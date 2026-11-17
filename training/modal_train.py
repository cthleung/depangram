"""Trains a notes-to-text (or paraphrase) adapter on Modal and converts it to a llama.cpp LoRA (plan 6.9, Phase 3).

An image from a CUDA registry, named volumes, one app. Everything heavy runs in the cloud; this file only ferries
data up and the adapter back.

    python training/run_modal.py run training/modal_train.py --action warm      # base weights -> the cache volume (CPU box, cheap)
    python training/run_modal.py run training/modal_train.py --action upload --pairs DATA/pairs_notes.jsonl
    python training/run_modal.py run training/modal_train.py --action upload --config hip_notes.yaml --pairs FILE
    python training/run_modal.py run training/modal_train.py --action train     # a GPU from GPUS, writes the adapter
    python training/run_modal.py run training/modal_train.py --action convert   # adapter -> llama.cpp LoRA gguf
    python training/run_modal.py run training/modal_train.py --action download  # the gguf -> MODELS_DIR/<run_name>/
    python training/run_modal.py run training/modal_train.py --action download --out FOLDER   # somewhere else
    python training/run_modal.py run training/modal_train.py --action train --qlora   # nf4 rank-32 fallback
    python training/run_modal.py run training/modal_train.py --action sample --config hip_notes2.yaml --jobs J --out O
        # draws from adapters on the outputs volume, a GPU from GPUS, no local card

--config picks the yaml next to this file (default hip_mistral.yaml). --pairs is the pairs file `upload` sends; it
has no default, since the data is yours and isn't in the repo. --out is where `download` saves (default
MODELS_DIR/<run_name>, so models.toml's `lora` can point at it).

The Modal names: the app is "<prefix>-train" and the volumes "<prefix>-hf-cache", "<prefix>-data" and
"<prefix>-outputs", with the prefix from MODAL_PREFIX in .env or the environment (default "depangram"). Renaming
them points the code at new, empty volumes.

Run it through training/run_modal.py, which loads MODAL_TOKEN_ID, MODAL_TOKEN_SECRET and HF_TOKEN from .env
(training/README.md). The HF token becomes the container's secret below, so there's no Modal dashboard step; the
Hugging Face account behind it has to have accepted the base model's terms.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from engine.settings import load_env, setting_path  # noqa: E402
    ENV = load_env()
except ModuleNotFoundError:  # this module is imported again inside the Modal container, where the repo isn't
    ENV, setting_path = {}, None

HERE = Path(__file__).resolve().parent
hf_secret = modal.Secret.from_dict({"HF_TOKEN": ENV.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")})
CONFIG = HERE / "hip_mistral.yaml"
# the Modal names' prefix: the same value inside the container, which gets it through the image's env below
PREFIX = ENV.get("MODAL_PREFIX") or os.environ.get("MODAL_PREFIX") or "depangram"

MINUTES = 60
# The cards the GPU functions take, first free one wins. No H100: Modal's Starter plan doesn't get them; put it back
# first if your plan does. The H200 and the A100 80 GB both hold the bf16 24B base and a rank-128 LoRA; the recipe
# doesn't change, and train records the card it got in training_metrics.json
GPUS = ["H200", "A100-80GB"]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    # unpinned on purpose: pin after the first successful run by copying the versions it logs
    .uv_pip_install("torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes",
                    "sentencepiece", "protobuf", "huggingface_hub[hf_xet]", "pyyaml", "gguf", "mistral-common")
    .run_commands("git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0", "MODAL_PREFIX": PREFIX})
)

# the Modal app and volume names (renaming them points the code at new, empty volumes)
hf_cache = modal.Volume.from_name(f"{PREFIX}-hf-cache", create_if_missing=True)
data = modal.Volume.from_name(f"{PREFIX}-data", create_if_missing=True)
outputs = modal.Volume.from_name(f"{PREFIX}-outputs", create_if_missing=True)

app = modal.App(f"{PREFIX}-train")
VOLUMES = {"/root/.cache/huggingface": hf_cache, "/data": data, "/outputs": outputs}


def load_config(name=None):
    """The yaml next to this file; `--config hip_notes.yaml` picks another."""
    import yaml
    path = HERE / name if name else CONFIG
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def pairs_local(pairs):
    """The pairs file `upload` sends: --pairs, which has no default."""
    if not pairs:
        raise SystemExit("upload needs --pairs FILE: the pairs file build_notes_pairs.py wrote in your data folder")
    path = Path(pairs)
    if not path.is_file():
        raise SystemExit(f"no pairs file at {path}")
    return path


def local_out(cfg, out=""):
    """Where `download` saves: --out, else MODELS_DIR/<run_name>."""
    if out:
        return Path(out)
    if setting_path is None:
        raise SystemExit("download needs --out FOLDER here (the repo's settings aren't importable)")
    return setting_path("MODELS_DIR") / cfg["run_name"]


@app.function(image=image, volumes={"/data": data}, timeout=10 * MINUTES)
def upload(pairs_bytes: bytes, name: str = "pairs.jsonl"):
    Path("/data") .joinpath(name).write_bytes(pairs_bytes)
    data.commit()
    n = sum(1 for _ in open(Path("/data") / name, encoding="utf-8"))
    return n


@app.function(image=image, gpu=GPUS, volumes=VOLUMES, secrets=[hf_secret], timeout=4 * 60 * MINUTES)
def train(cfg: dict, qlora: bool = False):
    import random

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(int(cfg["seed"]))
    device = torch.cuda.get_device_name(0)
    print(f"device {device}", flush=True)
    out_dir = Path("/outputs") / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # data: your passages as targets, the AI paraphrases as sources
    rows = [json.loads(l) for l in open(cfg["pairs_file"], encoding="utf-8")]
    rows = [r for r in rows if r["source"] in cfg["sources"] and (r["ok"] or not cfg["only_ok_pairs"])]
    random.Random(int(cfg["seed"])).shuffle(rows)
    if cfg["max_train_samples"]:
        rows = rows[: int(cfg["max_train_samples"])]
    hold = int(cfg["holdout"])
    tpl = cfg["template"]

    def to_text(r):
        return {"prompt": tpl.split("{target}")[0].format(source=r[cfg["source_text_field"]]),
                "completion": tpl.split("{target}")[1].join([r[cfg["target_text_field"]], ""])}

    train_ds = Dataset.from_list([to_text(r) for r in rows[hold:]])
    eval_ds = Dataset.from_list([to_text(r) for r in rows[:hold]]) if hold else None
    print(f"train rows {len(train_ds)}, eval rows {0 if eval_ds is None else len(eval_ds)}")

    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    kwargs = {"torch_dtype": torch.bfloat16, "attn_implementation": cfg["attn_implementation"],
              "device_map": {"": 0}}
    use_4bit = qlora or bool(cfg.get("load_in_4bit"))
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **kwargs)
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    rank = 32 if qlora else int(cfg["lora_r"])
    peft_config = LoraConfig(r=rank, lora_alpha=32 if qlora else int(cfg["lora_alpha"]),
                             lora_dropout=float(cfg["lora_dropout"]), bias="none", task_type=TaskType.CAUSAL_LM,
                             target_modules=list(cfg["lora_target_modules"]))

    import inspect
    accepted = set(inspect.signature(SFTConfig.__init__).parameters)
    # transformers 5 dropped warmup_ratio; a fractional warmup_steps means the same thing
    warm = ({"warmup_ratio": float(cfg["warmup_ratio"])} if "warmup_ratio" in accepted
            else {"warmup_steps": float(cfg["warmup_ratio"])})
    args = SFTConfig(
        **warm,
        output_dir=str(out_dir), run_name=cfg["run_name"],
        do_eval=eval_ds is not None, eval_strategy="epoch" if eval_ds is not None else "no",
        num_train_epochs=float(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]), lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=int(cfg["logging_steps"]),
        save_strategy="no", bf16=True, tf32=True,
        gradient_checkpointing=bool(cfg["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=int(cfg["max_length"]), completion_only_loss=True, packing=False,
        report_to=[], seed=int(cfg["seed"]))
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tok, peft_config=peft_config)
    result = trainer.train()
    metrics = dict(result.metrics)
    if eval_ds is not None:
        metrics.update(trainer.evaluate())
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "training_metrics.json").write_text(json.dumps(
        {"config": cfg, "qlora": qlora, "device": device, "metrics": metrics, "log_history": trainer.state.log_history}, indent=2))
    import importlib.metadata as im
    (out_dir / "versions.txt").write_text("\n".join(
        f"{p}=={im.version(p)}" for p in ["torch", "transformers", "peft", "trl", "datasets", "bitsandbytes"]))
    outputs.commit()
    return metrics


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=60 * MINUTES, cpu=4, memory=65536)
def convert(cfg: dict):
    """PEFT adapter -> llama.cpp LoRA GGUF (same tool we used on the CMU Qwen adapter)."""
    from huggingface_hub import snapshot_download
    out_dir = Path("/outputs") / cfg["run_name"]
    base = snapshot_download(cfg["model_name"], allow_patterns=["*.json", "*.safetensors", "tokenizer*",
                                                                 "*.model", "*.txt"])
    gguf = out_dir / f"{cfg['run_name']}-f16.gguf"
    cmd = ["python", "/opt/llama.cpp/convert_lora_to_gguf.py", str(out_dir), "--base", base,
           "--outfile", str(gguf), "--outtype", "f16"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    outputs.commit()
    return gguf.name, gguf.stat().st_size


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=60 * MINUTES, cpu=4, memory=32768)
def warm(cfg: dict):
    """Pull the base weights into the cache volume on a CPU box, so the GPU run starts training at once."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(cfg["model_name"], allow_patterns=["*.json", "*.safetensors", "tokenizer*",
                                                                 "*.model", "*.txt"])
    hf_cache.commit()
    size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return path, round(size / 1e9, 1)


SAMPLE_STOPS = ["</target_text>", "<notes>", "<source_text>"]      # engine/local_models.py ADAPTER_STOPS


def cut_stops(text, stops=SAMPLE_STOPS):
    """Everything before the first stop string (generate's stop_strings keeps the stop itself)."""
    cut = min([text.find(s) for s in stops if s in text] or [len(text)])
    return text[:cut]


@app.function(image=image, gpu=GPUS, volumes=VOLUMES, secrets=[hf_secret], timeout=2 * 60 * MINUTES)
def sample(model_name: str, jobs: list):
    """Draws from PEFT adapters on the outputs volume for the adapter comparison (experiments/exp11_adapter_ab.py),
    so it never needs the local card. jobs: [{"key", "adapter" (a run_name folder on the volume), "prompt", "n",
    "max_new", "seed"}] -> {key: [n draws]}. The base is bf16 and the sampler is transformers', so the draws
    differ from the loop's IQ4_XS llama.cpp ones; both adapters share them, which keeps the A/B fair. The sampling
    is the loop's: temperature 1.0, min_p 0.05, top_k and top_p off."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    class StopOnText(StoppingCriteria):
        """Per draw: done once its new text holds a stop string. generate's own stop_strings needs tokenizer
        internals the Mistral tokenizer backend doesn't have."""

        def __init__(self, start):
            self.start = start

        def __call__(self, input_ids, scores, **kw):
            texts = tok.batch_decode(input_ids[:, self.start:], skip_special_tokens=True)
            return torch.tensor([any(s in t for s in SAMPLE_STOPS) for t in texts], device=input_ids.device)

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map={"": 0},
                                                attn_implementation="sdpa")
    names = sorted({j["adapter"] for j in jobs})
    model = PeftModel.from_pretrained(base, f"/outputs/{names[0]}", adapter_name=names[0])
    for name in names[1:]:
        model.load_adapter(f"/outputs/{name}", adapter_name=name)
    model.eval()
    out = {}
    for k, j in enumerate(jobs):
        model.set_adapter(j["adapter"])
        torch.manual_seed(int(j.get("seed", 0)))
        ids = tok(j["prompt"], return_tensors="pt").to(0)
        with torch.no_grad():
            gen = model.generate(**ids, do_sample=True, temperature=1.0, min_p=0.05, top_k=0, top_p=1.0,
                                 max_new_tokens=int(j["max_new"]), num_return_sequences=int(j["n"]),
                                 stopping_criteria=StoppingCriteriaList([StopOnText(ids["input_ids"].shape[1])]),
                                 pad_token_id=tok.eos_token_id)
        texts = tok.batch_decode(gen[:, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        out[j["key"]] = [cut_stops(t) for t in texts]
        print(f"  {k + 1}/{len(jobs)} {j['key']}", flush=True)
    return out


@app.function(image=image, volumes={"/outputs": outputs}, timeout=30 * MINUTES)
def fetch(run_name: str, name: str) -> bytes:
    return (Path("/outputs") / run_name / name).read_bytes()


@app.local_entrypoint()
def main(action: str = "train", qlora: bool = False, config: str = "", jobs: str = "", out: str = "",
         pairs: str = ""):
    cfg = load_config(config or None)
    if action == "sample":                       # --jobs FILE --out FILE
        todo = json.loads(Path(jobs).read_text(encoding="utf-8"))
        draws = sample.remote(cfg["model_name"], todo)
        Path(out).write_text(json.dumps(draws, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{sum(len(v) for v in draws.values())} draws for {len(draws)} prompts -> {out}")
        return
    if action == "upload":
        src = pairs_local(pairs)
        n = upload.remote(src.read_bytes(), Path(cfg["pairs_file"]).name)
        print(f"uploaded {n} pairs from {src} to {cfg['pairs_file']}")
    elif action == "train":
        print(json.dumps(train.remote(cfg, qlora), indent=2))
    elif action == "warm":
        path, gb = warm.remote(cfg)
        print(f"base weights cached at {path} ({gb} GB)")
    elif action == "convert":
        print(convert.remote(cfg))
    elif action == "download":
        dest = local_out(cfg, out)
        dest.mkdir(parents=True, exist_ok=True)
        name = f"{cfg['run_name']}-f16.gguf"
        (dest / name).write_bytes(fetch.remote(cfg["run_name"], name))
        for extra in ("training_metrics.json", "versions.txt"):
            (dest / extra).write_bytes(fetch.remote(cfg["run_name"], extra))
        print(f"saved to {dest}")
        print("point a models.toml entry's `lora` at it (relative to MODELS_DIR):", dest / name)
    else:
        raise SystemExit(f"unknown action {action}")
