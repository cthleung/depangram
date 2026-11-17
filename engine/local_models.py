"""Starts and stops llama.cpp's server for one model at a time (plan 11.1).

The models come from the registry in models.toml (MODELS_FILE in .env points at another copy): each entry names a
GGUF file under MODELS_DIR, whether it's a chat model, its context size and parallel slots, any extra llama-server
arguments, and an optional LoRA adapter with the scale its writer arm uses. llama-server itself is found in
LLAMA_CPP_DIR, else on the PATH.

llama-server's --fit (on by default) splits layers and MoE experts between the GPU and system RAM on its own, so we
only pick a context size.
"""
import os
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

import requests

from .settings import Settings, setting_path

SERVER_EXE = "llama-server.exe" if os.name == "nt" else "llama-server"


def load_models(source=None, path=None):
    """The registry: {name: spec} from models.toml (or `path`, or MODELS_FILE). A spec holds "file", "chat" and any
    of "ctx", "parallel", "extra", "lora", "lora_scale". A missing file is an empty registry."""
    path = Path(path) if path else setting_path("MODELS_FILE", source)
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return {}
    return {name: dict(spec) for name, spec in (data.get("models") or {}).items()}


try:
    MODELS = load_models()          # the default registry, for callers that read it directly
except (OSError, tomllib.TOMLDecodeError):
    MODELS = {}


def find_server(source=None):
    """llama-server: in LLAMA_CPP_DIR (or its build/bin), else on the PATH; None when neither has it."""
    d = setting_path("LLAMA_CPP_DIR", source)
    for cand in (d / SERVER_EXE, d / "build" / "bin" / SERVER_EXE, d / "bin" / SERVER_EXE):
        if cand.is_file():
            return str(cand)
    return shutil.which("llama-server")


class LlamaServer:
    def __init__(self, name, port=8079, ctx=4096, parallel=4, settings=None, models=None):
        s = settings or Settings()
        registry = models if models is not None else load_models(s)
        if name not in registry:
            raise KeyError(f"no model named {name!r} in the registry ({setting_path('MODELS_FILE', s)});"
                           f" it has: {', '.join(sorted(registry)) or 'nothing'}")
        self.name = name
        self.spec = registry[name]
        self.port = port
        self.ctx = int(self.spec.get("ctx", ctx))
        self.parallel = int(self.spec.get("parallel", parallel))
        models_dir = setting_path("MODELS_DIR", s)
        self.lora_path = str(models_dir / self.spec["lora"]) if self.spec.get("lora") else None
        self.extra = list(self.spec.get("extra", [])) + (["--lora", self.lora_path] if self.lora_path else [])
        self.exe = find_server(s)
        self.model_path = str(models_dir / self.spec["file"])
        self.log_path = s.runs_dir / f"llama-server-{name}.log"
        self.proc = None
        self.started = False
        from .guard import config as guard_config
        try:
            self.low_priority = bool(guard_config(getattr(s, "config", None) or {})["low_priority"])
        except ValueError:
            self.low_priority = True

    def missing(self):
        """What start() can't find before it launches anything, as plain sentences; [] when all is there."""
        out = []
        if not self.exe:
            out.append(f"no {SERVER_EXE} in LLAMA_CPP_DIR or on the PATH; install llama.cpp and set LLAMA_CPP_DIR")
        if not os.path.isfile(self.model_path):
            out.append(f"the model file isn't there: {self.model_path} (MODELS_DIR plus the registry's `file`)")
        if self.lora_path and not os.path.isfile(self.lora_path):
            out.append(f"the adapter isn't there: {self.lora_path}; train one (training/README.md) or remove"
                       f" `lora` from {self.name} in the registry")
        return out

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def _answers(self):
        """True when something already answers HTTP on the port (any status counts)."""
        try:
            requests.get(f"{self.url}/health", timeout=2)
            return True
        except requests.RequestException:
            return False

    def model_there(self):
        """The model file the server on the port says it loaded (/props), or None if it doesn't say."""
        try:
            r = requests.get(f"{self.url}/props", timeout=5)
            return (r.json() or {}).get("model_path") if r.status_code == 200 else None
        except (requests.RequestException, ValueError):
            return None

    def serves_our_model(self):
        there = self.model_there()
        if not there:
            return True                     # an older build without model_path: our own process is the check
        return os.path.normcase(os.path.basename(there)) == os.path.normcase(os.path.basename(self.model_path))

    def start(self, timeout=900):
        # another job's server on the same port (a paraphrase run, a bake-off left going) would answer /health
        # before ours had loaded, and the loop would pay to judge another model's draws (M1 review, finding 4)
        if self._answers():
            raise RuntimeError(f"port {self.port} already has a server on it (model {self.model_there() or 'unknown'});"
                               f" {self.name} won't start until it's gone")
        gone = self.missing()
        if gone:
            raise RuntimeError(f"{self.name} can't start: " + "; ".join(gone))
        # a start after a pause (engine/guard.py) adds to the log, so the server's lines from before the pause stay
        log = open(self.log_path, "a" if self.started else "w", encoding="utf-8")
        self.started = True
        # [guard] low_priority: below-normal CPU priority from the start on Windows, nice 10 just after it elsewhere
        from .guard import after_start, priority_flags
        # each parallel slot gets ctx/parallel tokens, so scale ctx up
        self.proc = subprocess.Popen(
            [self.exe, "-m", self.model_path, "--port", str(self.port), "-c", str(self.ctx * self.parallel),
             "-np", str(self.parallel), "--no-webui", *self.extra],
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | priority_flags(low=self.low_priority))
        after_start(self.proc, low=self.low_priority)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early, see {self.log_path}")
            try:
                if requests.get(f"{self.url}/health", timeout=2).status_code == 200:
                    if not self.serves_our_model():
                        self.stop()
                        raise RuntimeError(f"the server on port {self.port} has {self.model_there()} loaded,"
                                           f" not {self.model_path}")
                    if self.has_lora:
                        try:
                            got = self.adapters()
                        except (requests.RequestException, ValueError) as e:
                            self.stop()
                            raise RuntimeError(f"{self.name} loads a LoRA, but /lora-adapters didn't answer ({e});"
                                               " this llama.cpp build can't scale the adapter per request") from None
                        if not got:
                            self.stop()
                            raise RuntimeError(f"{self.name} started without its LoRA adapter (/lora-adapters is empty)")
                    return time.time() - t0
            except requests.RequestException:
                pass
            time.sleep(2)
        self.stop()
        raise TimeoutError(f"{self.name} did not come up in {timeout}s")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    @property
    def has_lora(self):
        """The spec loads a LoRA adapter, so per-request scaling (complete's lora_scale) means something."""
        return "--lora" in self.extra

    def adapters(self):
        """What the server says it has loaded, from GET /lora-adapters: [{"id", "path", "scale"}]. Recent
        llama.cpp builds answer this and take a per-request `lora` list on /completion, so one process serves both
        writer arms: scale 1.0 is the adapter, 0.0 the plain base."""
        r = requests.get(f"{self.url}/lora-adapters", timeout=10)
        r.raise_for_status()
        return r.json()

    def complete(self, prompt, n_predict=220, temperature=1.0, min_p=0.05, seed=-1, stop=None, ban=None,
                 lora_scale=None):
        """Raw continuation, for base models. `ban` is a list of token ids the sampler may never pick
        (llama.cpp logit_bias set to false), the "ban" control from plan 6.8. `lora_scale` (the writer arms)
        sets the loaded adapter's scale for this request only: 1.0 is the adapter arm, 0.0 the plain
        base; None leaves the server's own setting, so older callers behave as before. Toggling the scale flushes
        the prompt cache for that slot (the prompt is read again), which costs about a second on a 2,000-token
        prompt, so callers draw a run's whole pool on one arm."""
        body = {"prompt": prompt, "n_predict": n_predict, "temperature": temperature, "min_p": min_p,
                "top_p": 1.0, "top_k": 0, "seed": seed, "stop": stop or [], "cache_prompt": True}
        if ban:
            body["logit_bias"] = [[t, False] for t in ban]
        if lora_scale is not None:
            body["lora"] = [{"id": 0, "scale": float(lora_scale)}]
        r = requests.post(f"{self.url}/completion", timeout=900, json=body)
        r.raise_for_status()
        d = r.json()
        return d["content"], d.get("timings", {})

    def tokenize(self, text):
        r = requests.post(f"{self.url}/tokenize", timeout=60, json={"content": text, "add_special": False})
        r.raise_for_status()
        return r.json()["tokens"]

    def digit_tokens(self):
        """Every single-token digit, with and without a leading space, for banning invented numbers.
        Only plain ASCII digits: a glyph like "½" tokenizes to a raw byte, and banning a byte corrupts every
        other character that uses it (run 8 of experiment 6 came back full of "�")."""
        ids = set()
        for d in "0123456789":
            for form in (d, " " + d):
                toks = self.tokenize(form)
                if len(toks) == 1:
                    ids.add(toks[0])
        return sorted(ids)

    def chat(self, user, max_tokens=260, temperature=1.0, min_p=0.1, seed=-1):
        """Chat turn, for instruct models like StyleTune."""
        r = requests.post(f"{self.url}/v1/chat/completions", timeout=900, json={
            "messages": [{"role": "user", "content": user}], "max_tokens": max_tokens,
            "temperature": temperature, "min_p": min_p, "seed": seed,
            # Gemma 4 thinks first by default; a bake-off spent every token thinking
            # and returned empty answers, so thinking is off for rewrites.
            "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
        d = r.json()
        return d["choices"][0]["message"]["content"], d.get("timings", {})


# ---------------------------------------------------------------------- the writer arms
# Both arms run on one server: a base model with an adapter loaded (the registry's `lora`). The "notes" arm is the
# adapter at its scale (the registry's lora_scale, else 1.0); the "fewshot" arm is the plain base at scale 0.0, the
# few-shot writer. The flavor pass draws on the fewshot arm too. A model with no LoRA has only the fewshot arm, at
# the server's own setting.
ARM_SCALE = {"notes": 1.0, "fewshot": 0.0, "flavor": 0.0}
ADAPTER_STOPS = ["</target_text>", "<notes>", "<source_text>"]


class ArmServer:
    """A server seen through one writer arm: every complete() carries the arm's LoRA scale and the arm's extra stop
    strings, and everything else passes through. draw_pool takes it where it takes a LlamaServer."""

    def __init__(self, srv, arm):
        self.srv, self.arm = srv, arm
        self.scale = ARM_SCALE.get(arm, 0.0) if getattr(srv, "has_lora", False) else None
        spec = getattr(srv, "spec", None)
        if self.scale is not None and arm == "notes" and isinstance(spec, dict) and "lora_scale" in spec:
            self.scale = float(spec["lora_scale"])          # the registry's scale for the adapter arm
        self.extra_stops = ADAPTER_STOPS if arm == "notes" else []

    def complete(self, prompt, n_predict=220, temperature=1.0, min_p=0.05, seed=-1, stop=None, ban=None, **kw):
        stops = list(stop or []) + [s for s in self.extra_stops if s not in (stop or [])]
        if self.scale is not None:
            kw["lora_scale"] = self.scale
        return self.srv.complete(prompt, n_predict=n_predict, temperature=temperature, min_p=min_p, seed=seed,
                                 stop=stops, ban=ban, **kw)

    def __getattr__(self, name):
        return getattr(self.srv, name)


def arms_available(srv):
    """The arms a server can serve: both when it loads the adapter, else the fewshot arm alone."""
    return ("notes", "fewshot") if getattr(srv, "has_lora", False) else ("fewshot",)
