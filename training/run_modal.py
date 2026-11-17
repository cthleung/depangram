"""Runs the Modal CLI with the tokens from .env, so `modal setup` and the dashboard are never needed.

    python training/run_modal.py run training/modal_train.py --action upload --pairs FILE
    python training/run_modal.py run training/modal_train.py --action train
    python training/run_modal.py volume ls depangram-outputs

Anything after run_modal.py goes straight to `modal`. Reads MODAL_TOKEN_ID and MODAL_TOKEN_SECRET (and
HF_TOKEN, which modal_train.py turns into the container's Hugging Face secret) from the repo's .env, and
MODAL_PREFIX when it's set there (the Modal app and volume names, training/modal_train.py).
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.settings import load_env  # noqa: E402

env = load_env()
missing = [k for k in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "HF_TOKEN") if not env.get(k)]
if missing:
    sys.exit(f"fill these in .env first: {', '.join(missing)}")
os.environ.update({k: v for k, v in env.items() if k.startswith("MODAL_") or k == "HF_TOKEN"})
modal_exe = Path(sys.executable).with_name("modal.exe")
cmd = [str(modal_exe) if modal_exe.exists() else "modal", *sys.argv[1:]]
raise SystemExit(subprocess.call(cmd))
