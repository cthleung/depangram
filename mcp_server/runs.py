"""The run records: one folder a run under RUNS_DIR/mcp/, each holding

    run.json     what was started, its pid and the time the process began, and how it ended
    out.log      the run's own output (the wrapper's, the module's and anything it prints)
    exit         the exit code, written by mcp_server/child.py when the module ends
    config.toml  for a night, the config it runs under (DEPANGRAM_CONFIG), with the caller's ceilings in [night]

One file a run, written whole with os.replace, so two threads or two servers never tear a record, and a server that
restarts reads every run back from here. A pid only counts as the run while the process with that pid also began
when the record says, so a reused pid never passes for a run.
"""
import json
import math
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

RUN_ID = re.compile(r"^(pass|night)-\d{8}-\d{6}-[0-9a-f]{6}$")
STATES = ("running", "finished", "stopped", "lost")
POSIX = os.name != "nt"


def new_id(kind, now=None):
    return f"{kind}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(now or time.time()))}-{secrets.token_hex(3)}"


def valid_id(run_id):
    return isinstance(run_id, str) and bool(RUN_ID.match(run_id))


class Store:
    """The run folders under `root` (RUNS_DIR/mcp)."""

    def __init__(self, root):
        self.root = Path(root)

    def folder(self, run_id):
        if not valid_id(run_id):
            raise KeyError(run_id)
        return self.root / run_id

    def make(self, run_id):
        d = self.folder(run_id)
        d.mkdir(parents=True, exist_ok=False)
        return d

    def save(self, rec):
        d = self.folder(rec["id"])
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"run.json.{os.getpid()}.{secrets.token_hex(2)}.tmp"
        tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        os.replace(tmp, d / "run.json")

    def load(self, run_id):
        path = self.folder(run_id) / "run.json"
        if not path.exists():
            raise KeyError(run_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def all(self):
        if not self.root.is_dir():
            return []
        out = []
        for d in sorted(self.root.iterdir()):
            if valid_id(d.name) and (d / "run.json").exists():
                try:
                    out.append(json.loads((d / "run.json").read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
        return out

    def exit_code(self, run_id):
        """The code child.py wrote, or None."""
        path = self.folder(run_id) / "exit"
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None


# ---------------------------------------------------------------------- processes
def create_time(pid):
    """When the process `pid` began (psutil's seconds since the epoch), or None when there's no such process."""
    import psutil
    try:
        return psutil.Process(pid).create_time()
    except (psutil.Error, OSError):
        return None


def same_process(pid, began, now_began=None):
    """True while `pid` is the process the record was made for: it exists and began at `began`."""
    got = create_time(pid) if now_began is None else now_began
    return got is not None and began is not None and abs(got - began) < 1.0


def spawn(cmd, out, env, cwd):
    """A child process with its output into `out`, the same way engine/night.py's spawn starts one (its own session on
    POSIX, so kill_tree can end its process group; no window on Windows), plus an environment of its own and a
    process group of its own on Windows, so a Ctrl+C sent to the server doesn't reach it."""
    if POSIX:
        kw = {"start_new_session": True}
    else:
        kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)
              | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return subprocess.Popen(cmd, cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env=env, **kw)


def child_cmd(exit_file, module, args, python=None):
    """The command line of a run: the wrapper, then the module and its arguments."""
    return [python or sys.executable, "-m", "mcp_server.child", str(exit_file), "-m", module, *args]


# ---------------------------------------------------------------------- a config file for the night
BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(k):
    return k if BARE_KEY.match(k) else json.dumps(k)


def _value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"a config value isn't a finite number: {v!r}")
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_key(str(k))} = {_value(x)}" for k, x in v.items()) + "}"
    raise ValueError(f"a config value of type {type(v).__name__} can't be written as TOML")


def toml_dump(data):
    """A config dict as TOML: plain keys first, then one [table] for each dict at the top, with any dict deeper down
    written inline. Enough for config.toml, which holds no arrays of tables and no dates."""
    lines = [f"{_key(k)} = {_value(v)}" for k, v in data.items() if not isinstance(v, dict)]
    for name, table in data.items():
        if not isinstance(table, dict):
            continue
        if lines:
            lines.append("")
        lines.append(f"[{_key(name)}]")
        lines += [f"{_key(k)} = {_value(v)}" for k, v in table.items()]
    return "\n".join(lines) + "\n"


def night_config(config, *, max_pangram_usd, max_claude_usd=None, rounds=None, sections=None):
    """A copy of `config` for one night: [night] max_pangram_usd, and max_claude_usd, rounds and sections when given.
    The repair step at the submission keeps its own [repair] ceilings, each held to no more than the caller's (it
    spends on its own, past the night's pass and finals)."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    nt = cfg.setdefault("night", {})
    nt["max_pangram_usd"] = float(max_pangram_usd)
    if max_claude_usd is not None:
        nt["max_claude_usd"] = float(max_claude_usd)
    if rounds is not None:
        nt["rounds"] = int(rounds)
    if sections is not None:
        nt["sections"] = [int(x) for x in sections]
    rp = cfg.setdefault("repair", {})
    rp["max_pangram_usd"] = min(float(rp.get("max_pangram_usd", max_pangram_usd)), float(max_pangram_usd))
    if max_claude_usd is not None:
        rp["max_claude_usd"] = min(float(rp.get("max_claude_usd", max_claude_usd)), float(max_claude_usd))
    return cfg
