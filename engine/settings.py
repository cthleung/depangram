"""Loads .env (keys and paths) and config.toml (caps and knobs).

SANDBOX is the repo root, the folder that holds engine/: config.toml, .env, jobs/, analytics/ and digests/ live
there. REPO is kept for older callers; the morning and night steps find the git checkout they commit in from the
job's document (engine/morning.py's repo_for), or from [git] repo_root in config.toml.

config.toml is your own copy and git ignores it. When it's missing, the engine reads config.example.toml, the
documented template, and says so once on stderr. Copy the template to config.toml to tune anything.
DEPANGRAM_CONFIG in the environment (not .env, which is read after the config) points at another config file.

Every setting below is read from .env or the environment (a value in .env wins), and each has a default that
works on any machine. `.env.example` lists them all with a line on each.

Paths, with their defaults:

    RUNS_DIR         the run logs, the SQLite ledger and the queues. Default: runs/ in the repo (git ignores it).
    WORK_DIR         the folder the Claude Code CLI runs in. It must sit outside any repo, or the CLI loads that
                     repo's own instructions. Default: ~/.depangram/work.
    MODELS_DIR       the GGUF files the model registry points at. Default: models/ in the repo (ignored).
    MODELS_FILE      the model registry. Default: models.toml in the repo.
    LLAMA_CPP_DIR    a folder holding llama.cpp's llama-server. Default: llama.cpp/ in the repo (ignored); when no
                     llama-server is there, the one on the PATH.
    OPEN_PANGRAM_DIR the Open Pangram (EditLens) model files. Default: open-pangram/ in MODELS_DIR.
    STOP_FILE        a file whose presence stops the loop and the night run. Default: STOP in RUNS_DIR.
    NOTIFY_DIR       where the file notifier writes its posts. Default: notifications/ in RUNS_DIR.
    TICS_DIR         a folder holding an AI-tells scanner (tic_scan.py and tic_patterns.py) for engine/draws.py
                     make_tics. Default: the repo's own engine/tics/ when that folder exists, else unset; with it
                     unset, make_tics uses its small built-in check.
    RULES_FILE       the house rules file (banned words, spelling, the swear and filler budgets). Default:
                     rules.toml in the repo. A job can name its own in job.toml (rules).

Optional paths, unset by default. With one unset, the code that uses it falls back quietly, as its own docstring
says:

    SAMPLES_FILE     your writing samples ("## Sample N" headings, engine/tics/voice_audit.py). Unset: no samples.
    SAMPLES_TAGGED   samples for the swear budget, with the AI-assisted ones tagged so they're left out. The same
                     file as SAMPLES_FILE works. Unset: the budget reads your lines in the document only.
    VOICE_BASELINE   where the voice baseline (Burrows' Delta, built from SAMPLES_FILE) is cached. Unset: next to
                     the samples file, as "<its name>.baseline.json".
    VOICE_GUIDE      your voice guide (markdown), for the restore pass's careful prompt. Unset: none.
    IDIOMS_DIR       a folder of your own writing (.txt and .md files, or a forum export's comments.jsonl and
                     posts.jsonl), for the idiom list. Unset: no idioms.
    CHAT_LOG_DIRS    folders of chat transcripts (*.jsonl), separated by the system's path separator, for
                     find_lines. Unset: none.
    CHAT_HISTORY_DIR a folder of chat exports (*.md), for find_lines. Unset: none.

Choices:

    CLAUDE_BACKEND   "cli" (the Claude Code CLI with a subscription token, the default) or "api" (the Anthropic
                     API with ANTHROPIC_API_KEY). Also [claude] backend in config.toml; the environment wins.
    NOTIFIER         "discord" (the default), "file" (posts and digests written to NOTIFY_DIR) or "none". Also
                     [notify] notifier in config.toml; the environment wins.
"""
import os
import sys
import tomllib
from pathlib import Path

SANDBOX = Path(__file__).resolve().parents[1]      # the repo root, the folder that holds engine/
REPO = SANDBOX
CONFIG_FILE = SANDBOX / "config.toml"
CONFIG_EXAMPLE = SANDBOX / "config.example.toml"
USER_HOME = Path.home() / ".depangram"              # the user data folder, for what can't live in the repo
_told = set()


def load_env(path=None):
    """Parse a NAME=value file. Blank lines and # comments are skipped."""
    path = Path(path or SANDBOX / ".env")
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def config_path(path=None):
    """The config file to read: `path` when given, else DEPANGRAM_CONFIG from the environment, else config.toml,
    else the example with a note on stderr."""
    if path:
        return Path(path)
    if os.environ.get("DEPANGRAM_CONFIG"):
        return Path(os.environ["DEPANGRAM_CONFIG"]).expanduser()
    if CONFIG_FILE.exists():
        return CONFIG_FILE
    if "config" not in _told:
        _told.add("config")
        print(f"depangram: no config.toml in {SANDBOX}, so the defaults in config.example.toml are in use; copy it"
              " to config.toml to change them", file=sys.stderr, flush=True)
    return CONFIG_EXAMPLE


def load_config(path=None):
    with open(config_path(path), "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------- the path settings and their defaults
def _tics_default(get):
    d = SANDBOX / "engine" / "tics"
    return d if d.is_dir() else None


PATH_DEFAULTS = {
    "RUNS_DIR": lambda get: SANDBOX / "runs",
    "WORK_DIR": lambda get: USER_HOME / "work",
    "MODELS_DIR": lambda get: SANDBOX / "models",
    "MODELS_FILE": lambda get: SANDBOX / "models.toml",
    "LLAMA_CPP_DIR": lambda get: SANDBOX / "llama.cpp",
    "OPEN_PANGRAM_DIR": lambda get: setting_path("MODELS_DIR", get) / "open-pangram",
    "STOP_FILE": lambda get: setting_path("RUNS_DIR", get) / "STOP",
    "NOTIFY_DIR": lambda get: setting_path("RUNS_DIR", get) / "notifications",
    "TICS_DIR": _tics_default,
}


def _getter(source):
    """A get(name, default) from a Settings, a settings-like object with .get, a dict of .env values, a get function,
    or None (.env and the environment)."""
    if source is None:
        env = load_env()
        return lambda k, d="": env.get(k) or os.environ.get(k) or d
    if isinstance(source, dict):
        return lambda k, d="": source.get(k) or os.environ.get(k) or d
    if callable(source) and not hasattr(source, "get"):
        return source
    if not hasattr(source, "get"):                # a bare object with a config only: the environment
        return lambda k, d="": os.environ.get(k) or d
    return lambda k, d="": source.get(k, d) or d


def setting_path(name, source=None):
    """A path setting: its value from .env or the environment, else its default (PATH_DEFAULTS), else None. `source`
    is a Settings (or anything with .get(name, default)), a dict of .env values, or None to read .env."""
    get = _getter(source)
    value = str(get(name, "") or "").strip()
    if value:
        return Path(value).expanduser()
    make = PATH_DEFAULTS.get(name)
    return make(get) if make else None


def path_setting(name, env=None):
    """An optional path from .env or the environment, as Settings.get reads it, else its default when it has one
    (TICS_DIR: the repo's engine/tics/ when it exists), else None."""
    return setting_path(name, load_env() if env is None else env)


def path_list_setting(name, env=None):
    """An optional list of paths (separated by os.pathsep) from .env or the environment; [] when unset."""
    env = load_env() if env is None else env
    value = (env.get(name) or os.environ.get(name) or "").strip()
    return [Path(v) for v in value.split(os.pathsep) if v.strip()]


def choice(source, name, section, key, default, allowed):
    """A choice setting: .env or the environment first (`name`), then config.toml's [section] key, else `default`.
    An unknown value raises ValueError naming the allowed ones."""
    get = _getter(source)
    value = str(get(name, "") or "").strip().lower()
    if not value:
        cfg = getattr(source, "config", None) or {}
        value = str((cfg.get(section) or {}).get(key) or "").strip().lower()
    value = value or default
    if value not in allowed:
        raise ValueError(f"{name} is {value!r}; it takes one of {', '.join(allowed)}")
    return value


class Settings:
    def __init__(self, env=None, config=None):
        self.env = load_env() if env is None else env
        self.config = load_config() if config is None else config

    def get(self, key, default=""):
        """A setting from .env or the environment. A path setting (PATH_DEFAULTS) that is unset gets its own default
        here, ahead of the caller's, so every reader of STOP_FILE or RUNS_DIR finds the same place."""
        value = self.env.get(key) or os.environ.get(key)
        if value:
            return value
        if key in PATH_DEFAULTS:
            got = PATH_DEFAULTS[key](self._raw)
            if got is not None:
                return str(got)
        return default

    def _raw(self, key, default=""):
        return self.env.get(key) or os.environ.get(key) or default

    def path(self, name):
        """A path setting with its default (setting_path)."""
        return setting_path(name, self)

    @property
    def runs_dir(self):
        d = setting_path("RUNS_DIR", self)
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def work_dir(self):
        return setting_path("WORK_DIR", self)

    @property
    def stop_file(self):
        """The STOP file the loop and the night both watch (STOP_FILE, else STOP in RUNS_DIR)."""
        return setting_path("STOP_FILE", self)

    @property
    def db_path(self):
        return self.runs_dir / "depangram.sqlite"
