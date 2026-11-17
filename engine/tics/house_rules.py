"""The house rules: the wording choices the code gates make, read from a rules file.

The engine's gates hold a few lists that are choices rather than facts: the words a candidate may never bring in,
whether dashes are allowed, which spelling the text keeps, which words count as swears and how many a section may
carry, the filler tags and connective openers that are capped, and the stray bits of forum markup a writer model
trained on forum posts tends to leave behind, and the extra rules the end-of-day reflection writes under. They
live in a TOML file so a user can change them without touching
code.

Where the rules come from, first match wins:
  1. a path passed to load_rules (a job that points at its own rules file);
  2. RULES_FILE, read from .env or the environment the way engine/settings.py reads paths;
  3. rules.toml at the repo root;
  4. the built-in DEFAULTS below, which match the shipped rules.toml line for line.

A file only needs the keys it changes: every key it leaves out keeps its default. An unknown section or key is an
error, so a typo never passes silently.

Word lists use a small pattern syntax. A plain entry matches that whole word, case ignored. A trailing * matches any
ending ("walk*" catches "walking") and a leading * any beginning ("*fuck*"). The "except" lists name whole
words that never count even when a pattern matches them.

This module is plain standard library, so the scanner can load it from its own folder.
"""
import copy
import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # the repo root, the folder that holds engine/
DEFAULT_FILE = ROOT / "rules.toml"

DEFAULTS = {
    "house": {
        "dashes": True,
        "banned_words": ["structur*", "restructur*", "load-bearing", "scaffold*"],
        "banned_words_check": ["framework", "architecture", "foundation", "pillar", "cornerstone", "blueprint"],
    },
    "locale": {
        "spelling": "american",
    },
    "swears": {
        "words": ["*fuck*", "shit*", "damn*", "ass", "asshole", "assholes", "arse*", "hell", "bloody", "crap*",
                  "bastard*", "piss*", "bollocks"],
        "except": ["damning"],
        "strong": ["*fuck*"],
        "multiplier": 1.5,
    },
    "filler": {
        "words": ["honestly"],
        "cap": 1,
    },
    "connectives": {
        "openers": ["Of course", "In fact", "Fortunately", "In other words", "we can see", "Looking at", "Naturally"],
        "cap": 1,
    },
    "reflection": {
        "banned_words": [],
        "invented_digits": [],
    },
    "furniture": {
        "patterns": ["&#x200B;", "\u200b", r"^\s*(?:edit|EDIT|Edit)\s*[:\-]", r"\bTL;?DR\b", r"\br/\w+", r"\bOP\b",
                     r"\bupvot", r"\bdownvot"],
    },
}
SPELLINGS = ("american", "any")


def setting_path():
    """RULES_FILE from .env or the environment, or None. engine/settings.py does the reading when it can be
    imported; on its own (the scanner run from its folder) only the environment is read."""
    try:
        from engine.settings import path_setting
        return path_setting("RULES_FILE")
    except ImportError:
        value = os.environ.get("RULES_FILE", "").strip()
        return Path(value) if value else None


def default_path():
    """The rules file in use when no job names one: RULES_FILE, else rules.toml at the repo root, else None."""
    p = setting_path()
    if p:
        return p
    return DEFAULT_FILE if DEFAULT_FILE.exists() else None


def merge(base, extra, where="rules"):
    """`extra` laid over a copy of `base`, section by section. Unknown sections or keys raise ValueError."""
    out = copy.deepcopy(base)
    for section, values in extra.items():
        if section not in out:
            raise ValueError(f"{where}: unknown section [{section}]; the sections are {', '.join(out)}")
        if not isinstance(values, dict):
            raise ValueError(f"{where}: [{section}] should be a table of keys")
        for key, value in values.items():
            if key not in out[section]:
                raise ValueError(f"{where}: unknown key {key!r} in [{section}]; the keys are {', '.join(out[section])}")
            out[section][key] = value
    return out


def validate(rules, where="rules"):
    if rules["locale"]["spelling"] not in SPELLINGS:
        raise ValueError(f"{where}: [locale] spelling must be one of {', '.join(SPELLINGS)}")
    for pat in rules["furniture"]["patterns"]:
        re.compile(pat)
    for d in rules["reflection"]["invented_digits"]:
        if not (isinstance(d, str) and len(d) == 1 and d.isdigit()):
            raise ValueError(f"{where}: [reflection] invented_digits holds single digits as strings, like \"4\"")
    return rules


def load_rules(path=None):
    """The rules as a dict shaped like DEFAULTS: the file at `path` (else default_path()) over the defaults."""
    path = Path(path) if path else default_path()
    if not path:
        return copy.deepcopy(DEFAULTS)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return validate(merge(DEFAULTS, data, where=str(path.name)), where=str(path.name))


def word_pattern(words, exceptions=()):
    """The regex source for a word list in the pattern syntax above, or None for an empty list."""
    alts = []
    for w in words:
        w = w.strip()
        core = re.escape(w.strip("*"))
        if not core:
            continue
        alts.append((r"\w*" if w.startswith("*") else "") + core + (r"\w*" if w.endswith("*") else ""))
    if not alts:
        return None
    ex = "|".join(re.escape(e) for e in exceptions if e.strip())
    guard = rf"(?!(?:{ex})\b)" if ex else ""
    return rf"\b{guard}(?:{'|'.join(alts)})\b"


def word_regex(words, exceptions=()):
    """word_pattern compiled, case ignored; a regex that never matches for an empty list."""
    pat = word_pattern(words, exceptions)
    return re.compile(pat if pat else r"(?!)", re.I)


def stems(words):
    """The words of a list with their wildcards taken off, for a prompt that names them."""
    return [w.strip().strip("*") for w in words if w.strip().strip("*")]
