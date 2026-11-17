"""The engine's own AI-tells scanner and the house rules it shares with the code gates.

    tic_patterns.py   the rules: one regex per tell, a label, a hard or check level, a family
    tic_scan.py       runs the rules over a text (engine/draws.py make_tics loads it from this folder)
    phrases.txt       the stock phrases, tagged as checks
    house_rules.py    reads the house rules file (rules.toml at the repo root, or RULES_FILE)
    voice_audit.py    Burrows' Delta against a baseline built from your own samples

FOLDER is this folder: point TICS_DIR at it (engine/draws.py make_tics loads tic_scan.py and tic_patterns.py from
the folder TICS_DIR names). engine/checks.py and engine/reflect.py import from here directly.
"""
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
