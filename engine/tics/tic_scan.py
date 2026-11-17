"""The AI-tells scanner: tags the phrases in a text that read as machine-written.

The rules live in tic_patterns.py. This module finds where each one matches and never changes the text.

    hard   a tell nearly every time; the engine's gate rejects a candidate that has one
    check  could be a tell, could be a literal use or a real voice; a reader decides

The engine uses it through engine/draws.py make_tics, which puts this folder on sys.path, imports this module and
tic_patterns, and calls:

    compile_rules()            every rule, compiled, hard rules first: [{"label", "level", "rx", ...}]
    scan_line(text, rules, pov)  the hits in one line or candidate: [(position, level, label)]; pov is unused

On the command line it tags a file the way a reader would see it, each tag right after the phrase it is about:

    python engine/tics/tic_scan.py FILE            # the tagged text, then a tally
    python engine/tics/tic_scan.py FILE --tally    # the tally only
"""
import argparse
import re
import sys
from collections import Counter

try:
    from . import tic_patterns as _patterns
except ImportError:          # loaded from its folder on sys.path, as engine/draws.py make_tics does it
    import tic_patterns as _patterns

RULES = _patterns.RULES
HOUSE_ONLY = _patterns.HOUSE_ONLY
FAMILY_OF = _patterns.FAMILY_OF

DIVIDER_RE = re.compile(r"^\s*(?:\*\s*){3,}$|^\s*-{3,}\s*$|^\s*_{3,}\s*$")
ITALIC_RE = re.compile(r"(?<![*\w])\*(?![*\s])(.+?)(?<![\s*])\*(?![*\w])")
NOSCAN_RE = re.compile(r"https?://\S+|<[^>]+>|\]\([^)]*\)")


def compile_rules(rules=None):
    """The rules compiled, each a copy with "rx" and its place in the list ("order"), hard rules first so a check
    never doubles up on a span a hard rule already tagged."""
    out = []
    for i, r in enumerate(RULES if rules is None else rules):
        flags = 0 if r.get("case") else re.I
        rule = dict(r, order=i, rx=re.compile(r["regex"], flags))
        if "ai_relabel" in r:          # a stock phrase is tagged with its group's name, not "banned phrase"
            rule["relabel"] = r["ai_relabel"]
        out.append(rule)
    out.sort(key=lambda r: (r["level"] != "hard", r["order"]))
    return out


def is_skip_line(s):
    """Lines that aren't prose: blank, headings, comments, tables, images, quotes, dividers, a line all in bold."""
    t = s.strip()
    if not t:
        return True
    if t.startswith(("#", "<!--", "|", "![", ">")):
        return True
    if DIVIDER_RE.match(t):
        return True
    return t.startswith("**") and t.endswith("**")


def masks(s):
    """Per-character zone: 'n' narration, 'q' inside double quotes, 'i' italic, 'x' don't scan (links, markup)."""
    zone = ["n"] * len(s)
    inq = False
    for i, ch in enumerate(s):
        if ch == '"':
            zone[i] = "q"
            inq = not inq
            continue
        if ch == "“":
            inq = True
        if inq or ch == "”":
            zone[i] = "q"
        if ch == "”":
            inq = False
    for m in ITALIC_RE.finditer(s):
        for i in range(m.start(), m.end()):
            if zone[i] == "n":
                zone[i] = "i"
    for m in NOSCAN_RE.finditer(s):
        for i in range(m.start(), m.end()):
            zone[i] = "x"
    return zone


def in_scope(rule, zone, start):
    z = zone[start] if start < len(zone) else "n"
    if z == "x":
        return False
    scope = rule["scope"]
    if scope == "all":
        return True
    if scope == "narration":
        return z == "n"
    if scope == "dialogue":
        return z == "q"
    return False


def scan_line(s, rules, pov=None):
    """[(position, level, label)] for one line. The position is where a tag would go: after the matched phrase,
    never inside a word. `pov` is accepted for old callers and unused."""
    zone = masks(s)
    taken = []       # spans already tagged by word-level rules
    taken_long = []  # spans claimed by sentence-level rules ("stack": True)
    hits = []
    for r in rules:
        claimed = taken_long if r.get("stack") else taken
        for m in r["rx"].finditer(s):
            a, b = m.start(), m.end()
            if a == b or not in_scope(r, zone, a):
                continue
            if r["level"] == "check" and any(a < tb and ta < b for ta, tb in claimed):
                continue
            claimed.append((a, b))
            pos = b
            while 0 < pos < len(s) and s[pos - 1].isalnum() and s[pos].isalnum():
                pos += 1
            while pos > a and s[pos - 1] in ' "“':
                pos -= 1
            label = r["relabel"].get(m.group(0).lower(), r["label"]) if "relabel" in r else r["label"]
            hits.append((pos, r["level"], label))
    return hits


def scan(text, rules=None):
    """(hard hit count, sorted check labels) for a text, line by line: the shape make_tics returns."""
    rules = rules or compile_rules()
    hits = [h for line in text.splitlines() for h in scan_line(line, rules)]
    return (sum(1 for _, lvl, _ in hits if lvl == "hard"), sorted({lab for _, lvl, lab in hits if lvl != "hard"}))


def label_text(text, rules=None):
    """(the text with " [level: label]" tags put in after each hit, [(line number, level, label)]). Lines that
    aren't prose are left alone. Taking every " [hard: ...]" and " [check: ...]" tag back out gives the text back."""
    rules = rules or compile_rules()
    out, findings = [], []
    for n, raw in enumerate(text.splitlines(keepends=True), 1):
        body = raw.rstrip("\r\n")
        nl = raw[len(body):]
        if is_skip_line(body):
            out.append(raw)
            continue
        seen = set()
        for pos, level, label in sorted(scan_line(body, rules), key=lambda h: (-h[0], h[1] != "hard")):
            if (pos, label) in seen:
                continue
            seen.add((pos, label))
            body = body[:pos] + f" [{level}: {label}]" + body[pos:]
            findings.append((n, level, label))
        out.append(body + nl)
    return "".join(out), findings


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tag the AI tells in a text file.")
    ap.add_argument("file", help="a text or markdown file, or - for standard input")
    ap.add_argument("--tally", action="store_true", help="print only the count per label")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if a.file == "-":
        sys.stdin.reconfigure(encoding="utf-8")
        text = sys.stdin.read()
    else:
        with open(a.file, encoding="utf-8") as f:
            text = f.read()
    labeled, findings = label_text(text)
    if not a.tally:
        print(labeled)
    counts = Counter((level, label) for _, level, label in findings)
    for (level, label), n in sorted(counts.items(), key=lambda kv: (kv[0][0] != "hard", -kv[1], kv[0][1])):
        print(f"{n:5d}  {level:5s}  {label}  ({FAMILY_OF.get(label, '')})")
    if not counts:
        print("no tags")


if __name__ == "__main__":
    main()
