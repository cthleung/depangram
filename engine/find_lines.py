"""Find your lines: which sentences of a document you wrote, and which a model wrote.

The engine only regrows the model's sentences, so it needs to know which are yours. It takes one of three sources:

  1. A "your lines" file you supply. The job's `your lines.md` (job.toml's your_lines) is a table with one row per
     sentence: its verdict in the first cell, the sentence in the last. Write it or edit it by hand and the engine
     reads it as it is (engine/job.py verdicts); this module isn't needed. The table main() writes below shows the
     format.
  2. A folder of chat transcripts, read by this module, which then writes `your lines.md` for you. Transcripts are
     the conversations you had with a model while you wrote the document. Two kinds are read:
       - JSONL files (*.jsonl), searched through every subfolder: one JSON record per line, holding a turn with a
         role and its content. The role can sit in "role", "type", "from", "speaker", "author" ({"role": ...}) or
         a nested "message"; a record can also hold a whole conversation as a list under "messages", "turns" or
         "conversations". Content can be a string, a list of blocks ({"type": "text", "text": ...}), a "parts"
         list or a "value". A tool call's input counts as the model's output (it is what the model wrote into a
         file); a tool result counts as nobody's (it is a file read back).
       - Markdown chat exports (*.md): a turn opens on a line that is only a speaker name (User, Human, You, Me,
         Assistant, Agent, AI, Model, Bot, Claude, ChatGPT, Gemini), with optional markdown around it (#, *, _, >)
         and an optional note in brackets ("_**User (morning)**_"), or on a line that opens with the name in
         bold and a colon ("**User:** text"; for User, Human, Assistant, Agent, Claude, ChatGPT and Gemini, case
         kept). Code fences and tool blocks are skipped in every turn.
     The folders come from settings (engine/settings.py): CHAT_LOG_DIRS for the JSONL folders (several, separated by
     the system's path separator) and CHAT_HISTORY_DIR for the markdown folder. Both are empty by default, and then
     nothing is read: every sentence comes out `yours?`.
  3. Everything is machine: every sentence is the model's to regrow. Either set verdicts = "file" in job.toml and
     leave `your lines.md` out (engine/job.py then reads no verdicts at all), or write the table with
     `python -m engine.find_lines --machine` (classify with source="machine") and edit it from there.

A sentence counts as found in a source when most of its 5-word runs appear there, so small hand edits still match.
Verdicts:
  yours        found in your messages, not in the model's output
  yours?       not found in the model's output (only trustworthy when every transcript is read)
  claude       found in the model's output and not in yours (and every sentence, with source="machine")
  edited       found in the model's output, but you changed part of it
  both         found in both (pasted back and forth)

    python -m engine.find_lines              # the job's frozen original, from the transcript folders
    python -m engine.find_lines --machine    # every sentence the model's
"""
import argparse
import json
import re
import sys
from pathlib import Path

from .settings import path_list_setting, path_setting

N = 5
FOUND = 0.7
PARTIAL = 0.3
SOURCES = ("transcripts", "machine")

USER_ROLES = {"user", "human", "you", "me"}
MODEL_ROLES = {"assistant", "agent", "ai", "model", "bot", "claude", "chatgpt", "gpt", "gemini"}
SPEAKERS = r"User|Human|You|Me|Assistant|Agent|AI|Model|Bot|Claude|ChatGPT|Gemini"
# a line that is only a speaker name, with markdown around it and an optional note in brackets
HEAD_RE = re.compile(rf"^[#>*_ \t]*({SPEAKERS})\b(?:[ \t]*\([^)\n]*\))?[*_ \t]*:?[*_ \t]*$", re.M | re.I)
# a line that opens with a speaker name in bold and a colon ("**User:** text"), the turn's text after it. Only the
# names that can't be ordinary words, in bold, with case kept, so "User: clicks save" inside a reply stays prose.
INLINE_SPEAKERS = r"User|Human|Assistant|Agent|Claude|ChatGPT|Gemini"
INLINE_RE = re.compile(rf"^[> \t]*\*\*({INLINE_SPEAKERS})(?::\*\*|\*\*:)[ \t]+(?=\S)", re.M)
TOOL_RE = re.compile(r"<tool-use.*?</tool-use>", re.S)
FENCE_RE = re.compile(r"```.*?```", re.S)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def norm_words(text):
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return re.findall(r"[a-z0-9']+", text)


def shingles(words, n=N):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def split_sentences(text):
    text = COMMENT_RE.sub("", text)
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para or para.startswith("#"):
            continue
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", para.replace("\n", " ")):
            if len(norm_words(s)) >= N:
                out.append(s.strip())
    return out


# ---------------------------------------------------------------------- markdown chat exports
def markdown_turns(text):
    """[(role, text)] from a markdown chat export, role "user" or "model"; text before the first turn is skipped.
    A speaker line inside a code fence doesn't open a turn, and code fences and tool blocks are left out of every
    turn's text (they hold file contents and commands, nobody's prose)."""
    fences = [(m.start(), m.end()) for m in FENCE_RE.finditer(text)]
    marks = sorted({(m.start(), m.end(), m.group(1)) for rx in (HEAD_RE, INLINE_RE) for m in rx.finditer(text)
                    if not any(a <= m.start() < b for a, b in fences)})
    turns, last_end = [], -1
    for i, (a, b, who) in enumerate(marks):
        if a < last_end:
            continue
        last_end = b
        nxt = next((m[0] for m in marks[i + 1:] if m[0] >= b), len(text))
        role = "user" if who.lower() in USER_ROLES else "model"
        turns.append((role, FENCE_RE.sub("", TOOL_RE.sub("", text[b:nxt]))))
    return turns


def markdown_sources(folder=None):
    """Shingles of your messages and the model's replies in the markdown chat folder (CHAT_HISTORY_DIR by default).
    Any markdown export in the format above is read."""
    user, model = set(), set()
    folder = folder or path_setting("CHAT_HISTORY_DIR")
    if not folder or not Path(folder).is_dir():
        return user, model
    for p in sorted(Path(folder).glob("*.md")):
        for role, body in markdown_turns(p.read_text(encoding="utf-8", errors="ignore")):
            (user if role == "user" else model).update(shingles(norm_words(body)))
    return user, model


# ---------------------------------------------------------------------- JSONL transcripts
def strings_in(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings_in(v)


def _role(rec):
    """"user", "model" or None for one turn record."""
    msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
    author = rec.get("author") if isinstance(rec.get("author"), dict) else {}
    for value in (msg.get("role"), rec.get("role"), author.get("role"), rec.get("from"), rec.get("speaker"),
                  rec.get("type")):
        if isinstance(value, str):
            v = value.strip().lower()
            if v in USER_ROLES:
                return "user"
            if v in MODEL_ROLES:
                return "model"
    return None


def _content(rec):
    msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
    for value in (msg.get("content"), rec.get("content"), rec.get("text"), rec.get("value"), msg.get("parts"),
                  rec.get("parts")):
        if value is not None:
            return value
    return None


def turn_texts(rec):
    """[(role, text)] for one JSON record: a turn, or a conversation held under "messages", "turns" or
    "conversations"."""
    for key in ("messages", "turns", "conversations"):
        if isinstance(rec.get(key), list):
            return [t for item in rec[key] if isinstance(item, dict) for t in turn_texts(item)]
    role = _role(rec)
    if not role:
        return []
    content = _content(rec)
    if isinstance(content, dict) and "parts" in content:
        content = content["parts"]
    out = []
    if isinstance(content, str):
        out.append((role, content))
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                out.append((role, block))
            elif isinstance(block, dict):
                kind = block.get("type")
                if kind == "tool_result":          # a file read back: nobody's words
                    continue
                if kind == "tool_use":
                    if role == "model":
                        out += [(role, s) for s in strings_in(block.get("input"))]
                    continue
                text = block.get("text")
                if isinstance(text, str):
                    out.append((role, text))
    return out


def log_sources(dirs=None):
    """Shingles of your messages and the model's output in the JSONL folders (CHAT_LOG_DIRS by default), and how
    many files were read."""
    user, model, files = set(), set(), 0
    for d in (dirs if dirs is not None else path_list_setting("CHAT_LOG_DIRS")):
        d = Path(d)
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.jsonl")):
            files += 1
            with p.open(encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    for role, text in turn_texts(rec):
                        (user if role == "user" else model).update(shingles(norm_words(text)))
    return user, model, files


# ---------------------------------------------------------------------- verdicts
def verdict(u, c):
    if c >= FOUND and u >= FOUND:
        return "both"
    if c >= FOUND:
        return "claude"
    if c >= PARTIAL:
        return "edited"
    if u >= FOUND:
        return "yours"
    return "yours?"


def classify(doc_text, source="transcripts", dirs=None, history=None):
    """([(sentence, verdict, share in your messages, share in the model's output)], JSONL files read). `source` is
    "transcripts" (the folders in settings, or `dirs` for JSONL and `history` for markdown) or "machine" (every
    sentence the model's, nothing read)."""
    if source not in SOURCES:
        raise ValueError(f"source must be one of {', '.join(SOURCES)}, not {source!r}")
    if source == "machine":
        return [(s, "claude", 0.0, 1.0) for s in split_sentences(doc_text)], 0
    su, sc = markdown_sources(history)
    lu, lc, nfiles = log_sources(dirs)
    user, model = su | lu, sc | lc
    rows = []
    for s in split_sentences(doc_text):
        sh = shingles(norm_words(s))
        u = len(sh & user) / len(sh)
        c = len(sh & model) / len(sh)
        rows.append((s, verdict(u, c), u, c))
    return rows, nfiles


def table(rows, nfiles, source="transcripts"):
    """The `your lines.md` text for classify's rows."""
    counts = {}
    for _, v, _, _ in rows:
        counts[v] = counts.get(v, 0) + 1
    if source == "machine":
        where = "none: every sentence is the model's (find_lines --machine)."
    else:
        where = (f"the transcript folders in settings, {nfiles} JSONL file(s)"
                 f"{'' if nfiles else ' (none found in CHAT_LOG_DIRS, so `yours?` is a guess)'}.")
    lines = [
        "# Your lines in the document", "",
        f"Sources: {where}",
        "", "Counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
        "Skim for mistakes: anything marked yours that you didn't write, or claude that you did.", "",
        "| Verdict | In your messages | In the model's output | Sentence |", "| --- | --- | --- | --- |",
    ]
    for s, v, u, c in rows:
        lines.append(f"| {v} | {u:.0%} | {c:.0%} | {s.replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    from .job import Job
    ap = argparse.ArgumentParser(description="Write the job's `your lines.md`: which sentences are yours.")
    ap.add_argument("--job", default=None, help="the job's name (default: the job engine/job.py picks)")
    ap.add_argument("--machine", action="store_true", help="every sentence is the model's; read no transcripts")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    job = Job(a.job)
    text = job.original.read_text(encoding="utf-8")
    style = getattr(job, "style", None)                # the job's section style (engine/job.py)
    heads = style.heads(text) if style is not None else []
    if heads:
        text = text[heads[0][1]:]                          # from the first section on: no title or front matter
    else:
        m = re.search(r"^## 1\.", text, re.M)
        text = text[m.start():] if m else text
    source = "machine" if a.machine else "transcripts"
    rows, nfiles = classify(text, source=source)
    out = job.your_lines
    out.write_text(table(rows, nfiles, source), encoding="utf-8")
    counts = {}
    for _, v, _, _ in rows:
        counts[v] = counts.get(v, 0) + 1
    print(f"{len(rows)} sentences -> {out}")
    print(counts)


if __name__ == "__main__":
    main()
