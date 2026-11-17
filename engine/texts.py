"""Loading texts: your writing samples and a document's numbered sections.

Your samples come from SAMPLES_FILE (engine/settings.py), any markdown file you write in the samples format that
engine/tics/voice_audit.py reads: each sample opens with a "## Sample N" heading, and one tagged
"- **Source:** AI-assisted" is left out. With SAMPLES_FILE unset, load_samples returns no samples and the prompts that
show them go without.

A document's sections are its "## N." headings by default; guide_sections takes a job's section style instead.
"""
import re

from .settings import path_setting

# The AI example lines a document quotes: they come out of the prose-only score. A job lists its own in
# "example lines.txt" (the sync step writes it). This default is empty; a caller passes a job's list instead.
AI_EXAMPLE_LINES = frozenset()

COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
SECTION_RE = re.compile(r"^## (\d+)\.\s", re.M)
TAIL_RE = re.compile(r"^## (?!\d)", re.M)


def load_samples(samples=None, tics_dir=None):
    """Your raw samples: [("sample N", text)]. `samples` defaults to SAMPLES_FILE; with none set, [] (no samples).
    `tics_dir` is accepted for older callers and unused: the engine reads samples with its own
    engine/tics/voice_audit.py."""
    from .tics import voice_audit

    samples = samples or path_setting("SAMPLES_FILE")
    if not samples:
        return []
    kept, _dropped = voice_audit.load_samples(str(samples))
    out = []
    for s in kept:
        body = "\n".join(line for _, line in s["lines"]).strip()
        if body:
            out.append((f"sample {s['num']}", body))
    return out


def strip_markdown(text):
    """What a reader would paste: no comments, no heading marks, no bold marks."""
    text = COMMENT_RE.sub("", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    text = text.replace("**", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def guide_sections(numbers, path=None, text=None, style=None):
    """{number: raw section text} for top-level sections, subsections included. Give the document as `path` or,
    already in memory, as `text`. `style` is a job's section style (engine/job.py SectionStyle, or anything with
    its texts(md, ids) method); without one, the numbered "## N." headings: each section runs to the next one, and
    the last to the first unnumbered "## " heading."""
    if text is None and path is None:
        raise ValueError("guide_sections needs the document: a path or its text")
    raw = text if text is not None else path.read_text(encoding="utf-8")
    if style is not None:
        return style.texts(raw, numbers)
    starts = [(int(m.group(1)), m.start()) for m in SECTION_RE.finditer(raw)]
    tail = TAIL_RE.search(raw)
    out = {}
    for i, (num, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else (tail.start() if tail else len(raw))
        if num in numbers:
            out[num] = raw[start:end]
    return out


def prose_only(text, examples=None):
    """Drop the AI example lines. `examples` is a job's own list (the sync step can add to it)."""
    ex = AI_EXAMPLE_LINES if examples is None else examples
    keep = [line for line in text.split("\n") if line.strip() not in ex]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()
