"""How close a text sits to your own writing: Burrows' Delta on function words, against a baseline you build from
your own samples.

Nothing here ships numbers. The baseline is computed from your samples file the first time it's needed and cached
as JSON; the cache is rebuilt whenever the samples file changes.

The samples file is markdown. Each sample opens with a heading "## Sample N" (N a number, any title after it), and
runs to the next one. Bullet lines of the form "- **Source:** ..." (also Medium, Register, Context) are metadata and
not counted; a sample whose Source starts with "AI" (say "- **Source:** AI-assisted") is left out, so one file can
hold your raw samples and the ones a model helped with. Headings, dividers and markup are skipped.

    load_samples(path)                 the samples kept and the ones left out
    build_baseline(path)               the baseline as a dict (function words, chunk vectors, chunk-vs-rest Deltas)
    cached_baseline(path, cache=None)  the same, read from the cache when the samples haven't changed
    delta_line(text, baseline)         "DELTA (soft): ..." for a text, or "" when there's no usable baseline
    audit(text, samples=None)          delta_line with the samples from SAMPLES_FILE, or ""

Settings (read from .env or the environment, as engine/settings.py reads paths):
    SAMPLES_FILE     your samples file. Unset: no baseline, and audit returns "".
    VOICE_BASELINE   where the baseline cache goes. Unset: next to the samples file, named
                     "<samples file name>.baseline.json".

On the command line, the way engine/regrow.py audit runs it:

    python engine/tics/voice_audit.py - < draft.md              # one DELTA line, or nothing with no samples
    python engine/tics/voice_audit.py draft.md --samples my.md
    python engine/tics/voice_audit.py --baseline --samples my.md  # build the cache and say how many chunks
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

CHUNK_WORDS = 300
TOP_FUNCTION_WORDS = 60
MIN_CHUNKS = 3

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)*(?:-[A-Za-z0-9]+)*")
SAMPLE_RE = re.compile(r"^##\s+Sample\s+(\d+)\b", re.I)
META_RE = re.compile(r"^- \*\*(Medium|Register|Context|Source):\*\*\s*(.*)$", re.I)
SKIP_RE = re.compile(r"^\s*(?:#{1,6}\s|-{3,}\s*$|\*{3,}\s*$|_{3,}\s*$)")

# Common English function words, the ones Burrows' Delta counts.
FUNCTION_WORDS = set(
    """
a about above after again against all also although am an and any are as at be because been
before being below between both but by can could did do does doing down during each either
even ever every few for from further had has have having he her here hers herself him himself
his how however i if in into is it its itself just least less like many may me might more most
much must my myself neither no nor not now of off often on once only or other our ours out over
own perhaps quite rather really same she should since so some still such than that the their
theirs them themselves then there these they this those though through thus to too under until
up upon us very was we were what whatever when where whether which while who whom whose why
will with within without would yet you your yours yourself yourselves
i'm it's don't that's i've i'd i'll can't isn't wasn't you're there's didn't doesn't won't
""".split()
)


# ---------------------------------------------------------------- reading text
def normalize(s):
    return (s.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
            .replace("\u00a0", " "))


def clean(line):
    """A line without list marks, links, URLs, HTML tags, code ticks and emphasis marks."""
    line = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
    line = re.sub(r"https?://\S+", "", line)
    line = re.sub(r"<[A-Za-z/!][^>]*>", "", line)
    line = line.replace("`", "").replace("**", "").replace("__", "")
    return re.sub(r"(?<!\w)\*(\S[^*]*?)\*(?!\w)", r"\1", line)


def to_paragraphs(numbered):
    """[{"line", "text", "words"}] from [(line number, raw line)], a blank or skipped line ending a paragraph."""
    paras, buf, start = [], [], None

    def flush():
        text = " ".join(buf).strip()
        words = len(WORD_RE.findall(text))
        if text and words:
            paras.append(dict(line=start, text=text, words=words))

    for no, raw in numbered:
        line = normalize(raw)
        if not line.strip() or SKIP_RE.match(line):
            flush()
            buf, start = [], None
            continue
        if start is None:
            start = no
        buf.append(clean(line))
    flush()
    return paras


def load_samples(path, exclude=()):
    """(kept, dropped) samples from a samples file, each {"num", "title", "source", "lines": [(line no, text)]}.
    A sample whose number is in `exclude`, or whose Source starts with "ai", is dropped."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    samples, cur = [], None
    for i, raw in enumerate(lines, 1):
        m = SAMPLE_RE.match(raw)
        if m:
            cur = dict(num=int(m.group(1)), title=raw.lstrip("# ").strip(), source="raw", lines=[])
            samples.append(cur)
            continue
        if cur is None:
            continue
        meta = META_RE.match(raw)
        if meta:
            if meta.group(1).lower() == "source":
                cur["source"] = meta.group(2).strip().lower()
            continue
        cur["lines"].append((i, raw))
    exclude = set(exclude)
    kept = [s for s in samples if s["num"] not in exclude and not s["source"].startswith("ai")]
    dropped = [s for s in samples if s not in kept]
    return kept, dropped


def text_lines(text):
    """[(line number, line)] for a text, a front-matter block between "---" lines at the top left out."""
    lines = text.split("\n")
    first = 0
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                first = j + 1
                break
    return [(i + 1, lines[i]) for i in range(first, len(lines))]


# ---------------------------------------------------------------- measuring
def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def chunk(paras):
    """Paragraphs grouped into chunks of at least CHUNK_WORDS words; a short tail joins the chunk before it."""
    chunks, cur, count = [], [], 0
    for p in paras:
        cur.append(p)
        count += p["words"]
        if count >= CHUNK_WORDS:
            chunks.append(cur)
            cur, count = [], 0
    if cur:
        if chunks and count < CHUNK_WORDS / 2:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    return chunks


def tokens(paras):
    return [w.lower() for p in paras for w in WORD_RE.findall(p["text"])]


def freq_vector(words, vocab):
    c, n = Counter(words), max(len(words), 1)
    return [c[w] / n for w in vocab]


def burrows_delta(vec, others):
    total, k = 0.0, 0
    for x, col in zip(vec, zip(*others)):
        sd = statistics.pstdev(col)
        if sd == 0:
            continue
        total += abs((x - statistics.mean(col)) / sd)
        k += 1
    return total / k if k else float("nan")


# ---------------------------------------------------------------- the baseline
def build_baseline(samples_path, exclude=()):
    """The baseline from a samples file: the most common function words, one frequency vector per chunk, and each
    chunk's Delta against the rest (its spread is what a text gets compared with)."""
    kept, dropped = load_samples(samples_path, exclude)
    numbered = [nl for s in kept for nl in s["lines"] + [(0, "")]]
    paras = to_paragraphs(numbered)
    chunks = chunk(paras)
    freq = Counter(w for w in tokens(paras) if w in FUNCTION_WORDS)
    vocab = [w for w, _ in freq.most_common(TOP_FUNCTION_WORDS)]
    vecs = [freq_vector(tokens(c), vocab) for c in chunks]
    loo = [burrows_delta(vecs[i], vecs[:i] + vecs[i + 1:]) for i in range(len(vecs))] if len(vecs) > 2 else []
    return dict(vocab=vocab, vecs=vecs, loo=loo, words=sum(p["words"] for p in paras),
                samples=[s["num"] for s in kept], dropped=[s["num"] for s in dropped])


def stamp(path):
    st = Path(path).stat()
    return [st.st_size, int(st.st_mtime)]


def default_cache(samples_path):
    return Path(str(samples_path) + ".baseline.json")


def cached_baseline(samples_path, cache=None, exclude=()):
    """build_baseline, read from the cache file while the samples file's size and time match what it was built
    from; otherwise built again and the cache rewritten."""
    cache = Path(cache) if cache else default_cache(samples_path)
    key = {"samples": stamp(samples_path), "exclude": sorted(exclude)}
    if cache.exists():
        try:
            got = json.loads(cache.read_text(encoding="utf-8"))
            if got.get("key") == key:
                return got["baseline"]
        except (ValueError, OSError, KeyError):
            pass
    b = build_baseline(samples_path, exclude)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps({"key": key, "baseline": b}), encoding="utf-8")
    tmp.replace(cache)
    return b


def delta_line(text, baseline):
    """The Delta line for a text against a baseline, or "" when the baseline has too few chunks to judge by."""
    loo = [x for x in (baseline or {}).get("loo", []) if not math.isnan(x)]
    if not loo or not baseline.get("vocab"):
        return ""
    paras = to_paragraphs(text_lines(normalize(text)))
    vec = freq_vector(tokens(paras), baseline["vocab"])
    delta = burrows_delta(vec, baseline["vecs"])
    if math.isnan(delta):
        return ""
    p90 = pct(loo, 0.9)
    verdict = "inside your range" if delta <= p90 else "further from you than 90% of your own chunks"
    return f"DELTA (soft): {delta:.2f} vs your chunks median {statistics.median(loo):.2f}, p90 {p90:.2f}: {verdict}"


def _setting(name):
    try:
        from engine.settings import path_setting
        return path_setting(name)
    except ImportError:
        value = os.environ.get(name, "").strip()
        return Path(value) if value else None


def audit(text, samples=None, cache=None):
    """The Delta line for a text against your samples (SAMPLES_FILE by default), or "" with none set or usable."""
    samples = samples or _setting("SAMPLES_FILE")
    if not samples or not Path(samples).exists():
        return ""
    try:
        return delta_line(text, cached_baseline(samples, cache or _setting("VOICE_BASELINE")))
    except (OSError, ValueError, statistics.StatisticsError):
        return ""


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))          # so engine/settings.py can be read when run from this folder
    ap = argparse.ArgumentParser(description="How close a text sits to your own samples (Burrows' Delta).")
    ap.add_argument("draft", nargs="?", help="a text file, or - for standard input")
    ap.add_argument("--samples", help="your samples file (default: SAMPLES_FILE)")
    ap.add_argument("--cache", help="the baseline cache (default: VOICE_BASELINE, else next to the samples)")
    ap.add_argument("--baseline", action="store_true", help="build or refresh the cache and report it")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    samples = a.samples or _setting("SAMPLES_FILE")
    cache = a.cache or _setting("VOICE_BASELINE")
    if a.baseline:
        if not samples:
            ap.error("no samples file: pass --samples or set SAMPLES_FILE")
        b = cached_baseline(samples, cache)
        print(f"{len(b['samples'])} samples, {b['words']} words, {len(b['vecs'])} chunks"
              + ("" if b["loo"] else f" (Delta needs at least {MIN_CHUNKS} chunks)"))
        return
    if not a.draft:
        ap.error("give a draft path (or -), or --baseline")
    if a.draft == "-":
        sys.stdin.reconfigure(encoding="utf-8")
        text = sys.stdin.read()
    else:
        with open(a.draft, encoding="utf-8") as f:
            text = f.read()
    line = audit(text, samples, cache)
    if line:
        print(line)


if __name__ == "__main__":
    main()
