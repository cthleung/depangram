"""Your idioms, pulled by code from a folder of your own writing, for the restore pass.

Sentence openers of two to four words, parenthetical asides built from the author's common words, swears, and the
tails a sentence ends on, each as a frequency list. The restore pass hands these to its editor as the only phrases it
may add for bite, so nothing the editor writes for bite is Claude's own. Every entry goes through checks.house_words,
so no banned word reaches a prompt. `cached_idioms` rebuilds the lists only when the raw corpus files change.

The voice material the editor reads next to the idioms (your voice guide, or a stretch of it, and two of your
samples) is read here too, from the files, never pasted by hand.

Where they come from (engine/settings.py), all three empty by default:
    IDIOMS_DIR    a folder of your own writing. Any .txt or .md file in it is read as text, one piece per file.
                  comments.jsonl and posts.jsonl in a forum export's shape are read too: one JSON object per line,
                  "body" for a comment, "title" and "selftext" for a post; "[deleted]" and "[removed]" are skipped.
                  Unset: the idiom lists come out empty.
    VOICE_GUIDE   a markdown file that says how you write. The whole file is used, unless voice_material is given
                  the headings a stretch starts and ends at.
    SAMPLES_FILE  your samples, in the format engine/tics/voice_audit.py reads ("## Sample N" headings).
With VOICE_GUIDE or SAMPLES_FILE unset voice_material returns empty text, so the editor's prompt goes without that
block.
"""
import json
import re
from collections import Counter
from pathlib import Path

from .checks import house_words
from .settings import path_setting

N_OPENERS = 40
N_ASIDES = 40
N_SWEARS = 15
TAILS = ("probably", "haha", "lol", "lmao", "though", "anyway", "anyways", "honestly", "tbh", "imo", "really",
         "actually", "basically", "i guess", "i think", "to be honest", "at least", "for sure", "right?", "i suppose",
         "or something", "or whatever", "and stuff", "and shit", "i mean", "you know", "etc", "i'd say", "i'd think",
         "if that makes sense", "if anything", "in my opinion", "to be fair", "mind you")
VOICE_FROM, VOICE_TO = None, None    # the stretch of the voice guide to use; None is its start and its end
SAMPLE_PICKS = None                   # sample numbers to show; None is the first two in the file
TEXT_SUFFIXES = (".txt", ".md")
# the swear list the idiom extractor counts with (draws.SWEAR, kept here as it was)
SWEAR = re.compile(r"\b(fuck\w*|shit\w*|damn\w*|ass|arse|hell|bloody|crap\w*|bastard\w*|piss\w*|bollocks)\b", re.I)


def _sents(p):
    from engine.draws import sents
    return sents(p)


def text_files(root):
    """The .txt and .md files in a folder, sorted by name."""
    return sorted(p for p in Path(root).iterdir() if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)


def reddit_texts(root=None):
    """Your writing from the idioms folder: comments (body) and posts (title and selftext) in a forum export's
    shape, deleted and removed ones dropped, then every .txt and .md file as one text each. Returns (texts,
    {"comments": n, "posts": n, "files": n}), with only the kinds found counted. `root` defaults to IDIOMS_DIR; with
    none set, ([], {})."""
    from training.build_notes_pairs import strip_reddit
    root = root or path_setting("IDIOMS_DIR")
    if not root or not Path(root).is_dir():
        return [], {}
    out, counts = [], Counter()
    for name, fields in (("comments.jsonl", ("body",)), ("posts.jsonl", ("title", "selftext"))):
        path = Path(root) / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for row in rows:
            got = False
            for f in fields:
                t = (row.get(f) or "").strip()
                if t and t not in ("[deleted]", "[removed]"):
                    out.append(strip_reddit(t))
                    got = True
            counts[name.split(".")[0]] += got
    for path in text_files(root):
        t = path.read_text(encoding="utf-8", errors="replace").strip()
        if t:
            out.append(t)
            counts["files"] += 1
    return out, dict(counts)


def _plain(text):
    return text.replace("’", "'").replace("‘", "'").lower()


def _clean_words(text):
    return re.findall(r"[a-z][a-z']*", _plain(text))


def _usable(phrase):
    """A phrase the editor may be handed: plain letters, no link, no digit, no house word."""
    return (bool(phrase) and "http" not in phrase and not re.search(r"\d", phrase) and not house_words(phrase)
            and re.fullmatch(r"[a-z' ,;:!?/-]+", phrase) is not None)


def _names(texts):
    """Words capitalized mid-sentence more often than not: names, clubs, products."""
    cap, low = Counter(), Counter()
    for t in texts:
        for m in re.finditer(r"(?<=[a-z,;] )([A-Z][a-z]+)", t):
            cap[m.group(1).lower()] += 1
        for w in re.findall(r"\b[a-z]+\b", t):
            low[w] += 1
    return {w for w, n in cap.items() if n >= 2 and n > low.get(w, 0)}


def extract_idioms(texts, n_openers=N_OPENERS, n_asides=N_ASIDES, common_words=1000):
    """Your idioms as frequency lists, lowercased and counted: sentence openers of two to four words (a longer
    opener is kept only when no shorter one on the list is its prefix, and none carries a name), parenthetical asides
    built from your common words (an aside that repeats is content, "(right-hand drive)"; the idiom-shaped ones
    are the short ones made of everyday words, "(to me)", "(whatever that means)"), swears, and sentence tails.
    Returns {"openers", "asides", "swears", "tails", "sentences", "words"} with [phrase, count] pairs."""
    openers, asides, swears, tails, vocab = Counter(), Counter(), Counter(), Counter(), Counter()
    names = _names(texts)
    n_sent = n_words = 0
    for t in texts:
        for inner in re.findall(r"\(([^()]{1,120})\)", t):
            a = " ".join(_plain(inner).split()).strip(" .,;:!?")
            if _usable(a) and 1 <= len(a.split()) <= 6:
                asides[a] += 1
        for w in SWEAR.findall(t):
            swears[w.lower()] += 1
        for para in re.split(r"\n\s*\n", t):
            for s in _sents(para.strip()):
                words = _clean_words(s)
                if not words:
                    continue
                n_sent += 1
                n_words += len(words)
                vocab.update(words)
                for k in (2, 3, 4):
                    if len(words) > k + 1:
                        openers[" ".join(words[:k])] += 1
                low = " ".join(_plain(s).split()).rstrip(" .!,;:")
                for tail in TAILS:
                    probe = low if tail.endswith("?") else low.rstrip("?")
                    if probe.endswith(" " + tail) or probe == tail:
                        tails[tail] += 1
    common = {w for w, _ in vocab.most_common(common_words)} | set(TAILS) | {w for w, _ in swears.items()}
    chosen = []
    for phrase, n in sorted(openers.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0])):
        if not _usable(phrase) or any(w in names for w in phrase.split()):
            continue
        if any(c == phrase or phrase.startswith(c + " ") or c.startswith(phrase + " ") for c, _ in chosen):
            continue
        chosen.append([phrase, n])
        if len(chosen) >= n_openers:
            break
    one_word_ok = set(TAILS) | set(swears)
    idiom_asides = [(a, n) for a, n in asides.items()
                    if all(w in common and w not in names for w in re.findall(r"[a-z']+", a))
                    and (len(a.split()) >= 2 or a in one_word_ok)]
    top_asides = [[a, n] for a, n in sorted(idiom_asides, key=lambda kv: (-kv[1], len(kv[0]), kv[0]))][:n_asides]
    return {"openers": chosen, "asides": top_asides,
            "swears": [[w, n] for w, n in swears.most_common(N_SWEARS)],
            "tails": [[w, n] for w, n in sorted(tails.items(), key=lambda kv: (-kv[1], kv[0])) if n >= 2],
            "sentences": n_sent, "words": n_words}


def idiom_block(idioms):
    """The idiom list as the editor sees it."""
    def line(label, pairs):
        return f"{label}: " + "; ".join(p for p, _ in pairs)
    return "\n".join([line("Sentence openers (most frequent first)", idioms["openers"]),
                      line("Parenthetical asides", [[f"({a})", n] for a, n in idioms["asides"]]),
                      line("Swears", idioms["swears"]),
                      line("Tails a sentence ends on", idioms["tails"])])


def corpus_stamp(root=None):
    """What the cache is keyed on: each corpus file's size and modification time ({} with no IDIOMS_DIR)."""
    root = root or path_setting("IDIOMS_DIR")
    out = {}
    if not root or not Path(root).is_dir():
        return out
    for p in [Path(root) / n for n in ("comments.jsonl", "posts.jsonl")] + text_files(root):
        if p.exists():
            st = p.stat()
            out[p.name] = [st.st_size, int(st.st_mtime)]
    return out


def cached_idioms(cache_path, root=None, log=None):
    """The idiom lists, rebuilt from the raw corpus only when its files changed since the cache was written."""
    root = root or path_setting("IDIOMS_DIR")
    cache_path = Path(cache_path)
    stamp = corpus_stamp(root)
    if cache_path.exists():
        try:
            got = json.loads(cache_path.read_text(encoding="utf-8"))
            if got.get("stamp") == stamp and got.get("idioms"):
                return got["idioms"]
        except (ValueError, OSError):
            pass
    texts, counts = reddit_texts(root)
    idioms = extract_idioms(texts)
    idioms["corpus"] = counts
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"stamp": stamp, "idioms": idioms, "as_given": idiom_block(idioms)}, indent=1,
                              ensure_ascii=False), encoding="utf-8")
    tmp.replace(cache_path)
    if log:
        log(f"idioms rebuilt from {counts}: {len(idioms['openers'])} openers, {len(idioms['asides'])} asides, "
            f"{len(idioms['swears'])} swears, {len(idioms['tails'])} tails from {idioms['sentences']} sentences")
    return idioms


# ---------------------------------------------------------------------- the voice material
def voice_section(guide_text, start=VOICE_FROM, end=VOICE_TO):
    """The stretch of the voice guide from the line holding `start` up to the one holding `end`. A missing or
    unfound `start` means the top of the file, and a missing or unfound `end` its bottom."""
    i = guide_text.find(start) if start else 0
    i = max(i, 0)
    j = guide_text.find(end, i) if end else -1
    return guide_text[i:j if j >= 0 else len(guide_text)].strip()


def sample_numbers(samples_text):
    """The sample numbers in a samples file, in file order."""
    return [int(n) for n in re.findall(r"^##\s+Sample\s+(\d+)\b", samples_text, re.M | re.I)]


def sample(samples_text, n):
    """Sample n's text from the samples file, its header and metadata bullets off."""
    m = re.search(rf"^##\s+Sample\s+{n}\b.*$", samples_text, re.M | re.I)
    if not m:
        raise ValueError(f"sample {n} not found")
    body = samples_text[m.end():]
    nxt = re.search(r"^##\s+Sample\s+\d+\b", body, re.M | re.I)
    body = body[:nxt.start()] if nxt else body
    lines = [ln for ln in body.strip().splitlines() if not ln.startswith("- **") and ln.strip() != "---"]
    return "\n".join(lines).strip()


def voice_material(guide=None, samples=None, picks=SAMPLE_PICKS, start=VOICE_FROM, end=VOICE_TO):
    """(the voice guide or its stretch from `start` to `end`, the samples block) from the files. `guide` and
    `samples` default to VOICE_GUIDE and SAMPLES_FILE; with either unset, ("", ""). `picks` are the sample numbers
    to show; a pick the file doesn't have is skipped, and with none left (or none given) the first two samples are
    shown."""
    guide, samples = guide or path_setting("VOICE_GUIDE"), samples or path_setting("SAMPLES_FILE")
    if not guide or not samples:
        return "", ""
    voice = voice_section(Path(guide).read_text(encoding="utf-8"), start, end)
    text = Path(samples).read_text(encoding="utf-8")
    have = sample_numbers(text)
    chosen = [n for n in (picks or ()) if n in have] or have[:2]
    block = "\n\n".join(f"Sample {n}:\n{sample(text, n)}" for n in chosen)
    return voice, block
