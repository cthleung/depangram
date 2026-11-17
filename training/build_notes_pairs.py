"""Builds the pairs for the notes-to-text adapter (plan 6.9, Phase 3, second training).

Every stage reads and writes in the data folder you name with --data (your corpus; nothing ships with the repo):

  python -m training.build_notes_pairs normalize --data DIR    # DIR/passages.jsonl -> DIR/passages_notes.jsonl
  python -m training.build_notes_pairs notes --data DIR [--budget USD]   # Sonnet notes per passage, cached, resumable
  python -m training.build_notes_pairs pairs --data DIR        # -> DIR/pairs_notes.jsonl
  python -m training.build_notes_pairs stats --data DIR [--file pairs_notes2.jsonl]

The retrain with registers:

  python -m training.build_notes_pairs essays --data DIR --essays FOLDER   # FOLDER/*.txt -> DIR/passages_essays.jsonl
  python -m training.build_notes_pairs guide --data DIR --job NAME   # the job's all-yours paragraphs + the loop's notes
  python -m training.build_notes_pairs notes --data DIR --estimate   # what the notes still to write would cost
  python -m training.build_notes_pairs notes --data DIR [--budget USD]   # passages still without notes
  python -m training.build_notes_pairs pairs2 --data DIR       # -> DIR/pairs_notes2.jsonl, every row with a register

The data folder holds passages.jsonl (one passage per line: id, source, text, words) and pairs.jsonl (paraphrase
pairs: id, source, ok, paraphrased_text, text); the stages write the rest next to them.

Why this exists: the first adapter trained on paraphrase -> original pairs with chat as 70% of the corpus, and
came out an editor that copies its source and writes lowercase texting with typos.
This run drops the chat, normalizes the targets to the document's register (capitalized sentence starts and "I",
straight quotes, American spelling, known typos fixed; slang, swearing, fragments and run-ons untouched) and
trains the job the loop actually does: a notes list in, the author's paragraph out. The notes come from the loop's
own NOTES_PROMPT (engine/job.py), with its preamble changed to describe the corpus, so the adapter learns the same
format Sonnet writes for the document. One row in four is the old paraphrase pair
for the same passage, with its target normalized the same way, so "same content, your words" is still learned.

The second build (pairs2) adds the author's essays and the all-yours paragraphs of the document, and tags every
row with its register (reddit, forum, essay, guide). The register rides as the first bullet of the notes block
("- register: reddit"), so the live adapter's prompt format is a subset of the new one and the loop doesn't change
until the author swaps adapters. Paraphrase rows have no notes block; they carry the register field only.

The lists that belong to one corpus (essay files to skip, drafts to prefer, paragraphs to drop, typos to fix) are
empty here; fill them in for your own corpus.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.job import NOTES_PROMPT, sha  # noqa: E402
from engine.settings import Settings  # noqa: E402
from engine.util import describe  # noqa: E402

OUT = None                  # the data folder (--data); set_data() fills in every path below
PASSAGES = PAIRS = NORMALIZED = NOTES_CACHE = PAIRS_OUT = None
SOURCES = ("reddit", "samples")
PARAPHRASE_EVERY = 4
NOTES_KEY = "notes-corpus-v1|"

# the retrain with registers
ESSAYS = ESSAY_PASSAGES = GUIDE_PASSAGES = PAIRS_OUT2 = None
JOB = None                  # the job whose all-yours paragraphs the guide stage reads (--job)


def set_data(folder, essays=None, job=None):
    """Point every stage at the data folder `folder` (and the essays folder, and the job for the guide stage)."""
    global OUT, PASSAGES, PAIRS, NORMALIZED, NOTES_CACHE, PAIRS_OUT, ESSAYS, ESSAY_PASSAGES, GUIDE_PASSAGES
    global PAIRS_OUT2, JOB
    OUT = Path(folder)
    PASSAGES, PAIRS = OUT / "passages.jsonl", OUT / "pairs.jsonl"
    NORMALIZED, NOTES_CACHE, PAIRS_OUT = OUT / "passages_notes.jsonl", OUT / "notes cache.json", OUT / "pairs_notes.jsonl"
    ESSAYS = Path(essays) if essays else OUT / "essays"
    ESSAY_PASSAGES, GUIDE_PASSAGES = OUT / "passages_essays.jsonl", OUT / "passages_guide.jsonl"
    PAIRS_OUT2 = OUT / "pairs_notes2.jsonl"
    JOB = job


ESSAY_KEY = "notes-essay-v1|"
REGISTER = {"reddit": "reddit", "samples": "forum", "essays": "essay", "guide": "guide"}
MIN_WORDS = 40                      # the Reddit passages' floor (build_pairs.py); shorter essay paragraphs join the next
MAX_WORDS = 350
# measured on a notes run: $4.96 for 786 passages, 85,426 words
NOTES_USD_PER_WORD = 4.96 / 85426
# essay files that aren't essay prose (school notes: fragments and page references) or are earlier drafts of a piece
# whose later draft is in; file names, one corpus's own
ESSAY_SKIP_FILES = set()
# drafts of one piece: the later or corrected draft goes first, so the dedup keeps it
ESSAY_FIRST = []
# paragraphs that open with one of these aren't the author's prose (an exam question quoted at the top, say)
ESSAY_SKIP_PARAS = ()
DUP_RATIO = 0.8

PREAMBLE_OLD = describe("Below are numbered paragraphs from <<DESCRIPTION>>.")
PREAMBLE_NEW = ("Below are numbered passages by one writer: comments, posts and messages, casual and "
                "opinionated.")
CORPUS_NOTES_PROMPT = NOTES_PROMPT.replace(PREAMBLE_OLD, PREAMBLE_NEW)
assert CORPUS_NOTES_PROMPT != NOTES_PROMPT, "NOTES_PROMPT's preamble changed; update PREAMBLE_OLD"
PREAMBLE_ESSAY = ("Below are numbered passages by one writer: analytical essays written for school, formal "
                  "register.")
ESSAY_NOTES_PROMPT = NOTES_PROMPT.replace(PREAMBLE_OLD, PREAMBLE_ESSAY)

QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": " - "})
TYPOS = {}                          # known typos in the corpus, as {typo: fix}; one corpus's own
# British -> American, whole words, case kept by apply_case()
AMERICAN = {"criticised": "criticized", "criticise": "criticize", "analysing": "analyzing", "analyse": "analyze",
            "realise": "realize", "realised": "realized", "realising": "realizing", "organise": "organize",
            "organised": "organized", "organising": "organizing", "recognise": "recognize", "recognised": "recognized",
            "colour": "color", "colours": "colors", "favourite": "favorite", "favourites": "favorites",
            "flavour": "flavor", "behaviour": "behavior", "honour": "honor", "humour": "humor", "rumour": "rumor",
            "centre": "center", "metre": "meter", "metres": "meters", "theatre": "theater", "defence": "defense",
            "offence": "offense", "licence": "license", "practise": "practice", "grey": "gray", "travelling": "traveling",
            "travelled": "traveled", "cancelled": "canceled", "modelling": "modeling", "labelled": "labeled",
            "programme": "program", "catalogue": "catalog", "dialogue": "dialogue", "judgement": "judgment",
            "ageing": "aging", "aluminium": "aluminum", "tyre": "tire", "tyres": "tires", "cheque": "check",
            "mum": "mom", "arse": "ass", "whilst": "while", "amongst": "among", "learnt": "learned", "spelt": "spelled",
            "dreamt": "dreamed", "burnt": "burned", "apologise": "apologize", "apologised": "apologized",
            "specialised": "specialized", "emphasise": "emphasize", "minimise": "minimize", "maximise": "maximize",
            "optimise": "optimize", "optimised": "optimized", "summarise": "summarize", "utilise": "utilize",
            "paralysed": "paralyzed", "neighbour": "neighbor", "neighbours": "neighbors", "armour": "armor",
            "endeavour": "endeavor", "savour": "savor", "vigour": "vigor", "fibre": "fiber", "litre": "liter",
            "sceptical": "skeptical", "pyjamas": "pajamas", "jewellery": "jewelry", "manoeuvre": "maneuver",
            "plough": "plow", "mould": "mold", "moult": "molt", "smoulder": "smolder", "storey": "story",
            "kerb": "curb", "pretence": "pretense", "enrol": "enroll", "fulfil": "fulfill", "skilful": "skillful",
            "instalment": "installment", "doughnut": "donut", "cosy": "cozy", "draught": "draft", "gaol": "jail",
            "mediaeval": "medieval", "encyclopaedia": "encyclopedia", "foetus": "fetus", "haemorrhage": "hemorrhage",
            "oestrogen": "estrogen", "paediatric": "pediatric", "anaemia": "anemia", "diarrhoea": "diarrhea",
            "aeroplane": "airplane", "aluminium": "aluminum", "maths": "math",
            # found in essays
            "emphasising": "emphasizing", "emphasised": "emphasized", "cognisant": "cognizant",
            "revelling": "reveling", "favour": "favor", "favours": "favors", "favoured": "favored"}
AMERICAN = {k: v for k, v in AMERICAN.items() if k != v}


def apply_case(src, repl):
    if src.isupper():
        return repl.upper()
    if src[0].isupper():
        return repl[0].upper() + repl[1:]
    return repl


def american(text):
    def sub(m):
        w = m.group(0)
        r = AMERICAN.get(w.lower())
        return apply_case(w, r) if r else w
    return re.sub(r"[A-Za-z]+", sub, text)


def capitalize(text):
    """Sentence starts and the pronoun I. Leaves everything after a URL, a code span or inside a quote alone
    beyond the first letter; never touches a word that's already capitalized."""
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append(para)
            continue
        # the pronoun I and its contractions
        para = re.sub(r"(?<![\w'])i(?=(?:'m|'ve|'ll|'d)?\b)", "I", para)
        # the first letter of the paragraph and the first letter after . ! ? (plus closing quotes or brackets)
        def up(m):
            return m.group(1) + m.group(2).upper()
        para = re.sub(r'(^|[.!?]["\')\]]*\s+|^[>\-*\s]*)([a-z])', up, para, count=0)
        out.append(para)
    return "\n".join(out)


REDDIT_JUNK = re.compile("&#x200B;|&nbsp;|[​‌‍﻿]")
EDIT_PARA = re.compile(r"^\s*(?:edit|EDIT|Edit)\s*\d*\s*[:\-]", re.M)


def strip_reddit(text):
    """Reddit's zero-width entities and trailing "Edit:" paragraphs (a three-epoch adapter learned to write
    both). An edit paragraph is dropped only when it comes last, so a passage that is one edit stays whole."""
    t = REDDIT_JUNK.sub("", text)
    parts = re.split(r"\n\s*\n", t.strip())
    while len(parts) > 1 and EDIT_PARA.match(parts[-1]):
        parts.pop()
    return "\n\n".join(parts)


def normalize(text):
    t = strip_reddit(text).translate(QUOTES)
    for a, b in TYPOS.items():
        t = re.sub(re.escape(a), b, t, flags=re.I)
    t = american(t)
    t = capitalize(t)
    return t.strip()


def passages():
    rows = [json.loads(l) for l in open(PASSAGES, encoding="utf-8")]
    return [r for r in rows if r["source"] in SOURCES]


def do_normalize():
    rows = passages()
    changed = 0
    with open(NORMALIZED, "w", encoding="utf-8") as f:
        for r in rows:
            n = normalize(r["text"])
            changed += n != r["text"]
            f.write(json.dumps({**r, "raw": r["text"], "text": n}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} passages normalized ({changed} changed) -> {NORMALIZED}")


def load_cache():
    return json.loads(NOTES_CACHE.read_text(encoding="utf-8")) if NOTES_CACHE.exists() else {}


def notes_jobs():
    """(rows, prompt, key) per batch of passages: the normalized corpus with the corpus preamble, the essays with
    their own preamble and key. The guide rows bring the loop's own notes (do_guide), so they're not here."""
    jobs = []
    if NORMALIZED.exists():
        jobs.append(([json.loads(l) for l in open(NORMALIZED, encoding="utf-8")], CORPUS_NOTES_PROMPT, NOTES_KEY))
    if ESSAY_PASSAGES.exists():
        jobs.append(([json.loads(l) for l in open(ESSAY_PASSAGES, encoding="utf-8")], ESSAY_NOTES_PROMPT, ESSAY_KEY))
    return jobs


def notes_todo(jobs, cache):
    return [(r, prompt, key) for rows, prompt, key in jobs for r in rows if sha(key + r["text"]) not in cache]


def estimate_usd(todo):
    return sum(r["words"] for r, _, _ in todo) * NOTES_USD_PER_WORD


def notes_chunks(todo, size=10):
    """Chunks of at most `size`, never mixing a corpus passage with an essay (one prompt per call)."""
    chunks, cur = [], []
    for item in todo:
        if cur and (len(cur) == size or cur[-1][2] != item[2]):
            chunks.append(cur)
            cur = []
        cur.append(item)
    if cur:
        chunks.append(cur)
    return chunks


def parse_notes(out):
    """Sonnet's reply ("### 1" then "- note" lines) -> {1: [notes], ...}."""
    blocks = re.split(r"^###\s*(\d+)\s*$", out, flags=re.M)
    return {int(blocks[k]): [ln[2:].strip() for ln in blocks[k + 1].splitlines() if ln.startswith("- ")]
            for k in range(1, len(blocks) - 1, 2)}


def do_notes(budget, estimate=False):
    cache = load_cache()
    todo = notes_todo(notes_jobs(), cache)
    print(f"{len(todo)} passages need notes ({sum(r['words'] for r, _, _ in todo)} words), about"
          f" ${estimate_usd(todo):.2f} at the first run's rate; budget ${budget:.2f}")
    if estimate or not todo:
        return
    from engine import claude_call, db
    settings = Settings()
    conn = db.connect(settings.db_path)
    spent, t0, done = 0.0, time.time(), 0
    for n, chunk in enumerate(notes_chunks(todo)):
        if spent >= budget:
            print(f"budget ${budget:.2f} reached after {done} passages")
            break
        prompt, key = chunk[0][1], chunk[0][2]
        listing = "\n\n".join(f"{k + 1}. {r['text']}" for k, (r, _, _) in enumerate(chunk))
        out, meta = claude_call.run(settings, conn, prompt.replace("<<PARAS>>", listing),
                                    role="hip notes", model="sonnet", max_budget_usd=0.6)
        spent += meta.get("cost_usd") or 0.0
        got = parse_notes(out)
        for k, (r, _, _) in enumerate(chunk):
            if got.get(k + 1):
                cache[sha(key + r["text"])] = got[k + 1]
            else:
                print(f"  no notes for passage {r['id']} in this chunk; it'll be retried next run")
        done += len(chunk)
        tmp = NOTES_CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(NOTES_CACHE)
        if n % 5 == 4:
            print(f"  {done}/{len(todo)}  ${spent:.2f}  {(time.time() - t0) / 60:.0f} min")
    print(f"done: {len(cache)} passages have notes, ${spent:.2f} this run")


def do_pairs():
    rows = [json.loads(l) for l in open(NORMALIZED, encoding="utf-8")]
    cache = load_cache()
    para = {}
    for l in open(PAIRS, encoding="utf-8"):
        d = json.loads(l)
        if d["source"] in SOURCES and d["ok"]:
            para[d["id"]] = d["paraphrased_text"]
    n_notes = n_para = skipped = 0
    with open(PAIRS_OUT, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            notes = cache.get(sha(NOTES_KEY + r["text"]))
            if not notes:
                skipped += 1
                continue
            src = "<notes>\n" + "\n".join(f"- {n}" for n in notes) + "\n</notes>"
            f.write(json.dumps({"id": r["id"], "source": r["source"], "where": r["where"], "words": r["words"],
                                "kind": "notes", "text": r["text"], "notes": notes, "prompt_source": src, "ok": True},
                               ensure_ascii=False) + "\n")
            n_notes += 1
            if i % PARAPHRASE_EVERY == 0 and r["id"] in para:
                src2 = "<source_text>\n" + para[r["id"]].translate(QUOTES).strip() + "\n</source_text>"
                f.write(json.dumps({"id": r["id"] + "-p", "source": r["source"], "where": r["where"], "words": r["words"],
                                    "kind": "paraphrase", "text": r["text"], "prompt_source": src2, "ok": True},
                                   ensure_ascii=False) + "\n")
                n_para += 1
    print(f"{n_notes} notes rows + {n_para} paraphrase rows -> {PAIRS_OUT} ({skipped} passages without notes skipped)")


# ---------------------------------------------------------------------- the retrain with registers
def normalize_essay(text):
    """The essays keep their formal register: straight quotes, spaced hyphens for dashes, American spelling. No
    capitalize() (they're cased already, and it would capitalize after "e.g." and "p.312"), no Reddit strip."""
    return american(text.translate(QUOTES)).strip()


SENTENCE_END = re.compile(r"[.!?][\"')\]’”]*$")


def essay_paragraphs(folder):
    """(file, paragraph) in reading order, preferred drafts first; school notes, titles, the exam question and
    near-duplicate paragraphs (the same piece in two drafts) left out."""
    import difflib
    files = sorted(p for p in Path(folder).glob("*.txt") if p.name not in ESSAY_SKIP_FILES)
    files.sort(key=lambda p: ESSAY_FIRST.index(p.name) if p.name in ESSAY_FIRST else len(ESSAY_FIRST))
    kept = []
    for f in files:
        for para in re.split(r"\n\s*\n", f.read_text(encoding="utf-8")):
            para = " ".join(para.split())
            if not para or not SENTENCE_END.search(para):
                continue                          # a title or a heading ("Rationale")
            if para.startswith(ESSAY_SKIP_PARAS):
                continue
            if any(difflib.SequenceMatcher(None, para, q).quick_ratio() > DUP_RATIO
                   and difflib.SequenceMatcher(None, para, q).ratio() > DUP_RATIO for _, q in kept):
                continue
            kept.append((f.name, para))
    return kept


def essay_passages(folder):
    """One passage per paragraph, inside the Reddit passages' size range: a paragraph under MIN_WORDS joins the next
    one in the same essay (a short last one joins the one before), and no passage grows past MAX_WORDS."""
    import hashlib
    groups = []
    for fname, para in essay_paragraphs(folder):
        g = groups[-1] if groups else None
        if g and g[0] == fname and len(" ".join(g[1]).split()) < MIN_WORDS \
                and len(" ".join(g[1] + [para]).split()) <= MAX_WORDS:
            g[1].append(para)
        else:
            groups.append((fname, [para]))
    merged = []
    for fname, paras in groups:
        if merged and merged[-1][0] == fname and len(" ".join(paras).split()) < MIN_WORDS \
                and len(" ".join(merged[-1][1] + paras).split()) <= MAX_WORDS:
            merged[-1][1].extend(paras)
        else:
            merged.append((fname, list(paras)))
    out = []
    for fname, paras in merged:
        text = normalize_essay("\n\n".join(paras))
        if len(text.split()) < MIN_WORDS:
            continue                              # still short with its neighbors (a one-line piece)
        out.append({"id": "e" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:11], "source": "essays",
                    "where": fname, "words": len(text.split()), "text": text})
    return out


def do_essays():
    rows = essay_passages(ESSAYS)
    with open(ESSAY_PASSAGES, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    w = [r["words"] for r in rows]
    print(f"{len(rows)} essay passages, {sum(w)} words (shortest {min(w)}, longest {max(w)}) -> {ESSAY_PASSAGES}")


def guide_rows(job, verdicts=None):
    """The guide paragraphs that are entirely the author's: every sentence "yours" in `your lines.md` (not "yours?",
    "edited" or "both"), cut from the frozen original with the AI example lines out. Each brings the loop's own
    notes from the job's notes cache (NOTES_VERSION), the notes the adapter sees at inference; a paragraph the loop
    never noted is left out and returned separately."""
    from engine.job import NOTES_VERSION
    from engine.texts import guide_sections, prose_only, strip_markdown
    from engine.draws import sents
    vm = verdicts if verdicts is not None else job.verdict_map()
    cache = json.loads(job.notes_cache.read_text(encoding="utf-8")) if job.notes_cache.exists() else {}
    out, missing = [], []
    for num, body in guide_sections(range(1, 100), path=job.original).items():
        for para in re.split(r"\n\s*\n", prose_only(body, job.example_lines())):
            para = para.strip()
            if not para or para[0] in "#-<|>":
                continue
            text = strip_markdown(para)
            ss = sents(text) if text else []
            if not ss or not all(vm.get(x) == "yours" for x in ss):
                continue
            notes = cache.get(sha(NOTES_VERSION + text))
            row = {"id": "g" + sha(text)[:11], "source": "guide", "where": f"section {num}",
                   "words": len(text.split()), "text": text.translate(QUOTES), "notes": notes}
            (out if notes else missing).append(row)
    return out, missing


def do_guide():
    from engine.job import Job
    rows, missing = guide_rows(Job(JOB))
    with open(GUIDE_PASSAGES, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} all-yours guide paragraphs, {sum(r['words'] for r in rows)} words -> {GUIDE_PASSAGES}"
          f" ({len(missing)} left out, no loop notes: {', '.join(r['where'] for r in missing) or 'none'})")


def render_notes(notes, register=None):
    """The adapter's notes block. With a register it's the first bullet, so a prompt without one (the live loop's)
    is the same format minus a line."""
    lines = ([f"- register: {register}"] if register else []) + [f"- {n}" for n in notes]
    return "<notes>\n" + "\n".join(lines) + "\n</notes>"


def pair_rows(corpus, essays, guide, cache, para):
    """Every pairs2 row in order, and how many passages had no notes. Corpus rows keep the first build's choice of
    paraphrase rows (one passage in PARAPHRASE_EVERY, counted over the corpus alone)."""
    rows, skipped = [], 0
    for i, r in enumerate(corpus):
        notes = cache.get(sha(NOTES_KEY + r["text"]))
        if not notes:
            skipped += 1
            continue
        reg = REGISTER[r["source"]]
        base = {"id": r["id"], "source": r["source"], "register": reg, "where": r["where"], "words": r["words"],
                "text": r["text"], "ok": True}
        rows.append({**base, "kind": "notes", "notes": notes, "prompt_source": render_notes(notes, reg)})
        if i % PARAPHRASE_EVERY == 0 and r["id"] in para:
            src2 = "<source_text>\n" + para[r["id"]].translate(QUOTES).strip() + "\n</source_text>"
            rows.append({**base, "id": r["id"] + "-p", "kind": "paraphrase", "prompt_source": src2})
    for r in list(essays) + list(guide):
        notes = r.get("notes") or cache.get(sha(ESSAY_KEY + r["text"]))
        if not notes:
            skipped += 1
            continue
        reg = REGISTER[r["source"]]
        rows.append({"id": r["id"], "source": r["source"], "register": reg, "where": r["where"], "words": r["words"],
                     "text": r["text"], "ok": True, "kind": "notes", "notes": notes,
                     "prompt_source": render_notes(notes, reg)})
    return rows, skipped


def count_rows(rows):
    by = {}
    for r in rows:
        t = by.setdefault((r["kind"], r.get("register") or r["source"]), [0, 0])
        t[0] += 1
        t[1] += r["words"]
    return by


def print_counts(by):
    for (kind, reg), (n, w) in sorted(by.items()):
        print(f"  {kind:10} {reg:8} {n:5} rows {w:7} words")


def do_pairs2():
    corpus = [json.loads(l) for l in open(NORMALIZED, encoding="utf-8")]
    essays = [json.loads(l) for l in open(ESSAY_PASSAGES, encoding="utf-8")] if ESSAY_PASSAGES.exists() else []
    guide = [json.loads(l) for l in open(GUIDE_PASSAGES, encoding="utf-8")] if GUIDE_PASSAGES.exists() else []
    para = {}
    for l in open(PAIRS, encoding="utf-8"):
        d = json.loads(l)
        if d["source"] in SOURCES and d["ok"]:
            para[d["id"]] = d["paraphrased_text"]
    rows, skipped = pair_rows(corpus, essays, guide, load_cache(), para)
    with open(PAIRS_OUT2, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {PAIRS_OUT2} ({skipped} passages without notes skipped)")
    print_counts(count_rows(rows))


def do_stats(name=None):
    rows = [json.loads(l) for l in open(OUT / name if name else PAIRS_OUT, encoding="utf-8")]
    print_counts(count_rows(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["normalize", "essays", "guide", "notes", "pairs", "pairs2", "stats"])
    ap.add_argument("--data", required=True, help="the data folder: passages.jsonl and pairs.jsonl in, the rest out")
    ap.add_argument("--essays", help="essays: the folder of .txt essays (default DATA/essays)")
    ap.add_argument("--job", help="guide: the job whose all-yours paragraphs to read (default [job] default)")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--estimate", action="store_true", help="notes: print what's left and its cost, make no calls")
    ap.add_argument("--file", default=None, help="stats: a pairs file in the data folder (default pairs_notes.jsonl)")
    a = ap.parse_args()
    set_data(a.data, a.essays, a.job)
    {"normalize": do_normalize, "essays": do_essays, "guide": do_guide,
     "notes": lambda: do_notes(a.budget, a.estimate), "pairs": do_pairs, "pairs2": do_pairs2,
     "stats": lambda: do_stats(a.file)}[a.stage]()
