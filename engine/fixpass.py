"""The fix pass: plan 6.5's few-word rule as a last pass over the loop's text.

After the loop regrows paragraphs, the text keeps small slips a reader trips on: a dangling "probably", a pronoun
with nothing to point at, British "organised", a line pointing at a list that sits in another section. The noise
floor test (analytics/noise_floor.md) showed trivial edits don't move Pangram, so a few-word fix on a sentence is
safe. Sonnet reads each paragraph next to its original and proposes fixes; code is the gate and does the writing.

    python -m engine.fixpass                         # dry run over the job's sections in scope
    python -m engine.fixpass --section 1             # one section
    python -m engine.fixpass --job NAME --para "s1w1 p2,s1w1 p3"
    python -m engine.fixpass --section 1 --write     # apply, re-score the section, keep it only if it holds

What the gate holds (code, never Sonnet): a fix rewrites one sentence that stands in the paragraph as written,
into one sentence, with at most fixpass.max_words_changed tokens changed (word-level diff, a punctuation mark
counts as a token), no new swear, no new em dash or banned word, no British spelling brought in, no new name or
number beyond the original paragraph's, no punctuation slip, and the run's must-keeps (hedges, repeats, names,
quotes, numbers, your protected words) still there. Only paragraphs the loop wrote, standing in the real guide
exactly as the loop has them, get fixed.

With --write: each touched section is built in memory with the block-level write-back (engine/wrapup.py), scored
as posted, and kept only when its AI share rose by no more than fixpass.noise. Kept sections go into the real guide
(prose blocks only, everything else byte for byte), the working copy, and the bank as a "fix" entry per window,
which ship_entry treats like a whole-text pick, so the loop ships the fixed text and the write-back and the sync
recognize it as the loop's. A report goes to analytics/ and a line to the fix pass log.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from . import checks as _checks
from .checks import (gb_spellings, house_words, invented_sighting, must_keeps, names_missing, names_required,
                     new_items, punctuation_problem, repeat_ok, to_us, words_changed)
from .job import NOTES_VERSION, Bank, Job, flat, protected_fallback, sha, style_of, window_text_of, word_in
from .settings import Settings
from .texts import COMMENT_RE, strip_markdown
from .util import describe, report_file
from .wrapup import (BLOCK_SPLIT, as_posted, is_prose_block, layout, match_paragraph, norm, run_versions,
                     section_spans, write_back)

FIX_VERSION = "fix-v1|"
CATEGORIES = {"grammar", "pronoun", "spelling", "word order", "fact"}

FIX_TEMPLATE = """Below are numbered paragraphs from <<DESCRIPTION>>. Each is shown as it stands now, after a rewrite, with the original paragraph it was rewritten from. The rewrite left a few small slips. Find only these, sentence by sentence:
- grammar: a slip a careful reader trips on, like a hedge tacked on where it doesn't attach ("the kettle doesn't have, probably"), a word missing or doubled, subject and verb that don't agree
- pronoun: a pronoun with nothing before it to point back to, or pointing at the wrong thing
- spelling: British spelling ("organised", "colour", "labelling"); the text uses American spelling
- word order: a word in the wrong place
- fact: something the rewrite says that drifted from the original paragraph (a number, a name, a claim, or a line pointing at something that isn't there); put back what the original says

Nothing else. The author's voice is not a slip: leave swearing, slang, jokes, rhythm, fragments, "there's" with a plural, sentences starting with "And" or "So", and every word choice you'd merely prefer. Most paragraphs need nothing.

A fix rewrites one sentence and keeps it one sentence, changing at most <<N>> words of it. Copy "sentence" exactly as it stands in the paragraph now, punctuation included.

<<PARAS>>

Reply with only JSON, every paragraph number present, an empty list where there's nothing to fix:
{"1": [{"sentence": "...", "replacement": "...", "category": "grammar", "why": "short reason"}], "2": []}"""
FIX_PROMPT = describe(FIX_TEMPLATE)        # the default description; FixPass fills the job's own


# ---------------------------------------------------------------------- the gate (code only)
def _sents(text):
    from engine.draws import sents
    return sents(text)


def _names(text):
    from engine.draws import names
    return {re.sub(r"(?:'s|s)$", "", n.lower()) for n in names(text)}


def _numbers(text):
    from engine.draws import numbers
    return numbers(text)


def _quoted(text):
    from engine.draws import quoted
    return quoted(text)


def run_keeps(orig_run, notes, author=False, protected=None):
    """What a fix must leave in a run: its original's hedges, repeats, names, quotes and numbers (and for a line of
    yours, its protected words). The gate only holds the ones the run still carries, so a fix that puts back
    what the original said (a fact drift) can drop what the drift brought in."""
    mk = must_keeps(orig_run, notes) if not author else {"hedges": [], "repeats": []}
    return {"hedges": mk["hedges"], "repeats": mk["repeats"], "names": names_required(orig_run, notes),
            "quotes": [q.strip('"').strip(",.;:") for q in _quoted(orig_run)], "numbers": sorted(_numbers(orig_run)),
            "protected": list(protected or []) if author else []}


def keeps_lost(old_run, new_run, keeps):
    """The must-keeps the old run text carries and the new one lost, as short labels."""
    lost = []
    for h in keeps["hedges"]:
        rx = re.compile(r"\b{}\b".format(re.escape(h)), re.I)
        if rx.search(old_run) and not rx.search(new_run):
            lost.append(f'hedge "{h}"')
    if repeat_ok(old_run, keeps["repeats"]) and not repeat_ok(new_run, keeps["repeats"]):
        lost.append("a repeat")
    for n in keeps["names"]:
        if not names_missing(old_run, [n]) and names_missing(new_run, [n]):
            lost.append(f'name "{n}"')
    for q in keeps["quotes"]:
        if q.lower() in old_run.lower() and q.lower() not in new_run.lower():
            lost.append(f'quote "{q[:30]}"')
    for n in keeps["numbers"]:
        if n in _numbers(old_run) and n not in _numbers(new_run):
            lost.append(f"number {n}")
    for w in keeps["protected"]:
        if word_in(w, old_run) and not word_in(w, new_run):
            lost.append(f'protected word "{w}"')
    return lost


def gate(prop, runs, keeps, original, max_words, first=None):
    """One proposal against a paragraph. `runs` is the paragraph's run texts as they stand, `keeps` each run's
    must-keeps, `original` the original paragraph's text. `first` maps a sentence as it stands to the text it had
    before this pass touched it, so two fixes on one sentence count together against the word limit.
    Returns (run index, the run's new text, the new sentence, None) or (None, None, None, reason)."""
    s, r = norm(prop.get("sentence") or ""), norm(prop.get("replacement") or "")
    no = (None, None, None)
    cat = (prop.get("category") or "").strip().lower()
    if cat not in CATEGORIES:
        return (*no, f"not a fix the pass makes ({cat or 'no category'})")
    if not s or not r:
        return (*no, "empty")
    if r == s:
        return (*no, "the replacement is the same sentence")
    ri = next((i for i, t in enumerate(runs) if s in [norm(x) for x in _sents(t)]), None)
    if ri is None:
        para = " ".join(runs)
        return (*no, "the sentence isn't in the paragraph as written" if s not in norm(para)
                else "the sentence spans two runs")
    if len(_sents(r)) != 1:
        return (*no, "adds or removes a sentence")
    base = (first or {}).get(s, s)
    n = words_changed(base, r)
    if n > max_words:
        return (*no, f"changes {n} words (the limit is {max_words})")
    if new_items(s, r, lambda t: _checks.SWEARS.findall(t)):          # the rules in use, read now
        return (*no, "adds a swear")
    if new_items(s, r, house_words):
        return (*no, "adds an em dash or a banned word")
    if new_items(s, r, gb_spellings):
        return (*no, "brings in a British spelling")
    if not _names(r) <= _names(s) | _names(original) | _names(" ".join(runs)):
        return (*no, "adds a name")
    if not _numbers(r) <= _numbers(s) | _numbers(original):
        return (*no, "adds a number")
    if invented_sighting(r, s + " " + original):
        return (*no, "adds a first-person sighting")
    old_run = runs[ri]
    new_run = norm(" ".join(r if norm(x) == s else x for x in _sents(old_run)))
    lost = keeps_lost(old_run, new_run, keeps[ri])
    if lost:
        return (*no, "loses a must-keep: " + ", ".join(lost))
    slip = punctuation_problem(new_run, old_run)
    if slip:
        return (*no, f"punctuation: {slip}")
    return ri, new_run, r, None


def code_proposals(runs, claimed):
    """Spelling fixes code makes itself: every sentence with a British spelling on the list that no proposal of
    Sonnet's already rewrites."""
    out = []
    for t in runs:
        for x in _sents(t):
            if gb_spellings(x) and norm(x) not in claimed:
                out.append({"sentence": x, "replacement": to_us(x), "category": "spelling",
                            "why": "British spelling: " + ", ".join(gb_spellings(x)), "by": "code"})
    return out


def parse_reply(text, n):
    """{1: [proposal, ...], ...} from Sonnet's JSON, fenced or not. Paragraphs missing from the reply get []."""
    m = re.search(r"\{.*\}", text or "", re.S)
    got = {}
    if m:
        try:
            raw = json.loads(m.group(0))
        except json.JSONDecodeError:
            raw = {}
        for k, v in (raw.items() if isinstance(raw, dict) else []):
            if str(k).strip().isdigit() and isinstance(v, list):
                got[int(k)] = [p for p in v if isinstance(p, dict)]
    return {i: got.get(i, []) for i in range(1, n + 1)}


# ---------------------------------------------------------------------- the text: what's fixable and where it sits
def run_forms(bank, wid, original):
    """{(pi, ri): {normalized text: the sentence list it was banked as}}, so a fixed window keeps the bank's own
    sentence lists (the loop compares runs as lists)."""
    out = {(pi, ri): {norm(" ".join(r)): list(r)} for pi, p in enumerate(original) for ri, r in enumerate(p)}
    shape = [len(p) for p in original]
    for e in bank.of(wid):
        runs = e.get("runs")
        if runs is None or [len(p) for p in runs] != shape:
            continue
        for pi, p in enumerate(runs):
            for ri, r in enumerate(p):
                out[(pi, ri)].setdefault(norm(" ".join(r)), list(r))
    return out


def gather(job, bank, real_text, sections, lay=None):
    """The window paragraphs the fix pass may work on. Every window paragraph with a run of Claude's or yours gets
    a row: its label, section, window and index, the run texts as the loop ships them (read from the working copy
    and split by its banked versions), the original paragraph, and `fixable` with the reason when it isn't. A
    paragraph is fixable when the real guide holds it exactly as the loop has it."""
    lay = lay or layout(job, bank)
    base_md, _ = job.placeholder_base()
    base = BLOCK_SPLIT.split(base_md)
    work = BLOCK_SPLIT.split(job.working.read_text(encoding="utf-8")) if job.working.exists() else base
    if len(work) != len(base):
        raise RuntimeError("the working copy's blocks don't line up with the original's; let the loop rewrite it")
    index = {}
    for i, blk in enumerate(base):
        k = norm(strip_markdown(COMMENT_RE.sub("", blk)))
        if k:
            index.setdefault(k, i)
    real_blocks = set()
    for n, (a, b) in section_spans(real_text, sections, style_of(job)).items():
        for k, blk in enumerate(BLOCK_SPLIT.split(real_text[a:b])):
            if k % 2 == 0 and is_prose_block(blk):
                real_blocks.add(norm(strip_markdown(blk)))
    rows = []
    for wid, w in lay.items():
        if w["section"] not in sections:
            continue
        versions = run_versions(bank, wid, w["original"])
        for pi, p in enumerate(w["original"]):
            if not any(r["claude"] or r.get("author") for r in w["runs"][pi]):
                continue
            label = f"{wid} p{pi + 1}"
            i = index.get(norm(flat(p)))
            text = norm(strip_markdown(work[i])) if i is not None else norm(flat(p))
            split = match_paragraph(text, versions[pi])
            row = {"label": label, "section": w["section"], "wid": wid, "pi": pi, "text": text,
                   "runs": split, "original": norm(flat(p)), "fixable": True, "why": ""}
            if split is None:
                row.update(fixable=False, why="the loop's text doesn't split into its runs")
            elif text not in real_blocks:
                row.update(fixable=False, why="the real guide doesn't hold this paragraph as the loop has it "
                                              "(your edit, or a write-back not done yet)")
            rows.append(row)
    return rows


def paragraph_keeps(job, w, pi):
    """Each run's must-keeps for paragraph `pi` of a layout window, from the cached notes (no paid call)."""
    cache = job._cache()
    para = " ".join(s for r in w["original"][pi] for s in r)
    notes = cache.get(sha(NOTES_VERSION + para)) or []
    out = []
    for r in w["runs"][pi]:
        line = " ".join(r["sents"])
        prot = None
        if r.get("author"):
            hit = job.protected_cached(line)
            prot = hit["words"] if hit else protected_fallback(line)
        out.append(run_keeps(line, notes, author=bool(r.get("author")), protected=prot))
    return out


# ---------------------------------------------------------------------- one pass
class FixPass:
    def __init__(self, *, job, bank, settings, conn, claude=None, pangram=None, budget=None, model=None,
                 max_words=None, noise=None, max_usd=None, purpose="loop", log=print):
        cfg = settings.config.get("fixpass", {}) if settings is not None else {}
        self.job, self.bank, self.settings, self.conn = job, bank, settings, conn
        self.claude, self.pg, self.budget = claude, pangram, budget
        self.model = model or cfg.get("model", "sonnet")
        self.max_words = int(max_words if max_words is not None else cfg.get("max_words_changed", 3))
        self.noise = float(noise if noise is not None else cfg.get("noise", 0.05))
        self.max_usd = float(max_usd if max_usd is not None else cfg.get("max_pangram_usd", 1.0))
        self.batch = int(cfg.get("batch", 6))
        self.purpose = purpose
        self.log = log
        self.spent = 0.0
        self.versions = {}             # the Pangram model version of each score, by label
        self.cache_path = (settings.runs_dir / job.name / "fixpass cache.json") if settings is not None else None

    def description(self):
        """The job's one line about the document for the prompts, or None (util.describe's default)."""
        return getattr(getattr(self, "job", None), "description", None)

    # ---- proposals
    def _cache(self):
        if self.cache_path and self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {}

    def _save(self, cache):
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.cache_path)

    def key(self, row):
        return sha(f"{FIX_VERSION}{self.model}|{self.max_words}|{row['text']}\x1f{row['original']}")

    def proposals(self, rows):
        """{label: [proposal]} from Sonnet, cached by the paragraph and its original, a batch of paragraphs a call."""
        cache = self._cache()
        todo = [r for r in rows if self.key(r) not in cache]
        for i in range(0, len(todo), self.batch):
            chunk = todo[i:i + self.batch]
            listing = "\n\n".join(f"{k + 1}. Now: {r['text']}\n   Original: {r['original']}" for k, r in enumerate(chunk))
            prompt = (describe(FIX_TEMPLATE, self.description())
                      .replace("<<N>>", str(self.max_words)).replace("<<PARAS>>", listing))
            if self.budget is not None:
                self.budget.claude_check("fix", 0.3)
            out, _ = self.claude(self.settings, self.conn, prompt, role="fixpass fix", model=self.model,
                                 max_budget_usd=0.5)
            got = parse_reply(out, len(chunk))
            for k, r in enumerate(chunk):
                cache[self.key(r)] = got[k + 1]
            self._save(cache)
        return {r["label"]: cache.get(self.key(r), []) for r in rows}

    # ---- gating
    def review(self, rows, lay, props):
        """Every proposal through the gate, paragraph by paragraph. Returns {label: {"runs": the new run texts,
        "accepted": [...], "rejected": [...]}} for every fixable row."""
        out = {}
        for row in rows:
            if not row["fixable"]:
                continue
            w = lay[row["wid"]]
            keeps = paragraph_keeps(self.job, w, row["pi"])
            runs = list(row["runs"])
            first, acc, rej = {}, [], []
            todo = [dict(p, by=p.get("by", "sonnet")) for p in props.get(row["label"], [])]
            claimed = {norm(p.get("sentence") or "") for p in todo}
            todo += code_proposals(runs, claimed)
            for p in todo:
                ri, new_run, new_sent, why = gate(p, runs, keeps, row["original"], self.max_words, first)
                if why:
                    rej.append(dict(p, reason=why))
                    continue
                first[norm(new_sent)] = first.pop(norm(p["sentence"]), norm(p["sentence"]))
                runs[ri] = new_run
                acc.append(dict(p, run=ri))
            out[row["label"]] = {"runs": runs, "accepted": acc, "rejected": rej}
        return out

    # ---- applying
    def section_text(self, raw, rows, results, n):
        """The document with section n's accepted fixes in (the block-level write-back), and its write-back
        report."""
        paras = []
        for row in rows:
            if row["section"] != n or not row["fixable"]:
                continue
            res = results.get(row["label"])
            new = " ".join(x for x in (res["runs"] if res else row["runs"]) if x)
            versions = [{v} for v in row["runs"]]
            paras.append((row["label"], versions, norm(new), {row["text"]}))
        return write_back(raw, [n], paras, style=style_of(self.job))

    def score(self, text, label):
        """(AI share, whether it was paid). Raises the client's CapReached at a cap, or RuntimeError past the
        pass's own limit."""
        hit = self.pg.cached(text)
        if hit is None:
            _, usd = self.pg.estimate(text)
            if self.spent + usd > self.max_usd + 1e-9:
                raise RuntimeError(f"the fix pass's Pangram limit (${self.max_usd:.2f}) would be crossed")
            if self.budget is not None and self.purpose == "loop":
                self.budget.pangram_check(usd, "loop")
        from .pangram import summarize
        resp, cached = self.pg.score(text, self.purpose, label)
        if not cached:
            self.spent += self.pg.estimate(text)[1]
        self.versions[label] = resp.get("version")
        return summarize(resp)["fraction_ai"], not cached

    def apply_sections(self, raw, rows, results, touched, cap_errors=()):
        """Section by section: the fixes in (in memory), the section scored as posted before and after, and the new
        text kept only when its AI share rose by no more than `noise`. Nothing is written here. Returns (the text
        with the kept sections' fixes in, {section: before, after, result, replaced}, [kept sections])."""
        out, kept = {}, []
        for n in touched:
            new_raw, rep = self.section_text(raw, rows, results, n)
            st = style_of(self.job)
            before, after = as_posted(raw, [n], style=st)[n], as_posted(new_raw, [n], style=st)[n]
            s = out.setdefault(n, {"replaced": rep["replaced"], "before": None, "after": None})
            try:
                s["before"], _ = self.score(before, f"fixpass s{n} before")
                s["after"], _ = self.score(after, f"fixpass s{n} after")
            except (RuntimeError, *cap_errors) as e:
                s["result"] = f"not applied: Pangram said no ({str(e)[:90]})"
                continue
            vb, va = self.versions.get(f"fixpass s{n} before"), self.versions.get(f"fixpass s{n} after")
            if vb and va and vb != va:
                # a cached score from an older Pangram model against a fresh one says nothing about the fixes
                s["result"] = f"not applied: the before score is Pangram {vb}, the after {va}"
                continue
            if s["after"] - s["before"] > self.noise + 1e-9:
                s["result"] = f"not applied: the AI share rose by {s['after'] - s['before']:.2f}"
                continue
            s["result"] = f"applied ({len(rep['replaced'])} paragraph(s))"
            raw = new_raw
            kept.append(n)
        return raw, out, kept

    def cached_score(self, text):
        """The AI share Pangram's cache holds for a text, or None. Free."""
        from .pangram import summarize
        hit = self.pg.cached(text)
        return None if hit is None else summarize(hit)["fraction_ai"]

    def window_runs(self, lay, rows, results, wid):
        """The window's full runs (every paragraph, in the bank's shape and sentence lists) with the fixes in."""
        w = lay[wid]
        forms = run_forms(self.bank, wid, w["original"])
        by_pi = {r["pi"]: r for r in rows if r["wid"] == wid}
        out = []
        for pi, p in enumerate(w["original"]):
            row = by_pi.get(pi)
            texts = (results.get(row["label"], {}).get("runs") if row and row["label"] in results else None) \
                or (row["runs"] if row and row["runs"] else None)
            if texts is None:
                out.append([list(r) for r in p])
                continue
            out.append([forms[(pi, ri)].get(norm(t)) or (_sents(t) if t else []) for ri, t in enumerate(texts)])
        return out


def report_md(when, dry, rows, results, sections_out, purpose, model, max_words, noise):
    L = [f"# Fix pass, {when}{' (dry run)' if dry else ''}", "",
         f"Sonnet model `{model}`, at most {max_words} tokens changed a sentence, a section keeps its fixes when its"
         f" AI share as posted rises by no more than {noise:.2f}. Pangram purpose `{purpose}`.", ""]
    if sections_out:
        L += ["## Sections", "", "| Section | Before | After | Result |", "| --- | --- | --- | --- |"]
        for n, s in sorted(sections_out.items()):
            fmt = lambda v: "not scored" if v is None else f"{v:.2f}"  # noqa: E731
            L.append(f"| {n} | {fmt(s.get('before'))} | {fmt(s.get('after'))} | {s.get('result', '')} |")
        L.append("")
    acc = [(row, a) for row in rows for a in results.get(row["label"], {}).get("accepted", [])]
    rej = [(row, a) for row in rows for a in results.get(row["label"], {}).get("rejected", [])]
    L += [f"## Accepted ({len(acc)})", ""]
    for row, a in acc:
        L += [f"- **{row['label']}** ({a['category']}, {a.get('by', 'sonnet')}): {a.get('why', '')}",
              f"  - was: {a['sentence']}", f"  - now: {a['replacement']}"]
    L += ["", f"## Rejected ({len(rej)})", ""]
    for row, a in rej:
        L += [f"- **{row['label']}** ({a.get('category', '?')}, {a.get('by', 'sonnet')}): {a['reason']}",
              f"  - was: {a.get('sentence', '')}", f"  - proposed: {a.get('replacement', '')}",
              f"  - why it was proposed: {a.get('why', '')}"]
    skipped = [r for r in rows if not r["fixable"] and r.get("asked", True)]
    if skipped:
        L += ["", f"## Not looked at ({len(skipped)})", ""]
        L += [f"- {r['label']}: {r['why']}" for r in skipped]
    return "\n".join(L) + "\n"


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="the fix pass: few-word fixes on the loop's text (plan 6.5)")
    ap.add_argument("--job", help="the job, a folder in jobs/ (default: [job] default in config.toml)")
    ap.add_argument("--real", help="the document (default: job.toml's document)")
    ap.add_argument("--section", help="only these sections, comma-separated (default: the job's scope)")
    ap.add_argument("--para", help="only these paragraphs or windows, comma-separated ('s1w1 p2,s3w0')")
    ap.add_argument("--write", action="store_true", help="apply and re-score; without this it's a dry run")
    ap.add_argument("--model", help="Sonnet model for the proposals (default: config's fixpass.model)")
    ap.add_argument("--purpose", default="loop", choices=["loop", "experiment"],
                    help="the Pangram purpose for the re-score (experiment when the day's loop cap is spent)")
    ap.add_argument("--max-usd", type=float, help="the most this pass may spend on Pangram")
    a = ap.parse_args(argv)

    from . import db
    from .budget import Budget, CapReached
    from .claude_call import ClaudeCallError, run as claude_run
    from .pangram import CapReached as ClientCap, Pangram
    from .schedule import Clock, Schedule
    settings = Settings()
    conn = db.connect(settings.db_path)
    job = Job(a.job)
    real = Path(a.real) if a.real else job.real_file
    sections = [int(x) for x in a.section.split(",")] if a.section else list(job.sections)
    only = {x.strip() for x in a.para.split(",")} if a.para else None
    bank = Bank(job.bank_path)
    clock = Clock()
    budget = Budget(settings, conn, clock, Schedule(settings.config))
    fp = FixPass(job=job, bank=bank, settings=settings, conn=conn, claude=claude_run, pangram=Pangram(settings, conn),
                 budget=budget, model=a.model, max_usd=a.max_usd, purpose=a.purpose)
    raw0 = real.read_bytes().decode("utf-8")
    lay = layout(job, bank)
    rows = gather(job, bank, raw0, sections, lay=lay)
    if only:                                   # every row stays (a fixed window is built from all of them)
        for r in rows:
            r["asked"] = r["label"] in only or r["wid"] in only
    todo = [r for r in rows if r["fixable"] and r.get("asked", True)]
    print(f"{len(todo)} paragraph(s) to read, {sum(1 for r in rows if r.get('asked', True)) - len(todo)} left alone")
    try:
        props = fp.proposals(todo)
    except (CapReached, ClaudeCallError) as e:
        print(f"no proposals: {e}")
        return 1
    results = fp.review(todo, lay, props)
    touched = sorted({r["section"] for r in rows if results.get(r["label"], {}).get("accepted")})
    for r in rows:
        res = results.get(r["label"])
        for x in (res or {}).get("accepted", []):
            print(f"  {r['label']} [{x['category']}] {x['sentence']}\n      -> {x['replacement']}")
        for x in (res or {}).get("rejected", []):
            print(f"  {r['label']} rejected ({x['reason']}): {x.get('replacement', '')[:110]}")
    sections_out = {}
    raw = raw0
    if a.write:
        raw, sections_out, kept = fp.apply_sections(raw0, rows, results, touched, (ClientCap, CapReached))
        if kept:
            if real.read_bytes().decode("utf-8") != raw0:
                print("the real guide changed while the fix pass ran; nothing written")
                return 1
            real.write_bytes(raw.encode("utf-8"))
            stamp = clock.stamp()
            best = {}
            for wid in sorted({r["wid"] for r in rows if r["section"] in kept}):
                fixes = [dict(x, para=r["pi"]) for r in rows if r["wid"] == wid
                         for x in results.get(r["label"], {}).get("accepted", [])]
                if not fixes:
                    continue
                runs = fp.window_runs(lay, rows, results, wid)
                w = lay[wid]
                bank.add({"kind": "fix", "window": wid, "time": stamp, "text": window_text_of(runs, w["heading"]),
                          "runs": runs, "ai": None, "section": w["section"],
                          "section_ai": sections_out[w["section"]]["after"],
                          "fixes": [{k: x.get(k) for k in ("para", "run", "category", "by", "sentence", "replacement", "why")}
                                    for x in fixes]})
                best[wid] = ([flat(p) for p in w["original"]], [flat(p) for p in runs])
            job.assemble(best, keep_others=True)
    else:
        for n in touched:
            _, rep = fp.section_text(raw, rows, results, n)
            sections_out[n] = {"result": f"dry run: {len(rep['replaced'])} paragraph(s) would change",
                               "before": fp.cached_score(as_posted(raw, [n], style=style_of(job))[n])}
    when = time.strftime("%Y-%m-%d %H%M")
    out = report_file(f"fixpass {when}{' (dry run)' if not a.write else ''}.md")
    out.write_text(report_md(when, not a.write, rows, results, sections_out, a.purpose, fp.model, fp.max_words,
                             fp.noise), encoding="utf-8")
    n_acc = sum(len(v["accepted"]) for v in results.values())
    n_rej = sum(len(v["rejected"]) for v in results.values())
    line = (f"{clock.stamp()} fix pass{' (dry run)' if not a.write else ''}: {len(todo)} paragraph(s) read,"
            f" {n_acc} fix(es) accepted, {n_rej} rejected; "
            + ("; ".join(f"section {n} {s.get('result', '')}" for n, s in sorted(sections_out.items())) or "no section touched")
            + f"; Pangram ${fp.spent:.2f}; report {out.name}")
    logp = settings.runs_dir / job.name / "fixpass.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    with logp.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
