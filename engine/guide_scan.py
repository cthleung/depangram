"""The guide as it stands, and the sections combined (plan 9; the author's ask of 2026-11-10, 16:30).

Two numbers for every digest:

    sections combined   the in-scope sections' as-posted scores from the day's final scan, combined into one
                        number weighted by each section's as-posted word count. Free: no new call.
    the guide as it     the whole guide file scored once as one document, the way a reader would paste it today:
    stands              every section, the in-scope ones with the shipped picks and the rest exactly as they stand
                        in the real file (stubs and drafts included), comments and marks out, examples in. One
                        Pangram call at the wrap-up, after the final as-posted scan, purpose `loop`, label
                        "guide as it stands" ("run guide as it stands" in the ledger), drawn from the day cap like
                        the final scan. The cache makes an unchanged guide free. [loop] guide_scan switches it.

Neither number is a target for anything: no round, no context round and no pick reads them. The bank gets one
`guide` entry per scoring (day, time, ai, flagged count, words, the sha of the text, the combined number), so the
loop history reads them from the bank rather than the digest; a scoring the day cap refused is banked with ai null
and the reason, as a round Pangram couldn't score is.

    python -m engine.guide_scan --dry     # assemble the whole from the working copy and the real file; print the
                                          # word count, the estimated cost and the day's spend; score nothing
    python -m engine.guide_scan           # score it (purpose loop, under the day cap) and bank the entry
"""
import argparse
import sys
import time

from .job import Bank, Job, style_of
from .pangram import summarize, text_sha
from .wrapup import guide_as_posted, layout, paragraphs_from_working, write_back

LABEL = "guide as it stands"


def combined(scores, words):
    """{ai, sections, left_out, words}: the sections' scores combined, weighted by their word counts. A section with
    no score (the cap said no) or no words is left out and named; None for `ai` when nothing could be combined."""
    used = sorted(n for n in scores if scores.get(n) is not None and words.get(n))
    left = sorted(n for n in scores if n not in used)
    total = sum(words[n] for n in used)
    ai = sum(scores[n] * words[n] for n in used) / total if total else None
    return {"ai": ai, "sections": used, "left_out": left, "words": total}


def section_list(secs):
    """"1, 2, 3 and 6" for the digest lines."""
    secs = [str(n) for n in secs]
    if len(secs) <= 1:
        return "".join(secs)
    return ", ".join(secs[:-1]) + " and " + secs[-1]


def assemble(job, bank, real=None):
    """The whole guide as a reader would paste it today, by hand: the loop's paragraphs from the working copy into
    the real guide (wrapup's dry run, nothing written), then the whole file as posted. Returns (text, the
    write-back report)."""
    real = real or job.real_file
    lay = layout(job, bank)
    paras = paragraphs_from_working(job, lay, bank)
    raw = real.read_bytes().decode("utf-8")
    new, report = write_back(raw, job.sections, paras, style=style_of(job))
    return guide_as_posted(new), report


def entry(day, when, text, ai=None, flagged=None, sections_combined=None, how="wrap-up", skipped=None,
          placeholder=False):
    """The bank's `guide` entry: no text and no segments (the whole guide's would swell the bank), the sha instead."""
    e = {"kind": "guide", "day": day, "time": when, "label": LABEL, "ai": ai, "flagged": flagged,
         "words": len(text.split()), "sha": text_sha(text), "sections_combined": sections_combined, "how": how}
    if skipped:
        e["skipped"] = skipped
    if placeholder:
        e["placeholder"] = True
    return e


def flagged_count(resp):
    return sum(1 for w in resp.get("windows") or [] if (w.get("ai_assistance_score") or 0) > 0.5)


def run_for_loop(loop, raw):
    """The wrap-up's call (engine/run.py's wrap_up, after the final as-posted scan): `raw` is the real guide with the
    shipped picks in (the write-back's result, written or a dry run). Sets loop.guide_scan for the digest, the card
    and the reflection, and banks the entry. Off unless [loop] guide_scan is on; a placeholder at zero spend in a
    rehearsal (the text is assembled for real, the call never made); skipped and banked as such when the day cap
    says no."""
    from .budget import PangramCapReached
    text = guide_as_posted(raw)
    words = len(text.split())
    comb = (getattr(loop, "sections_combined", None) or {}).get("ai")
    day, when = getattr(loop, "day", None), loop.clock.stamp()
    if not getattr(loop, "guide_scan_on", False):
        loop.guide_scan = {"skipped": "guide_scan is off in config.toml", "words": words, "ai": None}
        return loop.guide_scan
    if getattr(loop, "rehearsal", False):
        loop.log(f"{LABEL}: rehearsal, {words} words assembled; the scoring is a placeholder at zero spend")
        loop.guide_scan = {"placeholder": True, "words": words, "ai": None, "flagged": None, "sha": text_sha(text)}
        loop.bank.add(entry(day, when, text, sections_combined=comb, placeholder=True))
        return loop.guide_scan
    try:
        resp = loop.score(text, LABEL, whole=True)
    except PangramCapReached as e:
        loop.on_cap(e)
        loop.log(f"{LABEL}: not scored today (the day cap: {e.detail})")
        loop.guide_scan = {"skipped": "the day cap", "words": words, "ai": None, "sha": text_sha(text)}
        loop.bank.add(entry(day, when, text, sections_combined=comb, skipped="the day cap"))
        return loop.guide_scan
    s = summarize(resp)
    loop.guide_scan = {"ai": s["fraction_ai"], "flagged": flagged_count(resp), "words": words,
                       "humanized": s["humanized"], "sha": text_sha(text)}
    loop.bank.add(entry(day, when, text, ai=s["fraction_ai"], flagged=flagged_count(resp), sections_combined=comb))
    loop.log(f"{LABEL}: {s['fraction_ai']:.2f} ({flagged_count(resp)} flagged, {words} words)"
             + (f"; sections combined {comb:.2f}" if comb is not None else ""))
    return loop.guide_scan


# ---------------------------------------------------------------------- the lines the posts carry
def digest_lines(loop):
    """The two lines under the digest's as-posted table."""
    comb = getattr(loop, "sections_combined", None)
    gs = getattr(loop, "guide_scan", None)
    L = []
    if comb:
        if comb.get("ai") is not None:
            line = f"Sections {section_list(comb['sections'])} combined: {comb['ai']:.2f} (weighted by words)"
            if comb.get("left_out"):
                line += f"; section{'s' if len(comb['left_out']) > 1 else ''} {section_list(comb['left_out'])} not" \
                        " scanned today, so left out"
            L.append(line + ".")
        else:
            L.append("Sections combined: nothing to combine today (no section was scanned).")
    if gs:
        L.append("The guide as it stands: " + guide_phrase(gs) + " Every section, scored whole the way a reader"
                 " would paste the file today; a number for you, never a target.")
    return L


def guide_phrase(gs):
    if gs.get("ai") is not None:
        return (f"{gs['ai']:.2f} ({gs.get('flagged', 0)} flagged, {gs.get('words', 0):,} words"
                + (", humanized" if gs.get("humanized") else "") + ").")
    if gs.get("placeholder"):
        return f"a placeholder in a rehearsal ({gs.get('words', 0):,} words assembled, no call made)."
    return f"not scored today: {gs.get('skipped', 'no scoring')}."


def card_line(loop):
    """One line for the Discord card."""
    comb = getattr(loop, "sections_combined", None) or {}
    gs = getattr(loop, "guide_scan", None)
    if not gs and comb.get("ai") is None:
        return ""
    if not gs:
        guide = "not scored"
    elif gs.get("ai") is not None:
        guide = f"{gs['ai']:.2f}"
    elif gs.get("placeholder"):
        guide = "placeholder (rehearsal)"
    else:
        guide = f"not scored ({gs.get('skipped', 'no scoring')})"
    sec = f"{comb['ai']:.2f}" if comb.get("ai") is not None else "not combined"
    return f"Guide as it stands: {guide}, sections combined {sec}"


# ---------------------------------------------------------------------- by hand
def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="score the whole guide as it stands, once, by hand (plan 9)")
    ap.add_argument("--job", help="the job, a folder in jobs/ (default: [job] default in config.toml)")
    ap.add_argument("--dry", action="store_true", help="assemble and estimate only; score nothing, bank nothing")
    ap.add_argument("--purpose", default="loop", choices=["loop", "experiment"],
                    help="the Pangram purpose (experiment when the day's loop cap is spent)")
    a = ap.parse_args(argv)
    from . import db
    from .budget import Budget, CapReached
    from .pangram import CapReached as ClientCap, Pangram, PangramError
    from .schedule import Clock, Schedule
    from .settings import Settings
    settings = Settings()
    job = Job(a.job)
    bank = Bank(job.bank_path)
    text, report = assemble(job, bank)
    words = len(text.split())
    conn = db.connect(settings.db_path)
    pg = Pangram(settings, conn)
    credits, usd = pg.estimate(text)
    cached = pg.cached(text) is not None
    clock = Clock()
    budget = Budget(settings, conn, clock, Schedule(settings.config))
    cap = float(settings.config["pangram"]["nightly_cap_usd"])
    today = budget.pangram_today()
    print(f"{LABEL}: {words:,} words, {credits} credits, about ${usd:.2f}" + (" (cached: free)" if cached else ""))
    print(f"the loop's paragraphs into the real guide: {len(report['replaced'])} replaced, {len(report['same'])} as"
          f" they are, {len(report['missing'])} not found; sha {text_sha(text)[:12]}")
    print(f"Pangram today (purpose loop): ${today:.2f} of ${cap:.2f}, ${cap - today:.2f} left")
    if a.dry:
        print("dry run: nothing scored, nothing banked")
        return 0
    when = time.strftime("%Y-%m-%d %H:%M")
    day = time.strftime("%Y-%m-%d")
    try:
        if not cached and a.purpose == "loop":
            budget.pangram_check(usd, "loop")
        resp, from_cache = pg.score(text, a.purpose, f"run {LABEL}")
    except (CapReached, ClientCap) as e:
        print(f"not scored: a cap said no ({e})")
        bank.add(entry(day, when, text, how="by hand", skipped=f"a cap ({str(e)[:80]})"))
        return 1
    except PangramError as e:
        print(f"not scored: Pangram failed ({str(e)[:160]})")
        return 1
    s = summarize(resp)
    e = bank.add(entry(day, when, text, ai=s["fraction_ai"], flagged=flagged_count(resp), how="by hand"))
    print(f"{LABEL}: {s['fraction_ai']:.2f} AI share, {e['flagged']} of {len(resp.get('windows') or [])} segments"
          f" flagged, {words:,} words{' (from the cache)' if from_cache else ''}; banked in {bank.path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
