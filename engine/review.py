"""The reviewer's file for the paragraph pass, `jobs/<job>/review.json` (2026-11-14), written by hand:

    {"vetoes":  [{"label", "orig", "text", "why", "scoring"}],
     "repairs": [{"label", "orig", "from", "text", "moves", "by", "when", "scoring"}],
     "for_you": [{"label", "orig", "text", "went_wrong", "about", "point", "fact", "when", "scoring"}],
     "stuck":   [{"label", "orig", "why", "about", "point", "fact", "when"}]}

`scoring` (optional, 2026-11-14) is the label of the scoring the pass was serving when the entry was written, such as
"s6 round 8". The newest usable scoring a section's entries name is its review base, the scoring its repairs are read
against (RegrowPass.review_base), so a veto written beside a repair never makes the repair stale. Entries without it
keep the older rule.

A veto is a candidate span the pass must never pick. A repair is a pick mended with small edits (an invented sentence
cut, up to three words swapped, one original sentence spliced back): wherever the unit would carry the pick `from`, it
carries `text`, which ships as normal text. A hand fix ("for you") is a span the reviewer corrected by hand, for what
small edits couldn't save; every assembly the pass builds carries it from then on, and the write-back shows it with
its note. The author's order (2026-11-14): repair first, veto and redraw what a repair can't save, hand fix what is
still broken after that. Entries match a unit by its original span text, whitespace
squashed: the label is only for the human, since labels shift when the guide is re-synced. A missing file means no
vetoes and no hand fixes; a file that doesn't parse stops the run, so a typo never lets a vetoed text back in.

A stuck entry (2026-11-15, the author's rule: what the machine can't write goes in the guide in bold with a prompt
for the author as the human writer) is a unit the pass has found no pick for, whose standing text still holds
sentences of Claude's original word for word. It carries no `text` and no `scoring`, pins nothing and changes nothing
in what the pass draws, picks, scores or calls a best: the pass keeps trying the unit every run. While the unit has no
pick in what the pass serves and no hand fix, the write-back puts those sentences in bold under a FOR YOU TO REWRITE
marker with the entry's prompt (why it's the author's, what to write about, the point, the fact). When two entries
match one unit, the later one wins.
"""
import json
from pathlib import Path

FILE = "review.json"
FOR_YOU_KEYS = ("went_wrong", "about", "point", "fact")
STUCK_KEYS = ("why", "about", "point", "fact")
KINDS = ("vetoes", "repairs", "for_you", "stuck")


def squash(t):
    return " ".join((t or "").split())


def load(job_dir):
    """The review file of a job folder, as {"vetoes": [...], "repairs": [...], "for_you": [...], "stuck": [...]}."""
    path = Path(job_dir) / FILE
    if not path.exists():
        return {k: [] for k in KINDS}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ValueError(f"{path} doesn't parse: {e}") from e
    return {k: list(data.get(k) or []) for k in KINDS}


def forms(text):
    """A span as the matchers compare it: whitespace squashed, and also with quote_stops applied (a banked draw from
    before that fix still reads `"does the job.".`)."""
    from .regrow import quote_stops
    s = squash(text)
    return {s, squash(quote_stops(s))}


def veto_why(review, orig, text):
    """The reason the reviewer gave when `text` is a vetoed candidate for the unit whose original span is `orig`,
    else None. An empty reason still vetoes."""
    o = squash(orig)
    have = None
    for v in review.get("vetoes", ()):
        if squash(v.get("orig")) != o:
            continue
        have = have or forms(text)
        if forms(v.get("text")) & have:
            return v.get("why") or "no reason given"
    return None


def repair_fits(r, pick_text):
    """Whether repair `r` speaks for a unit carrying `pick_text`: the pick is its `from`, or already its `text`."""
    have = forms(pick_text)
    return bool(have & (forms(r.get("from")) | forms(r.get("text"))))


def repair(review, orig, pick_text):
    """The repair for the unit whose original span is `orig` when it carries `pick_text` (the repaired pick, or the
    repaired text itself), else None. A repair whose `from` is some other text never applies to this pick."""
    o = squash(orig)
    return next((r for r in review.get("repairs", ()) if squash(r.get("orig")) == o and r.get("text")
                 and repair_fits(r, pick_text)), None)


def hand_fix(review, orig):
    """The hand fix for the unit whose original span is `orig`, or None. When the file holds two for one span, the
    later one wins (the reviewer added a newer fix below the old one)."""
    o = squash(orig)
    return next((f for f in reversed(review.get("for_you", ())) if squash(f.get("orig")) == o and f.get("text")), None)


def note(fix):
    """The part of a hand fix the write-back shows: {"went_wrong", "about", "point", "fact"}."""
    return {k: fix.get(k) for k in FOR_YOU_KEYS}


def stuck(review, orig):
    """The stuck entry for the unit whose original span is `orig`, or None; the later one wins when two match. A
    review built without the list (an older file, a test) has none."""
    o = squash(orig)
    return next((s for s in reversed(review.get("stuck") or ()) if squash(s.get("orig")) == o), None)


def sentences(text):
    """A span's sentences, whitespace squashed, by the pass's own splitter."""
    from .checks import split_sentences
    return [squash(x) for x in split_sentences(squash(text)) if squash(x)]


def is_part(unit_orig, entry_orig):
    """Whether the unit's original span is a run of whole sentences of the entry's, and shorter: what is left of a
    stuck span once the author rewrote a sentence of it by hand and the sync made that sentence an anchor."""
    u, e = sentences(unit_orig), sentences(entry_orig)
    return 0 < len(u) < len(e) and any(e[i:i + len(u)] == u for i in range(len(e) - len(u) + 1))


def holds(unit_orig, entry_orig):
    """Whether the unit's original span holds the entry's whole as a run of whole sentences, and more: after the
    author rewrote a stuck paragraph by hand, the sync freezes it as the guide had it, so a line the loop wrote
    next to Claude's stuck sentences ("Yeah, that one.") joins their span, and the span grows past the entry's."""
    return is_part(entry_orig, unit_orig)


MIN_SHARED_WORDS = 4   # an overlap of shorter sentences than this ("Slow.") could be two spans saying the same word


def shared(unit_orig, entry_orig):
    """The sentences two original spans have in common, in the unit's order."""
    e = set(sentences(entry_orig))
    return [x for x in sentences(unit_orig) if x in e]


def overlaps(unit_orig, entry_orig):
    """Whether a stuck entry may speak for a unit it doesn't match exactly: the unit's span is a part of the
    entry's (is_part: you rewrote some of its sentences), or the two share sentences of MIN_SHARED_WORDS words or
    more between them (the span grew past the entry's at the sync, holds, or the loop had already rewritten one of
    the entry's sentences there, so the guide's text froze in its place)."""
    common = shared(unit_orig, entry_orig)
    return bool(common) and (is_part(unit_orig, entry_orig) or sum(len(x.split()) for x in common) >= MIN_SHARED_WORDS)


def stuck_match(review, orig):
    """(the stuck entry that speaks for the unit whose original span is `orig`, True when it speaks for it by
    overlap rather than exactly), or (None, False). An exact match always wins; then the overlapping entry
    (overlaps) that shares the most sentences with the unit, the later one among equals. Either way only the
    sentences both originals share can go in bold (engine/regrow.py's stuck_bold)."""
    exact = stuck(review, orig)
    if exact is not None:
        return exact, False
    best, most = None, 0
    for e in review.get("stuck") or ():
        if overlaps(orig, e.get("orig")):
            n = len(shared(orig, e.get("orig")))
            if n >= most:
                best, most = e, n
    return best, best is not None


def stuck_note(entry):
    """The part of a stuck entry the write-back shows: {"why", "about", "point", "fact"}."""
    return {k: entry.get(k) for k in STUCK_KEYS}
