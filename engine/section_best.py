"""A section's best as-posted text, remembered (plan 5.6; 2026-11-10 evening).

The whole-text round scores each section as posted and banks the scorings as `whole` entries, and the picks it ends
on as a `whole_end` entry. Nothing used to remember which picks scored a section's best, so on 2026-11-10 section 3
shipped at 0.29 with a 0.00 version in the bank: a settle round overnight re-rolled two of s3w0's runs, the window
scored 0.00 for the new text, the tie counted as a new window best, the window's best overrode the section's
whole-text pick, and the next round found the section at 0.29 as it stood and never looked back.

This module reads the memory out of the bank. One record per finished whole-text round pairs the standing score of
every section (the last `whole` scoring of it before the `whole_end`, which is the score of the picks the round ended
on, since a scoring the round turned down banks the standing score again) with the picks the round ended on. From
2026-11-10 (evening) the `whole_end` entry carries those scores and the run lists itself (`sections`, `runs`); for
the two older rounds the pairing above and a text lookup do the same job.

    whole_rounds(bank)            the records, oldest first
    section_best(bank, n, noise)  the record whose score for section n is the lowest, or the latest within `noise`
                                  of it, with that section's score (as `ai`, `flagged`, `segs`) on top
    section_standing(bank, n)     the latest record that scored section n
    runs_for(bank, rnd, wid, ...) a window's run lists in a record (from the record, else the bank entry whose text
                                  the record's pick is, else the original when the pick is the original text)

Nothing here calls Pangram or a model; it only reads the bank.
"""


def _rounds(bank):
    out, scores = [], {}
    for i, e in enumerate(bank.entries):
        kind = e.get("kind")
        if kind == "whole":
            for n, s in (e.get("sections") or {}).items():
                scores[str(n)] = {"ai": s.get("ai"), "flagged": s.get("flagged"), "segs": s.get("segs") or [],
                                  "time": e.get("time")}
        elif kind == "whole_end":
            secs = e.get("sections")
            if secs is not None:
                secs = {str(n): {"ai": s.get("ai"), "flagged": s.get("flagged"), "segs": s.get("segs") or [],
                                 "time": e.get("time")} for n, s in secs.items()}
            else:
                secs = scores
            out.append({"index": i, "day": e.get("day"), "time": e.get("time"), "sections": secs,
                        "picks": dict(e.get("picks") or {}), "runs": dict(e.get("runs") or {})})
            scores = {}
    return out


def whole_rounds(bank):
    """One record per finished whole-text round, oldest first, cached on the bank until it grows."""
    cache = getattr(bank, "_section_rounds", None)
    if cache is None or cache[0] != len(bank.entries):
        cache = (len(bank.entries), _rounds(bank))
        bank._section_rounds = cache
    return cache[1]


def _with_score(rnd, n):
    s = rnd["sections"].get(str(n))
    if s is None or s.get("ai") is None:
        return None
    return dict(rnd, ai=s["ai"], flagged=s.get("flagged"), segs=s.get("segs") or [], section=str(n))


def section_standing(bank, n):
    """The latest finished whole-text round that scored section n, with its score on top; None before any."""
    for rnd in reversed(whole_rounds(bank)):
        got = _with_score(rnd, n)
        if got:
            return got
    return None


def section_best(bank, n, noise=0.0):
    """The round that holds section n's lowest as-posted score, or the latest round within `noise` of it (the more
    current text, when the scores can't be told apart); None before any scoring."""
    scored = [r for r in (_with_score(rnd, n) for rnd in whole_rounds(bank)) if r]
    if not scored:
        return None
    low = min(r["ai"] for r in scored)
    return [r for r in scored if r["ai"] <= low + noise][-1]


def runs_for(bank, rnd, wid, original=None):
    """The run lists behind a record's pick for a window: the record's own (`runs`), else the newest bank entry of
    the window whose text is that pick, else `original` (its run lists) when the pick is the original's text
    (`original` is (text, runs)). None when nothing in the bank holds it."""
    if rnd["runs"].get(wid) is not None:
        return rnd["runs"][wid]
    text = rnd["picks"].get(wid)
    if text is None:
        return None
    for e in reversed(bank.entries):
        if e.get("window") == wid and e.get("runs") is not None and e.get("text") == text:
            return e["runs"]
    if original and original[0] == text:
        return original[1]
    return None
