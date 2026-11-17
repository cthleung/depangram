"""The run loop's reports: the hourly post, the digest at stop (a short card plus the full report) and the
couldn't-crack report it also writes to the job folder. They read the loop's state and return text; they never
send anything, hold a key or a URL. engine/discord.py sends them through whichever notifier is set, and keeps
these names as aliases for older callers."""
import datetime as dt
from pathlib import Path

LIMIT = 2000                    # the card's length: a Discord message's limit, kept for every notifier


def _ai(x):
    return "unscored" if x is None else f"{x:.2f}"


def _hm(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M") if ts else "?"


def spend_lines(b):
    rd = b.get("pangram_run_day")
    out = [f"Pangram ${b['pangram_today']:.2f} of ${b['pangram_day_cap']:.2f} today"
           + (f" (${rd:.2f} this run-day)" if rd is not None else "")
           + f", ${b['pangram_month']:.2f} of ${b['pangram_month_cap']:.2f} this month",
           f"Claude about ${b['claude_usd_today']:.2f} estimated today"
           + (f" (first-run backstop ${b['backstop']:.2f})" if b["backstop"] is not None else "")
           + f", {b['loop_points_today']:.1%} of your week (today's allowance {b['allowance']:.1%})",
           f"Loop share this week {b['loop_share']:.1%} of {b['share_cap']:.0%}; your week {b['real_total']:.0%}"
           f" (ceiling {b['ceiling']:.0%}); 5-hour window {b['five_hour']:.0%} (cap now {b['five_hour_cap']:.0%})"]
    return out


def hourly_text(loop):
    now = loop.clock.now()
    worked = ", ".join(f"{w} ({n} round{'s' if n != 1 else ''})" for w, n in loop.worked_this_hour.items()) or "none"
    best = " | ".join(f"{w.wid} {_ai(w.ai)}" for w in sorted(loop.windows, key=lambda w: w.wid))
    blocks = [f"{x.check} until {_hm(x.until)}" for x in (loop.claude_block, loop.pangram_block) if x]
    lines = [f"**depangram, {loop.clock.stamp()}** ({loop.schedule.phase(now)} phase)",
             f"Windows worked this hour: {worked}", f"Scores now: {best}", *spend_lines(loop.budget.snapshot())]
    if blocks:
        lines.append("Holding: " + "; ".join(blocks))
    return "\n".join(lines)[:LIMIT]


def window_rows(loop):
    rows = []
    for w in loop.windows:
        best = (loop.best(w) if getattr(w, "must", None) is not None else None) or loop.bank.start(w.wid)
        cur, runs = getattr(w, "current", None), getattr(w, "runs", None) or []
        yours = [(pi, ri) for pi, p in enumerate(runs) for ri, r in enumerate(p) if r.get("author")]
        rows.append({"wid": w.wid, "start": w.start_of_day, "now": w.ai, "best": best["ai"] if best else None,
                     "state": ("retired" if w.retired else w.state) + (" (pass open)" if w.pass_open else "")
                     + (" (rested on your word)" if getattr(w, "rest_call", False) else ""),
                     "passes": w.passes_today, "rounds": w.rounds_today,
                     "flagged": w.n_flagged() if w.segs else None,
                     "yours": (f"{sum(1 for pi, ri in yours if cur[pi][ri] != w.original[pi][ri])} of {len(yours)}"
                               if cur and yours else "")})
    return rows


def digest_card(loop, crashed=None):
    rows = window_rows(loop)
    head = (f"**depangram digest, {getattr(loop, 'day', '')}**"
            f"{' (CRASHED: ' + crashed[:120] + ')' if crashed else ''}\nStopped: {loop.stop_reason or 'crash'}.")
    table = "\n".join(f"{r['wid']}: {_ai(r['start'])} -> {_ai(r['now'])} ({r['state']})" for r in rows)
    caps = ", ".join(sorted({c["check"] for c in loop.cap_events})) or "none"
    back = sum(1 for c in loop.cuts if c.get("vetoed"))
    tail = (f"Caps that said no: {caps}. Cuts: {len(loop.cuts)}{f' ({back} put back on your word)' if back else ''}."
            f"{open_questions_line(loop)} Full report attached.")
    from engine.guide_scan import card_line
    guide = card_line(loop)                 # plan 9: "Guide as it stands: 0.31, sections combined 0.19"
    card = "\n".join([head, table, *([guide] if guide else []), *spend_lines(loop.budget.snapshot()), tail])
    # the reflection's first paragraph and its first "push next" line, in whatever room the 2,000 characters leave
    # (plan 9; the whole text is in the attached report)
    r = getattr(loop, "reflection", None) or {}
    room = LIMIT - len(card) - 1
    if r.get("ok") and r.get("text") and room > 120:
        from engine.reflect import card_snippet
        snippet = card_snippet(r["text"], room)
        if snippet:
            card += "\n" + snippet
    return card[:LIMIT]


def open_questions_line(loop):
    """The card's word on the loop's unanswered Discord questions (plan 9.1); empty with no bot token."""
    asked = getattr(loop, "asked", None)
    n = len(asked.open()) if asked is not None else 0
    return (f" Open questions on Discord: {n} (no answer means the default; a later answer still counts)."
            if n else "")


def reflection_lines(loop):
    """The reflection (engine/reflect.py; plan 9): Sonnet's read of the day for the author, in full. It's the one
    Claude role that sees the scores, and the digest and analytics/reflections/ are the only places its text goes."""
    r = getattr(loop, "reflection", None)
    if r is None:
        return []
    L = ["", "## Reflection", ""]
    if r.get("skipped"):
        L.append(f"Skipped today: {r['skipped']}.")
    elif r.get("ok"):
        L.append(r["text"].strip())
        if not r.get("placeholder"):
            cut = f", {len(r['dropped'])} sentence(s) cut by the house check" if r.get("dropped") else ""
            L += ["", f"Model {r.get('model', '?')}, about ${r.get('cost', 0.0):.2f} of Claude, {r.get('attempts', 1)}"
                  f" draft(s){cut}; the file is `{Path(r['path']).name if r.get('path') else 'analytics/reflections/'}`."]
    else:
        day = getattr(loop, "day", "")
        L.append(f"The reflection didn't pass the house check today; its draft is in analytics/reflections/{day}"
                 " (draft).md.")
    return L


def digest_report(loop, crashed=None):
    b = loop.budget.snapshot()
    rows = window_rows(loop)
    L = [f"# depangram digest, {getattr(loop, 'day', '')}{' (rehearsal)' if loop.rehearsal else ''}", "",
         f"Job `{loop.job.name}`, model `{loop.model}`. Stopped at {loop.clock.stamp()}: {loop.stop_reason or 'crash'}."]
    if crashed:
        L += ["", f"**Crashed:** {crashed}. Every finished round is in the bank, and the next start resumes from it."]
    L += ["", "## Windows", "",
          "Start is the score at the start of today, now is the score of the text the window stands at, best is the"
          " lowest the bank has ever held.", "",
          "| Window | Start | Now | Best | Flagged now | Your lines regrown | State | Passes today | Rounds today |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    L += [f"| {r['wid']} | {_ai(r['start'])} | {_ai(r['now'])} | {_ai(r['best'])} | "
          f"{'' if r['flagged'] is None else r['flagged']} | {r['yours']} | {r['state']} | {r['passes']} | {r['rounds']} |"
          for r in rows]
    L += efficiency_lines(loop)
    L += model_lines(loop)
    L += as_posted_lines(loop)
    L += reflection_lines(loop)
    L += ["", "## Spend against each cap", "", "| Cap | Used | Limit |", "| --- | --- | --- |",
          f"| Pangram today (since midnight) | ${b['pangram_today']:.2f} | ${b['pangram_day_cap']:.2f} |",
          *([f"| Pangram this run-day | ${b['pangram_run_day']:.2f} | ${b['pangram_day_cap']:.2f} |"]
            if b.get("pangram_run_day") is not None else []),
          f"| Pangram this month | ${b['pangram_month']:.2f} | ${b['pangram_month_cap']:.2f} |",
          f"| Claude today, estimated dollars | ${b['claude_usd_today']:.2f} | "
          + (f"${b['backstop']:.2f} (first-run backstop) |" if b["backstop"] is not None else "no backstop now |"),
          f"| Claude today, points of your week | {b['loop_points_today']:.1%} | {b['allowance']:.1%} (today's allowance) |",
          f"| Loop share this week | {b['loop_share']:.1%} | {b['share_cap']:.0%} |",
          f"| Your week, real total | {b['real_total']:.0%} | {b['ceiling']:.0%} |",
          f"| 5-hour window | {b['five_hour']:.0%} | {b['five_hour_cap']:.0%} (the cap for this phase) |", "",
          f"The week resets {loop.clock.stamp(b['week_reset'])}. Conversion rate in use: {b['rate']:.4f} points of the"
          " week per estimated dollar (config seed until a quiet night refits it)."]
    L += ["", f"Pools drawn while a cap held and judged later: {loop.pools_used} ({loop.pool_draws_used} draws);"
          f" still waiting in the queue: {len(loop.queue.pools)}."]
    L += ["", "## Cap checks that said no", ""]
    if loop.cap_events:
        L += ["| Time | Check | Detail | Holds until |", "| --- | --- | --- | --- |"]
        L += [f"| {c['time']} | {c['check']} | {c['detail']} | {c['until']} |" for c in loop.cap_events]
    else:
        L.append("None.")
    L += ["", "## Cuts (the drop rule, 5.4)", "",
          "A cut is a human edit, so each one is here with its text. Say the word and it goes back."]
    L += [f"- {c['window']} paragraph {c['para'] + 1}, run {c['run'] + 1} ({c['time']}): \"{c['text']}\""
          f" Judge: {c['why']}{' **Put back: you said keep it going on Discord.**' if c.get('vetoed') else ''}"
          for c in loop.cuts] or ["", "None."]
    if any(c.get("question") and not c.get("vetoed") for c in loop.cuts):
        L += ["", "Each cut has its question on Discord: answer 2 (or react with a cross) and it goes back in, today"
              " or on a later run-day."]
    L += ["", "## Couldn't crack so far", "", "The full list, with context, is in `couldnt crack.md` in the job folder.", ""]
    L += whole_flag_lines(loop)
    L += crack_lines(loop, short=True) or ["Nothing yet."]
    L += ["", "## Timeline", "", "```", *loop.timeline[-400:], "```"]
    return "\n".join(L) + "\n"


def efficiency_rows(loop):
    """Today's efficiency per window (engine/analytics.py): the score the day began with, the window score of what
    ships (a whole-text pick only ever scored in its section falls back to the best), Pangram dollars per 0.01 of
    AI share removed, and draws per accepted run."""
    from engine import analytics
    since = loop.budget.run_day_start or getattr(loop, "smoke_t0", None) or 0.0
    wins = []
    for w in loop.windows:
        if getattr(w, "must", None) is None:
            continue
        e = loop.ship_entry(w)
        hit = loop.latest_score(w.wid, e["text"]) if e and e.get("text") else None
        ship = hit[0] if hit else (loop.best(w) or {}).get("ai")
        wins.append((w.wid, w.start_of_day, ship))
    return analytics.efficiency(loop.conn, loop.bank.entries, since, loop.clock.now(), wins,
                                purpose=getattr(loop, "purpose", "loop"))


def efficiency_lines(loop):
    try:
        rows = efficiency_rows(loop)
    except Exception:  # noqa: BLE001  (the digest goes out whatever the analytics make of today)
        return []
    if not rows:
        return []
    L = ["", "## Efficiency today", "",
         "Pangram dollars per 0.01 of AI share removed, start of today to what ships, and draws per accepted run (a"
         " judged run counts as one draw of 48 unless the bank holds its count). The whole history is"
         " `python -m engine.analytics`.", "",
         "| Window | Start | Shipped | Pangram $ | $ per point | Accepted | Draws per accepted |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        L.append(f"| {r['wid']} | {_ai(r['start'])} | {_ai(r['shipped'])} | ${r['pangram_usd']:.2f} |"
                 f" {'no gain' if r['cost_per_point'] is None else '$' + format(r['cost_per_point'], '.3f')} |"
                 f" {r['accepted']} | {'' if r['draws_per_accepted'] is None else r['draws_per_accepted']} |")
    L += arm_lines(rows)
    L += tiny_lines(loop)
    L += restore_lines(loop)
    return L


def tiny_lines(loop):
    """The runs the tiny-line rule skipped today (engine/run.py's tiny_skip): under tiny_words words, tiny_tries tries
    with no accepted change."""
    items = []
    for w in loop.windows:
        for (pi, ri), text in sorted((getattr(w, "tiny_skipped", None) or {}).items()):
            n = (getattr(w, "tiny_misses", None) or {}).get((pi, ri), 0)
            items.append(f"- {w.wid} p{pi + 1}.{ri + 1} \"{text[:60]}\" ({n} tries with no change)")
    if not items:
        return []
    return ["", f"Skipped as tiny (under {getattr(loop, 'tiny_words', 6)} words, no change after"
            f" {getattr(loop, 'tiny_tries', 3)} tries): {len(items)} run(s).", *items]


def restore_lines(loop):
    """What the restore pass did at the wrap-up (engine/restore.py), when it ran."""
    r = getattr(loop, "restore_result", None)
    if not r or r.get("skipped") == "run_at_wrapup is off":
        return []
    if r.get("skipped"):
        return ["", f"Restore pass: skipped ({r['skipped']})."]
    secs = "; ".join(f"section {n} {s}" for n, s in (r.get("sections") or {}).items()) or "no section touched"
    return ["", f"Restore pass: {r.get('restored', 0)} of {r.get('looked_at', 0)} paragraph(s) restored; {secs};"
            f" Pangram ${r.get('pangram_usd', 0):.2f}, Claude ${r.get('claude_usd', 0):.2f}; report `{r.get('report', '')}`."]


def arm_lines(rows):
    """The writer arms today (6.9, Phase 3): the adapter against the few-shot writer, from the efficiency rows."""
    from engine import analytics
    arms = {}
    for r in rows:
        analytics.add_arm_stats(arms, r.get("arms") or {})
    table = analytics.arm_rows(arms)
    if not table:
        return []
    L = ["", "### Writer arms today", "",
         "| Arm | Draws | Attempts | Accepted | Acceptance | Draws per accepted | Share change credited |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for a in table:
        L.append(f"| {a['arm']} | {a['draws']} | {a['attempts']} | {a['accepted']} |"
                 f" {'' if a['acceptance'] is None else format(a['acceptance'], '.0%')} |"
                 f" {a['draws_per_accepted'] or ''} | {a['share_change']:+.2f} over {a['share_rounds']} round(s) |")
    return L


def model_lines(loop):
    """The Pangram model-change alert (plan 9), while it's unacknowledged."""
    st = getattr(loop, "model_status", None)
    if not st or not st.get("stale"):
        return []
    return ["", "## Pangram model change", "",
            f"**Pangram answered with version {st['latest']}** (first seen {st.get('changed_at_text', '?')}); the"
            f" banked scores were made with {st['trusted']}. Until you set `model_ack = \"{st['latest']}\"` under"
            " [pangram] in config.toml, a score banked before the change is rescored before the loop trusts it, the"
            " cache ignores answers from the old version, and the whole-text round runs again."]


def is_yours(w, seg):
    """A flagged segment that holds a line of yours (plan 5.1), so the reports can say whose line it is."""
    seg = " ".join(seg.split())
    for pi, p in enumerate(getattr(w, "runs", None) or []):
        for ri, r in enumerate(p):
            if r.get("author") and any(s[:40] in seg for s in w.current[pi][ri]):
                return True
    return False


def as_posted_lines(loop):
    """The whole-text round and the wrap-up (5.6): each section as posted, at today's first whole-text scoring, now
    and in the final scan, next to its windows' own scores (the prose the loop works on, examples out)."""
    now = getattr(loop, "as_posted_now", None) or {}
    final = getattr(loop, "final_scan", None) or {}
    first = getattr(loop, "as_posted_first", None) or {}
    secs = sorted(set(now) | set(final))
    if not secs:
        return ["", "## As posted", "", "No whole-text round ran today."]
    L = ["", "## As posted (the whole-text round, 5.6)", "",
         "Each section scored as one document the way a reader would paste it: example lines in, comments and marks"
         " out. The windows column is the prose the loop works on, scored window by window.", "",
         "| Section | First today | After the whole-text round | Final scan | Flagged now | Windows |",
         "| --- | --- | --- | --- | --- | --- |"]
    for n in secs:
        wins = ", ".join(f"{w.wid} {_ai(w.ai)}" for w in loop.windows if w.w.get("section") == n)
        cur = now.get(n, {})
        L.append(f"| {n} | {_ai(first.get(n))} | {_ai(cur.get('ai'))} | {_ai(final.get(n))} |"
                 f" {cur.get('flagged', '')} | {wins} |")
    # plan 9: the sections combined (each counted by its words) and the whole document as it stands, two lines
    # under the table (engine/guide_scan.py)
    from engine.guide_scan import digest_lines
    extra = digest_lines(loop)
    if extra:
        L += ["", *extra]
    if getattr(loop, "whole_log", None):
        L += ["", "Whole-text scorings: " + "; ".join(
            f"{r['time'][-5:]} pass {r['k']}, " + ", ".join(f"section {n} {_ai(v)}" for n, v in r["sections"].items())
            + (" (better)" if r["improved"] else " (best known restored)" if r.get("restored") else "")
            for r in loop.whole_log)]
    rep = getattr(loop, "writeback_report", None)
    if rep:
        L += ["", "**Write-back** into the document: " + ("written" if rep.get("written") else "not written"
              " (a dry run, or nothing to change)") + f". {len(rep['replaced'])} paragraph(s) replaced"
              + (f": {', '.join(rep['replaced'])}" if rep["replaced"] else "") + "."]
        if rep["missing"]:
            L.append("Not found in the document, so left alone (edited or moved since?): " + ", ".join(rep["missing"]))
        for n, txt in rep["unrecognized"]:
            L.append(f"- section {n}, left as it is (not text the loop knows): {txt}")
    return L


def whole_flag_lines(loop):
    """What Pangram still flags in each section as posted, with whose line it is."""
    out = []
    for n, sc in sorted((getattr(loop, "as_posted_now", None) or {}).items()):
        flagged = [t.strip() for t, f in sc.get("segs", []) if f and t.strip()]
        if not flagged:
            continue
        out += [f"### Section {n} as posted (AI share {_ai(sc.get('ai'))})", ""]
        for t in flagged:
            whose = next(("yours" if is_yours(w, t) else "the loop's or Claude's" for w in loop.windows
                          if w.w.get("section") == n and getattr(w, "current", None) and any(
                              s[:40] in " ".join(t.split()) for p in w.current for r in p for s in r)), "an example line or heading")
            out.append(f"- ({whose}) \"{t}\"")
        out.append("")
    return out


def crack_lines(loop, short=False):
    out = []
    for w in loop.windows:
        if not getattr(w, "current", None):
            continue
        flagged = [t for t, f in w.segs if f and t.strip()]
        facts = [(k, v) for k, v in w.history.items() if v and "fact check" in v[-1] and k[0] < len(w.current)
                 and k[1] < len(w.current[k[0]]) and w.current[k[0]][k[1]] == w.original[k[0]][k[1]]]
        if not (flagged or facts or w.couldnt_cut):
            continue
        out.append(f"### {w.wid} (AI share {_ai(w.ai)})")
        out.append("")
        for t in flagged:
            out.append(f"- Pangram still flags{' (yours)' if is_yours(w, t) else ''}: \"{t}\"")
        for (pi, ri), res in facts:
            out.append(f"- Fact check keeps failing (p{pi + 1}.{ri + 1}, {len(res)} tries): \"{' '.join(w.original[pi][ri])}\"")
        for pi, ri, why in w.couldnt_cut:
            out.append(f"- Not cut ({why}), p{pi + 1}.{ri + 1}: \"{' '.join(w.original[pi][ri])}\"")
        out.append("")
    return out


def couldnt_crack_report(loop):
    """jobs/<job>/couldnt crack.md (plan 5.6 step 2): for reading, not a to-do list. Each line with its
    neighbors and what was tried."""
    L = [f"# Couldn't crack ({loop.job.name})", "",
         f"Written {loop.clock.stamp()} by `engine/run.py`. Every sentence Pangram still flags, every run the drop"
         " rule cut, and every fact-check failure the loop couldn't resolve, so far. The loop keeps pushing these"
         " on later run-days.", ""]
    for w in loop.windows:
        if not getattr(w, "current", None):
            continue
        items = []
        flat_sents = [s for p in w.current for rr in p for s in rr]
        for t, f in w.segs:
            if not (f and t.strip()):
                continue
            i = next((k for k, s in enumerate(flat_sents) if t.strip()[:40] in s or s[:40] in t), None)
            ctx_b = flat_sents[i - 1] if i else ""
            ctx_a = flat_sents[i + 1] if i is not None and i + 1 < len(flat_sents) else ""
            items += [f"- **Pangram flags{' (your line)' if is_yours(w, t) else ''}:** \"{t.strip()}\"",
                      f"  - before: {ctx_b or '(start)'}", f"  - after: {ctx_a or '(end)'}"]
        for (pi, ri), res in w.history.items():
            if res and pi < len(w.current) and ri < len(w.current[pi]) and w.current[pi][ri] == w.original[pi][ri]:
                tried = "; ".join(f"{r.split('(')[0].strip()}" for r in res[-4:])
                whose = "your line, still as you wrote it" if w.runs[pi][ri].get("author") else "still Claude's text"
                items += [f"- **{whose[0].upper() + whose[1:]} after {len(res)} tries** (p{pi + 1}.{ri + 1}):"
                          f" \"{' '.join(w.original[pi][ri])}\"", f"  - last tries: {tried}"]
        for pi, ri, why in w.couldnt_cut:
            items.append(f"- **The drop rule couldn't cut it** ({why}): \"{' '.join(w.original[pi][ri])}\"")
        for c in loop.cuts:
            if c["window"] == w.wid:
                items.append(f"- **Cut by the drop rule** ({c['time']}): \"{c['text']}\" Judge: {c['why']}")
        if items:
            L += [f"## {w.wid} (AI share {_ai(w.ai)})", "", *items, ""]
    whole = whole_flag_lines(loop)
    if whole:
        L += ["## As posted", "", "What Pangram flags when each section is scored whole, the way a reader pastes it.", "",
              *whole]
    if len(L) == 4:
        L.append("Nothing so far.")
    return "\n".join(L) + "\n"
