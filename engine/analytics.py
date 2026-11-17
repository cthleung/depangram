"""The loop's history (plan 10.1 and 10.2): what each window, section and run-day cost and what it bought.

    python -m engine.analytics            # writes analytics/loop history.md and analytics/loop history.json

Everything comes from data the loop already keeps, read only: the SQLite ledger (pangram_calls, pangram_cache,
claude_calls, run_days, claude_rates, baseline_scores), the job's bank and the digests in digests/. Nothing here
calls Pangram, Claude or the GPU.

A real run-day is a row in run_days (smoke tests and rehearsals never write one there). Ledger rows and bank entries
count for a run-day when their time falls between its start and its end, so the smoke rounds before and between the
run-days stay out, and a run-day that crosses midnight keeps its rows after 00:00.

What the bank can't say exactly, and how it's read here:
- draws: a judged run banks "N passed filters", not how many candidates were drawn, so it counts as one draw of
  N_CANDS (48) unless the change says otherwise ("draws" in changes banked from 2026-11-09 on, or "(N drawn: ...)"
  in a run where nothing passed). Pools judged from the queue held up to 192, so the figure is a floor.
- the reject tally: only failed runs banked it before 2026-11-09, so the check counts lean to failed runs.
- a window's shipped share: the window score of the text the working copy ships (a whole-text pick when one was
  banked after the window's best, else the best). A whole-text pick was only ever scored inside its section, so its
  window share is unknown and the best's share stands in, marked.
- the start share is the window's score when its first real run-day began (the digest's "Start"), so a window
  seeded from the experiments (s1w0, s3w0) isn't credited with work the experiments paid for. The original's
  round 0 score is listed next to it.
- the must-keep filter isn't re-run here, so "best" is the lowest banked window score.
"""
import argparse
import datetime as dt
import json
import re
import sqlite3
import statistics
from pathlib import Path

from engine.settings import SANDBOX, Settings

N_CANDS = 48                       # candidates per draw (engine/draws.py)
RUN_LABEL = re.compile(r"^run (s\d+w\d+) (.*)$")
WHOLE_LABEL = re.compile(r"^run whole s(\d+) ")
DRAWN = re.compile(r"\((\d+) drawn: ([^)]*)\)")
WINDOW_KINDS = ("round", "context")
LOOP_ROLE = "loop %"


# ---------------------------------------------------------------------- small readers
def parse_stamp(s):
    """'2026-11-09 05:54' (the bank's and the digests' stamps) as local epoch seconds, or None."""
    try:
        return dt.datetime.strptime(s.strip()[:16], "%Y-%m-%d %H:%M").timestamp()
    except (AttributeError, ValueError):
        return None


def stamp(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""


def in_span(ts, start, end):
    """A minute stamp counts when it falls in [start, end]: the bank stamps to the minute, so start is floored."""
    return ts is not None and start - start % 60 <= ts <= end


def load_bank(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def open_ro(path):
    """The ledger, read only (db.connect would run the schema script)."""
    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)


def real_run_days(conn):
    rows = conn.execute("SELECT day, started, ended FROM run_days WHERE done = 1 AND started IS NOT NULL"
                        " AND ended IS NOT NULL ORDER BY started").fetchall()
    return [{"day": d, "start": s, "end": e} for d, s, e in rows]


def cost_per_point(usd, start, end):
    """Dollars per 0.01 of AI share removed; None when nothing was removed (never inf in the JSON)."""
    if usd is None or start is None or end is None:
        return None
    gain = (start - end) * 100
    return round(usd / gain, 4) if gain > 1e-9 else None


def ratio(a, b):
    return round(a / b, 1) if b else None


def tally_items(text):
    """'must-keep hedge 30, length 10' -> {'must-keep hedge': 30, 'length': 10}."""
    out = {}
    for item in (text or "").split(", "):
        name, _, n = item.strip().rpartition(" ")
        if name and n.isdigit():
            out[name] = out.get(name, 0) + int(n)
    return out


def change_stats(ch):
    """(draws, exact, accepted, tally) for one banked change. A judged run without a banked count is one draw."""
    res = ch.get("result", "") or ""
    m = DRAWN.search(res)
    tally = tally_items(ch["tally"]) if ch.get("tally") else (tally_items(m.group(2)) if m else {})
    if res.startswith("queued"):
        draws, exact = 0, True              # its pool goes back to the queue, and counts when it's judged
    elif ch.get("draws") is not None:
        draws, exact = int(ch["draws"]), True
    elif m:
        draws, exact = int(m.group(1)), True
    else:
        draws, exact = N_CANDS, False
    accepted = bool(ch.get("new")) and res.startswith("replaced")
    return draws, exact, accepted, tally


def arm_of_change(ch):
    """The writer arm a banked change came from (6.9, Phase 3). Changes banked before the arms existed (up to
    2026-11-09) were all the few-shot writer's, so a change with no arm reads as "fewshot"."""
    return ch.get("arm") or "fewshot"


def new_arm_stats():
    return {"draws": 0, "attempts": 0, "accepted": 0, "share_change": 0.0, "share_rounds": 0}


def add_arm_stats(into, more):
    for arm, a in more.items():
        t = into.setdefault(arm, new_arm_stats())
        for k in t:
            t[k] += a.get(k, 0)
    return into


def arm_rows(arms):
    """The per-arm table's rows: draws, attempts, accepted, acceptance rate, draws per accepted and the AI share
    change credited to the arm (the drop in a window's score over a round whose accepted runs were all its own;
    a round that accepted runs from two arms is credited to "mixed")."""
    out = []
    for arm, a in sorted(arms.items()):
        out.append({"arm": arm, "draws": a["draws"], "attempts": a["attempts"], "accepted": a["accepted"],
                    "acceptance": ratio(a["accepted"], a["attempts"]) if a["attempts"] else None,
                    "draws_per_accepted": ratio(a["draws"], a["accepted"]),
                    "share_change": round(a["share_change"], 2), "share_rounds": a["share_rounds"]})
    return out


# ---------------------------------------------------------------------- the digests
def parse_table(lines, i):
    """The markdown table starting at lines[i] (its header row) as a list of dicts, and the line after it."""
    head = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    rows, j = [], i + 2
    while j < len(lines) and lines[j].startswith("|"):
        cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
        rows.append(dict(zip(head, cells)))
        j += 1
    return rows, j


def share(cell):
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def parse_digest(path):
    """A run-day digest: its window table, its cap checks that said no, the "has lifted" lines of its timeline and
    its as-posted table."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out = {"windows": {}, "caps": [], "lifted": [], "as_posted": {}}
    section = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("## "):
            section = ln[3:].strip()
        elif ln.startswith("| ") and i + 1 < len(lines) and lines[i + 1].startswith("| ---"):
            rows, i = parse_table(lines, i)
            if section == "Windows":
                for r in rows:
                    out["windows"][r.get("Window")] = {"start": share(r.get("Start")), "now": share(r.get("Now")),
                                                       "best": share(r.get("Best")), "state": r.get("State", ""),
                                                       "passes": r.get("Passes today"), "rounds": r.get("Rounds today")}
            elif section == "Cap checks that said no":
                out["caps"] = [{"time": r.get("Time"), "check": r.get("Check"), "detail": r.get("Detail"),
                                "until": r.get("Holds until")} for r in rows]
            elif section and section.startswith("As posted"):
                for r in rows:
                    out["as_posted"][r.get("Section")] = {"first": share(r.get("First today")),
                                                          "after": share(r.get("After the whole-text round")),
                                                          "final": share(r.get("Final scan"))}
            continue
        m = re.match(r"^(\d{4}-\d\d-\d\d \d\d:\d\d) (.+) has lifted$", ln)
        if m:
            out["lifted"].append((m.group(2), m.group(1)))
        i += 1
    return out


def digests_by_day(ddir):
    out = {}
    for p in sorted(Path(ddir).glob("* digest.md")):
        day = p.name.split(" ")[0]
        out[day] = parse_digest(p)
    return out


def holds(digest, day_end):
    """Each cap hold of a run-day: when it said no, which check, why, and how many minutes it held inside the day
    (to whichever comes first: its "until", the "has lifted" line, the day's end)."""
    out = []
    for c in (digest or {}).get("caps", []):
        t0 = parse_stamp(c["time"])
        if t0 is None:
            continue
        ends = [day_end]
        until = parse_stamp(c.get("until") or "")
        if until:
            ends.append(until)
        lift = [parse_stamp(t) for chk, t in digest.get("lifted", []) if chk == c["check"]
                and (parse_stamp(t) or 0) >= t0]
        if lift:
            ends.append(min(lift))
        out.append({"time": c["time"], "check": c["check"], "detail": c["detail"],
                    "minutes": round(max(0.0, min(ends) - t0) / 60)})
    return out


# ---------------------------------------------------------------------- the ledger
def pangram_rows(conn, start, end, purpose="loop"):
    return conn.execute("SELECT ts, label, usd FROM pangram_calls WHERE ts >= ? AND ts <= ? AND purpose = ?"
                        " AND usd > 0", (start, end, purpose)).fetchall()


def pangram_kind(label):
    m = RUN_LABEL.match(label or "")
    if WHOLE_LABEL.match(label or ""):
        return "whole-text round"
    if (label or "").startswith("run as posted"):
        return "final scan"
    if (label or "").startswith("run guide as it stands"):
        return "guide scan"
    if not m:
        return "other"
    rest = m.group(2)
    if rest.startswith("round 0"):
        return "round 0"
    if rest.startswith("rescore"):
        return "rescores"
    if rest.startswith("cut"):
        return "drop-rule cuts"
    if re.match(r"g\d+", rest):
        return "window rounds"
    return "other"


def pangram_by_window(conn, start, end, purpose="loop"):
    out = {}
    for _, label, usd in pangram_rows(conn, start, end, purpose):
        m = RUN_LABEL.match(label or "")
        if m and not WHOLE_LABEL.match(label):
            out[m.group(1)] = out.get(m.group(1), 0.0) + usd
    return out


def claude_rows(conn, start, end):
    return conn.execute("SELECT role, cost_usd FROM claude_calls WHERE role LIKE ? AND ts >= ? AND ts <= ?",
                        (LOOP_ROLE, start, end)).fetchall()


def claude_by_window(conn, start, end):
    out = {}
    for role, cost in claude_rows(conn, start, end):
        parts = role.split()
        if len(parts) >= 3 and re.match(r"s\d+w\d+$", parts[1]):
            out[parts[1]] = out.get(parts[1], 0.0) + (cost or 0.0)
    return out


def rate_in_use(conn, cfg, kind, before):
    """engine/budget.py's rate() as it read when a run-day began: the median of the last seven refits before it,
    held between rate_clamp times the seed, else the seed."""
    seed = float(cfg["week_points_per_usd" if kind == "week" else "five_hour_points_per_usd"])
    rows = [r[0] for r in conn.execute("SELECT rate FROM claude_rates WHERE kind = ? AND ts < ? ORDER BY ts DESC"
                                       " LIMIT 7", (kind, before))]
    if not rows:
        return seed, "seed"
    lo, hi = (float(x) for x in cfg.get("rate_clamp", [0.5, 2.0]))
    return min(max(statistics.median(rows), lo * seed), hi * seed), "refit"


# ---------------------------------------------------------------------- the bank
def window_entries(entries, wid, start=None, end=None):
    out = []
    for i, e in enumerate(entries):
        if e.get("window") != wid:
            continue
        if start is not None and not in_span(parse_stamp(e.get("time", "")), start, end):
            continue
        out.append((i, e))
    return out


def best_entry(entries, wid, until=None):
    """The lowest-scoring banked window text (humanized texts never count), the latest on a tie, as (index, entry)."""
    cands = [(i, e) for i, e in window_entries(entries, wid) if e.get("runs") is not None and e.get("ai") is not None
             and not e.get("humanized") and (until is None or (parse_stamp(e.get("time", "")) or 0) <= until)]
    return min(cands, key=lambda ie: (ie[1]["ai"], -ie[0])) if cands else None


def window_score_of(entries, conn, wid, text):
    """The window score of exactly this text: the newest scored bank entry holding it, else the cache."""
    for _, e in reversed(window_entries(entries, wid)):
        if e.get("text") == text and e.get("ai") is not None:
            return e["ai"]
    if conn is not None and text:
        import hashlib
        row = conn.execute("SELECT response_json FROM pangram_cache WHERE sha = ?",
                           (hashlib.sha256(text.encode("utf-8")).hexdigest(),)).fetchone()
        if row:
            return json.loads(row[0]).get("fraction_ai")
    return None


def shipped(entries, conn, wid, until=None):
    """(share, how): what the working copy ships for the window, the way run.py's ship_entry picks it."""
    best = best_entry(entries, wid, until)
    # a "fix", "flavor" or "restore" entry (the fix, flavor and restore passes' edits on the shipped text) ships the
    # same way (the list mirrors run.py's SHIP_KINDS)
    ships = [(i, e) for i, e in window_entries(entries, wid) if e.get("kind") in ("ship", "fix", "flavor", "restore")
             and e.get("runs") is not None
             and (until is None or (parse_stamp(e.get("time", "")) or 0) <= until)]
    if ships and (best is None or ships[-1][0] > best[0]):
        s = window_score_of(entries, conn, wid, ships[-1][1].get("text"))
        if s is not None:
            return s, "whole-text pick"
        return (best[1]["ai"] if best else None), "whole-text pick (unscored alone; best shown)"
    return (best[1]["ai"] if best else None), "best"


def round_stats(pairs):
    """Rounds, context rounds, passes, draws, accepted runs, your lines regrown and the reject tally over a list of
    (index, entry) of one window."""
    st = {"rounds": 0, "context_rounds": 0, "passes": set(), "draws": 0, "draws_exact": True, "attempts": 0,
          "accepted": 0, "accepted_context": 0, "yours_regrown": 0, "tally": {}, "arms": {}}
    last_ai = None
    for _, e in pairs:
        k = e.get("kind")
        if k not in WINDOW_KINDS:
            if e.get("ai") is not None and (e.get("runs") is not None or k == "start"):
                last_ai = e["ai"]              # a start, seed, rescore or cut: the score the next round moves from
            continue
        if k == "round":
            st["rounds"] += 1
            if e.get("pass"):
                st["passes"].add(e["pass"])
        else:
            st["context_rounds"] += 1
        accepted_arms = set()
        for ch in e.get("changes") or []:
            if "run" not in ch:
                continue
            draws, exact, acc, tally = change_stats(ch)
            arm = st["arms"].setdefault(arm_of_change(ch), new_arm_stats())
            st["draws"] += draws
            arm["draws"] += draws
            st["draws_exact"] = st["draws_exact"] and exact
            if not (ch.get("result") or "").startswith("queued"):
                st["attempts"] += 1
                arm["attempts"] += 1
            if acc:
                st["accepted"] += 1
                arm["accepted"] += 1
                accepted_arms.add(arm_of_change(ch))
                st["accepted_context"] += k == "context"
                st["yours_regrown"] += ch.get("who") == "yours"
            for name, n in tally.items():
                st["tally"][name] = st["tally"].get(name, 0) + n
        # the share change credited to an arm: a scored round whose accepted runs were all that arm's
        if k == "round" and e.get("ai") is not None:
            if last_ai is not None and accepted_arms:
                arm = st["arms"].setdefault(next(iter(accepted_arms)) if len(accepted_arms) == 1 else "mixed",
                                            new_arm_stats())
                arm["share_change"] += last_ai - e["ai"]
                arm["share_rounds"] += 1
            last_ai = e["ai"]
    st["passes"] = len(st["passes"])
    return st


def efficiency(conn, entries, since, until, windows, purpose="loop"):
    """One row per window for a stretch of time (the digest's table, a run-day's): `windows` is [(wid, start share,
    shipped share)]. Pangram dollars per point and draws per accepted run."""
    pg = pangram_by_window(conn, since, until, purpose) if conn is not None else {}
    rows = []
    for wid, start, ship in windows:
        st = round_stats(window_entries(entries, wid, since, until))
        usd = round(pg.get(wid, 0.0), 2)
        rows.append({"wid": wid, "start": start, "shipped": ship, "pangram_usd": usd,
                     "cost_per_point": cost_per_point(usd, start, ship), "accepted": st["accepted"],
                     "draws": st["draws"], "draws_per_accepted": ratio(st["draws"], st["accepted"]),
                     "arms": st["arms"]})
    return rows


# ---------------------------------------------------------------------- stuck rule
def stuck_fires(changed, after_round, streak):
    """engine/run.py's stuck_rule, round by round: the first round count at which it fires, or None."""
    for n in range(1, len(changed) + 1):
        if n >= after_round + streak and not any(changed[n - streak:n]):
            return n
    return None


def passes_of(entries, days):
    """{(wid, pass): [round entries in order]} for every pass with a round in a real run-day."""
    out = {}
    for e in entries:
        if e.get("kind") != "round" or not e.get("pass"):
            continue
        ts = parse_stamp(e.get("time", ""))
        if not any(in_span(ts, d["start"], d["end"]) for d in days):
            continue
        out.setdefault((e["window"], e["pass"]), []).append(e)
    return out


def stuck_reading(entries, days, after_round, streak, early_after=1):
    """What the stuck rule did, and what an earlier reading would have done. For each pass: the round the rule fired
    at as configured, and the round it would fire at with `early_after` (the earlier reading); the rounds that
    reading would skip (the settle round after it still runs), the runs those rounds replaced, and whether the pass's
    lowest score came in one of them."""
    out = []
    for (wid, p), rounds in sorted(passes_of(entries, days).items()):
        done = [e for e in rounds if not e.get("interrupted")]
        changed = [bool(e.get("changed", True)) for e in done]
        now = stuck_fires(changed, after_round, streak)
        early = stuck_fires(changed, early_after, streak)
        skipped = done[early + 1:] if early is not None else []
        kept = done[:early + 1] if early is not None else done
        acc = sum(1 for e in skipped for ch in e.get("changes") or [] if change_stats(ch)[2])
        lows = [e["ai"] for e in kept if e.get("ai") is not None]
        lost = [e["ai"] for e in skipped if e.get("ai") is not None and (not lows or e["ai"] < min(lows) - 1e-9)]
        out.append({"window": wid, "pass": p, "rounds": len(done), "fires_now": now, "fires_early": early,
                    "skipped": len(skipped), "accepted_in_skipped": acc, "lower_score_in_skipped": bool(lost)})
    return out


# ---------------------------------------------------------------------- the history
def norm(s):
    return " ".join((s or "").replace("**", "").split())


def whose(vmap, seg):
    if vmap is None:
        return "?"
    v = vmap.get(norm(seg))
    if v is None:
        first = re.split(r"(?<=[.!?])\s+", norm(seg))[0]
        v = vmap.get(first)
    return {"yours": "yours", "yours?": "yours", "edited": "yours", "both": "both"}.get(v, "Claude's" if v else
                                                                                      "the loop's")


def always_flagged(entries, days, vmap=None):
    """Segments of a window's text Pangram flagged every time the text held them, over at least two scorings in at
    least two run-days (the couldn't-crack view across days)."""
    out = []
    wids = sorted({e["window"] for e in entries if e.get("window")})
    for wid in wids:
        seen = {}
        for _, e in window_entries(entries, wid):
            ts = parse_stamp(e.get("time", ""))
            day = next((d["day"] for d in days if in_span(ts, d["start"], d["end"])), None)
            if day is None or e.get("ai") is None or not e.get("segs") or not e.get("text"):
                continue
            body = norm(e["text"])
            flagged = {norm(t) for t, f in e["segs"] if f and norm(t)}
            for t in dict.fromkeys(norm(t) for t, _ in e["segs"] if norm(t)):
                if t in body:
                    s = seen.setdefault(t, {"present": 0, "flagged": 0, "days": set()})
                    s["present"] += 1
                    s["flagged"] += t in flagged
                    s["days"].add(day)
        for t, s in seen.items():
            if s["present"] >= 2 and s["present"] == s["flagged"] and len(s["days"]) >= 2:
                out.append({"window": wid, "text": t, "scorings": s["present"], "whose": whose(vmap, t)})
    return out


def section_history(conn, entries, digests, days):
    """Each section's as-posted score over time: the baseline, the first run-day's write-back scored by hand, every
    whole-text scoring, and each digest's final scan."""
    out = {}
    for label, ai, ts in conn.execute("SELECT label, fraction_ai, ts FROM baseline_scores WHERE kind = 'guide as posted'"
                                      " ORDER BY ts"):
        n = label.replace("section", "").strip()
        if n.isdigit():
            out.setdefault(n, {})["baseline"] = {"when": stamp(ts), "what": "baseline", "ai": ai}
    rows = []
    for n, b in out.items():
        rows.append({"section": n, **b["baseline"]})
    for label, ts, resp in conn.execute(
            "SELECT p.label, p.ts, k.response_json FROM pangram_calls p JOIN pangram_cache k ON k.sha = p.sha"
            " AND k.model = p.model WHERE p.label LIKE 'as-posted section % after loop' ORDER BY p.ts"):
        m = re.match(r"as-posted section (\d+) after loop", label)
        if m:
            rows.append({"section": m.group(1), "when": stamp(ts), "what": "after the first run-day, written back"
                         " and scored by hand", "ai": json.loads(resp).get("fraction_ai")})
    for e in entries:
        if e.get("kind") != "whole":
            continue
        for n, s in (e.get("sections") or {}).items():
            rows.append({"section": str(n), "when": e.get("time"), "what": f"whole-text pass {e.get('k')}"
                         + (" (kept)" if int(n) in (e.get("improved") or []) or str(n) in map(str, e.get("improved")
                                                                                              or []) else ""),
                         "ai": s.get("ai")})
    for d in days:
        dg = digests.get(d["day"]) or {}
        for n, v in (dg.get("as_posted") or {}).items():
            if v.get("final") is not None:
                rows.append({"section": str(n), "when": stamp(d["end"]), "what": "final scan (digest)",
                             "ai": v["final"]})
    hist = {}
    for r in sorted(rows, key=lambda r: (int(r["section"]), r["when"] or "")):
        hist.setdefault(r["section"], []).append({k: r[k] for k in ("when", "what", "ai")})
    return hist


def build(conn, entries, digests, cfg, job=None):
    """The whole history as one dict (the JSON file), from the ledger, the bank and the digests."""
    days = real_run_days(conn)
    wids = sorted({e["window"] for e in entries if e.get("window") and e.get("kind") != "seed"})
    vmap = None
    if job is not None:
        try:
            vmap = job.verdict_map()
        except Exception:  # noqa: BLE001
            vmap = None
    lc, cl, pgc = cfg.get("loop", {}), cfg.get("claude", {}), cfg.get("pangram", {})
    out = {"written": stamp(dt.datetime.now().timestamp()), "run_days": [], "windows": {}, "sections": {},
           "moved": {}, "notes": [f"draws for a judged run count as {N_CANDS} (one draw) unless banked"]}
    # ------------------------------------------------------------ per run-day
    month_cap = float(pgc.get("monthly_cap_usd", 180))
    day_cap = float(pgc.get("nightly_cap_usd", 6))
    for i, d in enumerate(days):
        s, e = d["start"], d["end"]
        by = {}
        for _, label, usd in pangram_rows(conn, s, e):
            k = pangram_kind(label)
            by[k] = round(by.get(k, 0.0) + usd, 2)
        month_start = dt.datetime.fromtimestamp(e).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month = conn.execute("SELECT COALESCE(SUM(usd), 0) FROM pangram_calls WHERE ts >= ? AND ts <= ? AND usd > 0",
                             (month_start.timestamp(), e)).fetchone()[0]
        kinds = {}
        for role, cost in claude_rows(conn, s, e):
            k = role.split()[-1]
            kinds[k] = round(kinds.get(k, 0.0) + (cost or 0.0), 2)
        claude_usd = round(sum(kinds.values()), 2)
        week_rate, week_src = rate_in_use(conn, cl, "week", s)
        five_rate, five_src = rate_in_use(conn, cl, "five_hour", s)
        refits = [{"kind": k, "measured": round(r, 4), "points": p, "dollars": round(dl, 2)} for k, r, p, dl in
                  conn.execute("SELECT kind, rate, points, dollars FROM claude_rates WHERE ts >= ? AND ts <= ?",
                               (s, e + 3600))]
        for r in refits:
            seed = float(cl["week_points_per_usd" if r["kind"] == "week" else "five_hour_points_per_usd"])
            lo, hi = (float(x) for x in cl.get("rate_clamp", [0.5, 2.0]))
            r["in_use_next"] = round(min(max(r["measured"], lo * seed), hi * seed), 4)
            r["clamped"] = not (lo * seed <= r["measured"] <= hi * seed)
        dg = digests.get(d["day"])
        hold = holds(dg, e)
        # the stuck rule and the settle rounds
        ends = [x for x in entries if x.get("kind") == "pass_end" and in_span(parse_stamp(x.get("time", "")), s, e)]
        reasons = {}
        for x in ends:
            reasons[x.get("reason")] = reasons.get(x.get("reason"), 0) + 1
        settles = [x for x in entries if x.get("kind") == "round" and x.get("settle_next")
                   and in_span(parse_stamp(x.get("time", "")), s, e)]
        settled = [{"window": x["window"], "time": x["time"], "orig": ch.get("orig"), "new": ch.get("new")}
                   for x in settles for ch in x.get("changes") or [] if change_stats(ch)[2]]
        # plan 9 (2026-11-10): the guide as it stands and the sections combined, from the day's last `guide` entry
        # (engine/guide_scan.py banks one per scoring; a day without one leaves both blank)
        guides = [x for x in entries if x.get("kind") == "guide" and in_span(parse_stamp(x.get("time", "")), s, e)]
        g = guides[-1] if guides else {}
        out["run_days"].append({
            "guide": {"ai": g.get("ai"), "sections_combined": g.get("sections_combined"), "words": g.get("words"),
                      "flagged": g.get("flagged"), "skipped": g.get("skipped"), "time": g.get("time")},
            "day": d["day"], "start": stamp(s), "end": stamp(e), "hours": round((e - s) / 3600, 1),
            "pangram": {"total": round(sum(by.values()), 2), "by_purpose": by, "day_cap": day_cap,
                        "reserve": float(lc.get("whole_text_reserve_usd", 0) or 0) if i > 0 else None,
                        "month_to_date": round(month, 2), "month_cap": month_cap},
            "claude": {"estimated_usd": claude_usd, "by_kind": kinds,
                       "points_of_week": round(claude_usd * week_rate, 4),
                       "backstop_usd": float(cl.get("first_night_cap_usd", 20)) if i < int(cl.get("backstop_run_days", 1))
                       else None},
            "rates": {"week": round(week_rate, 4), "week_from": week_src, "five_hour": round(five_rate, 4),
                      "five_hour_from": five_src, "refits_at_the_end": refits},
            "holds": hold, "held_minutes": sum(h["minutes"] for h in hold),
            "pass_ends": reasons, "settle_rounds": len(settles), "settle_replaced": settled})
    # ------------------------------------------------------------ per window
    total_end = days[-1]["end"] if days else None
    for wid in wids:
        start_entry = next((e for _, e in window_entries(entries, wid) if e.get("kind") == "start"), None)
        rows = []
        first_start = None
        for d in days:
            dg = (digests.get(d["day"]) or {}).get("windows", {}).get(wid, {})
            pairs = window_entries(entries, wid, d["start"], d["end"])
            st = round_stats(pairs)
            pg = round(pangram_by_window(conn, d["start"], d["end"]).get(wid, 0.0), 2)
            cl_usd = round(claude_by_window(conn, d["start"], d["end"]).get(wid, 0.0), 2)
            day_start = dg.get("start")
            if day_start is None:              # no digest: the first score banked in the day
                day_start = next((e["ai"] for _, e in pairs if e.get("ai") is not None), None)
            if first_start is None:
                first_start = day_start
            b = best_entry(entries, wid, d["end"])
            ship, how = shipped(entries, conn, wid, d["end"])
            rows.append({"day": d["day"], "start": day_start, "end": dg.get("now"),
                         "best": b[1]["ai"] if b else None, "shipped": ship, "shipped_is": how,
                         "state": dg.get("state"), "rounds": st["rounds"], "context_rounds": st["context_rounds"],
                         "passes": st["passes"], "draws": st["draws"], "accepted": st["accepted"],
                         "yours_regrown": st["yours_regrown"], "pangram_usd": pg, "claude_usd": cl_usd,
                         "unchanged": day_start is not None and dg.get("now") is not None
                         and round(day_start, 2) == round(dg["now"], 2)})
        all_pairs = [p for d in days for p in window_entries(entries, wid, d["start"], d["end"])]
        st = round_stats(all_pairs)
        pg = round(sum(r["pangram_usd"] for r in rows), 2)
        b = best_entry(entries, wid, total_end)
        ship, how = shipped(entries, conn, wid, total_end)
        out["windows"][wid] = {
            "section": int(re.match(r"s(\d+)", wid).group(1)), "original": start_entry.get("ai") if start_entry else None,
            "start": first_start, "best": b[1]["ai"] if b else None, "shipped": ship, "shipped_is": how,
            "rounds": st["rounds"], "context_rounds": st["context_rounds"], "passes": st["passes"],
            "attempts": st["attempts"], "draws": st["draws"], "draws_exact": st["draws_exact"],
            "accepted": st["accepted"], "accepted_context": st["accepted_context"], "yours_regrown": st["yours_regrown"],
            "pangram_usd": pg, "claude_usd": round(sum(r["claude_usd"] for r in rows), 2),
            "cost_per_point": cost_per_point(pg, first_start, ship),
            "draws_per_accepted": ratio(st["draws"], st["accepted"]),
            "attempts_per_accepted": ratio(st["attempts"], st["accepted"]),
            "tally": st["tally"], "arms": st["arms"], "days": rows}
    # the writer arms (6.9, Phase 3; 10.2): draws per accepted, acceptance rate and the share change each is credited
    arms = {}
    for w in out["windows"].values():
        add_arm_stats(arms, w["arms"])
    out["arms"] = arm_rows(arms)
    # ------------------------------------------------------------ per section
    out["sections"] = section_history(conn, entries, digests, days)
    whole_usd = {}
    for d in days:
        for _, label, usd in pangram_rows(conn, d["start"], d["end"]):
            m = WHOLE_LABEL.match(label or "")
            if m:
                whole_usd[m.group(1)] = round(whole_usd.get(m.group(1), 0.0) + usd, 2)
    out["section_whole_usd"] = whole_usd
    # every scoring of the guide as it stands, the by-hand ones included (plan 9, 2026-11-10)
    out["guide_scans"] = [{"day": x.get("day"), "time": x.get("time"), "ai": x.get("ai"), "words": x.get("words"),
                           "flagged": x.get("flagged"), "how": x.get("how", "wrap-up"), "skipped": x.get("skipped"),
                           "sections_combined": x.get("sections_combined")}
                          for x in entries if x.get("kind") == "guide" and not x.get("placeholder")]
    # ------------------------------------------------------------ what moved and what didn't
    tally = {}
    for w in out["windows"].values():
        for k, n in w["tally"].items():
            tally[k] = tally.get(k, 0) + n
    whole_flags = {}
    for e in entries:
        if e.get("kind") != "whole":
            continue
        for n, s in (e.get("sections") or {}).items():
            whole_flags.setdefault(str(n), []).append({norm(t) for t, f in s.get("segs") or [] if f and norm(t)})
    every_whole = {n: sorted(set.intersection(*sets)) for n, sets in whole_flags.items() if len(sets) >= 2}
    out["moved"] = {
        "unchanged_windows": [{"day": r["day"], "window": wid, "share": r["start"]}
                              for wid, w in out["windows"].items() for r in w["days"] if r["unchanged"]],
        "always_flagged": always_flagged(entries, days, vmap),
        "always_flagged_as_posted": {n: [{"text": t, "whose": whose(vmap, t)} for t in ts] for n, ts in
                                     every_whole.items()},
        "rejects": dict(sorted(tally.items(), key=lambda kv: -kv[1])),
        "stuck_reading": stuck_reading(entries, days, int(lc.get("stuck_after_round", 3)),
                                       int(lc.get("stuck_streak", 2))),
    }
    return out


# ---------------------------------------------------------------------- the markdown
def f2(x):
    return "" if x is None else f"{x:.2f}"


def usd(x):
    return "" if x is None else f"${x:.2f}"


def short(t, n=160):
    t = " ".join((t or "").split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " ..."


def cpp(x):
    return "no gain" if x is None else f"${x:.3f}"


def render(h):
    L = ["# Loop history", "",
         f"Written {h['written']} by `python -m engine.analytics`, from the ledger, the bank and the digests. Real"
         f" run-days only ({', '.join(d['day'] for d in h['run_days']) or 'none yet'}); the smoke tests and the"
         " rehearsals are left out. Shares are Pangram's AI share, 0 human to 1 AI.", ""]
    # windows
    L += ["## Windows", "",
          "Start is the window's score when its first real run-day began; original is round 0, the guide's text before"
          " the loop. Cost per point is Pangram dollars per 0.01 of AI share removed, start to shipped. Draws per"
          " accepted counts a judged run as one draw of 48 unless the bank says otherwise, so it's a floor.", "",
          "| Window | Original | Start | Best | Shipped | Rounds | Passes | Draws | Accepted | Yours regrown"
          " | Pangram $ | Claude $ (est.) | $ per point | Draws per accepted | Tries per accepted |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    marks = False
    for wid, w in h["windows"].items():
        mark = "*" if "unscored alone" in (w["shipped_is"] or "") else ""
        marks = marks or bool(mark)
        rounds = f"{w['rounds']}" + (f" + {w['context_rounds']} in context" if w["context_rounds"] else "")
        acc = f"{w['accepted']}" + (f" ({w['accepted_context']} in context)" if w["accepted_context"] else "")
        L.append(f"| {wid} | {f2(w['original'])} | {f2(w['start'])} | {f2(w['best'])} | {f2(w['shipped'])}{mark} |"
                 f" {rounds} | {w['passes']} | {w['draws']} | {acc} | {w['yours_regrown']} | {usd(w['pangram_usd'])} |"
                 f" {usd(w['claude_usd'])} | {cpp(w['cost_per_point'])} | {w['draws_per_accepted'] or ''} |"
                 f" {w['attempts_per_accepted'] or ''} |")
    if marks:
        L += ["", "\\* The window ships a whole-text pick, which was only scored inside its section, so the best's"
              " share stands in."]
    L += ["", "### Run-day by run-day", "",
          "| Window | Run-day | Start | End | Best | Rounds | Passes | Draws | Accepted | Yours regrown | Pangram $ |"
          " Claude $ (est.) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for wid, w in h["windows"].items():
        for r in w["days"]:
            L.append(f"| {wid} | {r['day']} | {f2(r['start'])} | {f2(r['end'])} | {f2(r['best'])} |"
                     f" {r['rounds']}{' + ' + str(r['context_rounds']) if r['context_rounds'] else ''} | {r['passes']} |"
                     f" {r['draws']} | {r['accepted']} | {r['yours_regrown']} | {usd(r['pangram_usd'])} |"
                     f" {usd(r['claude_usd'])} |")
    # the writer arms
    L += ["", "## Writer arms", "",
          "The notes-to-text adapter (notes) against the few-shot writer (fewshot), over every real run-day: draws"
          " and judged attempts per arm, how many were accepted, and the AI share change credited to the arm (a"
          " window's drop over a round whose accepted runs were all that arm's; mixed when two arms were accepted"
          " in one round). Changes banked before 2026-11-09 count as the few-shot writer's.", "",
          "| Arm | Draws | Attempts | Accepted | Acceptance | Draws per accepted | Share change credited | Rounds credited |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a in h.get("arms", []):
        L.append(f"| {a['arm']} | {a['draws']} | {a['attempts']} | {a['accepted']} |"
                 f" {'' if a['acceptance'] is None else format(a['acceptance'], '.0%')} |"
                 f" {a['draws_per_accepted'] or ''} | {a['share_change']:+.2f} | {a['share_rounds']} |")
    if not h.get("arms"):
        L.append("| none yet | | | | | | | |")
    # sections
    L += ["", "## Sections as posted", "",
          "Each section scored as one document the way a reader pastes it. The whole-text round's Pangram spend per"
          " section: " + (", ".join(f"section {n} {usd(v)}" for n, v in sorted(h["section_whole_usd"].items()))
                          or "none yet") + ".", ""]
    for n, rows in h["sections"].items():
        L += [f"**Section {n}:** " + "; ".join(f"{f2(r['ai'])} ({r['what']}, {r['when']})" for r in rows), ""]
    # the guide as it stands (plan 9, 2026-11-10): every section scored whole as a reader would paste the file
    scans = h.get("guide_scans") or []
    L += ["**The guide as it stands** (the whole file, every section, scored as one document): "
          + ("; ".join(f"{f2(x['ai']) if x['ai'] is not None else 'not scored (' + str(x.get('skipped')) + ')'}"
                       f" ({x.get('words') or 0:,} words, {x.get('flagged') if x.get('flagged') is not None else '?'}"
                       f" flagged, {x['time']}, {x['how']})" for x in scans) or "none yet") + ".", ""]
    # run-days
    L += ["## Run-days", "",
          "Sections combined is the final scan's in-scope sections as one number, weighted by each section's words;"
          " the guide as it stands is the whole file scored once at the wrap-up. Both come from the bank's `guide`"
          " entries, so a day before they existed is blank.", "",
          "| Run-day | Hours | Pangram $ | Claude $ (est.) | Sections combined | Guide as it stands |",
          "| --- | --- | --- | --- | --- | --- |"]
    for d in h["run_days"]:
        g = d.get("guide") or {}
        L.append(f"| {d['day']} | {d['hours']} | {usd(d['pangram']['total'])} | {usd(d['claude']['estimated_usd'])} |"
                 f" {f2(g.get('sections_combined'))} | {f2(g.get('ai'))}"
                 f"{' not scored (' + str(g['skipped']) + ')' if g.get('skipped') and g.get('ai') is None else ''} |")
    L.append("")
    for d in h["run_days"]:
        p, c, r = d["pangram"], d["claude"], d["rates"]
        L += [f"### {d['day']} ({d['start']} to {d['end']}, {d['hours']} hours)", "",
              f"- **Pangram:** {usd(p['total'])} of the {usd(p['day_cap'])} day cap"
              + (f" (the whole-text reserve {usd(p['reserve'])})" if p.get("reserve") else "") + ": "
              + ", ".join(f"{k} {usd(v)}" for k, v in sorted(p["by_purpose"].items(), key=lambda kv: -kv[1]))
              + f". Month to date at the end: {usd(p['month_to_date'])} of {usd(p['month_cap'])}.",
              f"- **Claude, estimated:** {usd(c['estimated_usd'])}"
              + (f" (first-run backstop {usd(c['backstop_usd'])})" if c.get("backstop_usd") else "") + ": "
              + ", ".join(f"{k} {usd(v)}" for k, v in sorted(c["by_kind"].items(), key=lambda kv: -kv[1]))
              + f". About {c['points_of_week']:.1%} of your week at the rate in use.",
              f"- **Conversion rate in use:** {r['week']:.4f} points of the week per estimated dollar ({r['week_from']}),"
              f" {r['five_hour']:.4f} of the 5-hour window ({r['five_hour_from']})."]
        for f in r["refits_at_the_end"]:
            L.append(f"  - Refit at the end ({f['kind']}): measured {f['measured']:.4f} ({f['points']:.2f} points over"
                     f" {usd(f['dollars'])}); in use from the next run-day: {f['in_use_next']:.4f}"
                     + (" (held at the clamp)" if f["clamped"] else "") + ".")
        if d["holds"]:
            L.append(f"- **Held by a cap:** {d['held_minutes']} minutes in all.")
            L += [f"  - {x['time'][-5:]} {x['check']}, {x['minutes']} minutes: {x['detail']}" for x in d["holds"]]
        else:
            L.append("- **Held by a cap:** never.")
        L.append("- **How passes ended:** " + (", ".join(f"{k} {v}" for k, v in d["pass_ends"].items()) or "none")
                 + f". Settle rounds after the stuck rule: {d['settle_rounds']}, which replaced {len(d['settle_replaced'])}"
                 " run(s)" + (":" if d["settle_replaced"] else "."))
        L += [f"  - {x['window']} {x['time'][-5:]}: \"{short(x['orig'])}\" became \"{short(x['new'])}\""
              for x in d["settle_replaced"]]
        L.append("")
    # what moved
    m = h["moved"]
    L += ["## What moved and what didn't", ""]
    L.append("**Windows whose share didn't change across a run-day:** " + (", ".join(
        f"{x['window']} on {x['day']} ({f2(x['share'])})" for x in m["unchanged_windows"]) or "none") + ".")
    L += ["", "**Flagged on every scoring, across run-days** (each time the window's text held it):", ""]
    L += [f"- {x['window']} ({x['whose']}, {x['scorings']} scorings): \"{x['text']}\"" for x in m["always_flagged"]] \
        or ["- none"]
    if m["always_flagged_as_posted"]:
        L += ["", "**Flagged in every whole-text scoring of its section:**", ""]
        for n, xs in m["always_flagged_as_posted"].items():
            L += [f"- section {n} ({x['whose']}): \"{x['text']}\"" for x in xs]
    L += ["", "**Checks that throw out the most candidates** (from the tallies the bank holds; before 2026-11-09 only"
          " runs where nothing passed kept one):", ""]
    L += [f"- {k}: {n}" for k, n in list(m["rejects"].items())[:12]] or ["- none"]
    sr = m["stuck_reading"]
    if sr:
        fired = sum(1 for x in sr if x["fires_now"] is not None)
        early = [x for x in sr if x["fires_early"] is not None]
        L += ["", "**The stuck rule** (M1 review finding 13): as configured it can fire after round 5 at the"
              f" earliest. It fired in {fired} of {len(sr)} passes. Read as \"two quiet rounds, checked from round 3\","
              f" it would have fired in {len(early)}, skipping {sum(x['skipped'] for x in early)} rounds in which"
              f" {sum(x['accepted_in_skipped'] for x in early)} runs were replaced, and"
              f" {sum(1 for x in early if x['lower_score_in_skipped'])} pass(es) found their lowest score in a"
              " skipped round."]
    L.append("")
    return "\n".join(L)


def write(out_dir=None, settings=None, bank_path=None, digests_dir=None, job=None):
    settings = settings or Settings()
    out_dir = Path(out_dir or SANDBOX / "analytics")
    conn = open_ro(settings.db_path)
    if job is None:
        from engine.job import Job
        job = Job()                    # the default job, [job] default in config.toml
    entries = load_bank(bank_path or job.bank_path)
    h = build(conn, entries, digests_by_day(digests_dir or SANDBOX / "digests"), settings.config, job=job)
    (out_dir / "loop history.json").write_text(json.dumps(h, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "loop history.md").write_text(render(h), encoding="utf-8")
    return h


def main(argv=None):
    ap = argparse.ArgumentParser(description="the loop's history from the ledger, the bank and the digests")
    ap.add_argument("--job", help="the job, a folder in jobs/ (default: [job] default in config.toml)")
    ap.add_argument("--out", help="folder for loop history.md and .json (default analytics/)")
    a = ap.parse_args(argv)
    from engine.job import Job
    h = write(out_dir=a.out, job=Job(a.job))
    for wid, w in h["windows"].items():
        print(f"{wid}: start {f2(w['start'])} shipped {f2(w['shipped'])} ${w['pangram_usd']:.2f} Pangram,"
              f" {cpp(w['cost_per_point'])} a point, {w['draws_per_accepted']} draws per accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
