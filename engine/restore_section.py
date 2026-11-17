"""Put a section back to its best known as-posted text, by hand (plan 5.6; 2026-11-10 evening).

    python -m engine.restore_section --section 3          # bank the restore and rewrite the working copy
    python -m engine.restore_section --section 3 --dry    # show what it would do, with the diff per window

The loop's whole-text round does this on its own now (Loop.restore_section_bests). This command is for a section
that shipped below its best known before that rule existed: section 3 ended 2026-11-10 at 0.29 as posted with a
0.00 version banked from the day before. For every window of the section whose text at the end of the last
whole-text round (what shipped) differs from the best known picks, it banks a "ship" entry holding the best known
run lists (stamped with the window's best score, so a tied window best can't override it), then writes the working
copy from what every window ships now. Nothing here calls Pangram: the best known score is already banked. The
real guide is not touched (python -m engine.wrapup --write does that by hand).
"""
import argparse
import difflib
import json
import sys
import time

from .job import Bank, Job, flat, window_text_of
from .section_best import runs_for, section_best, section_standing
from .wrapup import layout


def plan(bank, lay, n, noise=0.05):
    """What a restore of section n would change: the best known round, the standing one, and one row per window
    (what shipped at the end of the last whole-text round, the best known text, its run lists, and whether they
    differ). Returns {"error": ...} when the section has no whole-text scoring in the bank."""
    sb = section_best(bank, n, noise)
    if not sb:
        return {"error": f"section {n} has no whole-text scoring in the bank"}
    st = section_standing(bank, n)
    rows = []
    for wid, w in lay.items():
        if w["section"] != n:
            continue
        orig = window_text_of(w["original"], w["heading"])
        now = (st or sb)["picks"].get(wid, orig)
        best = sb["picks"].get(wid)
        runs = runs_for(bank, sb, wid, original=(orig, w["original"])) if best is not None else None
        rows.append({"wid": wid, "now": now, "best": best, "runs": runs,
                     "changed": best is not None and now != best,
                     "fits": runs is not None and [len(p) for p in runs] == [len(p) for p in w["runs"]]})
    return {"section": n, "best": sb, "standing": st, "rows": rows}


def entries(p, bank, stamp, day=None):
    """The bank entries a restore writes: one "ship" entry per window whose text changes, holding the best known
    run lists. Nothing for a window already at its best known text, or one whose best known runs no longer fit."""
    out = []
    for r in p["rows"]:
        if not r["changed"] or not r["fits"]:
            continue
        best = bank.best(r["wid"])
        out.append({"kind": "ship", "window": r["wid"], "day": day or stamp[:10], "time": stamp, "arm": "hand",
                    "arms": [], "text": r["best"], "runs": json.loads(json.dumps(r["runs"])), "ai": None,
                    "section": p["section"], "section_ai": p["best"]["ai"],
                    "window_ai": best["ai"] if best else None,
                    "why": f"section best restored by hand (scored {p['best']['ai']:.2f} on {p['best']['time']})"})
    return out


def diff_text(now, best, wid):
    """A unified diff of a window's text, paragraph by paragraph, from what shipped to the best known."""
    return "\n".join(difflib.unified_diff(now.split("\n"), best.split("\n"), f"{wid} as it shipped",
                                          f"{wid} best known", lineterm="", n=0))


def report(p):
    L = [f"section {p['section']}: best known {p['best']['ai']:.2f} as posted on {p['best']['time']}"
         + (f", standing {p['standing']['ai']:.2f} on {p['standing']['time']}" if p.get("standing") else "")]
    for r in p["rows"]:
        if not r["changed"]:
            L.append(f"{r['wid']}: already at the best known text")
        elif not r["fits"]:
            L.append(f"{r['wid']}: the best known runs no longer fit the window's shape; left alone")
        else:
            L.append(f"{r['wid']}: restored")
            L.append(diff_text(r["now"], r["best"], r["wid"]))
    return "\n".join(L)


def ship_map(bank, lay, noise=0.05, repeat_must=False, section=None):
    """{wid: (original paragraphs, shipped paragraphs)} for the working copy, through the loop's own ship rule
    (window rule plus section rule) on a Loop built without a server or a paid door. `section` keeps it to one
    section's windows (the command rewrites only the section it restored; the loop's next stop does the rest)."""
    from .run import Loop, WindowState
    lp = Loop.__new__(Loop)
    lp.bank, lp.noise, lp.repeat_must, lp.windows = bank, noise, repeat_must, []
    for wid, w in lay.items():
        ws = WindowState({"wid": wid, "text": "", "section": w["section"], "index": 0})
        ws.runs, ws.original, ws.heading, ws.must = w["runs"], w["original"], w["heading"], {}
        lp.load_calls(ws)                  # a cut you put back on Discord (plan 9.1) never ships from here either
        lp.windows.append(ws)
    return {ws.wid: ([flat(p) for p in ws.original], [flat(p) for p in Loop.pick_runs(lp, ws)])
            for ws in lp.windows if section is None or ws.w["section"] == section}


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="put a section back to its best known as-posted text (plan 5.6)")
    ap.add_argument("--job", help="the job, a folder in jobs/ (default: [job] default in config.toml)")
    ap.add_argument("--section", type=int, required=True)
    ap.add_argument("--dry", action="store_true", help="show the plan and the diffs; bank nothing, write nothing")
    a = ap.parse_args(argv)
    from .settings import Settings
    noise = float(Settings().config.get("fixpass", {}).get("noise", 0.05))
    job = Job(a.job)
    bank = Bank(job.bank_path)
    lay = layout(job, bank)
    p = plan(bank, lay, a.section, noise)
    if "error" in p:
        print(p["error"])
        return 1
    print(report(p))
    new = entries(p, bank, time.strftime("%Y-%m-%d %H:%M"))
    if not new:
        print("nothing to restore")
        return 0
    if a.dry:
        print(f"dry run: {len(new)} entry(ies) would be banked and the working copy rewritten")
        return 0
    for e in new:
        bank.add(e)
    job.assemble(ship_map(bank, lay, noise, section=a.section), keep_others=True)
    print(f"banked {len(new)} entry(ies) in {bank.path.name}; section {a.section}'s windows written into"
          f" {job.working.name} (the other windows keep what it had)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
