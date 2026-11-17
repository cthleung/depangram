"""Tests for the section-best rule and the reserve release (plan 5.6, 2026-11-10 evening): a section never ships
below its best known as-posted text, window ties don't override a whole-text pick, and the whole-text reserve goes
back to the window rounds once the round is done. The fixture is the s3w0 case of 2026-11-10, trimmed to the two
paragraphs that changed, with the real scores. Run from the repo root:

    python -m unittest discover tests
"""
import tempfile
import types
import unittest
from pathlib import Path

from engine.budget import WINDOW_SHARE, PangramCapReached
from engine.job import Bank, window_text_of
from engine.run import Loop, WindowState
from engine.section_best import runs_for, section_best, section_standing, whole_rounds

HEAD = "3. The Manual"
OLD_P3 = "The manual has a diagram for every button and not one for the lid."
OLD_P4 = "Still, the lid is the only part anybody struggles with."
NEW_P3 = "Every button gets a diagram in the manual, while the lid gets none at all."
NEW_P4 = "And the lid is exactly where people get stuck."
CLAUDE_P3 = "Claude's parallel-run paragraph."
CLAUDE_P4 = "Claude's repeat-for-emphasis paragraph."
WIN_P3 = "An earlier regrowth of the parallel-run paragraph."
S3W1 = "The second window's paragraph, unchanged all along."

R_SHIP = [[[OLD_P3]], [[OLD_P4]]]        # the 2026-11-09 06:36 whole-text pick (section 3 at 0.00)
R_WIN = [[[WIN_P3]], [[OLD_P4]]]         # the window's best when that pick was banked
R_NEW = [[[NEW_P3]], [[NEW_P4]]]         # the settle round's text: a window best, and section 3 at 0.29
R_1 = [[[S3W1]]]
BEST_AI, WORSE_AI = 0.0005931375781074166, 0.2896
TIE_OLD, TIE_NEW = 3.419491634293337e-10, 1.0019740592781545e-10
SEGS_BEST = [[OLD_P3, False], [OLD_P4, False], [S3W1, False]]
SEGS_WORSE = [[NEW_P3, True], [NEW_P4, True], [S3W1, False]]


def text(runs, head=HEAD):
    return window_text_of(runs, head)


def window(wid, runs, head, section=3):
    ws = WindowState({"wid": wid, "text": "", "section": section, "index": 0})
    ws.runs = [[{"sents": r[:], "claude": True} for r in p] for p in runs]
    ws.original = [[r[:] for r in p] for p in runs]
    ws.current = [[r[:] for r in p] for p in runs]
    ws.heading, ws.must = head, {(pi, ri): {"hedges": [], "repeats": []} for pi, p in enumerate(runs) for ri in range(len(p))}
    ws.scored_runs, ws.forced = None, set()
    return ws


def s3_bank(d, win_ai=TIE_OLD, new_ai=TIE_NEW, second_round=True):
    """The s3w0 case: a whole-text pick that shipped section 3 at 0.00, then a window best that overrode it and a
    second whole-text round that found the section at 0.29. Legacy entries (no runs on the whole_end), as the real
    bank has them."""
    b = Bank(Path(d) / "bank.jsonl")
    b.add({"kind": "start", "window": "s3w0", "time": "2026-11-09 02:10", "ai": 0.73, "segs": [],
           "text": text([[[CLAUDE_P3]], [[CLAUDE_P4]]])})
    b.add({"kind": "start", "window": "s3w1", "time": "2026-11-09 02:11", "ai": 0.0, "segs": [], "text": text(R_1, "")})
    b.add({"kind": "round", "window": "s3w0", "time": "2026-11-09 04:00", "ai": win_ai, "text": text(R_WIN), "runs": R_WIN})
    b.add({"kind": "context", "window": "s3w0", "time": "2026-11-09 05:59", "ai": None, "text": text(R_SHIP), "runs": R_SHIP})
    b.add({"kind": "whole", "day": "2026-11-09", "time": "2026-11-09 06:05", "k": 2,
           "sections": {"3": {"ai": BEST_AI, "flagged": 0, "segs": SEGS_BEST}}, "improved": [3]})
    b.add({"kind": "ship", "window": "s3w0", "day": "2026-11-09", "time": "2026-11-09 06:36", "text": text(R_SHIP),
           "runs": R_SHIP, "ai": None, "section": 3, "section_ai": BEST_AI})
    b.add({"kind": "whole_end", "day": "2026-11-09", "time": "2026-11-09 06:36",
           "picks": {"s3w0": text(R_SHIP), "s3w1": text(R_1, "")}})
    b.add({"kind": "round", "window": "s3w0", "time": "2026-11-10 07:22", "ai": new_ai, "text": text(R_NEW), "runs": R_NEW})
    if second_round:
        b.add({"kind": "whole", "day": "2026-11-10", "time": "2026-11-10 09:25", "k": 1,
               "sections": {"3": {"ai": WORSE_AI, "flagged": 6, "segs": SEGS_WORSE}}, "improved": []})
        b.add({"kind": "whole_end", "day": "2026-11-10", "time": "2026-11-10 10:05",
               "picks": {"s3w0": text(R_NEW), "s3w1": text(R_1, "")}})
    return b


def loop(b):
    lp = Loop.__new__(Loop)
    lp.bank, lp.noise, lp.repeat_must, lp.timeline = b, 0.05, False, []
    lp.day, lp.log = "2026-11-11", lambda m: lp.timeline.append(m)
    lp.clock = types.SimpleNamespace(stamp=lambda *a: "2026-11-11 09:25", now=lambda: 0.0)
    lp.arm_order = lambda is_last=False: ("notes",)
    lp.windows = [window("s3w0", [[[CLAUDE_P3]], [[CLAUDE_P4]]], HEAD), window("s3w1", R_1, "")]
    return lp


class Memory(unittest.TestCase):
    def test_the_bank_remembers_each_sections_best_as_posted_picks(self):
        with tempfile.TemporaryDirectory() as d:
            b = s3_bank(d)
            rounds = whole_rounds(b)
            self.assertEqual([round(r["sections"]["3"]["ai"], 2) for r in rounds], [0.0, 0.29])
            best, standing = section_best(b, 3, 0.05), section_standing(b, 3)
            self.assertEqual((best["time"], standing["time"]), ("2026-11-09 06:36", "2026-11-10 10:05"))
            self.assertEqual(best["picks"]["s3w0"], text(R_SHIP))
            self.assertEqual(runs_for(b, best, "s3w0"), R_SHIP)              # legacy: found by its text
            self.assertEqual(runs_for(b, best, "s3w1", original=(text(R_1, ""), R_1)), R_1)   # the original
            self.assertIsNone(section_best(b, 6, 0.05))
            # a later round within the noise of the lowest is the best (the more current text)
            b.add({"kind": "whole", "day": "2026-11-11", "time": "2026-11-11 09:25", "k": 1,
                   "sections": {"3": {"ai": 0.03, "flagged": 1, "segs": []}}, "improved": []})
            b.add({"kind": "whole_end", "day": "2026-11-11", "time": "2026-11-11 09:30",
                   "picks": {"s3w0": text(R_NEW), "s3w1": text(R_1, "")},
                   "sections": {"3": {"ai": 0.03, "flagged": 1, "segs": []}}, "runs": {"s3w0": R_NEW, "s3w1": R_1}})
            best = section_best(b, 3, 0.05)
            self.assertEqual(best["time"], "2026-11-11 09:30")
            self.assertEqual(runs_for(b, best, "s3w0"), R_NEW)                # the new whole_end carries its runs
            self.assertEqual(section_best(b, 3, 0.0)["time"], "2026-11-09 06:36")


class ShipRule(unittest.TestCase):
    def test_a_window_tie_never_overrides_the_whole_text_pick(self):
        with tempfile.TemporaryDirectory() as d:
            b = s3_bank(d, second_round=False)
            lp = loop(b)
            ws = lp.windows[0]
            # the pick's text was never scored as a window, so it stands on the window's best before it (3.4e-10);
            # 1.0e-10 is a tie, and the pick keeps shipping (flag: the override that cost section 3 its 0.00)
            self.assertEqual(Loop.window_ship(lp, ws)["runs"], R_SHIP)
            self.assertEqual(Loop.ship_entry(lp, ws)["runs"], R_SHIP)
            # pass: a window best past the noise floor takes over, and the next whole-text round checks it
            b.add({"kind": "round", "window": "s3w0", "time": "2026-11-10 08:00", "ai": 0.0, "text": "x", "runs": R_NEW})
            b.entries[-1]["ai"] = 0.0
            b.entries[2]["ai"] = 0.10                                          # what the pick stood on
            self.assertEqual(Loop.window_ship(lp, ws)["runs"], R_NEW)

    def test_the_section_rule_ships_the_best_known_picks_when_the_standing_text_scored_worse(self):
        with tempfile.TemporaryDirectory() as d:
            b = s3_bank(d, win_ai=0.10, new_ai=0.0)                            # the window rule says the new text
            lp = loop(b)
            w0, w1 = lp.windows
            self.assertEqual(Loop.window_ship(lp, w0)["runs"], R_NEW)
            e = Loop.ship_entry(lp, w0)                                        # flag: section 3 scored 0.29 on it
            self.assertEqual((e["kind"], e["runs"], round(e["section_ai"], 2)), ("section_best", R_SHIP, 0.0))
            self.assertEqual(Loop.pick_runs(lp, w0), R_SHIP)
            self.assertEqual(Loop.pick_runs(lp, w1), R_1)                      # pass: the other window is unchanged
            # pass: within the noise floor, the standing text ships
            b.entries[-2]["sections"]["3"]["ai"] = 0.04
            b._section_rounds = None
            self.assertEqual(Loop.pick_runs(lp, w0), R_NEW)
            b.entries[-2]["sections"]["3"]["ai"] = WORSE_AI
            b._section_rounds = None
            self.assertEqual(Loop.pick_runs(lp, w0), R_SHIP)
            # pass: a window best after the standing scoring hasn't been scored as a section, so it ships, and the
            # next whole-text round scores it as it stands
            b.add({"kind": "round", "window": "s3w0", "time": "2026-11-10 11:00", "ai": 0.0, "text": "newer",
                   "runs": [[["Newer still."]], [[NEW_P4]]]})
            self.assertEqual(Loop.pick_runs(lp, w0), [[["Newer still."]], [[NEW_P4]]])

    def test_the_whole_text_round_restores_a_section_that_scores_worse_than_its_best_known(self):
        with tempfile.TemporaryDirectory() as d:
            b = s3_bank(d, win_ai=0.10, new_ai=0.0, second_round=False)
            lp = loop(b)
            w0, w1 = lp.windows
            lp.job = types.SimpleNamespace(placeholder_base=lambda: ("", []),
                                           assemble=lambda m, write=False: "## " + HEAD + "\n\n"
                                           + "\n\n".join(p for _, cur in m.values() for p in cur) + "\n")
            picks = {"s3w0": Loop.pick_runs(lp, w0), "s3w1": Loop.pick_runs(lp, w1)}
            self.assertEqual(picks["s3w0"], R_NEW)
            scores = {3: {"ai": WORSE_AI, "segs": SEGS_WORSE, "flagged": 6, "humanized": False, "text": "t"}}
            got = Loop.restore_section_bests(lp, {3: [w0, w1]}, picks, scores)    # flag: 0.29 against 0.00 known
            self.assertEqual(got, [3])
            self.assertEqual(picks["s3w0"], R_SHIP)
            self.assertEqual(picks["s3w1"], R_1)
            self.assertEqual((round(scores[3]["ai"], 2), scores[3]["flagged"], scores[3]["segs"]), (0.0, 0, SEGS_BEST))
            self.assertIn(OLD_P4, scores[3]["text"])
            e = b.entries[-1]
            self.assertEqual((e["kind"], e["window"], e["runs"], e["section"]), ("ship", "s3w0", R_SHIP, 3))
            self.assertTrue(e["why"].startswith("section best restored"))
            self.assertEqual(e["window_ai"], 0.0)                              # what the window had: a tie can't undo it
            self.assertEqual(sum(1 for x in b.entries if x.get("kind") == "ship"), 2)   # s3w1 needed nothing
            self.assertIn("section 3: 0.29 as it stands, 0.00 best known, restored", lp.timeline)
            self.assertEqual(Loop.pick_runs(lp, w0), R_SHIP)                   # and that's what ships now
            # pass: inside the noise floor nothing moves
            picks2 = {"s3w0": R_NEW, "s3w1": R_1}
            scores2 = {3: {"ai": 0.03, "segs": SEGS_WORSE, "flagged": 1, "humanized": False, "text": "t"}}
            self.assertEqual(Loop.restore_section_bests(lp, {3: [w0, w1]}, picks2, scores2), [])
            self.assertEqual(picks2["s3w0"], R_NEW)
            # pass: already at the best known picks, whatever the score says
            picks3 = {"s3w0": R_SHIP, "s3w1": R_1}
            self.assertEqual(Loop.restore_section_bests(lp, {3: [w0, w1]}, picks3, dict(scores)), [])


class ByHand(unittest.TestCase):
    def test_the_restore_command_banks_the_best_known_picks_and_shows_the_diff(self):
        from engine.restore_section import entries, plan, report, ship_map
        with tempfile.TemporaryDirectory() as d:
            b = s3_bank(d)
            lay = {"s3w0": {"section": 3, "heading": HEAD, "runs": [[{"sents": [CLAUDE_P3], "claude": True}],
                                                                    [{"sents": [CLAUDE_P4], "claude": True}]],
                            "original": [[[CLAUDE_P3]], [[CLAUDE_P4]]]},
                   "s3w1": {"section": 3, "heading": "", "runs": [[{"sents": [S3W1], "claude": False}]], "original": R_1}}
            p = plan(b, lay, 3, 0.05)
            self.assertEqual([(r["wid"], r["changed"]) for r in p["rows"]], [("s3w0", True), ("s3w1", False)])
            rep = report(p)
            self.assertIn("-" + NEW_P4, rep)
            self.assertIn("+" + OLD_P4, rep)
            self.assertIn("s3w1: already at the best known text", rep)
            new = entries(p, b, "2026-11-10 20:00")
            self.assertEqual(len(new), 1)
            self.assertEqual((new[0]["window"], new[0]["runs"], new[0]["window_ai"]), ("s3w0", R_SHIP, TIE_NEW))
            for e in new:
                b.add(e)
            self.assertEqual(ship_map(b, lay)["s3w0"][1], [OLD_P3, OLD_P4])
            self.assertEqual(ship_map(b, lay)["s3w1"][1], [S3W1])
            self.assertIn("error", plan(b, lay, 6))


class Reserve(unittest.TestCase):
    def test_the_reserve_goes_back_to_the_windows_once_the_whole_text_round_is_done(self):
        from test_run_loop import loop_stub
        spent = {"today": 3.85, "run_day": 3.85}
        lp = loop_stub(reserve=3.0, floor=0.5,
                       budget=types.SimpleNamespace(pangram_today=lambda: spent["today"],
                                                    pangram_run_day=lambda: spent["run_day"]),
                       pangram_block=PangramCapReached(WINDOW_SHARE, "spent", until=10 ** 12))
        lp.log = lambda m: lp.timeline.append(m)
        Loop.release_reserve(lp)
        self.assertEqual(lp.reserve, 0.5)                                      # down to the floor
        self.assertIsNone(lp.pangram_block)                                    # the window-share hold lifts
        self.assertIn("whole-text round done; $1.65 of the reserve released to the window rounds ($0.50 kept for"
                      " a day-end round)", lp.timeline)
        Loop.release_reserve(lp)                                               # once a day
        self.assertEqual(len(lp.timeline), 1)
        # flag: a round that spent the reserve itself has nothing to release, and the hold stays
        lp2 = loop_stub(reserve=3.0, floor=0.5,
                        budget=types.SimpleNamespace(pangram_today=lambda: 5.90, pangram_run_day=lambda: None),
                        pangram_block=PangramCapReached(WINDOW_SHARE, "spent", until=10 ** 12))
        lp2.log = lambda m: lp2.timeline.append(m)
        Loop.release_reserve(lp2)
        self.assertEqual(lp2.reserve, 0.5)
        self.assertIsNotNone(lp2.pangram_block)
        self.assertIn("nothing to release", lp2.timeline[-1])

    def test_the_floor_keeps_one_section_scoring_for_a_day_end_round(self):
        from test_run_loop import make_budget, pangram_row, ts
        b, conn, _ = make_budget(ts(11))
        pangram_row(conn, ts(9, 30), 3.85)                                     # the window share and the round
        with self.assertRaises(PangramCapReached) as cm:
            b.pangram_check(0.20, reserve=3.0)                                 # before the release: held
        self.assertEqual(cm.exception.check, WINDOW_SHARE)
        b.pangram_check(1.60, reserve=0.5)                                     # after: $5.45 of the $5.50 share
        with self.assertRaises(PangramCapReached) as cm:
            b.pangram_check(1.70, reserve=0.5)                                 # the floor holds the last $0.50
        self.assertEqual(cm.exception.check, WINDOW_SHARE)
        b.pangram_check(1.70)                                                  # which the day-end round may spend


if __name__ == "__main__":
    unittest.main()
