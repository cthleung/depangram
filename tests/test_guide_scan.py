"""Tests for the two digest numbers of 2026-11-10 (plan 9): the sections combined, weighted by words, and the guide
as it stands, the whole file scored once at the wrap-up (engine/guide_scan.py). Run from the repo root:

    python -m unittest discover tests
"""
import tempfile
import types
import unittest
from pathlib import Path

from engine import analytics as an
from engine.budget import PangramCapReached
from engine.guide_scan import card_line, combined, digest_lines, run_for_loop, section_list
from engine.job import Bank
from engine.pangram import text_sha
from engine.wrapup import guide_as_posted

GUIDE = ("<!-- plan -->\r\n# A Guide\r\n\r\nAn intro **bold** line.\r\n\r\n## 1. One\r\n\r\n<!-- note -->\r\n"
         "Prose one.\r\n\r\n\"An example.\"\r\n\r\n## 2. Two\r\n\r\n## Further Reading\r\n\r\nA book.\r\n")


def resp(ai, flags):
    return {"fraction_ai": ai, "version": "4.1",
            "windows": [{"text": f"s{i}", "ai_assistance_score": 0.9 if f else 0.1, "is_humanized": False}
                        for i, f in enumerate(flags)]}


def scan_loop(bank, **over):
    """A loop with what run_for_loop reads: the bank, the clock, the switch, the day and a scorer."""
    from test_run_loop import loop_stub
    lp = loop_stub(bank=bank, guide_scan_on=True, rehearsal=False, day="2026-11-10", cap_events=[], windows=[],
                   sections_combined={"ai": 0.19, "sections": [1, 2, 3, 6], "left_out": [], "words": 2855})
    lp.score = lambda text, label, whole=False: resp(0.31, [True, False, True])
    for k, v in over.items():
        setattr(lp, k, v)
    return lp


class Combined(unittest.TestCase):
    def test_the_sections_combine_weighted_by_words(self):
        c = combined({1: 0.30, 2: 0.11, 3: 0.29, 6: 0.00}, {1: 1036, 2: 723, 3: 594, 6: 502})
        self.assertAlmostEqual(c["ai"], (0.30 * 1036 + 0.11 * 723 + 0.29 * 594) / 2855, places=6)   # pass
        self.assertEqual((c["sections"], c["left_out"], c["words"]), ([1, 2, 3, 6], [], 2855))
        self.assertEqual(combined({1: 1.0, 2: 0.0}, {1: 100, 2: 100})["ai"], 0.5)
        self.assertEqual(combined({1: 1.0, 2: 0.0}, {1: 300, 2: 100})["ai"], 0.75)                   # the weights

    def test_a_section_the_cap_left_unscanned_is_left_out_and_named(self):
        c = combined({1: 0.30, 2: None, 6: 0.10}, {1: 100, 2: 500, 6: 100})
        self.assertEqual((round(c["ai"], 2), c["sections"], c["left_out"]), (0.2, [1, 6], [2]))      # flag: named
        self.assertEqual(combined({1: None}, {1: 100})["ai"], None)                                  # nothing to combine
        self.assertEqual(section_list([1, 2, 3, 6]), "1, 2, 3 and 6")
        self.assertEqual(section_list([3]), "3")


class TheText(unittest.TestCase):
    def test_the_whole_guide_as_a_reader_pastes_it(self):
        t = guide_as_posted(GUIDE)
        self.assertNotIn("<!--", t)                                   # comments out
        self.assertNotIn("**", t)                                     # bold marks out
        self.assertNotIn("\r", t)                                     # CRLF and LF are one text to the cache
        self.assertTrue(t.startswith("A Guide\n\nAn intro bold line."))
        self.assertIn('"An example."', t)                             # examples in
        self.assertIn("2. Two\n\nFurther Reading\n\nA book.", t)      # every section, the stubs and the tail too
        self.assertEqual(text_sha(t), text_sha(guide_as_posted(GUIDE.replace("\r\n", "\n"))))


class AtTheWrapUp(unittest.TestCase):
    def test_a_scoring_banks_one_guide_entry_with_the_sha_and_no_text(self):
        with tempfile.TemporaryDirectory() as d:
            lp = scan_loop(Bank(Path(d) / "bank.jsonl"))
            got = run_for_loop(lp, GUIDE)
            self.assertEqual((got["ai"], got["flagged"]), (0.31, 2))
            e = lp.bank.entries[-1]
            self.assertEqual((e["kind"], e["day"], e["ai"], e["flagged"], e["sections_combined"], e["how"]),
                             ("guide", "2026-11-10", 0.31, 2, 0.19, "wrap-up"))
            self.assertEqual((e["words"], e["sha"]), (len(guide_as_posted(GUIDE).split()), text_sha(guide_as_posted(GUIDE))))
            self.assertNotIn("text", e)
            self.assertNotIn("segs", e)
            self.assertIn("guide as it stands: 0.31 (2 flagged", lp.timeline[-1])

    def test_the_day_cap_skips_it_logged_and_banked_with_no_score(self):
        with tempfile.TemporaryDirectory() as d:
            seen = []

            def capped(text, label, whole=False):
                raise PangramCapReached("Pangram daily", "$5.90 spent today, cap $6.00", until=10 ** 12)
            lp = scan_loop(Bank(Path(d) / "bank.jsonl"), score=capped, on_cap=seen.append)
            got = run_for_loop(lp, GUIDE)
            self.assertEqual((got["skipped"], got["ai"]), ("the day cap", None))
            self.assertEqual(seen[0].check, "Pangram daily")                          # the hold is the loop's
            e = lp.bank.entries[-1]
            self.assertEqual((e["kind"], e["ai"], e["skipped"], e["sections_combined"]), ("guide", None, "the day cap", 0.19))
            self.assertIn("The guide as it stands: not scored today: the day cap.", digest_lines(lp)[1])
            self.assertEqual(card_line(lp), "Guide as it stands: not scored (the day cap), sections combined 0.19")

    def test_a_rehearsal_gets_a_placeholder_at_zero_spend(self):
        with tempfile.TemporaryDirectory() as d:
            def paid(text, label, whole=False):
                raise AssertionError("Pangram was called in a rehearsal")
            lp = scan_loop(Bank(Path(d) / "bank.jsonl"), rehearsal=True, score=paid)
            got = run_for_loop(lp, GUIDE)
            self.assertTrue(got["placeholder"])
            self.assertEqual(got["ai"], None)
            self.assertTrue(lp.bank.entries[-1]["placeholder"])
            self.assertIn("a placeholder in a rehearsal", digest_lines(lp)[1])
            self.assertEqual(card_line(lp), "Guide as it stands: placeholder (rehearsal), sections combined 0.19")

    def test_the_switch_off_scores_nothing_and_banks_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            lp = scan_loop(Bank(Path(d) / "bank.jsonl"), guide_scan_on=False)
            self.assertEqual(run_for_loop(lp, GUIDE)["skipped"], "guide_scan is off in config.toml")
            self.assertEqual(lp.bank.entries, [])


def digest_loop(**over):
    lp = types.SimpleNamespace(as_posted_first={1: 0.30, 6: 0.13}, as_posted_now={1: {"ai": 0.30, "flagged": 8},
                               6: {"ai": 0.00, "flagged": 0}}, final_scan={1: 0.30, 6: 0.00}, windows=[],
                               whole_log=[], writeback_report=None,
                               sections_combined={"ai": 0.2023, "sections": [1, 6], "left_out": [], "words": 1538},
                               guide_scan={"ai": 0.31, "flagged": 2, "words": 3071})
    for k, v in over.items():
        setattr(lp, k, v)
    return lp


class TheDigest(unittest.TestCase):
    def test_the_two_lines_sit_under_the_as_posted_table(self):
        from engine.discord import as_posted_lines
        L = as_posted_lines(digest_loop())
        i = next(k for k, ln in enumerate(L) if ln.startswith("| 6 |"))
        self.assertEqual(L[i + 1], "")
        self.assertEqual(L[i + 2], "Sections 1 and 6 combined: 0.20 (weighted by words).")
        self.assertTrue(L[i + 3].startswith("The guide as it stands: 0.31 (2 flagged, 3,071 words)."))
        self.assertFalse(any(ln.startswith("| ") for ln in L[i + 1:]))          # no table row for parse_digest to misread

    def test_the_card_carries_one_line(self):
        from engine.discord import digest_card
        snap = {"pangram_today": 5.6, "pangram_day_cap": 6.0, "pangram_run_day": None, "pangram_month": 60.0,
                "pangram_month_cap": 180.0, "claude_usd_today": 1.0, "backstop": None, "loop_points_today": 0.01,
                "allowance": 0.05, "loop_share": 0.1, "share_cap": 0.5, "real_total": 0.3, "ceiling": 0.9,
                "five_hour": 0.1, "five_hour_cap": 0.9, "five_hour_reset": None, "week_reset": 0, "rate": 0.007}
        lp = digest_loop(day="2026-11-10", stop_reason="the stop", cap_events=[], cuts=[], reflection=None,
                         budget=types.SimpleNamespace(snapshot=lambda: snap))
        card = digest_card(lp)
        self.assertIn("\nGuide as it stands: 0.31, sections combined 0.20\n", card)
        lp.guide_scan, lp.sections_combined = None, None
        self.assertNotIn("Guide as it stands", digest_card(lp))                    # a day without them says nothing

    def test_a_section_left_out_of_the_combination_is_named(self):
        L = digest_lines(digest_loop(sections_combined={"ai": 0.3, "sections": [1], "left_out": [6], "words": 1036}))
        self.assertEqual(L[0], "Sections 1 combined: 0.30 (weighted by words); section 6 not scanned today, so left out.")


class TheHistory(unittest.TestCase):
    def test_the_run_day_reads_the_two_numbers_from_the_bank(self):
        from test_analytics import CFG, ledger, st
        conn = ledger()
        e = [{"kind": "guide", "day": "2026-11-08", "time": st(9, 0, 30), "ai": 0.31, "flagged": 2, "words": 3071,
              "sha": "abc", "sections_combined": 0.19, "how": "wrap-up"}]
        h = an.build(conn, e, {}, CFG)
        g = h["run_days"][0]["guide"]
        self.assertEqual((g["ai"], g["sections_combined"], g["words"]), (0.31, 0.19, 3071))
        md = an.render(h)
        self.assertIn("| Run-day | Hours | Pangram $ | Claude $ (est.) | Sections combined | Guide as it stands |", md)
        self.assertIn("| 2026-11-08 | 6.1 | $0.00 | $0.00 | 0.19 | 0.31 |", md)
        self.assertIn("0.31 (3,071 words, 2 flagged, 2026-11-09 00:30, wrap-up)", md)
        self.assertEqual(an.pangram_kind("run guide as it stands"), "guide scan")

    def test_a_day_without_them_is_blank(self):
        from test_analytics import CFG, ledger
        h = an.build(ledger(), [], {}, CFG)
        self.assertEqual(h["run_days"][0]["guide"]["ai"], None)
        self.assertIn("| 2026-11-08 | 6.1 | $0.00 | $0.00 |  |  |", an.render(h))


class TheReflection(unittest.TestCase):
    def test_the_material_says_what_the_whole_guide_does(self):
        from engine.reflect import guide_scan_from_bank, guide_scan_record, render_material
        rec = {"day": "2026-11-10", "stopped": "x", "model": "m", "arms": "a", "digest": "", "sections": [1],
               "now": {}, "start": {}, "changed": [], "passes": [], "couldnt": "", "previous": None, "decisions": [],
               "guide": [(1, 1036, True)],
               "guide_scan": guide_scan_record({"ai": 0.31, "words": 3071, "flagged": 2}, {"ai": 0.19})}
        m = render_material(rec)
        self.assertIn("Scored whole as it stands today, every section as a reader would paste the file: 0.31 AI"
                      " share over 3,071 words (2 segments flagged). The sections in scope combined, weighted by"
                      " words: 0.19.", m)
        rec["guide_scan"] = guide_scan_record({"skipped": "the day cap"}, {"ai": 0.19})
        self.assertIn("wasn't scored today (the day cap). The sections in scope combined", render_material(rec))
        self.assertIsNone(guide_scan_record(None, None))
        bank = [{"kind": "guide", "day": "2026-11-10", "ai": 0.4, "words": 10, "sections_combined": 0.2}]
        self.assertEqual(guide_scan_from_bank(bank, "2026-11-10")["sections_combined"], 0.2)
        self.assertIsNone(guide_scan_from_bank(bank, "2026-11-09"))


if __name__ == "__main__":
    unittest.main()
