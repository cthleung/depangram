"""Tests for the write-back with the master kept (engine/writeback.py), the sync step's side of it (the markers come
off, the ledger's paragraphs are the machine's) and the guide's trip between main and this branch
(engine/main_copy.py). Each check has a flag case and a pass case. Run from the repo root:

    python -m unittest discover tests
"""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine.job import Bank, Job
from engine.wrapup import LEDGER, as_posted, norm
from engine.writeback import (HAND, frozen_blocks, marker, paragraphs, report_lines, run, safe, section_notes,
                              set_aside, strip_markers, write_marked)
from test_m3 import ORIG, REGROWN, make_job, stub_scan

NL = "\r\n"
GUIDE = NL.join(["# Title", "", "## 1. First Section", "", "<!-- research: A1 -->", "", "Claude said a **thing**. Mine stays.",
                 "", "A lead-in the loop never writes:", "", '"An example line."', "", "A paragraph you wrote later by hand.",
                 "", "## 2. Second", "", "Untouched text.", ""])
ORIG_P = "Claude said a thing. Mine stays."
PASS_P = "The pass wrote this thing. Mine stays."
FROZEN = {"A lead-in the loop never writes:", ORIG_P}


def para(new=PASS_P, known=(), **kw):
    return dict({"label": "s1w0 p1", "section": 1, "versions": [{"Claude said a thing.", "Mistral said it."}, {"Mine stays."}],
                 "known": set(known), "orig": ORIG_P, "orig_block": "Claude said a **thing**. Mine stays.", "new": new,
                 "source": "the paragraph pass (s1 round 2)", "flagged": False}, **kw)


def write(text, p, stamp="2026-11-14 21:00", **kw):
    return write_marked(text, [1], [p], stamp, frozen_keys=FROZEN, sync_date="2026-11-12", **kw)


FIXED = "The reviewer fixed the thing. Mine stays."
FY = {"went_wrong": "the pass said the study ran in 2021 --> it ran in 2023", "about": "the study's year",
      "point": "", "fact": "the study ran in 2023,\n\nnot 2021"}


def for_you(new="The reviewer fixed the **thing**. Mine stays.", prompt=FY, **kw):
    return para(new=new, for_you=prompt, source=HAND, **kw)


class Marked(unittest.TestCase):
    def test_a_paragraph_goes_under_its_commented_original(self):
        new, rep = write(GUIDE, para(flagged=True), notes={1: "as posted, this is the text Pangram scored 0.33"})
        mark = ("<!-- depangram s1w0 p1 · written 2026-11-14 21:00 · the paragraph pass (s1 round 2) · Pangram flagged"
                " a span the pass wrote here" + NL + "the original (frozen at the sync of 2026-11-12):" + NL
                + "Claude said a **thing**. Mine stays." + NL + "-->")
        self.assertIn(NL * 2 + mark + NL * 2 + "The pass wrote this **thing**. Mine stays." + NL * 2, new)   # bold back
        self.assertIn("## 1. First Section" + NL * 2 + "<!-- depangram section 1 · written back 2026-11-14 21:00 · 1"
                      " paragraph(s) here sit under their commented originals · as posted, this is the text Pangram"
                      " scored 0.33 -->" + NL * 2 + "<!-- research: A1 -->", new)
        self.assertEqual(rep["replaced"], ["s1w0 p1"])
        self.assertEqual([(r["orig"], r["text"]) for r in rep["ledger"]], [(ORIG_P, PASS_P)])
        self.assertEqual([t for _, t, _ in rep["yours"]], ["A paragraph you wrote later by hand."])   # flag: not the job's
        self.assertNotIn("\n", new.replace(NL, ""))                                         # the line endings held
        self.assertNotIn("Claude said", as_posted(new, [1])[1])                              # the original isn't posted
        self.assertEqual(strip_markers(new).replace("The pass wrote this **thing**", "Claude said a **thing**"), GUIDE)

    def test_a_rerun_with_nothing_new_changes_no_byte(self):
        new, _ = write(GUIDE, para())
        again, rep = write(new, para(known={PASS_P}), stamp="2026-11-15 09:00")
        self.assertEqual(again, new)                                                        # pass
        self.assertEqual((rep["same"], rep["replaced"], rep["ledger"]), (["s1w0 p1"], [], []))
        _, rep2 = write(new, para())                              # flag: without the ledger the pass's text is nobody's
        self.assertEqual([(t, under) for _, t, under in rep2["yours"]][0], (PASS_P, True))

    def test_newer_text_goes_under_the_same_marker(self):
        new, _ = write(GUIDE, para())
        newer, rep = write(new, para(new="A later round wrote it. Mine stays.", known={PASS_P}), stamp="2026-11-15 09:00")
        self.assertEqual(newer.count("<!-- depangram s1w0 p1"), 1)
        self.assertIn("written 2026-11-15 09:00", newer)
        self.assertIn("Claude said a **thing**. Mine stays." + NL + "-->" + NL * 2 + "A later round wrote it. Mine stays.", newer)
        self.assertNotIn(PASS_P, strip_markers(newer))
        self.assertIn("section 1 · written back 2026-11-15 09:00", newer)
        self.assertEqual(rep["replaced"], ["s1w0 p1"])

    def test_a_paragraph_edited_by_hand_under_its_marker_stays(self):
        new, _ = write(GUIDE, para())
        edited = new.replace("The pass wrote this **thing**. Mine stays.", "I rewrote this one myself.")
        again, rep = write(edited, para(new="A later round wrote it. Mine stays.", known={PASS_P}), stamp="2026-11-15 09:00")
        self.assertEqual(again, edited)                                                     # pass: nothing moved
        self.assertIn((1, "I rewrote this one myself.", True), rep["yours"])
        self.assertEqual(rep["missing"], ["s1w0 p1"])

    def test_text_the_guide_already_holds_gets_its_original_and_keeps_its_words(self):
        held = GUIDE.replace("Claude said a **thing**.", "Mistral said it.")                 # an old unmarked write-back
        new, rep = write(held, para(new="Mistral said it. Mine stays."))
        self.assertEqual((rep["marked"], rep["replaced"], rep["ledger"]), (["s1w0 p1"], [], [{
            "when": "2026-11-14 21:00", "label": "s1w0 p1", "orig": ORIG_P, "text": "Mistral said it. Mine stays.",
            "source": "the paragraph pass (s1 round 2)"}]))
        self.assertEqual(strip_markers(new), held)
        self.assertIn("Claude said a **thing**. Mine stays." + NL + "-->" + NL * 2 + "Mistral said it. Mine stays.", new)

    def test_a_paragraph_back_at_its_original_loses_its_marker(self):
        new, _ = write(GUIDE, para())
        back, rep = write(new, para(new=ORIG_P, known={PASS_P}))
        self.assertEqual(back, GUIDE)                                                       # the note under the heading too
        self.assertEqual(rep["restored"], ["s1w0 p1"])
        same, rep2 = write(GUIDE, para(new=ORIG_P))
        self.assertEqual((same, rep2["same"]), (GUIDE, ["s1w0 p1"]))

    def test_an_original_cant_close_its_own_comment(self):
        self.assertEqual(safe("a --> b <!-- note --> c <!-- open"), "a --&gt; b  c &lt;!-- open")       # flag
        self.assertEqual(safe("plain **bold** text"), "plain **bold** text")                              # pass
        m = marker(para(orig_block="It ends --> here."), "2026-11-14 21:00")
        self.assertEqual(m.count("-->"), 1)
        self.assertEqual(strip_markers("Before.\n\n" + m + "\n\nAfter."), "Before.\n\nAfter.")

    def test_a_score_is_only_claimed_for_the_exact_text(self):
        class Cache:
            def cached(self, text):
                return {"fraction_ai": 0.33, "windows": []} if text == "1. First Section\n\nScored text." else None
        passed = {"sections": {1: {"ai": 0.33, "label": "s1 round 2", "text": "1. First Section\n\nScored text."}}}
        ok = section_notes("## 1. First Section\n\n<!-- x -->\n\nScored text.\n", passed, Cache())
        self.assertEqual(ok[1], "as posted, this is the text Pangram scored 0.33 (s1 round 2)")           # pass
        off = section_notes("## 1. First Section\n\nScored text, edited.\n", passed, Cache())
        self.assertIn("isn't exactly that text", off[1])                                                  # flag


class ForYou(unittest.TestCase):
    """A paragraph the reviewer hand-corrected for a factual slip: in bold, with a prompt for the author."""

    def test_a_for_you_paragraph_goes_in_bold_under_its_prompt(self):
        new, rep = write(GUIDE, for_you())
        mark = NL.join(["<!-- depangram s1w0 p1 · written 2026-11-14 21:00 · hand-corrected by the reviewer · FOR YOU"
                        " TO REWRITE", "what went wrong: the pass said the study ran in 2021 --&gt; it ran in 2023",
                        "write about: the study's year", "the fact: the study ran in 2023, not 2021",
                        "the original (frozen at the sync of 2026-11-12):", "Claude said a **thing**. Mine stays.", "-->"])
        self.assertIn(NL * 2 + mark + NL * 2 + "**" + FIXED + "**" + NL * 2, new)          # bold inside it came out
        self.assertNotIn("the point:", new)                                                 # an empty line is left out
        self.assertEqual(strip_markers(new).count("-->"), GUIDE.count("-->"))              # flag: nothing closed it early
        self.assertIn("1 paragraph(s) here sit under their commented originals · 1 in bold for you to rewrite by hand",
                      new)
        self.assertEqual((rep["for_you"], rep["prompts"], rep["replaced"]), (["s1w0 p1"], {"s1w0 p1": FY}, ["s1w0 p1"]))
        self.assertEqual(rep["ledger"][0]["for_you"], FY)
        self.assertEqual(as_posted(new, [1])[1].count(FIXED), 1)
        plain, rep2 = write(GUIDE, para())                                                  # pass: no for-you, no bold
        self.assertNotIn("FOR YOU", plain)
        self.assertNotIn("in bold for you", plain)
        self.assertEqual((rep2["for_you"], list(rep2["ledger"][0])), ([], ["when", "label", "orig", "text", "source"]))

    def test_a_rerun_changes_no_byte_and_a_new_prompt_moves_the_time(self):
        new, _ = write(GUIDE, for_you())
        again, rep = write(new, for_you(known={FIXED}), stamp="2026-11-15 09:00")
        self.assertEqual(again, new)                                                        # pass: same text, same prompt
        self.assertEqual((rep["same"], rep["for_you"]), (["s1w0 p1"], ["s1w0 p1"]))
        moved, rep2 = write(new, for_you(known={FIXED}, prompt=dict(FY, about="the year the study ran")),
                            stamp="2026-11-15 09:00")
        self.assertNotEqual(moved, new)                                                     # flag: the prompt changed
        self.assertIn("written 2026-11-15 09:00 · hand-corrected by the reviewer · FOR YOU TO REWRITE", moved)
        self.assertIn("write about: the year the study ran", moved)
        self.assertEqual(moved.count("<!-- depangram s1w0 p1"), 1)
        self.assertEqual((rep2["marked"], rep2["ledger"]), (["s1w0 p1"], []))
        self.assertIn("**" + FIXED + "**", moved)

    def test_a_paragraph_that_stops_being_for_you_goes_back_to_plain_text(self):
        new, _ = write(GUIDE, for_you())
        later, rep = write(new, para(new="A later round wrote this thing. Mine stays.", known={FIXED}),
                           stamp="2026-11-15 09:00")
        self.assertIn("written 2026-11-15 09:00 · the paragraph pass (s1 round 2)" + NL + "the original", later)
        self.assertIn("-->" + NL * 2 + "A later round wrote this **thing**. Mine stays." + NL, later)   # the original's bold
        self.assertNotIn("FOR YOU", later)
        self.assertNotIn("in bold for you", later)
        self.assertEqual((rep["replaced"], rep["for_you"]), (["s1w0 p1"], []))
        same_words, rep2 = write(new, para(new=FIXED, known={FIXED}), stamp="2026-11-15 09:00")
        self.assertIn("-->" + NL * 2 + FIXED.replace("thing", "**thing**") + NL, same_words)    # flag: the bold came off
        self.assertNotIn("FOR YOU", same_words)
        self.assertEqual(rep2["marked"], ["s1w0 p1"])
        kept, rep3 = write(new, para(new=None, known={FIXED}), stamp="2026-11-15 09:00")      # pass: no newer text at all
        self.assertEqual((kept, rep3["same"]), (new, ["s1w0 p1"]))                          # (a kept row after a sync)

    def test_a_for_you_paragraph_rewritten_by_hand_stays(self):
        new, _ = write(GUIDE, for_you())
        edited = new.replace("**" + FIXED + "**", "I wrote this one myself, with the right year.")
        again, rep = write(edited, for_you(known={FIXED}, prompt=dict(FY, about="something else")), stamp="2026-11-15 09:00")
        self.assertEqual(again, edited)                                                     # pass: nothing moved
        self.assertIn((1, "I wrote this one myself, with the right year.", True), rep["yours"])
        self.assertEqual((rep["for_you"], rep["missing"]), ([], ["s1w0 p1"]))

    def test_the_report_lists_them_with_what_to_write_about(self):
        _, rep = write(GUIDE, for_you())
        lines = report_lines(rep)
        self.assertIn("In bold for you to rewrite by hand: s1w0 p1", lines)                # flag
        self.assertIn("- s1w0 p1, write about: the study's year", lines)
        _, rep2 = write(GUIDE, para())
        self.assertFalse(any("for you" in ln for ln in report_lines(rep2)))                 # pass


MOVES = 'cut: "These are so common."; splice: "It ends --> here."'


class Repaired(unittest.TestCase):
    """A paragraph whose pick was mended with small edits: normal text, and its marker says how."""

    def test_the_marker_says_repaired_under_its_first_line_for_normal_and_for_you_markers(self):
        new, rep = write(GUIDE, para(repaired=MOVES + "\n\n  and more"))
        first, second = new.split("<!-- depangram s1w0 p1", 1)[1].split(NL)[:2]
        self.assertIn("the paragraph pass (s1 round 2)", first)
        self.assertEqual(second, 'repaired: cut: "These are so common."; splice: "It ends --&gt; here." and more')  # flag
        self.assertEqual(strip_markers(new).count("-->"), GUIDE.count("-->"))              # nothing closed it early
        self.assertIn(NL + "The pass wrote this **thing**. Mine stays." + NL, new)          # normal text, no bold
        self.assertEqual(rep["repaired"], ["s1w0 p1"])
        self.assertIn("Mended with small edits, the marker says which: s1w0 p1", report_lines(rep))
        fy, _ = write(GUIDE, for_you(repaired=MOVES))
        lines = fy.split("<!-- depangram s1w0 p1", 1)[1].split(NL)
        self.assertTrue(lines[0].endswith("FOR YOU TO REWRITE"))
        self.assertTrue(lines[1].startswith("repaired: cut:"))
        self.assertTrue(lines[2].startswith("what went wrong:"))
        plain, rep2 = write(GUIDE, para())                                                  # pass: no repair, no line
        self.assertNotIn("repaired:", plain)
        self.assertEqual(rep2["repaired"], [])
        self.assertFalse(any("Mended" in ln for ln in report_lines(rep2)))

    def test_a_rerun_with_the_same_repair_changes_no_byte_and_a_new_one_rewrites_the_marker(self):
        new, _ = write(GUIDE, para(repaired=MOVES))
        again, rep = write(new, para(known={PASS_P}, repaired=MOVES), stamp="2026-11-15 09:00")
        self.assertEqual(again, new)                                                        # pass
        self.assertEqual(rep["same"], ["s1w0 p1"])
        moved, rep2 = write(new, para(known={PASS_P}, repaired='cut: "These are so common."'), stamp="2026-11-15 09:00")
        self.assertIn("written 2026-11-15 09:00", moved)                                    # flag: the line changed
        self.assertIn(NL + 'repaired: cut: "These are so common."' + NL + "the original", moved)
        self.assertEqual(moved.count("<!-- depangram s1w0 p1"), 1)
        self.assertEqual((rep2["marked"], rep2["ledger"]), (["s1w0 p1"], []))
        self.assertIn(NL + "The pass wrote this **thing**. Mine stays." + NL, moved)        # the words stay
        gone, _ = write(new, para(known={PASS_P}), stamp="2026-11-15 09:00")                # flag: no repair now
        self.assertNotIn("repaired:", gone)
        fy, _ = write(GUIDE, for_you(repaired=MOVES))
        fy_moved, _ = write(fy, for_you(known={FIXED}, repaired='cut: "x."'), stamp="2026-11-15 09:00")
        self.assertIn("written 2026-11-15 09:00", fy_moved)                                 # flag: a for-you one too
        self.assertEqual(write(fy, for_you(known={FIXED}, repaired=MOVES), stamp="2026-11-15 09:00")[0], fy)  # pass


class WithAJob(unittest.TestCase):
    def test_the_rows_take_the_pass_s_text_over_the_working_copy_s(self):
        with tempfile.TemporaryDirectory() as d:
            make_job(d)
            job = Job(root=d)
            bank = Bank(job.bank_path)
            rows = paragraphs(job, bank, passed={"s1w0 p1": {"text": "Pass text here. I wrote this bit myself, mate.",
                                                             "round": "s1 round 2", "flagged": True}})
            first = rows[0]
            self.assertEqual((first["label"], first["new"], first["source"], first["flagged"]),
                             ("s1w0 p1", "Pass text here. I wrote this bit myself, mate.", "the paragraph pass (s1 round 2)", True))
            self.assertEqual(first["orig_block"], "Claude wrote this line about tokens and words. I wrote this bit myself, mate.")
            self.assertEqual(rows[1]["new"], rows[1]["orig"])                    # no pass text, no working copy: the original
            self.assertEqual(paragraphs(job, bank)[0]["source"], "the loop")
            self.assertIn(first["orig"], frozen_blocks(job))
            self.assertIsNone(first["for_you"])                                  # pass: a pass row with no prompt
            fy = paragraphs(job, bank, passed={"s1w0 p1": {"text": "Fixed text. I wrote this bit myself, mate.",
                                                           "round": "s1 round 2", "flagged": False, "for_you": FY}})[0]
            self.assertEqual((fy["for_you"], fy["source"]), (FY, "hand-corrected by the reviewer"))        # flag
            self.assertIsNone(first["repaired"])                                 # pass: no repair
            rp = paragraphs(job, bank, passed={"s1w0 p1": {"text": "Mended text. I wrote this bit myself, mate.",
                                                           "round": "s1 round 2", "flagged": False, "for_you": None,
                                                           "repaired": MOVES}})[0]
            self.assertEqual((rp["repaired"], rp["source"]), (MOVES, "the paragraph pass (s1 round 2)"))     # flag

    def test_a_write_back_straight_after_a_sync_keeps_the_loop_s_text(self):
        """A sync starts the working copy again from the original, so read as the loop's newest text every paragraph
        of the loop's in the guide would go back to its original at the morning's write-back. Since 2026-11-15 the
        sync carries the loop's text over where the original didn't change (sync.carry_working), and the write-back
        keeps the guide's text either way."""
        from engine.sync import sync
        none = {"paragraphs": {}, "sections": {}}
        with tempfile.TemporaryDirectory() as d:
            make_job(d)
            job = Job(root=d)
            job.working.write_text(ORIG.replace("Claude wrote this line about tokens and words.", REGROWN), encoding="utf-8")
            real = Path(d) / "real.md"
            real.write_text(ORIG, encoding="utf-8")
            rep, _, changed = run(job, real=real, passed=none, stamp="2026-11-14 21:00")
            self.assertEqual((rep["replaced"], changed), (["s1w0 p1"], True))
            written = real.read_text(encoding="utf-8")
            sync(d, real, write=True, classify=stub_scan, today="2026-11-15")
            job = Job(root=d)
            self.assertIn(REGROWN, job.working.read_text(encoding="utf-8"))      # the loop's text stays standing
            for after_sync in (True, False):                                     # pass: flagged, or seen in the working copy
                rep, _, changed = run(job, real=real, passed=none, stamp="2026-11-15 06:00", after_sync=after_sync)
                self.assertEqual((rep["restored"], rep["same"][:1], changed), ([], ["s1w0 p1"], False))
                self.assertEqual(real.read_text(encoding="utf-8"), written)
            # flag: once the loop has written its working copy, a paragraph it put back at its original goes back
            job.working.write_text(ORIG.replace("Another paragraph that stays", "Another paragraph, the loop's, that stays"),
                                   encoding="utf-8")
            rep, _, changed = run(job, write=False, real=real, passed=none, stamp="2026-11-15 06:00")
            self.assertIn("s1w0 p1", rep["restored"])

    def test_the_sync_knows_the_write_back_s_paragraphs(self):
        from engine.sync import sync
        pass_p = "The pass wrote about tokens here. I wrote this bit myself, mate."
        for with_ledger in (True, False):
            with tempfile.TemporaryDirectory() as d:
                make_job(d)
                job = Job(root=d)
                rows = paragraphs(job, Bank(job.bank_path), passed={"s1w0 p1": {"text": pass_p, "round": "s1 round 2",
                                                                              "flagged": False}})
                real, rep = write_marked(ORIG, [1], rows, "2026-11-14 21:00", frozen_keys=set(frozen_blocks(job)))
                self.assertEqual(rep["replaced"], ["s1w0 p1"])
                if with_ledger:
                    (Path(d) / LEDGER).write_text("".join(json.dumps(r) + "\n" for r in rep["ledger"]), encoding="utf-8")
                src = Path(d) / "real.md"
                src.write_text(real, encoding="utf-8")
                r = sync(d, src, write=True, classify=stub_scan, today="2026-11-15")
                orig = Job(root=d).original.read_text(encoding="utf-8")
                self.assertNotIn("depangram", orig)                              # the markers never reach the frozen copy
                if with_ledger:                                                  # pass: the machine's, back to the original
                    self.assertEqual((r["restored"], r["your_edits"], r["kept"]), (["s1w0 p1"], [], [("s1w0", "s1w0")]))
                    self.assertEqual(orig, ORIG)
                else:                                                            # flag: unknown text is taken for yours
                    self.assertEqual((r["restored"], [t for _, t in r["your_edits"]], r["retired"]), ([], [pass_p], ["s1w0"]))

    def test_a_section_out_of_scope_keeps_the_loop_s_words_and_gets_its_originals(self):
        """Section 3 on 2026-11-14: out of scope since the sync, the loop's text from the M2 write-back still in the
        real guide, and the first publish put it on main with no original beside it."""
        from engine.sync import sync
        second = "Claude wrote the second section's only line."
        with tempfile.TemporaryDirectory() as d:
            make_job(d)
            job = Job(root=d)
            job.original.write_text(ORIG.replace("<!-- BEATS 2\n- beat\n-->", "<!-- BEATS 2\n- beat\n-->\n\n" + second
                                                 + "\n\n   - a note to self, so the section is still drafting"), encoding="utf-8")
            bank = Bank(job.bank_path)
            bank.add({"kind": "start", "window": "s2w0~before-sync-2026-11-12", "ai": 1.0, "segs": [],
                      "text": "2. Second\n\n" + second})
            bank.add({"kind": "round", "window": "s2w0~before-sync-2026-11-12", "ai": 0.0, "text": "x",
                      "runs": [[["Mistral wrote the second section's line."]]]})
            rows = set_aside(job, Bank(job.bank_path), {1})
            self.assertEqual([(r["label"], r["section"], r["new"], r["orig_block"]) for r in rows],
                             [("s2w0 p1", 2, None, second)])
            self.assertEqual(set_aside(job, Bank(job.bank_path), {1, 2}), [])              # in scope: the layout's job
            real = job.original.read_text(encoding="utf-8").replace(second, "Mistral wrote the second section's line.")
            new, rep = write_marked(real, [1, 2], rows, "2026-11-14 22:00", frozen_keys=set(frozen_blocks(job)))
            self.assertEqual((rep["marked"], rep["replaced"], rep["missing"]), (["s2w0 p1"], [], []))
            self.assertEqual(strip_markers(new), real)                                    # pass: no word of it moved
            self.assertIn(second + "\n-->\n\nMistral wrote the second section's line.", new)
            at_orig, rep2 = write_marked(job.original.read_text(encoding="utf-8"), [1, 2], rows, "2026-11-14 22:00")
            self.assertEqual((at_orig, rep2["marked"], rep2["missing"]),                   # flag: nothing of the loop's there
                             (job.original.read_text(encoding="utf-8"), [], []))
            src = Path(d) / "real.md"
            src.write_text(new, encoding="utf-8")
            r = sync(d, src, write=True, classify=stub_scan, today="2026-11-15")
            frozen = Job(root=d).original.read_text(encoding="utf-8")
            self.assertIn(second, frozen)                                                 # the sync puts the original back
            self.assertNotIn("Mistral wrote", frozen)
            self.assertEqual(r["your_edits"], [])

    def test_a_pass_sentence_you_kept_in_an_edit_is_still_the_machine_s(self):
        from engine.sync import sync
        with tempfile.TemporaryDirectory() as d:
            make_job(d)
            (Path(d) / LEDGER).write_text(json.dumps({
                "when": "2026-11-14 21:00", "label": "s1w0 p1", "source": "the paragraph pass (s1 round 2)",
                "orig": "Claude wrote this line about tokens and words. I wrote this bit myself, mate.",
                "text": "The pass wrote about tokens here. I wrote this bit myself, mate."}) + "\n", encoding="utf-8")
            real = ORIG.replace("Claude wrote this line about tokens and words. I wrote this bit myself, mate.",
                                "The pass wrote about tokens here. I wrote this bit myself, mate, and a new bit too.")
            src = Path(d) / "real.md"
            src.write_text(real, encoding="utf-8")
            r = sync(d, src, write=True, classify=stub_scan, today="2026-11-15")
            self.assertEqual(r["new_verdicts"], {"loop": 1, "yours": 1})
            self.assertEqual(Job(root=d).verdicts().get("The pass wrote about tokens here."), "loop")
            self.assertNotEqual(Job(root=d).verdicts().get(REGROWN), "loop")       # flag: a sentence nobody wrote back


ST1 = {"why": "no rewrite got through the gate and the fact check", "about": "what Claude said",
       "point": "the thing is the point", "fact": "it said a thing --> once"}
ST2 = {"why": "the loop kept it", "about": "your own line", "point": "", "fact": ""}


def stuck(new=ORIG_P, groups=(("Claude said a thing.", ST1),), **kw):
    """A row of the loop's (no pick) whose stuck units hold `groups`: (bold sentences, prompt) each."""
    st = [dict(g, bold=[b] if isinstance(b, str) else list(b)) for b, g in groups]
    return para(new=new, source="the loop", stuck=st, **kw)


class Stuck(unittest.TestCase):
    """A unit the pass found no pick for: its sentences still Claude's word for word go in bold, with a prompt."""

    def test_the_stuck_sentence_goes_in_bold_under_its_prompt_even_at_the_original(self):
        new, rep = write(GUIDE, stuck())
        mark = NL.join(["<!-- depangram s1w0 p1 · written 2026-11-14 21:00 · the original · FOR YOU TO REWRITE",
                        "why it's yours: no rewrite got through the gate and the fact check",
                        "write about: what Claude said", "the point: the thing is the point",
                        "the fact: it said a thing --&gt; once",
                        "the original (frozen at the sync of 2026-11-12):", "Claude said a **thing**. Mine stays.", "-->"])
        # the original's bold word sits inside the bold sentence: no nested marks
        self.assertIn(NL * 2 + mark + NL * 2 + "**Claude said a thing.** Mine stays." + NL * 2, new)      # flag
        self.assertNotIn("****", new)
        self.assertEqual(strip_markers(new).count("-->"), GUIDE.count("-->"))              # nothing closed it early
        self.assertEqual((rep["for_you"], rep["stuck"], rep["replaced"]), (["s1w0 p1"], ["s1w0 p1"], []))
        self.assertEqual(rep["marked"], ["s1w0 p1"])
        self.assertIn("1 paragraph(s) here sit under their commented originals · 1 in bold for you to rewrite by hand",
                      new)
        self.assertEqual(as_posted(new, [1])[1], as_posted(GUIDE, [1])[1])                  # the words posted don't move
        from engine.writeback import is_for_you, marker_prompt
        block = new.split(mark + NL * 2, 1)[1].split(NL * 2)[0]
        lf = mark.replace(NL, "\n")                                     # the write-back reads LF
        self.assertTrue(is_for_you(lf))
        self.assertEqual(marker_prompt(lf)[0], "why it's yours: no rewrite got through the gate and the fact check")
        from engine.texts import strip_markdown
        from engine.wrapup import norm, recognized
        p = stuck()
        self.assertTrue(recognized(norm(strip_markdown(block)), p["versions"], p["known"]))   # still the machine's
        plain, rep2 = write(GUIDE, para(new=ORIG_P, source="the loop"))                       # pass: no stuck, no marker
        self.assertEqual((plain, rep2["same"], rep2["for_you"]), (GUIDE, ["s1w0 p1"], []))

    def test_the_original_s_bold_words_outside_a_bold_sentence_come_back(self):
        new, _ = write(GUIDE, stuck(groups=(("Mine stays.", ST1),)))
        self.assertIn("-->" + NL * 2 + "Claude said a **thing**. **Mine stays.**" + NL * 2, new)
        passed, _ = write(GUIDE, stuck(new=PASS_P, groups=(("Mine stays.", ST1),)))     # a newer text of the loop's
        self.assertIn("· the loop · FOR YOU TO REWRITE", passed)
        self.assertIn("-->" + NL * 2 + "The pass wrote this **thing**. **Mine stays.**" + NL * 2, passed)

    def test_two_entries_name_their_bold_by_its_first_words_and_run_together(self):
        new, rep = write(GUIDE, stuck(groups=(("Claude said a thing.", ST1), ("Mine stays.", ST2))))
        lines = new.split("<!-- depangram s1w0 p1", 1)[1].split(NL)
        self.assertEqual(lines[1:8], ['the bold starting "Claude said a thing.":',
                                      "why it's yours: no rewrite got through the gate and the fact check",
                                      "write about: what Claude said", "the point: the thing is the point",
                                      "the fact: it said a thing --&gt; once",
                                      'the bold starting "Mine stays.":', "why it's yours: the loop kept it"])
        self.assertEqual(lines[8], "write about: your own line")                            # an empty line left out
        self.assertIn("-->" + NL * 2 + "**Claude said a thing. Mine stays.**" + NL * 2, new)
        self.assertEqual(rep["prompts"]["s1w0 p1"]["about"], "what Claude said; your own line")   # the digest's line
        out = report_lines(rep)
        self.assertIn("In bold for you to rewrite by hand: s1w0 p1", out)
        self.assertIn('- s1w0 p1, the bold starting "Claude said a thing.": why it\'s yours: no rewrite got through'
                      " the gate and the fact check; write about: what Claude said; the point: the thing is the point;"
                      " the fact: it said a thing --> once", out)
        self.assertIn('- s1w0 p1, the bold starting "Mine stays.": why it\'s yours: the loop kept it; write about:'
                      " your own line", out)
        one, _ = write(GUIDE, stuck(groups=(("Claude said a thing.", ST1), ("Not in this paragraph.", ST2))))
        self.assertNotIn("the bold starting", one)                  # pass: a group with nothing in bold is left out
        self.assertNotIn("the loop kept it", one)

    def test_a_rerun_changes_no_byte_and_a_new_prompt_or_sentence_rewrites_it(self):
        new, _ = write(GUIDE, stuck())
        again, rep = write(new, stuck(), stamp="2026-11-15 09:00")
        self.assertEqual(again, new)                                                        # pass
        self.assertEqual((rep["same"], rep["for_you"], rep["ledger"]), (["s1w0 p1"], ["s1w0 p1"], []))
        moved, rep2 = write(new, stuck(groups=(("Claude said a thing.", dict(ST1, point="a sharper point")),)),
                            stamp="2026-11-15 09:00")
        self.assertIn("written 2026-11-15 09:00 · the original · FOR YOU TO REWRITE", moved)    # flag: the prompt
        self.assertIn(NL + "the point: a sharper point" + NL, moved)
        self.assertEqual(moved.count("<!-- depangram s1w0 p1"), 1)
        self.assertEqual(rep2["marked"], ["s1w0 p1"])
        other, _ = write(new, stuck(groups=(("Mine stays.", ST1),)), stamp="2026-11-15 09:00")
        self.assertIn("written 2026-11-15 09:00", other)                                    # flag: the sentences
        self.assertIn("-->" + NL * 2 + "Claude said a **thing**. **Mine stays.**" + NL * 2, other)
        self.assertEqual(other.count("<!-- depangram s1w0 p1"), 1)

    def test_out_of_bold_only_the_original_s_bold_words_come_back(self):
        new, _ = write(GUIDE, stuck(groups=(("Claude said a thing.", ST1), ("Mine stays.", ST2))))
        back, rep = write(new, para(new=ORIG_P, source="the loop"), stamp="2026-11-15 09:00")
        self.assertEqual(back, GUIDE)                                       # flag: marker, note and bold all gone
        self.assertEqual(rep["restored"], ["s1w0 p1"])
        later, rep2 = write(new, para(), stamp="2026-11-15 09:00")          # the pass found a pick since
        self.assertIn("written 2026-11-15 09:00 · the paragraph pass (s1 round 2)" + NL + "the original", later)
        self.assertIn("-->" + NL * 2 + "The pass wrote this **thing**. Mine stays." + NL * 2, later)
        self.assertNotIn("FOR YOU", later)
        self.assertNotIn("in bold for you", later)
        self.assertEqual((rep2["replaced"], rep2["for_you"]), (["s1w0 p1"], []))
        fy, _ = write(GUIDE, for_you(new=ORIG_P))           # a hand fix that is the original word for word
        self.assertIn("**" + ORIG_P + "**", fy)
        fy_back, rep3 = write(fy, para(new=ORIG_P), stamp="2026-11-15 09:00")
        self.assertEqual((fy_back, rep3["restored"]), (GUIDE, ["s1w0 p1"]))      # flag: its bold comes out the same way

    def test_a_hand_fix_on_the_same_paragraph_wins(self):
        new, rep = write(GUIDE, for_you(stuck=[dict(ST1, bold=["Mine stays."])]))
        self.assertIn("-->" + NL * 2 + "**" + FIXED + "**" + NL * 2, new)                # whole in bold
        self.assertIn("hand-corrected by the reviewer · FOR YOU TO REWRITE" + NL + "what went wrong:", new)
        self.assertNotIn("why it's yours", new)
        self.assertEqual((rep["for_you"], rep["stuck"], rep["prompts"]["s1w0 p1"]), (["s1w0 p1"], [], FY))

    def test_a_stuck_paragraph_with_a_pick_and_a_repair_keeps_the_repaired_line(self):
        row = stuck(new=PASS_P, groups=(("Mine stays.", ST1),), repaired=MOVES)
        row["source"] = "the paragraph pass (s1 round 2)"
        new, rep = write(GUIDE, row)
        lines = new.split("<!-- depangram s1w0 p1", 1)[1].split(NL)
        self.assertTrue(lines[0].endswith("the paragraph pass (s1 round 2) · FOR YOU TO REWRITE"))
        self.assertTrue(lines[1].startswith("repaired: cut:"))
        self.assertTrue(lines[2].startswith("why it's yours:"))
        self.assertEqual((rep["repaired"], rep["stuck"], rep["replaced"]), (["s1w0 p1"], ["s1w0 p1"], ["s1w0 p1"]))
        self.assertEqual(write(new, dict(row, known={PASS_P}), stamp="2026-11-15 09:00")[0], new)     # pass: no byte


def make_job2(d):
    """make_job with a second Claude sentence in the first paragraph (so a stuck span can hold two sentences)."""
    d = make_job(d)
    two = "Claude wrote this line about tokens and words. Claude wrote a second line about the cache."
    orig = ORIG.replace("Claude wrote this line about tokens and words.", two)
    (d / "orig.md").write_text(orig, encoding="utf-8")
    with (d / "your lines.md").open("a", encoding="utf-8") as f:
        f.write("| claude | 0% | 100% | AI | Claude wrote a second line about the cache. |\n")
    (d / "bank.jsonl").write_text("", encoding="utf-8")
    Bank(d / "bank.jsonl").add({"kind": "start", "window": "s1w0", "ai": 1.0, "segs": [],
                                "text": Job(root=d).windows()[0]["text"]})
    return d, orig, two


STUCK_ROW = {"text": "", "section": 1, "round": None, "ai": None, "flagged": False, "for_you": None,
             "repaired": None, "picked": False}


class StuckWithAJob(unittest.TestCase):
    def passed(self, groups):
        return {"paragraphs": {"s1w0 p1": dict(STUCK_ROW, stuck=[dict(g, bold=b) for b, g in groups])},
                "sections": {}}

    def test_a_row_the_pass_never_picked_keeps_the_loop_s_text_and_source(self):
        with tempfile.TemporaryDirectory() as d:
            make_job(d)
            job = Job(root=d)
            job.working.write_text(ORIG.replace("Claude wrote this line about tokens and words.", REGROWN),
                                   encoding="utf-8")
            row = paragraphs(job, Bank(job.bank_path), passed=self.passed([(["I wrote this bit myself, mate."],
                                                                           ST1)])["paragraphs"])[0]
            self.assertEqual((row["new"], row["source"], row["flagged"]),
                             (REGROWN + " I wrote this bit myself, mate.", "the loop", None))     # pass: not the pass's
            self.assertEqual(row["stuck"][0]["bold"], ["I wrote this bit myself, mate."])
            picked = paragraphs(job, Bank(job.bank_path), passed={"s1w0 p1": {
                "text": "Pass text. I wrote this bit myself, mate.", "round": "s1 round 2", "flagged": False}})[0]
            self.assertEqual(picked["source"], "the paragraph pass (s1 round 2)")       # flag: no `picked` is a pick
            self.assertIsNone(picked["stuck"])

    def test_after_a_sync_the_original_goes_in_bold_and_a_rerun_and_a_removal_hold(self):
        with tempfile.TemporaryDirectory() as d:
            d, orig, two = make_job2(d)
            job = Job(root=d)
            real = Path(d) / "real.md"
            real.write_text(orig, encoding="utf-8")
            passed = self.passed([([two.split(". ")[0] + ".", two.split(". ")[1]], ST1)])
            rep, _, changed = run(job, real=real, passed=passed, stamp="2026-11-15 06:00", after_sync=True)
            self.assertTrue(changed)
            written = real.read_text(encoding="utf-8")
            self.assertIn("· the original · FOR YOU TO REWRITE", written)
            self.assertIn("-->\n\n**" + two + "** I wrote this bit myself, mate.\n", written)   # flag
            self.assertEqual((rep["stuck"], rep["for_you"]), (["s1w0 p1"], ["s1w0 p1"]))
            again, _, changed = run(job, real=real, passed=passed, stamp="2026-11-15 07:00", after_sync=True)
            self.assertEqual((changed, real.read_text(encoding="utf-8")), (False, written))  # pass: no byte
            run(job, real=real, passed={"paragraphs": {}, "sections": {}}, stamp="2026-11-15 08:00", after_sync=True)
            self.assertEqual(real.read_text(encoding="utf-8"), orig)                         # flag: the entry went

    def test_with_no_newer_text_the_bold_still_comes_out_when_the_entry_goes(self):
        """After a sync the row has no newer text, and the guide holds the loop's text from an earlier write-back."""
        with tempfile.TemporaryDirectory() as d:
            d, orig, two = make_job2(d)
            job = Job(root=d)
            first, second = two.split(". ")[0] + ".", two.split(". ")[1]
            held = orig.replace(first, REGROWN)
            plain = norm(REGROWN + " " + second + " I wrote this bit myself, mate.")
            (Path(d) / LEDGER).write_text(json.dumps({"when": "2026-11-14 21:00", "label": "s1w0 p1",
                                                      "source": "the loop", "orig": norm(two + " I wrote this bit"
                                                                                         " myself, mate."),
                                                      "text": plain}) + "\n", encoding="utf-8")
            real = Path(d) / "real.md"
            real.write_text(held, encoding="utf-8")
            run(job, real=real, passed=self.passed([([second], ST1)]), stamp="2026-11-15 06:00", after_sync=True)
            self.assertIn("-->\n\n" + REGROWN + " **" + second + "** I wrote", real.read_text(encoding="utf-8"))
            rep, _, _ = run(job, real=real, passed={"paragraphs": {}, "sections": {}}, stamp="2026-11-15 07:00",
                            after_sync=True)
            out = real.read_text(encoding="utf-8")
            self.assertNotIn("FOR YOU", out)                                                    # flag: it came out
            self.assertNotIn("**", strip_markers(out))
            self.assertIn("-->\n\n" + plain + "\n", out)                       # the loop's words, under a plain marker
            self.assertEqual(rep["marked"], ["s1w0 p1"])

    def test_a_sync_takes_a_loop_paragraph_with_stuck_bold_back_to_its_original(self):
        from engine.sync import sync
        with tempfile.TemporaryDirectory() as d:
            d, orig, two = make_job2(d)
            job = Job(root=d)
            first, second = two.split(". ")[0] + ".", two.split(". ")[1]
            job.working.write_text(job.placeholder_base()[0].replace(first, REGROWN), encoding="utf-8")
            real = Path(d) / "real.md"
            real.write_text(orig, encoding="utf-8")
            rep, _, _ = run(job, real=real, passed=self.passed([([second], ST1)]), stamp="2026-11-15 06:00")
            self.assertEqual((rep["replaced"], rep["stuck"]), (["s1w0 p1"], ["s1w0 p1"]))
            self.assertIn(REGROWN + " **" + second + "**", real.read_text(encoding="utf-8"))
            r = sync(d, real, write=True, classify=stub_scan, today="2026-11-16")
            job = Job(root=d)
            self.assertEqual((r["restored"], r["your_edits"], r["retired"]), (["s1w0 p1"], [], []))
            self.assertEqual(job.original.read_text(encoding="utf-8"), orig)                    # pass: no leak
            self.assertNotIn("**", job.working.read_text(encoding="utf-8"))
            self.assertIn(REGROWN, job.working.read_text(encoding="utf-8"))      # the loop's text, carried

    def test_the_old_original_s_bold_comes_back_by_the_marker_s_original_even_under_an_older_label(self):
        """A marker that didn't change keeps the label of the cut it was written under, so the sync finds the old
        frozen block by the original the marker holds."""
        from engine.sync import sync
        with tempfile.TemporaryDirectory() as d:
            d, orig, two = make_job2(d)
            orig = orig.replace("about tokens and words", "about **tokens** and words")
            (d / "orig.md").write_text(orig, encoding="utf-8")
            job = Job(root=d)
            real = d / "real.md"
            real.write_text(orig, encoding="utf-8")
            first, second = "Claude wrote this line about tokens and words.", two.split(". ")[1]
            run(job, real=real, passed=self.passed([([first, second], ST1)]), stamp="2026-11-15 06:00",
                after_sync=True)
            text = real.read_text(encoding="utf-8")
            self.assertIn("**" + two + "**", text)                       # the bold word is inside the bold run
            mine = "I rewrote the cache line myself."
            text = text.replace("<!-- depangram s1w0 p1 ", "<!-- depangram s1w7 p9 ").replace(
                "**" + two + "**", f"**{first} {mine}**")
            real.write_text(text, encoding="utf-8")
            sync(d, real, write=True, classify=stub_scan, today="2026-11-16")
            frozen = Job(root=d).original.read_text(encoding="utf-8")
            self.assertIn(f"Claude wrote this line about **tokens** and words. {mine} I wrote", frozen)    # flag
            self.assertEqual(frozen.count("**"), 2)

    def test_a_sync_on_a_guide_with_stuck_bold_freezes_the_same_original_and_working_copy(self):
        from engine.sync import sync
        frozen = {}
        for bolded in (False, True):
            with tempfile.TemporaryDirectory() as d:
                d, orig, two = make_job2(d)
                job = Job(root=d)
                real = Path(d) / "real.md"
                real.write_text(orig, encoding="utf-8")
                if bolded:
                    run(job, real=real, passed=self.passed([([two.split(". ")[0] + ".", two.split(". ")[1]], ST1)]),
                        stamp="2026-11-15 06:00", after_sync=True)
                    self.assertIn("**" + two + "**", real.read_text(encoding="utf-8"))
                r = sync(d, real, write=True, classify=stub_scan, today="2026-11-16")
                job = Job(root=d)
                frozen[bolded] = (job.original.read_text(encoding="utf-8"), job.working.read_text(encoding="utf-8"),
                                  r["your_edits"], r["retired"])
        self.assertEqual(frozen[True], frozen[False])                       # pass: no bold mark leaks into either
        self.assertNotIn("**", frozen[True][0])
        self.assertEqual(frozen[True][2:], ([], []))
        with tempfile.TemporaryDirectory() as d:                            # bold you typed with no marker stays
            d, orig, two = make_job2(d)
            real = Path(d) / "real.md"
            real.write_text(orig.replace("tokens and words", "**tokens** and words"), encoding="utf-8")
            sync(d, real, write=True, classify=stub_scan, today="2026-11-16")
            self.assertIn("**tokens** and words", Job(root=d).original.read_text(encoding="utf-8"))

    def test_a_bold_sentence_you_rewrote_by_hand_comes_in_as_yours_and_the_bold_comes_out(self):
        """The whole trip (the reviewer's rule of 2026-11-15: under a FOR YOU marker, bold is the machine's flag).
        A write-back puts a stuck unit's two sentences in bold; you rewrite the second in the guide and keep the
        `**` around it; the sync freezes your paragraph with no bold the original didn't have; the next write-back
        leaves your words with the marker off, or, while the entry still speaks for what is left of its unit, puts
        only the first sentence back in bold under the old prompt; a second sync and write-back change no byte."""
        from engine.regrow import pass_paragraphs
        from engine.sync import sync

        class NoCache:
            def cached(self, text):
                return None

        for keep_entry in (False, True):
            with self.subTest(keep_entry=keep_entry), tempfile.TemporaryDirectory() as d:
                d, orig, two = make_job2(d)
                first, second = two.split(". ")[0] + ".", two.split(". ")[1]
                entry = dict(ST1, label="s1w0 p1", orig=two, when="2026-11-15")
                (d / "review.json").write_text(json.dumps({"vetoes": [], "repairs": [], "for_you": [],
                                                           "stuck": [entry]}), encoding="utf-8")
                (d / "regrow bank.jsonl").write_text(json.dumps({
                    "kind": "score", "section": 1, "label": "s1 round 2", "arm_set": "mixed", "ai": 0.3,
                    "humanized": False, "picks": {}, "span_flagged": {}, "spans": True}) + "\n", encoding="utf-8")
                real = d / "real.md"
                real.write_text(orig, encoding="utf-8")

                def write_back(stamp):
                    job = Job(root=d)
                    passed = pass_paragraphs(job, Bank(job.bank_path), [1])
                    return passed, run(job, real=real, passed=passed, stamp=stamp, pangram=NoCache())[0]

                passed, _ = write_back("2026-11-15 06:00")
                self.assertEqual(passed["paragraphs"]["s1w0 p1"]["stuck"][0]["bold"], [first, second])
                self.assertIn("**" + two + "** I wrote this bit myself, mate.", real.read_text(encoding="utf-8"))
                mine = "I rewrote the cache line myself."
                real.write_text(real.read_text(encoding="utf-8").replace("**" + two + "**", f"**{first} {mine}**"),
                                encoding="utf-8")
                r = sync(d, real, write=True, classify=stub_scan, today="2026-11-16")
                yours = f"{first} {mine} I wrote this bit myself, mate."
                self.assertEqual(r["your_edits"], [(1, yours)])                                 # your edit
                frozen = Job(root=d).original.read_text(encoding="utf-8")
                self.assertIn("\n\n" + yours + "\n\n", frozen)                    # flag: no bold mark frozen in
                self.assertNotIn("**", frozen)
                self.assertNotIn("**", Job(root=d).working.read_text(encoding="utf-8"))
                if not keep_entry:
                    (d / "review.json").write_text(json.dumps({"vetoes": [], "repairs": [], "for_you": [],
                                                               "stuck": []}), encoding="utf-8")
                passed, rep = write_back("2026-11-16 06:00")
                after = real.read_text(encoding="utf-8")
                self.assertEqual(passed["unmatched"], [])
                if keep_entry:                                    # the entry speaks for what is left: A, under its prompt
                    self.assertEqual(passed["paragraphs"]["s1w0 p1"]["stuck"][0]["bold"], [first])
                    self.assertIn("· the original · FOR YOU TO REWRITE\nwhy it's yours: " + ST1["why"], after)
                    self.assertIn("-->\n\n**" + first + "** " + mine + " I wrote this bit myself, mate.\n", after)
                    self.assertEqual(rep["stuck"], ["s1w0 p1"])
                else:                                             # your words, the marker off, no bold of the machine's
                    self.assertNotIn("FOR YOU", after)
                    self.assertNotIn("depangram", after)
                    self.assertEqual(after.replace("\r\n", "\n"), frozen.replace("\r\n", "\n"))
                    self.assertEqual(rep["restored"], ["s1w0 p1"])
                sync(d, real, write=True, classify=stub_scan, today="2026-11-17")
                self.assertEqual(Job(root=d).original.read_text(encoding="utf-8"), frozen)          # pass: no byte
                write_back("2026-11-17 06:00")
                self.assertEqual(real.read_text(encoding="utf-8"), after)

    def test_an_entry_that_speaks_for_no_unit_is_named_in_the_report(self):
        _, rep = write(GUIDE, para())
        self.assertFalse(any("stuck entry" in x for x in report_lines(rep)))                   # pass
        rep["stuck_unmatched"] = ["s4w9 p9"]
        self.assertIn("The stuck entry for s4w9 p9 matches no unit now: write a new one or remove it",
                      report_lines(rep))                                                        # flag

# ---------------------------------------------------------------------- the author's first rewrite, a whole night
A1, A2 = "Alpha claude sentence number one is here.", "Alpha claude sentence number two is here."
B1, B2 = "Bravo claude stuck sentence one is here.", "Bravo claude stuck sentence two is here."
B_MINE, B3 = "I wrote this bravo line myself, mate.", "Bravo claude stuck sentence three is here."
B_LOOP = "The loop added this bravo line here."
B_SHORT = "Short one."
B_NEW = "I rewrote the bravo three sentence myself. " + B_SHORT
C1, C2 = "Charlie claude sentence number one is here.", "Charlie claude sentence number two is here."
D1, D_MINE, D2 = ("Delta claude lead sentence is right here.", "I wrote the delta middle myself, mate.",
                  "Delta claude tail sentence is right here.")
D_LOOP = "Delta loop lead sentence stands right here."
E1, E_MINE, E2 = ("Echo claude lead sentence is right here.", "I wrote the echo middle myself, mate.",
                  "Echo claude tail sentence is right here.")
E_LOOP, PICK_E = "Echo loop lead sentence stands right here.", "Echo pass tail sentence went right here."
NIGHT_ORIG = "\n\n".join(["# Guide", "## 1. First Section", f"{A1} {A2}", f"{B1} {B2} {B_MINE} {B3}", f"{C1} {C2}",
                          f"{E1} {E_MINE} {E2}",
                          "## 2. Second", f"{D1} {D_MINE} {D2}", "## Further Reading", "Links."]) + "\n"
NIGHT_LINES = ("| Verdict | In your messages | In Claude's output | Pangram | Sentence |\n| --- | --- | --- | --- | --- |\n"
               + "".join(f"| claude | 0% | 100% | AI | {s} |\n" for s in (A1, A2, B1, B2, B3, C1, C2, D1, D2, E1, E2))
               + "".join(f"| yours | 100% | 0% | human | {s} |\n" for s in (B_MINE, D_MINE, E_MINE)))
PICK_A, PICK_C_FROM, PICK_C, PICK_D = ("Alpha pass wrote this one right here.", "Charlie pass wrote this, you know.",
                                       "Charlie pass wrote this.", "Delta pass tail sentence went right here.")


class AuthorsFirstRewrite(unittest.TestCase):
    """The reviewer's rehearsal of 2026-11-15 on synthetic fixtures: the author rewrites one bold sentence of a stuck
    paragraph in the guide, then the night's sync and write-back run, twice, and the window loop never runs between.

    The three wrongs it found, each pinned here:
    1. The sync freezes the author's paragraph as the guide had it, so the loop's line next to a stuck span joins
       that span and the span outgrows its stuck entry: the entry must still speak for it (review.holds).
    2. The edit changes the paragraph's length, the window cut moves, and every paragraph after it gets a new label:
       the pass's picks and draws must follow their span by its sha (RegrowPass.where), so no repair goes stale and
       the served picks stay the ones they were.
    3. The sync started the working copy again from the original, so a paragraph that mixes a pass pick with the
       loop's text lost the loop's part and a section nobody touched changed: the sync carries the loop's shipped
       text over (sync.carry_working), and the loop's versions of a window the sync retired still split that text
       into its runs (layout's retired_versions: E, whose window was re-cut).
    Found on the way: the sentences the author typed get "yours" at the sync, short ones too (the chat-log scan
    can't know them, and a sentence with no verdict reads as Claude's), and a pick whose text the sync froze as
    its unit's original is absorbed, not stale (case 4 of the rehearsal over every bold run)."""

    def setUp(self):
        from engine.regrow import orig_key
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch("engine.job.WINDOW_WORDS", 60)
        self.patch.start()
        d = self.d = Path(self.tmp.name)
        (d / "job.toml").write_text('[job]\nname = "t"\noriginal = "orig.md"\nworking = "work.md"\nsections = [1, 2]\n'
                                    'voice_samples = [9]\n', encoding="utf-8")
        (d / "orig.md").write_text(NIGHT_ORIG, encoding="utf-8")
        (d / "your lines.md").write_text(NIGHT_LINES, encoding="utf-8")
        job = Job(root=d)
        bank = Bank(d / "bank.jsonl")
        for w in job.windows():
            bank.add({"kind": "start", "window": w["wid"], "ai": 1.0, "segs": [], "text": w["text"]})
        from engine.wrapup import layout
        lay = layout(job, Bank(d / "bank.jsonl"))
        self.assertEqual(sorted(lay), ["s1w0", "s1w1", "s2w0"])
        self.assertEqual([len(w["original"]) for _, w in sorted(lay.items())], [3, 1, 1])     # A B C | E | D
        best = {}
        for wid, w in lay.items():                     # the loop's round: B gains a line, D's lead is rewritten
            runs = [[list(r) for r in p] for p in w["original"]]
            for p in runs:
                for r in p:
                    if r == [B1, B2]:
                        r.append(B_LOOP)
                    if r == [D1]:
                        r[0] = D_LOOP
                    if r == [E1]:
                        r[0] = E_LOOP
            bank.add({"kind": "round", "window": wid, "ai": 0.1, "text": "x", "runs": runs})
            best[wid] = ([" ".join(s for r in p for s in r) for p in w["original"]],
                         [" ".join(s for r in p for s in r) for p in runs])
        job.assemble(best)                             # the working copy as the loop's stop wrote it
        self.units0 = {"s1w0 p1": f"{A1} {A2}", "s1w0 p2 s1": f"{B1} {B2}", "s1w0 p2": B3, "s1w0 p3": f"{C1} {C2}",
                       "s1w1 p1 s1": E1, "s1w1 p1": E2, "s2w0 p1 s1": D1, "s2w0 p1": D2}
        sc = {"kind": "score", "arm_set": "mixed", "humanized": False, "span_flagged": {}, "spans": True}
        with (d / "regrow bank.jsonl").open("w", encoding="utf-8") as f:
            for e in (dict(sc, section=1, label="s1 round 2", ai=0.3,
                           picks={"s1w0 p1": PICK_A, "s1w0 p3": PICK_C_FROM, "s1w1 p1": PICK_E}),
                      dict(sc, section=2, label="s2 round 2", ai=0.2, picks={"s2w0 p1": PICK_D})):
                e["origs"] = {k: orig_key(self.units0[k]) for k in e["picks"]}
                f.write(json.dumps(e) + "\n")
        (d / "review.json").write_text(json.dumps({
            "vetoes": [], "for_you": [],
            "repairs": [{"label": "s1w0 p3", "orig": f"{C1} {C2}", "from": PICK_C_FROM, "text": PICK_C,
                         "moves": "cut: ', you know'", "by": "reviewer", "when": "2026-11-14"}],
            "stuck": [dict(ST1, label="s1w0 p2 s1", orig=f"{B1} {B2}", when="2026-11-15"),
                      dict(ST2, label="s1w0 p2", orig=B3, when="2026-11-15")]}), encoding="utf-8")
        self.real = d / "real.md"
        self.real.write_text(NIGHT_ORIG, encoding="utf-8")

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def night(self, stamp, after_sync=True):
        from engine.regrow import RegrowPass, pass_paragraphs

        class NoCache:
            def cached(self, text):
                return None
        job = Job(root=self.d)
        passed = pass_paragraphs(job, Bank(job.bank_path), [1, 2])
        rep = run(job, real=self.real, passed=passed, stamp=stamp, pangram=NoCache(), after_sync=after_sync)[0]
        rp = RegrowPass(job=job, bank=Bank(job.bank_path), settings=None, conn=None, log=lambda *_: None)
        rp.load_units([1, 2])
        rp.load_bank()
        return passed, rep, rp

    def section(self, text, n):
        from engine.wrapup import section_spans
        a, b = section_spans(text.replace("\r\n", "\n"), [n])[n]
        return text.replace("\r\n", "\n")[a:b]

    def test_a_retired_window_s_versions_count_for_the_same_original_runs_only(self):
        from engine.wrapup import layout, run_versions
        with tempfile.TemporaryDirectory() as d:
            make_job(d)
            lines = (Path(d) / "bank.jsonl").read_text(encoding="utf-8").replace('"s1w0"', '"s1w0~before-sync-x"')
            (Path(d) / "bank.jsonl").write_text(lines, encoding="utf-8")
            job, bank = Job(root=d), Bank(Path(d) / "bank.jsonl")
            bank.add({"kind": "start", "window": "s1w0", "ai": 1.0, "segs": [], "text": job.windows()[0]["text"]})
            lay = layout(job, bank)
            v = run_versions(bank, "s1w0", lay["s1w0"]["original"])
            self.assertIn(REGROWN, v[0][0])                               # flag: the retired window's version
            other = [[["A different first run."], ["I wrote this bit myself, mate."]]]
            self.assertEqual(run_versions(bank, "s1w0", other)[0][0], {"A different first run."})   # pass

    def edit_once_then_twice(self, edit, classify=stub_scan):
        """Night 1 (the write-back), your edit in the guide, then two nights. Returns (served before, the edited
        guide, the guide after night 2, the pass after night 2, the sync's report, the write-back's report)."""
        from engine.sync import sync
        _, _, rp0 = self.night("2026-11-15 06:00", after_sync=False)
        served0 = {n: rp0.section_pick(n) for n in (1, 2)}
        edited = edit(self.real.read_text(encoding="utf-8"))
        self.real.write_text(edited, encoding="utf-8")
        r = sync(self.d, self.real, write=True, classify=classify, today="2026-11-16")
        _, rep, rp = self.night("2026-11-16 02:10")
        after = self.real.read_text(encoding="utf-8")
        sync(self.d, self.real, write=True, classify=classify, today="2026-11-17")
        self.night("2026-11-17 02:10")
        self.assertEqual(self.real.read_text(encoding="utf-8"), after)                 # the second night: no byte
        return served0, edited, after, rp, r, rep

    def changed_blocks(self, edited, after, n):
        b0 = self.section(edited, n).split("\n\n")
        return [x for x in self.section(after, n).split("\n\n") if x not in b0] + [
            x for x in b0 if x not in self.section(after, n).split("\n\n")]

    def test_a_rewrite_inside_a_pass_paragraph_drops_only_that_pick(self):
        """The reviewer's round 4: a sentence the pass wrote, not in bold, rewritten by you. Its pick is gone, not
        stale: the scoring keeps its place and every other pick, the repairs of other units stand."""
        mine = "Alpha pass wrote this one, right over here."
        served0, edited, after, rp, r, rep = self.edit_once_then_twice(lambda t: t.replace(PICK_A, mine))
        self.assertEqual([t for _, t in r["your_edits"]], [mine])
        sp = rp.section_pick(1)
        self.assertEqual((sp["scoring"]["label"], sp["scored"]), ("s1 round 2", False))       # pass: same scoring
        self.assertEqual(sorted(sp["picks"].values()), sorted([PICK_C, PICK_E]))               # A's pick dropped
        self.assertEqual(list(sp["scoring"]["gone"]), ["s1w0 p1"])
        self.assertNotIn("stale", sp["scoring"])
        self.assertEqual((rp.stale_repairs, len(rp.repairs)), ([], 1))
        self.assertIn(mine, after)
        self.assertEqual(self.section(after, 2), self.section(edited, 2))
        changed = self.changed_blocks(edited, after, 1)
        self.assertTrue(all(mine in x or x.startswith("<!-- depangram") for x in changed), changed)
        self.assertNotIn(PICK_A, after)
        from engine.regrow import review_lines
        self.assertIn("s1 round 2 had a pick for s1w0 p1, whose paragraph you rewrote or removed",
                      "\n".join(review_lines(rp)))

    def test_a_repair_whose_paragraph_you_rewrote_is_named_and_its_pin_still_counts(self):
        """C's pick was repaired; you rewrite a word of the repaired text. The repair matches no unit now and is
        named for removal, and a scoring that lacked the repaired text there still can't become the best."""
        from engine.regrow import orig_key
        with (self.d / "regrow bank.jsonl").open("a", encoding="utf-8") as f:     # an older round, lower, no repair
            f.write(json.dumps({"kind": "score", "arm_set": "mixed", "humanized": False, "span_flagged": {},
                                "spans": True, "section": 1, "label": "s1 round 1", "ai": 0.1,
                                "picks": {"s1w0 p1": PICK_A, "s1w0 p3": "Charlie older pick, you know."},
                                "origs": {"s1w0 p1": orig_key(f"{A1} {A2}"), "s1w0 p3": orig_key(f"{C1} {C2}")}})
                    + "\n")
            f.write(json.dumps({"kind": "score", "arm_set": "final", "humanized": False, "span_flagged": {},
                                "spans": True, "section": 1, "label": "s1 final", "ai": 0.3,
                                "picks": {"s1w0 p1": PICK_A, "s1w0 p3": PICK_C, "s1w1 p1": PICK_E},
                                "origs": {"s1w0 p1": orig_key(f"{A1} {A2}"), "s1w0 p3": orig_key(f"{C1} {C2}"),
                                          "s1w1 p1": orig_key(E2)}}) + "\n")
        review = json.loads((self.d / "review.json").read_text(encoding="utf-8"))
        review["repairs"][0]["scoring"] = "s1 final"      # the repair was read against the final: it stands, a pin
        (self.d / "review.json").write_text(json.dumps(review), encoding="utf-8")
        mine = "Charlie pass wrote all of this."
        served0, edited, after, rp, r, rep = self.edit_once_then_twice(lambda t: t.replace(PICK_C, mine))
        self.assertEqual(served0[1]["scoring"]["label"], "s1 final")
        self.assertEqual(rp.section_pick(1)["scoring"]["label"], "s1 final")    # flag: round 1 (0.1) stays out
        self.assertIn(("repair", "s1w0 p3"), rep["review_unmatched"])
        self.assertIn("The repair for s1w0 p3 in review.json matches no unit now (its paragraph was rewritten or"
                      " removed): remove it", report_lines(rep))
        self.assertEqual(rp.stale_repairs, [])
        self.assertEqual(self.section(after, 2), self.section(edited, 2))

    def test_a_pass_paragraph_you_delete_takes_only_its_own_pick(self):
        served0, edited, after, rp, r, rep = self.edit_once_then_twice(
            lambda t: t.replace(PICK_A + chr(10) * 2, "", 1))
        self.assertNotIn(PICK_A, edited)
        sp = rp.section_pick(1)
        self.assertEqual(sp["scoring"]["label"], "s1 round 2")
        self.assertEqual((list(sp["scoring"].get("gone") or ()), sp["scoring"].get("stale")), (["s1w0 p1"], None))
        self.assertEqual(sorted(sp["picks"].values()), sorted([PICK_C, PICK_E]))
        self.assertEqual(rp.stale_repairs, [])
        self.assertEqual(self.section(after, 2), self.section(edited, 2))

    def test_a_span_gone_for_another_reason_is_still_stale(self):
        """Safety: a pick whose span vanished while its paragraph is still there (here a sentence of it turned
        into a line of yours in the verdicts, so the span split) is stale as before, and the scoring is out."""
        from engine.sync import sync
        self.night("2026-11-15 06:00", after_sync=False)
        lines = (self.d / "your lines.md").read_text(encoding="utf-8").replace(f"| claude | 0% | 100% | AI | {A2} |",
                                                                               f"| yours | 100% | 0% | human | {A2} |")
        (self.d / "your lines.md").write_text(lines, encoding="utf-8")
        _, _, rp = self.night("2026-11-16 02:10")
        sc = next(s for s in rp.scores if s["label"] == "s1 round 2")
        self.assertEqual((sc.get("stale"), sc.get("gone")), (["s1w0 p1"], None))                  # flag
        self.assertFalse(rp.eligible(sc))

    def test_carry_working_keeps_the_loop_s_block_only_where_the_original_block_is_the_same(self):
        from engine.sync import carry_working
        old_base = "# T\n\nSame one.\n\nChanged one.\n\nTwice.\n\nTwice.\n"
        old_work = "# T\n\nThe loop's one.\n\nThe loop's changed.\n\nLoop twice a.\n\nLoop twice b.\n"
        new_base = "# T\n\nA new one.\n\nSame one.\n\nChanged by you.\n\nTwice.\n\nTwice.\n"
        text, n = carry_working(old_base, old_work, new_base)
        self.assertEqual(n, 1)                                     # flag: the same block moved and kept its text
        self.assertEqual(text, "# T\n\nA new one.\n\nThe loop's one.\n\nChanged by you.\n\nTwice.\n\nTwice.\n")
        self.assertEqual(carry_working(old_base, "# T\n\nOne block only.\n", new_base), (new_base, 0))  # pass

    def test_a_rewrite_moves_nothing_but_the_rewritten_paragraph(self):
        from engine.sync import sync
        passed0, _, rp0 = self.night("2026-11-15 06:00", after_sync=False)
        served0 = {n: rp0.section_pick(n) for n in (1, 2)}
        first = self.real.read_text(encoding="utf-8")
        self.assertIn(f"**{B1} {B2}** {B_LOOP} {B_MINE} **{B3}**", first)
        self.assertIn(f"{D_LOOP} {D_MINE} {PICK_D}", first)                       # the mixed paragraphs
        self.assertIn(f"{E_LOOP} {E_MINE} {PICK_E}", first)
        self.real.write_text(first.replace(f"**{B3}**", f"**{B_NEW}**"), encoding="utf-8")   # the author's rewrite
        edited = self.real.read_text(encoding="utf-8")
        def claude_scan(text):                 # the chat-log scan can't find what you typed in the guide
            from engine.draws import sents
            return [(x, "claude", 0.0, 1.0) for p in text.split("\n\n") for x in sents(p)]
        r = sync(self.d, self.real, write=True, classify=claude_scan, today="2026-11-16")
        self.assertEqual([t for _, t in r["your_edits"]], [f"{B1} {B2} {B_LOOP} {B_MINE} {B_NEW}"])
        verdicts = Job(root=self.d).verdicts()
        self.assertEqual([verdicts.get(x) for x in ("I rewrote the bravo three sentence myself.", B_SHORT, B_LOOP)],
                         ["yours", "yours", "loop"])                          # typed by you, short one included
        self.assertTrue(r["retired"])                                     # the cut moved: labels after it shift
        job = Job(root=self.d)
        self.assertNotIn("**", job.original.read_text(encoding="utf-8"))
        self.assertIn(D_LOOP, job.working.read_text(encoding="utf-8"))   # wrong 3: the loop's text stays standing
        passed, rep, rp = self.night("2026-11-16 02:10")
        after = self.real.read_text(encoding="utf-8")
        # wrong 2: the picks followed their spans, the repair stands, the same picks are served
        self.assertEqual(rp.stale_repairs, [])
        self.assertEqual(len(rp.repairs), 1)
        moved = {lab: rp.orig_text[lab] for lab in rp.section_pick(1)["picks"]}
        self.assertEqual(sorted(rp.section_pick(1)["picks"].values()), sorted(served0[1]["picks"].values()))
        self.assertEqual(sorted(moved.values()), sorted([f"{A1} {A2}", f"{C1} {C2}", E2]))
        # the cut moved C into E's old window: "s1w1 p1" was E's tail and is C's now, and E's pick never lands on it
        self.assertEqual({lab: rp.orig_text[lab] for lab in ("s1w1 p1", "s1w1 p2")}, {"s1w1 p1": f"{C1} {C2}",
                                                                                     "s1w1 p2": E2})
        self.assertEqual((rp.section_pick(1)["picks"]["s1w1 p1"], rp.section_pick(1)["picks"]["s1w1 p2"]),
                         (PICK_C, PICK_E))
        self.assertFalse(any(B_SHORT in o or "bravo three" in o for o in rp.orig_text.values()))   # yours: no unit
        self.assertEqual(len(rp.section_pick(1)["picks"]), 3)
        self.assertEqual(rp.section_pick(1)["scoring"]["label"], "s1 round 2")
        # wrong 3: section 2 serves exactly what it served, and its blocks keep every byte
        self.assertEqual(rp.section_pick(2)["text"], served0[2]["text"])
        self.assertEqual(self.section(after, 2), self.section(edited, 2))
        self.assertIn(f"{E_LOOP} {E_MINE} {PICK_E}", after)          # E's window was retired: its loop text stands
        self.assertIn("s1w1", r["retired"])
        # wrong 1: what is left of the stuck span stays in bold under its prompt; the rewrite is plain
        self.assertIn(f"-->\n\n**{B1} {B2}** {B_LOOP} {B_MINE} {B_NEW}\n", after.replace("\r\n", "\n"))
        self.assertIn("why it's yours: " + ST1["why"], after)
        self.assertNotIn(ST2["why"], after)                         # the rewritten sentence's entry speaks for nothing
        self.assertEqual(passed["unmatched"], ["s1w0 p2"])
        # section 1 changes only in the rewritten paragraph, its marker and the section note
        b_before = [x for x in self.section(edited, 1).split("\n\n")]
        b_after = [x for x in self.section(after, 1).split("\n\n")]
        changed = [x for x in b_after if x not in b_before]
        self.assertTrue(all(B_NEW in x or x.startswith("<!-- depangram") for x in changed), changed)
        self.assertEqual(len(changed), 3)                  # the note, the paragraph's marker, the paragraph
        # the second night changes no byte
        sync(self.d, self.real, write=True, classify=claude_scan, today="2026-11-17")
        frozen = Job(root=self.d).original.read_text(encoding="utf-8")
        self.night("2026-11-17 02:10")
        self.assertEqual(self.real.read_text(encoding="utf-8"), after)                          # pass
        self.assertEqual(Job(root=self.d).original.read_text(encoding="utf-8"), frozen)
        self.assertEqual(r["carried"], 2)                                # D and E, the blocks the loop changed


def git(*args, cwd):
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout.strip()


REL = "docs/the guide.md"
IDENT = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t",
         "GIT_COMMITTER_EMAIL": "t@example.com"}


class MainCopy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patch = mock.patch.dict(os.environ, IDENT)
        patch.start()
        self.addCleanup(patch.stop)
        root = Path(self.tmp.name)
        self.origin, self.work, self.yours = root / "origin.git", root / "work", root / "yours"
        git("init", "--bare", "-b", "main", str(self.origin), cwd=root)
        git("clone", "-c", "core.autocrlf=false", str(self.origin), str(self.work), cwd=root)
        git("config", "core.autocrlf", "false", cwd=self.work)
        self.put(self.work, "v1\n", "the guide")
        (self.work / "other.txt").write_text("main's own file\n", encoding="utf-8")
        git("add", "-A", cwd=self.work)
        git("commit", "-m", "another file", cwd=self.work)
        git("push", "origin", "HEAD:main", cwd=self.work)
        git("checkout", "-b", "side", cwd=self.work)
        (self.work / "side only.txt").write_text("the side branch's code\n", encoding="utf-8")
        self.put(self.work, "v2, depangrammed\n", "the write-back")
        git("clone", "-c", "core.autocrlf=false", str(self.origin), str(self.yours), cwd=root)
        git("config", "core.autocrlf", "false", cwd=self.yours)

    def put(self, repo, text, message):
        p = repo / REL
        p.parent.mkdir(exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
        git("add", "-A", cwd=repo)
        git("commit", "-m", message, cwd=repo)

    def edit_on_main(self, text):
        git("pull", "origin", "main", cwd=self.yours)
        self.put(self.yours, text, "your edit")
        git("push", "origin", "HEAD:main", cwd=self.yours)

    def test_publish_puts_this_branch_s_copy_on_main_and_nothing_else(self):
        from engine.main_copy import publish, status
        self.assertEqual(status(self.work, REL)["state"], "main behind")
        out = publish(self.work, REL)
        self.assertIn("published to origin/main", out)
        self.assertEqual(git("show", f"origin/main:{REL}", cwd=self.work), "v2, depangrammed")
        self.assertEqual(git("ls-tree", "--name-only", "-r", "origin/main", cwd=self.work).split("\n"),
                         [REL, "other.txt"])                                     # the side branch's other file stayed home
        self.assertEqual(git("rev-list", "--count", "origin/main", cwd=self.work), "3")       # one commit on top
        self.assertEqual(git("rev-parse", "main", cwd=self.work), git("rev-parse", "origin/main", cwd=self.work))
        self.assertEqual(git("rev-parse", "--abbrev-ref", "HEAD", cwd=self.work), "side")     # no branch switch
        self.assertEqual(status(self.work, REL)["state"], "same")
        self.assertIn("nothing to publish", publish(self.work, REL))

    def test_publish_refuses_over_an_edit_of_yours_and_over_uncommitted_text(self):
        from engine.main_copy import Refused, publish, status
        publish(self.work, REL)
        self.edit_on_main("v3, your edit\n")
        self.assertEqual(status(self.work, REL)["state"], "main ahead")
        with self.assertRaises(Refused):
            publish(self.work, REL)                                               # flag: main has text this branch never had
        self.assertEqual(git("show", f"origin/main:{REL}", cwd=self.work), "v3, your edit")
        (self.work / REL).write_text("scribble\n", encoding="utf-8", newline="\n")
        with self.assertRaises(Refused):
            publish(self.work, REL)                                               # flag: not committed here

    def test_pull_brings_your_edit_over_and_holds_when_both_moved(self):
        from engine.main_copy import Refused, pull, status
        from engine.main_copy import publish
        self.assertIn("nothing to pull", pull(self.work, REL))                    # main is the older one
        publish(self.work, REL)
        self.edit_on_main("v3, your edit\n")
        self.assertIn("main's copy is in this branch's file", pull(self.work, REL))   # pass
        self.assertEqual((self.work / REL).read_text(encoding="utf-8"), "v3, your edit\n")
        self.assertTrue(status(self.work, REL)["dirty"])
        git("commit", "-am", "your edit, pulled", cwd=self.work)
        self.assertEqual(status(self.work, REL)["state"], "same")
        self.put(self.work, "v4, another write-back\n", "the write-back")
        self.edit_on_main("v5, another edit of yours\n")
        self.assertEqual(status(self.work, REL)["state"], "diverged")
        with self.assertRaises(Refused):
            pull(self.work, REL)                                                  # flag: both moved
        pull(self.work, REL, take_main=True)
        self.assertEqual((self.work / REL).read_text(encoding="utf-8"), "v5, another edit of yours\n")


class DryRunWritesNothing(unittest.TestCase):
    """A dry write-back (write=False) on a job with no regrow origs.json leaves the job folder byte for byte as it
    was: the pass is read read only, so not even the sidecar is written."""

    def test_a_dry_write_back_leaves_the_job_folder_as_it_was(self):
        import shutil
        fixture = Path(__file__).resolve().parent / "fixtures" / "balcony"
        with tempfile.TemporaryDirectory() as d:
            job_dir = Path(d) / "balcony"
            shutil.copytree(fixture, job_dir)
            draw = {"kind": "draw", "label": "s1w0 p2", "arm": "notes", "k": 0, "text": "A draw."}
            (job_dir / "regrow bank.jsonl").write_text(json.dumps(draw) + "\n", encoding="utf-8")

            def snapshot():
                return {p.relative_to(job_dir).as_posix(): p.read_bytes()
                        for p in sorted(job_dir.rglob("*")) if p.is_file()}
            before = snapshot()
            job = Job(root=job_dir)
            report, _, changed = run(job, write=False, stamp="2026-11-16 02:00", pangram=object())
            self.assertFalse(report["written"])
            self.assertFalse((job_dir / "regrow origs.json").exists())             # flag: the sidecar
            self.assertEqual(snapshot(), before)                                    # pass: every byte as it was


if __name__ == "__main__":
    unittest.main()
