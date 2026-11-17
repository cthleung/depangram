"""Tests for the repair step (engine/repair.py): the prompts, the answer parser, the code gate on the moves (a flag case
and a pass case per rule), the prompt for the author, and whole runs over a two-paragraph pass with Claude and Pangram
faked: apply, propose, dry, off, out of money, the stuck rule, and a veto beside a repair that leaves it standing.
No model, no Pangram, no Claude. Run from the repo root:

    python -m unittest discover tests
"""
import json
import re
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from engine import repair
from engine.job import NOTES_VERSION, sha
from engine.regrow import REGROW_FACT_PROMPT, RegrowPass
from test_regrow import FakePangram, fake_load_units, fake_section_text, scoring

DASH = "\u2014"
ORIG = ("Moreau et al. (2022) timed 12,904 tram stops. Doors stayed open past the schedule 64% of the time at the market"
        " stop. Elsewhere it happened 23% of the time. So the market stop sets the pace for the line.")
PICK = ("Moreau et al. (2022) timed 12,904 tram stops. At the market stop the doors overran 64% of the time."
        " Other stops managed 23%. Nobody ever seems to mind the wait.")


def moves(*ms, **kw):
    return repair.apply_moves(list(ms), PICK, ORIG, **kw)


class Prompts(unittest.TestCase):
    def test_the_five_kinds_come_from_the_pass_s_fact_check_word_for_word_and_a_sixth_is_added(self):
        five = re.findall(r"^[1-5]\. .*$", REGROW_FACT_PROMPT, re.M)
        self.assertEqual(len(five), 5)
        for line in five:
            self.assertIn(line, repair.KINDS)                                                       # pass: reused
        self.assertIn("6. A dropped point that matters", repair.KINDS)
        self.assertIn('"Step two is where..."', repair.KINDS)
        for prompt in (repair.READ_PROMPT, repair.CHECK_PROMPT):
            text = repair.fill(prompt, ORIG, ["a note"], PICK, repair.DEFAULTS)
            self.assertNotIn("a dropped point do not count", text)                                 # flag: the old tail
            self.assertNotIn("<<", text)
        read = repair.fill(repair.READ_PROMPT, ORIG, [], PICK, repair.DEFAULTS)
        for piece in ("CLEAN", "UNFIXABLE", '"move": "splice"', "At most 2 splices", "up to 3 words", "(no notes)"):
            self.assertIn(piece, read)
        self.assertIn("Reply with NONE", repair.CHECK_PROMPT)


class Answers(unittest.TestCase):
    def test_clean_unfixable_and_a_list_of_moves_parse_and_anything_else_is_bad(self):
        self.assertEqual(repair.parse_answer(" CLEAN\n"), ("clean", None))                         # pass
        self.assertEqual(repair.parse_answer("UNFIXABLE: the point is gone"), ("unfixable", "the point is gone"))
        cut = [{"move": "cut", "sentence": "X.", "why": "invented"}]
        self.assertEqual(repair.parse_answer(json.dumps(cut)), ("moves", cut))
        self.assertEqual(repair.parse_answer("```json\n" + json.dumps(cut) + "\n```"), ("moves", cut))   # pass: fenced
        for bad in ("Here are the moves: " + json.dumps(cut), json.dumps({"move": "cut"}), "[]", "[1, 2]"):
            self.assertEqual(repair.parse_answer(bad)[0], "bad")                                    # flag


class Gate(unittest.TestCase):
    def test_a_cut_names_exactly_one_sentence_of_the_rewrite(self):
        text, why = moves({"move": "cut", "sentence": "Nobody ever seems to mind the wait.", "why": "invented"})
        self.assertIsNone(why)                                                                      # pass
        self.assertTrue(text.endswith("Other stops managed 23%."))
        self.assertIn("names no sentence", moves({"move": "cut", "sentence": "Not in it."})[1])     # flag
        twice = PICK + " Other stops managed 23%."
        self.assertIn("twice", repair.apply_moves([{"move": "cut", "sentence": "Other stops managed 23%."}], twice, ORIG)[1])

    def test_a_swap_holds_the_fix_pass_s_rules(self):
        s = "Other stops managed 23%."
        ok, why = moves({"move": "swap", "sentence": s, "to": "Other stops managed only 23%."})
        self.assertIsNone(why)                                                                      # pass: one token
        self.assertIn("Other stops managed only 23%.", ok)
        self.assertIn("(the limit is 3)", moves({"move": "swap", "sentence": s,
                                                 "to": "Other stops did it only 23% of the time."})[1])  # flag
        self.assertIn("adds a number", moves({"move": "swap", "sentence": s, "to": "Other stops managed 26%."})[1])
        self.assertIn("adds a name", moves({"move": "swap", "sentence": s, "to": "Other stops and Lyon managed 23%."})[1])
        self.assertIn("em dash", moves({"move": "swap", "sentence": s, "to": f"Other stops {DASH} 23%."})[1])
        self.assertIn("adds a swear", moves({"move": "swap", "sentence": s, "to": "Other fucking stops managed 23%."})[1])
        self.assertIn("same sentence", moves({"move": "swap", "sentence": s, "to": s})[1])

    def test_a_splice_is_one_sentence_of_the_original_word_for_word_and_two_at_most(self):
        cut = {"move": "cut", "sentence": "Nobody ever seems to mind the wait."}
        back = "So the market stop sets the pace for the line."
        text, why = moves(cut, {"move": "splice", "after": "Other stops managed 23%.", "sentence": back})
        self.assertIsNone(why)                                                                      # pass
        self.assertTrue(text.endswith("Other stops managed 23%. " + back))
        text, why = moves({"move": "splice", "after": "START", "sentence": "Elsewhere it happened 23% of the time."}, cut)
        self.assertTrue(text.startswith("Elsewhere it happened 23% of the time. Moreau"))                 # pass: at the start
        self.assertIn("word for word", moves({"move": "splice", "after": "START",
                                              "sentence": "So the market stop sets the pace."})[1])  # flag: not exact
        three = [{"move": "splice", "after": "START", "sentence": x} for x in repair.sentences(ORIG)[1:]]
        self.assertIn("more than 2 splices", moves(*three)[1])                                      # flag

    def test_an_unknown_move_a_stray_field_or_two_moves_on_one_sentence_fail_the_proposal(self):
        s = "Nobody ever seems to mind the wait."
        self.assertIn("not a move", moves({"move": "rewrite", "sentence": s})[1])                   # flag
        self.assertIn("fields", moves({"move": "cut", "sentence": s, "to": "x"})[1])                # flag
        self.assertIn("fields", moves({"move": "swap", "sentence": s})[1])                          # flag
        self.assertIn("two moves", moves({"move": "cut", "sentence": s}, {"move": "swap", "sentence": s, "to": "x."})[1])
        self.assertIn("the original has", repair.apply_moves([{"move": "cut", "sentence": "Elsewhere it happened 23% of the time."}],
                                                             PICK + " Elsewhere it happened 23% of the time.", ORIG)[1])  # flag

    def test_a_result_made_mostly_of_the_original_is_no_repair_unless_the_span_is_short(self):
        pick = "Moreau et al. (2022) timed 12,904 tram stops. They saw the doors overrun 64% of the time."
        to_orig = [{"move": "cut", "sentence": "They saw the doors overrun 64% of the time."},
                   {"move": "splice", "after": "START", "sentence": "Doors stayed open past the schedule 64% of the time at the market stop."},
                   {"move": "splice", "after": "START", "sentence": "Elsewhere it happened 23% of the time."}]
        self.assertIn("no repair", repair.apply_moves(to_orig, pick, ORIG)[1])                     # flag: 19 of 26 words
        short_orig, short_pick = "Let's start with the easy part.", "The easy part comes first. It's quick."
        back = [{"move": "cut", "sentence": "The easy part comes first."}, {"move": "cut", "sentence": "It's quick."},
                {"move": "splice", "after": "START", "sentence": short_orig}]
        self.assertEqual(repair.apply_moves(back, short_pick, short_orig), (short_orig, None))     # pass: short span

    def test_an_after_may_name_a_run_of_sentences_the_splitter_cut(self):
        orig = "Step one is the dough. Step two is where it rests. Not even close to done."
        pick = "Step one is the dough, more or less. Not. Even. Close. It goes on and on for ages."
        self.assertEqual(repair.sentences(pick)[1:4], ["Not.", "Even.", "Close."])         # the splitter's three
        back = {"move": "splice", "after": "Not. Even. Close.", "sentence": "Step two is where it rests."}
        text, why = repair.apply_moves([back], pick, orig)
        self.assertIsNone(why)                                                                      # pass: the real case
        self.assertEqual(text, "Step one is the dough, more or less. Not. Even. Close. Step two is where it rests."
                               " It goes on and on for ages.")
        self.assertIn("names no sentence", repair.apply_moves([dict(back, after="Not. Even.  Close")], pick, orig)[1])

    def test_a_cut_may_take_a_run_and_each_sentence_of_it_follows_the_rules(self):
        text, why = moves({"move": "cut", "sentence": "Other stops managed 23%.  Nobody ever seems to mind the wait."})
        self.assertIsNone(why)                                                                      # pass: both go
        self.assertTrue(text.endswith("64% of the time."))
        pick = PICK + " Elsewhere it happened 23% of the time."
        self.assertIn("the original has", repair.apply_moves(
            [{"move": "cut", "sentence": "Nobody ever seems to mind the wait. Elsewhere it happened 23% of the time."}],
            pick, ORIG)[1])                                                                         # flag: one is the original's
        self.assertIn("two moves", moves({"move": "cut", "sentence": "Other stops managed 23%. Nobody ever seems to mind"
                                                                     " the wait."},
                                         {"move": "swap", "sentence": "Other stops managed 23%.",
                                          "to": "Other stops managed only 23%."})[1])                # flag: touched twice

    def test_a_run_found_twice_fails_and_a_swap_on_a_run_is_refused(self):
        twice = "Not. Even. Close. And then. Not. Even. Close."
        self.assertIn("twice", repair.apply_moves([{"move": "cut", "sentence": "Not. Even. Close."}], twice, "Fine.")[1])
        self.assertIn("twice", repair.apply_moves([{"move": "splice", "after": "Not. Even.", "sentence": "Fine."}],
                                                  twice, "Fine.")[1])                               # flag
        two = "Other stops managed 23%. Nobody ever seems to mind the wait."
        self.assertIn("a swap names 2 sentences, and a swap changes one",
                      moves({"move": "swap", "sentence": two, "to": two.replace("23%", "only 23%")})[1])  # flag

    def test_the_moves_line_reads_as_the_reviewer_writes_them(self):
        line = repair.moves_line([{"move": "cut", "sentence": " A. "}, {"move": "swap", "sentence": "B.", "to": "C."},
                                  {"move": "splice", "after": "START", "sentence": "D."}])
        self.assertEqual(line, 'cut: "A."; swap: "B." to "C."; splice: "D."')


class AuthorPrompt(unittest.TestCase):
    def test_the_prompt_comes_from_the_notes_without_the_keep_line(self):
        notes = ['Cite "Moreau et al. (2022)".', "The market stop sets the pace.", 'keep: "probably"', "Doors overran 64% of the time."]
        got = repair.author_prompt(notes, ORIG)
        self.assertEqual(got["about"], notes[0])
        self.assertEqual(got["point"], " ".join([notes[0], notes[1], notes[3]]))                   # flag: keep: is out
        self.assertEqual(got["fact"], notes[0] + " " + notes[3])                                    # the plain note isn't a fact
        first = "Moreau et al. (2022) timed 12,904 tram stops."
        self.assertEqual(repair.author_prompt([], ORIG), {"about": first, "point": first, "fact": first})  # pass: no notes
        self.assertEqual(repair.author_prompt([], "so it goes on.")["fact"], "")


class FakeClaude:
    """Answers by the call's role and the rewrite in the prompt; an unknown one is CLEAN or NONE. A retry with no answer
    of its own gets the read's answer again. A list of answers is handed out in order. Costs $0.10 a call (or the
    list `cost` in order). The prompts are kept in `prompts`."""

    def __init__(self, answers=None, cost=0.10):
        self.answers, self.cost, self.calls, self.prompts = answers or {}, cost, [], []

    def __call__(self, settings, conn, prompt, role, model, max_budget_usd):
        new = prompt.split("Rewrite:\n", 1)[1].split("\n\n", 1)[0]
        self.calls.append((role, new, model))
        self.prompts.append(prompt)
        default = "CLEAN" if role in ("repair read", "repair retry") else "NONE"
        got = self.answers.get((role, new))
        if got is None and role == "repair retry":
            got = self.answers.get(("repair read", new))
        if isinstance(got, list):
            got = got.pop(0)
        cost = self.cost.pop(0) if isinstance(self.cost, list) else self.cost
        return (default if got is None else got), {"cost_usd": cost}


NOTES_ONE = ["Start with the first point.", 'Quote "One." exactly.']


class Runs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        cache = {sha(NOTES_VERSION + "One."): NOTES_ONE}
        self.job = types.SimpleNamespace(dir=self.d, sections=[5], name="t", bank_path=self.d / "bank.jsonl",
                                         _cache=lambda: cache)
        for p in (mock.patch.object(RegrowPass, "load_units", fake_load_units),
                  mock.patch.object(RegrowPass, "section_text", fake_section_text)):
            p.start()
            self.addCleanup(p.stop)

    def bank(self, *entries):
        with (self.d / "regrow bank.jsonl").open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    def review(self, data=None, crlf=True):
        text = json.dumps(data or {"vetoes": [], "repairs": [], "for_you": []}, indent=1, ensure_ascii=False)
        (self.d / "review.json").write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))

    def run_(self, claude=None, pangram=None, mode=None, **cfg):
        self.claude = claude or FakeClaude()
        self.pg = pangram or FakePangram([{"ai": 0.20, "segs": []}] * 4)
        settings = types.SimpleNamespace(config={"repair": cfg})
        return repair.run(self.job, settings, None, mode=mode, claude=self.claude, pangram=self.pg, today="2026-11-15",
                          log=lambda *_: None)

    def saved(self):
        return json.loads((self.d / "review.json").read_text(encoding="utf-8"))

    CUT = json.dumps([{"move": "cut", "sentence": "Extra claim.", "why": "the original doesn't say it"}])

    def test_apply_writes_a_repair_logs_both_reads_and_scores_the_changed_section_once(self):
        keep = {"label": "old", "orig": "Gone.", "text": "Ida.", "why": "the reviewer's"}
        self.review({"vetoes": [keep], "repairs": [], "for_you": []})
        self.bank(scoring("s5 round 2", 0.30, {"s5w0 p1": "Un, you know. Extra claim.", "s5w0 p2": "Dos."}))
        res = self.run_(FakeClaude({("repair read", "Un, you know. Extra claim."): self.CUT}))
        self.assertEqual(res["counts"], {"repaired": 1, "clean": 1})
        saved = self.saved()
        self.assertEqual(saved["vetoes"], [keep])                                                   # pass: kept, in place
        self.assertEqual(saved["repairs"], [{"label": "s5w0 p1", "orig": "One.", "from": "Un, you know. Extra claim.",
                                             "text": "Un, you know.", "moves": 'cut: "Extra claim."', "by": "opus",
                                             "when": "2026-11-15", "scoring": "s5 round 2"}])     # the scoring it read
        self.assertIn(b"\r\n", (self.d / "review.json").read_bytes())                               # its line endings held
        roles = [c[0] for c in self.claude.calls]
        self.assertEqual(sorted(roles), ["repair check", "repair read", "repair read"])
        self.assertIn(("repair check", "Un, you know.", "opus"), self.claude.calls)
        self.assertEqual(len(repair.read_log(self.d)), 2)
        self.assertEqual(self.pg.calls, ["regrow s5 final"])                                        # flag: changed, scored
        self.assertEqual(res["rescored"], {5: {"ai": 0.20, "label": "s5 final", "humanized": False}})
        self.assertAlmostEqual(res["claude_usd"], 0.30)
        lines = repair.report_lines(res)
        self.assertIn('- s5w0 p1, repaired: cut: "Extra claim."', lines)
        self.assertIn("  before: Un, you know. Extra claim.", lines)
        self.assertIn("  after: Un, you know.", lines)
        self.assertIn("Section 5 scored again as served: 0.20 (s5 final)", lines)
        again = self.run_()                                                                         # pass: paid once
        self.assertEqual((self.claude.calls, again["spans"], again["seen"], self.pg.calls), ([], [], 1, []))

    def test_a_failed_span_is_vetoed_and_a_second_failure_on_another_pick_goes_to_the_author(self):
        self.review()
        self.bank(scoring("s5 round 2", 0.20, {"s5w0 p1": "Uno invented."}),
                  scoring("s5 round 3", 0.30, {"s5w0 p1": "Un invented."}))
        no = {("repair read", "Uno invented."): "UNFIXABLE: the point is gone",
              ("repair read", "Un invented."): json.dumps([{"move": "cut", "sentence": "Not there."}])}
        res = self.run_(FakeClaude(no))
        self.assertEqual([(s["label"], s["outcome"]) for s in res["spans"]],
                         [("s5w0 p1", "vetoed"), ("s5w0 p1", "for you")])                           # flag, then stuck
        saved = self.saved()
        self.assertEqual(saved["vetoes"], [{"label": "s5w0 p1", "orig": "One.", "text": "Uno invented.",
                                            "why": "unfixable: the point is gone", "scoring": "s5 round 2"}])
        fy, = saved["for_you"]
        self.assertEqual(fy["scoring"], "s5 round 3")         # no pins, so the veto moved the best: the re-read kept
        self.assertEqual((fy["text"], fy["about"], fy["point"], fy["fact"]),
                         ("One.", NOTES_ONE[0], " ".join(NOTES_ONE), NOTES_ONE[1]))
        self.assertEqual(fy["went_wrong"], "the gate said no: a cut names no sentence of the rewrite (on the second"
                                           " try)")                     # asked twice, the same answer twice
        self.assertEqual(self.pg.calls, ["regrow s5 final"])                                        # the original, scored

    def test_a_veto_and_a_repair_written_against_the_same_scoring_both_hold(self):
        self.review()
        self.bank(scoring("s5 round 2", 0.20, {"s5w0 p1": "A1 bad.", "s5w0 p2": "B1, you know. Extra claim."}),
                  scoring("s5 round 3", 0.30, {"s5w0 p1": "A2.", "s5w0 p2": "B2."}))
        res = self.run_(FakeClaude({("repair read", "A1 bad."): "UNFIXABLE: wrong",
                                    ("repair read", "B1, you know. Extra claim."): self.CUT}))
        read = [c[1] for c in self.claude.calls if c[0] == "repair read"]
        self.assertEqual(read, ["A1 bad.", "B1, you know. Extra claim."])          # pass: nothing moved, no re-read
        self.assertEqual(res["stale"], [])                                                          # the repair holds
        saved = self.saved()
        self.assertEqual((saved["vetoes"][0]["scoring"], saved["repairs"][0]["scoring"]), ("s5 round 2", "s5 round 2"))
        rp = repair.load_pass(self.job, None, None, [5], log=lambda *_: None)
        self.assertEqual(rp.section_pick(5)["text"], "One. B1, you know.")         # the vetoed unit served standing
        self.assertEqual(self.pg.calls, ["regrow s5 final"])

    def test_propose_writes_nothing_to_review_json_and_apply_replays_it_for_free(self):
        self.review()
        before = (self.d / "review.json").read_bytes()
        self.bank(scoring("s5 round 2", 0.30, {"s5w0 p1": "Un, you know. Extra claim."}))
        answers = {("repair read", "Un, you know. Extra claim."): self.CUT}
        res = self.run_(FakeClaude(answers), mode="propose")
        self.assertEqual(((self.d / "review.json").read_bytes(), self.pg.calls), (before, []))     # flag: nothing written
        self.assertIn("Proposals only", "\n".join(repair.report_lines(res)))
        self.assertIn("would be repaired", repair.report_lines(res)[0])
        self.assertEqual([r["mode"] for r in repair.read_log(self.d)], ["propose"])
        self.assertEqual(self.run_(mode="propose")["seen"], 1)                                      # not read twice
        dry = self.run_(mode="dry")
        self.assertEqual([(t["label"], t["free"]) for t in dry["to_read"]], [("s5w0 p1", True)])  # a replay, listed free
        res = self.run_()                                                                           # pass: apply replays it
        self.assertEqual((self.claude.calls, res["counts"], res["spans"][0]["replayed"]), ([], {"repaired": 1}, True))
        self.assertEqual(self.saved()["repairs"][0]["text"], "Un, you know.")
        self.assertEqual([r["mode"] for r in repair.read_log(self.d)], ["propose", "apply"])

    def test_out_of_money_leaves_the_rest_unread_and_says_so(self):
        self.bank(scoring("s5 round 2", 0.30, {"s5w0 p1": "Uno.", "s5w0 p2": "Dos."}))
        res = self.run_(max_claude_usd=0.05)
        self.assertEqual((len(self.claude.calls), res["unread"], res["unread_why"]), (1, 1, "the Claude ceiling"))  # flag
        self.assertIn("1 span went unread (the Claude ceiling)", "\n".join(repair.report_lines(res)))
        self.assertEqual(len(repair.read_log(self.d)), 1)
        res = self.run_(max_claude_usd=1.0)                                                         # pass: the rest, later
        self.assertEqual((len(self.claude.calls), res["unread"], res["seen"]), (1, 0, 1))

    def test_a_span_left_unread_is_counted_once_across_the_rounds(self):
        self.bank(scoring("s5 round 2", 0.30, {"s5w0 p1": "Un, you know. Extra claim.", "s5w0 p2": "Dos."}))
        res = self.run_(FakeClaude({("repair read", "Un, you know. Extra claim."): self.CUT}), max_claude_usd=0.15)
        self.assertEqual((res["counts"], res["written"]), ({"repaired": 1}, True))                 # then a second round
        self.assertEqual(res["unread"], 1)                                                          # flag: p2, once

    def test_a_veto_beside_the_reviewer_s_older_repair_leaves_the_repair_standing(self):
        mine = {"label": "s5w0 p2", "orig": "Two.", "from": "B1, you know.", "text": "B1.", "moves": "cut", "by": "reviewer",
                "when": "2026-11-14"}                                 # an older entry: no `scoring`
        self.review({"vetoes": [], "repairs": [mine], "for_you": []})
        self.bank(scoring("s5 round 2", 0.20, {"s5w0 p1": "A1 bad.", "s5w0 p2": "B1, you know."}),
                  scoring("s5 round 3", 0.30, {"s5w0 p1": "A2.", "s5w0 p2": "B2."}))
        res = self.run_(FakeClaude({("repair read", "A1 bad."): "UNFIXABLE: wrong"}))
        self.assertEqual([c[1] for c in self.claude.calls], ["A1 bad."])        # pass: B1 is pinned, nothing moved
        self.assertEqual(res["stale"], [])                                       # flag before the review base: stale
        self.assertNotIn("went stale", "\n".join(repair.report_lines(res)))

    def test_a_claude_error_stops_the_reads_and_logs_nothing_for_that_span(self):
        self.bank(scoring("s5 round 2", 0.30, {"s5w0 p1": "Uno.", "s5w0 p2": "Dos."}))

        def broken(*a, **k):
            raise RuntimeError("the five-hour cap")
        res = self.run_(broken)
        self.assertEqual((res["unread"], repair.read_log(self.d)), (2, []))                         # flag
        self.assertIn("Claude said no: RuntimeError: the five-hour cap", res["unread_why"])

    def test_a_dry_run_lists_the_spans_and_calls_and_writes_nothing(self):
        self.bank(scoring("s5 round 2", 0.30, {"s5w0 p1": "Uno.", "s5w0 p2": "Dos."}))
        res = self.run_(mode="dry")
        self.assertEqual(([t["label"] for t in res["to_read"]], self.claude.calls, self.pg.calls),
                         (["s5w0 p1", "s5w0 p2"], [], []))
        self.assertEqual(sorted(p.name for p in self.d.iterdir()), ["regrow bank.jsonl"])          # flag: not even the sidecar
        self.assertEqual(repair.report_lines(res)[1], "- s5w0 p1 (section 5): Uno.")

    def folder(self):
        return {p.name: p.read_bytes() for p in self.d.iterdir()}

    def test_reread_reads_logged_spans_again_prints_what_opus_said_and_writes_nothing(self):
        self.review()
        self.bank(scoring("s5 round 2", 0.30, {"s5w0 p1": "Un, you know. Extra claim.", "s5w0 p2": "Dos."}))
        answers = {("repair read", "Un, you know. Extra claim."): self.CUT}
        self.run_(FakeClaude(answers), mode="propose")                       # both read and logged once
        before = self.folder()
        claude = FakeClaude(answers, cost=0.12)
        settings = types.SimpleNamespace(config={"repair": {}})
        res = repair.reread(self.job, settings, None, ["s5w0 p1", "s5w0 p2"], claude=claude, log=lambda *_: None)
        self.assertEqual([c[0] for c in claude.calls], ["repair read", "repair check", "repair read"])  # flag: read again
        self.assertEqual(self.folder(), before)                              # pass: not a byte written in the job folder
        self.assertAlmostEqual(res["claude_usd"], 0.36)
        text = "\n".join(repair.reread_lines(res))
        self.assertIn("--reread, propose mode, nothing written: 2 spans read, Claude $0.36 of $3.00.", text)
        self.assertIn("s5w0 p1 (section 5): repair read $0.12, repair check $0.12", text)
        self.assertIn('  moves: cut: "Extra claim."', text)
        self.assertIn("  the code gate: let them through", text)
        self.assertIn("  the check: NONE", text)
        self.assertIn("  after: Un, you know.", text)
        self.assertIn("s5w0 p2 (section 5): repair read $0.12", text)
        self.assertIn("  clean: the step would log it and write nothing", text)

    def test_reread_says_what_the_gate_turned_down_and_lists_labels_it_can_t_read(self):
        fix = {"label": "s5w0 p2", "orig": "Two.", "text": "Dos, by hand.", "went_wrong": "x", "about": "y", "point": "z",
               "fact": "", "when": "2026-11-14"}
        self.review({"vetoes": [], "repairs": [], "for_you": [fix]})
        self.bank(scoring("s5 round 2", 0.30, {"s5w0 p1": "Uno.", "s5w0 p2": "Dos, by hand."}))
        claude = FakeClaude({("repair read", "Uno."): json.dumps([{"move": "cut", "sentence": "Not there."}])})
        settings = types.SimpleNamespace(config={"repair": {}})
        res = repair.reread(self.job, settings, None, ["s5w0 p1", "s5w0 p2", "s5w9 p1"], claude=claude,
                            log=lambda *_: None)
        text = "\n".join(repair.reread_lines(res))
        self.assertIn("  the code gate: a cut names no sentence of the rewrite (on the second try), so no", text)
        self.assertIn("  not usable, so it was asked once more: the gate said no: a cut names no sentence", text)
        self.assertIn("- s5w0 p2: not read, pinned by a hand fix for you", text)
        self.assertIn("- s5w9 p1: not read, no such span in those sections", text)
        self.assertEqual([c[0] for c in claude.calls], ["repair read", "repair retry"])                  # no check
        res = repair.reread(self.job, types.SimpleNamespace(config={"repair": {"max_claude_usd": 0.0}}), None,
                            ["s5w0 p1"], claude=claude, log=lambda *_: None)
        self.assertEqual(res["unread"], [("s5w0 p1", "the Claude ceiling")])                            # the ceiling holds

    def test_reread_reads_a_span_a_standing_repair_pins_from_the_pick_it_mended(self):
        mine = {"label": "s5w0 p2", "orig": "Two.", "from": "B1, you know.", "text": "B1.", "moves": 'cut: "you know"',
                "by": "reviewer", "when": "2026-11-14"}
        self.review({"vetoes": [], "repairs": [mine], "for_you": []})
        self.bank(scoring("s5 round 2", 0.20, {"s5w0 p1": "A1.", "s5w0 p2": "B1, you know."}))
        before = self.folder()
        claude = FakeClaude({("repair read", "B1, you know."): "CLEAN"})
        res = repair.reread(self.job, types.SimpleNamespace(config={"repair": {}}), None, ["s5w0 p2"], claude=claude,
                            log=lambda *_: None)
        self.assertEqual(claude.calls, [("repair read", "B1, you know.", "opus")])              # pass: read though pinned
        self.assertEqual(self.folder(), before)
        text = "\n".join(repair.reread_lines(res))
        self.assertIn("  the repair that stands (by reviewer, 2026-11-14): cut: \"you know\"", text)
        self.assertIn("  its text: B1.", text)
        self.assertIn("s5w0 p2 (section 5): repair read $0.10", text)

    def test_reread_refuses_apply_and_takes_only_labels(self):
        for argv in (["--reread", "s5w0 p1", "--mode", "apply"], ["--reread", "s5w0 p1", "--dry"],
                     ["--reread", "s5w0 p1", "--section", "5"], ["--reread", "s5w0"]):
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                repair.main(argv)                                                                       # flag
        got = []
        with mock.patch.object(repair, "reread", lambda job, settings, conn, labels: got.append(labels) or
                               {"spans": [], "missing": [], "unread": [], "claude_usd": 0.0, "max_claude_usd": 3.0}), \
                mock.patch.object(repair, "Job", lambda name: name), \
                mock.patch("engine.settings.Settings", lambda: None), mock.patch("builtins.print"):
            self.assertEqual(repair.main(["--reread", "s5w0 p1, s6w2 p5", "--mode", "propose"]), 0)    # pass
        self.assertEqual(got, [["s5w0 p1", "s6w2 p5"]])

    RUN_ORIG = "Step one is the dough. Step two is where it rests."
    RUN_PICK = "Step one is the dough. Not. Even. Close."

    def run_span(self):
        return {"label": "s6w2 p5", "section": 6, "orig": self.RUN_ORIG, "pick": self.RUN_PICK, "notes": [],
                "key": "k", "scoring": "s6 round 1"}

    def test_a_gate_refusal_gets_one_more_try_that_can_repair_it(self):
        bad = json.dumps([{"move": "splice", "after": "Close.", "sentence": "Step two is where it rests."},
                          {"move": "splice", "after": "Close.", "sentence": "Step one is the dough."}])
        good = json.dumps([{"move": "splice", "after": "Not. Even. Close.", "sentence": "Step two is where it rests."}])
        claude = FakeClaude({("repair read", self.RUN_PICK): bad, ("repair retry", self.RUN_PICK): good}, cost=[0.3, 0.2, 0.1])
        row, usd = repair.read_span(self.run_span(), claude, None, None, repair.config(None))
        self.assertEqual([c[0] for c in claude.calls], ["repair read", "repair retry", "repair check"])
        self.assertEqual(row["outcome"], "repaired")                                                # pass: mended on the retry
        self.assertEqual(row["text"], "Step one is the dough. Not. Even. Close. Step two is where it rests.")
        self.assertTrue(row["retried"])
        self.assertEqual((row["read"], row["retry_read"]), (bad, good))                             # both reads kept
        self.assertIn("the gate said no: a splice puts in a sentence the rewrite already has", row["first_why"])
        self.assertAlmostEqual(usd, 0.6)                                                            # every call counted
        self.assertAlmostEqual(row["usd"], 0.6)
        retry = claude.prompts[1]
        self.assertTrue(retry.startswith(claude.prompts[0]))                                        # the read prompt, then the note
        self.assertIn("Your last answer couldn't be used. It was:\n" + bad, retry)
        self.assertIn("The code that applies the moves said: the gate said no: a splice puts in", retry)
        lines = "\n".join(repair.reread_lines({"spans": [{"label": "s6w2 p5", "section": 6, "pick": self.RUN_PICK,
                                                           "row": row, "calls": [("repair read", 0.3), ("repair retry", 0.2),
                                                                                 ("repair check", 0.1)]}],
                                              "missing": [], "unread": [], "claude_usd": usd, "max_claude_usd": 3.0}))
        self.assertIn("s6w2 p5 (section 6): repair read $0.30, repair retry $0.20, repair check $0.10", lines)
        self.assertIn("  Opus read: " + bad, lines)
        self.assertIn("  Opus read again: " + good, lines)
        self.assertIn("  the code gate: let them through", lines)

    def test_a_second_failure_fails_the_span_and_a_parse_failure_is_retried_too(self):
        claude = FakeClaude({("repair read", self.RUN_PICK): "Sure! Here you go.",
                             ("repair retry", self.RUN_PICK): json.dumps([{"move": "cut", "sentence": "Not there."}])})
        row, usd = repair.read_span(self.run_span(), claude, None, None, repair.config(None))
        self.assertEqual([c[0] for c in claude.calls], ["repair read", "repair retry"])           # flag: no third call
        self.assertEqual(row["outcome"], "failed")
        self.assertEqual(row["first_why"], "the answer isn't CLEAN, UNFIXABLE or a list of moves")
        self.assertEqual(row["why"], "the gate said no: a cut names no sentence of the rewrite (on the second try)")
        self.assertAlmostEqual(usd, 0.2)
        self.assertIn("It was:\nSure! Here you go.", claude.prompts[1])                           # the raw answer quoted

    def test_unfixable_and_a_check_that_finds_a_slip_are_not_retried(self):
        claude = FakeClaude({("repair read", self.RUN_PICK): "UNFIXABLE: the second step is gone"})
        row, _ = repair.read_span(self.run_span(), claude, None, None, repair.config(None))
        self.assertEqual(([c[0] for c in claude.calls], row["retried"]), (["repair read"], False))  # pass: an answer
        good = json.dumps([{"move": "splice", "after": "Not. Even. Close.", "sentence": "Step two is where it rests."}])
        claude = FakeClaude({("repair read", self.RUN_PICK): good,
                             ("repair check", "Step one is the dough. Not. Even. Close. Step two is where it rests."):
                                 "- a joke the original never made"})
        row, _ = repair.read_span(self.run_span(), claude, None, None, repair.config(None))
        self.assertEqual([c[0] for c in claude.calls], ["repair read", "repair check"])            # flag: no retry
        self.assertEqual(row["why"], "the check still found: a joke the original never made")

    def test_a_pinned_span_is_never_read_and_off_does_nothing(self):
        fix = {"label": "s5w0 p2", "orig": "Two.", "text": "Dos, by hand.", "went_wrong": "x", "about": "y", "point": "z",
               "fact": "", "when": "2026-11-14"}
        self.review({"vetoes": [], "repairs": [], "for_you": [fix]})
        self.bank(scoring("s5 round 2", 0.30, {"s5w0 p1": "Uno.", "s5w0 p2": "Dos, by hand."}))
        res = self.run_(mode="dry")
        self.assertEqual([t["label"] for t in res["to_read"]], ["s5w0 p1"])                         # flag: p2 is pinned
        res = self.run_(mode="off")
        self.assertEqual((res["spans"], res["to_read"], self.claude.calls), ([], [], []))           # pass: off
        self.assertEqual(repair.report_lines(res), ['The repair step is off ([repair] mode = "off").'])


if __name__ == "__main__":
    unittest.main()
