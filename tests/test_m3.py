"""Tests for the M3 build (2026-11-09): your lines through the loop (plan 5.1), the whole-text round and the wrap-up
(5.6), the checks from the first run-day (M2), and the sync step. Each check has a flag case and a pass case.
Run from the repo root:

    python -m unittest discover tests
"""
import json
import tempfile
import types
import unittest
from pathlib import Path

from engine.job import (Bank, Job, clean_protected, protected_fallback, reshape_runs, split_author_runs, word_in)
from engine.run import Loop, failing_runs, protected_missing

ILK = "Next door, Hargreaves keeps a ramshackle potting-shed by the back fence."


def run(sents, claude=False, **kw):
    return dict({"sents": list(sents), "claude": claude}, **kw)


# ---------------------------------------------------------------------- piece 1: your lines (plan 5.1)
class AuthorRuns(unittest.TestCase):
    def setUp(self):
        self.locked = lambda s: s.startswith("X and X")
        self.runs0 = [[run(["Mine one.", "Mine two.", "Mine three."]), run(["Claude a.", "Claude b."], claude=True)],
                      [run(["X and X and X."])],
                      [run(["Only mine."])]]

    def test_your_stretches_split_one_line_a_run(self):
        runs, index = split_author_runs(self.runs0, self.locked, verdict_of=lambda s: "yours")
        self.assertEqual([[r["sents"] for r in p] for p in runs],
                         [[["Mine one."], ["Mine two."], ["Mine three."], ["Claude a.", "Claude b."]],
                          [["X and X and X."]], [["Only mine."]]])
        self.assertEqual([r["author"] for r in runs[0]], [True, True, True, False])   # Claude's run isn't yours
        self.assertFalse(runs[1][0]["author"])                                          # a template stays locked
        self.assertTrue(runs[2][0]["author"])
        self.assertEqual(index[0], [[0, 1, 2], [3]])

    def test_banked_entries_follow_into_the_new_shape(self):
        runs, index = split_author_runs(self.runs0, self.locked)
        old = [[["Mine one.", "Mine two.", "Mine three."], ["Mistral says it."]], [["X and X and X."]], [["Only mine."]]]
        new, remap = reshape_runs(old, self.runs0, index)
        self.assertEqual(new[0], [["Mine one."], ["Mine two."], ["Mine three."], ["Mistral says it."]])
        self.assertEqual(remap[(0, 1)], (0, 3))
        same, remap = reshape_runs(new, self.runs0, index)          # already in the new shape: left alone
        self.assertIs(same, new)
        self.assertIsNone(remap)
        broken = [[["A line of yours someone changed."], ["Mistral."]], [["X and X and X."]], [["Only mine."]]]
        self.assertEqual(reshape_runs(broken, self.runs0, index), (None, None))

    def test_bank_reshape_moves_run_indices(self):
        runs, index = split_author_runs(self.runs0, self.locked)
        with tempfile.TemporaryDirectory() as d:
            b = Bank(Path(d) / "bank.jsonl")
            b.add({"kind": "round", "window": "w", "ai": 0.2, "text": "t",
                   "runs": [[["Mine one.", "Mine two.", "Mine three."], ["New."]], [["X and X and X."]], [["Only mine."]]],
                   "changes": [{"para": 0, "run": 1, "result": "replaced"}], "attempts": {"0.1": 2}})
            self.assertEqual(b.reshape("w", self.runs0, index), (1, 0))
            e = b.of("w")[0]
            self.assertEqual(e["changes"][0]["run"], 3)
            self.assertEqual(e["attempts"], {"0.3": 2})
            self.assertEqual(b.reshape("w", self.runs0, index), (0, 0))   # a second load changes nothing

    def test_protected_words(self):
        self.assertTrue({"ramshackle", "potting-shed"} <= set(protected_fallback(ILK)))
        self.assertLessEqual(len(protected_fallback(ILK)), 3)
        self.assertEqual(protected_fallback("It all comes down to creosote."), ["creosote"])
        # Sonnet's picks: only words standing in the line, never names, quotes, numbers or everyday words
        self.assertEqual(clean_protected(["ramshackle", "potting-shed", "Hargreaves", "the", "unicorn"], ILK), ["ramshackle", "potting-shed"])
        self.assertEqual(clean_protected(["confetti"], 'it reads "word confetti" all day'), [])
        self.assertTrue(word_in("potting-shed", ILK))
        self.assertFalse(word_in("potting", ILK))                 # a whole word, so "potting-shed" isn't "potting"
        self.assertTrue(word_in("Forever", "and forever, really"))  # a capital first letter may differ

    def test_a_replacement_that_loses_a_protected_word_is_demoted(self):
        original = [[[ILK]]]
        must = {(0, 0): {"hedges": [], "repeats": [], "protected": ["ramshackle", "potting-shed"], "author": True, "numbers": []}}
        flag = [[["Next door, Hargreaves keeps a potting-shed by the back fence."]]]
        keep = [[["Hargreaves next door has a ramshackle potting-shed out by the fence."]]]
        self.assertEqual(failing_runs(must, flag, original), [(0, 0)])
        self.assertEqual(failing_runs(must, keep, original), [])
        self.assertEqual(protected_missing(flag[0][0][0], ["ramshackle", "potting-shed"]), ["ramshackle"])
        nums = {(0, 0): {"hedges": [], "repeats": [], "protected": [], "numbers": ["18"]}}
        self.assertEqual(failing_runs(nums, [[["eighteen writers"]]], [[["18 writers"]]]), [(0, 0)])

    def test_the_protected_words_call_parses_and_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "job.toml").write_text('[job]\nname = "t"\noriginal = "o.md"\nworking = "w.md"\nsections = [1]\n'
                                           'voice_samples = [9]\n', encoding="utf-8")
            job = Job("t", root=d)
            calls = []

            def fake(settings, conn, prompt, role, model, max_budget_usd):
                calls.append(prompt)
                return "### 1\nramshackle, potting-shed\n### 2\nthe", {}
            got = job.protected_for(fake, None, None, [ILK, "It all comes down to creosote."], role="t protect")
            self.assertEqual(got[ILK], ["ramshackle", "potting-shed"])
            self.assertEqual(got["It all comes down to creosote."], ["creosote"])        # nothing usable: code's pick
            self.assertEqual(job.protected_cached("It all comes down to creosote.")["from"], "code")
            job.protected_for(fake, None, None, [ILK], role="t protect")
            self.assertEqual(len(calls), 1)                                            # cached by the line's text


def window_state(runs, current=None, flagged=(), scored=True):
    from engine.run import WindowState
    ws = WindowState({"wid": "w", "text": "", "section": 1, "index": 0})
    ws.runs = runs
    ws.original = [[r["sents"][:] for r in p] for p in runs]
    ws.current = current or [[r["sents"][:] for r in p] for p in runs]
    ws.scored_runs = [[r[:] for r in p] for p in ws.current] if scored else None
    ws.flagged = list(flagged)
    ws.forced = set()
    return ws


class AuthorTargets(unittest.TestCase):
    def setUp(self):
        self.runs = [[run(["A line of mine that Pangram flags."], author=True), run(["Claude's line."], claude=True)],
                     [run(["Another line of mine."], author=True)]]

    def test_a_flagged_line_of_yours_is_targeted_once_its_paragraph_has_no_claude_text(self):
        flagged = ["A line of mine that Pangram flags.", "Another line of mine."]
        ws = window_state(self.runs, flagged=flagged)
        got = Loop.targets(None, ws, False)
        self.assertIn((1, 0, False), got)                       # flagged, paragraph clean: regrown
        self.assertNotIn((0, 0, False), got)                    # Claude's line still stands next to it: not yet
        self.assertIn((0, 1, False), got)
        ws.current[0][1] = ["Mistral's version."]
        ws.scored_runs = [[r[:] for r in p] for p in ws.current]
        self.assertIn((0, 0, False), Loop.targets(None, ws, False))

    def test_a_line_of_yours_pangram_passes_stays(self):
        ws = window_state(self.runs, flagged=["Claude's line."])
        self.assertEqual([t for t in Loop.targets(None, ws, False) if self.runs[t[0]][t[1]].get("author")], [])

    def test_your_lines_are_never_settled_and_the_judge_sees_your_original(self):
        lp = Loop.__new__(Loop)
        lp.model, lp.header = "m", "samples"
        lp.fmt_example = lambda n, p: "Notes:\n" + "\n".join(n) + "\n\nParagraph:\n" + p
        # the worked examples: one holds the line being regrown, which Mistral would copy (smoke test, 2026-11-09)
        lp.examples = [ILK + " And more of mine.", "A different paragraph of yours entirely."]
        lp.example_notes = [["a"], ["b"]]
        lp.examples_block = "\n\n---\n\n".join(lp.fmt_example(n, p) for n, p in zip(lp.example_notes, lp.examples))
        ws = window_state([[run([ILK], author=True)]])
        ws.notes = [["the point"]]
        ws.must = {(0, 0): {"hedges": [], "repeats": [], "protected": ["ramshackle", "potting-shed"], "author": True,
                            "numbers": [], "asked": True}}
        ra = Loop.run_args(lp, ws, 0, 0, False, 6, True)
        self.assertFalse(ra["is_last"])                          # no settle round, no starter ladder for your lines
        self.assertNotIn("And more of mine.", ra["prompt"])      # flag: the example holding your line is out
        self.assertIn("A different paragraph of yours entirely.", ra["prompt"])   # pass: the others stay
        self.assertTrue(ra["author"])
        reasons = [f("Next door, Hargreaves keeps a potting-shed by the fence.") for f in ra["checks"]]
        self.assertIn("protected word", reasons)
        self.assertFalse(any(f("Look, Hargreaves keeps a ramshackle potting-shed by the back fence.")
                             for f in ra["checks"]))
        from engine.draws import judge_pool
        prompts = []

        def fake(settings, conn, prompt, role, model, max_budget_usd):
            prompts.append(prompt)
            return json.dumps({"pick": None, "drops": [], "why": "x"}), {}
        pool = {"stats": {}, "origin": {}, "ranked": [("cand", 0.1)], "shown": ["cand"]}
        judge_pool(pool, settings=None, conn=None, tics=lambda c: (0, []), orig_text=ILK, cur_text="an earlier regrowth",
                   is_original=False, notes=["n"], before="", after="", pi=0, ri=0, rnd=1, rounds=2, role="w",
                   claude=fake, voice=["ramshackle", "potting-shed"])
        self.assertIn(ILK, prompts[0])                          # your original, even after an earlier regrowth
        self.assertIn('"ramshackle", "potting-shed"', prompts[0])
        judge_pool(pool, settings=None, conn=None, tics=lambda c: (0, []), orig_text="Claude's line.",
                   cur_text="Claude's line.", is_original=True, notes=["n"], before="", after="", pi=0, ri=0, rnd=1,
                   rounds=2, role="w", claude=fake)
        self.assertNotIn("<<VOICE>>", prompts[1])
        self.assertNotIn("voice reference", prompts[1])


# ---------------------------------------------------------------------- piece 2: the whole-text round and the wrap-up
GUIDE = ("# Title\r\n\r\n<!-- planning: keep me -->\r\n\r\n## 1. First Section\r\n\r\nClaude said a thing. Mine stays.\r\n\r\n"
         "<!-- BEATS 1\r\n- beat\r\n-->\r\n\r\n\"The kettle didn't whistle.\"\r\n\r\nA paragraph you wrote **later** by hand.\r\n\r\n"
         "## 2. Second\r\n\r\nUntouched **bold** text.\r\n\r\n## Further Reading\r\n\r\nLinks.\r\n")


class WholeText(unittest.TestCase):
    def test_the_reserve_keeps_part_of_the_day_for_the_whole_text_round(self):
        from engine.budget import WINDOW_SHARE, PangramCapReached
        from test_run_loop import make_budget, pangram_row, ts
        b, conn, _ = make_budget(ts(3))
        pangram_row(conn, ts(2, 30), 3.00)
        b.pangram_check(0.50, reserve=2.0)                          # pass: $3.50 of the window rounds' $4.00
        with self.assertRaises(PangramCapReached) as cm:
            b.pangram_check(1.10, reserve=2.0)                      # flag: past the window share
        self.assertEqual(cm.exception.check, WINDOW_SHARE)
        b.pangram_check(1.10)                                       # the whole-text round may spend the reserve
        with self.assertRaises(PangramCapReached) as cm:
            b.pangram_check(3.10)
        self.assertEqual(cm.exception.check, "Pangram daily")       # but never past the cap itself

    def test_the_whole_text_round_passes_the_window_share_hold(self):
        from engine.budget import WINDOW_SHARE, PangramCapReached
        from test_run_loop import loop_stub
        hold = PangramCapReached(WINDOW_SHARE, "spent", until=10 ** 12)
        pg = types.SimpleNamespace(cached=lambda t: None, estimate=lambda t: (3, 0.15),
                                   score=lambda text, purpose, label: ({"fraction_ai": 0.2}, False))
        lp = loop_stub(pg=pg, budget=types.SimpleNamespace(pangram_check=lambda usd, purpose, reserve=0.0: None),
                       pangram_block=hold)
        with self.assertRaises(PangramCapReached):
            lp.score("a window", "s1w0 g3")                         # flag: a window round waits
        self.assertEqual(lp.score("a section", "whole s1", whole=True), {"fraction_ai": 0.2})   # pass
        lp.pangram_block = PangramCapReached("Pangram daily", "spent", until=10 ** 12)
        with self.assertRaises(PangramCapReached):
            lp.score("a section", "whole s1", whole=True)           # the day cap holds everyone

    def test_as_posted_is_what_a_reader_pastes(self):
        from engine.wrapup import as_posted, restore_examples
        md = GUIDE.replace('"The kettle didn\'t whistle."', "{{example 1.1}}")
        got = as_posted(restore_examples(md, [("{{example 1.1}}", '"The kettle didn\'t whistle."')]), [1, 2])
        self.assertIn('"The kettle didn\'t whistle."', got[1])          # examples back in
        self.assertNotIn("BEATS", got[1])                            # comments out
        self.assertNotIn("**", got[1])
        self.assertTrue(got[1].startswith("1. First Section"))
        self.assertEqual(got, as_posted(restore_examples(md, [("{{example 1.1}}", '"The kettle didn\'t whistle."')])
                                        .replace("\r\n", "\n"), [1, 2]))   # CRLF and LF score as one text

    def test_a_paragraph_is_recognized_run_by_run(self):
        from engine.wrapup import match_paragraph
        versions = [{"Claude said a thing.", "Mistral said it.", ""}, {"Mine stays."}]
        self.assertIsNotNone(match_paragraph("Mistral said it. Mine stays.", versions))      # the loop's text
        self.assertIsNotNone(match_paragraph("Claude said a thing.  Mine stays.", versions))  # one run put back
        self.assertIsNotNone(match_paragraph("Mine stays.", versions))                        # a cut
        self.assertIsNone(match_paragraph("Mistral said it. Mine stays, you know.", versions))  # your edit since

    def test_the_write_back_touches_prose_blocks_only(self):
        from engine.wrapup import write_back
        versions = [{"Claude said a thing.", "Mistral said it."}, {"Mine stays."}]
        new, rep = write_back(GUIDE, [1], [("s1w0 p1", versions, "Mistral said it. Mine stays.")])
        self.assertIn("\r\n\r\nMistral said it. Mine stays.\r\n\r\n", new)
        self.assertEqual(rep["replaced"], ["s1w0 p1"])
        self.assertEqual(new.replace("Mistral said it. Mine stays.", "Claude said a thing. Mine stays."), GUIDE)
        self.assertEqual([t for _, t in rep["unrecognized"]], ["A paragraph you wrote later by hand."])
        again, rep2 = write_back(new, [1], [("s1w0 p1", versions, "Mistral said it. Mine stays.")])
        self.assertEqual((again, rep2["same"]), (new, ["s1w0 p1"]))                # already there: byte for byte
        _, rep3 = write_back(GUIDE, [1], [("s1w0 p1", [{"Something else."}], "x")])
        self.assertEqual(rep3["missing"], ["s1w0 p1"])                             # edited since: left alone
        from engine.wrapup import rebold
        self.assertEqual(rebold("a later edit", "written **later** by hand"), "a **later** edit")

    def test_a_prose_block_in_quote_marks_is_still_written_back(self):
        """A paragraph at the 2026-11-12 sync opened and closed on a quote mark, which looks like an example line
        to is_prose_block; a block one of the loop's paragraphs
        recognizes is prose and gets written. A true example line (no paragraph knows it) still stays."""
        from engine.wrapup import write_back
        nl = "\r\n"
        guide = nl.join(["## 1. One", "", '"Fine" and "totally fine," in two drafts, both "shattered."', "",
                         '"A quoted example line."', ""])
        versions = [{'"Fine" and "totally fine," in two drafts, both "shattered."', '"Fine" twice, both "shattered."'}]
        new, rep = write_back(guide, [1], [("s1w0 p1", versions, '"Fine" twice, both "shattered."')])
        self.assertIn(nl.join(["", '"Fine" twice, both "shattered."', "", '"A quoted example line."']), new)
        self.assertEqual((rep["replaced"], rep["unrecognized"]), (["s1w0 p1"], []))

    def test_a_comment_block_never_matches_a_paragraph_cut_to_nothing(self):
        """A paragraph whose every run has a banked cut matches the empty text; a comment-only block, a heading and
        the separators have an empty or heading key, so they're skipped before the lookup and stay byte for byte."""
        from engine.wrapup import write_back
        nl = "\r\n"
        guide = nl.join(["## 1. One", "", "<!-- plan: the beats -->", "", "### Sub", "", "Claude said a thing.", ""])
        versions = [{"Claude said a thing.", ""}]
        new, rep = write_back(guide, [1], [("s1w0 p1", versions, "")])
        self.assertIn("<!-- plan: the beats -->", new)
        self.assertIn("### Sub", new)
        self.assertEqual(rep["replaced"], ["s1w0 p1"])
        self.assertEqual(new.replace("Claude said a thing.", ""), guide.replace("Claude said a thing.", ""))

    def test_flagged_segments_map_to_runs_yours_included_never_locked(self):
        from engine.run import WindowState
        ws = WindowState({"wid": "s1w0", "text": "", "section": 1, "index": 0})
        ws.runs = [[run(["Claude said a thing."], claude=True), run(["Mine stays here today."], author=True)],
                   [run(["X and X and X."])]]
        picks = {"s1w0": [[["Mistral said it."], ["Mine stays here today."]], [["X and X and X."]]]}
        scores = {1: {"segs": [["Mistral said it.", True], ["Mine stays here today.", True], ["X and X and X.", True]]}}
        self.assertEqual(Loop.whole_targets(None, {1: [ws]}, picks, scores), {"s1w0": [(0, 0), (0, 1)]})
        scores = {1: {"segs": [["Mistral said it.", False], ["Mine stays here today.", False]]}}
        self.assertEqual(Loop.whole_targets(None, {1: [ws]}, picks, scores), {})

    def test_a_whole_text_pick_ships_until_the_window_finds_a_new_best(self):
        """The strict rule (2026-11-10 evening): a window best banked after the whole-text pick overrides it only
        when it's better than what the pick stood on by more than the noise floor; a tie never does."""
        lp = Loop.__new__(Loop)
        with tempfile.TemporaryDirectory() as d:
            lp.bank = Bank(Path(d) / "bank.jsonl")
            ws = window_state([[run(["Claude."], claude=True)]])
            ws.wid, ws.must = "w", {(0, 0): {"hedges": [], "repeats": []}}
            lp.bank.add({"kind": "round", "window": "w", "ai": 0.10, "text": "a", "runs": [[["Best."]]]})
            self.assertEqual(Loop.ship_entry(lp, ws)["runs"], [[["Best."]]])
            lp.bank.add({"kind": "ship", "window": "w", "ai": None, "text": "b", "runs": [[["Picked whole."]]]})
            self.assertEqual(Loop.ship_entry(lp, ws)["runs"], [[["Picked whole."]]])
            # the pick's text was never scored as a window, so it stands on the window's best before it (0.10)
            self.assertEqual(Loop.ship_window_ai(lp, ws, lp.bank.entries[-1]), 0.10)
            lp.bank.add({"kind": "round", "window": "w", "ai": 0.08, "text": "c", "runs": [[["Newer best."]]]})
            self.assertEqual(Loop.ship_entry(lp, ws)["runs"], [[["Picked whole."]]])      # flag: inside the noise
            lp.bank.add({"kind": "round", "window": "w", "ai": 0.0, "text": "d", "runs": [[["Much better."]]]})
            self.assertEqual(Loop.ship_entry(lp, ws)["runs"], [[["Much better."]]])       # pass: strictly better
            # a pick stamped with what it stood on uses that, and the s3w0 case is a tie: 3.4e-10 against 1.0e-10
            lp.bank.add({"kind": "ship", "window": "w", "ai": None, "text": "e", "runs": [[["Picked again."]]],
                         "window_ai": 3.419491634293337e-10})
            lp.bank.add({"kind": "round", "window": "w", "ai": 1.0019740592781545e-10, "text": "f",
                         "runs": [[["Tied best."]]]})
            self.assertEqual(Loop.ship_entry(lp, ws)["runs"], [[["Picked again."]]])

    def test_a_whole_text_round_that_cant_start_waits_before_the_next_try(self):
        from engine.run import WindowState
        from test_run_loop import loop_stub
        lp = loop_stub()
        lp.wait, lp.whole_mode, lp.whole_wait_until = 600.0, False, 0.0
        lp.job = types.SimpleNamespace(windows=lambda: [{"wid": "s1w0", "section": 1}], sections=[1])
        lp.windows = [WindowState({"wid": "s1w0", "text": "", "section": 1, "index": 0})]
        seen = []
        lp.prepare = lambda ws: seen.append(lp.whole_mode) or False      # a cap holds the window up
        self.assertFalse(Loop.whole_text_round(lp))
        self.assertEqual(seen, [True])                                    # readied with the reserve open to it
        self.assertFalse(lp.whole_mode)
        self.assertEqual(lp.whole_wait_until, lp.clock.now() + 600)       # not tried again for wait_minutes

    def test_a_cap_ends_a_smoke_test(self):
        from engine.budget import ClaudeCapReached
        from test_run_loop import loop_stub
        lp = loop_stub(cap_events=[], stop_reason=None)
        lp.budget = types.SimpleNamespace(snapshot=lambda: {"loop_share": 0, "real_total": 0, "week_reset": 0})
        Loop.on_cap(lp, ClaudeCapReached("Claude 5-hour (night)", "x", until=lp.clock.now() + 60))
        self.assertIsNone(lp.stop_reason)                                   # pass: a real run carries on
        lp.args.smoke = True
        Loop.on_cap(lp, ClaudeCapReached("smoke test limit", "x", until=lp.clock.now() + 60))
        self.assertIn("smoke test stops", lp.stop_reason)                   # flag: a smoke test ends

    def test_a_now_run_winds_down_before_its_stop(self):
        from test_run_loop import loop_stub, ts
        lp = loop_stub()
        lp.args.now, lp.stop_reason, lp.stop_file, lp.now_stop, lp.wind_down = True, None, None, ts(16), 1800.0
        lp.clock.advance(12 * 3600 + 25 * 60)                      # 15:25
        self.assertFalse(lp.must_stop())
        lp.clock.advance(5 * 60)                                    # 15:30
        self.assertTrue(lp.must_stop())
        self.assertIn("wind-down", lp.stop_reason)
        self.assertFalse(Loop.hard_stop(lp))                        # the day-end whole-text round may still run
        lp.clock.advance(26 * 60)                                   # 15:56
        self.assertTrue(Loop.hard_stop(lp))


# ---------------------------------------------------------------------- piece 3: the first run-day's fixes (M2)
NAMED = ('Version 4.2 of the camera app cut shutter lag to 0.3 seconds, and Priya Natarajan of the review team posted'
         ' about it on X in June 2024 after the firmware shipped ("the lag is finally gone for good").')
NAMED_BROKEN = ('4.2 of the camera app cut shutter lag to 0.3 seconds.: "the lag is finally gone for good").')
NAMED_DROPPED = ('Version 4.2 of the camera app cut shutter lag to 0.3 seconds, and someone on the review team posted'
                 ' about it on X in June 2024 after the firmware shipped ("the lag is finally gone for good").')
NAMED_NOTES = ['"Priya Natarajan" posted on X in "June 2024" after the firmware shipped, saying "the lag is finally'
               ' gone for good".']
LADDER_A = ('The other habit is the drumroll, where every line gets louder than the last as the recipe races toward'
            ' its grand reveal, which, of course, the cooking never actually earned.')
LADDER_B = "It's racing toward a grand reveal, but the cooking didn't put in any fucking effort to earn that."


class FirstRunDayChecks(unittest.TestCase):
    def test_punctuation(self):
        from engine.checks import punctuation_problem
        self.assertEqual(punctuation_problem(NAMED_BROKEN, NAMED), "stray colon after a full stop")      # flag
        self.assertIsNone(punctuation_problem(NAMED, NAMED))                                              # pass
        self.assertEqual(punctuation_problem('He said "never mind.', 'He said "never mind."'), "unmatched quote")
        self.assertEqual(punctuation_problem("a thing) and more.", "a (thing) and more."), "stray closing parenthesis")
        self.assertEqual(punctuation_problem("a (thing and more.", "a (thing) and more."), "unclosed parenthesis")
        self.assertEqual(punctuation_problem("They're everywhere:", "They're everywhere."), "ends on a colon")
        self.assertIsNone(punctuation_problem("They're all over:", "They're everywhere:"))   # the original has it too

    def test_names_are_counted(self):
        from engine.checks import citations, names_missing, names_required
        need = names_required(NAMED, NAMED_NOTES)
        self.assertIn("Priya Natarajan", need)
        self.assertEqual(names_missing(NAMED_BROKEN, need), ["Priya Natarajan"])                               # flag
        self.assertEqual(names_missing("Even Priya Natarajan bragged about it on X.", need), [])                 # pass
        self.assertEqual(names_required(NAMED, ["the joke is about old models"]), [])    # notes skip it: droppable
        self.assertEqual(names_required(NAMED, NAMED_NOTES + ["droppable: Priya Natarajan"]), [])
        self.assertEqual(names_required("Then Priya Natarajan did it.", ()), ["Priya Natarajan"])  # a sentence opener comes off
        self.assertEqual(citations("as Okafor et al. (2024) and (Lind, 2026) found"), ["Okafor et al.", "(Lind, 2026)"])
        self.assertEqual(names_missing("as a 2024 paper found", ["Okafor et al."]), ["Okafor et al."])
        # pass cases: a citation reworded with its name and year, and a capitalized word pair
        self.assertEqual(names_missing('From the Acme, 2023 paper: "Widget-2 is a model."', ["(Acme, 2023)"]), [])
        self.assertEqual(names_missing("From the Acme paper: it's a model.", ["(Acme, 2023)"]), ["(Acme, 2023)"])
        self.assertEqual(names_required("Each Part covers one cause.", ["each part covers a cause"]), [])

    def test_the_named_run_is_demoted(self):
        from engine.checks import names_required
        must = {(0, 0): {"hedges": [], "repeats": [], "names": names_required(NAMED, NAMED_NOTES)}}
        self.assertEqual(failing_runs(must, [[[NAMED_BROKEN]]], [[[NAMED]]]), [(0, 0)])
        self.assertEqual(failing_runs(must, [[[NAMED_DROPPED]]], [[[NAMED]]]), [(0, 0)])     # flag: the name alone
        good = ('Version 4.2 of the camera app brought shutter lag down to 0.3 seconds, and Priya Natarajan from the'
                ' review team cheered on X in June 2024 once the firmware shipped ("the lag is finally gone for'
                ' good").')
        self.assertEqual(failing_runs(must, [[[good]]], [[[NAMED]]]), [])

    def test_a_quote_closing_the_sentence_in_parentheses_keeps_its_opener(self):
        from engine.draws import draw_pool

        class Srv:
            def complete(self, prompt, **kw):
                self.stop = kw.get("stop")
                return "the camera app now has 0.3 seconds of lag, and Priya Natarajan even cheered on X in June 2024 about it.", {}
        srv = Srv()
        op = types.SimpleNamespace(score=lambda cands: [{"score": 0.1} for _ in cands])
        pool = draw_pool(srv=srv, op=op, tics=lambda c: (0, []), prompt="p ", orig_text=NAMED, n_sent=1,
                         is_original=True, known_names={"Version", "Priya", "Natarajan", "June", "X"},
                         swears_elsewhere=0, swear_budget=5, digit_ban=None, pi=0, ri=0, rnd=1, rounds=2,
                         extra_checks=[lambda c: None])
        self.assertIn("(", srv.stop)
        self.assertTrue(pool["shown"])
        self.assertTrue(pool["shown"][0].endswith(' about it ("the lag is finally gone for good").'))
        self.assertNotIn(".:", pool["shown"][0])

    def test_restatement(self):
        from engine.checks import restated_pairs, restatement, restates_in_context
        self.assertGreaterEqual(restatement(LADDER_A, LADDER_B), 0.6)                                       # flag
        ok_a = "The cook just goes by feel about which spices pair up."
        ok_b = 'The cook just "knows" that after "brown the onions", words like "add the garlic" come next.'
        self.assertLess(restatement(ok_a, ok_b), 0.6)                                                       # pass
        self.assertEqual([(i, j) for i, j, _ in restated_pairs([LADDER_A, LADDER_B])], [(0, 1)])
        orig = "So the diner's left staring at a dish that never arrived."
        self.assertTrue(restates_in_context(LADDER_B, orig, before=LADDER_A))                              # flag
        self.assertFalse(restates_in_context("Nobody wrote the middle bit of the story.", orig, before=LADDER_A))
        # an original that already says it twice isn't the candidate's fault
        self.assertFalse(restates_in_context(LADDER_B, "The cooking never earns its grand reveal.", before=LADDER_A))

    def test_a_replacement_that_says_it_twice_is_demoted(self):
        original = [[["The other habit is the drumroll, each line louder than the last.",
                       "The cooking never earns its grand reveal."]]]
        must = {(0, 0): {"hedges": [], "repeats": []}}
        self.assertEqual(failing_runs(must, [[[LADDER_A, LADDER_B]]], original), [(0, 0)])            # flag
        self.assertEqual(failing_runs(must, [[[LADDER_A, "Nobody wrote the middle bit."]]], original), [])   # pass

    def test_the_run_day_cap_holds_across_midnight(self):
        from engine.budget import PangramCapReached
        from test_run_loop import make_budget, pangram_row, ts
        b, conn, clock = make_budget(ts(0, 30))                    # 00:30, a --now run started at 18:40 yesterday
        pangram_row(conn, ts(20, days=-1), 4.60)                   # spent before midnight
        b.pangram_check(1.00)                                      # pass: the calendar day alone lets it through
        b.run_day_start = ts(18, 40, days=-1)
        with self.assertRaises(PangramCapReached) as cm:
            b.pangram_check(1.50)                                  # flag: $6.10 this run-day
        self.assertEqual(cm.exception.check, "Pangram run-day")
        self.assertAlmostEqual(b.snapshot()["pangram_run_day"], 4.60)
        b.pangram_check(1.00)                                      # $5.60 still fits


# ---------------------------------------------------------------------- piece 4: the sync step
ORIG = ("# Guide\n\n## 1. First Section\n\nClaude wrote this line about tokens and words. I wrote this bit myself, mate.\n\n"
        "Another paragraph that stays exactly the same here.\n\n## 2. Second\n\n<!-- BEATS 2\n- beat\n-->\n\n"
        "## 3. Third\n\n<!-- BEATS 3 -->\n\n## Further Reading\n\nLinks.\n")
LINES = ("| Verdict | In your messages | In Claude's output | Pangram | Sentence |\n| --- | --- | --- | --- | --- |\n"
         "| claude | 0% | 100% | AI | Claude wrote this line about tokens and words. |\n"
         "| yours | 100% | 0% | human | I wrote this bit myself, mate. |\n"
         "| yours | 100% | 0% | human | Another paragraph that stays exactly the same here. |\n")
REGROWN = "Mistral put this line about tokens and words."


def make_job(d):
    d = Path(d)
    (d / "job.toml").write_text('[job]\nname = "t"\noriginal = "orig.md"\nworking = "work.md"\nsections = [1]\n'
                                'voice_samples = [9]\n\n[seed]\nbanks = []\nwindows = ["s1w0"]\n', encoding="utf-8")
    (d / "orig.md").write_text(ORIG, encoding="utf-8")
    (d / "your lines.md").write_text(LINES, encoding="utf-8")
    text = Job(root=d).windows()[0]["text"]
    b = Bank(d / "bank.jsonl")
    b.add({"kind": "start", "window": "s1w0", "ai": 1.0, "segs": [], "text": text})
    b.add({"kind": "round", "window": "s1w0", "ai": 0.1, "text": "x",
           "runs": [[[REGROWN], ["I wrote this bit myself, mate."]], [["Another paragraph that stays exactly the same here."]]]})
    return d


def stub_scan(text):
    from engine.draws import sents
    return [(s, "yours", 1.0, 0.0) for p in text.split("\n\n") for s in sents(p)]


class Sync(unittest.TestCase):
    def test_a_handed_over_guide_comes_in(self):
        from engine.sync import sync
        with tempfile.TemporaryDirectory() as d:
            make_job(d)
            # the real guide after a write-back (the loop's line in), with section 2 drafted, an example line,
            # section 3 still only BEATS, and section 4 still notes outside a comment
            real = (ORIG.replace("Claude wrote this line about tokens and words.", REGROWN)
                    .replace("<!-- BEATS 2\n- beat\n-->", "<!-- BEATS 2\n- beat\n-->\n\nA brand new paragraph for the second section.\n\n"
                             "\"The door shut. Nobody moved.\"")
                    .replace("## Further Reading", "## 4. Fourth\n\n   - a note to self, not prose yet\n\n## Further Reading"))
            src = Path(d) / "real.md"
            src.write_text(real, encoding="utf-8")
            r = sync(d, src, write=True, classify=stub_scan, today="2026-11-09")
            self.assertEqual(r["sections_after"], [1, 2])                         # 3 and 4 stay out
            self.assertIn("still being drafted", r["left_out"][4])
            self.assertIn("no prose yet", r["left_out"][3])
            self.assertEqual(r["kept"], [("s1w0", "s1w0")])                       # its text is unchanged, so its history stays
            self.assertEqual(r["restored"], ["s1w0 p1"])                          # the loop's line, back to the frozen original
            self.assertEqual(r["examples_added"], ['"The door shut. Nobody moved."'])
            job = Job(root=d)
            orig = job.original.read_text(encoding="utf-8")
            self.assertIn("Claude wrote this line about tokens and words.", orig)
            self.assertNotIn(REGROWN, orig)
            self.assertEqual([w["wid"] for w in job.windows()], ["s1w0", "s2w0"])
            self.assertNotIn("The door shut", job.windows()[1]["text"])              # the example line is out of the prose
            self.assertEqual(job.verdicts().get("A brand new paragraph for the second section."), "yours")
            self.assertEqual(job.sections, [1, 2])
            self.assertTrue((Path(d) / "job (before sync 2026-11-09).toml").exists())
            again = sync(d, src, write=False, classify=stub_scan, today="2026-11-09")
            self.assertEqual((again["fresh"], again["retired"], again["new_verdicts"]), ([], [], {}))   # idempotent

    def test_an_edited_window_starts_again_and_the_loops_lines_get_their_own_verdict(self):
        from engine.sync import sync
        with tempfile.TemporaryDirectory() as d:
            make_job(d)
            # you edited the paragraph that holds the loop's line: it's yours now, but the loop's sentence in it is
            # still the loop's (a "loop" verdict, regrown like Claude's), and the window starts again
            real = ORIG.replace("Claude wrote this line about tokens and words. I wrote this bit myself, mate.",
                                REGROWN + " I wrote this bit myself, mate, and then some.")
            src = Path(d) / "real.md"
            src.write_text(real, encoding="utf-8")
            r = sync(d, src, write=True, classify=stub_scan, today="2026-11-09")
            self.assertEqual(r["retired"], ["s1w0"])
            self.assertEqual(r["fresh"], ["s1w0"])
            self.assertEqual(r["seed_windows"], [])
            self.assertEqual(r["new_verdicts"], {"loop": 1, "yours": 1})
            self.assertEqual(Job(root=d).verdicts().get(REGROWN), "loop")
            bank = Bank(Path(d) / "bank.jsonl")
            self.assertEqual({e["window"] for e in bank.entries}, {"s1w0~before-sync-2026-11-09"})
            self.assertIn("windows = []", (Path(d) / "job.toml").read_text(encoding="utf-8"))

    def test_a_partial_run_keeps_the_other_windows_in_the_working_copy(self):
        with tempfile.TemporaryDirectory() as d:
            make_job(d)
            job = Job(root=d)
            p1 = "Claude wrote this line about tokens and words. I wrote this bit myself, mate."
            p2 = "Another paragraph that stays exactly the same here."
            job.assemble({"s1w0": ([p1], ["Mistral says it. I wrote this bit myself, mate."])})
            # a --window run that only knows the second paragraph's window keeps the first one's text
            job.assemble({"other": ([p2], ["A regrown second paragraph."])}, keep_others=True)
            text = job.working.read_text(encoding="utf-8")
            self.assertIn("Mistral says it.", text)                                                       # pass
            self.assertIn("A regrown second paragraph.", text)
            job.assemble({"other": ([p2], ["A regrown second paragraph."])})                               # flag
            self.assertNotIn("Mistral says it.", job.working.read_text(encoding="utf-8"))
            job.assemble({"other": ([p2], [p2])}, keep_others=True)            # a window back at its original
            self.assertIn(p2, job.working.read_text(encoding="utf-8"))

    def test_the_write_back_still_knows_a_set_aside_windows_text(self):
        from engine.wrapup import paragraph_history, recognized
        with tempfile.TemporaryDirectory() as d:
            make_job(d)
            bank = Bank(Path(d) / "bank.jsonl")
            for e in bank.entries:
                e["window"] = "s1w0~before-sync-2026-11-09"
            known = paragraph_history(bank)["Claude wrote this line about tokens and words. I wrote this bit myself, mate."]
            self.assertTrue(recognized(REGROWN + " I wrote this bit myself, mate.", [], known))          # pass
            self.assertFalse(recognized(REGROWN + " Something you wrote.", [], known))                    # flag


if __name__ == "__main__":
    unittest.main()
