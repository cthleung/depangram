"""Tests for the writer arms (plan 6.9, Phase 3; 2026-11-09): the notes-to-text adapter as the primary writer on one
shared server, the few-shot writer as the fallback and the flavor pass, the arm in every key and record, the new
checks (each with a flag case and a pass case), the judge's new lines, rewrite_cleared both ways, and the per-arm
analytics. No model, no Pangram, no Claude anywhere. Run from the repo root:

    python -m unittest discover tests
"""
import json
import tempfile
import types
import unittest
from pathlib import Path

from engine import analytics as an
from engine.checks import connective_problem, connectives_in, reddit_tell
from engine.job import Bank, PoolQueue, best_seed
from engine.local_models import ArmServer, LlamaServer, arms_available
from engine.run import Loop
from engine.draws import DANGLE_LINE, judge_pool

CLAUDE = "The tram map shows every stop but never says which ones close after nine."
NOTES = ["the map hides which stops close late", "it skips the night timetable", 'keep: "probably"']


def run(sents, claude=False, **kw):
    return dict({"sents": list(sents), "claude": claude}, **kw)


def window(runs, current=None, flagged=(), ai=0.0):
    ws = types.SimpleNamespace(wid="s1w1", runs=runs, heading=None, notes=[NOTES for _ in runs], names=set(),
                               swear_budget=3, forced=set(), rewrites=set(), passes_today=1, segs=[], ai=ai,
                               attempts={}, history={})
    ws.original = [[r["sents"][:] for r in p] for p in runs]
    ws.current = [[r[:] for r in p] for p in (current or ws.original)]
    ws.scored_runs = [[r[:] for r in p] for p in ws.current]
    ws.flagged = list(flagged)
    ws.must = {(pi, ri): {"hedges": [], "repeats": [], "numbers": [], "names": []}
               for pi, p in enumerate(runs) for ri, _ in enumerate(p)}
    ws.n_flagged = lambda: len(ws.flagged)
    ws.claude_text_left = lambda: any(r["claude"] and ws.current[pi][ri] == ws.original[pi][ri]
                                      for pi, p in enumerate(runs) for ri, r in enumerate(p))
    return ws


def loop(bank=None, arms=("notes", "fewshot"), primary="notes", fallback="fewshot", rewrite=False):
    lp = Loop.__new__(Loop)
    lp.model, lp.header, lp.examples_block, lp.examples, lp.example_notes = "m", "samples", "EXAMPLES", [], []
    lp.fmt_example = lambda n, p: "Notes:\n" + "\n".join(f"- {x}" for x in n) + "\n\nParagraph:\n" + p
    lp.arms, lp.primary_arm, lp.fallback_arm, lp.rewrite_cleared = arms, primary, fallback, rewrite
    lp.bank = bank
    lp.swear_rates = None
    lp.pools_used = lp.pool_draws_used = 0
    lp.log = lambda msg: None
    return lp


# ---------------------------------------------------------------------- the checks (engine/checks.py)
class Checks(unittest.TestCase):
    def test_connective_openers_are_capped_per_paragraph(self):
        # flag: the candidate brings an opener into a paragraph that already has one
        self.assertEqual(connective_problem("In other words, it fails.", before="Of course it does.", after=""),
                         "connective glue (In other words)")
        # pass: one opener in the paragraph is the author's own habit
        self.assertIsNone(connective_problem("In other words, it fails.", before="It does.", after="Then it stops."))
        # pass: a paragraph already over the cap doesn't freeze a candidate that adds none
        self.assertIsNone(connective_problem("It fails.", before="Of course it does. In fact, twice.", after=""))
        # "e.g." counts mid-sentence; a quoted opener doesn't
        self.assertEqual(connectives_in('He wrote "Of course not." The room, e.g. the bed, is vague.'), ["e.g."])
        self.assertEqual(connectives_in("The course of fact. Looking atlas. Natural."), [])

    def test_single_asterisk_italics_count_as_markup(self):
        from engine.draws import MARKUP
        self.assertTrue(MARKUP.search("The app has no *night mode*, so it guesses."))             # flag: Reddit italics
        self.assertFalse(MARKUP.search("2 * 3 * 4 is big, and a * b is fine."))                 # pass: arithmetic

    def test_reddit_furniture_is_out(self):
        for tell in ("Edit: forgot to add this.", "TL;DR it sucks.", "see r/writing for more", "OP said so",
                     "Then everyone upvoted it.", "a&#x200B;b"):
            self.assertIsNotNone(reddit_tell(tell), tell)                              # flag
        for ok in ("haha, probably... this is fine", "the open door, e.g. a fridge", "an op-ed about Reddit"):
            self.assertIsNone(reddit_tell(ok), ok)                                      # pass
    def test_filler_tags_are_capped_per_section(self):
        # author, 2026-11-12: "honestly" tacked onto a clause is the editor's habit; one per section as posted, at most
        from engine.checks import section_counts, section_filler_problem
        rest = section_counts("The section so far, which is pretty absurd, honestly. More of it.")
        self.assertEqual(rest[3], 1)
        # flag: the candidate brings a second one into a section that already holds one
        self.assertEqual(section_filler_problem("It fails, honestly.", "It fails.", rest, cap=1), "filler tag (honestly)")
        # pass: the run it replaces already carried it (the author's own stays)
        self.assertIsNone(section_filler_problem("It fails, honestly.", "It fails, honestly.", rest, cap=1))
        # pass: a section with none takes the first
        self.assertIsNone(section_filler_problem("Honestly, it fails.", "It fails.", section_counts("Plain text."), cap=1))
        # cap 0: nothing new gets in
        self.assertEqual(section_filler_problem("Honestly, it fails.", "It fails.", section_counts("Plain."), cap=0),
                         "filler tag (honestly)")
        # a quoted one isn't counted, and an old three-item rest passes everything
        self.assertEqual(section_counts('She wrote "honestly" on the form.')[3], 0)
        self.assertIsNone(section_filler_problem("It fails, honestly.", "It fails.", (10, 0, 0), cap=1))


# ---------------------------------------------------------------------- the judge's new lines
class Judge(unittest.TestCase):
    def prompts(self, **kw):
        seen = []

        def fake(settings, conn, prompt, role, model, max_budget_usd):
            seen.append(prompt)
            return json.dumps({"pick": None, "drops": [], "why": "x"}), {}
        pool = {"stats": {}, "origin": {}, "ranked": [("cand", 0.1)], "shown": ["cand"]}
        judge_pool(pool, settings=None, conn=None, tics=lambda c: (0, []), orig_text=CLAUDE, cur_text=CLAUDE,
                   is_original=True, notes=NOTES, before="", after="", pi=0, ri=0, rnd=1, rounds=2, role="w",
                   claude=fake, **kw)
        return seen[0]

    def test_the_dangling_opener_line_goes_only_to_a_run_that_opens_its_paragraph(self):
        self.assertIn(DANGLE_LINE.strip(), self.prompts(opens_para=True))             # flag case: the line is there
        self.assertNotIn("answers nothing", self.prompts(opens_para=False))           # pass: a mid-paragraph run
        self.assertNotIn("<<DANGLE>>", self.prompts(opens_para=False))

    def test_the_calmer_register_is_accepted(self):
        p = self.prompts()
        self.assertIn("calmer register is accepted", p)
        self.assertIn("at least as funny", p)                                          # the joke rule still holds


# ---------------------------------------------------------------------- the server: one process, two arms
class Server(unittest.TestCase):
    def test_the_scale_goes_in_the_request_only_when_given(self):
        srv = LlamaServer("hip-notes-mistral24b", settings=types.SimpleNamespace(
            get=lambda k, d=None: d, runs_dir=Path(tempfile.gettempdir())))
        sent = []

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"content": "x", "timings": {}}
        import engine.local_models as lm
        real = lm.requests.post
        lm.requests.post = lambda url, timeout, json: sent.append(json) or R()
        try:
            srv.complete("p", lora_scale=0.0)
            srv.complete("p")
        finally:
            lm.requests.post = real
        self.assertEqual(sent[0]["lora"], [{"id": 0, "scale": 0.0}])                   # the few-shot arm, explicit
        self.assertNotIn("lora", sent[1])                                              # older callers: the server's own
        self.assertTrue(srv.has_lora)
        self.assertEqual(arms_available(srv), ("notes", "fewshot"))
        plain = LlamaServer("mistral-small-24b-base", settings=srv.__dict__ and types.SimpleNamespace(
            get=lambda k, d=None: d, runs_dir=Path(tempfile.gettempdir())))
        self.assertEqual(arms_available(plain), ("fewshot",))

    def test_the_arm_wrapper_sets_scale_and_stops(self):
        calls = []
        fake = types.SimpleNamespace(has_lora=True, complete=lambda prompt, **kw: calls.append(kw) or ("", {}),
                                     digit_tokens=lambda: [1])
        ArmServer(fake, "notes").complete("p", stop=["\n"])
        ArmServer(fake, "fewshot").complete("p", stop=["\n"])
        self.assertEqual(calls[0]["lora_scale"], 1.0)
        self.assertIn("</target_text>", calls[0]["stop"])
        self.assertEqual(calls[1]["lora_scale"], 0.0)
        self.assertEqual(calls[1]["stop"], ["\n"])
        self.assertEqual(ArmServer(fake, "notes").digit_tokens(), [1])                 # everything else passes through
        bare = types.SimpleNamespace(complete=lambda prompt, **kw: calls.append(kw) or ("", {}))
        ArmServer(bare, "fewshot").complete("p")
        self.assertNotIn("lora_scale", calls[2])                                       # no adapter: no scale


# ---------------------------------------------------------------------- the arm in run_args, the key and the records
class Arms(unittest.TestCase):
    def setUp(self):
        self.ws = window([[run(["Mine first."], author=False), run([CLAUDE], claude=True)]])
        self.lp = loop()

    def test_the_adapter_prompt_is_notes_then_the_paragraph_so_far_flush(self):
        ra = Loop.run_args(self.lp, self.ws, 0, 1, False, 6, True, arm="notes")
        self.assertEqual(ra["prompt"], "<notes>\n- " + "\n- ".join(NOTES) + "\n</notes>\n<target_text>\nMine first.")
        self.assertNotIn("samples", ra["prompt"])                                      # no header, no examples
        self.assertFalse(ra["is_last"])                                                # no settle, no ladder
        self.assertEqual(ra["arm"], "notes")
        self.assertFalse(ra["opens_para"])
        first = Loop.run_args(self.lp, self.ws, 0, 0, False, 1, False, arm="notes")
        self.assertTrue(first["prompt"].endswith("<target_text>\n"))                    # an empty prefix: nothing after
        self.assertTrue(first["opens_para"])

    def test_the_few_shot_prompt_is_as_before_and_the_arm_is_in_the_key(self):
        fs = Loop.run_args(self.lp, self.ws, 0, 1, False, 6, True, arm="fewshot")
        self.assertTrue(fs["prompt"].startswith("samples\n\n---\n\nEXAMPLES"))
        self.assertTrue(fs["prompt"].endswith("Mine first. "))                          # the trailing space stays here
        self.assertTrue(fs["is_last"])
        nt = Loop.run_args(self.lp, self.ws, 0, 1, False, 6, True, arm="notes")
        self.assertNotEqual(fs["key"], nt["key"])
        # the same arm, the same inputs: the same key (a queued pool is found again)
        self.assertEqual(nt["key"], Loop.run_args(self.lp, self.ws, 0, 1, False, 6, True, arm="notes")["key"])
        # the default arm is the primary
        self.assertEqual(Loop.run_args(self.lp, self.ws, 0, 1, False, 1, False)["arm"], "notes")

    def test_arm_order_falls_back_to_the_few_shot_writer_without_an_adapter(self):
        self.assertEqual(self.lp.arm_order(), ["notes", "fewshot"])
        self.assertEqual(loop(arms=("fewshot",)).arm_order(), ["fewshot"])
        self.assertEqual(loop(fallback=None).arm_order(), ["notes"])
        self.assertEqual(loop(primary="fewshot", fallback="notes").arm_order(), ["fewshot", "notes"])

    def regrow(self, pools, verdicts, is_last=False):
        """regrow() with a fake draw (pool per arm) and a fake judge (verdict per arm)."""
        lp = self.lp
        with tempfile.TemporaryDirectory() as d:
            lp.queue = PoolQueue(Path(d) / "q.json")
            lp.budget = types.SimpleNamespace(run_check=lambda f: None)
            lp.clock = types.SimpleNamespace(now=lambda: 0.0)
            lp.draw = lambda ws, ra, k_offset=0: pools[ra["arm"]]
            lp.judge = lambda ws, ra, pool: verdicts[ra["arm"]]
            return Loop.regrow(lp, self.ws, 0, 1, False, 1, is_last)

    def test_the_fallback_fires_when_the_adapter_passes_no_draw(self):
        empty = {"shown": [], "raw": 48, "tally": "length 48", "rejected": [], "stats": {"draws": 48}}
        full = {"shown": ["c"], "raw": 48, "tally": "", "rejected": [], "stats": {"draws": 48}}
        ok = ("Few-shot wrote this.", {"para": 0, "new": "Few-shot wrote this.", "result": "replaced (1 passed filters, head-to-head: SAME)"}, {})
        new, changes, status = self.regrow({"notes": empty, "fewshot": full}, {"fewshot": ok})
        self.assertEqual(new, "Few-shot wrote this.")                                  # flag: the fallback ran
        self.assertEqual([c["arm"] for c in changes], ["notes", "fewshot"])
        self.assertTrue(changes[0]["result"].startswith("no candidate passed"))
        self.assertIsNone(status)
        # pass: the adapter passed a draw and the judge took it, so the fallback never draws
        new, changes, _ = self.regrow({"notes": full, "fewshot": full},
                                      {"notes": ("Adapter wrote this.", {"para": 0, "new": "Adapter wrote this.", "result": "replaced"}, {})})
        self.assertEqual((new, [c["arm"] for c in changes]), ("Adapter wrote this.", ["notes"]))

    def test_in_a_settle_round_the_few_shot_arm_settles_what_the_adapter_couldnt(self):
        full = {"shown": ["c"], "raw": 48, "tally": "", "rejected": [], "stats": {"draws": 48}}
        kept = (None, {"para": 0, "result": "pick lost the head-to-head (3 passed filters)"}, {})
        settled = ("Few-shot settled it.", {"para": 0, "new": "Few-shot settled it.", "result": "replaced (settled in the last round)"}, {})
        new, changes, _ = self.regrow({"notes": full, "fewshot": full}, {"notes": kept, "fewshot": settled}, is_last=True)
        self.assertEqual((new, [c["arm"] for c in changes]), ("Few-shot settled it.", ["notes", "fewshot"]))   # flag
        new, changes, _ = self.regrow({"notes": full, "fewshot": full}, {"notes": kept, "fewshot": settled}, is_last=False)
        self.assertEqual((new, [c["arm"] for c in changes]), (None, ["notes"]))         # pass: an ordinary round stops

    def test_best_seed_skips_an_entry_from_another_writer_arm(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{"model": "m", "window": "w", "runs": [[["a"]]], "ai": 0.1, "arm": "notes"},
                    {"model": "m", "window": "w", "runs": [[["b"]]], "ai": 0.2, "arm": "fewshot"},
                    {"model": "m", "window": "w", "runs": [[["c"]]], "ai": 0.3, "arm": 2}]     # experiment 7's number
            (Path(d) / "exp.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            self.assertEqual(best_seed(d, ["exp.jsonl"], "w", "m", arm="fewshot")[1]["ai"], 0.2)   # flag: notes skipped
            self.assertEqual(best_seed(d, ["exp.jsonl"], "w", "m")[1]["ai"], 0.1)                 # pass: no arm asked

    def test_best_seed_reads_an_untagged_entry_only_for_the_window_named(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "exp.jsonl").write_text(json.dumps({"model": "m", "runs": [[["a"]]], "ai": 0.1}), encoding="utf-8")
            self.assertIsNone(best_seed(d, ["exp.jsonl"], "s1w0", "m"))                           # flag: no default
            self.assertIsNone(best_seed(d, ["exp.jsonl"], "s2w0", "m", untagged="s1w0"))          # flag: another window
            self.assertEqual(best_seed(d, ["exp.jsonl"], "s1w0", "m", untagged="s1w0")[1]["ai"], 0.1)   # pass


# ---------------------------------------------------------------------- rewrite_cleared and the flavor lock
class RewriteCleared(unittest.TestCase):
    def setUp(self):
        self.runs = [[run([CLAUDE], claude=True), run(["Second Claude line here."], claude=True)]]
        self.current = [[["Few-shot regrowth one."], ["Adapter regrowth two."]]]

    def bank(self, d):
        b = Bank(Path(d) / "bank.jsonl")
        b.add({"kind": "round", "window": "s1w1", "ai": 0.0, "text": "t", "runs": self.current,
               "changes": [{"para": 0, "run": 0, "new": "Few-shot regrowth one.", "result": "replaced"},
                           {"para": 0, "run": 1, "new": "Adapter regrowth two.", "result": "replaced", "arm": "notes"}]})
        return b

    def test_true_regrows_the_few_shot_written_runs_of_a_clean_window_once(self):
        with tempfile.TemporaryDirectory() as d:
            lp = loop(bank=self.bank(d), rewrite=True)
            ws = window(self.runs, current=self.current)
            self.assertEqual(Loop.targets(lp, ws, False), [(0, 0, False)])            # flag: the few-shot run, not the adapter's
            self.assertEqual(ws.rewrites, {(0, 0)})
            ws.passes_today = 2
            self.assertEqual(Loop.targets(lp, ws, False), [])                           # only the first pass of the day
            ws.passes_today, ws.flagged = 1, ["Few-shot regrowth one."]
            self.assertEqual(Loop.targets(lp, ws, False), [(0, 0, False)])            # a flagged run is an ordinary target
            self.assertEqual(ws.rewrites, set())

    def test_false_leaves_a_clean_window_alone(self):
        with tempfile.TemporaryDirectory() as d:
            lp = loop(bank=self.bank(d), rewrite=False)
            ws = window(self.runs, current=self.current)
            self.assertEqual(Loop.targets(lp, ws, False), [])                           # pass: the default path
            self.assertEqual(ws.rewrites, set())

    def test_a_flavored_run_is_locked_and_a_flavor_entry_ships(self):
        with tempfile.TemporaryDirectory() as d:
            b = self.bank(d)
            flavored = [[["Few-shot regrowth one."], ["Adapter regrowth two.", "Which, haha, is the whole point."]]]
            b.add({"kind": "flavor", "window": "s1w1", "ai": None, "text": "t2", "runs": flavored, "arm": "flavor",
                   "inserts": [{"para": 0, "run": 1, "sentence": "Which, haha, is the whole point."}]})
            lp = loop(bank=b)
            # the window resumes from its best round, which never holds the insert (a flavor entry has no window
            # score): apply_flavor carries it in, and only then is the run locked
            ws = window(self.runs, current=self.current, flagged=["Which, haha, is the whole point.", "Adapter regrowth two."])
            self.assertFalse(Loop.flavor_locked(lp, ws, 0, 1))
            self.assertEqual(Loop.apply_flavor(lp, ws), [(0, 1)])
            self.assertEqual(ws.current, flavored)
            self.assertEqual(Loop.apply_flavor(lp, ws), [])                               # once
            ws.scored_runs = [[r[:] for r in p] for p in ws.current]
            self.assertEqual(Loop.targets(lp, ws, False), [])                           # flag: locked even when flagged
            self.assertTrue(Loop.flavor_locked(lp, ws, 0, 1))
            self.assertFalse(Loop.flavor_locked(lp, ws, 0, 0))                          # pass: the other run is free
            lp.rewrite_cleared, ws.flagged = True, []
            self.assertEqual(Loop.targets(lp, ws, False), [(0, 0, False)])            # a rewrite never takes the locked run
            lp.best = lambda ws: b.entries[0]
            self.assertEqual(Loop.ship_entry(lp, ws)["kind"], "flavor")
            self.assertEqual(Loop.arm_of(lp, ws, 0, 0), "fewshot")                       # no arm banked: the few-shot writer
            # the whole-text round reads the shipped picks, which hold the insert, while ws.current may not
            ws.current = [[r[:] for r in p] for p in self.current]
            scores = {1: {"segs": [["Which, haha, is the whole point.", True]]}}
            picks = {"s1w1": [[r[:] for r in p] for p in flavored]}
            ws.w = {"section": 1}
            self.assertEqual(Loop.whole_targets(lp, {1: [ws]}, picks, scores), {})       # locked there too
            # a run that moved on from the text the insert sat on isn't re-flavored (the insert went with it)
            ws.current = [[["Few-shot regrowth one."], ["A newer regrowth."]]]
            self.assertEqual(Loop.apply_flavor(lp, ws), [])


# ---------------------------------------------------------------------- the analytics per arm
class ArmAnalytics(unittest.TestCase):
    def test_round_stats_split_by_arm_and_credit_the_share_change(self):
        def ch(arm, new=True, draws=48):
            c = {"para": 0, "run": 0, "orig": "o", "draws": draws, "result": "replaced (x)" if new else "no candidate passed (48 drawn: length 48)"}
            if new:
                c["new"] = "n"
            if arm:
                c["arm"] = arm
            return c
        pairs = [(0, {"kind": "start", "ai": 0.5, "runs": [[["o"]]]}),
                 (1, {"kind": "round", "ai": 0.3, "runs": [[["n"]]], "changes": [ch("notes")]}),
                 (2, {"kind": "round", "ai": 0.2, "runs": [[["n"]]], "changes": [ch("notes", new=False), ch("fewshot")]}),
                 (3, {"kind": "round", "ai": 0.1, "runs": [[["n"]]], "changes": [ch(None)]})]
        st = an.round_stats(pairs)
        arms = st["arms"]
        self.assertEqual(arms["notes"]["draws"], 96)
        self.assertEqual((arms["notes"]["attempts"], arms["notes"]["accepted"]), (2, 1))
        self.assertAlmostEqual(arms["notes"]["share_change"], 0.2)                      # 0.5 to 0.3, all notes
        self.assertAlmostEqual(arms["fewshot"]["share_change"], 0.1 + 0.1)              # the fallback's round and the old one
        self.assertEqual(arms["fewshot"]["accepted"], 2)
        self.assertEqual(an.arm_of_change(ch(None)), "fewshot")                         # before the arms: the few-shot writer
        rows = an.arm_rows(arms)
        self.assertEqual([r["arm"] for r in rows], ["fewshot", "notes"])
        self.assertEqual(rows[1]["acceptance"], 0.5)


if __name__ == "__main__":
    unittest.main()
