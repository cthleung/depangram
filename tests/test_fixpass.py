"""Tests for the 2026-11-09 build: the fix pass (plan 6.5's few-word rule as a final pass), the per-section swear
budget, and the joke rule in the judge. Each rule has a flag case and a pass case. No paid call anywhere: Sonnet and
Pangram are fakes. Run from the repo root:

    python -m unittest discover tests
"""
import json
import tempfile
import types
import unittest
from pathlib import Path

from engine.checks import gb_spellings, house_words, swear_counts, to_us, words_changed
from engine.fixpass import FixPass, code_proposals, gate, parse_reply, run_keeps
from engine.job import Bank
from engine.run import Loop

# the house words the gate keeps out, built from pieces (the repo's edit gate blocks them as literals)
DASH = "\u2014"
RZ = "struct" + "ural"

PROBABLY = "Most budget models skip the carry case, probably."
RUNS = ["Ferreira et al. (2021) listed this as a common cost cut.", PROBABLY]
ORIGINAL = "Ferreira et al. (2021) listed this as a common cost cut. Cheaper pairs tend to leave the case out, probably."


def keeps_for(runs=RUNS, notes=("their hypothesis", 'keep: "probably"')):
    return [run_keeps(r, list(notes)) for r in runs]


def fix(sentence, replacement, category="grammar"):
    return {"sentence": sentence, "replacement": replacement, "category": category, "why": "x"}


# ---------------------------------------------------------------------- piece 1: the fix pass
class Gate(unittest.TestCase):
    def check(self, prop, runs=RUNS, keeps=None, max_words=3):
        return gate(prop, list(runs), keeps or keeps_for(runs), ORIGINAL, max_words)

    def test_a_dangling_hedge_moved_back_passes(self):
        ri, new_run, _, why = self.check(fix(PROBABLY, "Most budget models probably skip the carry case."))
        self.assertIsNone(why)
        self.assertEqual((ri, new_run), (1, "Most budget models probably skip the carry case."))

    def test_the_word_limit(self):
        self.assertEqual(words_changed(PROBABLY, "Most budget models probably skip the carry case."), 3)
        why = self.check(fix(PROBABLY, "Most of the cheap models out there probably skip the carry case."))[3]
        self.assertIn("changes", why)                                                            # flag
        self.assertIsNone(self.check(fix(PROBABLY, "Most budget models leave out the case, probably."))[3])

    def test_the_sentence_has_to_stand_in_the_paragraph(self):
        self.assertIn("isn't in the paragraph", self.check(fix("Budget models have no case.", "Budget models have no cases."))[3])
        self.assertIn("same sentence", self.check(fix(PROBABLY, PROBABLY))[3])

    def test_no_sentence_added_or_removed(self):
        why = self.check(fix(PROBABLY, "Most budget models skip the carry case. Probably."))[3]
        self.assertEqual(why, "adds or removes a sentence")

    def test_only_the_allowed_categories(self):
        self.assertIn("not a fix", self.check(fix(PROBABLY, "Most budget models probably skip the carry case.",
                                                  category="style"))[3])

    def test_no_new_swear_dash_or_banned_word(self):
        self.assertEqual(self.check(fix(PROBABLY, "Most fucking budget models skip the carry case, probably."))[3],
                         "adds a swear")
        self.assertIn("em dash", self.check(fix(PROBABLY, f"Most budget models {DASH} skip the carry case, probably."))[3])
        self.assertIn("banned word", self.check(fix(RUNS[0], f"Ferreira et al. (2021) listed this as a common {RZ} cost cut."))[3])

    def test_american_spelling_only(self):
        runs = ["I've organized this recipe into Parts.", PROBABLY]
        why = gate(fix(runs[0], "I've organised this recipe into Parts.", "spelling"), runs, keeps_for(runs),
                   ORIGINAL, 3)[3]
        self.assertEqual(why, "brings in a British spelling")
        self.assertEqual(code_proposals(["I've organised this recipe into Parts."], set())[0]["replacement"],
                         "I've organized this recipe into Parts.")
        self.assertEqual(code_proposals(["I've organised this recipe into Parts."],
                                        {"I've organised this recipe into Parts."}), [])          # Sonnet's already on it
        self.assertEqual(to_us("Labelling behaviour, Colour"), "Labeling behavior, Color")
        self.assertEqual(gb_spellings("a promise, a surprise, the rise"), [])

    def test_must_keeps_stay(self):
        why = self.check(fix(PROBABLY, "Most budget models just skip the carry case."))[3]
        self.assertIn('hedge "probably"', why)                                                   # flag
        mine = ["Hargreaves keeps a ramshackle potting-shed by the fence."]
        k = [run_keeps(mine[0], [], author=True, protected=["ramshackle", "potting-shed"])]
        self.assertIn("protected word", gate(fix(mine[0], "Hargreaves keeps a shabby potting-shed by the "
                                                          "fence."), mine, k, mine[0], 3)[3])
        self.assertIsNone(gate(fix(mine[0], "Hargreaves keeps his ramshackle potting-shed by the fence."),
                               mine, k, mine[0], 3)[3])                                         # pass

    def test_no_new_name_or_number_but_the_originals_may_come_back(self):
        self.assertEqual(self.check(fix(PROBABLY, "Most budget models from Sony skip the carry case, probably."))[3],
                         "adds a name")
        runs = ["They had 12 testers try the headphones.", PROBABLY]
        orig = "They had 18 testers try the headphones."
        keeps = [run_keeps(orig, []), keeps_for()[1]]
        self.assertEqual(gate(fix(runs[0], "They had 30 testers try the headphones.", "fact"), runs, keeps, orig, 3)[3],
                         "adds a number")                                                       # flag
        self.assertIsNone(gate(fix(runs[0], "They had 18 testers try the headphones.", "fact"), runs, keeps, orig,
                               3)[3])                                                           # the drift goes back

    def test_no_punctuation_slip(self):
        self.assertIn("punctuation", self.check(fix(PROBABLY, 'Most budget models "skip the carry case, probably.'))[3])

    def test_two_fixes_on_one_sentence_count_together(self):
        fp = FixPass(job=types.SimpleNamespace(_cache=lambda: {}, protected_cached=lambda t: None), bank=None,
                     settings=None, conn=None)
        lay = {"w": {"original": [[[RUNS[0]], [PROBABLY]]], "runs": [[{"sents": [RUNS[0]], "claude": True},
                                                                     {"sents": [PROBABLY], "claude": True}]]}}
        row = {"label": "w p1", "wid": "w", "pi": 0, "runs": list(RUNS), "original": ORIGINAL, "fixable": True}
        props = {"w p1": [fix(PROBABLY, "Most budget models probably skip the carry case."),
                          fix("Most budget models probably skip the carry case.",
                              "Most budget models probably still skip the carry case.")]}
        res = fp.review([row], lay, props)["w p1"]
        self.assertEqual(len(res["accepted"]), 1)
        self.assertIn("changes", res["rejected"][0]["reason"])


class Proposals(unittest.TestCase):
    def test_the_reply_parses_fenced_or_not(self):
        got = parse_reply('```json\n{"1": [{"sentence": "a", "replacement": "b", "category": "grammar"}]}\n```', 2)
        self.assertEqual(len(got[1]), 1)
        self.assertEqual(got[2], [])
        self.assertEqual(parse_reply("no json here", 1), {1: []})

    def test_proposals_are_cached_so_the_write_run_reuses_the_dry_runs(self):
        calls = []

        def claude(settings, conn, prompt, role, model, max_budget_usd):
            calls.append(prompt)
            return json.dumps({"1": [fix(PROBABLY, "Most budget models probably skip the carry case.")]}), {}
        with tempfile.TemporaryDirectory() as d:
            st = types.SimpleNamespace(config={"fixpass": {}}, runs_dir=Path(d))
            fp = FixPass(job=types.SimpleNamespace(name="j"), bank=None, settings=st, conn=None, claude=claude)
            row = {"label": "w p1", "text": " ".join(RUNS), "original": ORIGINAL}
            self.assertEqual(len(fp.proposals([row])["w p1"]), 1)
            fp.proposals([row])
            self.assertEqual(len(calls), 1)
            self.assertIn("at most 3 words", calls[0])
            self.assertIn("Original: " + ORIGINAL, calls[0])


GUIDE = ("# Title\n\n<!-- planning: keep me -->\n\n## 1. First\n\n" + " ".join(RUNS) + "\n\n<!-- BEATS -->\n\n"
         "A paragraph you wrote **later** by hand.\n\n## 2. Second\n\nUntouched.\n")


class FakePangram:
    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def cached(self, text):
        return None

    def estimate(self, text):
        return 1, 0.05

    def score(self, text, purpose, label):
        self.calls.append((label, purpose))
        ai = self.scores["after"] if "after" in label else self.scores["before"]
        return {"fraction_ai": ai, "windows": []}, False


class Apply(unittest.TestCase):
    def setUp(self):
        self.row = {"label": "w p1", "section": 1, "wid": "w", "pi": 0, "text": " ".join(RUNS), "runs": list(RUNS),
                    "original": ORIGINAL, "fixable": True}
        self.results = {"w p1": {"runs": [RUNS[0], "Most budget models probably skip the carry case."],
                                 "accepted": [fix(PROBABLY, "x")], "rejected": []}}

    def fp(self, before, after):
        st = types.SimpleNamespace(config={"fixpass": {"noise": 0.05}}, runs_dir=Path("."))
        return FixPass(job=types.SimpleNamespace(name="j"), bank=None, settings=st, conn=None,
                       pangram=FakePangram({"before": before, "after": after}))

    def test_only_the_fixed_block_changes(self):
        new, rep = self.fp(0, 0).section_text(GUIDE, [self.row], self.results, 1)
        self.assertEqual(rep["replaced"], ["w p1"])
        self.assertEqual(new.replace("models probably skip the carry case.", "models skip the carry case, probably."), GUIDE)

    def test_a_section_whose_score_rises_past_the_noise_keeps_its_old_text(self):
        raw, out, kept = self.fp(0.20, 0.30).apply_sections(GUIDE, [self.row], self.results, [1])
        self.assertEqual((raw, kept), (GUIDE, []))                                              # flag
        self.assertIn("rose by 0.10", out[1]["result"])
        raw, out, kept = self.fp(0.20, 0.23).apply_sections(GUIDE, [self.row], self.results, [1])
        self.assertEqual(kept, [1])                                                             # pass: inside the noise
        self.assertIn("probably skip the carry case.", raw)

    def test_scores_from_two_pangram_versions_dont_decide(self):
        fp = self.fp(0.20, 0.20)
        real = fp.pg.score
        fp.pg.score = lambda text, purpose, label: (dict(real(text, purpose, label)[0],
                                                          version="4.1" if "before" in label else "4.2"), False)
        raw, out, kept = fp.apply_sections(GUIDE, [self.row], self.results, [1])
        self.assertEqual(kept, [])                                                              # flag
        self.assertIn("4.1", out[1]["result"])
        fp = self.fp(0.20, 0.20)
        self.assertEqual(fp.apply_sections(GUIDE, [self.row], self.results, [1])[2], [1])       # pass: same version

    def test_the_pass_stops_at_its_own_pangram_limit(self):
        fp = self.fp(0.2, 0.2)
        fp.max_usd = 0.05
        raw, out, kept = fp.apply_sections(GUIDE, [self.row], self.results, [1])
        self.assertEqual(kept, [])
        self.assertIn("limit", out[1]["result"])

    def test_a_fix_entry_ships_like_a_whole_text_pick(self):
        lp = Loop.__new__(Loop)
        with tempfile.TemporaryDirectory() as d:
            lp.bank = Bank(Path(d) / "bank.jsonl")
            ws = types.SimpleNamespace(wid="w", runs=[[{"sents": ["C."], "claude": True}]], original=[[["C."]]],
                                       must={(0, 0): {"hedges": [], "repeats": []}})
            lp.best = lambda ws: next((e for e in reversed(lp.bank.entries) if e.get("ai") is not None), None)
            lp.bank.add({"kind": "round", "window": "w", "ai": 0.10, "text": "a", "runs": [[["Best."]]]})
            lp.bank.add({"kind": "fix", "window": "w", "ai": None, "text": "b", "runs": [[["Best, fixed."]]]})
            self.assertEqual(Loop.ship_entry(lp, ws)["runs"], [[["Best, fixed."]]])
            # a newer window best inside the noise floor of what the fix stood on (0.10) leaves the fix shipping
            # (2026-11-10 evening: a tie never overrides); one strictly better takes over
            lp.bank.add({"kind": "round", "window": "w", "ai": 0.08, "text": "c", "runs": [[["Newer best."]]]})
            self.assertEqual(Loop.ship_entry(lp, ws)["runs"], [[["Best, fixed."]]])
            lp.bank.add({"kind": "round", "window": "w", "ai": 0.0, "text": "d", "runs": [[["Much better."]]]})
            self.assertEqual(Loop.ship_entry(lp, ws)["runs"], [[["Much better."]]])


class Words(unittest.TestCase):
    def test_house_words_and_swears(self):
        self.assertEqual(house_words(f"a {RZ} {DASH} thing -- framework"), [DASH, RZ, "framework", "--"])
        self.assertEqual(house_words("a plain sentence"), [])
        self.assertEqual(swear_counts("a clusterfuck of damning shit, busted-ass"), (6, 3, 1))


# ---------------------------------------------------------------------- piece 2: the per-section swear budget
class SwearFilter(unittest.TestCase):
    def test_the_loop_puts_it_in_the_filter_stack(self):
        lp = Loop.__new__(Loop)
        lp.model, lp.header, lp.examples_block = "m", "samples", "examples"
        lp.fmt_example = lambda n, p: p
        lp.swear_rates, lp.swear_mult = {"all": 0.01, "strong": 0.001}, 1.5
        lp.section_swear_rest = lambda ws, pi, ri: (1000, 5, 5)
        from engine.run import WindowState
        ws = WindowState({"wid": "w", "text": "", "section": 1, "index": 0})
        ws.runs = [[{"sents": ["The model has no clue at all."], "claude": True}]]
        ws.original = [[["The model has no clue at all."]]]
        ws.current = [[["The model has no clue at all."]]]
        ws.notes, ws.forced = [["the point"]], set()
        ws.must = {(0, 0): {"hedges": [], "repeats": []}}
        ra = Loop.run_args(lp, ws, 0, 0, False, 1, False)
        reasons = [f("The model has no fucking clue at all.") for f in ra["checks"]]
        self.assertIn("section swearing", reasons)                                              # flag
        self.assertFalse(any(f("The model hasn't got a clue at all.") for f in ra["checks"]))  # pass


# ---------------------------------------------------------------------- piece 3: the joke rule in the judge
VIBES = "Battery life? Think goldfish."
VIBES_NOTES = ["The battery runs down fast.",
               "Joke: the author compares the battery to a short-lived pet."]
PUNCHY = "Battery life? Mayfly territory, basically."
FLAT = "The battery does not last very long on a single charge."
FACT = "A full charge takes about 2 hours, by the maker's count."


class JokeRule(unittest.TestCase):
    def judge(self, orig, notes, shown, verdicts, h2h="SAME"):
        """judge_pool with a fake Sonnet: the judge's reply comes from `verdicts(prompt)`, the head-to-head says
        `h2h`, the fact check says NONE. Returns (new text or None, every prompt sent)."""
        from engine.draws import judge_pool
        prompts = []

        def fake(settings, conn, prompt, role, model, max_budget_usd):
            prompts.append((role, prompt))
            if role.endswith("judge"):
                return json.dumps(verdicts(prompt)), {}
            return (h2h if role.endswith("h2h") else "NONE"), {}
        pool = {"stats": {}, "origin": {}, "ranked": [(c, 0.1) for c in shown], "shown": list(shown)}
        new, _, _ = judge_pool(pool, settings=None, conn=None, tics=lambda c: (0, []), orig_text=orig, cur_text=orig,
                               is_original=True, notes=notes, before="", after="", pi=0, ri=0, rnd=1, rounds=2,
                               role="w", claude=fake)
        return new, prompts

    def test_the_judge_and_the_head_to_head_ask_about_the_punchline(self):
        _, prompts = self.judge(VIBES, VIBES_NOTES, [FLAT, PUNCHY], lambda p: {"pick": 2, "drops": [], "why": "x"})
        judge = dict(prompts)["w judge"]
        self.assertIn("The stretch as first written:\n" + VIBES, judge)
        self.assertIn("the pick has to end on one too", judge)
        self.assertIn('write "punchline"', judge)
        self.assertIn("Joke: the author compares the battery", judge)          # the notes' joke lines ride along
        self.assertIn("the one with the joke is the better piece of writing", dict(prompts)["w h2h"])
        self.assertNotIn("<<", judge + dict(prompts)["w h2h"])

    def test_a_pick_that_explains_the_punchline_away_is_rejected(self):
        new, _ = self.judge(VIBES, VIBES_NOTES, [FLAT], lambda p: {"pick": 1, "drops": ["punchline"], "why": "x"})
        self.assertIsNone(new)                                                                   # flag
        new, _ = self.judge(VIBES, VIBES_NOTES, [FLAT, PUNCHY], lambda p: {"pick": 2, "drops": [], "why": "x"})
        self.assertEqual(new, PUNCHY)                                                            # pass

    def test_a_plain_factual_run_doesnt_need_a_joke(self):
        notes = ['A full charge takes "2" hours.']
        new, prompts = self.judge(FACT, notes, ["By the maker's count, a full charge takes about 2 hours."],
                                  lambda p: {"pick": 1, "drops": [], "why": "x"})
        self.assertIsNotNone(new)
        self.assertIn("A stretch that ends on a plain point needs no joke.", dict(prompts)["w judge"])
        self.assertNotIn("notes mention", dict(prompts)["w judge"])

    def test_a_joke_cut_on_purpose_takes_the_rule_off(self):
        from engine.draws import joke_cut, joke_notes
        notes = VIBES_NOTES + ["joke cut: the author dropped the goldfish line"]
        self.assertTrue(joke_cut(notes))
        self.assertEqual(joke_notes(notes), [VIBES_NOTES[1]])
        _, prompts = self.judge(VIBES, notes, [FLAT], lambda p: {"pick": 1, "drops": [], "why": "x"})
        self.assertNotIn("first written", dict(prompts)["w judge"])
        self.assertNotIn("punchline", dict(prompts)["w judge"])
        self.assertNotIn("with the joke", dict(prompts)["w h2h"])


# ---------------------------------------------------------------------- the citation sentence (section 1 diagnosis)
CHAK = ('Ferreira et al. (2021), whose 18 testers wore a stack of AI-tuned earbuds for a month, put the problem under'
        ' its own label, "Poor Fit and Seal" (section 2.3.1 of the report), and they blame the cheap foam tips that'
        ' ship in the box.')
WHICH = "Which, as the reviews say, budget pairs never fix."


class CitationSentence(unittest.TestCase):
    def test_footnote_marks_come_out_before_the_number_check(self):
        from engine.checks import strip_footnote_marks
        self.assertEqual(strip_footnote_marks("1 Ferreira et al. (2021) had 18 testers rate them.", CHAK),
                         "Ferreira et al. (2021) had 18 testers rate them.")                # flag: a leading mark
        self.assertEqual(strip_footnote_marks("In Ferreira et al. (2021) 3, and their verdict.", CHAK),
                         "In Ferreira et al. (2021), and their verdict.")
        self.assertEqual(strip_footnote_marks("They called it that2, in the report.", CHAK), "They called it that, in the report.")
        kept = "18 testers, section 2.3.1, about 3/4 of them."
        self.assertEqual(strip_footnote_marks(kept, CHAK + " about 3/4"), kept)                 # pass: real numbers stay
        self.assertEqual(strip_footnote_marks("Part 1 is next.", "Part 1 is where we start."), "Part 1 is next.")

    def test_a_citation_without_its_brackets_still_keeps_it(self):
        from engine.checks import citation_kept, must_kept, names_missing
        self.assertTrue(citation_kept("(2021)", "Ferreira et al. 2021 had testers rate them.", CHAK))
        self.assertTrue(citation_kept("(2021)", "the 2021 Ferreira paper had testers rate them.", CHAK))
        self.assertFalse(citation_kept("(2021)", "a 2021 paper had testers rate them.", CHAK))     # flag: year alone
        self.assertFalse(must_kept("(2021)", "Ferreira and friends had testers rate them.", CHAK))
        self.assertTrue(must_kept("(2021)", "Ferreira et al. 2021 had testers rate them.", CHAK))   # pass: the filter's view
        self.assertTrue(citation_kept("(Acme, 2023)", "the Acme, 2023 paper", "built it (Acme, 2023) like"))
        self.assertEqual(names_missing("the 2021 Ferreira paper", ["Ferreira et al."]), [])
        self.assertEqual(names_missing("a 2021 paper", ["Ferreira et al."]), ["Ferreira et al."])

    def test_the_citing_name_is_no_new_name(self):
        from engine.draws import names_ok, run_names
        known = {"AI"}                            # names() skips "Ferreira", since it opens its sentence
        cand = "In a 2021 paper, Ferreira et al. had 18 testers rate AI-tuned earbuds."
        self.assertFalse(names_ok(cand, known))                                                  # the old miss
        self.assertTrue(names_ok(cand, known | run_names(CHAK)))                                 # pass
        self.assertFalse(names_ok("Then Gemini had 18 testers rate them.", known | run_names(CHAK)))  # flag

    def test_citations_go_back_exact(self):
        from engine.checks import splice_citations
        self.assertEqual(splice_citations("Ferreira et al. 2021 had testers.", CHAK),
                         "Ferreira et al. (2021) had testers.")
        self.assertEqual(splice_citations("Ferreira et al., (2021) had testers.", CHAK),
                         "Ferreira et al. (2021) had testers.")
        self.assertEqual(splice_citations("The 2021 Ferreira paper had testers.", CHAK),
                         "The 2021 Ferreira paper had testers.")                              # nothing to put back

    def test_a_short_title_is_spliced_and_the_model_writes_the_rest(self):
        from engine.draws import draw_pool, short_title
        self.assertEqual(short_title(CHAK), '"Poor Fit and Seal"')
        self.assertIsNone(short_title('its job is literally "taste as you go".'))           # not a title
        self.assertIsNone(short_title('after the line "brown the onions first", words'))

        class Srv:
            def __init__(self):
                self.calls = []

            def complete(self, prompt, n_predict=0, temperature=1.0, min_p=0.05, seed=0, stop=None, ban=None):
                self.calls.append((prompt, stop))
                if '"' in (stop or []):
                    return "Ferreira et al. (2021) had 18 testers wear AI-tuned earbuds for a month and named this", {}
                return (" (section 2.3.1 of the report) after the way they sound, and they blame the foam tips."
                        " Which, as the reviews say, budget pairs never fix."), {}

        class Op:
            def score(self, cands):
                return [{"score": 0.1} for _ in cands]
        srv = Srv()
        pool = draw_pool(srv=srv, op=Op(), tics=lambda c: (0, []), prompt="P ", orig_text=CHAK, n_sent=1,
                         is_original=True, known_names={"AI"}, swears_elsewhere=0, swear_budget=5,
                         digit_ban=None, pi=0, ri=3, rnd=1, rounds=6, next_sent=WHICH)
        self.assertTrue(pool["shown"], pool["tally"])
        c = pool["shown"][0]
        self.assertIn('named this "Poor Fit and Seal" (section 2.3.1 of the report)', c)
        self.assertNotIn("cheap foam tips", c)                    # Claude's words after the title aren't kept
        self.assertNotIn("Which, as the reviews say", c)           # the run-on into the next run is cut back
        self.assertEqual(srv.calls[1][0], 'P Ferreira et al. (2021) had 18 testers wear AI-tuned earbuds for a month'
                                          ' and named this "Poor Fit and Seal"')

    def test_a_run_on_into_the_next_sentence_is_cut_back_not_thrown_out(self):
        from engine.checks import trim_continuation
        two = "It comes from the foam tips. Which, as the reviews say, budget pairs never fix it."
        self.assertEqual(trim_continuation(two, 1, WHICH), "It comes from the foam tips.")   # flag
        self.assertEqual(trim_continuation(two, 1, "Anyway, on to the tells."), two)                 # pass: unrelated
        self.assertEqual(trim_continuation(two, 2, WHICH), two)                                     # a two-sentence run


if __name__ == "__main__":
    unittest.main()
