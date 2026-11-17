"""Tests for the paragraph pass (engine/regrow.py, 2026-11-14): the units (which paragraphs regrow, where the span
starts), the gate (a flag case and a pass case per rule), the prompts, the scoring memory (a humanized result never
counts as best, the ceiling holds) and the flagged-span mapping. No model, no Pangram, no Claude. Run from
the repo root:

    python -m unittest discover tests
"""
import json
import tempfile
import unittest
from pathlib import Path

from engine.job import Bank
from engine.regrow import (MAX_RATIO, RegrowPass, adapter_prompt, fewshot_prompt, gate, keep_words, new_names,
                           normalize, olmo_answer, olmo_prompt, quote_stops, seed_for, span_of, spans_of, units)

DASH = "\u2014"
ORIG = ('Lindqvist et al. (2023) logged 31,862 commutes, and riders on painted lanes braked hard at junctions 58% of'
        ' the time, next to 19% on separated tracks. The report sums it up: "Paint tells drivers where cyclists should'
        ' be, not where they actually are." So the corner is where most close calls happen.')
NOTES = ['Cite "Lindqvist et al. (2023)", a study of "31,862" logged commutes.',
         'On painted lanes riders braked hard at junctions "58%" of the time, on separated tracks "19%".',
         'Quote: "Paint tells drivers where cyclists should be, not where they actually are."',
         'Most close calls happen at the corner.', 'keep: "probably"']
GOOD = ('Lindqvist et al. (2023) followed 31,862 trips to work and counted hard stops at junctions in 58% of cases on'
        ' painted lanes, against 19% for the protected tracks, and their line beats mine: "Paint tells drivers where'
        ' cyclists should be, not where they actually are." Which is probably why the corner is where it goes wrong.')
KNOWN = {"Lindqvist", "Pangram"}


def check(cand, **kw):
    args = dict(orig=ORIG, standing=ORIG, notes=NOTES, prefix="", after="", known=KNOWN, tics=lambda c: (0, []))
    args.update(kw)
    return gate(cand, **args)


def window(runs, section=5):
    """A layout window from a list of paragraphs, each a list of (sentence, who) with who in C, A or L (locked)."""
    out_runs, original = [], []
    for p in runs:
        rr = [{"sents": [s], "claude": who == "C", "author": who == "A"} for s, who in p]
        out_runs.append(rr)
        original.append([[s] for s, _ in p])
    return {"section": section, "heading": None, "runs": out_runs, "original": original}


class Units(unittest.TestCase):
    def test_an_all_claude_paragraph_is_one_span_from_the_start(self):
        w = window([[("One.", "C"), ("Two.", "C")]])
        self.assertEqual(spans_of(w, 0), [(0, 2)])
        self.assertEqual(span_of(w, 0), {"start": 0, "prefix_runs": []})

    def test_a_line_of_yours_is_an_anchor_and_the_claude_text_around_it_is_regrown(self):
        w = window([[("One.", "C"), ("Mine.", "A"), ("Three.", "C")]])
        self.assertEqual(spans_of(w, 0), [(0, 1), (2, 3)])                                           # pass: both sides
        w = window([[("Mine.", "A"), ("Also mine.", "A")]])
        self.assertEqual(spans_of(w, 0), {"why": "nothing of Claude's"})                              # flag

    def test_a_locked_sentence_is_an_anchor(self):
        w = window([[("One.", "C"), ("X and X and X.", "L"), ("Three.", "C"), ("Four.", "C")]])
        self.assertEqual(spans_of(w, 0), [(0, 1), (2, 4)])
        self.assertEqual(span_of(w, 0), {"start": 2, "prefix_runs": [0, 1]})                           # the last span
        w = window([[("One.", "C"), ("X and X and X.", "L")]])
        self.assertEqual(spans_of(w, 0), [(0, 1)])                                                    # the lead, not held

    def test_units_take_the_standing_text_and_split_prefix_and_suffix(self):
        w = window([[("One.", "C"), ("X and X and X.", "L"), ("Three.", "C")], [("Only.", "C")]])
        lay = {"s5w0": w}
        got, skipped = units(lay, [5], {"s5w0 p1": ["Uno.", "X and X and X.", "Tres."]}, {"s5w0 p2": "Solo."})
        self.assertEqual(skipped, [])
        u1a, u1, u2 = got
        self.assertEqual((u1a["label"], u1a["prefix"], u1a["orig"], u1a["standing"], u1a["suffix"]),
                         ("s5w0 p1 s1", "", "One.", "Uno.", "X and X and X. Tres."))
        self.assertEqual((u1["label"], u1["prefix"], u1["orig"], u1["standing"], u1["suffix"]),
                         ("s5w0 p1", "Uno. X and X and X.", "Three.", "Tres.", ""))
        self.assertEqual((u2["prefix"], u2["orig"], u2["standing"], u2["para_standing"]), ("", "Only.", "Solo.", "Solo."))

    def test_a_section_out_of_scope_is_left_out(self):
        lay = {"s5w0": window([[("Only.", "C")]]), "s6w0": window([[("Six.", "C")]], section=6)}
        got, _ = units(lay, [6], {})
        self.assertEqual([u["label"] for u in got], ["s6w0 p1"])


class Gate(unittest.TestCase):
    def test_a_good_rewrite_passes(self):
        self.assertIsNone(check(GOOD))

    def test_copies_and_length(self):
        self.assertEqual(check(ORIG), "copied the original")                                           # flag
        self.assertEqual(check("Short."), "length 0.02")                                               # flag
        long = GOOD + " " + " ".join(["And on it goes with more words."] * 12)
        self.assertTrue(check(long).startswith("length"))                                              # flag
        self.assertIsNone(check(GOOD))                                                                 # pass

    def test_quotes_numbers_names_and_keeps(self):
        self.assertTrue(check(GOOD.replace('"Paint tells drivers where cyclists should be, not where they actually are."',
                                           '"the AI explains everything."')).startswith("quote lost"))  # flag
        self.assertTrue(check(GOOD.replace("31,862", "many")).startswith("number lost"))              # flag
        self.assertTrue(check(GOOD.replace("58%", "58% of 900")).startswith("new number"))             # flag
        self.assertTrue(check(GOOD.replace("Lindqvist et al. (2023)", "A 2023 paper")).startswith(("quote lost", "name lost")))  # flag
        self.assertEqual(check(GOOD.replace("probably", "surely")), "keep lost: probably")             # flag
        self.assertIsNone(check(GOOD.replace("probably", "surely"), keeps=False))                     # pass: the pass's setting
        self.assertEqual(check(GOOD.replace("the protected tracks", "Jane Doe")), "new name: Jane, Doe")      # flag
        self.assertIsNone(check(GOOD))                                                                 # pass

    def test_a_capital_after_a_quote_mark_or_in_quote_punctuation_is_an_opener(self):
        swap = ('Which is probably why ("you never looked") Or, "Who signals? Is anyone watching?"'
                ' decides the corner.')
        self.assertIsNone(check(GOOD.replace("Which is probably why the corner is where it goes wrong.", swap)))  # pass
        self.assertEqual(check(GOOD + " Like a Hallmark card."), "new name: Hallmark")                   # flag
        self.assertEqual(new_names("It skipped Section 2 and the Hallmark bit.", KNOWN), ["Hallmark"])      # flag / pass

    def test_house_words_markup_tells_and_furniture(self):
        self.assertEqual(check(GOOD.replace(", against", f" {DASH} against")), "dash or house word")   # flag
        self.assertEqual(check(GOOD.replace("beats mine", "**beats** mine")), "markup")                # flag
        self.assertEqual(check("Here's the thing: " + GOOD), "assistant tell")                         # flag
        self.assertEqual(check("TL;DR " + GOOD), "reddit tell")                                        # flag
        self.assertEqual(check("Sure, here's a version. " + GOOD), "assistant costume")                # flag
        self.assertEqual(check(GOOD + " I've seen it do this a dozen times."), "invented sighting")   # flag
        self.assertTrue(check(GOOD.replace("(2023)", "(2023")).startswith("punctuation"))              # flag
        self.assertEqual(check(GOOD, tics=lambda c: (1, [])), "hard tic")                              # flag

    def test_register_and_restating(self):
        low = GOOD.replace("Lindqvist", "lindqvist").replace("Which", "which").replace("So", "so")
        self.assertEqual(check(low.lower().replace("lindqvist et al. (2023)", "Lindqvist et al. (2023)")), "lowercase register")  # flag
        twice = GOOD + " The corner is the trouble spot, probably. The corner is the trouble spot, probably."
        self.assertEqual(check(twice), "says it twice")                                                # flag
        self.assertIsNone(check(GOOD))                                                                 # pass

    def test_the_section_swear_budget_and_the_filler_cap(self):
        rates = {"all": 0.005, "strong": 0.002}
        sweary = GOOD.replace("beats mine", "fucking beats mine")
        self.assertEqual(check(sweary, rest=(1000, 8, 3, 0), rates=rates), "section swearing")        # flag: no room
        self.assertIsNone(check(sweary, rest=(1000, 1, 0, 0), rates=rates))                            # pass: room left
        self.assertTrue(check(GOOD.replace("Which is", "Which is, honestly,"), rest=(1000, 1, 0, 1), rates=rates)
                        .startswith("filler tag"))                                                     # flag
        self.assertIsNone(check(GOOD, rest=(1000, 8, 3, 1), rates=rates))                              # pass: adds none


class Normalize(unittest.TestCase):
    def test_british_spelling_is_turned_not_rejected(self):
        c = normalize("My favourite colour is the one Lindqvist et al. (2023) found.", ORIG)
        self.assertEqual(c, "My favorite color is the one Lindqvist et al. (2023) found.")

    def test_a_nearly_right_quote_and_citation_come_back_exact(self):
        c = normalize('Lindqvist et al 2023 said "Paint tells drivers where cyclists should be, not where they actually are".', ORIG)
        self.assertIn("Lindqvist et al. (2023)", c)
        self.assertIn('"Paint tells drivers where cyclists should be, not where they actually are."', c)

    def test_a_stop_outside_a_spliced_quote_goes(self):
        self.assertEqual(quote_stops('more than ChatGPT.". Next, "the words,", "it,", or "hard,". This'),
                         'more than ChatGPT." Next, "the words," "it," or "hard." This')
        self.assertEqual(quote_stops('she said "fine." Then "what?" and left...'), 'she said "fine." Then "what?" and left...')

    def test_keep_words(self):
        self.assertEqual(keep_words(NOTES), ["probably"])
        self.assertEqual(keep_words(NOTES[:-1]), [])


class Prompts(unittest.TestCase):
    def test_the_adapter_prompt_is_flush_after_the_prefix(self):
        p = adapter_prompt(["one", "two"], "So far.")
        self.assertTrue(p.endswith("</notes>\n<target_text>\nSo far."))
        self.assertTrue(adapter_prompt(["one"], "").endswith("<target_text>\n"))

    def test_the_few_shot_prompt_ends_on_the_notes_and_a_space_only_with_a_prefix(self):
        p = fewshot_prompt("HEADER", "EXAMPLES", ["one"], "")
        self.assertTrue(p.endswith("Notes:\n- one\n\nParagraph:\n"))
        self.assertTrue(fewshot_prompt("HEADER", "EXAMPLES", ["one"], "So far.").endswith("Paragraph:\nSo far. "))


class OlmoArm(unittest.TestCase):
    def test_the_prompt_carries_the_notes_the_prefix_and_the_suffix(self):
        p = olmo_prompt("SAMPLE", NOTES, "Before.", "After.")
        for piece in ("SAMPLE", "- " + NOTES[0], "Before.", "After.", "Write the rest of the paragraph"):
            self.assertIn(piece, p)
        self.assertIn("Write the paragraph from", olmo_prompt("SAMPLE", NOTES, "", ""))
        self.assertNotIn("What follows", olmo_prompt("SAMPLE", NOTES, "", ""))

    def test_the_answer_loses_its_preamble_and_wrapping_quotes(self):
        self.assertEqual(olmo_answer("Here is the paragraph:\n\n" + GOOD), GOOD)                     # preamble out
        self.assertEqual(olmo_answer('"' + GOOD + '"'), GOOD)                                        # quotes out
        self.assertEqual(olmo_answer("One line.\n\nTwo line."), "One line. Two line.")               # joined for the length rule

    def test_an_instruct_preamble_is_an_assistant_tell(self):
        self.assertTrue(check("Okay, I'll do my best to follow your instructions here. " + GOOD).startswith("assistant"))  # flag
        self.assertEqual(check(GOOD + " The paragraph will be in the author's own voice."), "assistant tell")        # flag

    def test_the_arms_draw_on_different_seeds(self):
        u = {"wid": "s6w2", "pi": 1}
        self.assertEqual(len({seed_for(u, a, 0) for a in ("notes", "fewshot", "olmo")}), 3)

    def test_a_draw_on_the_olmo_arm_goes_through_the_chat_call_and_the_gate(self):
        class FakeChat:
            def __init__(self):
                self.calls = []

            def chat(self, user, **kw):
                self.calls.append(user)
                return "Sure! Here you go:\n\n" + GOOD, {}
        rp = RegrowPass.__new__(RegrowPass)
        rp.srv, rp.op, rp.pools, rp.known, rp.tics, rp.rates = FakeChat(), None, {}, KNOWN, (lambda c: (0, [])), None
        rp.swear_mult, rp.filler_cap, rp.filler_words, rp.header = 1.5, 1, (), "SAMPLE"
        rp.notes = {"s5w0 p1": NOTES}
        rp.section_rest = lambda u, picks: None
        unit = {"label": "s5w0 p1", "wid": "s5w0", "pi": 0, "orig": ORIG, "standing": ORIG, "prefix": "", "suffix": ""}
        with tempfile.TemporaryDirectory() as d:
            rp.rbank = Bank(Path(d) / "regrow bank.jsonl")
            e = rp.draw(unit, "olmo", 0, {})
        self.assertIsNone(e["why"])                                                                 # pass
        self.assertEqual(e["text"], GOOD)
        self.assertIn("- " + NOTES[0], rp.srv.calls[0])


class FakePangram:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def cached(self, text):
        return None

    def estimate(self, text):
        return 1, 0.50

    def score(self, text, purpose, label=""):
        self.calls.append(label)
        a = self.answers.pop(0)
        return {"fraction_ai": a["ai"], "windows": [{"text": t, "ai_assistance_score": 1.0 if f else 0.0,
                                                    "is_humanized": a.get("humanized", False)} for t, f in a["segs"]]}, False


class Scoring(unittest.TestCase):
    def pass_(self, answers, max_usd=10.0):
        rp = RegrowPass.__new__(RegrowPass)
        rp.pg, rp.max_usd, rp.spent, rp.purpose, rp.log = FakePangram(answers), max_usd, 0.0, "experiment", lambda s: None
        rp.scores, rp.best = [], {}
        rp.lay = {"s5w0": window([[("One.", "C")], [("Two.", "C")]])}
        rp.units = [{"label": "s5w0 p1", "plabel": "s5w0 p1", "wid": "s5w0", "pi": 0, "section": 5, "start": 0, "end": 1,
                     "prefix": "", "suffix": "", "runs": ["One."], "standing": "One.", "para_standing": "One."},
                    {"label": "s5w0 p2", "plabel": "s5w0 p2", "wid": "s5w0", "pi": 1, "section": 5, "start": 0, "end": 1,
                     "prefix": "", "suffix": "", "runs": ["Two."], "standing": "Two.", "para_standing": "Two."}]
        rp.section_text = lambda n, picks: " ".join(picks.get(f"s5w0 p{i}", t) for i, t in ((1, "One."), (2, "Two.")))
        with tempfile.TemporaryDirectory() as d:
            rp.rbank = Bank(Path(d) / "regrow bank.jsonl")
            yield rp

    def test_flags_map_to_spans_and_a_humanized_result_is_never_best(self):
        answers = [{"ai": 0.40, "segs": [("Uno. Two.", True)]},
                   {"ai": 0.10, "segs": [("Uno. Dos.", False)], "humanized": True},
                   {"ai": 0.30, "segs": [("Uno.", False), ("Dos.", True)]}]
        for rp in self.pass_(answers):
            s1 = rp.score_section(5, {"s5w0 p1": "Uno."}, "r1", "notes")
            self.assertEqual(s1["span_flagged"], {"s5w0 p1": True})                                 # flag
            s2 = rp.score_section(5, {"s5w0 p1": "Uno.", "s5w0 p2": "Dos."}, "r2", "notes")
            self.assertTrue(s2["humanized"])
            self.assertIs(rp.best[5], s1)                                                            # humanized never best
            s3 = rp.score_section(5, {"s5w0 p1": "Uno.", "s5w0 p2": "Dos."}, "r3", "mixed")
            self.assertEqual(s3["span_flagged"], {"s5w0 p1": False, "s5w0 p2": True})               # pass / flag
            self.assertIs(rp.best[5], s3)
            self.assertEqual(rp.best_picks()["s5w0 p1"]["flagged"], False)

    def test_regate_rejudges_banked_draws_and_banks_the_change(self):
        for rp in self.pass_([]):
            rp.units = [{"label": "s5w0 p1", "section": 5, "prefix": "", "suffix": "", "orig": ORIG, "standing": ORIG}]
            rp.notes, rp.known, rp.tics, rp.rates, rp.op = {"s5w0 p1": NOTES}, KNOWN, (lambda c: (0, [])), None, None
            rp.swear_mult, rp.filler_cap, rp.filler_words = 1.5, 1, ()
            rp.section_rest = lambda u, picks: None
            stale = {"kind": "draw", "label": "s5w0 p1", "arm": "notes", "k": 0, "text": GOOD, "why": "new name: Or"}
            still = {"kind": "draw", "label": "s5w0 p1", "arm": "notes", "k": 1, "text": GOOD + " Like a Hallmark card.",
                     "why": "new name: Hallmark"}
            rp.pools = {("s5w0 p1", "notes"): {0: stale, 1: still}}
            rp.regate()
            self.assertIsNone(rp.pools[("s5w0 p1", "notes")][0]["why"])                                  # rejudged
            self.assertEqual(rp.pools[("s5w0 p1", "notes")][1], still)                                   # unchanged, not rebanked
            banked = [e for e in rp.rbank.entries if e.get("kind") == "draw"]
            self.assertEqual([(e["k"], e["why"]) for e in banked], [(0, None)])

    def test_spent_today_reads_the_pass_s_own_calls(self):
        import sqlite3
        import time
        for rp in self.pass_([]):
            rp.conn = None
            self.assertEqual(rp.spent_today(), 0.0)
            rp.conn = sqlite3.connect(":memory:")
            rp.conn.execute("CREATE TABLE pangram_calls (ts REAL, usd REAL, label TEXT)")
            now = time.time()
            rp.conn.executemany("INSERT INTO pangram_calls VALUES (?, ?, ?)",
                                [(now, 0.65, "regrow s4 all-notes"), (now, 0.80, "regrow s5 round 2"),
                                 (now, 4.05, "run guide as it stands"), (now - 2 * 86400, 0.70, "regrow s6 all-notes")])
            self.assertAlmostEqual(rp.spent_today(), 1.45)                      # the loop's call and yesterday's left out

    def test_a_paragraph_is_assembled_around_its_anchors(self):
        for rp in self.pass_([]):
            rp.units = [{"label": "s5w0 p1 s1", "plabel": "s5w0 p1", "wid": "s5w0", "pi": 0, "start": 0, "end": 1,
                         "runs": ["One.", "Mine.", "Three.", "Four."], "standing": "One.", "prefix": "", "suffix": "Mine. Three. Four."},
                        {"label": "s5w0 p1", "plabel": "s5w0 p1", "wid": "s5w0", "pi": 0, "start": 2, "end": 4,
                         "runs": ["One.", "Mine.", "Three.", "Four."], "standing": "Three. Four.", "prefix": "One. Mine.", "suffix": ""}]
            self.assertEqual(rp.para_text("s5w0", 0, {}, "standing"), "standing")
            self.assertEqual(rp.para_text("s5w0", 0, {"s5w0 p1": "Tres y cuatro."}, ""), "One. Mine. Tres y cuatro.")
            self.assertEqual(rp.para_text("s5w0", 0, {"s5w0 p1 s1": "Uno.", "s5w0 p1": "Tres."}, ""), "Uno. Mine. Tres.")
            old = {"kind": "score", "picks": {"s5w0 p1": "One. Mine. Tres y cuatro."}, "ai": 0.3}
            self.assertEqual(rp.span_picks(old)["picks"], {"s5w0 p1": "Tres y cuatro."})      # a first-version pick, converted
            new = dict(old, spans=True)
            self.assertIs(rp.span_picks(new), new)                                                 # already spans: untouched

    def test_the_ceiling_holds(self):
        answers = [{"ai": 0.40, "segs": [("Uno.", False)]}, {"ai": 0.40, "segs": [("Uno.", False)]}]
        for rp in self.pass_(answers, max_usd=0.75):
            self.assertIsNotNone(rp.score_section(5, {"s5w0 p1": "Uno."}, "r1", "notes"))
            self.assertIsNone(rp.score_section(5, {"s5w0 p1": "Dos."}, "r2", "notes"))               # flag: $1.00 > $0.75
            self.assertEqual(rp.pg.calls, ["regrow r1"])


def lay2():
    return {"s5w0": window([[("One.", "C")], [("Two.", "C")]])}


def fake_load_units(self, sections, vm=None):
    """load_units without a job: the two-paragraph layout above, its units and the cross-section index."""
    self.lay = lay2()
    self.units, self.skipped = units(self.lay, sections, {})
    self.standing, self.swaps = {}, []
    self.index_units()


def fake_section_text(self, n, picks):
    return " ".join(self.para_text("s5w0", pi, picks, t) for pi, t in enumerate(("One.", "Two.")))


VETO = {"label": "s5w0 p1", "orig": "One.", "text": "Uno.", "why": "garbles the callback"}
FIX = {"label": "s5w0 p2", "orig": "Two.", "text": "Dos, by hand.", "went_wrong": "lost the quote",
       "about": "templates", "point": "one template", "fact": "the quote is the author's", "when": "2026-11-14"}


REPAIR = {"label": "s5w0 p1", "orig": "One.", "from": "Un, you know.", "text": "Un.", "moves": "cut: ', you know'",
          "by": "reviewer", "when": "2026-11-14"}


def bare(d, review=None, entries=(), sidecar=None, lay=None):
    """A pass over lay2() (or `lay`) with its bank in folder `d`, built without __init__, the bank loaded."""
    import json
    from types import SimpleNamespace
    d = Path(d)
    if sidecar is not None:
        (d / "regrow origs.json").write_text(json.dumps(sidecar), encoding="utf-8")
    with (d / "regrow bank.jsonl").open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    rp = RegrowPass.__new__(RegrowPass)
    rp.job, rp.lines = SimpleNamespace(dir=d, name="t"), []
    rp.log = rp.lines.append
    rp.review = review or {"vetoes": [], "for_you": []}
    fake_load_units(rp, [5])
    if lay is not None:
        rp.lay = lay
        rp.units = units(lay, [5], {})[0]
        rp.index_units()
    rp.rbank = Bank(d / "regrow bank.jsonl")
    rp.pools, rp.facts, rp.scores, rp.best, rp.no_checked = {}, {}, [], {}, {}
    rp.load_bank()
    return rp


def scoring(label, ai, picks, **kw):
    return dict({"kind": "score", "section": 5, "label": label, "arm_set": "mixed", "ai": ai, "humanized": False,
                 "picks": picks, "span_flagged": {}, "spans": True}, **kw)


class Review(unittest.TestCase):
    def test_a_missing_review_file_means_no_vetoes_and_no_hand_fixes(self):
        from engine import review as rv
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(rv.load(d), {"vetoes": [], "repairs": [], "for_you": [], "stuck": []})

    def test_a_file_that_does_not_parse_stops_the_run(self):
        from engine import review as rv
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "review.json").write_text('{"vetoes": [', encoding="utf-8")
            with self.assertRaises(ValueError):
                rv.load(d)

    def test_a_veto_matches_by_the_original_squashed_and_after_quote_stops(self):
        from engine import review as rv
        r = {"vetoes": [{"orig": "It  does\nits thing.", "text": 'A "thing."', "why": "wrong"}], "for_you": []}
        self.assertEqual(rv.veto_why(r, "It does its thing.", 'A  "thing."'), "wrong")             # flag: squashed
        self.assertEqual(rv.veto_why(r, "It does its thing.", 'A "thing.".'), "wrong")             # flag: the stop slip
        self.assertIsNone(rv.veto_why(r, "It does its thing.", 'A "thing" again.'))                # pass: another text
        self.assertIsNone(rv.veto_why(r, "Another span.", 'A "thing."'))                           # pass: another unit
        r["vetoes"][0]["why"] = ""
        self.assertEqual(rv.veto_why(r, "It does its thing.", 'A "thing."'), "no reason given")    # flag: still vetoed

    def test_a_hand_fix_matches_by_the_original_and_the_later_one_wins(self):
        from engine import review as rv
        r = {"vetoes": [], "for_you": [dict(FIX, text="old"), dict(FIX, label="moved")]}
        self.assertEqual(rv.hand_fix(r, " Two. ")["label"], "moved")                               # flag: the later one
        self.assertIsNone(rv.hand_fix(r, "One."))                                                  # pass: no fix
        self.assertEqual(rv.note(FIX), {"went_wrong": "lost the quote", "about": "templates", "point": "one template",
                                        "fact": "the quote is the author's"})


class VetoesAndHandFixes(unittest.TestCase):
    def test_a_vetoed_candidate_is_rejected_in_the_gate_path_and_regate_banks_it(self):
        with tempfile.TemporaryDirectory() as d:
            draws = [{"kind": "draw", "label": "s5w0 p1", "arm": "notes", "k": 0, "text": "Uno.", "why": None},
                     {"kind": "draw", "label": "s5w0 p1", "arm": "notes", "k": 1, "text": "Un.", "why": None}]
            rp = bare(d, {"vetoes": [VETO], "for_you": []}, draws)
            rp.notes, rp.known, rp.tics, rp.rates, rp.op = {"s5w0 p1": ["one"]}, set(), (lambda c: (0, [])), None, None
            rp.swear_mult, rp.filler_cap, rp.filler_words = 1.5, 1, ()
            rp.section_rest = lambda u, picks: None
            from unittest import mock
            with mock.patch("engine.regrow.gate", lambda cand, **kw: None):      # every code gate passes both
                rp.regate()
            pool = rp.pools[("s5w0 p1", "notes")]
            self.assertEqual(pool[0]["why"], "vetoed: garbles the callback")                       # flag
            self.assertIsNone(pool[1]["why"])                                                      # pass
            self.assertEqual(rp.tally("notes"), {"vetoed: garbles the callback": 1, "passed": 1})

    def test_a_scoring_with_a_vetoed_pick_or_without_the_hand_fix_is_never_best(self):
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d, {"vetoes": [VETO], "for_you": [FIX]}, [
                scoring("vetoed", 0.10, {"s5w0 p1": "Uno.", "s5w0 p2": FIX["text"]}),
                scoring("no fix", 0.20, {"s5w0 p1": "Un.", "s5w0 p2": "Dos."}),
                scoring("slip", 0.25, {"s5w0 p1": 'Un "x.".', "s5w0 p2": FIX["text"]}),
                scoring("clean", 0.30, {"s5w0 p1": "Un.", "s5w0 p2": " Dos,  by hand."})])
            self.assertEqual([rp.eligible(s) for s in rp.scores], [False, False, True, True])     # flag x2 / pass x2
            self.assertEqual(rp.best[5]["label"], "slip")

    def test_the_served_text_fixes_the_stop_slip_and_is_then_not_scored(self):
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d, entries=[scoring("slip", 0.25, {"s5w0 p1": 'Un "x.".'}),
                                  scoring("tidy", 0.30, {"s5w0 p1": "Un."})])
            rp.section_text = lambda n, picks: fake_section_text(rp, n, picks)
            sp = rp.section_pick(5)
            self.assertEqual((sp["picks"], sp["scoring"]["label"], sp["scored"]),
                             ({"s5w0 p1": 'Un "x."'}, "slip", False))                    # flag: fixed, unscored
            rp.best[5] = rp.scores[1]
            sp = rp.section_pick(5)
            self.assertEqual((sp["text"], sp["scored"]), ("Un. Two.", True))                       # pass: as scored

    def test_an_older_best_with_no_text_sha_is_scored_only_when_its_word_count_still_matches(self):
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d, entries=[scoring("old", 0.25, {"s5w0 p1": "Un."}, words=2)])
            rp.section_text = lambda n, picks: fake_section_text(rp, n, picks)
            self.assertTrue(rp.section_pick(5)["scored"])                                          # pass: "Un. Two."
            rp.best[5]["words"] = 9                                                     # the standing text moved
            self.assertFalse(rp.section_pick(5)["scored"])                                         # flag

    def test_the_report_counts_units_with_no_checked_candidate_and_unmatched_review_entries(self):
        from engine.regrow import report_md
        with tempfile.TemporaryDirectory() as d:
            gone = dict(VETO, label="gone", orig="A span the sync retired.")
            rp = bare(d, {"vetoes": [VETO, gone], "for_you": [FIX]})
            rp.arms, rp.purpose, rp.spent, rp.max_usd, rp.claude_spent, rp.max_claude_usd = ("notes",), "x", 0, 1, 0, 1
            rp.skipped = []
            rp.no_checked = {"s5w0 p1": "the fact check ran out of money"}
            md = report_md(rp, [5], {"notes": {}}, "today", True)
        self.assertIn("1 because the fact check ran out of money", md)                            # flag
        self.assertIn("the veto for gone matches no unit's original now", md)                      # flag
        self.assertNotIn("the hand fix for", md)                                                   # pass: it matches

    def test_an_assembly_carries_the_hand_fix_and_the_census_never_draws_for_it(self):
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d, {"vetoes": [], "for_you": [FIX]})
            rp.arms, rp.draws, rp.more = ("notes",), 1, 0
            drawn = []
            rp.fill = lambda u, arm, n, picks: drawn.append(u["label"])
            rp.passed = lambda u, arm=None: [{"text": "Un.", "k": 0, "arm": "notes"}] if u["label"] == "s5w0 p1" else []
            rp.no_facts = True
            rp.census([5])
            self.assertEqual(drawn, ["s5w0 p1"])                                       # flag: p2 is never drawn for
            picks, _ = rp.assembly("notes")
            self.assertEqual(picks, {"s5w0 p1": "Un.", "s5w0 p2": FIX["text"]})

    def test_a_later_round_re_picks_a_vetoed_unit_and_puts_the_hand_fix_in(self):
        for rp in Scoring().pass_([{"ai": 0.20, "segs": [("Un. Dos, by hand.", False)]}]):
            rp.review = {"vetoes": [VETO], "for_you": [FIX]}
            rp.units = units(rp.lay, [5], {})[0]
            rp.index_units()
            rp.arms, rp.draws, rp.more, rp.rounds = ("notes",), 1, 0, 2
            rp.pools = {("s5w0 p1", "notes"): {
                0: {"kind": "draw", "label": "s5w0 p1", "arm": "notes", "k": 0, "text": "Uno.", "why": "vetoed: x"},
                1: {"kind": "draw", "label": "s5w0 p1", "arm": "notes", "k": 1, "text": "Un.", "why": None, "op": 0.2}}}
            rp.facts, rp.no_checked, rp.no_facts = {}, {}, True
            rp.scores = [scoring("s5 all-notes", 0.30, {"s5w0 p1": "Uno.", "s5w0 p2": "Two-ish."}, arm_set="notes")]
            rp.run([5])
            self.assertEqual(rp.pg.calls, ["regrow s5 round 2"])
            self.assertEqual(rp.scores[-1]["picks"], {"s5w0 p1": "Un.", "s5w0 p2": FIX["text"]})
            self.assertEqual(rp.best[5]["label"], "s5 round 2")


class Repairs(unittest.TestCase):
    def test_the_matcher_takes_from_or_the_repaired_text_and_nothing_else(self):
        from engine import review as rv
        r = {"repairs": [dict(REPAIR, **{"from": 'Un "you know.".'})]}
        self.assertIs(rv.repair(r, " One. ", 'Un  "you know."'), r["repairs"][0])      # flag: from, after quote_stops
        self.assertIs(rv.repair(r, "One.", "Un."), r["repairs"][0])                    # flag: already repaired
        self.assertIsNone(rv.repair(r, "One.", "Something else."))                                 # pass: another pick
        self.assertIsNone(rv.repair(r, "Two.", 'Un "you know."'))                                  # pass: another unit

    def test_a_standing_repair_replaces_its_pick_in_an_assembly_and_is_never_drawn_for(self):
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d, {"vetoes": [], "repairs": [REPAIR], "for_you": []},
                      [scoring("s5 round 2", 0.20, {"s5w0 p1": "Un, you know."})])
            self.assertEqual(rp.pins, {"s5w0 p1": "Un."})
            rp.arms, rp.draws, rp.more, rp.no_facts = ("notes",), 1, 0, True
            drawn = []
            rp.fill = lambda u, arm, n, picks: drawn.append(u["label"])
            rp.passed = lambda u, arm=None: [{"text": "Cand.", "k": 0, "arm": "notes"}]
            rp.census([5])
            self.assertEqual(drawn, ["s5w0 p2"])                                       # flag: p1 is never drawn for
            picks, _ = rp.assembly("notes")
            self.assertEqual(picks, {"s5w0 p1": "Un.", "s5w0 p2": "Cand."})             # flag: repaired / pass: drawn
            self.assertEqual(rp.clean_picks(5, {"s5w0 p1": "Un, you know."}), {"s5w0 p1": "Un."})

    def test_a_stale_repair_is_ignored_and_logged(self):
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d, {"vetoes": [], "repairs": [REPAIR], "for_you": []},
                      [scoring("s5 round 3", 0.20, {"s5w0 p1": "A later pick."})])
            self.assertEqual((rp.repairs, rp.pins), ({}, {}))                                     # flag: not applied
            self.assertEqual(rp.stale_repairs, ["s5w0 p1"])
            self.assertTrue(any("stale" in line for line in rp.lines))
            self.assertEqual(rp.clean_picks(5, {"s5w0 p1": "A later pick."}), {"s5w0 p1": "A later pick."})
            self.assertIs(rp.best[5], rp.scores[0])                                                # pass: still best

    def test_a_scoring_holding_from_is_not_eligible_and_one_holding_the_repair_is(self):
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d, {"vetoes": [], "repairs": [REPAIR], "for_you": []},
                      [scoring("from", 0.20, {"s5w0 p1": "Un, you know."}),
                       scoring("repaired", 0.30, {"s5w0 p1": " Un. "})])
            self.assertEqual([rp.eligible(s) for s in rp.scores], [False, True])                   # flag / pass
            self.assertEqual(rp.best[5]["label"], "repaired")

    def test_final_scores_the_repaired_assembly_once_and_it_becomes_eligible(self):
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d, {"vetoes": [], "repairs": [REPAIR], "for_you": []},
                      [scoring("s5 round 2", 0.20, {"s5w0 p1": "Un, you know."})])
            self.assertNotIn(5, rp.best)
            rp.section_text = lambda n, picks: fake_section_text(rp, n, picks)
            rp.pg = FakePangram([{"ai": 0.22, "segs": [("Un. Two.", False)]}])
            rp.max_usd, rp.spent, rp.purpose = 9, 0, "x"
            sc = rp.final(5)
            self.assertEqual((sc["label"], sc["picks"]), ("s5 final", {"s5w0 p1": "Un."}))
            self.assertIs(rp.best[5], sc)
            self.assertTrue(rp.section_pick(5)["scored"])


class ReviewBase(unittest.TestCase):
    """Review entries name the scoring they were written against, and the repairs are read against the newest one."""
    OLD = {"label": "s5w0 p1", "orig": "One.", "from": "R6a.", "text": "Fixed a.", "moves": "cut", "by": "x",
           "when": "t"}
    NEW = {"label": "s5w0 p2", "orig": "Two.", "from": "R8b.", "text": "Fixed b.", "moves": "cut", "by": "x",
           "when": "t", "scoring": "s5 round 8"}
    ROUNDS = [scoring("s5 round 6", 0.33, {"s5w0 p1": "R6a.", "s5w0 p2": "R6b."}),
              scoring("s5 round 8", 0.40, {"s5w0 p1": "Fixed a.", "s5w0 p2": "R8b."})]

    def served(self, d, review, entries):
        rp = bare(d, dict({"vetoes": [], "repairs": [], "for_you": []}, **review), entries)
        rp.section_text = lambda n, picks: fake_section_text(rp, n, picks)
        return rp

    def test_a_repair_written_against_a_newer_round_that_scores_higher_stands(self):
        with tempfile.TemporaryDirectory() as d:
            rp = self.served(d, {"repairs": [self.OLD, self.NEW]}, self.ROUNDS)
            self.assertEqual(rp.bases[5]["label"], "s5 round 8")
            self.assertEqual(rp.pins, {"s5w0 p1": "Fixed a.", "s5w0 p2": "Fixed b."})             # pass: both stand
            self.assertEqual(rp.stale_repairs, [])
            sp = rp.section_pick(5)
            self.assertEqual((sp["text"], sp["scored"]), ("Fixed a. Fixed b.", False))            # the base, unscored
            from engine.regrow import review_lines
            self.assertIn("- section 5: the review base is s5 round 8 (0.40)", "\n".join(review_lines(rp)))
        with tempfile.TemporaryDirectory() as d:
            rp = self.served(d, {"repairs": [self.OLD, {k: v for k, v in self.NEW.items() if k != "scoring"}]},
                             self.ROUNDS)
            self.assertEqual(rp.stale_repairs, ["s5w0 p2"])          # flag: with no `scoring`, the older rule as before
            self.assertIsNone(rp.bases[5])

    def test_a_veto_and_a_repair_against_the_same_scoring_both_hold_and_the_veto_s_unit_stands(self):
        rounds = [scoring("s5 round 2", 0.20, {"s5w0 p1": "A1 bad.", "s5w0 p2": "B1 x."}),
                  scoring("s5 round 3", 0.30, {"s5w0 p1": "A2.", "s5w0 p2": "B2."})]
        veto = {"label": "s5w0 p1", "orig": "One.", "text": "A1 bad.", "why": "wrong", "scoring": "s5 round 2"}
        fix = {"label": "s5w0 p2", "orig": "Two.", "from": "B1 x.", "text": "B1.", "moves": "cut", "by": "x",
               "when": "t", "scoring": "s5 round 2"}
        with tempfile.TemporaryDirectory() as d:
            rp = self.served(d, {"vetoes": [veto], "repairs": [fix]}, rounds)
            self.assertEqual((rp.pins, rp.stale_repairs), ({"s5w0 p2": "B1."}, []))               # pass: it holds
            self.assertEqual(rp.section_pick(5)["text"], "One. B1.")                               # p1 standing
        with tempfile.TemporaryDirectory() as d:
            bare_ = [{k: v for k, v in e.items() if k != "scoring"} for e in (veto, fix)]
            rp = self.served(d, {"vetoes": bare_[:1], "repairs": bare_[1:]}, rounds)
            self.assertEqual(rp.stale_repairs, ["s5w0 p2"])           # flag: the veto moved the old anchor to round 3

    def test_a_named_scoring_gone_stale_after_a_re_sync_is_ignored_and_the_older_rule_applies(self):
        from engine.regrow import orig_key
        moved = scoring("s5 round 9", 0.10, {"s5w0 p1": "Other."}, origs={"s5w0 p1": orig_key("A retired span.")})
        named = dict(REPAIR, scoring="s5 round 9")
        with tempfile.TemporaryDirectory() as d:
            rp = self.served(d, {"repairs": [named]},
                             [scoring("s5 round 2", 0.20, {"s5w0 p1": "Un, you know."}), moved])
            self.assertIsNone(rp.bases[5])                                                     # flag: stale, ignored
            self.assertEqual(rp.anchor[5]["label"], "s5 round 2")
            self.assertEqual(rp.pins, {"s5w0 p1": "Un."})                                      # pass: read as before

    def test_final_on_the_base_makes_an_eligible_best_that_holds_everything(self):
        with tempfile.TemporaryDirectory() as d:
            rp = self.served(d, {"repairs": [self.OLD, self.NEW]}, self.ROUNDS)
            self.assertNotIn(5, rp.best)
            rp.pg, rp.max_usd, rp.spent, rp.purpose = FakePangram([{"ai": 0.25, "segs": []}]), 9, 0, "x"
            sc = rp.final(5)
            self.assertEqual(sc["picks"], {"s5w0 p1": "Fixed a.", "s5w0 p2": "Fixed b."})
            self.assertIs(rp.best[5], sc)                                                          # pass: eligible
            self.assertTrue(rp.section_pick(5)["scored"])
            rp2 = self.served(d, {"repairs": [self.OLD, self.NEW]}, self.ROUNDS + [sc])            # reloaded
            self.assertEqual((rp2.best[5]["label"], rp2.bases[5]["label"]), ("s5 final", "s5 round 8"))


class PassParagraphs(unittest.TestCase):
    def served(self, review, entries):
        import json
        from types import SimpleNamespace
        from unittest import mock
        from engine.regrow import pass_paragraphs
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "review.json").write_text(json.dumps(review), encoding="utf-8")
            with (Path(d) / "regrow bank.jsonl").open("w", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(e) + "\n")
            job = SimpleNamespace(dir=Path(d), sections=[5], name="t")
            with mock.patch.object(RegrowPass, "load_units", fake_load_units), \
                    mock.patch.object(RegrowPass, "section_text", fake_section_text):
                out = pass_paragraphs(job, None, [5])
            self.assertTrue((Path(d) / "regrow origs.json").exists())                             # the sidecar, written
            return out

    def test_with_no_eligible_scoring_the_vetoes_go_and_the_hand_fixes_come_in_unscored(self):
        out = self.served({"vetoes": [VETO], "for_you": [FIX]},
                          [scoring("s5 round 2", 0.20, {"s5w0 p1": "Uno.", "s5w0 p2": "Dos."},
                                   span_flagged={"s5w0 p1": True, "s5w0 p2": True}),
                           scoring("s5 round 3", 0.10, {"s5w0 p1": "Un."}, humanized=True)])
        self.assertEqual(out["sections"][5], {"ai": 0.20, "label": "s5 round 2", "text": "One. Dos, by hand.",
                                              "scored": False})                                    # flag: not scored
        self.assertEqual(set(out["paragraphs"]), {"s5w0 p2"})                                      # the veto is gone
        row = out["paragraphs"]["s5w0 p2"]
        self.assertEqual(row["for_you"], {"went_wrong": "lost the quote", "about": "templates", "point": "one template",
                                          "fact": "the quote is the author's"})
        self.assertFalse(row["flagged"])                                                           # the flag was Dos.'s
        self.assertEqual(set(row), {"text", "section", "round", "ai", "flagged", "for_you", "repaired", "picked",
                                    "stuck"})
        self.assertEqual((row["picked"], row["stuck"]), (True, []))
        self.assertIsNone(row["repaired"])                                             # a hand fix, not a repair

    def test_an_eligible_scoring_is_served_as_scored(self):
        from engine.job import sha
        out = self.served({"vetoes": [], "for_you": []},
                          [scoring("s5 round 2", 0.20, {"s5w0 p1": "Un."}, span_flagged={"s5w0 p1": True},
                                   text_sha=sha("Un. Two."))])
        self.assertEqual(out["sections"][5]["scored"], True)                                       # pass
        self.assertEqual(out["paragraphs"]["s5w0 p1"], {"text": "Un.", "section": 5, "round": "s5 round 2", "ai": 0.20,
                                                        "flagged": True, "for_you": None, "repaired": None,
                                                        "picked": True, "stuck": []})

    def test_a_repaired_paragraph_says_so_and_ships_the_repaired_text_unscored(self):
        out = self.served({"vetoes": [], "repairs": [REPAIR], "for_you": []},
                          [scoring("s5 round 2", 0.20, {"s5w0 p1": "Un, you know.", "s5w0 p2": "Dos."})])
        self.assertEqual((out["sections"][5]["text"], out["sections"][5]["scored"]), ("Un. Dos.", False))
        self.assertEqual(out["paragraphs"]["s5w0 p1"]["repaired"], REPAIR["moves"])                # flag: repaired
        self.assertEqual(out["paragraphs"]["s5w0 p1"]["for_you"], None)                            # not a for-you
        self.assertIsNone(out["paragraphs"]["s5w0 p2"]["repaired"])                                # pass: untouched

    def test_a_section_with_only_humanized_scorings_is_left_out(self):
        out = self.served({"vetoes": [], "for_you": [FIX]}, [scoring("s5 round 2", 0.10, {}, humanized=True)])
        self.assertEqual(out, {"paragraphs": {}, "sections": {}, "moot": [], "unmatched": [], "unmatched_review": []})


class Final(unittest.TestCase):
    def test_final_scores_the_served_text_once_and_it_becomes_the_best(self):
        answers = [{"ai": 0.25, "segs": [("Un. Dos, by hand.", False)]}]
        for rp in Scoring().pass_(answers):
            rp.review = {"vetoes": [VETO], "for_you": [FIX]}
            rp.units = units(rp.lay, [5], {})[0]
            rp.index_units()
            rp.scores = [scoring("s5 round 2", 0.20, {"s5w0 p1": "Uno.", "s5w0 p2": "Dos."})]
            sc = rp.final(5)
            self.assertEqual((sc["label"], sc["picks"]), ("s5 final", {"s5w0 p2": FIX["text"]}))
            self.assertIs(rp.best[5], sc)
            self.assertTrue(rp.section_pick(5)["scored"])
            self.assertIs(rp.final(5), sc)                                                    # pass: no second call
            self.assertEqual(rp.pg.calls, ["regrow s5 final"])

    def test_final_and_census_do_not_go_together(self):
        import contextlib
        import io
        from engine.regrow import main
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--final", "--census"])                                                          # flag

    def test_final_respects_the_ceiling(self):
        for rp in Scoring().pass_([{"ai": 0.25, "segs": []}], max_usd=0.25):
            rp.review = {"vetoes": [VETO], "for_you": []}
            rp.units = units(rp.lay, [5], {})[0]
            rp.index_units()
            rp.scores = [scoring("s5 round 2", 0.20, {"s5w0 p1": "Uno."})]
            self.assertIsNone(rp.final(5))                                                         # flag: $0.50 > $0.25
            self.assertEqual(rp.pg.calls, [])


class FactGate(unittest.TestCase):
    def pass_(self, claude=None, spent=0.0, gate="loose"):
        rp = RegrowPass.__new__(RegrowPass)
        rp.fact_gate = gate
        rp.lines = []
        rp.log = rp.lines.append
        rp.facts, rp.no_checked, rp.claude, rp.claude_spent, rp.max_claude_usd = {}, {}, claude, spent, 1.0
        rp.notes = {"s5w0 p1": NOTES, "s5w0 p2": NOTES}
        rp.settings = rp.conn = None
        rp.pools = {("s5w0 p1", "notes"): {0: {"label": "s5w0 p1", "arm": "notes", "k": 0, "text": GOOD, "why": None}},
                    ("s5w0 p2", "notes"): {0: {"label": "s5w0 p2", "arm": "notes", "k": 0, "text": GOOD, "why": None}}}
        return rp

    U1 = {"label": "s5w0 p1", "orig": ORIG}
    U2 = {"label": "s5w0 p2", "orig": ORIG + " Two."}

    def test_an_unchecked_candidate_is_never_picked_unless_the_run_said_no_facts(self):
        rp = self.pass_()
        self.assertIsNone(rp.choose(self.U1))                                                      # flag
        self.assertEqual(rp.no_checked, {"s5w0 p1": "there is no Claude caller"})
        rp.no_facts = True
        self.assertEqual(rp.choose(self.U1)["text"], GOOD)                                         # pass: --no-facts
        self.assertEqual(rp.no_checked, {})

    def test_out_of_money_is_logged_once_and_counted_per_unit(self):
        rp = self.pass_(claude=lambda *a, **k: ("NONE", {}), spent=1.0)
        self.assertIsNone(rp.choose(self.U1))
        self.assertIsNone(rp.choose(self.U2))
        self.assertEqual(len(rp.lines), 1)
        self.assertEqual(set(rp.no_checked.values()), {"the fact check ran out of money"})
        self.assertEqual(len(rp.no_checked), 2)

    def test_a_cached_wrong_verdict_holds_even_under_no_facts(self):
        from engine.regrow import fact_key, loose_key
        for gate, keyf in (("loose", loose_key), ("strict", fact_key)):
            rp = self.pass_(gate=gate)
            rp.no_facts = True
            rp.facts[keyf(ORIG, GOOD)] = "adds a claim"
            self.assertIsNone(rp.choose(self.U1))                                                  # flag
            rp.facts[keyf(ORIG, GOOD)] = ""
            self.assertEqual(rp.choose(self.U1)["text"], GOOD)                                     # pass

    def test_the_cache_is_versioned_and_the_pass_asks_with_its_own_prompt(self):
        from engine.job import sha
        from engine.regrow import FACTS_VERSION, REGROW_FACT_PROMPT, fact_key
        asked = []

        def claude(settings, conn, prompt, **kw):
            asked.append(prompt)
            return "- a claim the original never makes", {"cost_usd": 0.01}
        rp = self.pass_(claude=claude, gate="strict")
        rp.facts[sha(ORIG + "\x1f" + GOOD)] = ""                     # a loose verdict: strict mode doesn't reuse it
        with tempfile.TemporaryDirectory() as d:
            rp.rbank = Bank(Path(d) / "regrow bank.jsonl")
            self.assertIs(rp.facts_ok(self.U1, rp.pools[("s5w0 p1", "notes")][0]), False)          # flag
            banked = rp.rbank.entries[-1]
        self.assertEqual((banked["key"], banked["version"]), (fact_key(ORIG, GOOD), FACTS_VERSION))
        for piece in ("the salt rule from section 2", "Stir. Wait.", "winging it",
                      "But let's take a closer look", "Reply with NONE", ORIG, GOOD, NOTES[0]):
            self.assertIn(piece, asked[0])
        for ph in ("<<ORIGINAL>>", "<<NOTES>>", "<<NEW>>"):
            self.assertIn(ph, REGROW_FACT_PROMPT)
        self.assertIs(rp.facts_ok(self.U1, {"text": GOOD}), False)                                 # cached now
        self.assertEqual(len(asked), 1)

    def test_loose_mode_asks_with_fact_prompt_and_reuses_a_verdict_under_the_old_key(self):
        from engine.regrow import fact_key, loose_key
        from engine.draws import FACT_PROMPT
        asked = []

        def claude(settings, conn, prompt, **kw):
            asked.append(prompt)
            return "NONE", {"cost_usd": 0.01}
        rp = self.pass_(claude=claude)
        rp.facts[loose_key(ORIG, GOOD)] = ""                         # banked before tonight, no version
        self.assertIs(rp.facts_ok(self.U1, {"text": GOOD}), True)                                  # pass: no call
        self.assertEqual(asked, [])
        new = GOOD + " More."
        with tempfile.TemporaryDirectory() as d:
            rp.rbank = Bank(Path(d) / "regrow bank.jsonl")
            self.assertIs(rp.facts_ok(self.U1, {"text": new, "arm": "notes", "k": 3}), True)
            banked = rp.rbank.entries[-1]
        self.assertTrue(asked[0].startswith(FACT_PROMPT.split("<<ORIGINAL>>")[0]))                # FACT_PROMPT
        self.assertNotIn("the salt rule from section 2", asked[0])                             # not the strict one
        self.assertEqual(banked["key"], loose_key(ORIG, new))
        self.assertNotIn("version", banked)
        rp = self.pass_()
        rp.facts[fact_key(ORIG, GOOD)] = ""                          # a strict verdict: loose mode doesn't reuse it
        self.assertIsNone(rp.facts_ok(self.U1, {"text": GOOD}))                                   # flag

    def test_an_unknown_fact_gate_stops_the_pass(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            job = SimpleNamespace(dir=Path(d))
            with self.assertRaises(ValueError):
                RegrowPass(job=job, bank=None, settings=SimpleNamespace(config={"regrow": {"fact_gate": "medium"}}),
                           conn=None)                                                              # flag
            rp = RegrowPass(job=job, bank=None, settings=SimpleNamespace(config={}), conn=None)
            self.assertEqual(rp.fact_gate, "loose")                                                # pass: the default

    def test_a_loose_verdict_banked_under_the_unfixed_stop_slip_text_is_found(self):
        from engine.regrow import loose_key
        raw, pick = 'It "does its thing.". Then.', 'Un "x.".'
        entries = [{"kind": "draw", "label": "s5w0 p1", "arm": "notes", "k": 0, "text": raw, "why": None},
                   {"kind": "fact", "key": loose_key("One.", raw), "facts": ""},
                   scoring("s5 round 2", 0.3, {"s5w0 p1": pick}),
                   {"kind": "fact", "key": loose_key("One.", pick), "facts": ""}]
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d, entries=entries)
            rp.claude_spent, rp.max_claude_usd = 0.0, 1.0
            u = rp.units[0]
            e = rp.pools[("s5w0 p1", "notes")][0]
            self.assertEqual(e["text"], 'It "does its thing." Then.')
            self.assertIs(rp.facts_ok(u, e), True)                                    # pass: found, no Claude caller
            self.assertIs(rp.facts_ok(u, {"text": 'It "does its thing." Then.'}), True)   # through the bank's forms
            self.assertTrue(rp.keepable(u, pick))                                      # a kept pick, by its old text
            self.assertIsNone(rp.facts_ok(u, {"text": "Another text."}))              # flag: nothing banked for it


class DryFactCheck(unittest.TestCase):
    """When the fact check runs out of money (or is off without --no-facts), no assembly that lost a unit to it is
    scored, and the run scores nothing more."""

    def dry(self, d, entries, answers=(), cost=1.0, budget=1.0, lay=None, claude=True):
        rp = ready(bare(d, entries=entries, lay=lay), answers)
        rp.no_facts, rp.arms, rp.draws, rp.more, rp.rounds = False, ("notes",), 1, 0, 1
        rp.settings = rp.conn = None
        rp.claude_spent, rp.max_claude_usd = 0.0, budget
        rp.claude = (lambda *a, **k: ("NONE", {"cost_usd": cost})) if claude else None
        rp.notes = {label: ["a note"] for label in rp.orig_text}
        return rp

    @property
    def DRAWS(self):
        return [drawn("s5w0 p1", "notes", 0, "Un."), drawn("s5w0 p2", "notes", 0, "Dos.")]

    def test_a_run_whose_fact_check_runs_dry_scores_nothing_more_and_says_so(self):
        from engine.regrow import report_md
        with tempfile.TemporaryDirectory() as d:
            rp = self.dry(d, self.DRAWS)                         # the first check spends it all, p2 goes unchecked
            rp.run([5])
            self.assertEqual(rp.pg.calls, [])                                                     # flag
            self.assertEqual((rp.unscored, rp.dry_why), ({5: 1}, "the fact check ran out of money"))
            self.assertEqual(len([x for x in rp.lines if "scores nothing more" in x]), 1)
            rp.skipped, rp.purpose = [], "x"
            md = report_md(rp, [5], {"notes": {}}, "today", True)
        self.assertIn("Scoring stopped because the fact check ran out of money: section 5 left unscored with 1 unit(s)"
                      " waiting on a fact check.", md)
        with tempfile.TemporaryDirectory() as d:
            rp = self.dry(d, self.DRAWS, [{"ai": 0.3, "segs": []}], budget=5.0)
            rp.run([5])
            self.assertEqual((rp.pg.calls, rp.unscored), (["regrow s5 all-notes"], {}))            # pass: paid for
        with tempfile.TemporaryDirectory() as d:
            rp = self.dry(d, self.DRAWS, [{"ai": 0.3, "segs": []}], claude=False)
            rp.no_facts = True
            rp.run([5])
            self.assertEqual(rp.pg.calls, ["regrow s5 all-notes"])                                # pass: --no-facts

    def test_a_kept_pick_that_needs_a_check_stops_the_round(self):
        from engine.regrow import loose_key
        entries = [scoring("s5 all-notes", 0.30, {"s5w0 p1": "Uno."}, arm_set="notes"),
                   drawn("s5w0 p2", "notes", 0, "Dos."),
                   {"kind": "fact", "key": loose_key("Two.", "Dos."), "facts": ""}]
        with tempfile.TemporaryDirectory() as d:
            rp = self.dry(d, entries, [{"ai": 0.2, "segs": []}], claude=False)
            rp.rounds = 2
            rp.fill = lambda u, arm, n, picks: None                  # p1 has no draws, and none are made
            rp.run([5])
            self.assertEqual(rp.pg.calls, [])                    # flag: "Dos." alone, with "Uno." dropped, isn't scored
            self.assertEqual((rp.unscored, rp.dry_why), ({5: 1}, "there is no Claude caller"))

    def test_the_arm_test_stops_at_the_section_where_the_money_ran_out(self):
        two = {"s4w0": window([[("Four.", "C")]], section=4), "s5w0": window([[("Five.", "C")]])}
        draws = [drawn(f"s{n}w0 p1", arm, 0, f"{arm} {n}.") for n in (4, 5) for arm in ("notes", "fewshot")]
        with tempfile.TemporaryDirectory() as d:
            rp = self.dry(d, draws, [{"ai": 0.3, "segs": []}, {"ai": 0.2, "segs": []}], cost=0.5, lay=two)
            rp.units = units(two, [4, 5], {})[0]
            rp.arm_test([4, 5], min_units=1)
            self.assertEqual(rp.pg.calls, ["regrow s4 armtest notes", "regrow s4 armtest fewshot"])  # pass: paid for
            self.assertIn("ran out of money", rp.arm_tests[5]["why"])                             # flag
            self.assertEqual(rp.unscored, {5: 1})

    def test_the_base2_step_stops_when_a_stuck_unit_waits_on_a_check(self):
        with tempfile.TemporaryDirectory() as d:
            rp = SecondBase().stuck_pass(d, [SecondBase.SERVED, drawn("s5w0 p2", "base2", 0, "Dos.")])
            rp.no_facts, rp.arms, rp.draws, rp.more = False, ("base2",), 1, 0
            rp.claude_spent, rp.max_claude_usd = 0.0, 1.0
            rp.fill = lambda u, arm, n, picks: None
            rp.run([5], only_stuck=True)
            self.assertEqual((rp.pg.calls, rp.unscored), ([], {5: 1}))                            # flag


class Resync(unittest.TestCase):
    def test_the_sidecar_is_written_once_from_every_unit_and_never_rewritten(self):
        import json
        from engine.regrow import orig_key
        with tempfile.TemporaryDirectory() as d:
            bare(d)
            path = Path(d) / "regrow origs.json"
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             {"s5w0 p1": orig_key("One."), "s5w0 p2": orig_key("Two.")})
            path.write_text(json.dumps({"s5w0 p1": "kept"}), encoding="utf-8")
            rp = bare(d)
            self.assertEqual(rp.sidecar, {"s5w0 p1": "kept"})                                      # not rewritten

    def test_a_draw_or_pick_made_for_another_span_never_applies(self):
        from engine.regrow import orig_key
        old_side = {"s5w0 p1": orig_key("One."), "s5w0 p2": orig_key("An older second paragraph.")}
        def draw(k, **kw):
            e = {"kind": "draw", "label": "s5w0 p1", "arm": "notes", "k": k, "text": f"D{k}.", "why": None}
            return dict(e, **kw)
        entries = [draw(0, orig_sha=orig_key("One.")),                       # pass: its sha matches
                   draw(1, orig_sha=orig_key("Something else.")),            # flag: made for another span
                   draw(2),                                                  # pass: old, and the sidecar matches
                   dict(draw(3), label="s5w0 p2"),                           # flag: old, and the sidecar says otherwise
                   dict(draw(4, orig_sha=orig_key("Something else.")), k=0),  # flag: a newer stale entry at k 0
                   scoring("s5 round 2", 0.10, {"s5w0 p1": "D0.", "s5w0 p2": "P2."}),          # flag: p2 went stale
                   scoring("s5 round 3", 0.30, {"s5w0 p1": "D0."}, origs={"s5w0 p1": orig_key("One.")})]  # pass
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d, entries=entries, sidecar=old_side)
            self.assertEqual(sorted(rp.pools[("s5w0 p1", "notes")]), [0, 2])
            self.assertEqual(rp.pools[("s5w0 p1", "notes")][0]["text"], "D0.")                    # the valid one stands
            self.assertNotIn(("s5w0 p2", "notes"), rp.pools)
            stale = rp.scores[0]
            self.assertEqual((stale["picks"], stale["stale"]), ({"s5w0 p1": "D0."}, ["s5w0 p2"]))
            self.assertEqual(rp.best[5]["label"], "s5 round 3")                                    # never the stale one

    def test_a_banked_draw_s_stop_slip_is_fixed_in_memory_and_the_bank_file_is_left_alone(self):
        e = {"kind": "draw", "label": "s5w0 p1", "arm": "notes", "k": 0, "text": 'It "does its thing.". Then.',
             "why": None}
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d, entries=[e])
            before = (Path(d) / "regrow bank.jsonl").read_text(encoding="utf-8")
            self.assertEqual(rp.pools[("s5w0 p1", "notes")][0]["text"], 'It "does its thing." Then.')
            self.assertEqual(rp.rbank.entries[0]["text"], e["text"])
            self.assertEqual((Path(d) / "regrow bank.jsonl").read_text(encoding="utf-8"), before)

    def test_new_draws_and_scorings_carry_the_orig_sha(self):
        from engine.job import sha
        from engine.regrow import orig_key
        for rp in Scoring().pass_([{"ai": 0.20, "segs": [("Uno. Two.", False)]}]):
            rp.units = units(rp.lay, [5], {})[0]
            rp.index_units()
            sc = rp.score_section(5, {"s5w0 p1": "Uno."}, "r1", "notes")
            self.assertEqual(sc["origs"], {"s5w0 p1": orig_key("One.")})
            self.assertEqual(sc["text_sha"], sha("Uno. Two."))
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d)
            rp.srv = type("S", (), {"chat": lambda self, user, **kw: (GOOD, {})})()
            rp.notes, rp.known, rp.tics, rp.rates, rp.op = {"s5w0 p1": NOTES}, KNOWN, (lambda c: (0, [])), None, None
            rp.swear_mult, rp.filler_cap, rp.filler_words, rp.header = 1.5, 1, (), "S"
            rp.section_rest = lambda u, picks: None
            e = rp.draw(rp.units[0], "olmo", 0, {})
            self.assertEqual(e["orig_sha"], orig_key("One."))


class EverydayCapitals(unittest.TestCase):
    def test_an_idiom_s_capital_is_not_a_new_name(self):
        self.assertEqual(new_names("The kitchen was lit up like a Christmas tree.", KNOWN), [])              # pass
        self.assertEqual(new_names("It came out on a Monday in May, for God's sake.", KNOWN), [])       # pass
        self.assertEqual(new_names("It lights up like a Hallmark card.", KNOWN), ["Hallmark"])          # flag
        self.assertEqual(new_names("It lights up like a Christmasland sign.", KNOWN), ["Christmasland"])  # flag


def lay4():
    return {"s5w0": window([[("One.", "C")], [("Two.", "C")], [("Three.", "C")], [("Four.", "C")]])}


FIX4 = dict(FIX, label="s5w0 p4", orig="Four.", text="Cuatro, by hand.")


def lay_text(rp, n, picks):
    """section_text without a job: every paragraph of the layout, picks in, the rest as the original has it."""
    return " ".join(rp.para_text(wid, pi, picks, " ".join(s for run in p for s in run))
                    for wid, w in rp.lay.items() for pi, p in enumerate(w["original"]))


def drawn(label, arm, k, text, op=0.3):
    return {"kind": "draw", "label": label, "arm": arm, "k": k, "text": text, "why": None, "op": op}


def ready(rp, answers=(), max_usd=10.0):
    """A bare pass made ready to score: a fake Pangram, the ceiling, no fact check needed."""
    rp.pg, rp.max_usd, rp.spent, rp.purpose, rp.no_facts = FakePangram(list(answers)), max_usd, 0.0, "x", True
    rp.section_text = lambda n, picks: lay_text(rp, n, picks)
    return rp


class ArmTest(unittest.TestCase):
    def test_a_test_unit_needs_a_clean_candidate_on_both_arms_and_no_pin(self):
        draws = [drawn("s5w0 p1", "notes", 0, "N1."), drawn("s5w0 p1", "fewshot", 0, "F1."),
                 drawn("s5w0 p2", "notes", 0, "N2."),                                         # no few-shot candidate
                 drawn("s5w0 p3", "notes", 0, "N3."), drawn("s5w0 p3", "fewshot", 0, "F3."),
                 drawn("s5w0 p4", "notes", 0, "N4."), drawn("s5w0 p4", "fewshot", 0, "F4.")]   # pinned
        veto = {"label": "s5w0 p3", "orig": "Three.", "text": "F3.", "why": "garbles it"}
        with tempfile.TemporaryDirectory() as d:
            rp = ready(bare(d, {"vetoes": [veto], "for_you": [FIX4]}, draws, lay=lay4()))
            self.assertEqual(sorted(rp.arm_test_units(5)), ["s5w0 p1"])                # flag: p2, p3 and p4 left out
            rp.review = {"vetoes": [], "for_you": [FIX4]}
            self.assertEqual(sorted(rp.arm_test_units(5)), ["s5w0 p1", "s5w0 p3"])     # pass: the veto gone

    def test_the_two_assemblies_differ_only_in_the_test_units_and_hold_nothing_else_of_mistral_s(self):
        from engine.job import sha
        draws = [drawn(f"s5w0 p{i}", arm, 0, f"{arm[0].upper()}{i}.") for i in (1, 2, 3)
                 for arm in ("notes", "fewshot")]
        draws.append(drawn("s5w0 p4", "notes", 0, "N4."))                            # p4 has no few-shot candidate
        real = scoring("s5 round 2", 0.30, {"s5w0 p1": "N1.", "s5w0 p4": "N4.", "s5w0 p3": "F3."})
        with tempfile.TemporaryDirectory() as d:
            rp = ready(bare(d, entries=draws + [real], lay=lay4()),
                       [{"ai": 0.05, "segs": [("N1. N2. N3. Four.", False)]},
                        {"ai": 0.06, "segs": [("F1. F2. F3. Four.", True)]}])
            rp.arm_test([5], min_units=3)
            notes, fewshot = rp.tests
            self.assertEqual((notes["label"], fewshot["label"]), ("s5 armtest notes", "s5 armtest fewshot"))
            self.assertEqual(notes["picks"], {"s5w0 p1": "N1.", "s5w0 p2": "N2.", "s5w0 p3": "N3."})
            self.assertEqual(fewshot["picks"], {"s5w0 p1": "F1.", "s5w0 p2": "F2.", "s5w0 p3": "F3."})
            self.assertEqual(notes["text_sha"], sha("N1. N2. N3. Four."))             # p4 stands, not N4.
            self.assertEqual(fewshot["text_sha"], sha("F1. F2. F3. Four."))
            self.assertTrue(notes["test"] and fewshot["test"] and notes["arm_set"] == "armtest")
            self.assertEqual(rp.best[5]["label"], "s5 round 2")                        # flag: 0.05 is no best
            self.assertEqual(rp.section_pick(5)["scoring"]["label"], "s5 round 2")
            self.assertNotIn(notes, rp.scores)
            banked = [e for e in rp.rbank.entries if e.get("kind") == "score" and e.get("test")]
            self.assertEqual(len(banked), 2)
            from engine.regrow import arm_test_lines
            md = "\n".join(arm_test_lines(rp))
            self.assertIn("| notes | 3 | 3 | 0.05 | no |  | 0 of 3 |", md)
            self.assertIn("| fewshot | 3 | 3 | 0.06 | no |  | 3 of 3 |", md)
            self.assertIn("On AI share the two are within the noise", md)
            rp.arm_test([5], min_units=3)                                               # a rerun pays nothing twice
            self.assertEqual(rp.pg.calls, ["regrow s5 armtest notes", "regrow s5 armtest fewshot"])

    def test_a_banked_test_scoring_is_never_a_best_an_anchor_or_served(self):
        test = scoring("s5 armtest notes", 0.05, {"s5w0 p1": "Un, you know."}, arm_set="armtest", test=True)
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d, entries=[test])
            self.assertEqual((rp.best, rp.scores, len(rp.tests)), ({}, [], 1))
            self.assertIsNone(rp.section_pick(5))                                       # flag: nothing to serve
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d, {"vetoes": [], "repairs": [REPAIR], "for_you": []}, [test])
            self.assertEqual((rp.anchor[5], rp.repairs), (None, {}))                    # flag: no anchor from a test
        with tempfile.TemporaryDirectory() as d:
            rp = ready(bare(d, entries=[test, scoring("s5 round 2", 0.30, {"s5w0 p1": "Un."})]))
            self.assertEqual(rp.best[5]["label"], "s5 round 2")                         # pass: the real one
            self.assertEqual(rp.section_pick(5)["scoring"]["label"], "s5 round 2")

    def test_the_rounds_never_keep_a_test_scoring_s_pick(self):
        entries = [drawn("s5w0 p1", "notes", 0, "Un.", op=0.2),
                   scoring("s5 all-notes", 0.30, {"s5w0 p1": "Uno.", "s5w0 p2": "Two."}, arm_set="notes",
                           span_flagged={"s5w0 p1": True}),
                   scoring("s5 armtest notes", 0.05, {"s5w0 p1": "Test text."}, arm_set="armtest", test=True)]
        with tempfile.TemporaryDirectory() as d:
            rp = ready(bare(d, {"vetoes": [], "for_you": [FIX]}, entries),
                       [{"ai": 0.20, "segs": [("Un. Dos, by hand.", False)]}])
            rp.arms, rp.draws, rp.more, rp.rounds = ("notes",), 1, 0, 2
            rp.run([5])
            self.assertEqual(rp.pg.calls, ["regrow s5 round 2"])
            self.assertEqual(rp.scores[-1]["picks"], {"s5w0 p1": "Un.", "s5w0 p2": FIX["text"]})   # not "Test text."

    def test_the_ceiling_scores_both_or_neither(self):
        draws = [drawn(f"s5w0 p{i}", arm, 0, f"{arm[0].upper()}{i}.") for i in (1, 2) for arm in ("notes", "fewshot")]
        answers = [{"ai": 0.4, "segs": []}, {"ai": 0.3, "segs": []}]
        with tempfile.TemporaryDirectory() as d:
            rp = ready(bare(d, entries=draws), answers, max_usd=0.75)
            rp.arm_test([5], min_units=2)
            self.assertEqual(rp.pg.calls, [])                                           # flag: $1.00 > $0.75
            self.assertIn("neither was scored", rp.arm_tests[5]["why"])
        with tempfile.TemporaryDirectory() as d:
            rp = ready(bare(d, entries=draws), answers, max_usd=1.0)
            rp.arm_test([5], min_units=2)
            self.assertEqual(len(rp.pg.calls), 2)                                       # pass: both fit
        with tempfile.TemporaryDirectory() as d:
            rp = ready(bare(d, entries=draws), answers)
            rp.arm_test([5], min_units=3)
            self.assertEqual(rp.pg.calls, [])                                           # flag: two units, three needed
            self.assertIn("under the minimum of 3", rp.arm_tests[5]["why"])

    def test_the_verdict_sentence_names_the_lower_arm_or_the_noise(self):
        from types import SimpleNamespace
        from engine.regrow import arm_test_lines

        def sc(ai, hum):
            return {"ai": ai, "humanized": hum > 0.5, "max_humanizer": hum, "picks": {"a": "x y"}, "span_flagged": {}}
        rp = SimpleNamespace(noise=0.05, arm_tests={6: {"units": ["a"], "notes": sc(0.45, 0.90),
                                                        "fewshot": sc(0.30, 0.88), "why": None}})
        md = "\n".join(arm_test_lines(rp))
        self.assertIn("On AI share the few-shot arm scored lower (0.30 against 0.45); on the humanizer score the two"
                      " are within the noise (0.90 and 0.88)", md)
        rp.arm_tests[6]["notes"] = sc(0.31, 0.40)
        self.assertIn("on the humanizer score the adapter arm scored lower (0.40 against 0.88)",
                      "\n".join(arm_test_lines(rp)))


class FakeBase(object):
    """A base model's server: the completion call only, no LoRA, a per-slot context and a word-count tokenizer."""
    has_lora = False

    def __init__(self, ctx=None):
        self.ctx, self.calls = ctx, []

    def complete(self, prompt, **kw):
        self.calls.append((prompt, kw))
        return GOOD + "\nNext line.", {}

    def tokenize(self, text):
        return text.split()


def base_pass(srv, parts):
    rp = RegrowPass.__new__(RegrowPass)
    rp.srv, rp.op, rp.pools, rp.known, rp.tics, rp.rates = srv, None, {}, KNOWN, (lambda c: (0, [])), None
    rp.swear_mult, rp.filler_cap, rp.filler_words, rp.header = 1.5, 1, (), "SAMPLE"
    rp.example_parts, rp.examples_block = tuple(parts), "\n\n---\n\n".join(parts)
    rp.notes = {"s5w0 p1": NOTES}
    rp.section_rest = lambda u, picks: None
    rp.lines = []
    rp.log = rp.lines.append
    return rp


UNIT = {"label": "s5w0 p1", "wid": "s5w0", "pi": 0, "orig": ORIG, "standing": ORIG, "prefix": "", "suffix": ""}


class SecondBase(unittest.TestCase):
    def test_a_base2_draw_takes_the_completion_call_with_the_few_shot_prompt_and_its_own_seeds(self):
        rp = base_pass(FakeBase(), ["EX1", "EX2"])
        with tempfile.TemporaryDirectory() as d:
            rp.rbank = Bank(Path(d) / "regrow bank.jsonl")
            e = rp.draw(UNIT, "base2", 0, {})
        self.assertIsNone(e["why"])                                                               # pass
        self.assertEqual((e["text"], e["arm"]), (GOOD, "base2"))                                  # cut at the newline
        prompt, kw = rp.srv.calls[0]
        self.assertEqual(prompt, fewshot_prompt("SAMPLE", "EX1\n\n---\n\nEX2", NOTES, ""))
        self.assertEqual((kw["stop"], kw["temperature"], kw["min_p"]), (["\n"], 1.0, 0.05))
        self.assertNotIn("lora_scale", kw)                                                        # no adapter, no scale
        self.assertEqual(kw["seed"], seed_for(UNIT, "base2", 0))
        self.assertEqual(len({seed_for(UNIT, a, k) for a in ("notes", "fewshot", "olmo", "base2") for k in range(40)}),
                         160)                                                                     # no seed shared

    def test_a_prompt_too_long_for_the_context_drops_worked_examples_from_the_front_and_says_so_once(self):
        long_ex = "Notes:\n- a\n\nParagraph:\n" + "word " * 300
        cut = fewshot_prompt("SAMPLE", "EX2", NOTES, "")
        n_tok = int(len(ORIG.split()) * 2.4) + 60
        rp = base_pass(FakeBase(ctx=len(cut.split()) + n_tok), [long_ex, "EX2"])
        with tempfile.TemporaryDirectory() as d:
            rp.rbank = Bank(Path(d) / "regrow bank.jsonl")
            rp.draw(UNIT, "base2", 0, {})
            rp.draw(UNIT, "base2", 1, {})
        self.assertEqual([p for p, _ in rp.srv.calls], [cut, cut])                               # flag: trimmed
        self.assertEqual(len([x for x in rp.lines if "dropped" in x]), 1)
        rp = base_pass(FakeBase(ctx=100000), [long_ex, "EX2"])
        self.assertIn(long_ex, rp.prompt_for(UNIT, "base2", n_tok))                               # pass: it fits
        self.assertEqual(rp.lines, [])

    def stuck_pass(self, d, entries, review=None):
        return ready(bare(d, review or {"vetoes": [], "for_you": [FIX4]}, entries, lay=lay4()),
                     [{"ai": 0.10, "segs": [("Uno. Dos. Tres. Cuatro, by hand.", False)]}])

    SERVED = scoring("s5 round 2", 0.30, {"s5w0 p1": "Uno.", "s5w0 p3": "Tres.", "s5w0 p4": FIX4["text"]})

    def test_only_stuck_takes_the_units_with_no_served_pick_and_leaves_the_pinned_ones_out(self):
        veto = {"label": "s5w0 p3", "orig": "Three.", "text": "Tres.", "why": "wrong"}
        with tempfile.TemporaryDirectory() as d:
            rp = self.stuck_pass(d, [self.SERVED])
            self.assertEqual(rp.stuck_units(5), ["s5w0 p2"])                     # flag p2 / pass p1, p3 / p4 pinned
            rp = self.stuck_pass(d, [self.SERVED], {"vetoes": [veto], "for_you": [FIX4]})
            self.assertEqual(rp.stuck_units(5), ["s5w0 p2", "s5w0 p3"])          # flag: a vetoed pick isn't served
            rp.arms, rp.draws, rp.more = ("notes", "fewshot"), 1, 0
            got = []
            rp.fill = lambda u, arm, n, picks: got.append((u["label"], arm))
            rp.run([5], census_only=True, only_stuck=True)
            self.assertEqual(set(got), {("s5w0 p2", "notes"), ("s5w0 p2", "fewshot"),
                                        ("s5w0 p3", "notes"), ("s5w0 p3", "fewshot")})      # p1 and p4 never drawn for

    def test_the_base2_assembly_is_the_served_picks_plus_the_base2_spans_and_can_be_the_best(self):
        from engine.regrow import base2_lines
        with tempfile.TemporaryDirectory() as d:
            rp = self.stuck_pass(d, [self.SERVED, drawn("s5w0 p2", "base2", 0, "Dos.", op=0.25),
                                     drawn("s5w0 p2", "notes", 0, "Deux.", op=0.6)])
            rp.arms, rp.draws, rp.more = ("base2",), 1, 0
            rp.fill = lambda u, arm, n, picks: None
            rp.run([5], only_stuck=True)
            self.assertEqual(rp.pg.calls, ["regrow s5 base2"])                               # one scoring, no rounds
            sc = rp.scores[-1]
            self.assertEqual((sc["label"], sc["arm_set"]), ("s5 base2", "base2"))
            self.assertEqual(sc["picks"], {"s5w0 p1": "Uno.", "s5w0 p2": "Dos.", "s5w0 p3": "Tres.",
                                           "s5w0 p4": FIX4["text"]})
            self.assertIs(rp.best[5], sc)                                                        # pass: eligible
            self.assertIs(rp.section_pick(5)["scoring"], sc)
            line = base2_lines(rp)[0]
            for piece in ("1 unit(s) stuck, 1 got a base2 candidate", "base2 0.25 (1 draws)",
                          "notes 0.60 (1 draws; 0.60 on the stuck units, 1 draws)", "scored 0.10 against its best"
                          " before of 0.30 (s5 round 2)", "not humanized", "0 of 1 base2 span(s) flagged"):
                self.assertIn(piece, line)

    def test_the_report_carries_the_arm_test_and_the_second_base_model_parts(self):
        from engine.regrow import report_md
        with tempfile.TemporaryDirectory() as d:
            rp = self.stuck_pass(d, [self.SERVED])
            rp.arms, rp.skipped, rp.claude_spent, rp.max_claude_usd = ("base2",), [], 0.0, 1.0
            self.assertNotIn("## Arm test", report_md(rp, [5], {"base2": {}}, "today", True))     # pass: nothing run
            rp.base2_step(5)
            rp.arm_tests = {5: {"units": [], "notes": None, "fewshot": None, "why": "0 test unit(s)"}}
            md = report_md(rp, [5], {"base2": {}}, "today", True)
        self.assertIn("## Arm test", md)                                                         # flag
        self.assertIn("- section 5: not scored, 0 test unit(s).", md)
        self.assertIn("## Second base model", md)
        self.assertIn("- section 5: 1 unit(s) stuck, 0 got a base2 candidate", md)

    def test_a_section_with_no_base2_candidate_is_not_scored(self):
        with tempfile.TemporaryDirectory() as d:
            rp = self.stuck_pass(d, [self.SERVED])
            rp.arms = ("base2",)
            self.assertIsNone(rp.base2_step(5))                                                  # flag
            self.assertEqual(rp.pg.calls, [])

    def test_base2_or_olmo_mixed_with_another_arm_is_refused(self):
        import contextlib
        import io
        from unittest import mock
        from engine.regrow import main
        for arms in ("base2,notes", "fewshot,base2", "olmo,base2", "olmo,notes"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--arms", arms])                                                           # flag
        for argv in (["--arm-test", "--only-stuck"], ["--final", "--only-stuck"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(argv)                                                                       # flag
        with mock.patch("engine.regrow.Settings", side_effect=RuntimeError("past the checks")):
            for argv in (["--arms", "base2"], [], ["--arm-test"]):
                with self.assertRaises(RuntimeError):
                    main(argv)                                                                   # pass: on to the run


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------- stuck units in bold, and --test-scoring
def lay5():
    return {"s5w0": window([[("One.", "C")], [("Two.", "C")],
                            [("Gamma one.", "C"), ("Mine.", "A"), ("Delta one.", "C")],
                            [("Four.", "C")],
                            [("Alpha one.", "C"), ("Alpha two.", "C"), ("Mine here.", "A"), ("Beta one.", "C"),
                             ("Beta two.", "C")]])}


STAND5 = {"s5w0 p4": ["Four by the loop."],
          "s5w0 p5": ["Alpha one.", "Alpha too, by the loop.", "Mine here.", "Beta one.", "Beta two."]}


def fake_load_units5(self, sections, vm=None):
    """load_units without a job: lay5(), with the loop's text standing in p4 (all of it) and p5 (one sentence)."""
    self.lay = lay5()
    self.standing = {k: " ".join(v) for k, v in STAND5.items()}
    self.units, self.skipped = units(self.lay, sections, STAND5, self.standing)
    self.swaps = []
    self.index_units()


def st(orig, label, **kw):
    return dict({"label": label, "orig": orig, "why": "nothing got through", "about": f"what {label} says",
                 "point": "the point", "fact": "the fact", "when": "2026-11-15"}, **kw)


STUCK5 = [st("One.", "s5w0 p1"), st("Two.", "s5w0 p2"), st("Gamma one.", "s5w0 p3 s1"), st("Delta one.", "s5w0 p3"),
          st("Four.", "s5w0 p4"), st("Alpha one. Alpha two.", "s5w0 p5 s1")]
FIX5 = dict(FIX, label="s5w0 p1", orig="One.", text="Uno, by hand.")
REPAIR5 = dict(REPAIR, label="s5w0 p5", orig="Beta one. Beta two.", **{"from": "Beta, you know."}, text="Beta.",
               moves="cut: ', you know'")
ROUND5 = scoring("s5 round 2", 0.30, {"s5w0 p1": "Uno.", "s5w0 p5": "Beta, you know."})


def bare5(d, review, entries=(ROUND5,)):
    """bare() over lay5(), with the loop's text standing as fake_load_units5 has it (the labels don't move)."""
    rp = bare(d, dict({"vetoes": [], "repairs": [], "for_you": [], "stuck": []}, **review), entries, lay=lay5())
    rp.standing = {k: " ".join(v) for k, v in STAND5.items()}
    rp.units = units(rp.lay, [5], STAND5, rp.standing)[0]
    rp.section_text = lambda n, picks: lay_text(rp, n, picks)
    return rp


class StuckUnits(unittest.TestCase):
    """A stuck entry in the reviewer's file: the unit's sentences still Claude's go in bold, and nothing else moves."""

    def test_the_matcher_takes_the_later_entry_and_a_review_without_the_list_has_none(self):
        from engine import review as rv
        r = {"stuck": [st("Two.", "old"), st(" Two. ", "new")]}
        self.assertEqual(rv.stuck(r, "Two.")["label"], "new")                                      # flag: the later
        self.assertIsNone(rv.stuck(r, "One."))                                                     # pass
        self.assertIsNone(rv.stuck({"vetoes": []}, "Two."))
        self.assertEqual(rv.stuck_note(r["stuck"][0]), {"why": "nothing got through", "about": "what old says",
                                                        "point": "the point", "fact": "the fact"})

    def test_pass_paragraphs_puts_the_original_sentences_of_each_stuck_unit_in_the_row(self):
        import json
        from types import SimpleNamespace
        from unittest import mock
        from engine.regrow import pass_paragraphs
        review = {"vetoes": [], "repairs": [REPAIR5], "for_you": [FIX5], "stuck": STUCK5}
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "review.json").write_text(json.dumps(review), encoding="utf-8")
            (Path(d) / "regrow bank.jsonl").write_text(json.dumps(ROUND5) + "\n", encoding="utf-8")
            job = SimpleNamespace(dir=Path(d), sections=[5], name="t")
            with mock.patch.object(RegrowPass, "load_units", fake_load_units5), \
                    mock.patch.object(RegrowPass, "section_text", lambda self, n, picks: lay_text(self, n, picks)):
                out = pass_paragraphs(job, None, [5])
        rows = out["paragraphs"]
        self.assertEqual(sorted(rows), ["s5w0 p1", "s5w0 p2", "s5w0 p3", "s5w0 p5"])
        p1 = rows["s5w0 p1"]                                              # a hand fix wins over its stuck entry
        self.assertEqual((p1["picked"], p1["stuck"], p1["for_you"]["about"]), (True, [], "templates"))
        p2 = rows["s5w0 p2"]                                              # all original, no pick: a row all the same
        self.assertEqual({k: p2[k] for k in ("text", "picked", "round", "ai", "for_you", "repaired")},
                         {"text": "Two.", "picked": False, "round": None, "ai": None, "for_you": None,
                          "repaired": None})
        self.assertEqual(p2["stuck"], [{"bold": ["Two."], "why": "nothing got through", "about": "what s5w0 p2 says",
                                        "point": "the point", "fact": "the fact"}])
        p3 = rows["s5w0 p3"]                                              # two entries, in unit order
        self.assertEqual([g["bold"] for g in p3["stuck"]], [["Gamma one."], ["Delta one."]])
        self.assertEqual([g["about"] for g in p3["stuck"]], ["what s5w0 p3 s1 says", "what s5w0 p3 says"])
        self.assertFalse(p3["picked"])
        p5 = rows["s5w0 p5"]                                              # a sub-span beside a pick and a repair
        self.assertEqual((p5["text"], p5["picked"], p5["repaired"]),
                         ("Alpha one. Alpha too, by the loop. Mine here. Beta.", True, "cut: ', you know'"))
        self.assertEqual([g["bold"] for g in p5["stuck"]], [["Alpha one."]])          # flag: the loop's line isn't
        self.assertEqual(out["moot"], ["s5w0 p4"])                                     # the loop rewrote all of it
        self.assertNotIn("s5w0 p4", rows)

    def test_a_stuck_entry_never_changes_a_best_a_pick_a_pin_or_what_is_drawn(self):
        entries = [ROUND5, scoring("s5 round 3", 0.20, {"s5w0 p1": "Uno, by hand.", "s5w0 p5": "Beta.",
                                                      "s5w0 p2": "Dos."}),
                   scoring("s5 round 4", 0.10, {"s5w0 p2": "Dos."}, humanized=True)]
        stray = [dict(x, scoring="s5 round 2") for x in STUCK5]      # a stray `scoring` names no review base either
        got = {}
        for name, stuck in (("none", []), ("stuck", STUCK5), ("stray", stray)):
            with tempfile.TemporaryDirectory() as d:
                rp = bare5(d, {"repairs": [REPAIR5], "for_you": [FIX5], "stuck": stuck}, entries)
                sp = rp.section_pick(5)
                got[name] = ((rp.best[5]["label"], rp.pins, rp.bases[5], rp.anchor[5]["label"], rp.stale_repairs,
                              [rp.eligible(s) for s in rp.scores], sp["picks"], sp["text"], sp["scored"],
                              rp.stuck_units(5)), rp)
        self.assertEqual(got["stuck"][0], got["none"][0])                                          # pass
        self.assertEqual(got["stray"][0], got["none"][0])
        self.assertIn("s5w0 p4", got["stuck"][0][-1])                         # a stuck unit stays a stuck unit
        rp = got["stuck"][1]
        self.assertEqual(len(rp.stuck_entries), 6)
        rp.arms, rp.draws, rp.more = ("notes",), 1, 0
        drawn = []
        rp.fill = lambda u, arm, n, picks: drawn.append(u["label"])
        rp.passed = lambda u, arm=None: []
        rp.census([5])
        self.assertIn("s5w0 p2", drawn)                                       # the pass keeps trying it
        self.assertNotIn("s5w0 p1", drawn)                                    # the hand fix is pinned, as before

    def test_the_report_says_what_each_stuck_entry_comes_to(self):
        from engine.regrow import review_lines
        gone = st("A span the sync retired.", "gone")
        with tempfile.TemporaryDirectory() as d:
            rp = bare5(d, {"repairs": [REPAIR5], "for_you": [FIX5], "stuck": STUCK5 + [gone]})
            text = "\n".join(review_lines(rp))
        self.assertIn("0 veto(es), 1 repair(s), 1 hand fix(es) and 7 stuck unit(s) in review.json", text)
        self.assertIn("- stuck, s5w0 p2: 1 sentence(s) of the original in bold for you to rewrite", text)
        self.assertIn("- stuck, s5w0 p4: moot, the loop already rewrote every sentence of its original", text)
        self.assertIn("- stuck, s5w0 p1: it has a hand fix, which goes in bold whole instead", text)
        self.assertIn("- the stuck entry for gone matches no unit now: write a new one or remove it", text)                # flag
        with tempfile.TemporaryDirectory() as d:
            rp = bare5(d, {"stuck": [st("Beta one. Beta two.", "s5w0 p5")]})
            self.assertIn("- stuck, s5w0 p5: the pass serves a pick for it now, so nothing goes in bold",
                          review_lines(rp))                                                          # pass: picked
        with tempfile.TemporaryDirectory() as d:
            rp = bare5(d, {})
            self.assertEqual(review_lines(rp),
                             ["No review file entries (no vetoes, no repairs, no hand fixes, no stuck units)."])


class TestScoring(unittest.TestCase):
    """--test-scoring: a banked scoring with the reviewer's repairs in that fit its picks, scored once as a test."""
    OLD = {"label": "s5w0 p1", "orig": "One.", "from": "R6a.", "text": "Fixed a.", "moves": "cut a", "by": "x",
           "when": "t"}
    OLD_B = {"label": "s5w0 p2", "orig": "Two.", "from": "R6b.", "text": "Fixed 6b.", "moves": "cut 6b", "by": "x",
             "when": "t", "scoring": "s5 round 6"}
    NEW = {"label": "s5w0 p2", "orig": "Two.", "from": "R8b.", "text": "Fixed b.", "moves": "cut 8b", "by": "x",
           "when": "t", "scoring": "s5 round 8"}
    ROUNDS = [scoring("s5 round 6", 0.33, {"s5w0 p1": "R6a.", "s5w0 p2": "R6b."}),
              scoring("s5 round 8", 0.40, {"s5w0 p1": "Fixed a.", "s5w0 p2": "R8b."})]

    def served(self, d, review, entries=None, answers=(), max_usd=10.0):
        rp = bare(d, dict({"vetoes": [], "repairs": [], "for_you": []}, **review),
                  self.ROUNDS if entries is None else entries)
        return ready(rp, answers, max_usd)

    def test_a_repair_tagged_for_an_older_scoring_applies_and_the_test_is_never_a_best(self):
        with tempfile.TemporaryDirectory() as d:
            rp = self.served(d, {"repairs": [self.OLD, self.OLD_B, self.NEW]},
                             answers=[{"ai": 0.05, "segs": [("Fixed a. Fixed 6b.", False)]}])
            self.assertEqual(rp.bases[5]["label"], "s5 round 8")                  # the review base is the newer one
            self.assertEqual(rp.stale_repairs, ["s5w0 p2"])                       # OLD_B is stale against it
            before = (rp.best.get(5), rp.section_pick(5)["picks"])
            sc = rp.test_scoring("s5 round 6")
            self.assertEqual((sc["label"], sc["test"], sc["arm_set"]), ("s5 round 6 repaired test", True, "repaired test"))
            self.assertEqual(sc["picks"], {"s5w0 p1": "Fixed a.", "s5w0 p2": "Fixed 6b."})            # flag: applied
            self.assertEqual(rp.test_runs["s5 round 6"]["log"],
                             {"s5w0 p1": {"applied": ["cut a"], "unfit": [], "vetoed": False, "hand_fix": False},
                              "s5w0 p2": {"applied": ["cut 6b"], "unfit": ["cut 8b"], "vetoed": False,
                                          "hand_fix": False}})
            self.assertTrue(any("s5w0 p2: applied cut 6b, no fitting `from` for cut 8b" in x for x in rp.lines))
            self.assertNotIn(sc, rp.scores)
            self.assertIn(sc, rp.tests)
            self.assertEqual((rp.best.get(5), rp.section_pick(5)["picks"]), before)    # pass: never a best or served
            self.assertIs(rp.test_scoring("s5 round 6"), sc)                          # a relaunch pays nothing twice
            self.assertEqual(rp.pg.calls, ["regrow s5 round 6 repaired test"])
            banked = [e for e in rp.rbank.entries if e.get("kind") == "score"]
        with tempfile.TemporaryDirectory() as d:
            rp2 = self.served(d, {"repairs": [self.OLD, self.OLD_B, self.NEW]}, entries=self.ROUNDS + banked[-1:])
            self.assertNotIn(5, rp2.best)                                              # reloaded: still no best
            self.assertEqual([t["label"] for t in rp2.tests], ["s5 round 6 repaired test"])
            self.assertEqual(rp2.section_pick(5)["scoring"]["label"], "s5 round 8")

    def test_vetoes_go_the_stop_slip_is_fixed_and_the_hand_fixes_come_in(self):
        veto = {"label": "s5w0 p1", "orig": "One.", "text": "R6a.", "why": "wrong"}
        entries = [scoring("s5 round 6", 0.33, {"s5w0 p1": "R6a.", "s5w0 p2": 'R6 "b.".'})]
        with tempfile.TemporaryDirectory() as d:
            rp = self.served(d, {"vetoes": [veto]}, entries=entries, answers=[{"ai": 0.2, "segs": []}])
            sc = rp.test_scoring("s5 round 6")
            self.assertEqual(sc["picks"], {"s5w0 p2": 'R6 "b."'})                       # flag: veto out, slip fixed
            self.assertTrue(rp.test_runs["s5 round 6"]["log"]["s5w0 p1"]["vetoed"])
        with tempfile.TemporaryDirectory() as d:
            rp = self.served(d, {"for_you": [FIX]}, entries=entries, answers=[{"ai": 0.2, "segs": []}])
            self.assertEqual(rp.test_scoring("s5 round 6")["picks"], {"s5w0 p1": "R6a.", "s5w0 p2": FIX["text"]})

    def test_an_unknown_label_is_refused_and_the_ceiling_holds(self):
        with tempfile.TemporaryDirectory() as d:
            rp = self.served(d, {})
            with self.assertRaises(ValueError) as e:
                rp.test_scoring("s5 round 99")
            self.assertIn("no banked scoring is called 's5 round 99'", str(e.exception))                 # flag
            self.assertIn("s5 round 6", str(e.exception))
            rp.max_usd = 0.25
            self.assertIsNone(rp.test_scoring("s5 round 6"))                                         # $0.50 > $0.25
            self.assertEqual((rp.pg.calls, rp.tests), ([], []))

    def test_the_report_carries_the_test_scoring_beside_the_arm_test(self):
        from engine.regrow import report_md
        with tempfile.TemporaryDirectory() as d:
            rp = self.served(d, {"repairs": [self.OLD, self.OLD_B, self.NEW]},
                             answers=[{"ai": 0.05, "segs": []}])
            rp.test_scoring("s5 round 6")
            rp.arms, rp.purpose, rp.claude_spent, rp.max_claude_usd, rp.skipped = ("notes",), "x", 0, 1, []
            md = report_md(rp, [5], {"notes": {}}, "today", False, test_scoring=True)
        self.assertIn("# The paragraph pass, today (test scoring)", md)
        self.assertIn("## Test scorings", md)
        self.assertIn("| s5 round 6 repaired test | 5 | s5 round 6 (0.33) | 2 | 1 | 0.05 | no |  |", md)
        self.assertNotIn("## Scores", md)                                     # no voice audit, no side by side

    def test_the_command_line_takes_it_on_its_own_and_from_its_label(self):
        import contextlib
        import io
        from engine.regrow import main, test_scoring_section
        self.assertEqual((test_scoring_section("s4 round 3"), test_scoring_section("s12 final")), (4, 12))
        self.assertIsNone(test_scoring_section("round 3"))
        for argv in (["--test-scoring", "s4 round 3", "--final"], ["--test-scoring", "s4 round 3", "--arm-test"],
                     ["--test-scoring", "s4 round 3", "--census"], ["--test-scoring", "s4 round 3", "--only-stuck"],
                     ["--test-scoring", "round 3"], ["--test-scoring", "s4 round 3", "--section", "5"]):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                main(argv)                                                                           # flag
            self.assertIn("--test-scoring", err.getvalue())


class StuckParts(unittest.TestCase):
    """A stuck entry also speaks for what is left of its unit after you rewrote a sentence of it by hand."""

    def test_a_part_is_a_run_of_whole_sentences_and_an_exact_match_wins(self):
        from engine import review as rv
        self.assertTrue(rv.is_part("Alpha one.", "Alpha one. Alpha two."))                          # flag
        self.assertTrue(rv.is_part("Beta.  Gamma.", "Alpha. Beta. Gamma. Delta."))
        self.assertFalse(rv.is_part("Alpha. Gamma.", "Alpha. Beta. Gamma."))                        # pass: not a run
        self.assertFalse(rv.is_part("Alpha", "Alpha one. Alpha two."))                     # pass: part of a sentence
        self.assertFalse(rv.is_part("Alpha one. Alpha two.", "Alpha one. Alpha two."))     # the whole is exact
        old, new, exact = st("One. Two.", "old"), st("One. Two. Three.", "new"), st("One.", "exact")
        self.assertEqual(rv.stuck_match({"stuck": [old, new]}, "One."), (new, True))       # the later among parts
        self.assertEqual(rv.stuck_match({"stuck": [exact, new]}, "One."), (exact, False))  # exact over a later part
        self.assertEqual(rv.stuck_match({"stuck": [new]}, "Four."), (None, False))

    def test_the_part_gets_its_own_bold_under_the_old_prompt_and_the_report_says_so(self):
        from engine.regrow import review_lines
        whole = st("Alpha one. Alpha two. Alpha three.", "s5w0 p5 s1", about="the old prompt")
        gone = st("Nothing here any more.", "gone")
        with tempfile.TemporaryDirectory() as d:
            rp = bare5(d, {"stuck": [whole, gone]})
            self.assertIs(rp.stuck_entries["s5w0 p5 s1"], whole)
            self.assertEqual((rp.stuck_parts, rp.stuck_unmatched), (("s5w0 p5 s1",), ("gone",)))
            st5 = rp.stuck_state(5, rp.served(5)[1])
            self.assertEqual(st5["bold"], {"s5w0 p5 s1": ["Alpha one."]})     # from the unit: the loop's line isn't
            text = "\n".join(review_lines(rp))
        self.assertIn("- stuck, s5w0 p5 s1: the entry for s5w0 p5 s1 speaks for it by overlap", text)
        self.assertIn("- the stuck entry for gone matches no unit now: write a new one or remove it", text)


class FollowTheSpan(unittest.TestCase):
    """A pick or a draw follows its span by its sha when a re-cut at a sync moves the span to another label."""

    def moved(self, d, entries, lay=None):
        """The pass over a layout where "Two." moved from p2 to p3 (a paragraph came in before it)."""
        lay = lay or {"s5w0": window([[("One.", "C")], [("New.", "C")], [("Two.", "C")]])}
        return bare(d, {"vetoes": [], "repairs": [], "for_you": []}, entries, lay=lay)

    def test_a_pick_follows_its_span_to_its_new_label_and_never_onto_another_span(self):
        from engine.regrow import orig_key
        sc = scoring("s5 round 2", 0.2, {"s5w0 p1": "Un.", "s5w0 p2": "Dos."}, span_flagged={"s5w0 p2": True},
                     origs={"s5w0 p1": orig_key("One."), "s5w0 p2": orig_key("Two.")})
        with tempfile.TemporaryDirectory() as d:
            rp = self.moved(d, [sc])
            got = rp.scores[0]
            self.assertEqual(got["picks"], {"s5w0 p1": "Un.", "s5w0 p3": "Dos."})                    # flag: moved
            self.assertEqual((got["moved"], got["span_flagged"]), ({"s5w0 p2": "s5w0 p3"}, {"s5w0 p3": True}))
            self.assertNotIn("stale", got)
            self.assertIs(rp.best[5], got)                                       # still eligible, still the best
            self.assertNotIn("s5w0 p2", got["picks"])                            # "New." never gets Two.'s pick
        gone = dict(sc, origs={"s5w0 p1": orig_key("One."), "s5w0 p2": orig_key("A span nobody holds now.")})
        with tempfile.TemporaryDirectory() as d:
            got = self.moved(d, [gone]).scores[0]
            self.assertEqual((got["picks"], got["stale"]), ({"s5w0 p1": "Un."}, ["s5w0 p2"]))       # pass: dropped

    def test_a_sha_two_units_share_or_another_section_s_span_follows_nowhere(self):
        from engine.regrow import orig_key
        sc = scoring("s5 round 2", 0.2, {"s5w0 p2": "Dos."}, origs={"s5w0 p2": orig_key("Two.")})
        twice = {"s5w0": window([[("One.", "C")], [("New.", "C")], [("Two.", "C")], [("Two.", "C")]])}
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.moved(d, [sc], twice).scores[0]["stale"], ["s5w0 p2"])           # flag: ambiguous
        across = {"s5w0": window([[("One.", "C")], [("New.", "C")]]), "s6w0": window([[("Two.", "C")]], section=6)}
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.moved(d, [sc], across).scores[0]["stale"], ["s5w0 p2"])    # flag: not into s6

    def test_a_draw_follows_its_span_too_and_the_bank_file_stays_as_it_is(self):
        from engine.regrow import orig_key
        draw = {"kind": "draw", "label": "s5w0 p2", "arm": "notes", "k": 0, "text": "Dos.", "why": None,
                "orig_sha": orig_key("Two.")}
        with tempfile.TemporaryDirectory() as d:
            rp = self.moved(d, [draw])
            self.assertEqual(list(rp.pools), [("s5w0 p3", "notes")])
            self.assertEqual(rp.pools[("s5w0 p3", "notes")][0]["was_label"], "s5w0 p2")
            self.assertEqual(json.loads((Path(d) / "regrow bank.jsonl").read_text(encoding="utf-8"))["label"],
                             "s5w0 p2")

    def test_a_stuck_entry_speaks_for_a_span_that_grew_past_it_or_shares_sentences_with_it(self):
        from engine import review as rv
        who = "Then. Then it boils. Slow."
        self.assertTrue(rv.holds(who + " Yep, that one.", who))                                           # flag
        self.assertFalse(rv.holds("Then. Slow.", who))                                                       # pass
        self.assertEqual(rv.stuck_match({"stuck": [st(who, "grown")]}, who + " Yep, that one.")[1], True)
        # the loop rewrote the entry's last sentence before the author's edit froze the guide's text in its place
        long1, long2 = "It does the same to the soup as well.", "Butter fixes most things, and there's more."
        entry = st(f"{long1} {long2}", "overlap")
        self.assertEqual(rv.stuck_match({"stuck": [entry]}, f"{long1} Butter fixes most things."), (entry, True))
        self.assertEqual(rv.shared(f"{long1} Butter fixes most things.", entry["orig"]), [long1])
        # too little in common to be the same span: one short sentence ("Slow.") is no overlap, unless the unit is
        # all that's left of the entry's span
        self.assertEqual(rv.stuck_match({"stuck": [st("Slow. Another sentence that is long.", "x")]},
                                        "Slow. Something else here."), (None, False))                     # pass
        entry = st("Beta one. Beta two.", "shrunk")
        self.assertEqual(rv.stuck_match({"stuck": [entry]}, "Beta one."), (entry, True))
        with tempfile.TemporaryDirectory() as d:
            rp = bare5(d, {"stuck": [st("Beta one.", "grown")]}, entries=())
            self.assertEqual(rp.stuck_entries, {})                    # "Beta one." alone is too short to overlap


class Absorbed(unittest.TestCase):
    """A pick whose text a sync froze as its unit's original (you rewrote another sentence of its paragraph)."""

    def test_the_pick_is_absorbed_not_stale_and_the_scoring_keeps_its_place(self):
        from engine.regrow import orig_key
        frozen = {"s5w0": window([[("Un.", "C")], [("Two.", "C")]])}            # p1's pick is now its original
        sc = scoring("s5 final", 0.2, {"s5w0 p1": "Un.", "s5w0 p2": "Dos."},
                     origs={"s5w0 p1": orig_key("One."), "s5w0 p2": orig_key("Two.")})
        with tempfile.TemporaryDirectory() as d:
            rp = bare(d, {"vetoes": [], "repairs": [], "for_you": []}, [sc], lay=frozen)
            got = rp.scores[0]
            self.assertEqual((got["picks"], got["absorbed"]), ({"s5w0 p2": "Dos."}, {"s5w0 p1": "s5w0 p1"}))
            self.assertNotIn("stale", got)
            self.assertIs(rp.best[5], got)                                                       # flag: still best
        repaired = {"s5w0": window([[("Un.", "C")], [("Two.", "C")]])}
        sc2 = dict(sc, picks={"s5w0 p1": "Un, you know.", "s5w0 p2": "Dos."})
        with tempfile.TemporaryDirectory() as d:          # the guide held the repair of the pick: that froze
            got = bare(d, {"vetoes": [], "repairs": [REPAIR], "for_you": []}, [sc2], lay=repaired).scores[0]
            self.assertEqual(got.get("absorbed"), {"s5w0 p1": "s5w0 p1"})
        other = {"s5w0": window([[("Something you wrote.", "C")], [("Two.", "C")]])}
        with tempfile.TemporaryDirectory() as d:          # pass: the span is gone and nothing holds the pick's text
            got = bare(d, {"vetoes": [], "repairs": [], "for_you": []}, [sc], lay=other).scores[0]
            self.assertEqual((got["stale"], got.get("absorbed")), (["s5w0 p1"], None))
