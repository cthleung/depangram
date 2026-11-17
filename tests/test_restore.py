"""Tests for the restore pass (plan 6.9; 2026-11-10): the gate's cases from experiments 10 and 10b, the scope rules
(min_words, locked runs, untouched paragraphs), the runs under an edited paragraph, the retry trigger, the jokes
switch, the lock the loop honors for restore entries, the wrap-up hook (off by default, a placeholder in rehearsal),
the two check fixes ("The thing is," off the assistant tells; the digit split) and the tiny-line rule. No model, no
Pangram, no Claude anywhere. Run from the repo root:

    python -m unittest discover tests
"""
import tempfile
import types
import unittest
from pathlib import Path

from engine import discord
from engine.checks import names_required, restated_pairs
from engine.job import Bank
from engine.restore import (GiveUp, RestorePass, bold_changes, fact_strings, gate, locked_inserts, parse_punch_list,
                            run_for_loop, split_runs, split_skipped)
from engine.run import Loop
from engine.draws import ASSISTANT_TELL, sents

DASH = "\u2014"
R1 = "The trouble is that the map has no idea which entrance is open today."
R2 = ("It treats the museum as one big box, so the floors, the wings and the closed stairwells all vanish, and "
      "it can't plan the route you actually walk.")
R3 = "Haddad et al. (2024), who had 18 volunteers test museum apps, filed this under its own heading."
R4 = "Which is a thing most maps don't have, probably."
RUNS = [R1, R2, R3, R4]
REGROWN = " ".join(RUNS)
ORIGINAL = ("Here's the catch: the map doesn't know which door is unlocked on any given day. It flattens the building "
            "into a single box, so it really can't plan the route you walk. Haddad et al. (2024), whose 18 "
            "volunteers tried museum apps for a week, filed this under its own heading, \"Lost in the Lobby\" "
            "(section 2.3.1). Most maps still don't handle it, probably.")
ITEMS = [{"n": 1, "kind": "fact", "text": '"really" is the softener on the claim about the route.'},
         {"n": 2, "kind": "bite", "text": "the contempt in the opening claim."}]


def edit(old, new):
    return REGROWN.replace(old, new, 1)


# ---------------------------------------------------------------------- the gate
class Gate(unittest.TestCase):
    def check(self, after, **kw):
        kw.setdefault("original", ORIGINAL)
        return gate(REGROWN, after, **kw)

    def test_a_line_edit_passes_and_a_long_insertion_is_out(self):
        self.assertIsNone(self.check(edit("can't plan", "can't really plan")))                        # pass
        self.assertIsNone(self.check(edit("has no idea", "has no fucking idea")))             # pass
        twelve = " ".join(["word"] * 12)
        self.assertIsNone(self.check(edit("as one big box", f"as one big box, {twelve},")))               # pass: 12
        self.assertIn("insertion of 13 words", self.check(edit("as one big box", f"as one big box, {twelve} word,")))
        self.assertIn("insertion of 7 words", self.check(edit("as one big box", "as one big box, a b c d e f g,"),
                                                         max_insert=6))                                       # flag: the knob

    def test_a_rewrite_a_merge_and_a_split_are_out(self):
        rewritten = edit(R2, "It sees one flat plan, so every route it gives you is a guess.")
        self.assertIsNotNone(self.check(rewritten))                                                         # flag
        self.assertIn("sentences became", self.check(edit("open today. It treats", "open today, and it treats")))
        self.assertIn("sentences became", self.check(edit("one big box, so", "one big box. So")))    # flag: split

    def test_facts_names_numbers_and_quotes_hold(self):
        self.assertIn("number lost", self.check(edit(" (2024)", "")))                                        # flag
        self.assertIn("adds a number", self.check(edit("had 18", "had 19")))                                 # flag
        self.assertIn("adds a name", self.check(edit("the map has no idea", "GPT has no idea")))             # flag
        self.assertIn("adds a name", self.check(edit("the map has no idea", "Sam has no idea")))             # flag
        self.assertIsNone(self.check(edit("has no idea", "REALLY has no idea")))                   # pass: CAPS stress
        after = edit("its own heading.", "its own heading, \"Lost in the Lobby\" (section 2.3.1).")
        self.assertIsNone(self.check(after, facts=["Lost in the Lobby", "section 2.3.1"]))    # pass: listed
        before = 'He said "the smell of the kitchen" twice. Then he left the room.'
        self.assertIn("quote lost", gate(before, 'He said "the smell of it" twice. Then he left the room.'))
        self.assertIn("name lost", gate("Jane Doe took a bow on X in 2025. Then she left.",
                                        "Jane took a bow on X in 2025. Then she left.", notes=["Jane Doe celebrated on X"]))

    def test_locked_sentences_house_words_and_keep_words(self):
        self.assertEqual(self.check(edit("don't have, probably", "don't have"), locked=[R4]), "touched a locked sentence")
        self.assertIn("em dash", self.check(edit(", so the floors", f" {DASH} so the floors")))         # flag
        self.assertIn("keep lost", self.check(edit("don't have, probably.", "don't have."), notes=['keep: "probably"']))
        self.assertEqual(self.check(REGROWN), "no change")

    def test_the_authors_own_opener_passes_and_a_tell_is_out(self):
        before = ("Every ticket desk hands you a paper map and a sticker for your coat. Most visitors lose the map "
                  "before the second gallery and keep the sticker on all day.")
        self.assertIsNone(gate(before, before.replace("Every ticket desk", "The thing is, every ticket desk")))  # pass
        after = before.replace("lose the map", "lose, at the end of the day, the map")
        self.assertEqual(gate(before, after), "assistant tell")                                             # flag
        self.assertIsNone(gate(before, after, original="At the end of the day, this gets exhausting."))     # pass: the original's own

    def test_a_made_up_sighting_is_out(self):
        before = ("Guides print bigger maps these days, because a survey found visitors kept getting lost. Try asking "
                  "at the desk and they hand you a bigger one.")
        after = before.replace("Try asking", "I've tested it myself, so try asking")
        self.assertIn("first-person sighting", gate(before, after))                                         # flag
        self.assertIsNone(gate(before, after, original="I've tested it myself on a museum map."))         # pass: the original's own
        self.assertIsNone(gate(before, before.replace("Try asking", "Honestly, try asking")))              # pass
        self.assertIsNone(gate(before, before.replace("bigger one.", "bigger one, I've got you.")))             # pass: an idiom
        self.assertIn("first-person sighting", gate(before, before.replace("Try asking", "I've done this myself, so try asking")))

    def test_a_restatement_the_edit_made_is_out(self):
        before = "The map labels every room twice. Then it moves on to the next floor and does it again."
        after = before.replace("does it again", "labels every room twice again")
        self.assertEqual(gate(before, after), "says it twice")                                              # flag
        self.assertIsNone(gate(before, before.replace("does it again", "does it again, honestly")))          # pass

    def test_a_digit_can_open_a_sentence(self):
        before = "Older guides mostly gave up, since nobody read them. 3.5 is the average rating now, so it's a sign."
        self.assertEqual(len(sents(before)), 2)                                                              # flag: two sentences
        self.assertIsNone(gate(before, before.replace(" 3.5 is", " Model 3.5 is"), original="Model 3.5 is the average"))  # pass
        self.assertEqual(len(sents("See No. 5 for more. Then it stopped.")), 2)                             # pass: an abbreviation holds
        # the checks that split sentences behave the same on the new split
        self.assertEqual(names_required("Jane Doe took a bow. 3.5 did too, says Jane Doe."), ["Jane Doe"])
        self.assertEqual(restated_pairs(sents("It costs a lot. 5 dollars is a lot to pay for it.")), [])


# ---------------------------------------------------------------------- the runs under a paragraph
class Runs(unittest.TestCase):
    def test_an_edited_paragraph_cuts_back_into_its_runs(self):
        after = edit("can't plan", "can't really plan").replace("The trouble is", "The trouble, though, is")
        got = split_runs(RUNS, after)
        self.assertEqual(got[0], "The trouble, though, is that the map has no idea which entrance is open today.")
        self.assertEqual(got[2:], [R3, R4])                                                                  # untouched runs byte for byte
        self.assertEqual(split_runs([R1, "", R2], R1 + " " + R2.replace("can't", "can't really")),
                         [R1, "", R2.replace("can't", "can't really")])                                      # a cut run stays cut
        self.assertIsNone(split_runs(RUNS, R1 + " " + R2 + " " + R4))                                       # a lost run: no cut
        self.assertEqual(split_runs([R1, R2], "The thing is, " + R1[0].lower() + R1[1:] + " " + R2)[0],
                         "The thing is, " + R1[0].lower() + R1[1:])                                          # an opener joins its run

    def test_bold_marks_the_changes(self):
        self.assertEqual(bold_changes("It can't find the stairs.", "It can't really find the stairs, honestly."),
                         "It can't **really** find the stairs**, honestly**.")

    def test_the_skip_line_comes_off(self):
        para, line, nums = split_skipped("Edited paragraph here.\nskipped: 2 (it asked for a split)", 3)
        self.assertEqual((para, nums), ("Edited paragraph here.", [2]))
        self.assertEqual(split_skipped("Edited.\nskipped: none")[1:], ("skipped: none", []))


# ---------------------------------------------------------------------- the punch list and the jokes switch
class PunchList(unittest.TestCase):
    TEXT = ('1. fact: "section 2.3.1" names the subsection.\n2. joke: the target is the jargon.\n'
            "3. bite: the contempt in the first claim.\n4. flat: the last sentence should carry the shrug.\n"
            "5. swear: second sentence.\nkeep: the last sentence.")

    def test_jokes_off_drops_joke_items_and_the_experiments_swear_label(self):
        items, keep = parse_punch_list(self.TEXT, ("fact", "point", "bite", "flat"))
        self.assertEqual([i["kind"] for i in items], ["fact", "bite", "flat"])                              # flag: joke out
        self.assertEqual([i["n"] for i in items], [1, 2, 3])                                                  # renumbered
        self.assertEqual(keep, "the last sentence.")
        self.assertEqual(fact_strings(items), ["section 2.3.1"])

    def test_jokes_on_keeps_them(self):
        items, _ = parse_punch_list(self.TEXT, ("fact", "point", "bite", "flat", "joke"))
        self.assertEqual([i["kind"] for i in items], ["fact", "joke", "bite", "flat"])                      # pass

    def test_the_director_prompt_names_jokes_only_when_on(self):
        for on in (False, True):
            rp = fake_pass(jokes=on)
            seen = {}
            rp.call = lambda prompt, role, model, usd, system, seen=seen: (seen.update(prompt=prompt, model=model), ("1. nothing\nkeep: all", {"cost_usd": 0.0}))[1]
            rp.direct(target())
            self.assertEqual(seen["model"], "opus")
            self.assertEqual("- joke:" in seen["prompt"], on)
            self.assertEqual("Jokes are off" in seen["prompt"], not on)


# ---------------------------------------------------------------------- the pass: scope, retry, locks
def fake_pass(jokes=False, draws=2, tmp=None):
    rp = RestorePass.__new__(RestorePass)
    rp.director_model, rp.editor_model, rp.retry_model = "opus", "sonnet", "opus"
    rp.jokes, rp.max_words, rp.max_change, rp.min_words, rp.draws = jokes, 12, 0.30, 25, draws
    rp.labels = tuple(x for x in ("fact", "point", "bite", "flat", "joke") if x != "joke" or jokes)
    rp.voice, rp.samples, rp.idiom_text = "VOICE", "SAMPLES", "IDIOMS"
    rp.swear_rates, rp.swear_mult = {"all": 0.02, "strong": 0.01}, 1.5
    rp.claude_spent, rp.claude_usd, rp.budget, rp.settings, rp.conn = 0.0, 3.0, None, None, None
    rp.log = lambda msg: None
    rp.cache_path = Path(tmp) / "restore cache.json" if tmp else None
    rp._cache = lambda: getattr(rp, "_mem", None) if tmp is None else RestorePass._cache(rp)
    if tmp is None:
        rp._mem = {}
        rp._save = lambda cache: None
    return rp


def target(items=None):
    return {"label": "s1w1 p2", "wid": "s1w1", "pi": 1, "section": 1, "original": ORIGINAL, "regrown": REGROWN,
            "runs": list(RUNS), "run_lists": [sents(r) for r in RUNS], "notes": [], "provenance": [], "locked": [],
            "locked_runs": [], "rest": (1000, 4, 1), "swear_room": "room for 2", "items": list(ITEMS if items is None else items),
            "keep": "", "facts": ["really"]}


class Editing(unittest.TestCase):
    GOOD = edit("can't plan", "can't really plan").replace("The trouble is", "The trouble, honestly, is")
    BROKEN = edit("can't plan", "can't plan really")
    BAD = edit(R2, "It sees one flat plan, so every route it gives you is a guess.")

    def run_edit(self, replies, checks):
        """replies: the editor's raw replies per (arm, draw); checks: the checker's JSON per after text."""
        rp = fake_pass()
        calls = []

        def call(prompt, role, model, usd, system):
            calls.append((role, model))
            if role in ("editor", "editor retry"):
                return replies.pop(0) + "\nskipped: none", {"cost_usd": 0.1}
            after = prompt.split("After:\n", 1)[1].rsplit("\n\nReply", 1)[0].strip()
            return checks.get(after, '{"applied": [1, 2], "broken": [], "rewritten": []}'), {"cost_usd": 0.02}
        rp.call = call
        t = target()
        picked = rp.edit(t)
        return picked, t, calls

    def test_a_clean_sonnet_draw_is_taken(self):
        picked, t, calls = self.run_edit([self.GOOD], {})
        self.assertEqual((picked, t["arm"]), (self.GOOD, "editor"))                                            # pass
        self.assertEqual([m for _, m in calls], ["sonnet", "sonnet"])                                          # no Opus

    def test_every_sonnet_draw_failing_the_gate_brings_opus_once(self):
        picked, t, calls = self.run_edit([self.BAD, self.BAD, self.GOOD], {})
        self.assertEqual((picked, t["arm"]), (self.GOOD, "retry"))                                             # flag: Opus edited
        self.assertEqual([r for r, _ in calls], ["editor", "editor", "editor retry", "check"])
        self.assertEqual([d["arm"] for d in t["draws"]], ["editor", "editor", "retry"])

    def test_a_broken_sentence_from_the_checker_brings_opus_and_then_the_pass_gives_up(self):
        broken = '{"applied": [1], "broken": ["can\'t picture really"], "rewritten": []}'
        picked, t, calls = self.run_edit([self.BROKEN, self.GOOD], {self.BROKEN: broken})
        self.assertEqual((picked, t["arm"]), (self.GOOD, "retry"))                                             # flag: the retry fired
        self.assertEqual([r for r, _ in calls], ["editor", "check", "editor retry", "check"])
        picked, t, calls = self.run_edit([self.BROKEN, self.BROKEN], {self.BROKEN: broken})
        self.assertIsNone(picked)                                                                              # flag: given up
        self.assertIn("Opus's draw did too", t["why"])
        picked, t, _ = self.run_edit([self.BAD, self.BAD, self.BAD], {})
        self.assertIn("Opus's draw failed the gate", t["why"])

    def test_an_editor_that_declines_every_item_stops_the_draws(self):
        rp = fake_pass()
        rp.call = lambda prompt, role, model, usd, system: (REGROWN + "\nskipped: 1, 2 (no room)", {"cost_usd": 0.1})
        t = target()
        self.assertIsNone(rp.edit(t))
        self.assertTrue(t["why"].startswith("declined"))
        self.assertEqual(len(t["draws"]), 1)

    def test_the_claude_ceiling_ends_the_pass(self):
        rp = fake_pass()
        rp.claude_spent = 3.0
        with self.assertRaises(GiveUp):
            rp.call("p", "editor", "sonnet", 0.3, "s")


class Scope(unittest.TestCase):
    def lay_and_rows(self, bank_dir, regrown=True):
        orig = [[["Author line one."]], [["Claude a."], ["Claude b."]], [["Claude c."]]]
        runs = [[{"sents": ["Author line one."], "claude": False, "author": True}],
                [{"sents": ["Claude a."], "claude": True}, {"sents": ["Claude b."], "claude": True}],
                [{"sents": ["Claude c."], "claude": True}]]
        lay = {"s1w1": {"section": 1, "heading": None, "runs": runs, "original": orig}}
        long_a = "Regrown a with enough words in it to pass the floor, which takes a sentence this long to reach."
        long_b = "Regrown b with enough words in it as well, so the paragraph clears twenty-five words easily."
        rows = [{"label": "s1w1 p1", "section": 1, "wid": "s1w1", "pi": 0, "text": "Author line one.",
                 "runs": ["Author line one."], "original": "Author line one.", "fixable": True, "why": ""},
                {"label": "s1w1 p2", "section": 1, "wid": "s1w1", "pi": 1, "text": long_a + " " + long_b,
                 "runs": [long_a, long_b], "original": "Claude a. Claude b.", "fixable": True, "why": ""},
                {"label": "s1w1 p3", "section": 1, "wid": "s1w1", "pi": 2, "text": "Short regrown c.",
                 "runs": ["Short regrown c."], "original": "Claude c.", "fixable": True, "why": ""}]
        bank = Bank(Path(bank_dir) / "bank.jsonl")
        bank.add({"kind": "round", "window": "s1w1", "ai": 0.1, "text": "t",
                  "runs": [[["Author line one."]], [[long_a], [long_b]], [["Short regrown c."]]]})
        return lay, rows, bank, long_a, long_b

    def test_untouched_short_and_locked_paragraphs_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            lay, rows, bank, long_a, long_b = self.lay_and_rows(d)
            rp = fake_pass(tmp=d)
            rp.bank, rp.job = bank, types.SimpleNamespace(_cache=lambda: {})
            got = rp.scope(rows, lay, "")
            self.assertEqual(list(got), ["s1w1 p2"])                                                           # only the regrown long one
            self.assertIn("didn't regrow", rows[0]["why"])                                                      # flag: author paragraph
            self.assertIn("under 25 words", rows[2]["why"])                                                     # flag: min_words
            # a restore entry locks the runs it edited: one locked run stays byte for byte, every run locked skips
            edited_a = long_a.replace("enough", "really enough")
            bank.add({"kind": "restore", "window": "s1w1", "ai": None, "text": "t2", "arm": "restore",
                      "runs": [[["Author line one."]], [[edited_a], [long_b]], [["Short regrown c."]]],
                      "restores": [{"para": 1, "run": 0, "before": [long_a], "after": [edited_a]}]})
            rows[1]["runs"], rows[1]["text"] = [edited_a, long_b], edited_a + " " + long_b
            rows[1]["fixable"] = True
            got = rp.scope(rows, lay, "")
            self.assertEqual(got["s1w1 p2"]["locked"], [edited_a])                                             # pass: the other run is free
            self.assertEqual(got["s1w1 p2"]["locked_runs"], [0])
            bank.add({"kind": "restore", "window": "s1w1", "ai": None, "text": "t3", "arm": "restore",
                      "runs": [[["Author line one."]], [[edited_a], [long_b]], [["Short regrown c."]]],
                      "restores": [{"para": 1, "run": 1, "before": [long_b], "after": [long_b]}]})
            rows[1]["fixable"] = True
            self.assertEqual(rp.scope(rows, lay, ""), {})
            self.assertIn("every run is locked", rows[1]["why"])                                               # flag


class BankRoundTrip(unittest.TestCase):
    def test_the_lock_reads_what_the_pass_writes(self):
        """bank_entries writes a restore entry; locked_inserts and the loop's lock read it back."""
        with tempfile.TemporaryDirectory() as d:
            bank = Bank(Path(d) / "bank.jsonl")
            bank.add({"kind": "round", "window": "s1w1", "ai": 0.1, "text": "t", "runs": [[[R1], [R2]]]})
            rp = fake_pass(tmp=d)
            rp.bank = bank
            lay = {"s1w1": {"section": 1, "heading": None, "original": [[["Claude one."], ["Claude two."]]],
                            "runs": [[{"sents": ["Claude one."], "claude": True}, {"sents": ["Claude two."], "claude": True}]]}}
            rows = [{"label": "s1w1 p1", "section": 1, "wid": "s1w1", "pi": 0, "runs": [R1, R2], "text": R1 + " " + R2,
                     "original": "Claude one. Claude two.", "fixable": True}]
            new_r1 = "The trouble, honestly, is that the map has no idea which entrance is open today."
            t = dict(target(), label="s1w1 p1", pi=0, runs=[R1, R2], run_lists=[[R1], [R2]], arm="editor",
                     check={"applied": [1], "partly": [], "ignored": [], "note": "fine"})
            results = {"s1w1 p1": {"runs": [new_r1, R2], "rejected": [],
                                   "accepted": [{"run": 0, "para": 0, "sentence": R1, "replacement": new_r1,
                                                 "category": "restore", "by": "editor", "why": "1. bite: contempt"}]}}
            best = rp.bank_entries(lay, rows, results, {"s1w1 p1": t}, {1: {"before": 0.2, "after": 0.21}}, [1], "now")
            self.assertEqual(best["s1w1"][1], [new_r1 + " " + R2])
            e = bank.entries[-1]
            self.assertEqual((e["kind"], e["arm"], e["section_ai"]), ("restore", "restore", 0.21))
            self.assertEqual(e["restores"][0]["arm"], "editor")
            self.assertEqual(locked_inserts(bank, "s1w1"), {(0, 0): ([R1], [new_r1])})
            lp = Loop.__new__(Loop)
            lp.bank, lp.log = bank, lambda m: None
            ws = types.SimpleNamespace(wid="s1w1", current=[[[R1], [R2]]], original=[[[R1], [R2]]])
            self.assertEqual(Loop.apply_flavor(lp, ws), [(0, 0)])                                               # the window resumes with it
            self.assertEqual(ws.current, e["runs"])
            self.assertTrue(Loop.flavor_locked(lp, ws, 0, 0))                                                   # and never targets it
            self.assertFalse(Loop.flavor_locked(lp, ws, 0, 1))


class Locks(unittest.TestCase):
    def test_the_loop_reads_restore_entries_like_flavor_inserts(self):
        with tempfile.TemporaryDirectory() as d:
            b = Bank(Path(d) / "bank.jsonl")
            before, after = [R1], ["The trouble, honestly, is that the map has no idea which entrance is open today."]
            b.add({"kind": "round", "window": "s1w1", "ai": 0.0, "text": "t", "runs": [[before, [R2]]]})
            b.add({"kind": "restore", "window": "s1w1", "ai": None, "text": "t2", "arm": "restore", "runs": [[after, [R2]]],
                   "restores": [{"para": 0, "run": 0, "before": before, "after": after}]})
            b.add({"kind": "flavor", "window": "s1w1", "ai": None, "text": "t3", "runs": [[after, [R2, "Haha, no."]]],
                   "inserts": [{"para": 0, "run": 1, "sentence": "Haha, no."}]})
            locks = locked_inserts(b, "s1w1")
            self.assertEqual(locks[(0, 0)], (before, after))
            self.assertEqual(locks[(0, 1)], ([R2], [R2, "Haha, no."]))
            lp = Loop.__new__(Loop)
            lp.bank, lp.log = b, lambda m: None
            ws = types.SimpleNamespace(wid="s1w1", current=[[before[:], [R2]]], original=[[before[:], [R2]]])
            self.assertFalse(Loop.flavor_locked(lp, ws, 0, 0))                                                  # pass: not yet applied
            self.assertEqual(Loop.apply_flavor(lp, ws), [(0, 0), (0, 1)])
            self.assertEqual(ws.current, [[after, [R2, "Haha, no."]]])
            self.assertTrue(Loop.flavor_locked(lp, ws, 0, 0))                                                   # flag: locked
            self.assertTrue(Loop.flavor_locked(lp, ws, 0, 1))
            self.assertEqual(Loop.apply_flavor(lp, ws), [])                                                     # once
            ws.must = {}
            lp.best = lambda ws: b.entries[0]
            lp.bank = b
            ws.runs = [[{"sents": before, "claude": True}, {"sents": [R2], "claude": True}]]
            self.assertEqual(Loop.ship_entry(lp, ws)["kind"], "flavor")                                         # the newest pass entry ships


# ---------------------------------------------------------------------- the wrap-up hook
class WrapUp(unittest.TestCase):
    def fake_loop(self, cfg, smoke=False, window=None, rehearsal=False):
        logs = []
        return types.SimpleNamespace(settings=types.SimpleNamespace(config={"restore": cfg}), rehearsal=rehearsal,
                                     args=types.SimpleNamespace(smoke=smoke, window=window), whole_sections=lambda: [1],
                                     log=logs.append, logs=logs, real_file=None)

    def test_off_by_default_and_a_placeholder_in_rehearsal(self):
        lp = self.fake_loop({})
        self.assertEqual(run_for_loop(lp), {"skipped": "run_at_wrapup is off"})                               # pass: nothing runs
        self.assertEqual(discord.restore_lines(lp), [])
        lp = self.fake_loop({"run_at_wrapup": True}, smoke=True)
        self.assertEqual(run_for_loop(lp)["skipped"], "a smoke or partial run")
        lp = self.fake_loop({"run_at_wrapup": True}, rehearsal=True)
        self.assertTrue(run_for_loop(lp)["placeholder"])                                                       # flag: the placeholder
        self.assertIn("placeholder", lp.logs[0])
        self.assertIn("Restore pass: skipped", discord.restore_lines(lp)[1])


# ---------------------------------------------------------------------- the check fix: the author's opener
class Tells(unittest.TestCase):
    def test_the_thing_is_is_not_a_tell_and_heres_the_thing_still_is(self):
        self.assertIsNone(ASSISTANT_TELL.search("The thing is, all of this gets exhausting."))                 # pass
        self.assertIsNotNone(ASSISTANT_TELL.search("Here's the thing, all of this gets exhausting."))          # flag


# ---------------------------------------------------------------------- the tiny-line rule
class TinyLines(unittest.TestCase):
    def loop(self):
        lp = Loop.__new__(Loop)
        lp.tiny_words, lp.tiny_tries, lp.bank, lp.rewrite_cleared = 6, 3, Bank.__new__(Bank), False
        lp.bank.entries = []
        lp.logs = []
        lp.log = lp.logs.append
        lp.flavor_inserts = lambda ws: {}
        return lp

    def window(self):
        runs = [[{"sents": ["They're everywhere:"], "claude": True},
                 {"sents": ["A longer Claude sentence that the loop keeps trying to regrow all day."], "claude": True}]]
        ws = types.SimpleNamespace(wid="s2w1", runs=runs, forced=set(), rewrites=set(), scored_runs=None, flagged=[],
                                   tiny_misses={}, tiny_skipped={}, passes_today=1, ai=None, segs=[])
        ws.original = [[r["sents"][:] for r in p] for p in runs]
        ws.current = [[r[:] for r in p] for p in ws.original]
        ws.n_flagged = lambda: 0
        ws.claude_text_left = lambda: True
        return ws

    def test_three_misses_on_a_tiny_run_skip_it_and_a_long_run_never(self):
        lp, ws = self.loop(), self.window()
        for _ in range(3):
            Loop.tiny_note(lp, ws, 0, 0, accepted=False)
            Loop.tiny_note(lp, ws, 0, 1, accepted=False)
        self.assertTrue(Loop.tiny_skip(lp, ws, 0, 0))                                                           # flag: tiny, 3 misses
        self.assertFalse(Loop.tiny_skip(lp, ws, 0, 1))                                                          # pass: 12 words
        self.assertEqual(Loop.targets(lp, ws, False), [(0, 1, False)])                                          # the tiny run is out
        self.assertEqual(len([m for m in lp.logs if "skipped as tiny" in m]), 1)                                # logged once
        Loop.tiny_skip(lp, ws, 0, 0)
        self.assertEqual(len([m for m in lp.logs if "skipped as tiny" in m]), 1)
        self.assertEqual(ws.tiny_skipped, {(0, 0): "They're everywhere:"})
        lines = discord.tiny_lines(types.SimpleNamespace(windows=[ws], tiny_words=6, tiny_tries=3))
        self.assertIn("s2w1 p1.1 \"They're everywhere:\" (3 tries with no change)", lines[-1])

    def test_two_misses_and_an_accepted_change_keep_it_in_play(self):
        lp, ws = self.loop(), self.window()
        Loop.tiny_note(lp, ws, 0, 0, accepted=False)
        Loop.tiny_note(lp, ws, 0, 0, accepted=False)
        self.assertFalse(Loop.tiny_skip(lp, ws, 0, 0))                                                          # pass: under the count
        Loop.tiny_note(lp, ws, 0, 0, accepted=True)
        Loop.tiny_note(lp, ws, 0, 0, accepted=False)
        self.assertEqual(ws.tiny_misses[(0, 0)], 1)                                                             # an accepted change resets it
        self.assertEqual(Loop.targets(lp, ws, False), [(0, 0, False), (0, 1, False)])
        lp.tiny_tries = 0
        ws.tiny_misses[(0, 0)] = 9
        self.assertFalse(Loop.tiny_skip(lp, ws, 0, 0))                                                          # pass: the rule off


if __name__ == "__main__":
    unittest.main()
