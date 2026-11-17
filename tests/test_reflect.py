"""Tests for the reflection (engine/reflect.py; plan 9, 6.8): the gather from a bank, the house check, the retry
and the fail-twice paths, the digest sections, the rehearsal placeholder and the smoke skip. Every test builds its
own job folder and ledger; nothing reads the real ones or calls Claude. Run from the repo root:

    python -m unittest discover tests
"""
import datetime as dt
import sqlite3
import tempfile
import types
import unittest
from pathlib import Path

from engine import reflect as R
from engine.budget import RUN_DAYS
from engine.job import Bank, Job
from engine.settings import Settings
from test_m3 import LINES, ORIG, REGROWN

from engine import checks

# A test rules file with two [reflection] rules, so the tests below have a word rule and a number rule to break.
# The shipped rules.toml has neither (RulesFromTheFile checks that).
TEST_RULES = '[reflection]\nbanned_words = ["half*"]\ninvented_digits = ["4", "7"]\n'
_rules_dir = None


def setUpModule():
    global _rules_dir
    _rules_dir = tempfile.TemporaryDirectory()
    path = Path(_rules_dir.name) / "rules.toml"
    path.write_text(TEST_RULES, encoding="utf-8")
    checks.use_rules(path)


def tearDownModule():
    checks.use_rules()
    _rules_dir.cleanup()

EM = "\u2014"                                  # the em dash, as an escape: this file carries none itself


def ts(day, h, m=0):
    return dt.datetime(2026, 11, day, h, m).timestamp()


def make_job(d):
    d = Path(d)
    (d / "job.toml").write_text('[job]\nname = "t"\noriginal = "orig.md"\nworking = "work.md"\nsections = [1]\n'
                                'voice_samples = [9]\n\n[seed]\nbanks = []\nwindows = ["s1w0"]\n', encoding="utf-8")
    (d / "orig.md").write_text(ORIG, encoding="utf-8")
    (d / "your lines.md").write_text(LINES, encoding="utf-8")
    job = Job(root=d)
    text = job.windows()[0]["text"]
    claude_line = "Claude wrote this line about tokens and words."
    yours = ["I wrote this bit myself, mate."]
    same = ["Another paragraph that stays exactly the same here."]
    b = Bank(d / "bank.jsonl")
    b.add({"kind": "start", "window": "s1w0", "ai": 1.0, "segs": [], "text": text, "time": "2026-11-08 18:41"})
    b.add({"kind": "round", "window": "s1w0", "ai": 0.5, "text": "a", "time": "2026-11-08 19:00", "day": "2026-11-08",
           "pass": 1, "runs": [[[claude_line], yours], [same]], "changes": []})
    b.add({"kind": "round", "window": "s1w0", "ai": 0.1, "text": "b", "time": "2026-11-09 03:10", "day": "2026-11-09",
           "pass": 2, "runs": [[[REGROWN], yours], [same]],
           "changes": [{"para": 0, "run": 0, "orig": claude_line, "new": REGROWN, "arm": "notes",
                        "result": "replaced (3 passed filters, head-to-head: B)"}]})
    conn = sqlite3.connect(":memory:")
    conn.executescript(RUN_DAYS)
    conn.execute("INSERT INTO run_days VALUES ('2026-11-09', ?, ?, 1)", (ts(9, 3, 1), ts(9, 10, 38)))
    return job, b, conn


def clean_draft(extra_working="", extra_push="", drop_heading=None, words=270):
    """A draft that passes the house check: four headings, enough plain sentences, nothing on the lists."""
    filler = ("The adapter wrote the world-model paragraph in two draws and Pangram read the section as human. "
              "The judge kept every citation. The few-shot arm still needs more draws on the filler sentence. ")
    body = (filler * (words // 100 + 1)).strip()                    # about a third of `words` per heading
    parts = []
    for h in R.HEADINGS:
        if h == drop_heading:
            continue
        extra = extra_working if h == R.HEADINGS[0] else extra_push if h == R.HEADINGS[3] else ""
        text = f"1. {extra} {body[:120]}".strip() if h == R.HEADINGS[3] else f"{body} {extra}".strip()
        parts.append(f"### {h}\n\n{text}\n")
    return "\n".join(parts)


class Gather(unittest.TestCase):
    def test_the_record_builds_from_a_bank_and_the_prompt_carries_it(self):
        with tempfile.TemporaryDirectory() as d:
            job, bank, conn = make_job(d)
            settings = Settings(env={}, config={"loop": {"model": "m", "primary_arm": "notes", "fallback_arm": "fewshot"}})
            rec = R.record_from_files("2026-11-09", settings, job, bank, conn, digests_dir=d, now=ts(9, 16))
            self.assertIn(REGROWN, rec["now"][1])                        # the day's pick is what ships now
            self.assertIn("Claude wrote this line", rec["start"][1])      # the day began on the 2026-11-08 best
            self.assertEqual([(r["label"], r["arms"], r["yours"]) for r in rec["changed"]], [("s1w0 p1", ["notes"], 1)])
            self.assertIn("Claude wrote this line", rec["changed"][0]["before"])
            self.assertIn(REGROWN, rec["changed"][0]["after"])
            self.assertEqual(rec["passes"], [])
            self.assertIsNone(rec["previous"])
            prompt = R.render_prompt(rec)
            for h in R.HEADINGS:
                self.assertIn(f"### {h}", prompt)
            self.assertIn("written by the notes arm", prompt)
            self.assertIn("(no digest file for 2026-11-09)", prompt)
            self.assertIn(R.banned_word_stems()[0], prompt)                  # the rules name the banned words

    def test_a_fix_banked_after_the_loop_stopped_still_counts_as_today(self):
        with tempfile.TemporaryDirectory() as d:
            job, bank, conn = make_job(d)
            fixed = REGROWN.replace("put", "placed")
            bank.add({"kind": "fix", "window": "s1w0", "ai": None, "time": "2026-11-09 11:18", "section": 1,
                      "runs": [[[fixed], ["I wrote this bit myself, mate."]],
                               [["Another paragraph that stays exactly the same here."]]],
                      "fixes": [{"para": 0, "run": 0, "category": "grammar", "sentence": REGROWN, "replacement": fixed,
                                 "why": "test"}]})
            rec = R.record_from_files("2026-11-09", Settings(env={}, config={}), job, bank, conn, digests_dir=d,
                                      now=ts(9, 16))
            self.assertIn(fixed, rec["now"][1])
            self.assertEqual(rec["changed"][0]["passes"], ["fix pass"])
            self.assertEqual(len(rec["passes"]), 1)

    def test_the_digest_is_cut_before_the_couldnt_crack_part(self):
        report = "# digest\n\n## Windows\n\ntable\n\n## Reflection\n\nold text\n\n## Spend\n\nmoney\n\n## Couldn't crack so far\n\nstuck\n\n## Timeline\n"
        cut = R.cut_digest(report)
        self.assertIn("## Windows", cut)
        self.assertIn("## Spend", cut)
        self.assertNotIn("Couldn't crack", cut)
        self.assertNotIn("old text", cut)                                 # an earlier reflection is never input


class HouseCheck(unittest.TestCase):
    def test_an_em_dash_is_fixed_not_flagged(self):
        fixed = R.fix_dashes(f"It moved {EM} a lot {EM}today. Also--this.")
        self.assertEqual(fixed, "It moved - a lot - today. Also - this.")
        self.assertEqual(R.gate(clean_draft(extra_working=f"One more {EM} fine."), set()), [])   # reflect() fixes first
        hits = R.gate(R.fix_dashes(clean_draft(extra_working=f"One more {EM} fine.")), set())
        self.assertEqual(hits, [])

    def test_a_banned_word_half_and_an_invented_47_are_caught(self):
        word = R.banned_word_stems()[0]
        text = clean_draft(extra_working=f"The {word} problem is the opener. Most of it took half a day. It took 47 tries.")
        rules = {(h["rule"], h["match"]) for h in R.gate(text, set())}
        self.assertIn(("banned word", word), rules)
        self.assertIn(("half", "half"), rules)
        self.assertIn(("invented number with a 4 or a 7", "47"), rules)
        # the same 47 quoted from the material is counted, so fine
        allowed = R.allowed_numbers("the table says 47 draws and 0.43")
        rules = {(h["rule"], h["match"]) for h in R.gate(text, allowed)}
        self.assertNotIn(("invented number with a 4 or a 7", "47"), rules)
        self.assertIn(("half", "half"), rules)

    def test_the_banned_phrases_the_triple_and_a_quote(self):
        text = clean_draft(extra_working="It's not the quotes, it's the lead-ins. The prose is calm, flat, and glued."
                           " The joke landed. Here's the thing. The kind of line that reads well."
                           " \"Half the fun is the swearing\" is quoted and fine.")
        rules = [h["rule"] for h in R.gate(text, set())]
        for r in ("it's not X, it's Y", "rule of three", "landed for impact", "here's the thing", "the kind of"):
            self.assertIn(r, rules)
        self.assertNotIn("half", rules)

    def test_headings_bullets_and_length(self):
        hits = R.gate(clean_draft(drop_heading="The guide") + "\n### Extra\n\n- a bullet\n", set())
        matches = {(h["rule"], h["match"]) for h in hits}
        self.assertIn(("headings", "missing: The guide"), matches)
        self.assertIn(("headings", "a heading beyond the five: Extra"), matches)
        self.assertIn(("bullet list", "- a bullet"), matches)
        short = "\n".join(f"### {h}\n\nShort.\n" for h in R.HEADINGS)
        self.assertEqual([h["rule"] for h in R.gate(short, set())], ["length"])
        self.assertEqual(R.gate(short, set(), strict_length=False), [])


def fake_claude(replies):
    calls = []

    def fn(prompt):
        calls.append(prompt)
        return replies[min(len(calls), len(replies)) - 1], {"cost_usd": 0.1}
    fn.calls = calls
    return fn


class Call(unittest.TestCase):
    def setUp(self):
        self.rec = {"day": "2026-11-09", "stopped": "test", "model": "m", "arms": "a", "digest": "tables with 0.43",
                    "sections": [1], "now": {1: "now text"}, "start": {1: "start text"}, "changed": [], "passes": [],
                    "couldnt": "nothing", "previous": None, "decisions": [], "guide": []}

    def test_a_clean_first_draft_is_one_call(self):
        fn = fake_claude([clean_draft()])
        out = R.reflect(self.rec, fn, log=lambda *a: None)
        self.assertTrue(out["ok"])
        self.assertEqual((out["attempts"], out["cost"], out["dropped"]), (1, 0.1, []))
        self.assertEqual(len(fn.calls), 1)

    def test_a_hit_gets_one_retry_with_the_hits_named(self):
        word = R.banned_word_stems()[0]
        fn = fake_claude([clean_draft(extra_working=f"The {word} problem is the opener."), clean_draft()])
        out = R.reflect(self.rec, fn, log=lambda *a: None)
        self.assertTrue(out["ok"])
        self.assertEqual((out["attempts"], out["cost"]), (2, 0.2))
        self.assertIn("Your first draft broke these house rules", fn.calls[1])
        self.assertIn(f'banned word: "{word}"', fn.calls[1])
        self.assertEqual(out["dropped"], [])

    def test_a_hit_on_the_retry_drops_its_sentence(self):
        bad = "Most of it took half a day."
        fn = fake_claude([clean_draft(extra_working=bad), clean_draft(extra_working=bad)])
        out = R.reflect(self.rec, fn, log=lambda *a: None)
        self.assertTrue(out["ok"])
        self.assertEqual(out["dropped"], [bad])
        self.assertNotIn("half", out["text"])
        self.assertEqual(out["attempts"], 2)

    def test_failing_twice_keeps_the_draft(self):
        fn = fake_claude([clean_draft(drop_heading="The guide"), clean_draft(drop_heading="The guide")])
        out = R.reflect(self.rec, fn, log=lambda *a: None)
        self.assertFalse(out["ok"])
        self.assertIsNone(out["text"])
        self.assertIn("headings", [h["rule"] for h in out["hits"]])
        with tempfile.TemporaryDirectory() as d:
            p = R.write_reflection(out, "2026-11-09", out_dir=d)
            self.assertEqual(p.name, "2026-11-09 (draft).md")
            self.assertIn("## What the house check caught", p.read_text(encoding="utf-8"))
            ok = R.write_reflection(dict(out, ok=True, text=clean_draft(), dropped=[]), "2026-11-09", out_dir=d,
                                    tag="by hand")
            self.assertEqual(ok.name, "2026-11-09 (by hand).md")
            self.assertIn("### What's working", ok.read_text(encoding="utf-8"))
            # the next day's gather reads the by-hand one and never the draft
            self.assertEqual(R.previous_reflection(d, before_day="2026-11-10")[0], "2026-11-09 (by hand)")
            self.assertIsNone(R.previous_reflection(d, before_day="2026-11-09"))


SNAP = {"pangram_today": 0.0, "pangram_day_cap": 6.0, "pangram_month": 0.0, "pangram_month_cap": 180.0,
        "claude_usd_today": 0.0, "backstop": None, "loop_points_today": 0.0, "allowance": 0.1, "loop_share": 0.0,
        "share_cap": 0.5, "real_total": 0.3, "ceiling": 0.9, "five_hour": 0.0, "five_hour_cap": 0.5,
        "week_reset": ts(10, 14), "rate": 0.007, "pangram_run_day": None}


def digest_loop(reflection):
    return types.SimpleNamespace(
        day="2026-11-09", rehearsal=False, job=types.SimpleNamespace(name="t"), model="m", stop_reason="the STOP file",
        clock=types.SimpleNamespace(stamp=lambda t=None: "2026-11-09 16:00"), windows=[],
        budget=types.SimpleNamespace(snapshot=lambda: SNAP), pools_used=0, pool_draws_used=0,
        queue=types.SimpleNamespace(pools={}), cap_events=[], cuts=[], timeline=[], reflection=reflection)


class Digest(unittest.TestCase):
    def test_the_report_carries_the_reflection_between_as_posted_and_the_spend(self):
        from engine.discord import digest_card, digest_report
        text = clean_draft(extra_working="The adapter did it.", extra_push="Run the flavor pass ($0.20 Pangram).")
        rep = digest_report(digest_loop({"ok": True, "text": text, "cost": 0.21, "model": "sonnet", "attempts": 1,
                                         "dropped": [], "path": "x/2026-11-09.md"}))
        i, j, k = rep.index("## As posted"), rep.index("## Reflection"), rep.index("## Spend against each cap")
        self.assertTrue(i < j < k)
        self.assertIn("### Where to push next", rep)
        self.assertIn("about $0.21 of Claude", rep)
        card = digest_card(digest_loop({"ok": True, "text": text}))
        self.assertIn("Reflection: The adapter wrote", card)
        self.assertIn("Push next: 1. Run the flavor pass ($0.20 Pangram).", card)
        self.assertLessEqual(len(card), 2000)

    def test_a_failed_or_skipped_reflection_says_so(self):
        from engine.discord import digest_card, digest_report
        rep = digest_report(digest_loop({"ok": False, "draft": "x", "hits": [{"rule": "headings", "match": "m"}]}))
        self.assertIn("didn't pass the house check today; its draft is in analytics/reflections/2026-11-09 (draft).md", rep)
        self.assertNotIn("Reflection:", digest_card(digest_loop({"ok": False})))
        self.assertIn("Skipped today: a smoke test.", digest_report(digest_loop({"skipped": "a smoke test"})))
        self.assertNotIn("## Reflection", digest_report(digest_loop(None)))


def run_loop(**over):
    def no_claude(*a, **k):
        raise AssertionError("Claude was called")
    lp = types.SimpleNamespace(args=types.SimpleNamespace(smoke=False, window=None), rehearsal=False,
                               whole_sections=lambda: [1], claude=no_claude, reflection=None,
                               settings=Settings(env={}, config={}), conn=None, log=lambda *a: None)
    for k, v in over.items():
        setattr(lp, k, v)
    return lp


class Skips(unittest.TestCase):
    def test_a_rehearsal_gets_the_placeholder_at_zero_spend(self):
        with tempfile.TemporaryDirectory() as d:
            logged = []
            lp = run_loop(rehearsal=True, reflections_dir=d, log=logged.append)
            out = R.run_for_loop(lp)
            self.assertTrue(out["ok"] and out["placeholder"])
            self.assertEqual(out["cost"], 0.0)
            self.assertIn("### Where to push next", out["text"])
            self.assertEqual(list(Path(d).iterdir()), [])                 # no reflection file for a rehearsal
            # the gather still runs in a rehearsal, and this stub has nothing for it to read: the log says so loudly
            self.assertTrue(any("REHEARSAL GATHER FAILED" in line for line in logged))

    def test_a_gather_that_fails_on_a_real_day_skips_with_the_reason(self):
        logged = []
        out = R.run_for_loop(run_loop(log=logged.append))
        self.assertIn("the gather failed", out["skipped"])
        self.assertTrue(any("the gather failed" in line for line in logged))

    def test_a_smoke_test_and_a_partial_run_are_skipped(self):
        self.assertEqual(R.run_for_loop(run_loop(args=types.SimpleNamespace(smoke=True, window="s1w0")))["skipped"],
                         "a smoke test")
        out = R.run_for_loop(run_loop(args=types.SimpleNamespace(smoke=False, window="s1w0")))
        self.assertIn("partial run", out["skipped"])
        self.assertIn("partial run", R.run_for_loop(run_loop(whole_sections=lambda: []))["skipped"])


class RulesFromTheFile(unittest.TestCase):
    def tearDown(self):
        checks.use_rules(Path(_rules_dir.name) / "rules.toml")

    def test_the_test_file_rules_reach_the_prompt(self):
        prompt = R.render_prompt({}, material="m")
        self.assertIn('Never "half" in any form.', prompt)
        self.assertIn("never one with a 4 or a 7 in it", prompt)

    def test_the_shipped_defaults_have_no_word_or_number_rule(self):
        checks.use_rules()
        self.assertEqual(R.reflection_rules(), {"banned_words": [], "invented_digits": []})
        hits = {rule for rule, _ in R.sentence_hits("Most of it took half a day and 47 tries.", set())}
        self.assertEqual(hits, set())
        prompt = R.render_prompt({}, material="m")
        self.assertNotIn("in any form.", prompt)
        self.assertNotIn("never one with", prompt)


class FollowsTheJob(unittest.TestCase):
    def test_the_description_is_filled_at_call_time(self):
        rec = {"description": "a field guide to garden birds"}
        self.assertIn("a field guide to garden birds", R.render_prompt(rec, material="m"))
        self.assertIn("a field guide to garden birds", R.system_prompt(rec))
        self.assertNotIn("<<DESCRIPTION>>", R.render_prompt({}, material="m"))
        self.assertNotIn("<<DESCRIPTION>>", R.system_prompt({}))

    def test_guide_state_follows_the_section_style(self):
        from engine.job import SectionStyle
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "doc.md"
            p.write_text("# Title\n\n### Robins\n\nRed chest, loud song.\n\n### Wrens\n\nTiny and brown.\n",
                         encoding="utf-8")
            self.assertEqual(R.guide_state(p, [2], SectionStyle(level=3)),
                             [(1, 5, False), (2, 4, True)])
            self.assertEqual(R.guide_state(p, [2]), [])                  # no "## N." headings in this file


if __name__ == "__main__":
    unittest.main()
