"""Tests for M1's run loop: the capper, the schedule's phases, the stuck rule, the drop rule, the Phase 2 fix list
(hedge and repeat must-keeps, the invented-sightings filter, the narrowed notation lock) and bank resume.
Run from the repo root:

    python -m unittest discover tests
"""
import datetime as dt
import sqlite3
import tempfile
import unittest
from pathlib import Path

from engine import db
from engine.budget import Budget, ClaudeCapReached, PangramCapReached
from engine.checks import (carries_fact, extra_checks_for, hedges_in, invented_sighting, is_template, keep_phrases,
                           must_keeps, pattern_quotes, repeat_ok, repeats_in, splice_quotes)
from engine.claude_call import ensure_schema
from engine.job import Bank, VerdictMap, align
from engine.run import drop_eligible, flag_list, stuck_rule
from engine.schedule import CompressedClock, Schedule
from engine.settings import Settings

CONFIG = {
    "pangram": {"usd_per_credit": 0.05, "words_per_credit": 100, "monthly_cap_usd": 180.0, "nightly_cap_usd": 6.0,
                "calibration_cap_usd": 15.0, "poll_seconds": 0, "timeout_seconds": 5},
    "claude": {"loop_share": 0.50, "total_ceiling": 0.90, "claude_until": "09:00", "night_claude_max_5h": 0.90,
               "day_claude_max_5h": 0.50, "first_night_cap_usd": 20.0, "week_points_per_usd": 0.01,
               "five_hour_points_per_usd": 0.05, "backstop_run_days": 1, "expected_cost_calls": 20},
    "schedule": {"start": "02:00", "last_attempt": "15:30", "stop": "16:00"},
}
DAY = dt.datetime(2026, 11, 9)
BULK = "loop earlier notes"        # bulk spend logged as notes calls, so it leaves the judge's expected cost alone


def ts(h, m=0, days=0):
    return (DAY + dt.timedelta(days=days, hours=h, minutes=m)).timestamp()


def make_budget(at):
    conn = sqlite3.connect(":memory:")
    conn.executescript(db.SCHEMA)
    ensure_schema(conn)
    clock = CompressedClock(at, speed=0.0)
    b = Budget(Settings(env={}, config=CONFIG), conn, clock, Schedule(CONFIG))
    return b, conn, clock


def pangram_row(conn, when, usd, purpose="loop"):
    conn.execute("INSERT INTO pangram_calls (ts, sha, model, words, credits, usd, purpose, label, status)"
                 " VALUES (?,?,?,?,?,?,?,?,?)", (when, "x", "pangram-4.1", 100, 1, usd, purpose, "", "ok"))
    conn.commit()


def claude_row(conn, when, cost, role="loop s1w0 judge", week=0.30, five=0.10, week_reset=None, five_reset=None):
    conn.execute("INSERT INTO claude_calls (ts, role, model, ok, cost_usd, week_util, five_hour_util, week_resets_at,"
                 " five_hour_resets_at) VALUES (?,?,?,?,?,?,?,?,?)",
                 (when, role, "sonnet", 1, cost, week, five, week_reset or ts(14, days=2), five_reset))
    conn.commit()


class PangramCaps(unittest.TestCase):
    def test_daily_cap_counts_only_the_loop_since_midnight(self):
        b, conn, _ = make_budget(ts(3))
        pangram_row(conn, ts(2, 30), 5.90)
        pangram_row(conn, ts(1), 3.00, purpose="experiment")       # not the loop's daily money
        pangram_row(conn, ts(23, days=-1), 4.00)                    # yesterday
        b.pangram_check(0.05)
        with self.assertRaises(PangramCapReached) as cm:
            b.pangram_check(0.20)
        self.assertEqual(cm.exception.check, "Pangram daily")

    def test_monthly_cap_counts_everything_this_month(self):
        b, conn, _ = make_budget(ts(3))
        pangram_row(conn, ts(12, days=-5), 100.0, purpose="calibration")
        pangram_row(conn, ts(12, days=-3), 79.90, purpose="experiment")
        with self.assertRaises(PangramCapReached) as cm:
            b.pangram_check(0.20, purpose="experiment")
        self.assertEqual(cm.exception.check, "Pangram monthly")

    def test_last_month_does_not_count(self):
        b, conn, _ = make_budget(ts(3))
        pangram_row(conn, ts(12, days=-12), 179.0)                  # 2026-10-28
        b.pangram_check(1.0)


class ClaudeCaps(unittest.TestCase):
    def test_ceiling_reads_the_real_total(self):
        b, conn, _ = make_budget(ts(3))
        claude_row(conn, ts(2, 50), 0.01, role="loop probe", week=0.898)
        with self.assertRaises(ClaudeCapReached) as cm:
            b.claude_check("judge", 0.3)
        self.assertEqual(cm.exception.check, "Claude weekly ceiling")

    def test_share_is_loop_dollars_times_the_rate(self):
        b, conn, _ = make_budget(ts(3))
        claude_row(conn, ts(12, days=-2), 49.9, role=BULK, week=0.55)  # 0.499 of the week, earlier this week
        self.assertAlmostEqual(b.loop_share(), 0.499)
        with self.assertRaises(ClaudeCapReached) as cm:
            b.claude_check("judge", 0.3)
        self.assertEqual(cm.exception.check, "Claude weekly share")

    def test_your_own_use_never_counts_as_the_loops(self):
        b, conn, _ = make_budget(ts(3))
        claude_row(conn, ts(2, 30), 1.0, role=BULK, week=0.70)       # your 3am session moved the reading
        self.assertAlmostEqual(b.loop_share(), 0.01)
        self.assertAlmostEqual(b.real_total(), 0.70)

    def test_allowance_spreads_the_share_over_the_days_left(self):
        b, conn, _ = make_budget(ts(3))
        claude_row(conn, ts(12, days=-2), 20.0, role=BULK, week=0.40)  # 0.20 used before today
        # reset is two days on at 14:00: run-days today, tomorrow and the day after are left
        self.assertAlmostEqual(b.allowance(), (0.50 - 0.20) / 3)
        claude_row(conn, ts(2, 30), 9.9, role=BULK, week=0.45)       # 0.099 today
        with self.assertRaises(ClaudeCapReached) as cm:
            b.claude_check("judge", 0.3)
        self.assertEqual(cm.exception.check, "Claude allowance today")

    def test_unused_allowance_carries_to_later_days(self):
        b, conn, _ = make_budget(ts(3, days=1))                       # tomorrow, nothing spent yesterday
        claude_row(conn, ts(12, days=-2), 0.0, role="loop probe")     # a reading, for the reset time
        self.assertAlmostEqual(b.allowance(), 0.50 / 2)

    def test_five_hour_cap_is_the_night_cap_before_claude_until(self):
        b, conn, _ = make_budget(ts(3))
        claude_row(conn, ts(2, 59), 0.02, five=0.90, five_reset=ts(5))
        with self.assertRaises(ClaudeCapReached) as cm:
            b.claude_check("judge", 0.3)
        self.assertEqual(cm.exception.check, "Claude 5-hour (night)")
        self.assertEqual(cm.exception.until, ts(5))

    def test_five_hour_cap_is_the_day_cap_after_claude_until(self):
        b, conn, clock = make_budget(ts(10))
        claude_row(conn, ts(9, 59), 0.02, five=0.45, five_reset=ts(13))
        b.claude_check("judge", 0.03)                                 # 0.45 + 0.0015 < 0.50
        claude_row(conn, ts(10), 0.02, five=0.50, five_reset=ts(13))
        with self.assertRaises(ClaudeCapReached) as cm:
            b.claude_check("judge", 0.3)
        self.assertEqual(cm.exception.check, "Claude 5-hour (day)")

    def test_the_same_reading_passes_at_night_and_fails_by_day(self):
        for at, ok in ((ts(8, 30), True), (ts(9, 30), False)):
            b, conn, _ = make_budget(at)
            claude_row(conn, at - 60, 0.02, five=0.60, five_reset=at + 3600)
            if ok:
                b.claude_check("judge", 0.3)
            else:
                with self.assertRaises(ClaudeCapReached):
                    b.claude_check("judge", 0.3)

    def test_a_reset_five_hour_window_reads_zero(self):
        b, conn, _ = make_budget(ts(10))
        claude_row(conn, ts(4), 0.02, five=0.95, five_reset=ts(7))
        self.assertEqual(b.five_hour_util()[0], 0.0)
        b.claude_check("judge", 0.3)

    def test_backstop_until_the_first_run_day_finishes(self):
        b, conn, _ = make_budget(ts(3))
        claude_row(conn, ts(2, 30), 19.9, role=BULK, week=0.30, five=0.10, five_reset=ts(7))
        CONFIG["claude"]["week_points_per_usd"] = 0.001               # keep the allowance out of the way
        try:
            with self.assertRaises(ClaudeCapReached) as cm:
                b.claude_check("judge", 0.3)
            self.assertEqual(cm.exception.check, "Claude first-run backstop")
            b.day_started("2026-11-08")
            b.day_finished("2026-11-08")
            b.claude_check("judge", 0.3)
        finally:
            CONFIG["claude"]["week_points_per_usd"] = 0.01

    def test_expected_cost_is_the_costly_end_of_recent_calls(self):
        b, conn, _ = make_budget(ts(3))
        self.assertEqual(b.expected_cost("judge", 0.3), 0.3)          # no history: the call's own ceiling
        for c in (0.02, 0.05, 0.03):
            claude_row(conn, ts(2, 30), c)
        claude_row(conn, ts(2, 31), 0.09, role="loop s1w0 h2h")
        self.assertEqual(b.expected_cost("judge", 0.3), 0.05)

    def test_week_reset_starts_the_share_again(self):
        b, conn, _ = make_budget(ts(15))
        claude_row(conn, ts(12, days=-2), 40.0, role=BULK, week=0.80, week_reset=ts(14))   # the week reset an hour ago
        self.assertEqual(b.loop_share(), 0.0)
        self.assertEqual(b.real_total(), 0.0)

    def test_an_overshoot_comes_off_next_weeks_share(self):
        b, conn, _ = make_budget(ts(3))
        claude_row(conn, ts(12, days=-9), 60.0, role=BULK, week=0.70)  # last week: 0.60 against a 0.50 share
        self.assertAlmostEqual(b.share_cap(), 0.40)


class SchedulePhases(unittest.TestCase):
    def setUp(self):
        self.s = Schedule(CONFIG)

    def test_phases(self):
        for (h, m), want in {(1, 59): "before", (2, 0): "night", (8, 59): "night", (9, 0): "day", (15, 29): "day",
                             (15, 30): "wind-down", (15, 59): "wind-down", (16, 0): "after"}.items():
            self.assertEqual(self.s.phase(ts(h, m)), want, f"{h}:{m:02d}")

    def test_five_hour_cap_by_phase(self):
        self.assertEqual(self.s.five_hour_cap(ts(3)), 0.90)
        self.assertEqual(self.s.five_hour_cap(ts(9)), 0.50)
        self.assertEqual(self.s.five_hour_cap(ts(15, 45)), 0.50)

    def test_attempts_only_between_start_and_last_attempt(self):
        self.assertFalse(self.s.may_start_attempt(ts(1, 59)))
        self.assertTrue(self.s.may_start_attempt(ts(2)))
        self.assertTrue(self.s.may_start_attempt(ts(15, 29)))
        self.assertFalse(self.s.may_start_attempt(ts(15, 30)))

    def test_run_days_left_and_next_start(self):
        self.assertEqual(self.s.run_days_between(ts(3), ts(14, days=2)), 3)
        self.assertEqual(self.s.run_days_between(ts(17), ts(14, days=2)), 2)   # today's run is over
        self.assertEqual(self.s.run_days_between(ts(3), ts(1, days=1)), 1)
        self.assertEqual(self.s.next_start(ts(16)), ts(2, days=1))
        self.assertEqual(self.s.next_start(ts(1)), ts(2))


class StuckRule(unittest.TestCase):
    def test_rounds_one_to_three_always_run(self):
        self.assertFalse(stuck_rule([False, False, False], 3, 2))
        self.assertFalse(stuck_rule([False, False, False, False], 3, 2))

    def test_two_quiet_rounds_after_round_three(self):
        self.assertTrue(stuck_rule([True, True, True, False, False], 3, 2))
        self.assertTrue(stuck_rule([False] * 5, 3, 2))

    def test_a_change_resets_it(self):
        self.assertFalse(stuck_rule([True, True, True, False, True], 3, 2))
        self.assertFalse(stuck_rule([True, False, False, False, True, False], 3, 2))
        self.assertTrue(stuck_rule([True, False, False, False, True, False, False], 3, 2))


class DropRule(unittest.TestCase):
    def setUp(self):
        self.runs = [[{"sents": ["A joke about soup."], "claude": True}, {"sents": ["Yours here."], "claude": False}],
                     [{"sents": ["In 2023 a study found five things."], "claude": True},
                      {"sents": ["Yours too."], "claude": False}],
                     [{"sents": ["The only run."], "claude": True}],
                     [{"sents": ["Another flourish."], "claude": True}, {"sents": ["Mine."], "claude": False}]]
        self.orig = [[r["sents"][:] for r in p] for p in self.runs]

    def go(self, current=None, attempts=None, rounds=6, verdict="claude"):
        cur = current or [[r[:] for r in p] for p in self.orig]
        att = attempts if attempts is not None else {(pi, ri): 6 for pi, p in enumerate(self.runs) for ri in range(len(p))}
        return drop_eligible(self.runs, cur, self.orig, att, rounds, lambda s: verdict, lambda t: set())

    def test_conditions(self):
        cut, keep = self.go()
        self.assertEqual([(p, r) for p, r, _ in cut], [(0, 0), (3, 0)])
        self.assertIn((1, 0, "carries a fact"), keep)
        self.assertIn((2, 0, "it's the whole paragraph"), keep)

    def test_a_replaced_run_is_not_cut(self):
        cur = [[r[:] for r in p] for p in self.orig]
        cur[0][0] = ["A regrown line."]
        cut, _ = self.go(current=cur)
        self.assertNotIn(0, [p for p, _, _ in cut])

    def test_a_run_not_tried_every_round_is_not_cut(self):
        att = {(0, 0): 3, (3, 0): 6}
        cut, _ = self.go(attempts=att)
        self.assertEqual([(p, r) for p, r, _ in cut], [(3, 0)])

    def test_only_claude_written_runs(self):
        cut, keep = self.go(verdict="both")
        self.assertEqual(cut, [])

    def test_carries_fact(self):
        self.assertTrue(carries_fact('a "quote" here'))
        self.assertTrue(carries_fact("Okafor et al. (2024) found it"))
        self.assertTrue(carries_fact("this is called proofing the dough"))
        self.assertTrue(carries_fact("Somebody hit it", names_in_run={"Claude"}))
        self.assertFalse(carries_fact("Somebody ring the damn bell already."))


class FixList(unittest.TestCase):
    COFFEE = "A pro might coast through that corner once in a race, and a beginner will probably try it every lap."
    AGAIN = "The chain slipped again, and again, and again."
    SEE = ('Coaching notes lean on "brake X, lean Y" and "eyes on X, not Y" for every bend, which gets old by the'
           " third lesson.")

    def test_hedges_found_in_the_run(self):
        self.assertEqual(sorted(hedges_in(self.COFFEE)), ["might", "once", "probably"])
        self.assertEqual(hedges_in("(about 3/4 of a cup)"), ["about"])
        self.assertEqual(hedges_in("the new tyres last up to 30% longer"), ["up to"])

    def test_words_that_are_only_sometimes_hedges(self):
        self.assertNotIn("once", hedges_in("once you've ridden it, the whole route"))
        self.assertNotIn("sort of", hedges_in("the smell of the kitchen, that sort of thing"))
        self.assertNotIn("about", hedges_in("what it's about"))
        self.assertEqual(hedges_in('a "probably" inside quotes'), [])

    def test_repeats(self):
        self.assertEqual(repeats_in(self.AGAIN), [("again", 3)])
        self.assertEqual(repeats_in("The chain slipped again."), [])
        self.assertTrue(repeat_ok("and again and again and again", [("again", 3)]))
        self.assertFalse(repeat_ok("The chain slipped again.", [("again", 3)]))

    def test_keep_line(self):
        notes = ["the model repeats itself", 'keep: "again, and again, and again", "probably"']
        self.assertEqual(keep_phrases(notes), ["again, and again, and again", "probably"])
        mk = must_keeps(self.AGAIN, notes)
        self.assertEqual(mk["repeats"], [("again", 3)])
        self.assertEqual(mk["hedges"], [])                 # "probably" isn't in this run, so it isn't its must-keep
        self.assertIn("I reckon", must_keeps("I reckon it's fine.", ['keep: "I reckon"'])["hedges"])
        self.assertEqual(must_keeps("it's basically a chunk", ['keep: "basically"'])["hedges"], [])
        self.assertIn("once", must_keeps("once you've seen it, it sticks", ['keep: "once"'])["hedges"])

    def test_invented_sightings(self):
        self.assertEqual(invented_sighting("I've seen a beginner try it every lap.", self.COFFEE), "I've seen")
        self.assertTrue(invented_sighting("and that beginner just showed me another wobble", self.COFFEE))
        self.assertIsNone(invented_sighting(self.COFFEE, self.COFFEE))
        self.assertIsNone(invented_sighting("I saw it, I said.", "I saw it, I said."))
        checks = extra_checks_for(self.AGAIN, [("again", 3)])
        reasons = lambda c: [r for r in (f(c) for f in checks) if r]  # noqa: E731
        self.assertEqual(reasons("The chain slipped again."), ["must-keep repeat"])
        self.assertEqual(reasons("I tried it again and again and again."), ["invented sighting"])

    def test_notation_lock_is_narrowed_to_templates(self):
        self.assertTrue(is_template("X and X and X."))
        self.assertTrue(is_template('"Stir X. Stir Y. Stir Z." is the usual one.'))
        self.assertFalse(is_template(self.SEE))             # prose that quotes patterns regrows now
        self.assertEqual(pattern_quotes(self.SEE), ['"brake X, lean Y"', '"eyes on X, not Y"'])
        self.assertEqual(pattern_quotes('literally "taste as you go"'), [])

    def test_quotes_spliced_back(self):
        q = ['"brake X, lean Y"']
        self.assertEqual(splice_quotes('You see "Brake X, lean Y." a lot', q),
                         'You see "brake X, lean Y" a lot')
        self.assertEqual(splice_quotes('You see "brake X" a lot', q), 'You see "brake X" a lot')

    def test_flag_list_joins_neighbors(self):
        segs = [["Okafor et al.", True], [" (2024) saw it too.", True], ["Fine.", False]]
        fl = flag_list(segs)
        self.assertIn("Okafor et al. (2024) saw it too.", fl)
        self.assertTrue(any("Okafor et al. (2024) saw it to"[:40] in f for f in fl))


class BankResume(unittest.TestCase):
    def test_open_pass_best_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bank.jsonl"
            b = Bank(p)
            runs = [[["a."], ["b."]]]
            b.add({"kind": "start", "window": "s1w0", "ai": 1.0, "segs": [], "text": "a. b."})
            b.add({"kind": "round", "window": "s1w0", "pass": 1, "round": 1, "g": 1, "ai": 0.6, "text": "t1", "runs": runs})
            b.add({"kind": "round", "window": "s1w0", "pass": 1, "round": 2, "g": 2, "ai": None, "text": "t2", "runs": runs})
            b.add({"kind": "round", "window": "s3w0", "pass": 1, "round": 1, "g": 1, "ai": 0.1, "text": "u", "runs": runs})
            b2 = Bank(p)                                     # a restart reads the file back
            self.assertEqual(len(b2.entries), 4)
            op = b2.open_pass("s1w0")
            self.assertEqual(op[0], 1)
            self.assertEqual([e["round"] for e in op[1]], [1, 2])
            self.assertEqual(b2.best("s1w0")["text"], "t1")
            self.assertEqual([e["text"] for e in b2.unscored("s1w0")], ["t2"])
            self.assertEqual(b2.g("s1w0"), 2)
            b2.add({"kind": "pass_end", "window": "s1w0", "pass": 1, "state": "stuck"})
            self.assertIsNone(Bank(p).open_pass("s1w0"))
            b2.add({"kind": "round", "window": "s1w0", "pass": 2, "round": 1, "g": 3, "ai": 0.6, "text": "t3",
                    "runs": runs, "humanized": True})
            self.assertEqual(Bank(p).best("s1w0")["text"], "t1")   # a tie goes to the latest, a humanized never wins

    def test_align_keeps_your_lines(self):
        runs = [[{"sents": ["Yours one.", "Yours two."], "claude": False}, {"sents": ["Claude says it thrice."], "claude": True},
                 {"sents": ["Yours three."], "claude": False}]]
        same, exact = align(runs, [[["Yours one.", "Yours two."], ["Mistral says it."], ["Yours three."]]])
        self.assertTrue(exact)
        self.assertEqual(same[0][1], ["Mistral says it."])
        # the experiment regrew one of your lines: it comes back as written, and the text needs a fresh score
        fixed, exact = align(runs, [[["A regrown line."], ["Yours two."], ["Mistral says it."], ["Yours three."]]])
        self.assertFalse(exact)
        self.assertEqual(fixed[0], [["Yours one.", "Yours two."], ["Mistral says it."], ["Yours three."]])

    def test_verdict_map_combines_pieces(self):
        vm = VerdictMap({"Back again are our friends Okafor et al.": "both",
                         "(2025), whose cooks tested a thousand recipes.": "claude"})
        self.assertEqual(vm.get("Back again are our friends Okafor et al. (2025), whose cooks tested"
                                " a thousand recipes."), "claude")
        self.assertIsNone(vm.get("Why?"))


class SessionLimit(unittest.TestCase):
    def test_reset_time_from_the_message_else_the_reading(self):
        from engine.run import Loop

        class FakeBudget:
            def five_hour_util(self, now):
                return 0.5, now + 7200

        loop = type("L", (), {"budget": FakeBudget()})()
        now = ts(3, 10)
        self.assertEqual(Loop.session_reset(loop, "You've hit your session limit · resets 5am", now), ts(5))
        self.assertEqual(Loop.session_reset(loop, "You've hit your session limit", now), now + 7200)
        self.assertEqual(Loop.session_reset(loop, "limit reached, resets 9pm", now), now + 7200)   # too far off to trust


# ---------------------------------------------------------------------- the M1 review's fixes (2026-11-08)
def loop_stub(**over):
    """A Loop with only what the two paid doors and the retry path need: no models, no network."""
    import types
    from engine.run import Loop
    lp = Loop.__new__(Loop)
    clock = CompressedClock(ts(3), speed=0.0)
    lp.clock, lp.schedule, lp.timeline, lp.alerted = clock, Schedule(CONFIG), [], set()
    lp.args = types.SimpleNamespace(smoke=False, now=False)
    lp.retry_tries, lp.retry_waits, lp.error_block = 3, [30.0, 90.0], 600.0
    lp.pangram_block = lp.claude_block = None
    lp.purpose, lp.errors_in_row, lp.reserve = "loop", 0, 0.0
    lp.notifier = types.SimpleNamespace(send=lambda *a, **k: None)
    lp.settings = Settings(env={}, config=CONFIG)
    for k, v in over.items():
        setattr(lp, k, v)
    return lp


class Finding1StartTime(unittest.TestCase):
    def test_a_start_outside_the_window_waits_for_the_next_two_oclock(self):
        from engine.run import start_time
        s = Schedule(CONFIG)
        self.assertEqual(start_time(s, ts(17)), ts(2, days=1))          # an evening start waits for tonight
        self.assertEqual(start_time(s, ts(23, 59)), ts(2, days=1))
        self.assertEqual(start_time(s, ts(15, 45)), ts(2, days=1))      # the wind-down too
        self.assertEqual(start_time(s, ts(1)), ts(2))                    # after midnight: today's 2:00
        self.assertEqual(start_time(s, ts(3)), ts(3))                    # inside the window: straight away
        self.assertEqual(start_time(s, ts(10)), ts(10))
        self.assertEqual(start_time(s, ts(17), ignore_schedule=True), ts(17))


class Finding2Retries(unittest.TestCase):
    def test_pangram_errors_retry_then_hold_the_door(self):
        import types
        from engine.pangram import PangramError
        tries = []

        def fail(text, purpose, label):
            tries.append(1)
            raise PangramError("create task failed: 502")
        pg = types.SimpleNamespace(cached=lambda t: None, estimate=lambda t: (3, 0.15), score=fail)
        lp = loop_stub(pg=pg, budget=types.SimpleNamespace(pangram_check=lambda usd, purpose, reserve=0.0: None))
        t0 = lp.clock.now()
        with self.assertRaises(PangramCapReached) as cm:
            lp.score("some text", "s1w0 g1")
        self.assertEqual(len(tries), 3)
        self.assertEqual(cm.exception.check, "Pangram errors")
        self.assertAlmostEqual(lp.clock.now() - t0, 120)                 # 30 s, then 90 s, between the tries
        self.assertAlmostEqual(cm.exception.until, lp.clock.now() + 600)

    def test_a_pangram_error_that_clears_on_a_retry_just_carries_on(self):
        import types
        from engine.pangram import PangramError
        calls = []

        def flaky(text, purpose, label):
            calls.append(1)
            if len(calls) == 1:
                raise PangramError("timed out")
            return {"fraction_ai": 0.1}, False
        pg = types.SimpleNamespace(cached=lambda t: None, estimate=lambda t: (3, 0.15), score=flaky)
        lp = loop_stub(pg=pg, budget=types.SimpleNamespace(pangram_check=lambda usd, purpose, reserve=0.0: None))
        self.assertEqual(lp.score("text", "x"), {"fraction_ai": 0.1})

    def test_claude_errors_retry_then_hold_and_a_session_limit_holds_at_once(self):
        import types
        from engine.claude_call import ClaudeCallError
        calls = []

        def broken(settings, conn, prompt, role, model, max_budget_usd, **kw):
            calls.append(role)
            raise ClaudeCallError("Internal server error")
        budget = types.SimpleNamespace(claude_check=lambda kind, usd: usd, five_hour_util=lambda now: (0.2, None))
        lp = loop_stub(claude_raw=broken, budget=budget)
        with self.assertRaises(ClaudeCapReached) as cm:
            lp.claude(None, None, "prompt", role="s1w0 judge")
        self.assertEqual((len(calls), cm.exception.check), (3, "Claude errors"))
        calls.clear()

        def limited(*a, **k):
            calls.append(1)
            raise ClaudeCallError("You've hit your session limit · resets 5am")
        lp = loop_stub(claude_raw=limited, budget=budget)
        with self.assertRaises(ClaudeCapReached) as cm:
            lp.claude(None, None, "prompt", role="s1w0 judge")
        self.assertEqual((len(calls), cm.exception.check), (1, "Claude session limit"))

    def test_a_timed_out_claude_call_is_counted_at_its_ceiling(self):
        import subprocess
        from unittest import mock
        from engine import claude_call
        conn = sqlite3.connect(":memory:")
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(claude_call.subprocess, "run",
                                  side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=600)):
            s = Settings(env={"WORK_DIR": d, "CLAUDE_CLI": "claude"}, config=CONFIG)
            with self.assertRaises(claude_call.ClaudeCallError):
                claude_call.run(s, conn, "p", role="loop s1w0 judge", max_budget_usd=0.3)
        row = conn.execute("SELECT role, ok, cost_usd FROM claude_calls").fetchone()
        self.assertEqual(row, ("loop s1w0 judge", 0, 0.3))

    def test_the_day_restarts_and_five_failures_in_a_row_end_it(self):
        import types
        from engine.run import run_with_restarts
        clock = CompressedClock(ts(3), speed=0.0)
        events = []

        class FakeLoop:
            settings = Settings(env={}, config=dict(CONFIG, loop={"failures_in_row": 5}))
            rounds_done, error_block, stop_reason = 0, 600.0, None
            schedule = Schedule(CONFIG)

            def __init__(self, fail_times, progress=False):
                self.fail_times, self.progress, self.calls = fail_times, progress, 0

            def run_day(self):
                self.calls += 1
                if self.calls <= self.fail_times:
                    if self.progress:
                        self.rounds_done += 1          # a round finished before this failure
                    raise RuntimeError("llama-server went away")
                self.stop_reason = "wind-down"

            def stop_servers(self):
                pass

            def log(self, m):
                events.append(m)

            def alert(self, key, text):
                events.append("ALERT " + key)

            def must_stop(self):
                return False

            def tick(self):
                pass
        import contextlib
        import io
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        job = types.SimpleNamespace(setup=lambda: None)
        a = types.SimpleNamespace(now=True)
        lp = FakeLoop(2)
        self.assertIsNone(run_with_restarts(lp, job, lp.schedule, clock, Path("no STOP here"), a))
        self.assertEqual(lp.calls, 3)                                     # two errors, then a clean day
        lp = FakeLoop(99)
        self.assertIn("llama-server", run_with_restarts(lp, job, lp.schedule, clock, Path("no STOP here"), a))
        self.assertEqual(lp.calls, 5)
        self.assertTrue(any(e == "ALERT crash" for e in events))
        lp = FakeLoop(8, progress=True)                                   # rounds between failures: never "in a row"
        self.assertIsNone(run_with_restarts(lp, job, lp.schedule, clock, Path("no STOP here"), a))
        self.assertEqual(lp.calls, 9)

    def test_notes_that_dont_parse_raise_notes_error(self):
        from engine.job import Job, NotesError
        with tempfile.TemporaryDirectory() as d:
            Path(d, "job.toml").write_text('[job]\nname = "t"\noriginal = "o.md"\nworking = "w.md"\nsections = [1]\n'
                                           'voice_samples = [9]\n', encoding="utf-8")
            job = Job("t", root=d)
            ok = job.notes_for(lambda *a, **k: ("### 1\n- a point\n- keep: \"probably\"", {}), None, None,
                               ["Para one."], role="t notes")
            self.assertEqual(ok, [["a point", 'keep: "probably"']])
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["job.toml", "notes cache.json"])  # no .tmp
            with self.assertRaises(NotesError):
                job.notes_for(lambda *a, **k: ("NONE", {}), None, None, ["Para two."], role="t notes")


class Finding3ClockOnlyInRehearsal(unittest.TestCase):
    def test_clock_and_speed_need_rehearse(self):
        import contextlib
        import io
        from engine.run import main
        for argv in (["--clock", "02:00"], ["--speed", "60"], ["--clock", "2026-11-09 01:55", "--now"]):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                main(argv)
            self.assertEqual(cm.exception.code, 2, argv)


class MainWiring(unittest.TestCase):
    """main() past argparse, with every outside part faked and a STOP file waiting, so it builds the loop and returns
    before any work: the job comes from --job or config.toml's [job] default, and --smoke with no --window takes the
    job's first window."""

    def main(self, argv, config):
        import types
        from unittest import mock
        import engine.run as run
        made = {}
        with tempfile.TemporaryDirectory() as d:
            stop = Path(d) / "STOP"
            stop.write_text("", encoding="utf-8")
            settings = types.SimpleNamespace(config=config, runs_dir=Path(d), db_path=Path(d) / "x.sqlite",
                                             stop_file=stop, get=lambda key, default="": default)
            job = types.SimpleNamespace(name="x", windows=lambda: [{"wid": "s2w0"}, {"wid": "s3w0"}])

            def make_job(name):
                made["name"] = name
                return job

            def loop(**kw):
                made.update(kw)
                return types.SimpleNamespace()
            with mock.patch.multiple(run, Settings=lambda: settings, Job=make_job, LoopPangram=mock.DEFAULT,
                                     real_parts=lambda s: (None, None, None), Notifier=mock.DEFAULT,
                                     Schedule=mock.DEFAULT, Budget=mock.DEFAULT, Loop=loop), \
                    mock.patch.object(run.db, "connect"), mock.patch("builtins.print"):
                code = run.main(argv)
        return code, made

    def test_the_job_comes_from_the_flag_or_config(self):
        code, made = self.main(["--job", "x", "--now"], {})
        self.assertEqual((code, made["name"], str(made["stop_file"]).endswith("STOP")), (0, "x", True))  # pass
        code, made = self.main(["--now"], {"job": {"default": "essay"}})
        self.assertEqual(made["name"], "essay")                                                          # pass
        from engine.job import JobError
        with self.assertRaises(JobError):
            self.main(["--now"], {})                                                                    # flag

    def test_smoke_takes_the_jobs_first_window(self):
        _, made = self.main(["--job", "x", "--smoke"], {})
        self.assertEqual(made["args"].window, "s2w0")                                                   # flag
        _, made = self.main(["--job", "x", "--smoke", "--window", "s3w0"], {})
        self.assertEqual(made["args"].window, "s3w0")                                                   # pass


class Finding4OwnServer(unittest.TestCase):
    def test_a_busy_port_is_refused_before_anything_starts(self):
        import types
        from unittest import mock
        from engine import local_models
        srv = local_models.LlamaServer("mistral-small-24b-base", settings=Settings(env={}, config=CONFIG))
        props = types.SimpleNamespace(status_code=200, json=lambda: {"model_path": r"C:\m\Mistral-Instruct.gguf"})
        with mock.patch.object(local_models.requests, "get", return_value=props), \
                mock.patch.object(local_models.subprocess, "Popen", side_effect=AssertionError("must not launch")):
            with self.assertRaises(RuntimeError) as cm:
                srv.start()
        self.assertIn("Mistral-Instruct.gguf", str(cm.exception))

    def test_the_props_check_matches_the_file_name(self):
        import types
        from unittest import mock
        from engine import local_models
        srv = local_models.LlamaServer("mistral-small-24b-base", settings=Settings(env={}, config=CONFIG))
        ours = types.SimpleNamespace(status_code=200, json=lambda: {"model_path": srv.model_path.upper()})
        other = types.SimpleNamespace(status_code=200, json=lambda: {"model_path": r"D:\x\other.gguf"})
        with mock.patch.object(local_models.requests, "get", return_value=ours):
            self.assertTrue(srv.serves_our_model())
        with mock.patch.object(local_models.requests, "get", return_value=other):
            self.assertFalse(srv.serves_our_model())


class Finding5PangramSpend(unittest.TestCase):
    def test_failed_and_timed_out_rows_with_a_cost_count(self):
        b, conn, _ = make_budget(ts(3))
        pangram_row(conn, ts(2, 10), 5.80)
        conn.execute("INSERT INTO pangram_calls (ts, sha, model, words, credits, usd, purpose, label, status)"
                     " VALUES (?,?,?,?,?,?,?,?,?)", (ts(2, 20), "y", "pangram-4.1", 300, 3, 0.15, "loop", "", "timeout"))
        conn.execute("INSERT INTO pangram_calls (ts, sha, model, words, credits, usd, purpose, label, status)"
                     " VALUES (?,?,?,?,?,?,?,?,?)", (ts(2, 30), "z", "pangram-4.1", 300, 0, 0.0, "loop", "", "http_502"))
        conn.commit()
        self.assertAlmostEqual(b.pangram_today(), 5.95)
        with self.assertRaises(PangramCapReached):
            b.pangram_check(0.10)
        from engine.pangram import Pangram
        self.assertAlmostEqual(Pangram(Settings(env={}, config=CONFIG), conn).spent_usd(ts(0), "loop"), 5.95)


class Finding6RateClamp(unittest.TestCase):
    def test_a_refit_stays_between_point_five_and_two_times_the_seed(self):
        for stored, want in ((0.0013, 0.005), (0.05, 0.02), (0.012, 0.012)):
            b, conn, _ = make_budget(ts(3))
            conn.execute("INSERT INTO claude_rates (ts, day, kind, points, dollars, rate) VALUES (?,?,?,?,?,?)",
                         (ts(1), "2026-11-09", "week", 0.02, 15.0, stored))
            conn.commit()
            self.assertAlmostEqual(b.rate("week"), want)                 # the seed is 0.01 here


class Finding7NowStops(unittest.TestCase):
    def test_now_runs_stop_at_the_next_stop_time(self):
        from engine.run import now_stop_time
        s = Schedule(CONFIG)
        self.assertEqual(now_stop_time(s, ts(10)), ts(16))
        self.assertEqual(now_stop_time(s, ts(17)), ts(16, days=1))
        lp = loop_stub()
        lp.args.now, lp.stop_reason, lp.stop_file, lp.now_stop = True, None, None, ts(16)
        self.assertFalse(lp.must_stop())                                  # 03:00, a --now run, before its stop
        lp.clock.advance(13 * 3600)
        self.assertTrue(lp.must_stop())
        self.assertIn("--now", lp.stop_reason)


class Finding8MustKeepDemotion(unittest.TestCase):
    def test_a_replacement_that_breaks_a_must_keep_fails_and_never_ships(self):
        from engine.run import failing_runs
        original = [[["The chain slipped again, and again, and again."], ["Mine."]]]
        bad = [[["The chain slipped again."], ["Mine."]]]
        good = [[["And then again and again and again."], ["Mine."]]]
        must = {(0, 0): {"hedges": [], "repeats": [("again", 3)]}}
        self.assertEqual(failing_runs(must, bad, original), [(0, 0)])
        self.assertEqual(failing_runs(must, good, original), [])
        self.assertEqual(failing_runs(must, original, original), [])     # Claude's own text is never "failing"
        with tempfile.TemporaryDirectory() as d:
            bank = Bank(Path(d) / "bank.jsonl")
            bank.add({"kind": "round", "window": "w", "ai": 0.0, "text": "bad", "runs": bad})
            bank.add({"kind": "round", "window": "w", "ai": 0.3, "text": "good", "runs": good})
            ok = lambda e: not failing_runs(must, e["runs"], original)  # noqa: E731
            self.assertEqual(bank.best("w")["text"], "bad")
            self.assertEqual(bank.best("w", ok=ok)["text"], "good")


class Finding12NoWebhookInPrints(unittest.TestCase):
    def test_a_failed_post_prints_the_class_only(self):
        import contextlib
        import io
        from unittest import mock
        from engine import discord
        from engine.run import Notifier, redact
        url = "https://discord.com/api/webhooks/123/SECRETTOKEN"
        n = Notifier(Settings(env={}, config=CONFIG), CompressedClock(ts(3)), Path(tempfile.mkdtemp()) / "d.log")
        out = io.StringIO()
        with mock.patch.object(discord, "post", side_effect=ConnectionError(f"Max retries exceeded with url: {url}")), \
                contextlib.redirect_stdout(out):
            n.send("hello")
        self.assertNotIn("SECRETTOKEN", out.getvalue())
        self.assertIn("ConnectionError", out.getvalue())
        self.assertNotIn("SECRETTOKEN", redact(f"error at {url} with sk-ant-oat01-abc"))


if __name__ == "__main__":
    unittest.main()
