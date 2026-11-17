"""Tests for the analytics basics (plan 10.1, 10.2), engine/analytics.py, and the digest's efficiency table. Every
test builds its own ledger in memory and its own bank; nothing reads the real ones. Run from
the repo root:

    python -m unittest discover tests
"""
import datetime as dt
import json
import sqlite3
import tempfile
import types
import unittest
from pathlib import Path

from engine import analytics as an
from engine import db
from engine.budget import RUN_DAYS
from engine.claude_call import ensure_schema

CFG = {"loop": {"stuck_after_round": 3, "stuck_streak": 2, "whole_text_reserve_usd": 3.0},
       "pangram": {"nightly_cap_usd": 6.0, "monthly_cap_usd": 180.0},
       "claude": {"week_points_per_usd": 0.007, "five_hour_points_per_usd": 0.05, "rate_clamp": [0.5, 2.0],
                  "first_night_cap_usd": 20.0, "backstop_run_days": 1}}


def ts(day, h, m=0):
    return dt.datetime(2026, 11, day, h, m).timestamp()


def st(day, h, m=0):
    return dt.datetime(2026, 11, day, h, m).strftime("%Y-%m-%d %H:%M")


def ledger():
    conn = sqlite3.connect(":memory:")
    conn.executescript(db.SCHEMA)
    ensure_schema(conn)
    conn.executescript(RUN_DAYS)
    conn.execute("INSERT INTO run_days VALUES ('2026-11-08', ?, ?, 1)", (ts(8, 18, 41) + 22, ts(9, 0, 48)))
    return conn


def pg(conn, when, label, usd, purpose="loop"):
    conn.execute("INSERT INTO pangram_calls (ts, sha, model, words, credits, usd, purpose, label, status)"
                 " VALUES (?,?,?,?,?,?,?,?,?)", (when, "x", "pangram-4.1", 100, 1, usd, purpose, label, "ok"))


def cl(conn, when, role, cost):
    conn.execute("INSERT INTO claude_calls (ts, role, model, ok, cost_usd) VALUES (?,?,?,?,?)",
                 (when, role, "sonnet", 1, cost))


def rnd(day, h, m, ai, changes=(), p=1, changed=True, settle=False, text="t", runs=None, **kw):
    return dict({"kind": "round", "window": "s1w0", "day": "2026-11-08", "time": st(day, h, m), "pass": p,
                 "ai": ai, "changes": list(changes), "changed": changed, "settle_next": settle, "text": text,
                 "runs": runs if runs is not None else [[[text]]], "segs": []}, **kw)


ACCEPTED = {"para": 0, "run": 0, "orig": "Claude's.", "new": "Mistral's.", "who": "claude",
            "result": "replaced (5 passed filters, head-to-head: B)"}
NOTHING = {"para": 0, "run": 1, "orig": "Again.", "result": "no candidate passed (44 drawn: length 43, must-keep repeat 1)"}
KEPT = {"para": 0, "run": 1, "orig": "Again.", "result": "judge kept the original (7 passed filters)"}


class Reading(unittest.TestCase):
    def test_a_change_counts_its_draws_exactly_when_the_bank_has_them(self):
        self.assertEqual(an.change_stats(NOTHING), (44, True, False, {"length": 43, "must-keep repeat": 1}))
        self.assertEqual(an.change_stats(KEPT)[:2], (an.N_CANDS, False))           # one draw, estimated
        self.assertEqual(an.change_stats({"result": "queued for when the cap lifts"})[:2], (0, True))
        self.assertEqual(an.change_stats(dict(ACCEPTED, draws=192, tally="protected word 40"))[:4],
                         (192, True, True, {"protected word": 40}))              # banked from 2026-11-09 on
        self.assertFalse(an.change_stats(dict(ACCEPTED, result="pick lost the head-to-head (2 passed filters)",
                                              new=None))[2])

    def test_cost_per_point_is_none_without_a_gain(self):
        self.assertEqual(an.cost_per_point(0.60, 0.17, 0.00), 0.0353)
        self.assertIsNone(an.cost_per_point(0.95, 0.00, 0.00))
        self.assertIsNone(an.cost_per_point(0.95, 0.00, 0.27))                    # worse: no gain, never negative

    def test_a_minute_stamp_counts_from_the_minute_the_run_day_began(self):
        start, end = ts(8, 18, 41) + 22, ts(9, 0, 48) + 18
        self.assertTrue(an.in_span(an.parse_stamp(st(8, 18, 41)), start, end))     # 18:41 when the day began 18:41:22
        self.assertFalse(an.in_span(an.parse_stamp(st(8, 14, 59)), start, end))    # the smoke test that afternoon
        self.assertTrue(an.in_span(an.parse_stamp(st(9, 0, 31)), start, end))      # after midnight, same run-day
        self.assertFalse(an.in_span(an.parse_stamp(st(9, 2, 10)), start, end))     # the next morning's smoke test

    def test_whole_text_scorings_belong_to_the_section_not_a_window(self):
        conn = ledger()
        pg(conn, ts(8, 19), "run s1w0 g3", 0.20)
        pg(conn, ts(8, 19, 5), "run s1w0 cut", 0.20)
        pg(conn, ts(8, 20), "run whole s1 pass 1", 0.55)
        pg(conn, ts(8, 14, 58), "run s1w0 rescore", 0.15, purpose="experiment")    # the smoke test
        self.assertEqual(an.pangram_by_window(conn, ts(8, 18), ts(9, 1)), {"s1w0": 0.4})
        self.assertEqual([an.pangram_kind(x) for x in ("run s1w0 g3", "run s1w1 round 0", "run s1w0 rescore",
                                                       "run s1w0 cut", "run whole s3 pass 2", "run as posted s1")],
                         ["window rounds", "round 0", "rescores", "drop-rule cuts", "whole-text round", "final scan"])


class Shipped(unittest.TestCase):
    def test_a_whole_text_pick_ships_and_its_window_score_is_used_when_banked(self):
        e = [rnd(8, 19, 0, 0.40, text="a"), rnd(8, 19, 5, 0.10, text="b"),
             {"kind": "ship", "window": "s1w0", "time": st(8, 20), "text": "c", "runs": [[["c"]]], "ai": None}]
        self.assertEqual(an.shipped(e, None, "s1w0"), (0.10, "whole-text pick (unscored alone; best shown)"))
        e.append(rnd(8, 21, 0, 0.30, text="c"))                                    # a later round scored the pick
        # that later round is a scored entry of the pick's text, but it's not a new best, so the pick still ships
        self.assertEqual(an.shipped(e, None, "s1w0"), (0.30, "whole-text pick"))
        e.append(rnd(8, 22, 0, 0.05, text="d"))                                    # a new best wins again
        self.assertEqual(an.shipped(e, None, "s1w0"), (0.05, "best"))

    def test_humanized_texts_never_count_as_best(self):
        e = [rnd(8, 19, 0, 0.40), rnd(8, 19, 5, 0.00, humanized=True)]
        self.assertEqual(an.best_entry(e, "s1w0")[1]["ai"], 0.40)


class Digests(unittest.TestCase):
    DIGEST = "\n".join([
        "# depangram digest, 2026-11-08", "", "## Windows", "",
        "| Window | Start | Now | Best | Flagged now | State | Passes today | Rounds today |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        "| s1w0 | 0.05 | 0.04 | 0.00 | 1 | stuck | 2 | 17 |", "",
        "## Cap checks that said no", "", "| Time | Check | Detail | Holds until |", "| --- | --- | --- | --- |",
        "| 2026-11-08 19:20 | Claude 5-hour (day) | the 5-hour window is at 50% | 2026-11-08 23:00 |",
        "| 2026-11-09 00:30 | Pangram daily | $6.00 spent | 2026-11-10 00:00 |", "",
        "## Timeline", "", "```", "2026-11-08 21:10 Claude 5-hour (day) has lifted", "```"])

    def test_the_window_table_and_the_holds(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "2026-11-08 digest.md"
            p.write_text(self.DIGEST, encoding="utf-8")
            (Path(d) / "2026-11-09 digest (smoke).md").write_text(self.DIGEST, encoding="utf-8")
            got = an.digests_by_day(d)
        self.assertEqual(list(got), ["2026-11-08"])                                 # smoke digests stay out
        self.assertEqual(got["2026-11-08"]["windows"]["s1w0"]["start"], 0.05)
        h = an.holds(got["2026-11-08"], ts(9, 0, 48))
        self.assertEqual([(x["check"], x["minutes"]) for x in h],
                         [("Claude 5-hour (day)", 110), ("Pangram daily", 18)])      # lifted at 21:10; the day's end


class StuckRule(unittest.TestCase):
    def test_the_rule_round_by_round(self):
        self.assertEqual(an.stuck_fires([True, False, False, False, False, True], 3, 2), 5)
        self.assertEqual(an.stuck_fires([True, False, False, False, False, True], 1, 2), 3)
        self.assertIsNone(an.stuck_fires([True, True, False, True, False, True], 3, 2))

    def test_the_earlier_reading_counts_what_it_would_skip(self):
        days = [{"day": "2026-11-08", "start": ts(8, 18), "end": ts(9, 1)}]
        e = [rnd(8, 19, i, 0.5, changed=c) for i, c in enumerate([True, False, False, False, False])]
        e.append(rnd(8, 19, 10, 0.2, changes=[ACCEPTED], changed=True, settle=True))   # round 6 found the win
        got = an.stuck_reading(e, days, 3, 2)[0]
        self.assertEqual((got["fires_now"], got["fires_early"], got["skipped"]), (5, 3, 2))
        self.assertEqual(got["accepted_in_skipped"], 1)
        self.assertTrue(got["lower_score_in_skipped"])


class History(unittest.TestCase):
    def history(self):
        conn = ledger()
        pg(conn, ts(8, 18, 42), "run s1w0 round 0", 0.20)
        pg(conn, ts(8, 19), "run s1w0 g1", 0.20)
        pg(conn, ts(8, 14, 58), "run s1w0 rescore", 0.15, purpose="experiment")
        cl(conn, ts(8, 19), "loop s1w0 judge", 0.30)
        cl(conn, ts(8, 14, 59), "loop s1w0 judge", 5.00)                            # the smoke test's
        conn.execute("INSERT INTO baseline_scores (ts, run, kind, label, sha, model, words, fraction_ai)"
                     " VALUES (?, 'baseline', 'guide as posted', 'section 1', 'x', 'pangram-4.1', 900, 0.72)",
                     (ts(7, 21),))
        e = [{"kind": "start", "window": "s1w0", "time": st(8, 14, 58), "ai": 1.0, "segs": [], "text": "o"},
             rnd(8, 14, 59, 0.9, changes=[ACCEPTED], text="smoke"),                  # the smoke test: left out
             rnd(8, 19, 0, 0.6, changes=[ACCEPTED, NOTHING], text="a", segs=[["a", True]]),
             rnd(8, 20, 0, 0.2, changes=[dict(ACCEPTED, who="yours"), KEPT], text="b"),
             {"kind": "pass_end", "window": "s1w0", "time": st(8, 20, 1), "pass": 1, "reason": "rounds done"},
             {"kind": "whole", "day": "2026-11-08", "time": st(8, 23), "k": 1, "improved": [],
              "sections": {"1": {"ai": 0.40, "segs": [["a", True]]}}}]
        return an.build(conn, e, {}, CFG)

    def test_one_window_from_the_ledger_and_the_bank(self):
        w = self.history()["windows"]["s1w0"]
        self.assertEqual((w["original"], w["start"], w["best"], w["shipped"]), (1.0, 0.6, 0.2, 0.2))
        self.assertEqual((w["rounds"], w["accepted"], w["yours_regrown"]), (2, 2, 1))
        self.assertEqual(w["draws"], 48 + 44 + 48 + 48)                              # the smoke round is out
        self.assertEqual((w["pangram_usd"], w["claude_usd"]), (0.4, 0.3))
        self.assertEqual(w["cost_per_point"], 0.01)                                  # $0.40 for 0.60 to 0.20
        self.assertEqual(w["draws_per_accepted"], 94.0)

    def test_the_run_day_and_the_sections(self):
        h = self.history()
        d = h["run_days"][0]
        self.assertEqual(d["pangram"]["by_purpose"], {"round 0": 0.2, "window rounds": 0.2})
        self.assertEqual(d["claude"]["estimated_usd"], 0.3)
        self.assertEqual(d["claude"]["backstop_usd"], 20.0)
        self.assertEqual((d["rates"]["week"], d["rates"]["week_from"]), (0.007, "seed"))
        self.assertEqual([r["ai"] for r in h["sections"]["1"]], [0.72, 0.40])
        self.assertEqual(h["moved"]["rejects"], {"length": 43, "must-keep repeat": 1})
        json.dumps(h, allow_nan=False)                                               # no inf or nan in the file
        self.assertIn("## Windows", an.render(h))

    def test_a_refit_under_the_clamp_is_reported_held(self):
        conn = ledger()
        conn.execute("INSERT INTO claude_rates (ts, day, kind, points, dollars, rate) VALUES (?,?,?,?,?,?)",
                     (ts(9, 0, 48), "2026-11-08", "five_hour", 0.12, 6.35, 0.0189))
        d = an.build(conn, [], {}, CFG)["run_days"][0]
        self.assertEqual(d["rates"]["refits_at_the_end"][0]["in_use_next"], 0.025)
        self.assertTrue(d["rates"]["refits_at_the_end"][0]["clamped"])
        self.assertEqual(an.rate_in_use(conn, CFG["claude"], "five_hour", ts(10, 3)), (0.025, "refit"))


class DigestTable(unittest.TestCase):
    def loop(self):
        conn = ledger()
        pg(conn, ts(8, 19), "run s1w0 g1", 0.20)
        e = [rnd(8, 19, 0, 0.2, changes=[ACCEPTED, NOTHING], text="a")]
        w = types.SimpleNamespace(wid="s1w0", must={}, start_of_day=0.6)
        return types.SimpleNamespace(
            conn=conn, windows=[w], purpose="loop", clock=types.SimpleNamespace(now=lambda: ts(8, 23)),
            budget=types.SimpleNamespace(run_day_start=ts(8, 18, 41)), bank=types.SimpleNamespace(entries=e),
            ship_entry=lambda w: e[0], latest_score=lambda wid, text: (0.2, []), best=lambda w: e[0])

    def test_the_digest_shows_efficiency_per_window(self):
        from engine.discord import efficiency_lines
        L = efficiency_lines(self.loop())
        self.assertIn("## Efficiency today", L)
        self.assertIn("| s1w0 | 0.60 | 0.20 | $0.20 | $0.005 | 1 | 92.0 |", L)

    def test_a_loop_the_analytics_cant_read_leaves_the_digest_alone(self):
        from engine.discord import efficiency_lines
        self.assertEqual(efficiency_lines(types.SimpleNamespace(windows=[])), [])


# ---------------------------------------------------------------------- the M1 review's deferred findings
TRICK = "And that's the whole trick, really."


def drop_loop(bank, replies):
    """A Loop with only what drop_rule needs, and a Claude door that counts its calls."""
    from engine.run import Loop, WindowState
    lp = Loop.__new__(Loop)
    lp.bank, lp.cuts, lp.day, lp.settings, lp.conn, lp.claude_raw = bank, [], "2026-11-08", None, None, None
    lp.clock = types.SimpleNamespace(stamp=lambda ts=None: "2026-11-08 20:00", now=lambda: 0.0)
    lp.blocked = lambda: None
    lp.must_stop = lambda: False
    lp.log = lambda msg: None
    calls = []

    def claude(settings, conn, prompt, role, model, max_budget_usd):
        calls.append(role)
        return replies.pop(0), {}
    lp.claude = claude
    ws = WindowState({"wid": "s1w0", "text": "", "section": 1, "index": 0})
    ws.runs = [[{"sents": [TRICK], "claude": True}, {"sents": ["I wrote this bit myself."], "claude": False}]]
    ws.original = [[[TRICK], ["I wrote this bit myself."]]]
    ws.current = json.loads(json.dumps(ws.original))
    ws.attempts, ws.completed, ws.pass_no, ws.heading = {(0, 0): 6}, 6, 1, ""
    ws.who, ws.notes = {TRICK: "claude"}, [["the trick"]]
    return lp, ws, calls


class Findings(unittest.TestCase):
    def test_14_a_score_only_carries_over_onto_the_text_it_belongs_to(self):
        from engine.run import score_is_fresh
        old, new = [[["Claude's line."]]], [[["Mistral's line."]]]
        self.assertTrue(score_is_fresh(0.40, old, [[["Claude's line."]]]))      # pass: nothing moved since the score
        # flag: a round changed the text, Pangram was capped, and the next round changes nothing: the old score
        # belongs to the old text, so that round has to score the text it banks
        self.assertFalse(score_is_fresh(0.40, old, new))
        self.assertFalse(score_is_fresh(None, None, new))

    def test_9_a_cut_banked_without_a_score_gets_scored_later(self):
        from engine.job import Bank
        with tempfile.TemporaryDirectory() as d:
            b = Bank(Path(d) / "bank.jsonl")
            b.add({"kind": "round", "window": "w", "ai": 0.3, "text": "a", "runs": [[["a"]]]})
            b.add({"kind": "cut", "window": "w", "ai": None, "text": "b", "runs": [[["b"]]], "pass": 1})
            self.assertEqual([e["text"] for e in b.unscored("w")], ["b"])       # flag: the cut waits for a score
            b.add({"kind": "rescore", "window": "w", "ai": 0.2, "text": "b", "runs": [[["b"]]]})
            self.assertEqual(b.unscored("w"), [])                               # pass: scored, nothing waits

    def test_11_a_resumed_pass_keeps_its_cut_and_doesnt_ask_again(self):
        from engine.job import Bank
        from engine.run import resume_base
        with tempfile.TemporaryDirectory() as d:
            b = Bank(Path(d) / "bank.jsonl")
            old_cut = b.add({"kind": "cut", "window": "w", "pass": 1, "para": 0, "run": 0, "text": "x", "runs": [[[]]]})
            r = b.add({"kind": "round", "window": "w", "pass": 2, "text": "a", "runs": [[["a"]]]})
            self.assertIs(resume_base(b, "w", (2, [r])), r)                     # pass: an older pass's cut stays put
            cut = b.add({"kind": "cut", "window": "w", "pass": 2, "para": 0, "run": 0, "text": "", "runs": [[[]]]})
            self.assertIs(resume_base(b, "w", (2, [r])), cut)                   # flag: the crash came after the cut
            self.assertIsNot(old_cut, cut)

    def test_11_the_drop_rule_banks_a_no_and_reads_it_back(self):
        from engine.job import Bank
        from engine.run import Loop
        with tempfile.TemporaryDirectory() as d:
            b = Bank(Path(d) / "bank.jsonl")
            lp, ws, calls = drop_loop(b, ['{"pass": false, "lost": [], "why": "the joke carries the point"}'])
            Loop.drop_rule(lp, ws)
            self.assertEqual(len(calls), 1)
            self.assertEqual([e["kind"] for e in b.entries], ["drop_no"])
            lp2, ws2, calls2 = drop_loop(b, [])                                 # the same pass, after a restart
            Loop.drop_rule(lp2, ws2)
            self.assertEqual(calls2, [])                                        # no second Sonnet call
            self.assertIn("the joke carries the point", ws2.couldnt_cut[-1][2])
            ws2.pass_no = 2                                                     # a new pass asks afresh
            lp2.claude = lambda *a, **k: calls2.append("asked") or ('{"pass": false, "why": "still no"}', {})
            Loop.drop_rule(lp2, ws2)
            self.assertEqual(calls2, ["asked"])

    def test_11_a_cut_banked_this_pass_blocks_a_second_cut_in_its_paragraph(self):
        from engine.job import Bank
        from engine.run import Loop
        with tempfile.TemporaryDirectory() as d:
            b = Bank(Path(d) / "bank.jsonl")
            b.add({"kind": "cut", "window": "s1w0", "pass": 1, "para": 0, "run": 2, "text": "", "runs": [[[]]]})
            lp, ws, calls = drop_loop(b, [])
            Loop.drop_rule(lp, ws)
            self.assertEqual(calls, [])                                         # one cut a paragraph a pass, banked

    def test_13_the_stuck_rule_reading_stays_and_the_history_says_why(self):
        # the review's note: as configured the rule can only end a pass after round 5, which with 6 rounds a pass
        # skips nothing. The bank shows the late rounds earn their keep (the history's stuck-rule line), so the
        # reading stays; the analytics measure what an earlier one would lose, and this pins the rule as it reads
        from engine.run import stuck_rule
        self.assertFalse(stuck_rule([True, False, False, False], 3, 2))
        self.assertTrue(stuck_rule([True, False, False, False, False], 3, 2))
        self.assertEqual(an.stuck_fires([True, False, False, False, False, False], 3, 2), 5)


# ---------------------------------------------------------------------- the Pangram model-change alert (plan 9)
PG_CFG = {"pangram": {"usd_per_credit": 0.05, "words_per_credit": 100, "monthly_cap_usd": 180.0,
                      "nightly_cap_usd": 6.0, "calibration_cap_usd": 15.0, "poll_seconds": 0.0,
                      "timeout_seconds": 5, "model_ack": "4.1"}}


class FakeResp:
    def __init__(self, body):
        self.status_code, self.text, self.body = 200, "", body

    def json(self):
        return self.body

    def raise_for_status(self):
        return None


class FakeHTTP:
    """Pangram's two endpoints, answering with the versions given, one a call."""

    def __init__(self, versions):
        self.versions = list(versions)

    def post(self, url, headers=None, json=None, timeout=None):
        return FakeResp({"task_id": "t"})

    def get(self, url, headers=None, timeout=None):
        return FakeResp({"stage": "STAGE_SUCCESS", "version": self.versions.pop(0), "fraction_ai": 0.1,
                         "prediction_short": "Mixed", "windows": []})


def pangram(versions, ack="4.1", conn=None):
    from engine.pangram import Pangram
    from engine.settings import Settings
    conn = conn or sqlite3.connect(":memory:")
    conn.executescript(db.SCHEMA)
    cfg = json.loads(json.dumps(PG_CFG))
    cfg["pangram"]["model_ack"] = ack
    return Pangram(Settings(env={"PANGRAM_API_KEY": "fake"}, config=cfg), conn, session=FakeHTTP(versions)), conn


class ModelChange(unittest.TestCase):
    def test_the_ledger_learns_the_version_of_the_rows_it_already_has(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(db.SCHEMA)
        conn.execute("INSERT INTO pangram_calls (ts, sha, model, words, credits, usd, purpose, label, status)"
                     " VALUES (1, 'abc', 'pangram-4.1', 100, 1, 0.05, 'loop', 'old', 'ok')")
        conn.execute("INSERT INTO pangram_cache VALUES ('abc', 'pangram-4.1', ?, 1)", (json.dumps({"version": "4.1"}),))
        pg, _ = pangram([], conn=conn)
        self.assertEqual(pg.last_version(), "4.1")                         # backfilled from the cached answer
        self.assertFalse(pg.model_status()["stale"])

    def test_a_new_version_is_a_change_and_the_old_answers_stop_counting(self):
        pg, _ = pangram(["4.1", "4.2"])
        pg.score("the first text", "loop")
        self.assertIsNone(pg.model_change)                                 # pass: the version everyone trusts
        self.assertIsNotNone(pg.cached("the first text"))
        pg.score("the second text", "loop")
        self.assertEqual(pg.model_change, ("4.1", "4.2"))                  # flag: Pangram moved on
        st = pg.model_status()
        self.assertEqual((st["latest"], st["trusted"], st["stale"]), ("4.2", "4.1", True))
        self.assertIsNone(pg.cached("the first text"))                     # 4.1's answer is a miss now
        self.assertIsNotNone(pg.cached("the second text"))

    def test_acknowledging_the_new_version_clears_it(self):
        pg, conn = pangram(["4.1", "4.2"], ack="4.2")
        pg.score("the first text", "loop")
        pg.score("the second text", "loop")
        self.assertFalse(pg.model_status()["stale"])
        self.assertIsNone(pg.cached("the first text"))                     # still scored again by the new model

    def test_one_version_all_along_is_never_stale_whatever_the_ack_says(self):
        pg, _ = pangram(["rehearsal", "rehearsal"])                        # the rehearsal's fake answers
        pg.score("the first text", "loop")
        pg.score("the second text", "loop")
        self.assertFalse(pg.model_status()["stale"])
        self.assertIsNotNone(pg.cached("the first text"))

    def stub(self, versions):
        from engine.run import WindowState
        from test_run_loop import loop_stub
        pg, conn = pangram(versions)
        sent = []
        lp = loop_stub(pg=pg, budget=types.SimpleNamespace(pangram_check=lambda usd, purpose, reserve=0.0: None),
                       notifier=types.SimpleNamespace(send=lambda text, **k: sent.append(text)))
        ws = WindowState({"wid": "s1w0", "text": "", "section": 1, "index": 0})
        ws.ready, ws.ai, ws.scored_runs = True, 0.30, [[["x"]]]
        lp.windows = [ws]
        return lp, ws, sent

    def test_the_loop_alerts_once_and_stops_trusting_the_standing_score(self):
        lp, ws, sent = self.stub(["4.1", "4.2", "4.2"])
        lp.score("a window", "s1w0 g1")
        self.assertEqual(sent, [])                                         # pass: same model
        self.assertEqual(ws.ai, 0.30)
        lp.score("a window, regrown", "s1w0 g2")
        self.assertEqual(len(sent), 1)                                     # flag: the ALERT, posted
        self.assertIn("model version 4.2", sent[0])
        self.assertIn('model_ack = "4.2"', sent[0])
        self.assertIn("ALERT", lp.timeline[-1])
        self.assertEqual((ws.ai, ws.scored_runs), (None, None))           # the next round scores its text
        lp.score("a window, regrown again", "s1w0 g3")
        self.assertEqual(len(sent), 1)                                     # once a run

    def test_scores_banked_before_the_change_count_as_unscored(self):
        from engine.job import Bank
        from engine.run import Loop
        lp, ws, _ = self.stub([])
        with tempfile.TemporaryDirectory() as d:
            lp.bank = Bank(Path(d) / "bank.jsonl")
            lp.bank.add({"kind": "round", "window": "s1w0", "time": "2026-11-09 04:00", "ai": 0.0, "text": "t",
                         "runs": [[["t"]]]})
            self.assertEqual(Loop.latest_score(lp, "s1w0", "t")[0], 0.0)    # pass: nothing changed
            lp.model_status = {"stale": True, "changed_at": ts(9, 5), "latest": "4.2", "trusted": "4.1"}
            self.assertIsNone(Loop.latest_score(lp, "s1w0", "t"))           # flag: made by the old model
            lp.bank.add({"kind": "rescore", "window": "s1w0", "time": "2026-11-09 05:10", "ai": 0.2, "text": "t",
                         "runs": [[["t"]]]})
            self.assertEqual(Loop.latest_score(lp, "s1w0", "t")[0], 0.2)    # scored again since: trusted

    def test_the_whole_text_round_is_due_until_every_section_is_rescored(self):
        from engine.job import Bank
        from engine.run import Loop
        lp, _, _ = self.stub([])
        lp.windows, lp.whole_snapshot = [], None
        lp.whole_sections = lambda: [1, 3]
        with tempfile.TemporaryDirectory() as d:
            lp.bank = Bank(Path(d) / "bank.jsonl")
            lp.bank.add({"kind": "whole_end", "time": "2026-11-09 04:00", "picks": {"gone": "x"}})
            self.assertFalse(Loop.whole_due(lp))                             # pass: nothing changed since
            lp.model_status = {"stale": True, "changed_at": ts(9, 5), "latest": "4.2", "trusted": "4.1"}
            self.assertTrue(Loop.whole_due(lp))                              # flag: the model changed
            lp.bank.add({"kind": "whole", "time": "2026-11-09 05:20", "sections": {"1": {"ai": 0.3}}})
            self.assertTrue(Loop.whole_due(lp))                              # a cap stopped it after section 1
            lp.bank.add({"kind": "whole", "time": "2026-11-09 05:30", "sections": {"3": {"ai": 0.0}}})
            self.assertFalse(Loop.whole_due(lp))

    def test_the_digest_says_so_while_it_holds(self):
        from engine.discord import model_lines
        self.assertEqual(model_lines(types.SimpleNamespace(model_status={"stale": False})), [])
        L = model_lines(types.SimpleNamespace(model_status={"stale": True, "latest": "4.2", "trusted": "4.1",
                                                            "changed_at_text": "2026-11-09 05:00"}))
        self.assertIn("## Pangram model change", L)
        self.assertIn('model_ack = \\"4.2\\"', json.dumps(L))


if __name__ == "__main__":
    unittest.main()
