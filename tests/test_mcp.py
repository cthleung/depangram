"""Tests for the MCP server (mcp_server/): each tool, with fakes for Pangram, Open Pangram, the child process, the busy
check and the process tree killer, a ledger in a temporary folder, and the fixture job copied into a temporary jobs
folder, so nothing here reads .env, config.toml or the real runs folder, and nothing calls out. Also: a run refused
while the GPU is taken, a missing Pangram ceiling, a path outside the repo, a restart that reads the run records back,
stop_run on a fake process tree, and the tool list as the SDK's own client lists it. Run from the repo root:

    python -m unittest discover tests
"""
import asyncio
import contextlib
import importlib.util
import io
import json
import logging
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from engine.settings import CONFIG_EXAMPLE, Settings, load_config
from mcp_server import child
from mcp_server import runs as rs
from mcp_server.core import App, Refused, plain_error

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "balcony"
KEY = "pk-test-0123456789abcdef"
T = time.mktime((2026, 11, 16, 2, 0, 0, 0, 0, -1))          # 02:00 local, inside the night's hours
AFTERNOON = time.mktime((2026, 11, 16, 14, 0, 0, 0, 0, -1))


class FakePangram:
    """Pangram's client: estimate() and score(), with the purpose and label of each call kept."""

    def __init__(self, cached=False, error=None):
        self.calls, self.cached, self.error = [], cached, error

    def estimate(self, text):
        return 2, 0.10

    def score(self, text, purpose, label=""):
        if self.error:
            raise self.error
        self.calls.append((text, purpose, label))
        windows = [{"text": "Delve into the rich tapestry.", "ai_assistance_score": 0.93, "is_humanized": False},
                   {"text": "I drilled extra holes in my first pot.", "ai_assistance_score": 0.04}]
        return {"prediction_short": "Mixed", "fraction_ai": 0.5, "fraction_ai_assisted": 0.1, "fraction_human": 0.4,
                "windows": windows, "version": "4.1"}, self.cached


class FakeOpen:
    loads = 0

    def __init__(self):
        FakeOpen.loads += 1

    def score(self, texts):
        return [{"score": 0.9 if "tapestry" in t else 0.1} for t in texts]


class FakeProc:
    def __init__(self, pid=4242, rc=None):
        self.pid, self.rc = pid, rc

    def poll(self):
        return self.rc


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.repo = root / "repo"
        self.jobs = self.repo / "jobs"
        self.runs = root / "runs"
        self.jobs.mkdir(parents=True)
        shutil.copytree(FIXTURE, self.jobs / "balcony")
        shutil.copytree(REPO / "jobs" / "example", self.jobs / "example")
        self.config = load_config(CONFIG_EXAMPLE)
        self.settings = Settings(env={"RUNS_DIR": str(self.runs), "PANGRAM_API_KEY": KEY}, config=self.config)
        self.spawned, self.killed = [], []
        self.busy_why = None
        self.clock = [T]
        self.alive = {}

    def tearDown(self):
        self.tmp.cleanup()

    def spawn(self, cmd, out, env):
        self.spawned.append((cmd, env))
        out.write(b"2026-11-16 02:00 starting the pass\n")
        return self.proc

    def kill(self, proc):
        self.killed.append(proc.pid)
        return [proc.pid, proc.pid + 1], []

    def app(self, **kw):
        args = dict(repo=self.repo, jobs_dir=self.jobs, pangram_fn=lambda s, c: self.pangram,
                    open_pangram_fn=FakeOpen, busy_fn=lambda: self.busy_why, spawn_fn=self.spawn,
                    kill_fn=self.kill, port_fn=lambda: False, now=lambda: self.clock[0],
                    alive_fn=lambda pid, began: self.alive.get(pid, False), began_fn=lambda pid: 1000.0,
                    config_fn=lambda: self.config, python="python")
        args.update(kw)
        return App(self.settings, **args)

    def start(self, app=None, **kw):
        self.proc = getattr(self, "proc", None) or FakeProc()
        args = dict(job="balcony", max_pangram_usd=2.5)
        args.update(kw)
        return (app or self.app()).start_run(**args)

    def run_dir(self, run_id):
        return self.runs / "mcp" / run_id


# ---------------------------------------------------------------------- score_text
class ScoreText(Base):
    def test_pangram_goes_through_the_client_and_reports_its_cost(self):
        self.pangram = FakePangram()
        got = self.app().score_text("Delve into the rich tapestry. I drilled extra holes in my first pot.", "pangram")
        self.assertEqual(got["ai_share"], 0.5)
        self.assertEqual(got["flagged"], ["Delve into the rich tapestry."])                 # flag: over 0.5
        self.assertFalse(got["humanized"])
        self.assertEqual(got["cost_usd"], 0.10)
        self.assertEqual(self.pangram.calls[0][1], "mcp")                                   # its own purpose
        self.assertTrue((self.runs / "depangram.sqlite").exists())                          # the ledger is the run's

    def test_a_cached_pangram_answer_costs_nothing(self):
        self.pangram = FakePangram(cached=True)
        got = self.app().score_text("Some text.", "pangram")
        self.assertEqual((got["cost_usd"], got["from_cache"]), (0.0, True))

    def test_a_pangram_cap_is_a_plain_refusal(self):
        from engine.pangram import CapReached
        self.pangram = FakePangram(error=CapReached("monthly cap $180.00 would be crossed"))
        with self.assertRaises(Refused) as e:
            self.app().score_text("Some text.", "pangram")
        self.assertIn("monthly cap", str(e.exception))

    def test_a_pangram_error_never_echoes_the_key(self):
        from engine.pangram import PangramError
        self.pangram = FakePangram(error=PangramError(f"create task failed: 401 bad key {KEY}"))
        with self.assertRaises(Refused) as e:
            self.app().score_text("Some text.", "pangram")
        self.assertNotIn(KEY, str(e.exception))
        self.assertIn("[redacted]", str(e.exception))

    def test_open_pangram_loads_once_and_has_no_humanized_flag(self):
        FakeOpen.loads = 0
        app = self.app()
        got = app.score_text("Delve into the rich tapestry. I drilled extra holes in my first pot.")
        app.score_text("Another one.", "open")
        self.assertEqual(FakeOpen.loads, 1)
        self.assertEqual(got["detector"], "open")
        self.assertIsNone(got["humanized"])
        self.assertEqual(got["flagged"], ["Delve into the rich tapestry."])
        self.assertEqual(got["cost_usd"], 0.0)

    def test_empty_text_and_an_unknown_detector_are_refused(self):
        with self.assertRaises(Refused):
            self.app().score_text("   ")
        with self.assertRaises(Refused):
            self.app().score_text("Text.", "gptzero")


# ---------------------------------------------------------------------- list_jobs and review_queue
def passed_fake(job, bank, sections, readonly=False):
    if not readonly:
        raise AssertionError("the server reads the pass read only")
    return {"sections": {1: {"ai": 0.12, "label": "s1 round 3", "text": "...", "scored": True}}, "left_out": 0,
            "paragraphs": {
                "s1w0 p2": {"text": "Para one.", "section": 1, "round": "s1 round 3", "ai": 0.12, "flagged": False,
                            "for_you": None, "repaired": "cut an invented sentence", "picked": True, "stuck": []},
                "s2w0 p3": {"text": "Para two.", "section": 2, "round": None, "ai": None, "flagged": False,
                            "for_you": None, "repaired": None, "picked": False,
                            "stuck": [{"bold": ["A machine sentence."], "why": "no pick passed", "about": "pots",
                                       "point": "drainage", "fact": "holes"}]},
                "s3w0 p2": {"text": "Hand fixed.", "section": 3, "round": "s3 round 1", "ai": 0.2, "flagged": False,
                            "for_you": {"went_wrong": "a made-up number", "about": "soil", "point": "mix",
                                        "fact": "one part grit"}, "repaired": None, "picked": True, "stuck": []}},
            "moot": ["s2w1 p1"], "unmatched": ["s9w0 p1"], "unmatched_review": [["veto", "s7w0 p1"]]}


class Jobs(Base):
    def give_bank(self, origs=True):
        d = self.jobs / "balcony"
        (d / "regrow bank.jsonl").write_text("", encoding="utf-8")
        if origs:
            (d / "regrow origs.json").write_text("{}", encoding="utf-8")

    def test_list_jobs_reads_the_served_scores(self):
        self.give_bank()
        got = {r["folder"]: r for r in self.app(pass_paragraphs_fn=passed_fake).list_jobs()["jobs"]}
        b = got["balcony"]
        self.assertEqual(b["sections"], [1, 2, 3])
        self.assertEqual(b["scores"], {"1": {"ai_share": 0.12, "scoring": "s1 round 3", "scored": True}})
        self.assertIn("2, 3", b["scores_note"])
        self.assertIsNone(got["example"]["scores"])                                          # no pass has run

    def test_a_bank_without_its_sidecar_is_read_without_writing_one(self):
        self.give_bank(origs=False)

        def fake(job, bank, sections, readonly=False):
            return dict(passed_fake(job, bank, sections, readonly), left_out=3)
        got = self.app(pass_paragraphs_fn=fake).list_jobs()["jobs"]
        b = next(r for r in got if r["folder"] == "balcony")
        self.assertEqual(b["scores"]["1"]["ai_share"], 0.12)                      # read, not skipped
        self.assertIn("3 older entries", b["scores_note"])
        self.assertFalse((self.jobs / "balcony" / "regrow origs.json").exists())

    def test_the_real_pass_read_writes_nothing_in_the_job(self):
        self.give_bank(origs=False)
        before = sorted(p.name for p in (self.jobs / "balcony").iterdir())
        got = {r["folder"]: r for r in self.app().list_jobs()["jobs"]}
        self.assertNotIn("error", got["balcony"])
        self.assertEqual(got["balcony"]["scores"], {})
        self.assertEqual(sorted(p.name for p in (self.jobs / "balcony").iterdir()), before)

    def test_each_job_is_read_under_its_own_rules_and_nothing_is_left_behind(self):
        from engine import checks
        mine = self.jobs / "frank"
        shutil.copytree(FIXTURE, mine)
        toml = (mine / "job.toml").read_text(encoding="utf-8").replace('name = "balcony"',
                                                                       'name = "frank"\nrules = "rules.toml"')
        (mine / "job.toml").write_text(toml, encoding="utf-8")
        (mine / "rules.toml").write_text('[filler]\nwords = ["frankly"]\n', encoding="utf-8")
        for d in (mine, self.jobs / "balcony"):
            (d / "regrow bank.jsonl").write_text("", encoding="utf-8")
        seen = {}

        def fake(job, bank, sections, readonly=False):
            seen[job.name] = checks.FILLER_TAGS
            return passed_fake(job, bank, sections, readonly)
        base = checks.FILLER_TAGS
        self.app(pass_paragraphs_fn=fake).list_jobs()
        self.assertEqual(seen["frank"], ("frankly",))                              # its own file
        self.assertEqual(seen["balcony"], base)                                     # the next job: the base again
        self.assertEqual(checks.FILLER_TAGS, base)                                  # and nothing left behind

    def test_one_bad_job_doesnt_sink_the_list(self):
        (self.jobs / "broken").mkdir()
        (self.jobs / "broken" / "job.toml").write_text("[job\nname = ", encoding="utf-8")
        got = {r["folder"]: r for r in self.app().list_jobs()["jobs"]}
        self.assertIn("error", got["broken"])
        self.assertNotIn(str(self.repo), got["broken"]["error"])                              # the repo is <repo>
        self.assertIn("sections", got["balcony"])

    def test_review_queue_lists_the_bold_paragraphs_and_the_review_file(self):
        self.give_bank()
        review = {"vetoes": [{"label": "s1w0 p2", "orig": "o", "text": "bad pick", "why": "wrong fact"}],
                  "repairs": [{"label": "s1w0 p2", "orig": "o", "from": "a", "text": "b", "moves": "swap 1"}],
                  "for_you": [], "stuck": []}
        (self.jobs / "balcony" / "review.json").write_text(json.dumps(review), encoding="utf-8")
        got = self.app(pass_paragraphs_fn=passed_fake).review_queue("balcony")
        kinds = sorted((x["paragraph"], x["kind"]) for x in got["for_you"])
        self.assertEqual(kinds, [("s2w0 p3", "stuck"), ("s3w0 p2", "hand fix")])
        stuck = next(x for x in got["for_you"] if x["kind"] == "stuck")
        self.assertEqual(stuck["bold"], ["A machine sentence."])
        self.assertEqual(stuck["prompt"]["point"], "drainage")
        self.assertEqual(got["repaired"], [{"paragraph": "s1w0 p2", "section": 1, "moves": "cut an invented sentence"}])
        self.assertEqual(got["vetoes"][0]["why"], "wrong fact")
        self.assertEqual(got["repairs"][0]["moves"], "swap 1")
        self.assertEqual((got["unmatched"], got["unmatched_review"]), (["s9w0 p1"], [["veto", "s7w0 p1"]]))

    def test_review_queue_without_a_pass_lists_the_stuck_entries_from_the_file(self):
        review = {"stuck": [{"label": "s2w0 p3", "orig": "A machine sentence.", "why": "no pick", "about": "pots"}]}
        (self.jobs / "balcony" / "review.json").write_text(json.dumps(review), encoding="utf-8")
        got = self.app().review_queue("balcony")
        self.assertIsNone(got["for_you"])
        self.assertEqual(got["stuck_entries"][0]["about"], "pots")

    def test_a_path_outside_the_repo_is_refused(self):
        outside = Path(self.tmp.name) / "elsewhere"
        shutil.copytree(FIXTURE, outside)
        for job in (str(outside), "../elsewhere", "jobs/../../elsewhere"):
            with self.assertRaises(Refused, msg=job) as e:
                self.app().review_queue(job)
            self.assertIn("isn't inside", str(e.exception))
        with self.assertRaises(Refused):
            self.start(job=str(outside))
        self.assertEqual(self.spawned, [])

    def test_a_job_folder_inside_the_repo_is_taken_by_its_path(self):
        shutil.copytree(FIXTURE, self.repo / "elsewhere")
        got = self.start(job=str(self.repo / "elsewhere"))
        cmd = self.spawned[0][0]
        self.assertIn("--job-dir", cmd)
        self.assertEqual(got["job"], "balcony")


# ---------------------------------------------------------------------- start_run
class StartRun(Base):
    def test_a_pass_starts_in_the_background_with_its_ceilings(self):
        got = self.start(sections=[1, 3], rounds=2, max_claude_usd=1.5)
        self.assertTrue(rs.valid_id(got["run_id"]))
        cmd, env = self.spawned[0]
        self.assertEqual(cmd[:6], ["python", "-m", "mcp_server.child", str(self.run_dir(got["run_id"]) / "exit"),
                                   "-m", "engine.regrow"])
        args = cmd[6:]
        self.assertEqual(args[args.index("--max-usd") + 1], "2.50")
        self.assertEqual(args[args.index("--max-claude-usd") + 1], "1.50")
        self.assertEqual(args[args.index("--rounds") + 1], "3")                    # regrow counts the arm-set round
        self.assertEqual(args[args.index("--purpose") + 1], "loop")
        self.assertEqual(args[args.index("--section") + 1], "1,3")
        self.assertEqual(args[:2], ["--job", "balcony"])
        self.assertEqual(env.get("DEPANGRAM_CONFIG"), os.environ.get("DEPANGRAM_CONFIG"))   # a pass reads the config
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        rec = json.loads((self.run_dir(got["run_id"]) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((rec["pid"], rec["state"], rec["max_pangram_usd"]), (4242, "running", 2.5))
        self.assertFalse((self.repo / "analytics").exists())                       # the engine makes its own folder

    def test_no_pangram_ceiling_no_run(self):
        for bad in (None, 0, -1, float("nan"), float("inf"), "lots"):
            with self.assertRaises(Refused, msg=repr(bad)):
                self.start(max_pangram_usd=bad)
        self.assertEqual(self.spawned, [])

    def test_the_gpu_taken_refuses_the_run(self):
        self.busy_why = "python -m engine.regrow is running (pid 99)"
        with self.assertRaises(Refused) as e:
            self.start()
        self.assertIn("GPU", str(e.exception))
        self.assertEqual(self.spawned, [])
        self.busy_why = None
        self.start()                                                               # pass: free again
        self.assertEqual(len(self.spawned), 1)

    def test_a_run_of_its_own_still_going_refuses_the_next(self):
        app = self.app()
        self.start(app)
        with self.assertRaises(Refused) as e:
            self.start(app)
        self.assertIn("still going", str(e.exception))
        self.assertEqual(len(self.spawned), 1)

    def test_the_stop_file_refuses_the_run(self):
        self.runs.mkdir(parents=True, exist_ok=True)
        (self.runs / "STOP").write_text("", encoding="utf-8")
        with self.assertRaises(Refused) as e:
            self.start()
        self.assertIn("STOP", str(e.exception))
        self.assertEqual(self.spawned, [])

    def test_sections_outside_the_scope_and_bad_rounds_are_refused(self):
        with self.assertRaises(Refused):
            self.start(sections=[9])
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(Refused, msg=repr(bad)):
                self.start(rounds=bad)
        self.assertEqual(self.spawned, [])

    def test_a_night_gets_its_ceilings_as_flags(self):
        got = self.start(kind="night", max_pangram_usd=1.25, max_claude_usd=0.75, sections=[2], rounds=2)
        cmd, env = self.spawned[0]
        self.assertEqual(cmd[5:8], ["engine.night", "--job", "balcony"])
        args = cmd[8:]
        self.assertEqual(args[args.index("--max-pangram-usd") + 1], "1.25")
        self.assertEqual(args[args.index("--max-claude-usd") + 1], "0.75")
        self.assertEqual(args[args.index("--rounds") + 1], "2")                    # the night adds the arm-set round
        self.assertEqual(args[args.index("--section") + 1], "2")
        self.assertNotIn("--pass-minutes", args)
        self.assertEqual(env.get("DEPANGRAM_CONFIG"), os.environ.get("DEPANGRAM_CONFIG"))   # no config copy
        self.assertEqual(os.listdir(self.run_dir(got["run_id"])), ["out.log", "run.json"])
        self.assertFalse(any("skips the paragraph pass" in n for n in got["notes"]))
        self.assertTrue(any("covers the whole night" in n for n in got["notes"]))

    def test_a_night_started_in_the_afternoon_says_it_runs_the_submission_only(self):
        self.clock[0] = AFTERNOON
        got = self.start(kind="night")
        self.assertTrue(any("skips the paragraph pass" in n for n in got["notes"]))

    def test_pass_minutes_runs_a_night_s_pass_at_any_time(self):
        self.clock[0] = AFTERNOON
        got = self.start(kind="night", pass_minutes=45)
        args = self.spawned[0][0]
        self.assertEqual(args[args.index("--pass-minutes") + 1], "45")
        self.assertFalse(any("skips the paragraph pass" in n for n in got["notes"]))

    def test_pass_minutes_is_for_a_night_and_a_whole_number(self):
        for kw in ({"kind": "pass", "pass_minutes": 30}, {"kind": "night", "pass_minutes": 0},
                   {"kind": "night", "pass_minutes": 2.5}, {"kind": "night", "pass_minutes": True}):
            with self.assertRaises(Refused, msg=repr(kw)):
                self.start(**kw)
        self.assertEqual(self.spawned, [])


# ---------------------------------------------------------------------- run_status, run_report, restarts
class Runs(Base):
    def ledger(self, rows_pangram, rows_claude):
        from engine import db
        from engine.claude_call import ensure_schema
        self.runs.mkdir(parents=True, exist_ok=True)
        conn = db.connect(self.runs / "depangram.sqlite")
        ensure_schema(conn)
        for ts, usd in rows_pangram:
            conn.execute("INSERT INTO pangram_calls (ts, sha, model, words, credits, usd, purpose, label, status)"
                         " VALUES (?, 'x', 'm', 100, 1, ?, 'loop', '', 'ok')", (ts, usd))
        for ts, usd in rows_claude:
            conn.execute("INSERT INTO claude_calls (ts, role, model, ok, cost_usd) VALUES (?, 'r', 'sonnet', 1, ?)",
                         (ts, usd))
        conn.commit()
        conn.close()

    def test_status_while_running_then_finished(self):
        app = self.app()
        got = self.start(app)
        rid = got["run_id"]
        with (self.run_dir(rid) / "out.log").open("a", encoding="utf-8") as f:
            f.write("2026-11-16 02:10 s1 all-notes: section 1 0.41 (3 flagged); Pangram $0.20\n")
            f.write(f"2026-11-16 02:11 a line with the key {KEY} in it\n")
        self.ledger([(T - 100, 9.0), (T + 10, 0.20), (T + 20, 0.05)], [(T + 15, 0.30), (T - 5, 4.0)])
        self.clock[0] = T + 60
        st = app.run_status(rid)
        self.assertEqual(st["state"], "running")
        self.assertEqual(st["spent"]["pangram_usd"], 0.25)                         # only the run's own window
        self.assertEqual(st["spent"]["claude_usd"], 0.30)
        self.assertEqual(st["scores"], [{"scoring": "s1 all-notes", "section": 1, "ai_share": 0.41, "flagged": 3,
                                         "humanized": False}])
        self.assertFalse(any(KEY in x for x in st["last_lines"]))
        self.proc.rc = 0
        (self.run_dir(rid) / "exit").write_text("0", encoding="utf-8")
        st = app.run_status(rid)
        self.assertEqual((st["state"], st["exit_code"]), ("finished", 0))
        self.assertIsNotNone(st["ended"])

    def test_a_locked_or_missing_ledger_gives_no_spend_instead_of_failing(self):
        app = self.app()
        rid = self.start(app)["run_id"]
        st = app.run_status(rid)                                                    # no ledger file yet
        self.assertEqual((st["spent"]["pangram_usd"], st["spent"]["claude_usd"]), (0.0, 0.0))

    def test_a_restart_reads_the_records_back(self):
        first = self.app()
        a = self.start(first)["run_id"]
        second = self.app()                                                        # a new server: no Popen held
        self.alive[4242] = True
        self.assertEqual(second.run_status(a)["state"], "running")
        self.alive[4242] = False
        (self.run_dir(a) / "exit").write_text("3", encoding="utf-8")
        st = second.run_status(a)
        self.assertEqual((st["state"], st["exit_code"]), ("finished", 3))
        # another run whose process went with no exit code is lost
        self.proc = FakeProc(pid=5151)
        self.clock[0] = T + 5
        b = self.start(self.app())["run_id"]
        st = self.app().run_status(b)
        self.assertEqual((st["state"], st["exit_code"]), ("lost", None))
        rec = json.loads((self.run_dir(b) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["state"], "lost")                                     # saved, so it stays lost

    def test_a_reused_pid_is_not_the_run(self):
        me = os.getpid()
        began = rs.create_time(me)
        self.assertTrue(rs.same_process(me, began))
        self.assertFalse(rs.same_process(me, began - 100))                          # flag: another process's pid

    def test_report_reads_what_the_run_wrote(self):
        app = self.app()
        rid = self.start(app)["run_id"]
        (self.repo / "analytics").mkdir(exist_ok=True)
        old = self.repo / "analytics" / "regrow 2026-11-15 0200.md"
        new = self.repo / "analytics" / "regrow 2026-11-16 0230.md"
        old.write_text("# old report\n", encoding="utf-8")
        new.write_text(f"# new report\n\nThe repo is at {self.repo}.\n", encoding="utf-8")
        os.utime(old, (T - 86400, T - 86400))
        os.utime(new, (T + 1800, T + 1800))
        self.clock[0] = T + 1900
        got = app.run_report(rid)
        self.assertEqual(got["files"], ["analytics/regrow 2026-11-16 0230.md"])
        self.assertIn("# new report", got["markdown"])
        self.assertIn("<repo>", got["markdown"])
        self.assertNotIn("old report", got["markdown"])

    def test_report_finds_a_report_named_in_the_log_and_a_night_digest(self):
        app = self.app()
        rid = self.start(app, kind="night")["run_id"]
        (self.repo / "analytics").mkdir(exist_ok=True)
        (self.repo / "digests").mkdir(exist_ok=True)
        named = self.repo / "analytics" / "regrow 2026-11-16 0300 (final).md"
        named.write_text("# finals\n", encoding="utf-8")
        os.utime(named, (T - 3600, T - 3600))                                      # outside the window, but named
        digest = self.repo / "digests" / "2026-11-16 morning submission.md"
        digest.write_text("# the digest\n", encoding="utf-8")
        os.utime(digest, (T + 600, T + 600))
        night_log = self.runs / "balcony" / "night.log"
        night_log.parent.mkdir(parents=True, exist_ok=True)
        night_log.write_text("2026-11-16 03:00 done: Pangram $0.40; report regrow 2026-11-16 0300 (final).md\n",
                             encoding="utf-8")
        os.utime(night_log, (T + 700, T + 700))
        self.clock[0] = T + 900
        got = app.run_report(rid)
        self.assertEqual(sorted(got["files"]), ["analytics/regrow 2026-11-16 0300 (final).md",
                                                "digests/2026-11-16 morning submission.md"])

    def test_a_night_reads_only_its_own_lines_of_night_log(self):
        app = self.app()
        rid = self.start(app, kind="night")["run_id"]
        night_log = self.runs / "balcony" / "night.log"
        night_log.parent.mkdir(parents=True, exist_ok=True)
        night_log.write_text("2026-11-15 02:30 s1 round 2: section 1 0.80 (5 flagged); Pangram $0.20\n"
                             "plain output from last night\n"
                             "2026-11-16 02:00:05 the night run of 2026-11-16 starts\n"
                             "2026-11-16 02:30 s1 round 3: section 1 0.31 (1 flagged, HUMANIZED); Pangram $0.20\n"
                             "plain output from tonight\n", encoding="utf-8")
        os.utime(night_log, (T + 1800, T + 1800))
        self.clock[0] = T + 1900
        st = app.run_status(rid)
        self.assertEqual(st["scores"], [{"scoring": "s1 round 3", "section": 1, "ai_share": 0.31, "flagged": 1,
                                         "humanized": True}])
        self.assertIn("plain output from tonight", st["last_lines"])
        self.assertNotIn("plain output from last night", st["last_lines"])

    def test_an_empty_report_says_so(self):
        app = self.app()
        rid = self.start(app)["run_id"]
        got = app.run_report(rid)
        self.assertEqual(got["files"], [])
        self.assertIn("hasn't written", got["note"])

    def test_a_bad_run_id_is_refused_before_it_touches_a_path(self):
        for bad in ("../../etc", "pass-1", "", "night-20261116-020000-zzzzzz"):
            with self.assertRaises(Refused, msg=bad):
                self.app().run_status(bad)
        with self.assertRaises(Refused):
            self.app().run_status("pass-20261116-020000-abcdef")                    # well formed, not there


# ---------------------------------------------------------------------- stop_run
class StopRun(Base):
    def test_stop_ends_the_tree_of_a_run_this_server_started(self):
        app = self.app()
        rid = self.start(app)["run_id"]
        got = app.stop_run(rid)
        self.assertEqual(self.killed, [4242])
        self.assertEqual((got["state"], got["ended_pids"]), ("stopped", [4242, 4243]))
        self.assertEqual(app.run_status(rid)["state"], "stopped")

    def test_stop_after_a_restart_ends_the_tree_by_its_pid(self):
        rid = self.start(self.app())["run_id"]
        self.alive[4242] = True
        got = self.app(port_fn=lambda: True).stop_run(rid)
        self.assertEqual(self.killed, [4242])
        self.assertTrue(any("llama server" in n for n in got["notes"]))

    def test_a_finished_run_isnt_stopped_again(self):
        app = self.app()
        rid = self.start(app)["run_id"]
        self.proc.rc = 0
        (self.run_dir(rid) / "exit").write_text("0", encoding="utf-8")
        got = app.stop_run(rid)
        self.assertEqual(self.killed, [])
        self.assertEqual(got["state"], "finished")

    def test_a_reused_pid_is_never_killed(self):
        rid = self.start(self.app())["run_id"]
        self.alive[4242] = False                                                   # the pid is some other process now
        got = self.app().stop_run(rid)
        self.assertEqual(self.killed, [])
        self.assertEqual(got["state"], "lost")


# ---------------------------------------------------------------------- the wrapper and plain errors
class Wrapper(unittest.TestCase):
    def test_the_wrapper_writes_the_exit_code(self):
        # a real child, started the way start_run starts one (rs.spawn): the wrapper runs python -m engine.night
        # --help, which only prints its usage and exits 0
        with tempfile.TemporaryDirectory() as d:
            exit_file, log = Path(d) / "exit", Path(d) / "out.log"
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
            with log.open("ab") as out:
                proc = rs.spawn(rs.child_cmd(exit_file, "engine.night", ["--help"]), out, env, REPO)
            began = rs.create_time(proc.pid)
            self.assertTrue(began is None or rs.same_process(proc.pid, began))
            self.assertEqual(proc.wait(120), 0)
            self.assertIn("--job-dir", log.read_text(encoding="utf-8"))
            self.assertEqual(exit_file.read_text(encoding="utf-8"), "0")
            self.assertFalse(rs.same_process(proc.pid, began))                  # gone, so no longer the run

    def test_the_wrapper_runs_only_the_engine_s_two_modules(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(child.main([str(Path(d) / "exit"), "-m", "os"]), 2)
            self.assertFalse((Path(d) / "exit").exists())

    def test_plain_error_strips_the_repo_and_keys(self):
        app = App(Settings(env={"PANGRAM_API_KEY": KEY}, config={}), repo=REPO)
        msg = plain_error(app, FileNotFoundError(f"{REPO / 'jobs' / 'x' / 'job.toml'} with {KEY}"))
        self.assertNotIn(str(REPO), msg)
        self.assertNotIn(KEY, msg)
        self.assertTrue(msg.startswith("FileNotFoundError: <repo>"))


# ---------------------------------------------------------------------- through the SDK
@unittest.skipUnless(importlib.util.find_spec("mcp"), "the mcp package isn't installed (requirements.txt)")
class ThroughTheSdk(Base):
    def call(self, coro_fn):
        from mcp import Client
        from mcp_server.server import build

        async def go():
            async with Client(build(self.app())) as c:
                return await coro_fn(c)
        logging.disable(logging.INFO)          # the SDK logs each refused call at INFO
        try:
            return asyncio.run(go())
        finally:
            logging.disable(logging.NOTSET)

    def test_the_sdk_lists_the_seven_tools_with_their_schemas(self):
        tools = {t.name: t for t in self.call(lambda c: c.list_tools()).tools}
        self.assertEqual(sorted(tools), ["list_jobs", "review_queue", "run_report", "run_status", "score_text",
                                         "start_run", "stop_run"])
        sr = tools["start_run"].input_schema
        self.assertEqual(sorted(sr["required"]), ["job", "max_pangram_usd"])
        self.assertEqual(sr["properties"]["kind"]["enum"], ["pass", "night"])
        self.assertEqual(tools["score_text"].input_schema["properties"]["detector"]["enum"], ["open", "pangram"])
        self.assertEqual(tools["score_text"].input_schema["required"], ["text"])
        for name in ("run_status", "run_report", "stop_run"):
            self.assertEqual(tools[name].input_schema["required"], ["run_id"])
        for t in tools.values():
            self.assertTrue(t.description)

    def test_a_call_without_a_ceiling_comes_back_as_an_error(self):
        self.proc = FakeProc()
        r = self.call(lambda c: c.call_tool("start_run", {"job": "balcony"}))
        self.assertTrue(r.is_error)
        self.assertIn("max_pangram_usd", r.content[0].text)
        self.assertEqual(self.spawned, [])

    def test_a_refusal_comes_back_as_one_plain_line(self):
        r = self.call(lambda c: c.call_tool("review_queue", {"job": "../../outside"}))
        self.assertTrue(r.is_error)
        self.assertNotIn("Traceback", r.content[0].text)
        self.assertIn("isn't inside", r.content[0].text)


if __name__ == "__main__":
    unittest.main()
