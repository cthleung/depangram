"""Tests for the night run (engine/night.py): the order of the steps, each guard, a failing pass that still submits,
the deadline that ends the pass's process tree, the night's Pangram ceiling handed to each step as what's left, the
late start, the Claude ceiling that skips the pass and turns the repair step off, and a dry run that touches nothing.
Every step is a fake (the clock, the ledger, the child processes, main first, the submission, Discord), except one
test that ends a real process tree of two sleeping Pythons. Run from the repo root:

    python -m unittest discover tests
"""
import datetime as dt
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

from engine import night

NIGHT = {"branch": "work", "sections": [2, 3, 5], "rounds": 1, "max_pangram_usd": 3.50,
         "max_claude_usd": 5.00, "latest_start": "05:00", "pass_until": "06:00"}
AFTER = types.SimpleNamespace(name="garden-notes")          # the job main first hands on (loaded again after a sync)


class Clock:
    """The night's clock: sleep moves it on without waiting."""

    def __init__(self, hhmm="02:00"):
        h, m = (int(x) for x in hhmm.split(":"))
        self.t = dt.datetime(2026, 11, 16, h, m)

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += dt.timedelta(seconds=seconds)


class Ledger:
    """Spend rows at clock times; spent(since) sums the rows at or after `since`. One row before the night started
    stands for the day's earlier spend, which the night's ceiling must not count."""

    def __init__(self, clock, week=0.40, ceiling=0.90):
        self.clock, self.week_at, self.ceiling = clock, week, ceiling
        self.rows = [(clock.now().timestamp() - 3600, 5.0, 1.0)]

    def add(self, pangram, claude=0.0):
        self.rows.append((self.clock.now().timestamp(), pangram, claude))

    def spent(self, since):
        got = [r for r in self.rows if r[0] >= since]
        return sum(r[1] for r in got), sum(r[2] for r in got)

    def week(self):
        return self.week_at, self.ceiling


class Proc:
    """A child process: it ends with `rc` after `polls` looks, spending on the ledger as it ends; polls=None never
    ends on its own."""

    def __init__(self, t, cmd, rc=0, polls=1, pangram=0.0, claude=0.0):
        self.t, self.cmd, self.rc, self.polls, self.pangram, self.claude = t, cmd, rc, polls, pangram, claude
        self.pid, self.seen, self.done = 4242, 0, False

    def poll(self):
        if self.polls is None:
            return None
        self.seen += 1
        if self.seen < self.polls:
            self.t.clock.sleep(600)
            return None
        if not self.done:
            self.done = True
            self.t.ledger.add(self.pangram, self.claude)
            self.t.clock.sleep(60)            # the next step starts a minute later, so its spend is its own
        return self.rc


class NightBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.events, self.posts, self.killed, self.out = [], [], [], []

    def night(self, hhmm="02:00", branch="work", busy=None, week=0.40, dry=False, children=None,
              early=None, port=False, spawn_error=False, stop_at=None):
        self.clock = Clock(hhmm)
        self.ledger = Ledger(self.clock, week=week)
        self.children = list(children if children is not None else [dict(pangram=2.25, claude=0.40),
                                                                    dict(pangram=0.30)])
        self.early = early or {"stopped": None, "pulled": None, "state": "same", "commits": [], "problems": [],
                               "synced": False, "job": AFTER}
        self.stop = self.d / "STOP"
        settings = types.SimpleNamespace(config={"night": dict(NIGHT)}, get=lambda k, d="": d, runs_dir=self.d)
        job = types.SimpleNamespace(name="garden-notes")

        def spawn(cmd, out):
            if spawn_error:
                raise OSError("no python")
            self.events.append(("spawn", cmd[3:]))
            if stop_at == len([e for e in self.events if e[0] == "spawn"]):
                self.stop.write_text("")
            return Proc(self, cmd, **self.children.pop(0))

        def kill(proc):
            self.killed.append(proc.cmd[3:])
            self.events.append(("kill", self.clock.now().strftime("%H:%M")))
            return [proc.pid, 4243], []

        def main_first(job, repo=None, dry=False, settings=None, log=None):
            self.events.append(("main first", dry))
            self.main_first_settings = settings
            return self.early

        def submit(job, **kw):
            self.events.append(("submit", job))
            self.submitted = kw
            return {"digest": "the digest", "posted_line": "Posted on Discord: yes", "stopped": None, "problems": []}

        return night.Night(job, settings, dry=dry, repo=self.d, now=self.clock.now, sleep=self.clock.sleep,
                           ledger=lambda: self.ledger, busy_fn=lambda: busy, branch_fn=lambda: branch,
                           spawn_fn=spawn, kill_fn=kill, port_fn=lambda: port, main_first=main_first, submit=submit,
                           notify=lambda s, text: self.posts.append(text) or True, stop_file=self.stop,
                           log_path=self.d / "night.log", out=self.out.append)

    def spawns(self):
        return [e[1] for e in self.events if e[0] == "spawn"]

    def flag(self, args, name):
        return args[args.index(name) + 1]

    def lines(self):
        return "\n".join(self.submitted["night"]["lines"])


class NightTest(NightBase):
    def test_the_steps_run_in_order_and_each_gets_what_s_left_of_the_night_s_pangram(self):
        res = self.night().run()
        self.assertEqual([e[0] for e in self.events], ["main first", "spawn", "spawn", "submit"])     # pass: the order
        self.assertIs(self.events[-1][1], AFTER)                                                 # main first's job
        pass_args, final_args = self.spawns()
        self.assertEqual(pass_args, ["--job", "garden-notes", "--section", "2,3,5", "--purpose", "loop", "--rounds", "2",
                                     "--max-usd", "3.50", "--max-claude-usd", "5.00"])       # the day's earlier $5 not counted
        self.assertIs(self.main_first_settings.config["night"]["rounds"], 1)                    # main first gets the settings
        self.assertEqual(final_args[:1], ["--final"])
        self.assertEqual((self.flag(final_args, "--purpose"), self.flag(final_args, "--max-usd")), ("loop", "1.25"))
        self.assertIs(self.submitted["early"], self.early)
        self.assertEqual((self.submitted["repair_mode"], self.submitted["dry"]), (None, False))
        text = self.lines()
        self.assertIn("- The paragraph pass (sections 2, 3, 5, 1 round, Pangram ceiling $3.50, Claude $5.00): finished"
                      " at 02:01; Pangram $2.25, Claude $0.40.", text)
        self.assertIn("- The finals (sections 2, 3, 5, under $1.25): finished at 02:02; Pangram $0.30", text)
        self.assertIn("- The night before the submission: Pangram $2.55 of $3.50, Claude $0.40.", text)
        self.assertEqual(self.submitted["night"]["short"], "pass ran, Pangram $2.25")
        self.assertEqual(night.exit_code(res), 0)
        log = (self.d / "night.log").read_text(encoding="utf-8")
        self.assertIn("2026-11-16 02:00:00 the night run of 2026-11-16 starts", log)                # stamped, in the log
        self.assertIn("Posted on Discord: yes", log)
        self.assertIn("the digest", self.out)

    def test_the_pass_gets_one_more_round_than_the_night_counts_since_regrow_counts_the_arm_sets(self):
        n = self.night()
        n.cfg["rounds"] = 2
        n.run()
        self.assertEqual(self.flag(self.spawns()[0], "--rounds"), "3")                              # flag: two new rounds
        self.assertIn("sections 2, 3, 5, 2 rounds,", self.lines())
        self.events.clear()
        self.night().run()
        self.assertEqual(self.flag(self.spawns()[0], "--rounds"), "2")                              # pass: one new round
        self.assertNotIn("--rounds", self.spawns()[1])                                              # the finals draw nothing

    def test_the_code_default_for_the_pass_s_claude_is_the_regrow_ceiling(self):
        self.assertEqual(night.config(None)["max_claude_usd"], 5.0)
        self.assertEqual(night.config(types.SimpleNamespace(config={"night": {"max_claude_usd": 1.0}}))["max_claude_usd"],
                         1.0)

    def test_a_pass_that_spends_the_whole_ceiling_leaves_the_finals_nothing(self):
        self.night(children=[dict(pangram=3.50), dict()]).run()
        self.assertEqual(self.flag(self.spawns()[1], "--max-usd"), "0.00")                          # flag: none left

    def test_another_branch_stops_the_night_before_anything(self):
        res = self.night(branch="main").run()
        self.assertEqual(self.events, [])                                                           # flag: nothing at all
        self.assertIn("this checkout is on main, not work", res["halted"])
        self.assertIn("stopped: this checkout is on main", self.posts[0])
        self.assertEqual(night.exit_code(res), 1)

    def test_the_stop_file_stops_the_night_before_any_spend(self):
        n = self.night()
        self.stop.write_text("")
        res = n.run()
        self.assertEqual(self.events, [])                                                           # flag: no submission
        self.assertIn("the STOP file is there", res["halted"])
        self.assertEqual(len(self.posts), 1)

    def test_the_stop_file_made_during_the_pass_stops_the_night_before_the_finals(self):
        res = self.night(stop_at=1).run()
        self.assertEqual([e[0] for e in self.events], ["main first", "spawn"])                      # flag: no finals, no submit
        self.assertEqual(res["halted"], "the STOP file")
        self.assertIn("The STOP file was there before the finals", self.posts[0])

    def test_another_run_on_the_gpu_skips_the_pass_and_the_finals_and_still_submits(self):
        self.night(busy="python -m engine.run is running (pid 77)").run()
        self.assertEqual([e[0] for e in self.events], ["main first", "submit"])                     # flag: no child
        self.assertIn("- The paragraph pass and the finals: skipped, python -m engine.run is running (pid 77).",
                      self.lines())
        self.assertEqual(self.submitted["night"]["short"], "pass skipped (python -m engine.run is running)")

    def test_a_late_start_skips_the_pass_and_does_the_submission_only(self):
        self.night(hhmm="05:00").run()
        self.assertEqual([e[0] for e in self.events], ["main first", "submit"])                     # flag: at 05:00
        self.assertIn("skipped, a late start (05:00, at or after [night] latest_start 05:00)", self.lines())
        self.events.clear()
        self.night(hhmm="04:59").run()
        self.assertEqual([e[0] for e in self.events], ["main first", "spawn", "spawn", "submit"])  # pass: 04:59 runs

    def test_the_claude_ceiling_skips_the_pass_and_turns_the_repair_step_off(self):
        self.night(week=0.90).run()
        self.assertEqual([e[0] for e in self.events], ["main first", "submit"])                     # flag: at the ceiling
        self.assertEqual(self.submitted["repair_mode"], "off")
        self.assertIn("- The repair step: off tonight, your Claude week is at 90%, at or past [claude] total_ceiling"
                      " 90%.", self.lines())
        self.events.clear()
        self.night(week=0.89).run()
        self.assertEqual(self.submitted["repair_mode"], None)                                       # pass: under it

    def test_a_failing_pass_still_runs_the_finals_and_submits(self):
        res = self.night(children=[dict(rc=1, pangram=0.50), dict()]).run()
        self.assertEqual([e[0] for e in self.events], ["main first", "spawn", "spawn", "submit"])
        self.assertIn("ended with exit code 1 at 02:01 (its log: regrow.log in the runs folder); Pangram $0.50",
                      self.lines())
        self.assertIn("- Problem: the paragraph pass ended with exit code 1", self.lines())
        self.assertEqual(self.flag(self.spawns()[1], "--max-usd"), "3.00")
        self.assertEqual(night.exit_code(res), 1)
        self.events.clear()
        self.night(spawn_error=True).run()                                                          # a child that can't start
        self.assertEqual([e[0] for e in self.events], ["main first", "submit"])
        self.assertIn("didn't start: OSError: no python", self.lines())

    def test_a_pass_still_going_at_pass_until_has_its_tree_ended_and_the_night_carries_on(self):
        res = self.night(hhmm="04:30", children=[dict(polls=None), dict(pangram=0.25)], port=True).run()
        self.assertEqual([e[0] for e in self.events], ["main first", "spawn", "kill", "spawn", "submit"])
        self.assertEqual(self.events[2], ("kill", "06:00"))                                          # flag: at the deadline
        self.assertEqual(self.killed, [self.spawns()[0]])
        text = self.lines()
        self.assertIn("still going at [night] pass_until 06:00, so its process tree was ended at 06:00", text)
        self.assertIn("the llama server still answers on port 8079 after the paragraph pass was ended", text)
        self.assertEqual(self.submitted["night"]["short"], "pass ended at 06:00; 1 night problem, see the report")
        self.assertEqual(res["steps"]["pass"]["outcome"], "killed")
        self.events.clear()
        self.night(hhmm="04:30").run()
        self.assertNotIn("kill", [e[0] for e in self.events])                                       # pass: it ended in time

    def test_a_kill_that_tore_the_pass_s_bank_says_so_and_the_next_load_goes_on_without_it(self):
        from engine.job import Bank
        job = types.SimpleNamespace(name="garden-notes", dir=self.d)
        early = {"stopped": None, "pulled": None, "state": "same", "commits": [], "problems": [], "synced": False,
                 "job": job}
        bank = self.d / "regrow bank.jsonl"
        bank.write_bytes(b'{"kind": "draw"}\n{"kind": "dr')
        self.night(hhmm="04:30", children=[dict(polls=None), dict()], early=early).run()
        self.assertIn("- Problem: the paragraph pass was ended while it wrote regrow bank.jsonl: the file's last line"
                      " wasn't whole, so the next step loads the bank without it, and the bank's first write cuts it"
                      " off (the text is kept in regrow bank.jsonl.torn)", self.lines())           # flag: in the digest
        self.assertEqual(bank.read_bytes(), b'{"kind": "draw"}\n{"kind": "dr')                     # the night writes nothing
        b = Bank(bank)
        self.assertEqual((b.entries, b.torn), ([{"kind": "draw"}], '{"kind": "dr'))                # and the bank loads
        bank.write_bytes(b'{"kind": "draw"}\n')
        self.night(hhmm="04:30", children=[dict(polls=None), dict()], early=early).run()
        self.assertNotIn("wasn't whole", self.lines())                                              # pass: whole

    def test_a_stop_in_main_first_skips_the_pass_and_the_submission_still_runs(self):
        early = {"stopped": "this branch's copy of the guide has changes that aren't committed", "pulled": None,
                 "state": "same", "commits": [], "problems": [], "synced": False, "job": AFTER}
        self.night(early=early).run()
        self.assertEqual([e[0] for e in self.events], ["main first", "submit"])                     # flag
        self.assertIn("- Main first: stopped, this branch's copy of the guide has changes", self.lines())
        self.assertIn("skipped, main first stopped", self.lines())
        pulled = dict(early, stopped=None, pulled="Main's copy was pulled and synced in.", commits=["abc1234"],
                      synced=True)
        self.events.clear()
        self.night(early=pulled).run()
        self.assertIn("- Main first: Main's copy was pulled and synced in. Committed as abc1234.", self.lines())
        self.assertEqual(len(self.spawns()), 2)                                                     # pass: the pass runs

    def test_a_dry_run_spends_writes_commits_and_posts_nothing(self):
        res = self.night(dry=True).run()
        self.assertEqual([e[0] for e in self.events], ["main first", "submit"])                     # no child at all
        self.assertEqual(self.events[0], ("main first", True))
        self.assertTrue(self.submitted["dry"])
        self.assertFalse((self.d / "night.log").exists())                                           # flag: no log file
        self.assertEqual(self.posts, [])
        self.assertIn("- The paragraph pass (sections 2, 3, 5, 1 round, Pangram ceiling $3.50, Claude $5.00): would run"
                      " python -m engine.regrow --job garden-notes --section 2,3,5 --purpose loop --rounds 2 --max-usd 3.50"
                      " --max-claude-usd 5.00.", self.lines())
        self.assertIn("would run python -m engine.regrow --final", self.lines())
        self.assertEqual(res["halted"], None)
        self.night(dry=True, branch="main").run()
        self.assertEqual(self.posts, [])                                                            # a dry halt posts nothing


class Settings_(NightBase):
    """Where the night's branch, sections and STOP file come from when config.toml leaves them out."""

    def bare(self, night_cfg, git_cfg=None, job=None, branch=NIGHT["branch"]):
        n = self.night(branch=branch)
        n.settings = types.SimpleNamespace(config={"night": night_cfg, "git": git_cfg or {}}, get=lambda k, d="": d,
                                           runs_dir=self.d)
        n.cfg = night.config(n.settings)
        if job is not None:
            n.job = self.early["job"] = job                     # main first hands the same job on
        return n

    def test_the_sections_come_from_the_job_when_config_has_none(self):
        cfg = {k: v for k, v in NIGHT.items() if k not in ("sections", "branch")}
        job = types.SimpleNamespace(name="doc", night_sections=[2, 3], sections=[1, 2, 3])
        self.bare(cfg, job=job).run()
        self.assertEqual(self.flag(self.spawns()[0], "--section"), "2,3")                       # pass: the job's own
        self.events.clear()
        self.bare(cfg, job=types.SimpleNamespace(name="doc", sections=[1, 2])).run()
        self.assertEqual(self.flag(self.spawns()[0], "--section"), "1,2")                       # its whole scope
        self.events.clear()
        self.bare(dict(cfg, sections=[5]), job=job).run()
        self.assertEqual(self.flag(self.spawns()[0], "--section"), "5")                         # config wins
        self.events.clear()
        res = self.bare(cfg, job=types.SimpleNamespace(name="doc", sections=[])).run()
        self.assertEqual(self.spawns(), [])                                                      # flag: none at all
        self.assertIn("no sections for the night", "\n".join(self.submitted["night"]["lines"]))
        self.assertIsNone(res["halted"])

    def test_the_branch_guard_with_no_work_branch_set(self):
        cfg = {k: v for k, v in NIGHT.items() if k != "branch"}
        job = types.SimpleNamespace(name="doc", sections=[1])
        self.assertIsNone(self.bare(cfg, job=job, branch="main").run()["halted"])               # pass: any branch
        res = self.bare(cfg, {"publish": True}, job=job, branch="main").run()
        self.assertIn("the publish branch", res["halted"])                                       # flag: publishing
        res = self.bare(cfg, job=job, branch="HEAD").run()
        self.assertIn("detached", res["halted"])                                                 # flag: no branch
        res = self.bare(cfg, {"work_branch": "nightly"}, job=job, branch="main").run()
        self.assertIn("not nightly", res["halted"])                                              # [git] work_branch
        job.branch = "from-the-job"
        self.assertIn("not from-the-job", self.bare(cfg, job=job, branch="main").run()["halted"])

    def test_a_job_folder_anywhere_goes_to_the_pass_as_job_dir(self):
        cfg = {k: v for k, v in NIGHT.items() if k != "branch"}
        job = types.SimpleNamespace(name="doc", sections=[1])
        n = self.bare(cfg, job=job)
        n.job_dir = self.d / "elsewhere"
        n.run()
        spawns = self.spawns()
        self.assertEqual(len(spawns), 2)                                                         # pass and finals
        for args in spawns:
            self.assertEqual(self.flag(args, "--job-dir"), str(self.d / "elsewhere"))
            self.assertNotIn("--job", args)
        self.events.clear()
        n = self.bare(cfg, job=job)                                                              # no folder given
        n.run()
        self.assertTrue(all(self.flag(args, "--job") == "doc" for args in self.spawns()))

    def test_the_stop_file_defaults_to_runs_dir(self):
        s = types.SimpleNamespace(config={"night": dict(NIGHT)}, get=lambda k, d="": {"RUNS_DIR": str(self.d)}.get(k, d),
                                  runs_dir=self.d)
        n = night.Night(types.SimpleNamespace(name="doc"), s, repo=self.d)
        self.assertEqual(n.stop_file, self.d / "STOP")


class KillTreePosix(unittest.TestCase):
    """The POSIX branch of kill_tree with the platform faked: the child's own process group gets SIGTERM, then
    SIGKILL for what outlived the wait; a child in the night's own group is signalled one process at a time."""

    def run_kill(self, child_group, survivors):
        from unittest import mock
        sent = []

        class P:
            def __init__(self, pid):
                self.pid = pid

            def children(self, recursive=False):
                return [P(11), P(12)]

            def send_signal(self, sig):
                sent.append(("proc", self.pid, sig))

        rounds = iter([survivors, []])

        def wait_procs(procs, timeout=None):
            still = next(rounds)
            alive = [p for p in procs if p.pid in still]
            return [p for p in procs if p not in alive], alive

        fake = types.SimpleNamespace(Process=P, NoSuchProcess=ProcessLookupError, wait_procs=wait_procs)
        with mock.patch.object(night, "POSIX", True), mock.patch.dict(sys.modules, {"psutil": fake}), \
                mock.patch.object(night.os, "getpgid", lambda pid: child_group if pid == 10 else 1, create=True), \
                mock.patch.object(night.os, "killpg", lambda g, sig: sent.append(("group", g, sig)), create=True), \
                mock.patch.object(night.signal, "SIGKILL", 9, create=True), \
                mock.patch.object(night.signal, "SIGTERM", 15):
            ended, alive = night.kill_tree(types.SimpleNamespace(pid=10), wait=0)
        return ended, alive, sent

    def test_the_group_gets_sigterm_then_sigkill_for_what_outlived_it(self):
        ended, alive, sent = self.run_kill(child_group=10, survivors=[12])
        self.assertEqual((ended, alive), ([10, 11, 12], []))
        self.assertEqual(sent[0], ("group", 10, 15))                                             # the group first
        self.assertIn(("proc", 12, 9), sent)                                                     # flag: the survivor
        self.assertNotIn(("proc", 11, 9), sent)                                                  # pass: gone already
        self.assertEqual([x for x in sent if x[0] == "group"], [("group", 10, 15), ("group", 10, 9)])

    def test_a_child_in_the_night_s_own_group_never_gets_a_group_signal(self):
        ended, alive, sent = self.run_kill(child_group=1, survivors=[])
        self.assertEqual([x for x in sent if x[0] == "group"], [])
        self.assertEqual(sorted(x[1] for x in sent), [10, 11, 12])

    def test_spawn_starts_its_own_session_on_posix(self):
        from unittest import mock
        got = {}
        with mock.patch.object(night, "POSIX", True), \
                mock.patch.object(night.subprocess, "Popen", lambda cmd, **kw: got.update(kw) or "proc"):
            night.spawn(["x"], None)
        self.assertTrue(got["start_new_session"])
        self.assertNotIn("creationflags", got)


class KillTree(unittest.TestCase):
    def test_the_child_and_the_process_it_started_are_both_ended(self):
        import psutil
        code = ("import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', 'import time;"
                " time.sleep(120)']); print(p.pid, flush=True); time.sleep(120)")
        proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        grandchild = int(proc.stdout.readline())
        self.assertTrue(psutil.pid_exists(grandchild))
        ended, alive = night.kill_tree(proc)
        self.assertIn(proc.pid, ended)
        self.assertIn(grandchild, ended)
        self.assertEqual(alive, [])
        proc.wait(10)
        proc.stdout.close()
        self.assertFalse(psutil.pid_exists(grandchild))


if __name__ == "__main__":
    unittest.main()
