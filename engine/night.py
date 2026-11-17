"""The night run: the paragraph pass and the finals while you sleep, then the morning submission, so the newest
write-back is committed (and published, when [git] publish is on) by morning with a digest sent, without asking.
A scheduler starts it: `night task.ps1` registers a Windows Task Scheduler task, and `scheduling/` holds a cron
line and a systemd timer for Linux.

    python -m engine.night          # do it
    python -m engine.night --dry    # say what it would do: nothing spent, written, committed or posted

In order, each step logged with its time to RUNS_DIR/<job>/night.log and to stdout. A step's failure is caught,
goes in the digest and never stops the steps after it, unless a guard says so:

  1. The guards. A detached checkout, a checkout on another branch than the work branch ([night] branch, else
     [git] work_branch, else the job's branch in job.toml; none set means any branch), or one on the publish branch
     while publishing is on, stops the night (nothing committed, nothing published), and so does the STOP file the
     loop honors (STOP_FILE, default STOP in RUNS_DIR), before any spend: the submission doesn't run either. The
     STOP file is looked for again before the finals and before the submission, and stops the night there too.
     Another depangram run holding the llama server's port, or a python -m engine.run, engine.regrow or
     engine.night alive, skips the paragraph pass and the finals; so does a start at or after [night]
     latest_start (a computer that was off at the start time and ran the task late). Your week at or past
     [claude] total_ceiling (the last reading, as engine/budget.py reads it for every Claude call) skips the pass
     and the finals and turns the repair step off for the night.
  2. Main first (engine/morning.py's main_first): with publishing on, when you edited the publish branch's copy,
     your copy is pulled, synced in and committed before the pass, so the pass works on your newest text. A stop
     there skips the pass and the finals.
  3. The paragraph pass: python -m engine.regrow on the night's sections ([night] sections, else the job's
     night_sections, else its whole scope) with --purpose loop, --rounds [night] rounds plus one (regrow counts
     the arm-set round in it), --max-usd (what's left of the night's Pangram ceiling) and --max-claude-usd [night]
     max_claude_usd, in a child process. Still going at [night] pass_until, its whole process tree is ended (the
     llama server it started included) and the night carries on.
  4. The finals: python -m engine.regrow --final on the same sections, in a child process, under what's left of the
     night's Pangram ceiling. They run whenever the pass started, since a pass that was ended never wrote its picks.
     A finals run still going after FINAL_MINUTES is ended the same way.
  5. The submission (engine/morning.py's submit, with the night's main-first result), the night's own lines at the
     top of its digest. It prints and logs whether the notifier took the digest.

The night's Pangram ceiling ([night] max_pangram_usd) is one number for steps 3 and 4 together: what the night has
spent is read from the ledger (pangram_calls, purpose loop, since the night started, Budget.pangram_spent) before
each, and each gets only what's left. The repair step keeps its own [repair] ceilings.

Ending a process tree: on Windows each process is killed and taskkill /T /F gets a second go at any still alive;
on Linux and macOS the child starts in its own process group, which gets SIGTERM, then SIGKILL for anything still
alive after the wait.
"""
import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .settings import SANDBOX, setting_path

POSIX = os.name != "nt"       # Linux and macOS: process groups and signals; Windows: taskkill and no window
DEFAULTS = {"branch": "", "sections": None, "rounds": 1, "max_pangram_usd": 3.5,
            "max_claude_usd": 5.0, "latest_start": "05:00", "pass_until": "06:00"}
PORT = 8079                  # LlamaServer's port, the one every depangram run starts its server on
BUSY_MODULES = ("engine.run", "engine.regrow", "engine.night")
FINAL_MINUTES = 30           # the finals draw nothing and start no server; one still going after this is ended
POLL_SECONDS = 20            # how often the night looks at a child process
TORN_FILES = ("regrow bank.jsonl",)   # what the pass appends to as it goes, looked at after a kill


def config(settings):
    cfg = dict(DEFAULTS)
    cfg.update((settings.config.get("night") if settings is not None else None) or {})
    return cfg


def at(day_dt, hhmm):
    """That day at "HH:MM"."""
    h, m = (int(x) for x in str(hhmm).split(":"))
    return day_dt.replace(hour=h, minute=m, second=0, microsecond=0)


def money(x):
    return f"${x:.2f}"


# ---------------------------------------------------------------------- the real parts (tests pass fakes)
def branch_of(repo):
    from .main_copy import git
    return git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo)


def busy(port=PORT):
    """Why the paragraph pass can't have the GPU tonight, or None: something answers on the llama server's port, or
    another depangram run is alive (this process and the processes that started it don't count)."""
    import psutil
    import requests
    try:
        requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        return f"something already answers on the llama server's port {port}, so another run holds it"
    except requests.RequestException:
        pass
    me = psutil.Process()
    mine = {me.pid} | {p.pid for p in me.parents()}
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = p.info["cmdline"] or []
        except (psutil.Error, KeyError):
            continue
        if p.info["pid"] in mine:
            continue
        for i, x in enumerate(cmd[:-1]):
            if x == "-m" and cmd[i + 1] in BUSY_MODULES:
                return f"python -m {cmd[i + 1]} is running (pid {p.info['pid']})"
    return None


def port_held(port=PORT):
    import requests
    try:
        requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        return True
    except requests.RequestException:
        return False


def spawn(cmd, out):
    """A child process in the depangram folder, its output into `out` (a file, never a pipe nobody reads). On Windows
    it has no window of its own; elsewhere it starts its own session, so kill_tree can end its process group without
    touching the night's own."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    kw = {"start_new_session": True} if POSIX else {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return subprocess.Popen(cmd, cwd=str(SANDBOX), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            env=env, **kw)


def kill_tree(proc, wait=30):
    """End a child and everything it started (the venv's python can start the real one as its child, and the pass
    starts llama-server): the children are listed before the parent goes, then all are ended. Returns (pids ended,
    pids still alive). Windows: each is killed, and taskkill /T /F gets a second go at any still alive. POSIX: the
    child's process group gets SIGTERM, then SIGKILL for anything still alive after `wait` (kill_tree_posix)."""
    import psutil
    try:
        parent = psutil.Process(proc.pid)
        procs = [parent] + parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return [], []
    if POSIX:
        return kill_tree_posix(proc, procs, wait)
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=wait)
    for p in alive:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
    _, alive = psutil.wait_procs(alive, timeout=wait)
    return [p.pid for p in procs], [p.pid for p in alive]


def kill_tree_posix(proc, procs, wait=30):
    """The POSIX branch of kill_tree: SIGTERM to the child's process group (when it has its own, never the night's
    group) and to each listed process, a wait, then SIGKILL the same way for whatever is still alive."""
    import psutil
    try:
        group = os.getpgid(proc.pid)
        if group == os.getpgid(0):
            group = None                     # the child shares the night's group: signal the processes one by one
    except (OSError, AttributeError):
        group = None

    def hit(sig, targets):
        if group is not None:
            try:
                os.killpg(group, sig)
            except OSError:
                pass
        for p in targets:
            try:
                p.send_signal(sig)
            except psutil.NoSuchProcess:
                pass

    hit(signal.SIGTERM, procs)
    _, alive = psutil.wait_procs(procs, timeout=wait)
    if alive:
        hit(signal.SIGKILL, alive)
        _, alive = psutil.wait_procs(alive, timeout=wait)
    return [p.pid for p in procs], [p.pid for p in alive]


class Ledger:
    """What the night has spent, from the one SQLite ledger through engine/budget.py: Pangram (purpose loop) and
    Claude (every call's estimated dollars) since a time, and whether your week is at the Claude ceiling."""

    def __init__(self, settings):
        from . import db
        from .budget import Budget
        from .schedule import Clock, Schedule
        self.conn = db.connect(settings.db_path)
        self.budget = Budget(settings, self.conn, Clock(), Schedule(settings.config))

    def spent(self, since):
        pangram = self.budget.pangram_spent(since, "loop")
        claude = self.conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM claude_calls WHERE ts >= ?",
                                   (since,)).fetchone()[0]
        return float(pangram), float(claude)

    def week(self):
        """(your week as the last reading says, the ceiling)."""
        return float(self.budget.real_total()), float(self.budget.cl["total_ceiling"])


# ---------------------------------------------------------------------- the night
class Night:
    def __init__(self, job, settings, dry=False, *, repo=None, now=None, sleep=None, ledger=None, busy_fn=None,
                 branch_fn=None, spawn_fn=None, kill_fn=None, port_fn=None, main_first=None, submit=None,
                 notify=None, stop_file=None, log_path=None, out=print):
        from . import morning
        self.job, self.settings, self.dry, self.repo = job, settings, dry, repo
        self.cfg = config(settings)
        self.now = now or dt.datetime.now
        self.sleep = sleep or time.sleep
        self.ledger_fn = ledger or (lambda: Ledger(settings))
        self.busy_fn, self.branch_fn = busy_fn or busy, branch_fn or (lambda: branch_of(self.checkout()))
        self.spawn_fn, self.kill_fn, self.port_fn = spawn_fn or spawn, kill_fn or kill_tree, port_fn or port_held
        self.main_first = main_first or morning.main_first
        self.submit = submit or morning.submit
        self.notify = notify
        self.stop_file = Path(stop_file or setting_path("STOP_FILE", settings))
        self.log_path = Path(log_path) if log_path else settings.runs_dir / job.name / "night.log"
        self.out = out
        self.lines, self.problems, self.steps = [], [], {}
        self.short, self.ledger, self.result = [], None, None
        self.start = self.now()
        self.since = self.start.timestamp()

    def checkout(self):
        """The git checkout the night commits in: the repo it was given, else the one holding the job's document."""
        from .main_copy import repo_for
        if self.repo is None:
            self.repo = repo_for(self.job, self.settings)
        return self.repo

    def sections(self):
        """The night's sections: [night] sections, else the job's night_sections, else its whole scope."""
        got = self.cfg.get("sections")
        if not got:
            got = getattr(self.job, "night_sections", None) or getattr(self.job, "sections", None) or []
        return [int(x) for x in got]

    def job_args(self):
        """How the pass's child finds this job: --job-dir FOLDER when the night was given a folder (main() sets
        self.job_dir from python -m engine.night --job-dir), else --job NAME, a folder in jobs/."""
        d = getattr(self, "job_dir", None)
        return ["--job-dir", str(d)] if d else ["--job", self.job.name]

    def guard_branch(self, here):
        """Why this branch can't take the night's commits, or None."""
        from .main_copy import git_config
        from .morning import work_branch_of
        want = work_branch_of(self.job, self.settings)
        g = git_config(self.settings)
        if here == "HEAD":
            return "this checkout is on no branch (a detached HEAD); nothing was committed or published"
        if want and here != want:
            return f"this checkout is on {here}, not {want}; nothing was committed or published"
        if g["publish"] and here == g["publish_branch"]:
            return (f"this checkout is on {here}, the publish branch, and the night only commits on a work branch;"
                    " nothing was committed or published")
        return None

    # ------------------------------------------------------------------ the log
    def say(self, msg):
        line = f"{self.now():%Y-%m-%d %H:%M:%S} {msg}"
        self.out(line)
        if self.dry:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def problem(self, text):
        self.problems.append(text)
        self.say(f"problem: {text}")

    # ------------------------------------------------------------------ the run
    def run(self):
        from .util import redact
        cfg = self.cfg
        self.start = self.now()
        self.since = self.start.timestamp()       # the ledger's rows from here on are the night's
        self.say(f"the night run of {self.start:%Y-%m-%d} starts{' (a dry run)' if self.dry else ''}")
        try:
            here = self.branch_fn()
        except Exception as e:  # noqa: BLE001
            return self.halt(f"the checkout's branch couldn't be read ({type(e).__name__}: {redact(e)[:200]})")
        why = self.guard_branch(here)
        if why:
            return self.halt(why)
        if self.stop_file.exists():
            return self.halt(f"the STOP file is there ({self.stop_file}); delete it to let the night run")
        self.lines.append(f"- Started {self.start:%Y-%m-%d %H:%M} on {here}" + (", a dry run." if self.dry else "."))

        skip = None
        if self.start >= at(self.start, cfg["latest_start"]):
            skip = f"a late start ({self.start:%H:%M}, at or after [night] latest_start {cfg['latest_start']})"
        if not skip:
            try:
                why = self.busy_fn()
            except Exception as e:  # noqa: BLE001
                why = f"the check for another run failed ({type(e).__name__}: {redact(e)[:160]})"
            if why:
                skip = why
        try:
            self.ledger = self.ledger_fn()
        except Exception as e:  # noqa: BLE001
            self.ledger = None
            self.problem(f"the ledger couldn't be read ({type(e).__name__}: {redact(e)[:200]})")
            skip = skip or "the ledger couldn't be read, so the night's Pangram ceiling can't be held"
        repair_off = self.claude_over()
        if repair_off and not skip:
            skip = repair_off

        from .main_copy import git_config
        g = git_config(self.settings)
        pb = g["publish_branch"]
        self.say(f"main first: is there an edit of yours on {pb} to take in?" if g["publish"] else
                 "main first: publishing is off, so only the branch and uncommitted changes are checked")
        try:
            early = self.main_first(self.job, repo=self.repo, dry=self.dry, settings=self.settings, log=self.say)
        except Exception as e:  # noqa: BLE001
            early = None
            self.problem(f"main first failed: {type(e).__name__}: {redact(e)[:200]}")
        if early:
            self.job = early.get("job") or self.job
            if early.get("stopped"):
                self.lines.append(f"- Main first: stopped, {early['stopped']}.")
                skip = skip or "main first stopped, so the pass would work on the wrong text"
            elif early.get("pulled"):
                self.lines.append(f"- Main first: {early['pulled']}"
                                  + (f" Committed as {', '.join(early['commits'])}." if early.get("commits") else ""))
            else:
                self.lines.append(f"- Main first: nothing to pull ({pb}'s copy and this branch's: {early.get('state')})."
                                  if g["publish"] else "- Main first: publishing is off, so there's nothing to pull.")
            self.problems += [f"main first: {p}" for p in early.get("problems") or []]
        self.say(self.lines[-1][2:] if early else "main first: failed")

        secs = ", ".join(map(str, self.sections()))
        if not self.sections():
            skip = skip or "no sections for the night ([night] sections, or the job's sections)"
        if skip:
            self.lines.append(f"- The paragraph pass and the finals: skipped, {skip}.")
            self.short.append(f"pass skipped ({skip.split(' (')[0]})")
            self.say(f"the paragraph pass and the finals: skipped, {skip}")
        else:
            self.paragraph_pass(secs)
            if self.stopped_now("the finals"):
                return self.finish(early, repair_off, submit=False)
            self.finals(secs)
        if self.stopped_now("the submission"):
            return self.finish(early, repair_off, submit=False)
        repair_off = repair_off or self.claude_over(again=True)
        return self.finish(early, repair_off)

    def claude_over(self, again=False):
        """The reason to turn the repair step off (and skip the pass), or None: your week at or past the ceiling."""
        if self.ledger is None:
            return None
        try:
            week, ceiling = self.ledger.week()
        except Exception as e:  # noqa: BLE001
            self.problem(f"your Claude week couldn't be read ({type(e).__name__})")
            return None
        if week >= ceiling:
            why = f"your Claude week is at {week:.0%}, at or past [claude] total_ceiling {ceiling:.0%}"
            self.say(("before the submission: " if again else "") + why + ", so the repair step is off tonight")
            return why
        return None

    def stopped_now(self, before):
        if self.stop_file.exists():
            self.lines.append(f"- The STOP file was there before {before}, so the night stopped there; nothing after"
                              " it ran.")
            self.say(f"the STOP file is there; the night stops before {before}")
            return True
        return False

    def left(self):
        """What's left of the night's Pangram ceiling, from the ledger."""
        pangram, _ = self.ledger.spent(self.since)
        return max(0.0, float(self.cfg["max_pangram_usd"]) - pangram)

    def paragraph_pass(self, secs):
        cfg = self.cfg
        try:
            left = self.left()
        except Exception as e:  # noqa: BLE001
            self.problem(f"the night's Pangram spend couldn't be read ({type(e).__name__})")
            self.lines.append("- The paragraph pass: skipped, the night's Pangram spend couldn't be read.")
            return
        until = at(self.start, cfg["pass_until"])
        if left <= 0:
            self.lines.append(f"- The paragraph pass: skipped, nothing left of the night's {money(cfg['max_pangram_usd'])}.")
            return
        if self.now() >= until:
            self.lines.append(f"- The paragraph pass: skipped, it's past [night] pass_until {cfg['pass_until']}.")
            return
        argv = [*self.job_args(), "--section", ",".join(map(str, self.sections())), "--purpose", "loop",
                # regrow's --rounds counts the arm-set round too (RegrowPass.run's later rounds are range(base + 1,
                # base + rounds), so 1 runs none); [night] rounds counts the new scoring rounds tonight, hence + 1
                "--rounds", str(int(cfg["rounds"]) + 1), "--max-usd", f"{left:.2f}",
                "--max-claude-usd", f"{float(cfg['max_claude_usd']):.2f}"]
        res = self.child("the paragraph pass", argv, until)
        self.steps["pass"] = res
        head = (f"- The paragraph pass (sections {secs}, {cfg['rounds']} round{'' if cfg['rounds'] == 1 else 's'},"
                f" Pangram ceiling {money(left)}, Claude {money(float(cfg['max_claude_usd']))})")
        self.lines.append(head + ": " + self.outcome_text(res, f"at [night] pass_until {cfg['pass_until']}"))
        self.short.append({"dry": "pass would run", "finished": "pass ran", "killed": f"pass ended at {cfg['pass_until']}"}
                          .get(res["outcome"], "pass failed") + (f", Pangram {money(res['pangram'])}"
                                                                 if res.get("pangram") else ""))

    def finals(self, secs):
        try:
            left = self.left()
        except Exception as e:  # noqa: BLE001
            self.problem(f"the night's Pangram spend couldn't be read before the finals ({type(e).__name__})")
            return
        if self.dry:
            left_text = "what's left of the night's ceiling after the pass"
        else:
            left_text = money(left)
        argv = ["--final", *self.job_args(), "--section", ",".join(map(str, self.sections())),
                "--purpose", "loop", "--max-usd", f"{left:.2f}"]
        res = self.child("the finals", argv, self.now() + dt.timedelta(minutes=FINAL_MINUTES))
        self.steps["finals"] = res
        self.lines.append(f"- The finals (sections {secs}, under {left_text}): "
                          + self.outcome_text(res, f"after {FINAL_MINUTES} minutes"))

    def outcome_text(self, res, deadline):
        if res["outcome"] == "dry":
            return f"would run python -m engine.regrow {res['args']}."
        spend = f"; Pangram {money(res['pangram'])}, Claude {money(res['claude'])}" if "pangram" in res else ""
        if res["outcome"] == "finished":
            return f"finished at {res['end']:%H:%M}{spend}."
        if res["outcome"] == "killed":
            return f"still going {deadline}, so its process tree was ended at {res['end']:%H:%M}{spend}."
        if res["outcome"] == "error":
            return f"didn't start: {res['why']}."
        return f"ended with exit code {res['rc']} at {res['end']:%H:%M} (its log: regrow.log in the runs folder){spend}."

    def child(self, name, argv, deadline):
        """Run python -m engine.regrow with `argv` as a child, until it ends or `deadline`; then its process tree is
        ended. Returns {"outcome": finished | killed | exit | error | dry, "rc", "end", "pangram", "claude"}."""
        from .util import redact
        args = " ".join(f'"{x}"' if " " in x else x for x in argv)
        self.say(f"{name}: {'would run' if self.dry else 'starts'} python -m engine.regrow {args}"
                 f" (ended if still going at {deadline:%H:%M})")
        if self.dry:
            return {"outcome": "dry", "args": args}
        t0 = self.now().timestamp()
        res = {"outcome": None, "rc": None}
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                proc = self.spawn_fn([sys.executable, "-m", "engine.regrow", *argv], fh)
                while True:
                    rc = proc.poll()
                    if rc is not None:
                        res.update(outcome="finished" if rc == 0 else "exit", rc=rc)
                        break
                    wait = (deadline - self.now()).total_seconds()
                    if wait <= 0:
                        ended, alive = self.kill_fn(proc)
                        res.update(outcome="killed", ended=ended, alive=alive)
                        break
                    self.sleep(min(POLL_SECONDS, wait))
        except Exception as e:  # noqa: BLE001
            res.update(outcome="error", why=f"{type(e).__name__}: {redact(e)[:200]}")
            self.problem(f"{name} didn't start: {res['why']}")
        res["end"] = self.now()
        if res["outcome"] == "killed":
            self.say(f"{name}: still going at {deadline:%H:%M}, so its process tree was ended"
                     f" ({len(res['ended'])} process(es))")
            if res["alive"]:
                self.problem(f"{name}: process(es) {', '.join(map(str, res['alive']))} outlived the kill")
            try:
                if self.port_fn():
                    self.problem(f"the llama server still answers on port {PORT} after {name} was ended; stop it"
                                 " by hand, or the next night skips its pass")
            except Exception:  # noqa: BLE001
                pass
            for f in self.torn():
                self.problem(f"{name} was ended while it wrote {f}: the file's last line wasn't whole, so the next"
                             f" step loads the bank without it, and the bank's first write cuts it off (the text is"
                             f" kept in {f}.torn)")
        elif res["outcome"] == "exit":
            self.problem(f"{name} ended with exit code {res['rc']}")
        try:
            res["pangram"], res["claude"] = self.ledger.spent(t0)
        except Exception as e:  # noqa: BLE001
            self.problem(f"what {name} spent couldn't be read ({type(e).__name__})")
        self.say(f"{name}: {res['outcome']}" + (f", Pangram {money(res['pangram'])}, Claude {money(res['claude'])}"
                                                if "pangram" in res else ""))
        return res

    def torn(self):
        """The pass's append-only files in the job folder whose last line isn't whole JSON (a kill can land in the
        middle of a write). Only read: Bank (engine/job.py) loads such a file without that line and cuts it at its
        first add, keeping the text in `<name>.torn`."""
        import json
        out = []
        d = getattr(self.job, "dir", None)
        for name in TORN_FILES:
            path = Path(d) / name if d else None
            if not path or not path.exists():
                continue
            lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
            try:
                if lines:
                    json.loads(lines[-1])
            except ValueError:
                out.append(name)
        return out

    # ------------------------------------------------------------------ the end
    def night_info(self, repair_off):
        lines = list(self.lines)
        if repair_off:
            lines.append(f"- The repair step: off tonight, {repair_off}.")
            self.short.append("repair step off")
        if self.ledger is not None and not self.dry:
            try:
                pangram, claude = self.ledger.spent(self.since)
                lines.append(f"- The night before the submission: Pangram {money(pangram)} of"
                             f" {money(float(self.cfg['max_pangram_usd']))}, Claude {money(claude)}.")
            except Exception:  # noqa: BLE001
                pass
        lines += [f"- Problem: {p}" for p in self.problems]
        short = list(self.short)
        if self.problems:
            short.append(f"{len(self.problems)} night problem{'' if len(self.problems) == 1 else 's'}, see the report")
        return {"lines": lines, "short": "; ".join(short) or None}

    def finish(self, early, repair_off, submit=True):
        from .util import redact
        self.result = {"halted": None, "lines": self.lines, "problems": self.problems, "steps": self.steps,
                       "submission": None, "repair_off": repair_off}
        if not submit:
            info = self.night_info(repair_off)
            self.say("the night stops before the submission")
            self.post_line("\n".join([f"**depangram night run, {self.start:%Y-%m-%d}:** stopped before the"
                                      " submission (the STOP file)."] + info["lines"]))
            self.result["halted"] = "the STOP file"
            return self.result
        self.say("the submission")
        try:
            now = self.now()
            r = self.submit(self.job, settings=self.settings, dry=self.dry, early=early,
                            night=self.night_info(repair_off), repair_mode="off" if repair_off else None, log=self.say,
                            today=now.date().isoformat(), stamp=f"{now:%Y-%m-%d %H:%M}")
            self.result["submission"] = r
            self.out(r["digest"])
            self.say(r.get("posted_line") or f"{self.label()}: no, the submission didn't say")
        except Exception as e:  # noqa: BLE001
            self.problem(f"the submission failed: {type(e).__name__}: {redact(e)[:200]}")
        self.say("the night run is done")
        return self.result

    def halt(self, why):
        self.say(f"the night stops: {why}")
        self.post_line(f"**depangram night run, {self.start:%Y-%m-%d}:** stopped: {why}.")
        self.result = {"halted": why, "lines": self.lines, "problems": self.problems, "steps": self.steps,
                       "submission": None, "repair_off": None}
        return self.result

    def post_line(self, text):
        """A short post when the night stops before the submission (which posts its own digest)."""
        if self.dry:
            self.say("a dry run: nothing posted")
            return
        from .morning import notifier_of
        if self.notify is None and notifier_of(self.settings) == "none":
            self.say(f"{self.label()}: no, the notifier is none (NOTIFIER)")
            return
        try:
            if self.notify is None:
                from . import discord
                got = discord.post(self.settings, text[:1900])
            else:
                got = self.notify(self.settings, text)
            self.say(f"{self.label()}: yes" if got else f"{self.label()}: no, the notifier didn't take it")
        except Exception as e:  # noqa: BLE001
            self.say(f"{self.label()}: no, the post failed ({type(e).__name__})")

    def label(self):
        from .morning import posted_label
        return posted_label(self.settings)


def exit_code(res):
    sub = res.get("submission") or {}
    return 1 if (res.get("halted") or res.get("problems") or sub.get("stopped") or sub.get("problems")
                 or not sub) else 0


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="the night run: main first, the paragraph pass, the finals and the"
                                             " morning submission, for a scheduler to start")
    ap.add_argument("--job", help="the job (a folder in jobs/); default [job] default in config.toml")
    ap.add_argument("--job-dir", help="a job folder anywhere, instead of --job")
    ap.add_argument("--dry", action="store_true", help="say what it would do: nothing spent, written, committed or posted")
    a = ap.parse_args(argv)
    from .job import JobError
    from .main_copy import load_job
    from .settings import Settings
    settings = Settings()
    try:
        job = load_job(a.job, a.job_dir, settings)
    except JobError as e:
        print(f"the night can't start: {e}")
        return 1
    n = Night(job, settings, dry=a.dry)
    n.job_dir = a.job_dir
    res = n.run()
    return 0 if a.dry and not res.get("halted") else exit_code(res)


if __name__ == "__main__":
    sys.exit(main())
