"""What each tool does, with every outside part passed in so the tests can hand over fakes: the settings, the Pangram
and Open Pangram clients, the busy check, the spawn, the process tree killer and the read of the pass's bank.

Every method returns a plain dict for the tool to hand back. A refusal or a failure raises Refused with a message
meant for the caller; server.py turns any other exception into a short message too (plain_error), with keys
taken out and the repo's own folder written as <repo>.
"""
import math
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

from engine.settings import SANDBOX, Settings

from . import runs as rs

TAIL_LINES = 20
REPORT_MAX_CHARS = 200_000
OPEN_FLAG_AT = 0.5        # a sentence Open Pangram scores at or above this on its own is listed as flagged
PANGRAM_FLAG_AT = 0.5     # a Pangram window over this is flagged, as engine/regrow.py's score_section reads it
SECRET_SETTINGS = ("PANGRAM_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "DISCORD_WEBHOOK_URL",
                   "DISCORD_BOT_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "HF_TOKEN")
SCORE_LINE = re.compile(r"\b(s\d+ [^:\n]{1,60}): section (\d+) (\d+(?:\.\d+)?) \((\d+) flagged(, HUMANIZED)?")
REPORT_LINE = re.compile(r"; report (regrow [^;\n]*?\.md)")
STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2})")


class Refused(Exception):
    """A tool said no, or couldn't do it; the message is for the caller as it stands."""


# ---------------------------------------------------------------------- plain messages
def scrub(text, secrets=(), roots=()):
    """`text` with keys and tokens out (engine/util.py's redact, plus the exact values of the key settings) and each
    root folder written as its short name."""
    from engine.util import redact
    text = redact(str(text))
    for s in secrets:
        if s and len(s) >= 8:
            text = text.replace(s, "[redacted]")
    for root, short in roots:
        for form in {str(root), str(root).replace("\\", "/"), str(root).replace("/", "\\")}:
            if form and len(form) > 3:
                text = text.replace(form, short)
    return text


def finite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


class App:
    def __init__(self, settings=None, *, repo=SANDBOX, jobs_dir=None, pangram_fn=None, open_pangram_fn=None,
                 busy_fn=None, spawn_fn=None, kill_fn=None, port_fn=None, alive_fn=None, began_fn=None,
                 pass_paragraphs_fn=None, config_fn=None, now=None, python=None):
        self._settings = settings
        self.config_fn = config_fn
        self.repo = Path(repo).resolve()
        self.jobs_dir = Path(jobs_dir).resolve() if jobs_dir else self.repo / "jobs"
        self.pangram_fn = pangram_fn
        self.open_pangram_fn = open_pangram_fn
        self.busy_fn, self.spawn_fn, self.kill_fn, self.port_fn = busy_fn, spawn_fn, kill_fn, port_fn
        self.alive_fn = alive_fn or rs.same_process
        self.began_fn = began_fn or rs.create_time
        self.pass_paragraphs_fn = pass_paragraphs_fn
        self.now = now or time.time
        self.python = python
        self.procs = {}                       # run id -> the Popen this server started, while it lives
        self.start_lock = threading.Lock()    # the busy check and the spawn go together
        self.open_lock = threading.Lock()
        self._open = None

    # ------------------------------------------------------------------ settings and plain text
    @property
    def settings(self):
        if self._settings is None:
            self._settings = Settings()
        return self._settings

    def secrets(self):
        out = []
        for name in SECRET_SETTINGS:
            try:
                v = self.settings.get(name)
            except Exception:  # noqa: BLE001
                v = None
            if v:
                out.append(str(v))
        return out

    def plain(self, text):
        return scrub(text, self.secrets(), [(self.repo, "<repo>"), (Path.home(), "~")])

    def fresh_config(self):
        """The config as it is on disk now (a night's copy is made from it, so an edit since the server started
        counts, as it does for a pass, whose child reads the config itself). The tests pass config_fn."""
        if self.config_fn is not None:
            return self.config_fn()
        from engine.settings import load_config
        return load_config()

    def runs_root(self):
        return Path(self.settings.runs_dir) / "mcp"

    def store(self):
        return rs.Store(self.runs_root())

    # ------------------------------------------------------------------ paths
    def inside(self, path, *roots):
        p = Path(path).resolve()
        return any(p == r or r in p.parents for r in roots)

    def job_folder(self, job):
        """The folder of `job`: a folder name in jobs/, or a path that resolves inside jobs/ or the repo."""
        if not isinstance(job, str) or not job.strip():
            raise Refused("name a job: a folder in jobs/, or a job folder inside the repo")
        job = job.strip()
        plain_name = re.fullmatch(r"[\w][\w .-]*", job) is not None and ".." not in job
        d = (self.jobs_dir / job) if plain_name else Path(job)
        if not d.is_absolute():
            d = self.repo / d
        d = d.resolve()
        if not self.inside(d, self.jobs_dir, self.repo):
            raise Refused(f"the job {job!r} isn't inside the jobs folder or the repo")
        if not (d / "job.toml").exists():
            raise Refused(f"there's no job.toml in the job folder {self.plain(d)}")
        return d

    def load_job(self, folder):
        from engine.job import Job
        return Job(root=folder)

    def job_args(self, folder):
        """How a child finds the job: --job NAME for a folder in jobs/, else --job-dir FOLDER."""
        if folder.parent == self.jobs_dir:
            return ["--job", folder.name]
        return ["--job-dir", str(folder)]

    # ------------------------------------------------------------------ score_text
    def score_text(self, text, detector="open"):
        if not isinstance(text, str) or not text.strip():
            raise Refused("there's no text to score")
        if detector == "pangram":
            return self.score_pangram(text)
        if detector == "open":
            return self.score_open(text)
        raise Refused(f"the detector is \"open\" or \"pangram\", not {detector!r}")

    def make_pangram(self, conn):
        if self.pangram_fn is not None:
            return self.pangram_fn(self.settings, conn)
        from engine.pangram import Pangram
        return Pangram(self.settings, conn)

    def score_pangram(self, text):
        from engine import db
        from engine.pangram import CapReached, PangramError, summarize
        conn = db.connect(self.settings.db_path)
        try:
            pg = self.make_pangram(conn)
            credits, usd = pg.estimate(text)
            try:
                resp, cached = pg.score(text, "mcp", "mcp score_text")
            except CapReached as e:
                raise Refused(f"Pangram's cap says no: {e}") from None
            except PangramError as e:
                raise Refused(f"Pangram failed: {self.plain(e)}") from None
        finally:
            conn.close()
        s = summarize(resp)
        flagged = [w.get("text", "").strip() for w in resp.get("windows") or []
                   if (w.get("ai_assistance_score") or 0) > PANGRAM_FLAG_AT and (w.get("text") or "").strip()]
        return {"detector": "pangram", "ai_share": s["fraction_ai"], "ai_assisted_share": s["fraction_ai_assisted"],
                "prediction": s["prediction"], "humanized": bool(s["humanized"]), "flagged": flagged,
                "flagged_means": f"Pangram's windows scored over {PANGRAM_FLAG_AT}, each a stretch of one or more"
                                 " sentences",
                "words": len(text.split()), "cost_usd": 0.0 if cached else round(float(usd), 4),
                "credits": 0 if cached else credits, "from_cache": bool(cached), "model_version": s["version"]}

    def open_pangram(self):
        """The Open Pangram model, loaded once on first use (it loads torch)."""
        with self.open_lock:
            if self._open is None:
                if self.open_pangram_fn is not None:
                    self._open = self.open_pangram_fn()
                else:
                    try:
                        from engine.open_pangram import OpenPangram
                    except ImportError as e:
                        raise Refused("Open Pangram needs torch, transformers, peft, numpy, scipy and emoji (the"
                                      f" optional lines in requirements.txt); this Python has no {e.name}") from None
                    try:
                        self._open = OpenPangram("roberta", settings=self.settings)
                    except OSError:
                        raise Refused("Open Pangram's model files aren't there: put editlens_roberta-large in"
                                      " OPEN_PANGRAM_DIR (default: open-pangram/ in MODELS_DIR)") from None
            return self._open

    def score_open(self, text):
        from engine.checks import split_sentences
        op = self.open_pangram()
        sents = [s.strip() for s in split_sentences(text) if s.strip()]
        got = op.score([text] + sents)
        whole, each = got[0], got[1:]
        flagged = [s for s, r in zip(sents, each) if r["score"] >= OPEN_FLAG_AT]
        return {"detector": "open", "ai_share": round(float(whole["score"]), 4), "humanized": None,
                "flagged": flagged,
                "flagged_means": f"each sentence scored on its own by Open Pangram, listed at {OPEN_FLAG_AT} or more;"
                                 " Open Pangram has no humanized call and no windows of its own",
                "words": len(text.split()), "cost_usd": 0.0}

    # ------------------------------------------------------------------ list_jobs and review_queue
    def pass_state(self, job):
        """Whether the pass's bank can be read without a write: None when it can, else why not."""
        if not (job.dir / "regrow bank.jsonl").exists():
            return "no paragraph pass has run on this job yet"
        if not (job.dir / "regrow origs.json").exists():
            return ("the pass's bank has no regrow origs.json beside it, and reading it would write one; the next"
                    " pass writes it")
        return None

    def passed(self, job, sections):
        from engine.job import Bank
        fn = self.pass_paragraphs_fn
        if fn is None:
            from engine.regrow import pass_paragraphs as fn
        return fn(job, Bank(job.bank_path), sections)

    def doc_name(self, path):
        if path is None:
            return None
        p = Path(path)
        return self.plain(p) if self.inside(p, self.repo) else p.name

    def list_jobs(self):
        out = []
        if not self.jobs_dir.is_dir():
            return {"jobs": out, "note": "there's no jobs folder"}
        for d in sorted(self.jobs_dir.iterdir()):
            if not (d / "job.toml").exists():
                continue
            row = {"folder": d.name}
            try:
                job = self.load_job(d)
                secs = list(job.sections)
                row.update(name=job.name, description=job.description, document=self.doc_name(job.real_file),
                           document_there=bool(job.real_file and Path(job.real_file).exists()),
                           sections=secs, night_sections=list(job.night_sections), branch=job.branch)
                why = self.pass_state(job)
                if why:
                    row.update(scores=None, scores_note=why)
                else:
                    got = self.passed(job, secs)["sections"]
                    row["scores"] = {str(n): {"ai_share": s["ai"], "scoring": s["label"], "scored": s["scored"]}
                                     for n, s in sorted(got.items())}
                    missing = [n for n in secs if n not in got]
                    if missing:
                        row["scores_note"] = ("no clean scoring yet for section(s) "
                                              + ", ".join(map(str, missing)))
            except Exception as e:  # noqa: BLE001
                row["error"] = self.plain(f"{type(e).__name__}: {e}")
            out.append(row)
        return {"jobs": out}

    def review_queue(self, job):
        from engine import review as rv
        folder = self.job_folder(job)
        j = self.load_job(folder)
        try:
            review = rv.load(j.dir)
        except ValueError as e:
            raise Refused(self.plain(e)) from None
        out = {"job": j.name,
               "repairs": [{k: r.get(k) for k in ("label", "from", "text", "moves", "by", "when", "scoring")}
                           for r in review["repairs"]],
               "vetoes": [{k: v.get(k) for k in ("label", "text", "why", "scoring")} for v in review["vetoes"]]}
        why = self.pass_state(j)
        if why:
            out.update(for_you=None, repaired=None, moot=None, unmatched=None, unmatched_review=None,
                       stuck_entries=[dict({"label": s.get("label"), "orig": s.get("orig")}, **rv.stuck_note(s))
                                      for s in review["stuck"]],
                       note=why + "; the stuck entries are listed from review.json, not matched to paragraphs")
            return out
        got = self.passed(j, list(j.sections))
        for_you, repaired = [], []
        for plabel, row in got["paragraphs"].items():
            for st in row.get("stuck") or []:
                for_you.append({"paragraph": plabel, "section": row["section"], "kind": "stuck",
                                "bold": st.get("bold"), "prompt": {k: st.get(k) for k in rv.STUCK_KEYS}})
            if row.get("for_you"):
                for_you.append({"paragraph": plabel, "section": row["section"], "kind": "hand fix",
                                "text": row.get("text"), "prompt": row["for_you"]})
            if row.get("repaired"):
                repaired.append({"paragraph": plabel, "section": row["section"], "moves": row["repaired"]})
        out.update(for_you=for_you, repaired=repaired, moot=list(got.get("moot") or []),
                   unmatched=list(got.get("unmatched") or []),
                   unmatched_review=[list(x) for x in got.get("unmatched_review") or []])
        return out

    # ------------------------------------------------------------------ start_run
    def busy(self):
        if self.busy_fn is not None:
            return self.busy_fn()
        from engine.night import busy
        return busy()

    def stop_file(self):
        return Path(self.settings.stop_file)

    def night_clock_note(self, cfg):
        """What the night's own clock guards will do with a start now, or None when it runs in full."""
        import datetime as dt
        from engine import night
        nc = dict(night.DEFAULTS)
        nc.update(cfg.get("night") or {})
        now = dt.datetime.fromtimestamp(self.now())
        for key, what in (("latest_start", "[night] latest_start"), ("pass_until", "[night] pass_until")):
            try:
                if now >= night.at(now, nc[key]):
                    return (f"it's {now:%H:%M}, at or after {what} {nc[key]}, so this night skips the paragraph pass"
                            " and the finals and runs the morning submission only (the repair step, the write-back,"
                            " a commit, and a publish when [git] publish is on)")
            except (ValueError, KeyError):
                continue
        return None

    def start_run(self, job, max_pangram_usd=None, kind="pass", sections=None, rounds=1, max_claude_usd=None):
        if kind not in ("pass", "night"):
            raise Refused(f"kind is \"pass\" or \"night\", not {kind!r}")
        if max_pangram_usd is None:
            raise Refused("give max_pangram_usd, the most this run may spend on Pangram; there's no default")
        if not finite(max_pangram_usd) or float(max_pangram_usd) <= 0:
            raise Refused("max_pangram_usd is a number of dollars over zero")
        if max_claude_usd is not None and (not finite(max_claude_usd) or float(max_claude_usd) < 0):
            raise Refused("max_claude_usd is a number of dollars, zero or more")
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
            raise Refused("rounds is a whole number, 1 or more")
        folder = self.job_folder(job)
        j = self.load_job(folder)
        if sections is not None:
            try:
                sections = [int(x) for x in sections]
            except (TypeError, ValueError):
                raise Refused("sections is a list of section ids") from None
            if not sections:
                raise Refused("sections is a list of section ids; leave it out for the job's own")
            scope = set(j.sections)
            out_of = [n for n in sections if n not in scope]
            if out_of:
                raise Refused(f"section(s) {', '.join(map(str, out_of))} aren't in the job's scope"
                              f" ({', '.join(map(str, sorted(scope))) or 'none'})")
        stop = self.stop_file()
        if stop.exists():
            raise Refused(f"the STOP file is there ({self.plain(stop)}); delete it to let a run start")
        with self.start_lock:
            for rec in self.store().all():
                if self.refresh(rec)["state"] == "running":
                    raise Refused(f"run {rec['id']} is still going; stop it or wait for it")
            why = self.busy()
            if why:
                raise Refused(f"the GPU is taken: {self.plain(why)}")
            return self.launch(folder, j, kind, sections, rounds, float(max_pangram_usd),
                               None if max_claude_usd is None else float(max_claude_usd))

    def launch(self, folder, j, kind, sections, rounds, max_pangram_usd, max_claude_usd):
        store = self.store()
        run_id = rs.new_id(kind, self.now())
        d = store.make(run_id)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        notes = []
        # regrow (the pass, and a night's pass and finals) writes its report into analytics/ without making it
        (self.repo / "analytics").mkdir(exist_ok=True)
        if kind == "pass":
            module = "engine.regrow"
            # regrow's --rounds counts the arm-set round too, so 1 new scoring round is --rounds 2, as the night does
            args = [*self.job_args(folder), "--purpose", "loop", "--rounds", str(rounds + 1),
                    "--max-usd", f"{max_pangram_usd:.2f}"]
            if sections:
                args += ["--section", ",".join(map(str, sections))]
            if max_claude_usd is not None:
                args += ["--max-claude-usd", f"{max_claude_usd:.2f}"]
        else:
            module = "engine.night"
            args = self.job_args(folder)
            cfg = rs.night_config(self.fresh_config(), max_pangram_usd=max_pangram_usd,
                                  max_claude_usd=max_claude_usd, rounds=rounds, sections=sections)
            (d / "config.toml").write_text(rs.toml_dump(cfg), encoding="utf-8")
            env["DEPANGRAM_CONFIG"] = str(d / "config.toml")
            clock = self.night_clock_note(cfg)
            if clock:
                notes.append(clock)
            notes.append(f"the night's pass and finals stay under ${max_pangram_usd:.2f} of Pangram together; the"
                         " repair step at the submission spends on its own, under [repair]'s ceilings held to the"
                         " same numbers, so the night as a whole can spend up to about twice the figure given")
            notes.append("the night ends with the morning submission: the repair step, the write-back, a commit"
                         " on the work branch, a publish when [git] publish is on, and a digest through the notifier")
        cmd = rs.child_cmd(d / "exit", module, args, self.python)
        started = self.now()
        rec = {"id": run_id, "kind": kind, "job": j.name, "job_folder": self.plain(folder), "module": module,
               "args": [self.plain(a) for a in args], "sections": sections, "rounds": rounds,
               "max_pangram_usd": max_pangram_usd, "max_claude_usd": max_claude_usd, "pid": None,
               "began": None, "started": started, "ended": None, "state": "running", "exit_code": None,
               "notes": notes}
        spawn = self.spawn_fn or (lambda c, out, e: rs.spawn(c, out, e, self.repo))
        with (d / "out.log").open("ab") as out:
            try:
                proc = spawn(cmd, out, env)
            except Exception as e:  # noqa: BLE001
                rec.update(state="lost", ended=self.now(), notes=notes + [f"it didn't start: {self.plain(e)}"])
                store.save(rec)
                raise Refused(f"the run didn't start: {self.plain(e)}") from None
        rec["pid"] = proc.pid
        rec["began"] = self.began_fn(proc.pid)
        self.procs[run_id] = proc
        store.save(rec)
        return {"run_id": run_id, "kind": kind, "job": j.name, "command": f"python -m {module} " + " ".join(
            f'"{a}"' if " " in a else a for a in rec["args"]), "state": "running", "notes": notes}

    # ------------------------------------------------------------------ run records
    def record(self, run_id):
        if not rs.valid_id(run_id):
            raise Refused("that isn't a run id (they look like pass-20261116-020000-a1b2c3)")
        try:
            rec = self.store().load(run_id)
        except KeyError:
            raise Refused(f"there's no run {run_id}") from None
        return self.refresh(rec)

    def refresh(self, rec):
        """The record with its state brought up to date (and saved when it changed): a run whose process is gone is
        finished with the code its wrapper wrote, else lost."""
        if rec.get("state") != "running":
            return rec
        proc = self.procs.get(rec["id"])
        rc = None
        if proc is not None:
            rc = proc.poll()
            if rc is None:
                return rec
            self.procs.pop(rec["id"], None)
        elif rec.get("pid") and self.alive_fn(rec["pid"], rec.get("began")):
            return rec
        code = self.store().exit_code(rec["id"])
        code = code if code is not None else rc
        rec.update(state="finished" if code is not None else "lost", exit_code=code, ended=self.now())
        self.store().save(rec)
        return rec

    def log_text(self, rec):
        try:
            return (self.store().folder(rec["id"]) / "out.log").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def night_log(self, rec):
        """A night's own log in RUNS_DIR/<job>/night.log, where its pass's output goes; "" when it isn't this run's."""
        path = Path(self.settings.runs_dir) / rec["job"] / "night.log"
        try:
            if path.stat().st_mtime + 1 < rec["started"]:
                return ""
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        # the file holds every night: keep the lines from this run's start on (a line with no time of its own, a
        # child's plain output, goes with the lines around it)
        since = time.strftime("%Y-%m-%d %H:%M", time.localtime(rec["started"]))
        keep, out = False, []
        for line in text.splitlines():
            m = STAMP.match(line)
            if m:
                keep = m.group(1) >= since
            if keep:
                out.append(line)
        return "\n".join(out)

    def spend(self, rec):
        """(Pangram, Claude) dollars in the ledger from the run's start to its end (or now), or None for one the ledger
        couldn't give. Read only: the server never writes the ledger here."""
        end = rec.get("ended") or self.now()
        path = Path(self.settings.runs_dir) / "depangram.sqlite"
        if not path.exists():
            return 0.0, 0.0
        out = []
        try:
            conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=2)
        except sqlite3.Error:
            return None, None
        try:
            for q in ("SELECT COALESCE(SUM(usd), 0) FROM pangram_calls WHERE ts >= ? AND ts <= ? AND usd > 0",
                      "SELECT COALESCE(SUM(cost_usd), 0) FROM claude_calls WHERE ts >= ? AND ts <= ?"):
                try:
                    out.append(round(float(conn.execute(q, (rec["started"], end)).fetchone()[0]), 4))
                except sqlite3.OperationalError as e:
                    out.append(0.0 if "no such table" in str(e) else None)
                except sqlite3.Error:
                    out.append(None)
        finally:
            conn.close()
        return tuple(out)

    def when(self, ts):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else None

    def run_status(self, run_id):
        rec = self.record(run_id)
        text = self.log_text(rec)
        if rec["kind"] == "night":
            text = text + "\n" + self.night_log(rec)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        scores = [{"scoring": m.group(1), "section": int(m.group(2)), "ai_share": float(m.group(3)),
                   "flagged": int(m.group(4)), "humanized": bool(m.group(5))} for m in SCORE_LINE.finditer(text)]
        pangram, claude = self.spend(rec)
        return {"run_id": rec["id"], "kind": rec["kind"], "job": rec["job"], "state": rec["state"],
                "exit_code": rec["exit_code"], "started": self.when(rec["started"]), "ended": self.when(rec["ended"]),
                "spent": {"pangram_usd": pangram, "claude_usd": claude,
                          "ceilings": {"pangram_usd": rec["max_pangram_usd"], "claude_usd": rec["max_claude_usd"]},
                          "means": "every ledger row from the run's start to its end, whoever wrote it"},
                "scores": scores, "last_lines": [self.plain(x) for x in lines[-TAIL_LINES:]],
                "notes": rec.get("notes") or []}

    def report_files(self, rec):
        """The reports a run wrote: the regrow reports its log names, plus any regrow report or morning digest
        written while it ran."""
        start, end = rec["started"] - 1, (rec.get("ended") or self.now()) + 5
        named = set(REPORT_LINE.findall(self.log_text(rec) + "\n" + (self.night_log(rec) if rec["kind"] == "night"
                                                                      else "")))
        found = []
        for folder, pattern in ((self.repo / "analytics", "regrow *.md"), (self.repo / "digests", "*.md")):
            if folder == self.repo / "digests" and rec["kind"] != "night":
                continue
            if not folder.is_dir():
                continue
            for p in sorted(folder.glob(pattern)):
                try:
                    in_window = start <= p.stat().st_mtime <= end
                except OSError:
                    continue
                if p.name in named or in_window:
                    found.append(p)
        return found

    def run_report(self, run_id):
        rec = self.record(run_id)
        files = self.report_files(rec)
        parts = []
        for p in files:
            try:
                parts.append(f"# {p.parent.name}/{p.name}\n\n" + p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        md = self.plain("\n\n---\n\n".join(parts))
        cut = len(md) > REPORT_MAX_CHARS
        if cut:
            md = md[:REPORT_MAX_CHARS] + "\n\n(cut here: the report is longer than the server sends)"
        out = {"run_id": rec["id"], "state": rec["state"], "files": [f"{p.parent.name}/{p.name}" for p in files],
               "markdown": md}
        if not files:
            out["note"] = ("the run hasn't written a report yet" if rec["state"] == "running"
                           else "the run wrote no report")
        return out

    def stop_run(self, run_id):
        rec = self.record(run_id)
        if rec["state"] != "running":
            return {"run_id": rec["id"], "state": rec["state"], "exit_code": rec["exit_code"],
                    "note": f"the run isn't running (it's {rec['state']}), so nothing was stopped"}
        proc = self.procs.get(rec["id"])
        if proc is None:
            from types import SimpleNamespace
            proc = SimpleNamespace(pid=rec["pid"])
        kill = self.kill_fn
        if kill is None:
            from engine.night import kill_tree as kill
        ended, alive = kill(proc)
        self.procs.pop(rec["id"], None)
        notes = list(rec.get("notes") or [])
        if alive:
            notes.append(f"process(es) {', '.join(map(str, alive))} outlived the stop")
        held = self.port_fn
        if held is None:
            from engine.night import port_held as held
        try:
            if held():
                notes.append("the llama server still answers on its port; stop it by hand, or the next run sees the"
                             " GPU as taken")
        except Exception:  # noqa: BLE001
            pass
        rec.update(state="stopped", ended=self.now(), notes=notes)
        self.store().save(rec)
        return {"run_id": rec["id"], "state": "stopped", "ended_pids": list(ended), "alive_pids": list(alive),
                "notes": notes}


def plain_error(app, e):
    """An exception as one short line for the caller: no traceback, no key, the repo written as <repo>."""
    if isinstance(e, Refused):
        return app.plain(e)
    return app.plain(f"{type(e).__name__}: {e}")[:500]
