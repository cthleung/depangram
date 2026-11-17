"""Tests for the morning submission (engine/morning.py) and the loop's call to it at its digest. Each runs in a temp
git repo with a bare origin, the way the MainCopy tests do; the write-back, the sync and Discord are fakes. Each check
has a flag case and a pass case. Run from the repo root:

    python -m unittest discover tests
"""
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from engine import morning
from engine.job import Job
from engine.wrapup import LEDGER
from test_writeback import IDENT, REL, git

DAY, STAMP = "2026-11-15", "2026-11-15 06:10"
GUIDE = "# The guide\n\n## 1. First\n\nIntro.\n\nClaude's paragraph.\n"
NEWER = "The pass's paragraph."
FY = {"went_wrong": "a wrong year", "about": "the study's year", "point": "the year matters", "fact": "it ran in 2023"}
SYNC_REPORT = {"source": "the guide", "sections_before": [1], "sections_after": [1], "left_out": {}, "restored": [],
               "your_edits": [(1, "Intro, edited by you.")], "kept": [("s1w0", "s1w0")], "retired": [], "fresh": [],
               "examples_added": [], "new_verdicts": {}, "seed_windows": [], "written": True}


class FakeWriteback:
    """engine.writeback.run's shape: it swaps Claude's paragraph for `text` and adds a line to the ledger."""

    def __init__(self, text=NEWER, for_you=False):
        self.text, self.for_you, self.calls = text, for_you, []

    def __call__(self, job, write=True, stamp=None, real=None, after_sync=False):
        self.calls.append({"write": write, "after_sync": after_sync})
        raw = Path(real).read_text(encoding="utf-8")
        new = raw if self.text is None else raw.replace("Claude's paragraph.", self.text)
        changed = new != raw
        labels = ["s1w0 p1"] if changed else []
        rep = {"replaced": labels, "marked": [], "same": [] if changed else ["s1w0 p1"], "restored": [], "yours": [],
               "skipped": [], "missing": [], "ledger": [{"label": "s1w0 p1", "text": self.text}] if changed else [],
               "live": {1: 1}, "for_you": ["s1w0 p1"] if self.for_you else [],
               "prompts": {"s1w0 p1": FY} if self.for_you else {}}
        if write and changed:
            Path(real).write_text(new, encoding="utf-8", newline="\n")
            with (job.dir / LEDGER).open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rep["ledger"][0]) + "\n")
        return rep, {1: "the pass's best assembly scored 0.20 (s1 round 2)"}, changed


class FakeSync:
    """engine.sync.sync's shape: the frozen original becomes the guide, with a backup; and a file another job writes
    in the job folder at the same time (the GPU job's regrow bank), which stays out of the sync's commit and goes in
    the morning's own (since the repair step, 2026-11-14)."""

    def __init__(self):
        self.calls = []

    def __call__(self, job_dir, source, write=False, today=None):
        self.calls.append({"write": write, "today": today})
        d = Path(job_dir)
        (d / "orig.md").write_text(Path(source).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        (d / f"orig (before sync {today}).md").write_text("the old original\n", encoding="utf-8", newline="\n")
        (d / "regrow bank.jsonl").write_text('{"another": "job"}\n', encoding="utf-8", newline="\n")
        return dict(SYNC_REPORT)


class FakeRepair:
    """engine.repair.run's shape: it notes when it ran (how many sync and write-back calls came before) and, when
    `writes`, appends a repair to the reviewer's file and a line to the repair log, as the real step does."""

    def __init__(self, morning_test, writes=False, fail=False):
        self.t, self.writes, self.fail, self.calls = morning_test, writes, fail, []

    def __call__(self, job, settings, conn, mode=None, today=None, log=print):
        self.calls.append({"mode": mode, "after_sync": len(self.t.sync.calls), "after_writeback": len(self.t.wb.calls)})
        if self.fail:
            raise RuntimeError("the repair step broke")
        spans = []
        if self.writes:
            (job.dir / "review.json").write_text(json.dumps({"vetoes": [], "repairs": [REPAIRED], "for_you": []}, indent=1),
                                                 encoding="utf-8", newline="\n")
            with (job.dir / "repair log.jsonl").open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"key": "k", "outcome": "repaired"}) + "\n")
            spans = [{"label": "s1w0 p1", "section": 1, "outcome": "repaired", "before": "Claude's, with a slip.",
                      "after": "Claude's.", "moves": 'cut: "With a slip."', "why": None, "replayed": False}]
        return {"mode": mode or "apply", "spans": spans, "counts": {"repaired": len(spans)}, "seen": 0, "to_read": [],
                "unread": 0, "unread_why": None, "claude_usd": 0.2, "max_claude_usd": 3.0, "pangram_usd": 0.0,
                "rescored": {}, "stale": [], "written": self.writes}


REPAIRED = {"label": "s1w0 p1", "orig": "Claude's paragraph.", "from": "x", "text": "y", "moves": "cut", "by": "opus",
            "when": DAY}


class Poster:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def __call__(self, settings, content, file_name=None, file_text=None):
        self.calls.append((content, file_name, file_text))
        if self.fail:
            raise ConnectionError("no network")
        return True


def settings(**morning_cfg):
    """A settings double. These tests run with publishing on ([git] publish = true), the private setup the morning
    was built for; MorningRepo's publishing-off tests pass publish=False. [git] push follows publish."""
    return types.SimpleNamespace(config={"morning": morning_cfg, "git": {"publish": morning_cfg.pop("publish", True)}})


class MorningRepo(unittest.TestCase):
    """The temp repo with a bare origin, the job and the helpers, shared by the test classes below."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patch = mock.patch.dict(os.environ, IDENT)
        patch.start()
        self.addCleanup(patch.stop)
        root = Path(self.tmp.name).resolve()
        self.origin, self.work, self.yours, self.digests = root / "origin.git", root / "work", root / "yours", root / "digests"
        git("init", "--bare", "-b", "main", str(self.origin), cwd=root)
        git("clone", "-c", "core.autocrlf=false", str(self.origin), str(self.work), cwd=root)
        git("config", "core.autocrlf", "false", cwd=self.work)
        self.guide = self.work / REL
        self.guide.parent.mkdir()
        self.guide.write_text(GUIDE, encoding="utf-8", newline="\n")
        (self.work / "other.txt").write_text("main's own file\n", encoding="utf-8", newline="\n")
        git("add", "-A", cwd=self.work)
        git("commit", "-m", "the guide", cwd=self.work)
        git("push", "origin", "HEAD:main", cwd=self.work)
        git("checkout", "-b", "side", cwd=self.work)
        self.jobdir = self.work / "jobs" / "t"
        self.jobdir.mkdir(parents=True)
        (self.jobdir / "job.toml").write_text(
            '[job]\nname = "t"\noriginal = "orig.md"\nworking = "work.md"\nsections = [1]\nvoice_samples = [9]\n'
            f"real_file = '{self.guide.as_posix()}'\n", encoding="utf-8", newline="\n")
        (self.jobdir / "orig.md").write_text(GUIDE, encoding="utf-8", newline="\n")
        (self.jobdir / LEDGER).write_text("", encoding="utf-8")
        git("add", "-A", cwd=self.work)
        git("commit", "-m", "the job", cwd=self.work)
        git("push", "origin", "side", cwd=self.work)
        git("clone", "-c", "core.autocrlf=false", str(self.origin), str(self.yours), cwd=root)
        git("config", "core.autocrlf", "false", cwd=self.yours)

    def submit(self, wb=None, sync=None, post=None, cfg=None, dry=False, repair=None, origs=None, repair_cfg=None):
        self.wb, self.sync, self.post = wb or FakeWriteback(), sync or FakeSync(), post or Poster()
        self.repair = repair or FakeRepair(self)
        self.origs = origs or (lambda job, log=None: False)
        s = settings(**(cfg or {}))
        if repair_cfg:
            s.config["repair"] = repair_cfg
        return morning.submit(Job(root=self.jobdir), repo=self.work, settings=s, dry=dry,
                              writeback=self.wb, sync_fn=self.sync, post=self.post, repair_fn=self.repair,
                              origs_fn=self.origs, today=DAY, stamp=STAMP, digests=self.digests, log=lambda *_: None)

    def edit_on_main(self, text):
        git("pull", "origin", "main", cwd=self.yours)
        (self.yours / REL).write_text(text, encoding="utf-8", newline="\n")
        git("commit", "-am", "your edit", cwd=self.yours)
        git("push", "origin", "HEAD:main", cwd=self.yours)

    def files_in(self, commit):
        return git("show", "--name-only", "--format=", commit, cwd=self.work).split("\n")


class Morning(MorningRepo):
    def test_the_write_back_is_committed_pushed_and_published_with_nothing_else(self):
        (self.jobdir / "junk.txt").write_text("not the morning's\n", encoding="utf-8")
        (self.work / "staged.txt").write_text("staged by someone else\n", encoding="utf-8")
        git("add", "staged.txt", cwd=self.work)
        r = self.submit(wb=FakeWriteback(for_you=True))
        self.assertIsNone(r["stopped"])
        self.assertEqual((r["state"], len(r["commits"]), r["pushed"], r["problems"]), ("same", 1, "side", []))
        self.assertEqual(sorted(self.files_in("HEAD")), [REL, f"jobs/t/{LEDGER}"])            # pass: those two only
        self.assertEqual(git("log", "-1", "--format=%B", cwd=self.work),
                         "t: the morning submission of 2026-11-15 (1 paragraph newer, 1 for you)")
        self.assertIn("A  staged.txt", git("status", "--porcelain", cwd=self.work))             # still staged, not ours
        self.assertIn("?? jobs/t/junk.txt", git("status", "--porcelain", cwd=self.work))
        self.assertEqual(git("rev-parse", "origin/side", cwd=self.work), git("rev-parse", "HEAD", cwd=self.work))
        self.assertIn(NEWER, git("show", f"origin/main:{REL}", cwd=self.work))                # published
        self.assertEqual(git("ls-tree", "--name-only", "-r", "origin/main", cwd=self.work).split("\n"), [REL, "other.txt"])
        self.assertEqual(r["published"], git("rev-parse", "--short", "origin/main", cwd=self.work))
        self.assertEqual(git("rev-parse", "--abbrev-ref", "HEAD", cwd=self.work), "side")       # no branch switch
        self.assertEqual(self.wb.calls, [{"write": True, "after_sync": False}])
        (content, name, text), = self.post.calls
        self.assertEqual(name, "2026-11-15 morning submission.md")
        self.assertIn("1 paragraph newer, 1 in bold for you to rewrite; on main as", content)
        self.assertIn("## For you to rewrite by hand", text)
        self.assertIn("- s1w0 p1: the study's year", text)
        self.assertIn("- Section 1: the pass's best assembly scored 0.20", text)
        self.assertIn(f"- On side: {r['commits'][0]} (pushed)", text)
        self.assertEqual((self.digests / name).read_text(encoding="utf-8"), text + "\nPosted on Discord: yes\n")
        self.assertEqual(r["posted_line"], "Posted on Discord: yes")

    def test_nothing_new_posts_one_line(self):
        head = git("rev-parse", "HEAD", cwd=self.work)
        r = self.submit(wb=FakeWriteback(text=None))
        self.assertEqual((r["commits"], r["published"], git("rev-parse", "HEAD", cwd=self.work)), ([], None, head))
        (content, name, text), = self.post.calls
        self.assertIsNone(name)                                                                 # pass: one line, no file
        self.assertEqual(content, "The morning submission of 2026-11-15: nothing new to write back, nothing to pull,"
                                  " and main already has this branch's copy.")
        r = self.submit(wb=FakeWriteback(text=None, for_you=True))                              # one still in bold
        (content, name, _), = self.post.calls
        self.assertIsNone(name)
        self.assertTrue(content.endswith("main already has this branch's copy. 1 paragraph still in bold for you to"
                                         " rewrite."))
        r = self.submit(wb=FakeWriteback())                                                     # flag: something new
        self.assertIsNotNone(self.post.calls[0][1])

    def test_the_morning_commits_every_file_the_real_sync_writes(self):
        from engine.sync import sync
        from test_m3 import ORIG, REGROWN, make_job, stub_scan
        d = self.work / "jobs" / "m3"
        d.mkdir()
        make_job(d)
        src = self.work / "real.md"
        src.write_text(ORIG.replace("Claude wrote this line about tokens and words.", REGROWN).replace("Links.", "Links, yours."),
                       encoding="utf-8", newline="\n")
        git("add", "-A", cwd=self.work)
        git("commit", "-m", "a second job", cwd=self.work)
        sync(d, src, write=True, classify=stub_scan, today=DAY)
        wrote = set(git("ls-files", "-m", "-o", "--exclude-standard", "--", "jobs/m3", cwd=self.work).split("\n")) - {""}
        listed = {p.relative_to(self.work).as_posix() for p in morning.sync_paths(Job(root=d), DAY)}
        self.assertIn("jobs/m3/bank (before sync 2026-11-15).jsonl", wrote)
        self.assertEqual(wrote - listed, set())                                                 # pass: nothing missed
        self.assertNotIn("jobs/m3/notes cache.json", listed)                                    # flag: not the sync's

    def test_uncommitted_changes_in_the_guide_stop_it(self):
        self.guide.write_text(GUIDE + "\nA line being typed.\n", encoding="utf-8", newline="\n")
        head = git("rev-parse", "HEAD", cwd=self.work)
        r = self.submit()
        self.assertIn("aren't committed", r["stopped"])                                         # flag
        self.assertEqual((self.wb.calls, r["commits"], git("rev-parse", "HEAD", cwd=self.work)), ([], [], head))
        self.assertIn("A line being typed.", self.guide.read_text(encoding="utf-8"))
        self.assertIn("stopped", self.post.calls[0][0])
        git("checkout", "--", REL, cwd=self.work)
        self.assertIsNone(self.submit()["stopped"])                                             # pass

    def test_an_edit_on_main_is_pulled_synced_and_committed_before_the_write_back(self):
        self.edit_on_main(GUIDE.replace("Intro.", "Intro, edited by you."))
        main_blob = git("rev-parse", f"HEAD:{REL}", cwd=self.yours)
        r = self.submit()
        self.assertEqual((r["state"], r["problems"], len(r["commits"])), ("main ahead", [], 2))
        self.assertEqual(self.sync.calls, [{"write": True, "today": DAY}])
        self.assertEqual(self.wb.calls, [{"write": True, "after_sync": True}])                # the loop's text stays
        first, second = r["commits"]
        self.assertEqual(sorted(self.files_in(first)), [REL, "jobs/t/orig (before sync 2026-11-15).md", "jobs/t/orig.md"])
        self.assertEqual(git("rev-parse", f"{first}:{REL}", cwd=self.work), main_blob)          # main's copy, as it was
        self.assertEqual(sorted(self.files_in(second)), sorted([REL, f"jobs/t/{LEDGER}", "jobs/t/regrow bank.jsonl"]))
        self.assertNotIn("jobs/t/regrow bank.jsonl", self.files_in(first))                     # the pass's bank: not the sync's
        self.assertEqual(self.repair.calls, [{"mode": None, "after_sync": 1, "after_writeback": 0}])  # between the two
        posted = git("show", f"origin/main:{REL}", cwd=self.work)
        self.assertIn("Intro, edited by you.", posted)                                          # pass: yours and the newer text
        self.assertIn(NEWER, posted)
        self.assertIn("## What the sync took in", r["digest"])
        self.assertIn("- your edit, kept as you wrote it (section 1): Intro, edited by you.", r["digest"])

    def test_when_both_moved_main_s_copy_wins_and_the_write_back_runs_again(self):
        self.guide.write_text(GUIDE.replace("Claude's paragraph.", "Yesterday's write-back."), encoding="utf-8",
                              newline="\n")
        git("commit", "-am", "yesterday's write-back, never published", cwd=self.work)
        self.edit_on_main(GUIDE.replace("Intro.", "Intro, edited by you."))
        r = self.submit()
        self.assertEqual((r["state"], r["problems"]), ("diverged", []))
        posted = git("show", f"origin/main:{REL}", cwd=self.work)
        self.assertIn("Intro, edited by you.", posted)
        self.assertNotIn("Yesterday's write-back.", posted)                                     # flag: main's copy won
        self.assertIn(NEWER, posted)                                                            # pass: rebuilt on top

    def test_a_failed_push_publish_and_post_are_reported_and_undo_nothing(self):
        git("remote", "set-url", "--push", "origin", str(Path(self.tmp.name) / "nowhere.git"), cwd=self.work)
        r = self.submit(post=Poster(fail=True))
        self.assertIsNone(r["stopped"])
        self.assertEqual(len(r["commits"]), 1)                                                  # the commit stays
        self.assertNotIn(NEWER, git("show", f"origin/main:{REL}", cwd=self.work))
        self.assertEqual(len(r["problems"]), 2)                                                 # flag: push and publish
        self.assertIn("the push of side failed", r["problems"][0])
        self.assertIn("the publish to main failed", r["problems"][1])
        self.assertIs(r["posted"], False)
        self.assertIn("the push of side failed", (self.digests / "2026-11-15 morning submission.md").read_text(encoding="utf-8"))
        git("remote", "set-url", "--push", "origin", str(self.origin), cwd=self.work)
        r = self.submit(wb=FakeWriteback(text=None))                                            # pass: the next morning
        self.assertEqual(r["problems"], [])
        self.assertIn(NEWER, git("show", f"origin/main:{REL}", cwd=self.work))                # main was behind: published

    def test_the_config_can_turn_off_the_publish_and_the_post(self):
        r = self.submit(cfg={"publish": False, "post": False})
        self.assertEqual((len(r["commits"]), r["published"], self.post.calls), (1, None, []))  # flag: both off
        self.assertNotIn(NEWER, git("show", f"origin/main:{REL}", cwd=self.work))
        self.assertIn("[git] publish is off", r["digest"])
        self.assertTrue((self.digests / "2026-11-15 morning submission.md").exists())

    def test_a_dry_run_writes_nothing_commits_nothing_and_posts_nothing(self):
        self.edit_on_main(GUIDE.replace("Intro.", "Intro, edited by you."))
        head = git("rev-parse", "HEAD", cwd=self.work)
        r = self.submit(dry=True)
        self.assertEqual((self.sync.calls, self.post.calls, r["commits"]), ([], [], []))
        self.assertEqual(self.wb.calls, [{"write": False, "after_sync": False}])
        self.assertEqual((git("rev-parse", "HEAD", cwd=self.work), self.guide.read_text(encoding="utf-8")), (head, GUIDE))
        self.assertFalse(self.digests.exists())
        self.assertIn("It would pull main's copy and sync it in first", r["digest"])
        self.assertIn("A dry run: nothing committed, pushed or published.", r["digest"])
        self.assertEqual([c["mode"] for c in self.repair.calls], ["dry"])                        # the repair step's own dry run
        origs = []
        self.submit(dry=True, origs=lambda job, log=None: origs.append(job))
        self.assertEqual(origs, [])                                                             # no sidecar written in a dry run

    def test_the_repair_step_runs_before_the_write_back_and_its_files_are_committed(self):
        (self.jobdir / "regrow picks.json").write_text("{}\n", encoding="utf-8", newline="\n")
        (self.jobdir / "regrow origs.json").write_text("{}\n", encoding="utf-8", newline="\n")
        r = self.submit(repair=FakeRepair(self, writes=True))
        self.assertEqual(self.repair.calls, [{"mode": None, "after_sync": 0, "after_writeback": 0}])
        self.assertEqual(sorted(self.files_in("HEAD")), sorted([REL, f"jobs/t/{LEDGER}", "jobs/t/regrow origs.json",
                                                                "jobs/t/regrow picks.json", "jobs/t/repair log.jsonl",
                                                                "jobs/t/review.json"]))         # flag: the pass's files too
        self.assertIn("repair step: 1 repaired", git("log", "-1", "--format=%B", cwd=self.work))
        self.assertIn("## Repairs", r["digest"])
        self.assertIn('- s1w0 p1, repaired: cut: "With a slip."', r["digest"])
        self.assertIn("  after: Claude's.", r["digest"])
        self.assertIn("repair step: 1 repaired", self.post.calls[0][0])
        r = self.submit(wb=FakeWriteback(text=None))                                            # pass: nothing changed
        self.assertEqual(r["commits"], [])
        self.assertIsNone(self.post.calls[0][1])                                                # one line: clean reads only

    def test_an_error_in_the_repair_step_is_reported_and_the_submission_goes_on(self):
        r = self.submit(repair=FakeRepair(self, fail=True))
        self.assertIsNone(r["stopped"])
        self.assertIn("the repair step failed, and the write-back went on without it: RuntimeError", r["problems"][0])  # flag
        self.assertIn(NEWER, git("show", f"origin/main:{REL}", cwd=self.work))                # published anyway
        r = self.submit(repair_cfg={"mode": "off"})                                             # pass: off, never called
        self.assertEqual((self.repair.calls, r["problems"]), ([], []))

    def test_the_sidecar_is_made_sure_of_before_the_pull_and_a_failure_with_a_sync_due_stops_it(self):
        def broken(job, log=None):
            raise OSError("disk full")
        r = self.submit(origs=broken)                                                           # no sync due: a problem only
        self.assertIsNone(r["stopped"])
        self.assertIn("regrow origs.json couldn't be written (OSError: disk full)", r["problems"][0])
        order = []

        def origs(job, log=None):
            order.append(("origs", len(self.sync.calls)))
            return True
        self.edit_on_main(GUIDE.replace("Intro.", "Intro, edited by you."))
        self.submit(origs=origs)
        self.assertEqual(order, [("origs", 0)])                                                 # pass: before the sync
        self.edit_on_main(GUIDE.replace("Intro.", "Intro, edited by you again."))
        head = git("rev-parse", "HEAD", cwd=self.work)
        r = self.submit(origs=broken)
        self.assertIn("regrow origs.json couldn't be written (OSError: disk full), and a sync is due", r["stopped"])  # flag
        self.assertEqual((self.sync.calls, self.wb.calls, git("rev-parse", "HEAD", cwd=self.work)), ([], [], head))
        self.assertNotIn("edited by you again", self.guide.read_text(encoding="utf-8"))         # nothing pulled

    def test_ensure_origs_writes_the_sidecar_once_and_only_when_the_pass_has_a_bank(self):
        from engine.regrow import RegrowPass
        from test_regrow import fake_load_units
        job = Job(root=self.jobdir)
        with mock.patch.object(RegrowPass, "load_units", lambda rp, sections, vm=None: fake_load_units(rp, [5])):
            self.assertFalse(morning.ensure_origs(job, log=lambda *_: None))                    # pass: no bank, nothing to do
            self.assertFalse((self.jobdir / "regrow origs.json").exists())
            (self.jobdir / "regrow bank.jsonl").write_text("", encoding="utf-8")
            self.assertTrue(morning.ensure_origs(job, log=lambda *_: None))                     # flag: missing, written
            self.assertEqual(sorted(json.loads((self.jobdir / "regrow origs.json").read_text(encoding="utf-8"))),
                             ["s5w0 p1", "s5w0 p2"])
            self.assertFalse(morning.ensure_origs(job, log=lambda *_: None))                    # there now: left alone


class MainFirstAndThePostedLine(MorningRepo):
    """The step pulled out for the night run (main_first, then submit(early=...)), the night's lines at the top of the
    digest, the repair step turned off for one run, and the line that says whether Discord took the digest."""

    def main_first(self, dry=False):
        self.sync = FakeSync()
        return morning.main_first(Job(root=self.jobdir), repo=self.work, dry=dry, settings=settings(), sync_fn=self.sync,
                                  origs_fn=lambda job, log=None: False, today=DAY, stamp=STAMP, log=lambda *_: None)

    def submit_after(self, early, **kw):
        self.wb, self.post, self.repair = kw.pop("wb", None) or FakeWriteback(), Poster(), FakeRepair(self)
        later_sync = FakeSync()
        r = morning.submit(early["job"], repo=self.work, settings=settings(), writeback=self.wb, sync_fn=later_sync,
                           post=self.post, repair_fn=self.repair, origs_fn=lambda job, log=None: False, today=DAY,
                           stamp=STAMP, digests=self.digests, log=lambda *_: None, early=early, **kw)
        return r, later_sync

    def test_main_first_pulls_syncs_and_commits_and_the_submission_finds_nothing_more_to_pull(self):
        self.edit_on_main(GUIDE.replace("Intro.", "Intro, edited by you."))
        early = self.main_first()
        self.assertIsNone(early["stopped"])
        self.assertEqual((early["state"], early["synced"], len(early["commits"])), ("main ahead", True, 1))
        self.assertEqual(self.sync.calls, [{"write": True, "today": DAY}])
        self.assertIn("Intro, edited by you.", self.guide.read_text(encoding="utf-8"))
        self.assertEqual(git("rev-parse", "--short", "HEAD", cwd=self.work), early["commits"][0])  # committed, not pushed
        self.assertNotEqual(git("rev-parse", "origin/side", cwd=self.work), git("rev-parse", "HEAD", cwd=self.work))
        r, later_sync = self.submit_after(early)
        self.assertEqual(later_sync.calls, [])                                                  # pass: nothing more to pull
        self.assertEqual(self.wb.calls, [{"write": True, "after_sync": True}])                 # the night's sync counts
        self.assertEqual(r["commits"][0], early["commits"][0])
        self.assertEqual(len(r["commits"]), 2)
        self.assertEqual(r["pushed"], "side")
        self.assertIn("your edit on main synced in first", git("log", "-1", "--format=%B", cwd=self.work))
        self.assertIn("Main's copy and this branch's at the start: main ahead. Main's copy was pulled and synced in.",
                      r["digest"])
        self.assertIn("## What the sync took in", r["digest"])
        self.assertIn("Intro, edited by you.", git("show", f"origin/main:{REL}", cwd=self.work))

    def test_the_night_s_sync_commit_is_pushed_even_when_the_write_back_has_nothing_new(self):
        self.edit_on_main(GUIDE.replace("Intro.", "Intro, edited by you."))
        early = self.main_first()
        (self.jobdir / "regrow bank.jsonl").unlink()           # the fake sync's stand-in for another job's file
        r, _ = self.submit_after(early, wb=FakeWriteback(text=None))
        self.assertEqual((r["commits"], r["pushed"]), (early["commits"], "side"))               # flag: pushed anyway
        self.assertEqual(git("rev-parse", "origin/side", cwd=self.work), git("rev-parse", "HEAD", cwd=self.work))

    def test_main_first_says_what_it_would_do_in_a_dry_run_and_stops_on_a_dirty_guide(self):
        self.edit_on_main(GUIDE.replace("Intro.", "Intro, edited by you."))
        head = git("rev-parse", "HEAD", cwd=self.work)
        early = self.main_first(dry=True)
        self.assertEqual((self.sync.calls, early["commits"], early["synced"]), ([], [], False))
        self.assertIn("It would pull main's copy", early["pulled"])
        self.assertEqual(git("rev-parse", "HEAD", cwd=self.work), head)                         # pass: nothing done
        self.guide.write_text(GUIDE + "\nA line being typed.\n", encoding="utf-8", newline="\n")
        early = self.main_first()
        self.assertIn("aren't committed", early["stopped"])                                     # flag: the same refusal
        self.assertEqual(self.sync.calls, [])

    def test_the_night_s_lines_open_the_digest_in_every_form(self):
        night = {"lines": ["- The paragraph pass: skipped, a late start."], "short": "pass skipped"}
        self.wb, self.post, self.repair = FakeWriteback(text=None), Poster(), FakeRepair(self)
        r = morning.submit(Job(root=self.jobdir), repo=self.work, settings=settings(), writeback=self.wb,
                           sync_fn=FakeSync(), post=self.post, repair_fn=self.repair,
                           origs_fn=lambda job, log=None: False, today=DAY, stamp=STAMP, digests=self.digests,
                           log=lambda *_: None, night=night)
        self.assertTrue(r["digest"].startswith("# The morning submission, 2026-11-15 06:10\n\n## The night run\n\n"
                                               "- The paragraph pass: skipped, a late start.\n"))   # flag: no one-liner
        (content, name, _), = self.post.calls
        self.assertIsNotNone(name)
        self.assertIn("**depangram morning submission, 2026-11-15:** the night: pass skipped; 0 paragraphs", content)
        self.guide.write_text(GUIDE + "\nA line being typed.\n", encoding="utf-8", newline="\n")
        r = morning.submit(Job(root=self.jobdir), repo=self.work, settings=settings(), writeback=self.wb,
                           sync_fn=FakeSync(), post=self.post, repair_fn=self.repair,
                           origs_fn=lambda job, log=None: False, today=DAY, stamp=STAMP, digests=self.digests,
                           log=lambda *_: None, night=night)
        self.assertIn("## The night run", r["digest"].split("Stopped:")[0])                     # pass: above the stop

    def test_repair_mode_off_for_one_run_skips_the_repair_step(self):
        r = self.submit()
        self.assertEqual(len(self.repair.calls), 1)                                             # pass: config's mode
        self.wb, self.post, self.repair = FakeWriteback(), Poster(), FakeRepair(self)
        morning.submit(Job(root=self.jobdir), repo=self.work, settings=settings(), writeback=self.wb,
                       sync_fn=FakeSync(), post=self.post, repair_fn=self.repair,
                       origs_fn=lambda job, log=None: False, today=DAY, stamp=STAMP, digests=self.digests,
                       log=lambda *_: None, repair_mode="off")
        self.assertEqual(self.repair.calls, [])                                                 # flag: off tonight

    def test_the_posted_line_says_no_and_why_and_never_undoes_the_submission(self):
        name = "2026-11-15 morning submission.md"
        r = self.submit(post=Poster(fail=True))
        self.assertEqual(r["posted_line"], "Posted on Discord: no, the post failed (ConnectionError)")   # flag
        self.assertTrue((self.digests / name).read_text(encoding="utf-8").endswith(
            "\nPosted on Discord: no, the post failed (ConnectionError)\n"))
        self.assertEqual(len(r["commits"]), 1)
        self.assertNotIn("no network", r["posted_line"])                                        # the class only
        r = self.submit(post=lambda *a, **k: False, wb=FakeWriteback(text=None))
        self.assertEqual(r["posted_line"], "Posted on Discord: no, Discord didn't take it (the post came back empty)")
        r = self.submit(cfg={"post": False}, wb=FakeWriteback(text=None))
        self.assertEqual(r["posted_line"], "Posted on Discord: no, [morning] post is off")
        self.assertTrue((self.digests / name).read_text(encoding="utf-8").endswith(
            "Posted on Discord: no, [morning] post is off\n"))
        r = self.submit(dry=True)
        self.assertEqual(r["posted_line"], "Posted on Discord: no, a dry run")                  # nothing saved in a dry run
        r = self.submit(wb=FakeWriteback(text=None))
        self.assertEqual(r["posted_line"], "Posted on Discord: yes")                            # pass

    def test_take_main_off_stops_before_any_pull_when_main_moved_and_changes_nothing(self):
        self.edit_on_main(GUIDE.replace("Intro.", "Intro, edited by you."))
        head, origs = git("rev-parse", "HEAD", cwd=self.work), []
        main_before = git("rev-parse", "HEAD", cwd=self.yours)                                 # main with your edit
        r = self.submit(cfg={"take_main": False}, origs=lambda job, log=None: origs.append(job) or True)
        self.assertIn("main has an edit of yours (main ahead), and taking an edit on main in is switched off", r["stopped"])
        self.assertIn("Nothing was pulled, synced, written, committed or published: your edit on main is untouched and"
                      " stays as it is, and the bold and the comments stay in the document one more day", r["stopped"])
        self.assertEqual((self.sync.calls, self.wb.calls, self.repair.calls, origs, r["commits"]), ([], [], [], [], []))
        self.assertEqual((git("rev-parse", "HEAD", cwd=self.work), git("rev-parse", "origin/main", cwd=self.work)),
                         (head, main_before))                                                   # flag: nothing moved
        self.assertEqual(self.guide.read_text(encoding="utf-8"), GUIDE)
        self.assertIn("Stopped: main has an edit of yours", (self.digests / "2026-11-15 morning submission.md")
                      .read_text(encoding="utf-8"))
        self.assertIn("stopped: main has an edit of yours", self.post.calls[0][0])              # posted as the digest
        self.sync = FakeSync()
        early = morning.main_first(Job(root=self.jobdir), repo=self.work, settings=settings(take_main=False),
                                   sync_fn=self.sync, origs_fn=lambda job, log=None: False, today=DAY, stamp=STAMP,
                                   log=lambda *_: None)
        self.assertIn("switched off ([morning] take_main = false)", early["stopped"])            # main first too
        self.assertEqual((self.sync.calls, early["commits"]), ([], []))
        r = self.submit(cfg={"take_main": True})
        self.assertEqual((r["stopped"], r["state"], len(self.sync.calls)), (None, "main ahead", 1))  # pass: on, taken in

    def test_take_main_off_changes_nothing_when_main_has_not_moved(self):
        r = self.submit(cfg={"take_main": False})
        self.assertIsNone(r["stopped"])                                                         # pass: as before
        self.assertIn(NEWER, git("show", f"origin/main:{REL}", cwd=self.work))

    def test_the_night_with_take_main_off_skips_the_pass_and_posts_the_stop(self):
        import datetime as dt
        import functools
        from engine import night
        self.edit_on_main(GUIDE.replace("Intro.", "Intro, edited by you."))
        self.wb, self.sync, self.post, self.repair = FakeWriteback(), FakeSync(), Poster(), FakeRepair(self)
        s = types.SimpleNamespace(config={"morning": {"take_main": False}, "night": {"branch": "side"},
                                          "git": {"publish": True}},
                                  get=lambda k, d="": d, runs_dir=Path(self.tmp.name))
        led = types.SimpleNamespace(spent=lambda since: (0.0, 0.0), week=lambda: (0.1, 0.9))
        spawned = []
        kw = dict(sync_fn=self.sync, origs_fn=lambda job, log=None: False)
        n = night.Night(Job(root=self.jobdir), s, repo=self.work, now=lambda: dt.datetime(2026, 11, 16, 2, 0),
                        sleep=lambda x: None, ledger=lambda: led, busy_fn=lambda: None, branch_fn=lambda: "side",
                        spawn_fn=lambda cmd, out: spawned.append(cmd), port_fn=lambda: False,
                        main_first=functools.partial(morning.main_first, **kw),
                        submit=functools.partial(morning.submit, repo=self.work, writeback=self.wb, post=self.post,
                                                 repair_fn=self.repair, digests=self.digests, **kw),
                        stop_file=Path(self.tmp.name) / "STOP", log_path=Path(self.tmp.name) / "night.log",
                        out=lambda *_: None)
        res = n.run()
        self.assertEqual((spawned, self.sync.calls, self.wb.calls), ([], [], []))                # flag: no pass, no pull
        self.assertIn("skipped, main first stopped", "\n".join(res["lines"]))
        sub = res["submission"]
        self.assertIn("switched off ([morning] take_main = false)", sub["stopped"])
        self.assertIn("## The night run", sub["digest"])
        self.assertIn("Stopped: main has an edit of yours", sub["digest"])
        self.assertIn("stopped: main has an edit of yours", self.post.calls[0][0])
        self.assertNotIn("edited by you", self.guide.read_text(encoding="utf-8"))

    def test_main_prints_the_digest_and_the_posted_line(self):
        out = []
        fake = {"digest": "the digest", "posted_line": "Posted on Discord: yes", "stopped": None, "problems": []}
        from engine import main_copy
        with mock.patch.object(morning, "submit", lambda job, dry=False, **kw: fake), \
                mock.patch.object(main_copy, "load_job", lambda name, job_dir, settings: name), \
                mock.patch("builtins.print", lambda *a, **k: out.append(" ".join(map(str, a)))):
            self.assertEqual(morning.main(["--dry"]), 0)
        self.assertEqual(out, ["the digest", "Posted on Discord: yes"])


class GitSettings(MorningRepo):
    """[git] from config.toml: publishing off by default (a commit on the work branch, no fetch, push or publish),
    the repo found from the job's document, a job folder outside that repo or ignored by it, the work branch, and the
    notifier named in the posted line. Each check has a flag case and a pass case."""

    def bare(self, git_cfg=None, **morning_cfg):
        cfg = {"morning": morning_cfg}
        if git_cfg is not None:
            cfg["git"] = git_cfg
        return types.SimpleNamespace(config=cfg)

    def run_with(self, s, job=None, repo="given", dry=False):
        self.wb, self.sync, self.post, self.repair = FakeWriteback(), FakeSync(), Poster(), FakeRepair(self)
        return morning.submit(job or Job(root=self.jobdir), repo=self.work if repo == "given" else repo, settings=s,
                              dry=dry, writeback=self.wb, sync_fn=self.sync, post=self.post, repair_fn=self.repair,
                              origs_fn=lambda job, log=None: False, today=DAY, stamp=STAMP, digests=self.digests,
                              log=lambda *_: None)

    def test_publishing_is_off_by_default_and_the_morning_only_commits(self):
        self.edit_on_main(GUIDE.replace("Intro.", "Intro, edited by you."))              # main moved: not pulled
        side, main = (git("rev-parse", f"origin/{b}", cwd=self.work) for b in ("side", "main"))
        r = self.run_with(self.bare())
        self.assertIsNone(r["stopped"])
        self.assertEqual((len(r["commits"]), r["pushed"], r["published"], r["state"]), (1, None, None, "not published"))
        self.assertEqual(self.sync.calls, [])                                             # flag: nothing pulled or synced
        self.assertEqual(git("rev-parse", "origin/side", cwd=self.work), side)            # not pushed
        self.assertEqual(git("rev-parse", "origin/main", cwd=self.work), main)            # not fetched or published
        self.assertIn(NEWER, self.guide.read_text(encoding="utf-8"))
        self.assertIn("Publishing is off ([git] publish)", r["digest"])
        self.assertIn("not published ([git] publish is off)", r["digest"])
        made = []
        self.wb, self.sync, self.post, self.repair = FakeWriteback(text=None), FakeSync(), Poster(), FakeRepair(self)
        morning.submit(Job(root=self.jobdir), repo=self.work, settings=self.bare(), writeback=self.wb, sync_fn=self.sync,
                       post=self.post, repair_fn=self.repair, origs_fn=lambda job, log=None: made.append(job) or True,
                       today=DAY, stamp=STAMP, digests=self.digests, log=lambda *_: None)
        self.assertEqual(len(made), 1)                                                    # the sidecar still made sure of
        r = self.run_with(self.bare({"publish": True}))                                     # pass: on, it pulls
        self.assertEqual((r["state"], len(self.sync.calls)), ("diverged", 1))

    def test_push_follows_publish_unless_it_is_set(self):
        r = self.run_with(self.bare({"push": True}))
        self.assertEqual((r["pushed"], r["published"]), ("side", None))                   # pushed, not published
        self.assertEqual(git("rev-parse", "origin/side", cwd=self.work), git("rev-parse", "HEAD", cwd=self.work))

    def test_an_older_morning_publish_still_counts_and_git_wins(self):
        from engine.main_copy import git_config
        self.assertTrue(git_config(self.bare(publish=True))["publish"])
        self.assertTrue(git_config(self.bare(publish=True))["push"])
        self.assertFalse(git_config(self.bare({"publish": False}, publish=True))["publish"])
        self.assertEqual(git_config(None)["publish_branch"], "main")
        self.assertFalse(git_config(None)["publish"])

    def test_the_repo_is_found_from_the_document(self):
        from engine.main_copy import NoRepo, repo_for
        job = Job(root=self.jobdir)
        self.assertEqual(repo_for(job), self.work.resolve())
        r = self.run_with(self.bare(), repo=None)
        self.assertEqual((r["stopped"], len(r["commits"])), (None, 1))                    # pass: committed there
        lone = Path(self.tmp.name) / "lone"
        lone.mkdir()
        (lone / "doc.md").write_text(GUIDE, encoding="utf-8")
        job.real_file = lone / "doc.md"
        with self.assertRaises(NoRepo):                                                   # flag: not a repo
            repo_for(job)
        r = self.run_with(self.bare(), job=job, repo=None)
        self.assertIn("isn't in a git repo", r["stopped"])
        job.real_file = None
        self.assertIn("names no document", self.run_with(self.bare(), job=job, repo=None)["stopped"])

    def test_a_job_folder_outside_the_repo_or_ignored_stays_out_of_the_commit(self):
        import shutil
        outside = Path(self.tmp.name) / "jobs elsewhere" / "t"
        shutil.copytree(self.jobdir, outside)
        r = self.run_with(self.bare(), job=Job(root=outside))
        self.assertEqual((r["stopped"], r["problems"]), (None, []))
        self.assertEqual(self.files_in("HEAD"), [REL])                                    # pass: the document alone
        (self.work / ".gitignore").write_text("jobs/*/repair log.jsonl\n", encoding="utf-8")
        git("add", ".gitignore", cwd=self.work)
        git("commit", "-m", "ignore", cwd=self.work)
        self.guide.write_text(GUIDE, encoding="utf-8", newline="\n")
        git("commit", "-am", "back", cwd=self.work)
        self.wb, self.sync, self.post = FakeWriteback(), FakeSync(), Poster()
        r = morning.submit(Job(root=self.jobdir), repo=self.work, settings=self.bare(), writeback=self.wb,
                           sync_fn=self.sync, post=self.post, repair_fn=FakeRepair(self, writes=True),
                           origs_fn=lambda job, log=None: False, today=DAY, stamp=STAMP, digests=self.digests,
                           log=lambda *_: None)
        self.assertEqual((r["stopped"], r["problems"]), (None, []))
        files = self.files_in("HEAD")
        self.assertIn("jobs/t/review.json", files)                                         # pass: not ignored
        self.assertNotIn("jobs/t/repair log.jsonl", files)                                 # flag: ignored, left out

    def test_the_work_branch_stops_a_checkout_on_another(self):
        r = self.run_with(self.bare({"work_branch": "nightly"}))
        self.assertIn("not the work branch nightly", r["stopped"])                      # flag
        self.assertEqual(self.wb.calls, [])
        self.assertIsNone(self.run_with(self.bare({"work_branch": "side"}))["stopped"])    # pass
        on_side = {"publish_branch": "side"}                                             # this checkout is the publish branch
        self.assertIsNone(self.run_with(self.bare(on_side), dry=True)["stopped"])         # off: just a branch
        self.assertIn("only commits on a work branch",
                      self.run_with(self.bare(dict(on_side, publish=True)), dry=True)["stopped"])

    def test_the_posted_line_names_the_notifier(self):
        from engine.settings import Settings
        r = self.run_with(Settings(env={"NOTIFIER": "none"}, config={}))
        self.assertEqual(r["posted_line"], "Posted: no, the notifier is none (NOTIFIER)")
        self.assertEqual(self.post.calls, [])
        r = self.run_with(Settings(env={"NOTIFIER": "file"}, config={}))
        self.assertEqual(r["posted_line"], "Written to the notifications folder: yes")


class AtTheLoopsDigest(unittest.TestCase):
    def fake_loop(self, on=True, rehearsal=False, smoke=False, window=None):
        logs = []
        return types.SimpleNamespace(morning_on=on, rehearsal=rehearsal, args=types.SimpleNamespace(smoke=smoke, window=window),
                                     job="the job", settings="the settings", log=logs.append, logs=logs)

    def test_it_runs_on_a_real_run_and_is_skipped_everywhere_else(self):
        calls = []
        submit = lambda job, **kw: calls.append((job, kw["settings"])) or {"stopped": None}  # noqa: E731
        lp = self.fake_loop()
        self.assertEqual(morning.run_for_loop(lp, submit_fn=submit), {"stopped": None})       # pass: a real run
        self.assertEqual(calls, [("the job", "the settings")])
        for lp, crashed, why in [(self.fake_loop(on=False), None, "at_digest is off"),        # flag: each skip
                                 (self.fake_loop(), "boom", "the day crashed"),
                                 (self.fake_loop(rehearsal=True), None, "a rehearsal, smoke or partial run"),
                                 (self.fake_loop(smoke=True), None, "a rehearsal, smoke or partial run"),
                                 (self.fake_loop(window="s1w0"), None, "a rehearsal, smoke or partial run")]:
            self.assertEqual(morning.run_for_loop(lp, crashed, submit_fn=submit), {"skipped": why})
            self.assertEqual(lp.morning_result, {"skipped": why})
        self.assertEqual(len(calls), 1)
        del lp.morning_on                                                                       # a test double never submits
        self.assertEqual(morning.run_for_loop(lp, submit_fn=submit)["skipped"], "at_digest is off")

    def finishing_loop(self, d, rehearsal=False):
        from engine.run import Loop
        lp = Loop.__new__(Loop)
        lp.timeline, lp.rehearsal, lp.day, lp.morning_on, lp.morning_result = [], rehearsal, DAY, True, None
        lp.log = lp.timeline.append
        for name in ("stop_servers", "read_answers", "ask_open_cuts", "write_working_copy", "wrap_up"):
            setattr(lp, name, lambda *a, **k: None)
        lp.clock = types.SimpleNamespace(now=lambda: 1000.0)
        lp.schedule = types.SimpleNamespace(bounds=lambda now: {"start": 0, "claude_until": 0}, day_key=lambda now: DAY)
        lp.budget = types.SimpleNamespace(measure_rates=lambda *a: {}, day_finished=lambda day: None)
        lp.args = types.SimpleNamespace(smoke=False, window=None, digests_dir=str(d))
        lp.job = types.SimpleNamespace(couldnt_crack=Path(d) / "couldnt crack.md")
        lp.settings = settings()
        sent = []
        lp.notifier = types.SimpleNamespace(send=lambda content, **k: sent.append(content))
        return lp, sent

    def test_the_loop_calls_it_after_its_digest_and_an_error_there_takes_nothing_with_it(self):
        from engine import discord
        seen = []
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(discord, "digest_report", lambda *a, **k: "the report"), \
                mock.patch.object(discord, "digest_card", lambda *a, **k: "the card"), \
                mock.patch.object(discord, "couldnt_crack_report", lambda *a, **k: "none"), \
                mock.patch("engine.restore.run_for_loop", lambda lp: None), \
                mock.patch("engine.reflect.run_for_loop", lambda lp: None):
            lp, sent = self.finishing_loop(d)

            def boom(job, **kw):
                seen.append(list(sent))
                raise RuntimeError("the submission broke")
            with mock.patch.object(morning, "submit", boom):
                lp.finish()                                                                     # flag: it raised, finish didn't
            self.assertEqual((seen, sent), ([["the card"]], ["the card"]))                      # after the loop's digest
            self.assertIn("the morning submission didn't run: the submission broke", lp.timeline[-1])
            self.assertIn("an error", lp.morning_result["skipped"])
            lp, sent = self.finishing_loop(d, rehearsal=True)
            with mock.patch.object(morning, "submit", boom):
                lp.finish()                                                                     # pass: a rehearsal skips it
            self.assertEqual((len(seen), sent, lp.morning_result), (1, ["the card"],
                                                                    {"skipped": "a rehearsal, smoke or partial run"}))


class StuckEntryLines(unittest.TestCase):
    """A stuck entry in review.json that speaks for no unit reaches the reviewer through the digest."""

    def r(self, changed, lost):
        rep = {"replaced": [], "marked": [], "same": [], "restored": [], "yours": [], "skipped": [], "missing": [],
               "for_you": [], "prompts": {}, "repaired": [], "stuck_unmatched": lost}
        return {"dry": False, "day": "2026-11-16", "stamp": "2026-11-16 06:10", "stopped": None, "changed": changed,
                "pulled": None, "commits": [], "published": None, "problems": [], "repair_news": False, "night": None,
                "report": rep, "state": "the same", "notes": {}, "repair_lines": ["- 1 repaired"], "sync": None,
                "branch": "work", "pushed": False, "publish_note": None}

    def test_it_goes_under_the_repairs_and_in_the_one_line_digest(self):
        line = "the stuck entry for s4w1 p3 matches no unit now: write a new one or remove it"
        full = morning.digest(self.r(True, ["s4w1 p3"]))
        self.assertIn("## Repairs\n\n- 1 repaired\n- " + line + "\n", full)                     # flag
        self.assertEqual(full.count("s4w1 p3"), 1)                     # not again under the write-back
        self.assertIn(" The" + line[3:] + ".", morning.digest(self.r(False, ["s4w1 p3"])))
        self.assertNotIn("stuck entry", morning.digest(self.r(True, [])))                       # pass


if __name__ == "__main__":
    unittest.main()
