"""Tests for a job that is any markdown document (Phase 2): job.toml's fields, the section styles, and a fresh job
on a synthetic document (tests/fixtures/balcony) run end to end and dry: the layout, the units, a dry write-back and
a dry sync, with nothing in any of it about an older job. Each check has a flag case and a pass case. Run from the
repo root:

    python -m unittest discover tests
"""
import shutil
import tempfile
import unittest
from pathlib import Path

from engine.job import NUMBERED, Bank, Job, JobError, SectionStyle, job_name, plain_samples

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "balcony"
DESCRIPTION = "a short practical how-to on growing herbs on a balcony"
OLD_JOB_WORDS = ("ai-guide", "ai guide", "guide")

# a numbered document with a heading that isn't numbered between two sections, and one at the end
NUMBERED_DOC = ("# Title\n\nIntro.\n\n## 1. One\n\nA.\n\n### Sub\n\nB.\n\n## Aside\n\nX.\n\n## 2. Two\n\nC.\n\n"
                "## Closing\n\nZ.\n")


class FreshCopy:
    """The fixture job copied into a temporary folder, so setup() and a stray write never touch the repo."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "balcony"
        shutil.copytree(FIXTURE, self.dir)
        return self.dir

    def __exit__(self, *exc):
        self.tmp.cleanup()


def texts_of(*parts):
    """Everything given as one lower-case string, to look for an older job's name in it."""
    return "\n".join(str(p) for p in parts).lower()


# ---------------------------------------------------------------------- the section styles
class Styles(unittest.TestCase):
    def test_numbered_default_keeps_both_old_end_rules(self):
        texts = NUMBERED.texts(NUMBERED_DOC, [1, 2])
        # texts(): section 1 runs on to section 2 across "## Aside"; the last ends at the first unnumbered "## "
        self.assertTrue(texts[1].startswith("## 1. One") and "## Aside" in texts[1])                       # flag
        self.assertEqual(texts[2], "## 2. Two\n\nC.\n\n")                                                 # pass
        spans = NUMBERED.spans(NUMBERED_DOC, [1, 2])
        # spans(): every section ends at the next "## " heading of any kind
        self.assertEqual(NUMBERED_DOC[slice(*spans[1])], "## 1. One\n\nA.\n\n### Sub\n\nB.\n\n")
        self.assertEqual(NUMBERED_DOC[slice(*spans[2])], "## 2. Two\n\nC.\n\n")
        self.assertEqual(NUMBERED.ids(NUMBERED_DOC), [1, 2])
        self.assertTrue(SectionStyle(pattern=r"^## (\d+)\.\s").numbered)                                   # the same style

    def test_a_heading_level_counts_sections_by_place(self):
        md = "# T\n\n### Part one\n\nA.\n\n#### Deep\n\nD.\n\n### Part two\n\nB.\n\n## Back\n\nZ.\n\n### Part three\n\nC.\n"
        st = SectionStyle(level=3)
        self.assertEqual(st.ids(md), [1, 2, 3])
        self.assertIn("#### Deep", st.texts(md, [1])[1])                    # pass: a deeper heading stays inside
        self.assertEqual(st.texts(md, [2])[2], "### Part two\n\nB.\n\n")    # flag: a shallower heading ends it
        self.assertEqual(st.titles(md), {1: "Part one", 2: "Part two", 3: "Part three"})
        self.assertEqual(st.texts(md, [1, 2, 3]), {n: md[slice(*sp)] for n, sp in st.spans(md, [1, 2, 3]).items()})

    def test_a_pattern_takes_its_number_group_or_the_place(self):
        md = "Intro\n\nPART ONE\n\nA.\n\nPART TWO\n\nB.\n"
        self.assertEqual(SectionStyle(pattern=r"^PART [A-Z]+$").texts(md, [2]), {2: "PART TWO\n\nB.\n"})   # place
        md2 = "## Step 5: mix\n\nA.\n\n## Step 9: bake\n\nB.\n"
        self.assertEqual(SectionStyle(pattern=r"^## Step (\d+):").ids(md2), [5, 9])                       # its number

    def test_a_bad_style_is_named(self):
        with self.assertRaises(JobError):
            SectionStyle(pattern="^## (", level=None)                                                      # flag
        with self.assertRaises(JobError):
            SectionStyle(pattern="^## ", level=2)                                                          # flag
        with self.assertRaises(JobError):
            SectionStyle(level=9)                                                                          # flag
        self.assertEqual(SectionStyle(level=2).level, 2)                                                   # pass


# ---------------------------------------------------------------------- job.toml
class Fields(unittest.TestCase):
    def test_the_job_name_comes_from_the_flag_or_config(self):
        self.assertEqual(job_name("mine", {}), "mine")                                                     # pass
        self.assertEqual(job_name(None, {"job": {"default": "essay"}}), "essay")                          # pass
        with self.assertRaises(JobError) as cm:
            job_name(None, {})                                                                             # flag
        self.assertIn("--job", str(cm.exception))
        self.assertIn("[job]", str(cm.exception))
        from unittest import mock
        with mock.patch("engine.settings.load_config", lambda: {"job": {"default": "from-config"}}):
            self.assertEqual(job_name(None), "from-config")                                               # config.toml
        with mock.patch("engine.settings.load_config", lambda: {}), self.assertRaises(JobError):
            job_name(None)
        with mock.patch("engine.settings.load_config", side_effect=OSError("no config.toml")), \
                self.assertRaises(JobError):
            job_name(None)

    def test_a_heading_the_splitter_cuts_stays_locked(self):
        with FreshCopy() as d:
            md = (d / "original.md").read_text(encoding="utf-8")
            (d / "original.md").write_text(md.replace("### Part two: light", "### Light? Mostly sun"), encoding="utf-8")
            job = Job(root=d)
            self.assertTrue({"Light? Mostly sun", "Light?", "Mostly sun"} <= job.headings())
            from engine.wrapup import layout
            w = layout(job, Bank(job.bank_path))["s2w0"]
            self.assertFalse(any(r["claude"] or r.get("author") for r in w["runs"][0]))    # never regrown

    def test_an_old_job_toml_still_loads(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "job.toml").write_text('[job]\nname = "t"\noriginal = "o.md"\nworking = "w.md"\nsections = [1, 2]\n'
                                        'voice_samples = [9]\nreal_file = "somewhere/doc.md"\n', encoding="utf-8")
            (d / "o.md").write_text(NUMBERED_DOC, encoding="utf-8")
            job = Job(root=d)
            from engine.settings import SANDBOX
            self.assertTrue(job.style.numbered)                                           # the numbered default
            self.assertEqual((job.sections, job.voice_samples, job.description), ([1, 2], ["sample 9"], None))
            self.assertEqual(job.real_file, (SANDBOX / "somewhere/doc.md").resolve())    # old: from the repo root
            self.assertEqual(job.verdict_source, "scan")
            self.assertEqual([w["wid"] for w in job.windows()], ["s1w0", "s2w0"])

    def test_document_is_relative_to_the_job_folder(self):
        with FreshCopy() as d:
            job = Job(root=d)
            self.assertEqual(job.real_file, (d / "document.md").resolve())
            self.assertEqual(job.document, job.real_file)

    def test_scope_all_leaves_out_a_section_still_in_notes(self):
        with FreshCopy() as d:
            md = (d / "original.md").read_text(encoding="utf-8")
            (d / "original.md").write_text(md.replace("Rosemary forgives", "- a note to self\n\nRosemary forgives"),
                                           encoding="utf-8")
            self.assertEqual(Job(root=d).sections, [1, 2])                                                # flag
            toml = (d / "job.toml").read_text(encoding="utf-8")
            (d / "job.toml").write_text(toml.replace("[job]\n", "[job]\nskip_list_sections = false\n"), encoding="utf-8")
            self.assertEqual(Job(root=d).sections, [1, 2, 3])                                             # pass

    def test_verdicts_from_a_file_that_isnt_there_mean_every_line_is_the_machines(self):
        with FreshCopy() as d:
            (d / "your lines.md").unlink()
            self.assertEqual(Job(root=d).verdicts(), {})                                                  # pass
            toml = (d / "job.toml").read_text(encoding="utf-8")
            (d / "job.toml").write_text(toml.replace('verdicts = "file"', 'verdicts = "scan"'), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                Job(root=d).verdicts()                                                                     # flag
            (d / "job.toml").write_text(toml.replace('verdicts = "file"', 'verdicts = "chat"'), encoding="utf-8")
            with self.assertRaises(JobError):
                Job(root=d)                                                                                # flag

    def test_plain_samples_split_at_headings(self):
        self.assertEqual(plain_samples("# A\n\none\n\n# B\n\ntwo\n"), [("sample 1", "one"), ("sample 2", "two")])
        self.assertEqual(plain_samples("just text\n"), [("sample 1", "just text")])

    @unittest.skipUnless((FIXTURE.parents[2] / "jobs" / "example" / "job.toml").exists(), "no job template")
    def test_the_job_template_loads(self):
        with FreshCopy() as d:
            shutil.copy(FIXTURE.parents[2] / "jobs" / "example" / "job.toml", d / "job.toml")
            job = Job(root=d)
            self.assertEqual((job.style.level, job.sections, job.verdict_source), (3, [1, 2, 3], "file"))
            self.assertTrue(job.description)

    def test_a_missing_job_says_how_to_make_one(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(JobError) as cm:
            Job(root=d)
        self.assertIn("jobs/README.md", str(cm.exception))


# ---------------------------------------------------------------------- the fresh job, end to end and dry
class FreshJob(unittest.TestCase):
    def test_layout_units_writeback_and_sync_run_dry(self):
        from engine.regrow import units
        from engine.sync import report_text, sync
        from engine.wrapup import layout
        from engine.writeback import paragraphs, report_lines, run
        with FreshCopy() as d:
            job = Job(root=d)
            self.assertEqual((job.style.level, job.sections), (3, [1, 2, 3]))
            self.assertEqual([w["wid"] for w in job.windows()], ["s1w0", "s2w0", "s3w0"])
            self.assertIn("Basil does not, and it says so by noon.", job.headings())     # a protected line is locked
            self.assertEqual([k for k, _ in job.samples()], ["sample 1", "sample 2"])
            swaps = job.setup()
            locks = (d / "locks.md").read_text(encoding="utf-8")
            self.assertEqual(swaps, [])

            bank = Bank(job.bank_path)
            lay = layout(job, bank)
            self.assertEqual(sorted(lay), ["s1w0", "s2w0", "s3w0"])
            heads = [w["original"][0] for _, w in sorted(lay.items())]
            self.assertEqual(heads, [[["Part one: pots"]], [["Part two: light"]], [["Part three: watering"]]])
            got, skipped = units(lay, job.sections, {})
            self.assertEqual([u["label"] for u in got], ["s1w0 p2", "s2w0 p2", "s2w0 p4", "s3w0 p2"])
            self.assertEqual([x for x, _ in skipped], ["s1w0 p3", "s3w0 p3"])            # yours, and a locked line
            rows = paragraphs(job, bank, lay, {}, job.sections)
            self.assertNotIn("s1w0 p1", [r["label"] for r in rows])                       # the heading is never written
            from engine.fixpass import gather
            fix_rows = gather(job, bank, job.real_file.read_text(encoding="utf-8"), job.sections, lay=lay)
            self.assertEqual([r["label"] for r in fix_rows], [r["label"] for r in rows])
            self.assertTrue(all(r["fixable"] for r in fix_rows), [r.get("why") for r in fix_rows])  # the document's cut

            before = job.real_file.read_bytes()
            new = "Clay pots let air through the walls, so the soil dries evenly. Plastic ones hold water longer."
            passed = {"paragraphs": {"s1w0 p2": {"text": new, "section": 1, "round": "s1 round 1", "ai": 0.1,
                                                 "flagged": False}}, "sections": {}}
            rep, notes, changed = run(job, write=False, stamp="2026-01-05 10:00", passed=passed)
            self.assertTrue(changed)                                                                      # flag
            self.assertEqual(rep["replaced"], ["s1w0 p2"])
            self.assertFalse(rep["written"])
            self.assertEqual(job.real_file.read_bytes(), before)                                         # pass: dry
            self.assertFalse((d / "written back.jsonl").exists())

            def no_scan(text):
                raise AssertionError("verdicts = \"file\" never runs the chat-log scan")
            toml_before = (d / "job.toml").read_bytes()
            r = sync(job.dir, job.real_file, write=False, classify=no_scan)
            self.assertEqual((r["sections_before"], r["sections_after"], r["written"]), ([1, 2, 3], [1, 2, 3], False))
            self.assertEqual([a for a, _ in r["kept"]], ["s1w0", "s2w0", "s3w0"])
            self.assertEqual((d / "job.toml").read_bytes(), toml_before)
            self.assertEqual(sorted(p.name for p in d.iterdir() if "before sync" in p.name), [])

            said = texts_of(locks, "\n".join(report_lines(rep)), report_text(r), job.summary(), job.headings(),
                            [w["text"] for w in job.windows()])
            for word in OLD_JOB_WORDS:
                self.assertNotIn(word, said, word)

    def test_a_rehearsal_copies_the_jobs_own_files(self):
        import types
        from unittest import mock
        from engine import rehearsal
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shutil.copytree(FIXTURE, root / "jobs" / "balcony")
            with mock.patch("engine.settings.SANDBOX", root):
                rehearsal.prepare_dir(types.SimpleNamespace(env={}), root / "runs", "balcony")
            got = sorted(p.name for p in (root / "runs" / "jobs" / "balcony").iterdir())
            self.assertEqual(got, ["job.toml", "original.md", "protected lines.txt", "samples.md", "your lines.md"])
            self.assertTrue((root / "runs" / "document.md").exists())          # the stand-in for the document

    def test_the_description_reaches_the_prompts(self):
        from engine.draws import judge_pool
        prompts = []

        def claude(settings, conn, prompt, **kw):
            prompts.append(prompt)
            if "notes its writer" in prompt:
                return "### 1\n- pots breathe\n", {}
            if "Reply with only A, B or SAME" in prompt:
                return "A", {}
            return '{"pick": 1, "drops": [], "why": "fine"}' if "Pick the best" in prompt else "NONE", {}
        with FreshCopy() as d:
            job = Job(root=d)
            para = job.windows()[0]["text"].split("\n\n")[1]
            self.assertEqual(job.notes_for(claude, None, None, [para], role="t"), [["pots breathe"]])
            pool = {"stats": {}, "origin": {}, "ranked": [["A new line.", 0.1]], "shown": ["A new line."]}
            judge_pool(pool, settings=None, conn=None, tics=lambda c: (0, []), orig_text="Old line.",
                       cur_text="Old line.", is_original=True, notes=["n"], before="", after="", pi=0, ri=0, rnd=1,
                       rounds=2, role="t", claude=claude, description=job.description)
        self.assertEqual(len(prompts), 4)                         # notes, judge, head-to-head, fact check
        for p in prompts[:3]:
            self.assertIn(DESCRIPTION, p)                                                                   # pass
        self.assertNotIn("<<DESCRIPTION>>", "".join(prompts))                                              # flag


if __name__ == "__main__":
    unittest.main()
