"""Tests for the engine's own AI-tells scanner (engine/tics/), the house rules file (rules.toml, read by
engine/tics/house_rules.py and engine/checks.py), the voice baseline (engine/tics/voice_audit.py) and the
transcript reader (engine/find_lines.py). Every text here is made up for the test. Run from the repo root:

    python -m unittest discover tests
"""
import copy
import json
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from engine import checks, find_lines, idioms, texts
from engine.draws import make_tics
from engine.tics import FOLDER, house_rules, tic_patterns, tic_scan, voice_audit

EM, EN = "\u2014", "\u2013"                    # as escapes, so this file carries no dash itself
BANNED = "scaff" + "olding"                     # a banned word, built so the file never spells it
CHECKED = "frame" + "work"                      # one from the check list, the same way
CLEAN = ("The bus left at nine, so we walked to the market and bought bread. "
         "My sister paid for it, and I carried the bag home.")


class Scanner(unittest.TestCase):
    """Through make_tics, the way the engine loads the scanner: (hard hit count, check labels)."""

    @classmethod
    def setUpClass(cls):
        cls.tics = staticmethod(make_tics(FOLDER))

    def test_an_em_dash_an_en_dash_and_a_banned_word_are_hard(self):
        self.assertGreaterEqual(self.tics(f"We left early {EM} the bus was gone.")[0], 1)
        self.assertGreaterEqual(self.tics(f"Pages 3{EN}9 cover it.")[0], 1)
        self.assertGreaterEqual(self.tics(f"The {BANNED} of the essay holds up.")[0], 1)

    def test_a_stock_ai_phrase_is_hard(self):
        self.assertGreaterEqual(self.tics("Let's delve into what the numbers say.")[0], 1)
        self.assertGreaterEqual(self.tics("The garden was a testament to her patience.")[0], 1)

    def test_a_rule_of_three_is_a_check(self):
        hard, labels = self.tics("The report was long, dull and late.")
        self.assertEqual(hard, 0)
        self.assertIn("rule of three", labels)

    def test_clean_prose_passes(self):
        self.assertEqual(self.tics(CLEAN), (0, []))

    def test_a_tell_inside_quotes_is_left_to_dialogue_rules(self):
        # "the kind of" is a narration rule: inside quotes it's a character talking
        self.assertEqual(self.tics('She said, "It was the kind of day you forget."')[0], 0)
        self.assertGreaterEqual(self.tics("It was the kind of day you forget.")[0], 1)

    def test_the_interface_make_tics_needs(self):
        rules = tic_scan.compile_rules()
        self.assertTrue(all({"label", "level", "rx"} <= set(r) for r in rules))
        self.assertEqual([r["level"] for r in rules], sorted((r["level"] for r in rules), key=lambda x: x != "hard"))
        hits = tic_scan.scan_line(f"It moved {EM} fast.", rules, None)
        self.assertTrue(all(len(h) == 3 and isinstance(h[0], int) for h in hits))
        self.assertIn((10, "hard", "em dash"), hits)
        self.assertEqual(tic_patterns.HOUSE_ONLY, frozenset())

    def test_label_text_round_trips(self):
        text = f"# A heading\n\nThe {BANNED} held {EM} and the report was long, dull and late.\n"
        labeled, findings = tic_scan.label_text(text)
        self.assertIn("[hard: banned word]", labeled)
        self.assertIn("[hard: em dash]", labeled)
        self.assertEqual(re.sub(r" \[(?:hard|check): [^\]]*\]", "", labeled), text)
        self.assertTrue(all(n == 3 for n, _, _ in findings))

    def test_every_label_has_a_family(self):
        labels = {r["label"] for r in tic_patterns.RULES} | set(tic_patterns.PHRASE_FAMILIES)
        self.assertEqual(sorted(labels - tic_patterns.FAMILY_OF.keys()), [])


class Labels(unittest.TestCase):
    def test_the_scanner_emits_the_new_banned_word_labels(self):
        hits = tic_scan.scan_line(f"The {BANNED} and the {CHECKED}.", tic_scan.compile_rules())
        labels = {lab for _, _, lab in hits}
        self.assertIn("banned word", labels)
        self.assertIn("banned word (literal?)", labels)

    def test_the_reflection_names_banned_words(self):
        from engine import reflect as R
        word = R.banned_word_stems()[0]
        hits = R.sentence_hits(f"The {word} part is fine.", set())
        self.assertIn(("banned word", word), hits)


class RulesFile(unittest.TestCase):
    def tearDown(self):
        checks.use_rules()                     # back to the default file for every other test

    def test_the_shipped_file_matches_the_defaults(self):
        path = house_rules.ROOT / "rules.toml"
        with open(path, "rb") as f:
            self.assertEqual(house_rules.merge(house_rules.DEFAULTS, tomllib.load(f)), house_rules.DEFAULTS)
        self.assertEqual(house_rules.load_rules(path), house_rules.DEFAULTS)

    def test_a_job_file_changes_only_its_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rules.toml"
            p.write_text('[filler]\nwords = ["frankly"]\ncap = 0\n\n[locale]\nspelling = "any"\n', encoding="utf-8")
            got = house_rules.load_rules(p)
            want = copy.deepcopy(house_rules.DEFAULTS)
            want["filler"] = {"words": ["frankly"], "cap": 0}
            want["locale"]["spelling"] = "any"
            self.assertEqual(got, want)

            checks.use_rules(p)
            self.assertEqual((checks.FILLER_TAGS, checks.FILLER_CAP), (("frankly",), 0))
            self.assertEqual(checks.filler_count("Frankly, it fails, honestly."), 1)
            self.assertEqual(checks.section_filler_problem("It fails, frankly.", "It fails.",
                                                           checks.section_counts("Plain.")), "filler tag (frankly)")
            self.assertEqual(checks.gb_spellings("the colour"), [])
            self.assertEqual(checks.to_us("the colour"), "the colour")
        checks.use_rules()
        self.assertEqual(checks.FILLER_TAGS, ("honestly",))
        self.assertEqual(checks.gb_spellings("the colour"), ["colour"])

    def test_banned_words_and_dashes_follow_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rules.toml"
            p.write_text('[house]\ndashes = false\nbanned_words = ["synerg*"]\nbanned_words_check = []\n',
                         encoding="utf-8")
            checks.use_rules(p)
            self.assertEqual(checks.house_words(f"Synergies {EM} and {BANNED} -- fine."), ["Synergies"])
            self.assertEqual(checks.banned_words(), ["synerg"])

    def test_swears_and_furniture_come_from_the_file(self):
        self.assertEqual(checks.swear_counts("Damn, the damning shit."), (4, 2, 0))
        self.assertIsNotNone(checks.reddit_tell("TL;DR it works"))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rules.toml"
            p.write_text("[swears]\nwords = [\"heck\"]\nstrong = [\"heck\"]\nmultiplier = 2.0\n\n"
                         "[furniture]\npatterns = ['\\bPS:']\n", encoding="utf-8")
            checks.use_rules(p)
            self.assertEqual(checks.swear_counts("Heck, the shit."), (3, 1, 1))
            self.assertEqual(checks.SWEAR_MULTIPLIER, 2.0)
            self.assertIsNone(checks.reddit_tell("TL;DR it works"))
            self.assertIsNotNone(checks.reddit_tell("PS: it works"))

    def test_a_typo_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rules.toml"
            p.write_text("[filler]\nword = [\"x\"]\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                house_rules.load_rules(p)
            p.write_text("[fillers]\nwords = [\"x\"]\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                house_rules.load_rules(p)
            p.write_text("[locale]\nspelling = \"klingon\"\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                house_rules.load_rules(p)

    def test_the_word_pattern_syntax(self):
        rx = house_rules.word_regex(["walk*", "*ish", "cat"], ["walkman"])
        self.assertEqual(rx.findall("Walking a catfish, the cat walks, a walkman, reddish."),
                         ["Walking", "catfish", "cat", "walks", "reddish"])
        self.assertEqual(house_rules.word_regex([]).findall("anything"), [])

    def test_the_budgets_come_from_the_rules_and_an_old_config_wins_with_a_warning(self):
        said = []
        self.assertEqual(checks.budgets({"loop": {}}, warn=said.append), (1.5, 1, ("honestly",)))
        self.assertEqual(said, [])
        old = {"loop": {"swear_multiplier": 2.0, "filler_words": ["frankly"]}}
        self.assertEqual(checks.budgets(old, warn=said.append), (2.0, 1, ("frankly",)))
        self.assertEqual(len(said), 1)
        self.assertIn("swear_multiplier, filler_words", said[0])

    def test_a_job_loads_its_own_rules_file(self):
        from engine.job import Job, JobError
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "job.toml").write_text('[job]\nname = "t"\nrules = "my rules.toml"\n', encoding="utf-8")
            (Path(d) / "my rules.toml").write_text('[filler]\nwords = ["frankly"]\n', encoding="utf-8")
            job = Job(root=d)
            self.assertEqual(job.rules_file, Path(d) / "my rules.toml")
            self.assertEqual(checks.FILLER_TAGS, ("frankly",))
            (Path(d) / "my rules.toml").write_text('[filler]\nword = ["x"]\n', encoding="utf-8")
            with self.assertRaises(JobError):
                Job(root=d)
            (Path(d) / "job.toml").write_text('[job]\nname = "t"\nrules = "missing.toml"\n', encoding="utf-8")
            with self.assertRaises(JobError):
                Job(root=d)


def samples_file(d, n=5, words=120):
    """A made-up samples file: n samples of plain sentences, one tagged as AI-assisted."""
    lines = []
    for i in range(1, n + 1):
        lines.append(f"## Sample {i} - a note")
        lines.append("- **Medium:** a forum")
        if i == n:
            lines.append("- **Source:** AI-assisted")
        body = []
        while sum(len(x.split()) for x in body) < words:
            body.append(f"I think the {i} bus was late again, and it was not my fault that we missed the start of it."
                        if i % 2 else "So we went anyway, because the tickets were paid for and the night was warm.")
        lines.append(" ".join(body))
        lines.append("")
    p = Path(d) / "samples.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


class Voice(unittest.TestCase):
    def test_load_samples_drops_the_ai_tagged_one(self):
        with tempfile.TemporaryDirectory() as d:
            kept, dropped = voice_audit.load_samples(samples_file(d))
            self.assertEqual([s["num"] for s in kept], [1, 2, 3, 4])
            self.assertEqual([s["num"] for s in dropped], [5])
            self.assertEqual(len(texts.load_samples(samples_file(d))), 4)

    def test_the_baseline_is_built_cached_and_measured(self):
        with tempfile.TemporaryDirectory() as d:
            p = samples_file(d, n=13, words=320)
            cache = Path(d) / "base.json"
            b = voice_audit.cached_baseline(p, cache)
            self.assertTrue(cache.exists())
            self.assertGreaterEqual(len(b["vecs"]), 3)
            self.assertEqual(voice_audit.cached_baseline(p, cache), json.loads(cache.read_text())["baseline"])
            line = voice_audit.audit("I think the bus was late, and we went anyway. " * 30, p, cache)
            self.assertTrue(line.startswith("DELTA (soft): "), line)

    def test_no_samples_no_line(self):
        with mock.patch.object(voice_audit, "_setting", return_value=None):
            self.assertEqual(voice_audit.audit("Some text."), "")
        self.assertEqual(voice_audit.delta_line("Some text.", {}), "")
        with tempfile.TemporaryDirectory() as d:          # too few chunks to judge by
            p = samples_file(d, n=2, words=50)
            self.assertEqual(voice_audit.audit("I think so. " * 40, p, Path(d) / "c.json"), "")

    def test_the_script_prints_one_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = samples_file(d, n=13, words=320)
            r = subprocess.run([sys.executable, str(FOLDER / "voice_audit.py"), "-", "--samples", str(p),
                                "--cache", str(Path(d) / "c.json")], input="I think the bus was late. " * 40,
                               capture_output=True, text=True, encoding="utf-8", cwd=str(FOLDER), timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertRegex(r.stdout.strip(), r"^DELTA \(soft\): [\d.]+ vs your chunks median")

    def test_the_script_reads_its_settings_the_way_regrow_audit_runs_it(self):
        # engine/regrow.py audit passes only "-", from this folder; the samples and the cache come from settings
        import os
        with tempfile.TemporaryDirectory() as d:
            p = samples_file(d, n=13, words=320)
            cache = Path(d) / "voice cache.json"
            env = dict(os.environ, SAMPLES_FILE=str(p), VOICE_BASELINE=str(cache))
            r = subprocess.run([sys.executable, str(FOLDER / "voice_audit.py"), "-"], input="I think so. " * 60,
                               capture_output=True, text=True, encoding="utf-8", cwd=str(FOLDER), env=env,
                               timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(r.stdout.startswith("DELTA (soft): "), r.stdout + r.stderr)
            self.assertTrue(cache.exists())
            env.pop("SAMPLES_FILE")
            r = subprocess.run([sys.executable, str(FOLDER / "voice_audit.py"), "-"], input="I think so.",
                               capture_output=True, text=True, encoding="utf-8", cwd=str(FOLDER), env=env,
                               timeout=120)
            self.assertEqual((r.returncode, r.stdout), (0, ""))


class Transcripts(unittest.TestCase):
    MINE = "I wrote this whole sentence about the old harbour wall myself."
    THEIRS = "The model wrote this sentence about the lighthouse keeper's long night."

    def test_markdown_exports(self):
        md = (f"# A chat\n\n_**User (morning)**_\n\n{self.MINE}\n\n---\n\n_**Agent (model x)**_\n\n{self.THEIRS}\n\n"
              f"```\n{self.MINE}\n```\n\n**User:** And one more line of mine for the record here.\n"
              f"\n## Assistant\n\nAnother reply from the model, with nothing else in it.\n")
        turns = find_lines.markdown_turns(md)
        self.assertEqual([r for r, _ in turns], ["user", "model", "user", "model"])
        self.assertIn(self.MINE, turns[0][1])
        self.assertNotIn(self.MINE, turns[1][1])             # a code fence in a model turn is skipped
        self.assertIn("one more line of mine", turns[2][1])
        # a plain "User:" line inside a reply, and a speaker line inside a fence, stay part of the reply
        reply = "## Assistant\n\nFirst do this.\nUser: clicks save and waits for the page.\n```\n## User\n```\nDone.\n"
        self.assertEqual([r for r, _ in find_lines.markdown_turns(reply)], ["model"])

    def test_jsonl_shapes(self):
        recs = [
            {"type": "user", "message": {"role": "user", "content": self.MINE}},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": self.THEIRS},
                {"type": "tool_use", "input": {"new_string": "A line the model typed into a file for me today."}}]}},
            {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "read back"}]}},
            {"role": "assistant", "content": "A reply in the plain chat format, one record per turn."},
            {"conversations": [{"from": "human", "value": "A question asked in a whole conversation record."},
                               {"from": "gpt", "value": "An answer given in a whole conversation record."}]},
            {"type": "summary", "summary": "skipped"},
        ]
        got = [t for r in recs for t in find_lines.turn_texts(r)]
        self.assertIn(("user", self.MINE), got)
        self.assertIn(("model", self.THEIRS), got)
        self.assertIn(("model", "A line the model typed into a file for me today."), got)
        self.assertNotIn(("user", "read back"), got)
        self.assertIn(("user", "A question asked in a whole conversation record."), got)
        self.assertIn(("model", "An answer given in a whole conversation record."), got)
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "logs" / "deep"
            sub.mkdir(parents=True)
            (sub / "a.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\nnot json\n", encoding="utf-8")
            hist = Path(d) / "md"
            hist.mkdir()
            rows, n = find_lines.classify(f"{self.MINE} {self.THEIRS}", dirs=[Path(d) / "logs"], history=hist)
            self.assertEqual(n, 1)
            self.assertEqual([v for _, v, _, _ in rows], ["yours", "claude"])

    def test_nothing_configured_and_everything_machine(self):
        with mock.patch.object(find_lines, "path_setting", return_value=None), \
                mock.patch.object(find_lines, "path_list_setting", return_value=[]):
            rows, n = find_lines.classify(f"{self.MINE} {self.THEIRS}")
        self.assertEqual(([v for _, v, _, _ in rows], n), (["yours?", "yours?"], 0))
        rows, n = find_lines.classify(f"{self.MINE} {self.THEIRS}", source="machine")
        self.assertEqual(([v for _, v, _, _ in rows], n), (["claude", "claude"], 0))
        table = find_lines.table(rows, n, "machine")
        self.assertIn("| claude | 0% | 100% | " + self.MINE + " |", table)
        with self.assertRaises(ValueError):
            find_lines.classify("x", source="guess")


class OtherFiles(unittest.TestCase):
    def test_guide_sections_takes_a_section_style(self):
        doc = "# Title\n\n## 1. One\n\nA.\n\n## 2. Two\n\nB.\n\n## Notes\n\nC.\n"
        self.assertEqual(texts.guide_sections([1, 2], text=doc),
                         {1: "## 1. One\n\nA.\n\n", 2: "## 2. Two\n\nB.\n\n"})

        class Style:                                   # the shape of engine/job.py SectionStyle
            def texts(self, md, ids):
                return {i: md for i in ids}
        self.assertEqual(texts.guide_sections([3], text="x", style=Style()), {3: "x"})
        with self.assertRaises(ValueError):
            texts.guide_sections([1])

    def test_voice_material_from_any_files(self):
        with tempfile.TemporaryDirectory() as d:
            guide = Path(d) / "voice.md"
            guide.write_text("# How I write\n\nShort lines. Swear a bit.\n", encoding="utf-8")
            samples = samples_file(d, n=3, words=20)
            voice, block = idioms.voice_material(guide, samples)
            self.assertIn("Short lines", voice)
            self.assertTrue(block.startswith("Sample 1:\n") and "Sample 2:\n" in block)
            voice, block = idioms.voice_material(guide, samples, picks=(9, 3))
            self.assertTrue(block.startswith("Sample 3:\n"))
            self.assertEqual(idioms.voice_section("## A\nx\n## B\ny\n", "## B"), "## B\ny")

    def test_idioms_read_plain_text_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "notes.txt").write_text("Honestly, the bus was late again, probably.", encoding="utf-8")
            (Path(d) / "skip.json").write_text("{}", encoding="utf-8")
            got, counts = idioms.reddit_texts(d)
            self.assertEqual((got, counts), (["Honestly, the bus was late again, probably."], {"files": 1}))
            self.assertIn("notes.txt", idioms.corpus_stamp(d))


class RulesScopedToAJob(unittest.TestCase):
    """A job's rules file reaches its own reads only: a job loaded after another reads its own (or the base), a
    scope puts a job's rules in use for one block in one thread, and nothing is left behind."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(checks.use_rules)
        root = Path(self.tmp.name)
        self.a, self.b = root / "a", root / "b"
        for d in (self.a, self.b):
            d.mkdir()
        (self.a / "job.toml").write_text('[job]\nname = "a"\nrules = "rules.toml"\n', encoding="utf-8")
        (self.a / "rules.toml").write_text('[filler]\nwords = ["frankly"]\n', encoding="utf-8")
        (self.b / "job.toml").write_text('[job]\nname = "b"\n', encoding="utf-8")
        checks.use_rules()
        self.base = checks.FILLER_TAGS

    def test_a_job_loaded_after_another_never_reads_the_first_one_s_file(self):
        from engine.job import Job
        a = Job(root=self.a)
        self.assertEqual(checks.FILLER_TAGS, ("frankly",))                     # the command line's way: a's
        b = Job(root=self.b)
        self.assertEqual(checks.FILLER_TAGS, self.base)                         # flag: b has none, so the base
        with a.rules_in_use():
            self.assertEqual(checks.FILLER_TAGS, ("frankly",))
            self.assertEqual(checks.filler_count("It works, frankly."), 1)
            with b.rules_in_use():
                self.assertEqual(checks.FILLER_TAGS, self.base)
            self.assertEqual(checks.FILLER_TAGS, ("frankly",))
        self.assertEqual(checks.FILLER_TAGS, self.base)                         # pass: nothing left behind

    def test_a_job_loaded_without_binding_leaves_the_rules_alone(self):
        from engine.job import Job
        a = Job(root=self.a, bind_rules=False)
        self.assertEqual(checks.FILLER_TAGS, self.base)
        self.assertEqual(a.rules.FILLER_TAGS, ("frankly",))
        with a.rules_in_use():
            self.assertEqual(checks.house_rules()["filler"]["words"], ["frankly"])
        self.assertEqual(checks.FILLER_TAGS, self.base)

    def test_a_scope_in_one_thread_never_reaches_another(self):
        import threading
        from engine.job import Job
        a = Job(root=self.a, bind_rules=False)
        inside, other = threading.Event(), {}
        done = threading.Event()

        def hold():
            with a.rules_in_use():
                inside.set()
                done.wait(5)
        t = threading.Thread(target=hold)
        t.start()
        inside.wait(5)
        other["seen"] = checks.FILLER_TAGS                                      # read while the scope is open
        done.set()
        t.join(5)
        self.assertEqual(other["seen"], self.base)

    def test_a_scope_ends_even_when_its_block_fails(self):
        from engine.job import Job
        a = Job(root=self.a, bind_rules=False)
        with self.assertRaises(RuntimeError):
            with a.rules_in_use():
                raise RuntimeError("boom")
        self.assertEqual(checks.FILLER_TAGS, self.base)

    def test_use_rules_clears_a_job_s_rules(self):
        from engine.job import Job
        Job(root=self.a)
        checks.use_rules()
        self.assertEqual(checks.FILLER_TAGS, self.base)


if __name__ == "__main__":
    unittest.main()
