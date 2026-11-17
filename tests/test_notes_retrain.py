"""Tests for the notes adapter's retrain with registers (plan 6.9, Phase 3; 2026-11-11): the essay passages (school
notes, earlier drafts, titles and the exam question out, near-duplicates dropped, short paragraphs joined, formal
register kept), the all-yours guide paragraphs with the loop's own notes, the register bullet (and that a prompt
without it is the live loop's byte for byte), the pairs2 rows, the notes batching and estimate, the new yaml
matching the live recipe. Synthetic fixtures only: corpus/ is gitignored,
and there's no model, no Modal, no Pangram and no Claude anywhere. Run from the repo root:

    python -m unittest discover tests
"""
import json
import tempfile
import types
import unittest
from pathlib import Path

from unittest import mock

import yaml

from engine.job import NOTES_VERSION, sha
from engine.run import Loop
from training import build_notes_pairs as bnp

HERE = Path(__file__).resolve().parents[1]
# invented for the tests (no essay text in the repo): long enough to stand alone as a passage
LONG = ("The brochure leans on the reader's fondness for the family dog from its very first line, a slogan that "
        "promises a quiet green field for a loyal friend, and it keeps coming back to that promise with a restraint "
        "that feels unusual for a leaflet whose real job is to sell a plot of land to grieving owners.")


def essay_dir(files):
    d = Path(tempfile.mkdtemp())
    for name, text in files.items():
        (d / name).write_text(text, encoding="utf-8")
    return d


class EssayNormalize(unittest.TestCase):
    def test_quotes_dashes_and_spelling(self):
        src = "The \u2018model pupil\u2019 \u2013 as the writer puts it \u2013 is \u201ccognisant\u201d of the colour of favour."
        t = bnp.normalize_essay(src)
        self.assertEqual(t, "The 'model pupil' - as the writer puts it - is \"cognizant\" of the color of favor.")

    def test_formal_register_kept_and_no_capitalizing(self):
        src = "It is, e.g. in the opening, rather verbose; whilst the author does not say so."
        self.assertEqual(bnp.normalize_essay(src), "It is, e.g. in the opening, rather verbose; while the author does not say so.")


# a corpus's own lists, as one corpus would fill them in (build_notes_pairs ships them empty)
CORPUS_LISTS = {"ESSAY_SKIP_FILES": {"piece_notes.txt", "piece_a_1.txt"}, "ESSAY_FIRST": ["piece_b_fixed.txt"],
                "ESSAY_SKIP_PARAS": ("Discuss the claim that every story needs a villain.",)}


class EssayPassages(unittest.TestCase):
    def setUp(self):
        patches = [mock.patch.object(bnp, k, v) for k, v in CORPUS_LISTS.items()]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_skips_notes_files_drafts_titles_and_the_exam_question(self):
        d = essay_dir({
            "piece_notes.txt": LONG,
            "piece_a_1.txt": "An earlier draft. " + LONG,
            "piece_a_2.txt": "Discuss the claim that every story needs a villain. How far do you agree?"
                             "\n\nRationale\n\n" + LONG,
        })
        got = bnp.essay_paragraphs(d)
        self.assertEqual(got, [("piece_a_2.txt", LONG)])

    def test_near_duplicates_keep_the_preferred_draft(self):
        fixed = LONG.replace("grieving owners", "grieving-owners")
        d = essay_dir({"piece_b_fixed.txt": fixed, "aaa_other.txt": LONG})
        got = bnp.essay_paragraphs(d)
        self.assertEqual(got, [("piece_b_fixed.txt", fixed)])

    def test_with_the_lists_empty_nothing_is_skipped(self):
        with mock.patch.object(bnp, "ESSAY_SKIP_FILES", set()):
            d = essay_dir({"piece_notes.txt": LONG})
            self.assertEqual(bnp.essay_paragraphs(d), [("piece_notes.txt", LONG)])

    def test_short_paragraphs_join_and_lone_short_ones_drop(self):
        short = "This is a short paragraph of only a few words."
        other = ("In the next section the writer turns to the shopkeeper, whose patient sales talk and tired smile "
                 "are drawn in such careful detail that the scene reads less like a complaint about a pair of shoes "
                 "and more like a small portrait of a market town slowly going quiet.")
        d = essay_dir({"a.txt": short + "\n\n" + LONG + "\n\n" + other,
                       "b.txt": "A one-line piece that never reaches the floor."})
        rows = bnp.essay_passages(d)
        self.assertEqual([r["where"] for r in rows], ["a.txt", "a.txt"])
        self.assertTrue(rows[0]["text"].startswith(short + "\n\n"))
        self.assertTrue(all(bnp.MIN_WORDS <= r["words"] <= bnp.MAX_WORDS for r in rows))
        self.assertTrue(all(r["source"] == "essays" and r["id"].startswith("e") for r in rows))

    def test_a_short_last_paragraph_joins_the_one_before(self):
        d = essay_dir({"a.txt": LONG + "\n\nAnd that is why it works so well."})
        rows = bnp.essay_passages(d)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["text"].endswith("so well."))


class GuideRows(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self.yours = "I wrote this whole paragraph myself. It has two sentences."
        self.mixed = "I wrote this sentence. Claude wrote this one."
        self.unnoted = "This one is mine too. But the loop never noted it."
        self.maybe = "This one might be mine. Nobody is sure."
        (d / "orig.md").write_text(f"## 1. Intro\n\n{self.yours}\n\n{self.mixed}\n\n## 2. More\n\n{self.unnoted}\n\n"
                                   f"{self.maybe}\n", encoding="utf-8")
        cache = d / "notes cache.json"
        cache.write_text(json.dumps({sha(NOTES_VERSION + self.yours): ["the writer wrote it"],
                                     sha(NOTES_VERSION + self.maybe): ["maybe"]}), encoding="utf-8")
        self.job = types.SimpleNamespace(original=d / "orig.md", notes_cache=cache, example_lines=lambda: set())
        self.verdicts = {"I wrote this whole paragraph myself.": "yours", "It has two sentences.": "yours",
                         "I wrote this sentence.": "yours", "Claude wrote this one.": "claude",
                         "This one is mine too.": "yours", "But the loop never noted it.": "yours",
                         "This one might be mine.": "yours?", "Nobody is sure.": "yours"}

    def test_only_all_yours_paragraphs_with_loop_notes(self):
        rows, missing = bnp.guide_rows(self.job, self.verdicts)
        self.assertEqual([r["text"] for r in rows], [self.yours])
        self.assertEqual(rows[0]["notes"], ["the writer wrote it"])
        self.assertEqual((rows[0]["source"], rows[0]["where"]), ("guide", "section 1"))
        self.assertEqual([r["text"] for r in missing], [self.unnoted])


class RegisterBullet(unittest.TestCase):
    NOTES = ["the app has no idea why the steps are there", 'keep: "probably"']

    def test_bullet_goes_first(self):
        self.assertEqual(bnp.render_notes(self.NOTES, "essay"),
                         "<notes>\n- register: essay\n- the app has no idea why the steps are there\n"
                         "- keep: \"probably\"\n</notes>")

    def test_untagged_is_the_live_format(self):
        live = "<notes>\n" + "\n".join(f"- {n}" for n in self.NOTES) + "\n</notes>"
        self.assertEqual(bnp.render_notes(self.NOTES), live)
        self.assertEqual(bnp.render_notes(self.NOTES) + "\n<target_text>\n", Loop.adapter_prompt(self.NOTES, ""))


class PairRows(unittest.TestCase):
    def test_registers_paraphrase_rows_and_skips(self):
        corpus = [{"id": "r1", "source": "reddit", "where": "x", "words": 50, "text": "reddit one"},
                  {"id": "s1", "source": "samples", "where": "y", "words": 60, "text": "forum one"},
                  {"id": "r2", "source": "reddit", "where": "z", "words": 70, "text": "no notes here"}]
        essays = [{"id": "e1", "source": "essays", "where": "a.txt", "words": 80, "text": "essay one"},
                  {"id": "e2", "source": "essays", "where": "a.txt", "words": 90, "text": "essay two"}]
        guide = [{"id": "g1", "source": "guide", "where": "section 1", "words": 30, "text": "guide one",
                  "notes": ["loop note"]}]
        cache = {sha(bnp.NOTES_KEY + "reddit one"): ["r note"], sha(bnp.NOTES_KEY + "forum one"): ["f note"],
                 sha(bnp.ESSAY_KEY + "essay one"): ["e note"],
                 sha(bnp.NOTES_KEY + "essay two"): ["the corpus key never counts for an essay"]}
        rows, skipped = bnp.pair_rows(corpus, essays, guide, cache, {"r1": "a paraphrase"})
        self.assertEqual(skipped, 2)
        self.assertEqual([(r["id"], r["kind"], r["register"]) for r in rows],
                         [("r1", "notes", "reddit"), ("r1-p", "paraphrase", "reddit"), ("s1", "notes", "forum"),
                          ("e1", "notes", "essay"), ("g1", "notes", "guide")])
        self.assertTrue(rows[0]["prompt_source"].startswith("<notes>\n- register: reddit\n- r note"))
        self.assertEqual(rows[1]["prompt_source"], "<source_text>\na paraphrase\n</source_text>")
        self.assertIn("- register: guide\n- loop note", rows[4]["prompt_source"])
        counts = bnp.count_rows(rows)
        self.assertEqual(counts[("notes", "essay")], [1, 80])


class NotesBatching(unittest.TestCase):
    def test_chunks_never_mix_prompts(self):
        a = [({"id": i, "words": 10, "text": str(i)}, "P1", "k1|") for i in range(12)]
        b = [({"id": i, "words": 10, "text": str(i)}, "P2", "k2|") for i in range(3)]
        chunks = bnp.notes_chunks(a + b)
        self.assertEqual([len(c) for c in chunks], [10, 2, 3])
        self.assertTrue(all(len({x[2] for x in c}) == 1 for c in chunks))
        self.assertAlmostEqual(bnp.estimate_usd(a + b), 150 * bnp.NOTES_USD_PER_WORD)

    def test_todo_uses_each_jobs_key(self):
        rows = [{"id": 1, "words": 5, "text": "t"}]
        cache = {sha("k1|t"): ["n"]}
        todo = bnp.notes_todo([(rows, "P1", "k1|"), (rows, "P2", "k2|")], cache)
        self.assertEqual([(t[1], t[2]) for t in todo], [("P2", "k2|")])

    def test_parse_notes(self):
        got = bnp.parse_notes("### 1\n- one\n- two\n\n### 2\n- three\nnot a note\n")
        self.assertEqual(got, {1: ["one", "two"], 2: ["three"]})

    def test_essay_prompt_has_its_own_preamble(self):
        self.assertIn("analytical essays", bnp.ESSAY_NOTES_PROMPT)
        self.assertNotIn("Reddit", bnp.ESSAY_NOTES_PROMPT.split("\n")[0])
        self.assertNotEqual(bnp.ESSAY_KEY, bnp.NOTES_KEY)


class Recipe(unittest.TestCase):
    def test_new_yaml_is_the_live_recipe_with_new_names(self):
        old = yaml.safe_load((HERE / "training" / "hip_notes.yaml").read_text(encoding="utf-8"))
        new = yaml.safe_load((HERE / "training" / "hip_notes2.yaml").read_text(encoding="utf-8"))
        changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
        self.assertEqual(changed, {"run_name", "pairs_file", "sources"})
        for k in ("run_name", "pairs_file"):
            self.assertNotEqual(old[k], new[k])
        self.assertTrue(set(old["sources"]) < set(new["sources"]))
        self.assertEqual(set(new["sources"]), set(bnp.REGISTER))

    def test_no_yaml_names_a_local_path(self):
        for name in ("hip_mistral.yaml", "hip_notes.yaml", "hip_notes2.yaml"):
            cfg = yaml.safe_load((HERE / "training" / name).read_text(encoding="utf-8"))
            self.assertFalse({"pairs_local", "local_out"} & set(cfg), name)              # paths are arguments now

    def test_the_data_paths_come_from_the_data_folder(self):
        d = Path(tempfile.mkdtemp())
        bnp.set_data(d, job="j")
        self.assertEqual((bnp.PASSAGES, bnp.PAIRS_OUT2, bnp.ESSAYS, bnp.JOB),
                         (d / "passages.jsonl", d / "pairs_notes2.jsonl", d / "essays", "j"))
        bnp.set_data(d, essays=d / "mine")
        self.assertEqual(bnp.ESSAYS, d / "mine")


class Registry(unittest.TestCase):
    """The model registry in models.toml: names, files under MODELS_DIR, the adapter entry an example to fill in."""

    def server(self, name, registry=None, **env):
        from engine.local_models import LlamaServer
        env.setdefault("RUNS_DIR", tempfile.mkdtemp())
        get = lambda k, d="": env.get(k) or d        # noqa: E731
        return LlamaServer(name, settings=types.SimpleNamespace(get=get, runs_dir=Path(env["RUNS_DIR"])),
                           models=registry)

    def test_the_shipped_registry_has_the_loop_s_models_and_an_adapter_example(self):
        from engine.local_models import load_models
        reg = load_models(path=HERE / "models.toml")
        for name in ("hip-notes-mistral24b", "mistral-small-24b-base", "olmo3-32b-iq3"):        # config.toml's
            self.assertIn(name, reg)
        live = reg["hip-notes-mistral24b"]
        self.assertEqual(live["file"], reg["mistral-small-24b-base-2501"]["file"])
        self.assertTrue(live["lora"].endswith(".gguf"))
        self.assertFalse(Path(live["lora"]).is_absolute())                                   # relative to MODELS_DIR
        for spec in reg.values():
            self.assertNotIn("\\", spec["file"])                                            # forward slashes only

    def test_paths_resolve_under_models_dir_and_the_adapter_loads(self):
        d = Path(tempfile.mkdtemp())
        srv = self.server("a", {"a": {"file": "base/m.gguf", "lora": "ad/l.gguf", "lora_scale": 0.7, "ctx": 2048,
                                      "parallel": 2, "extra": ["-x"]}}, MODELS_DIR=str(d))
        self.assertEqual(Path(srv.model_path), d / "base" / "m.gguf")
        self.assertEqual(srv.extra, ["-x", "--lora", str(d / "ad" / "l.gguf")])
        self.assertEqual((srv.ctx, srv.parallel, srv.has_lora), (2048, 2, True))
        from engine.local_models import ArmServer
        self.assertEqual(ArmServer(srv, "notes").scale, 0.7)                                 # the registry's scale
        self.assertEqual(ArmServer(srv, "fewshot").scale, 0.0)

    def test_a_missing_model_or_adapter_or_server_is_named_before_any_start(self):
        d = Path(tempfile.mkdtemp())
        srv = self.server("a", {"a": {"file": "m.gguf", "lora": "l.gguf"}}, MODELS_DIR=str(d),
                          LLAMA_CPP_DIR=str(d / "nowhere"))
        srv.exe = None
        gone = " ".join(srv.missing())
        self.assertIn("llama-server", gone)
        self.assertIn("model file isn't there", gone)
        self.assertIn("adapter isn't there", gone)
        (d / "m.gguf").write_text("")
        (d / "l.gguf").write_text("")
        srv.exe = "llama-server"
        self.assertEqual(srv.missing(), [])                                                  # pass: all there
        with self.assertRaises(KeyError) as cm:
            self.server("nope", {"a": {"file": "m.gguf"}})
        self.assertIn("it has: a", str(cm.exception))

    def test_llama_server_is_found_in_its_folder_then_on_the_path(self):
        from unittest import mock
        from engine import local_models
        d = Path(tempfile.mkdtemp())
        get = {"LLAMA_CPP_DIR": str(d)}
        with mock.patch.object(local_models.shutil, "which", return_value="/usr/bin/llama-server"):
            self.assertEqual(local_models.find_server(get), "/usr/bin/llama-server")         # nothing in the folder
            (d / "build" / "bin").mkdir(parents=True)
            exe = d / "build" / "bin" / local_models.SERVER_EXE
            exe.write_text("")
            self.assertEqual(local_models.find_server(get), str(exe))                        # the folder first


class SampleStops(unittest.TestCase):
    def test_cut_stops(self):
        from training.modal_train import SAMPLE_STOPS, cut_stops
        from engine.local_models import ADAPTER_STOPS
        self.assertEqual(SAMPLE_STOPS, ADAPTER_STOPS)
        self.assertEqual(cut_stops("text here\n</target_text>\n<notes>"), "text here\n")
        self.assertEqual(cut_stops("no stop"), "no stop")


if __name__ == "__main__":
    unittest.main()
