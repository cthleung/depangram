"""Tests for the docs figures (docs/figures/make_figures.py): the script runs into a temp folder, every figure the
README lists has a light and a dark SVG, a second run is byte-identical, no generated file or source carries a
dash or a word the house text rules keep out, the data has no `source` key anywhere, and the committed README
matches what the script writes. Skipped when matplotlib isn't installed (it isn't an engine dependency):

    pip install matplotlib==3.11.2
    python -m unittest discover tests -p test_figures.py
"""
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
HAS_MPL = importlib.util.find_spec("matplotlib") is not None

DASHES = tuple(chr(c) for c in (0x2014, 0x2013, 0x2012, 0x2015, 0x2212))   # by code point, so this file carries none
WORDS = ["struct" + "ural", "scaff" + "olding", "frame" + "work", "archi" + "tecture", "found" + "ation",
         "load-" + "bearing", "ha" + "lf", "we" + "ight", "grav" + "ity"]
WORD_RE = re.compile(r"(?i)(" + "|".join(re.escape(w) for w in WORDS) + r")")


def load_script():
    spec = importlib.util.spec_from_file_location("make_figures", FIG / "make_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def listed(readme):
    """Figure names the README embeds, from its srcset and src attributes."""
    return sorted(set(re.findall(r'(?:srcset|src)="(?:figures/)?([a-z0-9-]+)-(?:light|dark)\.svg"', readme)))


def keys_anywhere(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from keys_anywhere(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from keys_anywhere(v)


class Data(unittest.TestCase):
    def test_no_source_key_anywhere(self):
        data = json.loads((FIG / "data.json").read_text(encoding="utf-8"))
        self.assertNotIn("source", set(keys_anywhere(data)))

    def test_every_dataset_has_a_note(self):
        data = json.loads((FIG / "data.json").read_text(encoding="utf-8"))
        for d in data["datasets"]:
            self.assertTrue(d.get("note"), d["id"])


@unittest.skipUnless(HAS_MPL, "matplotlib isn't installed")
class Figures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_script()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.a = Path(cls.tmp.name) / "a"
        cls.b = Path(cls.tmp.name) / "b"
        cls.names = cls.mod.build(cls.a)
        cls.mod.build(cls.b)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_listed_figure_has_both_svgs(self):
        for folder in (self.a, FIG):
            names = listed((folder / "README.md").read_text(encoding="utf-8"))
            self.assertEqual(names, sorted(self.names))
            for n in names:
                for mode in ("light", "dark"):
                    self.assertTrue((folder / f"{n}-{mode}.svg").is_file(), f"{folder.name}: {n}-{mode}.svg")

    def test_second_run_is_byte_identical(self):
        files = sorted(p.name for p in self.a.iterdir())
        self.assertEqual(files, sorted(p.name for p in self.b.iterdir()))
        for f in files:
            self.assertEqual((self.a / f).read_bytes(), (self.b / f).read_bytes(), f)

    def test_svgs_keep_text_as_text_in_the_system_sans(self):
        svg = (self.a / "noise-floor-light.svg").read_text(encoding="utf-8")
        self.assertIn("<text", svg)
        self.assertIn("system-ui", svg)
        self.assertNotIn("<dc:date>", svg)

    def test_no_dash_or_ruled_word_in_any_output_or_source(self):
        paths = list(self.a.iterdir()) + [FIG / "data.json", FIG / "README.md", FIG / "make_figures.py",
                                          Path(__file__)]
        for p in paths:
            text = p.read_text(encoding="utf-8")
            for d in DASHES:
                self.assertNotIn(d, text, f"{p.name}: U+{ord(d):04X}")
            m = WORD_RE.search(text)
            self.assertIsNone(m, f"{p.name}: {m.group(0) if m else ''}")

    def test_committed_readme_matches_the_script(self):
        self.assertEqual((FIG / "README.md").read_text(encoding="utf-8"),
                         (self.a / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
