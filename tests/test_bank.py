"""Tests for the bank's torn last line (engine/job.py, Bank): a load leaves a torn last line out and says so, a bad line
anywhere else still raises, a load never writes, the first add cuts the torn text off into the `.torn` file and
starts the new entry on its own line, a clean file loads and appends as it always did, and an add onto a whole last
line with no newline after it puts the newline in first. Run from the repo root:

    python -m unittest discover tests
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from engine.job import Bank

NL = json.dumps({"a": 1}) + "\n"     # what an add writes, before the text mode's own newline


class TornLastLine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "regrow bank.jsonl"
        self.side = Path(self.tmp.name) / "regrow bank.jsonl.torn"

    def write(self, data):
        self.path.write_bytes(data)

    def lines(self):
        return [json.loads(x) for x in self.path.read_bytes().decode("utf-8").splitlines() if x.strip()]

    def test_a_torn_last_line_loads_without_it_and_says_so(self):
        self.write(b'{"kind": "draw", "n": 1}\n{"kind": "draw", "n": 2}\n{"kind": "dr')
        b = Bank(self.path)
        self.assertEqual([e["n"] for e in b.entries], [1, 2])                                   # flag: left out
        self.assertEqual(b.torn, '{"kind": "dr')
        self.write(b'{"kind": "draw", "n": 1}\r\n{"kind": "dr')
        self.assertEqual(Bank(self.path).torn, '{"kind": "dr')                                  # CRLF before it too
        self.write(b'{"text": "caf\xc3')                                                        # a cut through a character
        b = Bank(self.path)
        self.assertEqual((b.entries, b.torn), ([], '{"text": "caf\ufffd'))

    def test_a_bad_line_anywhere_but_the_end_still_raises(self):
        self.write(b'{"n": 1}\n{"n": \n{"n": 3}\n')
        with self.assertRaises(ValueError):                                                     # flag: not a kill's work
            Bank(self.path)
        self.write(b'{"n": 1}\n{"n": \n\n  \n')                                                 # blank lines after it
        self.assertEqual(Bank(self.path).torn, '{"n": ')                                        # pass: still the last line

    def test_a_load_alone_changes_no_byte(self):
        data = b'{"n": 1}\r\n{"n": 2}\n{"n": 3'
        self.write(data)
        b = Bank(self.path)
        b.of("s1w0")
        self.assertEqual(self.path.read_bytes(), data)                                          # pass: read only
        self.assertFalse(self.side.exists())

    def test_the_first_add_cuts_the_torn_text_keeps_it_and_appends_a_whole_line(self):
        self.write(b'{"n": 1}\n{"n": 2}\n{"n": 3, "te')
        b = Bank(self.path)
        b.add({"n": 4})
        b.add({"n": 5})
        self.assertEqual([e["n"] for e in self.lines()], [1, 2, 4, 5])                          # flag: cut, then whole lines
        self.assertTrue(self.path.read_bytes().startswith(b'{"n": 1}\n{"n": 2}\n{"n": 4}'))     # the lines before kept
        side = self.side.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(side), 2)
        self.assertRegex(side[0], r"^# \d{4}-\d\d-\d\d \d\d:\d\d:\d\d the torn last line cut from regrow bank\.jsonl:$")
        self.assertEqual(side[1], '{"n": 3, "te')
        self.assertEqual([e["n"] for e in Bank(self.path).entries], [1, 2, 4, 5])               # pass: loads clean again
        self.write(b'{"n": 3, "te')                                                             # the only line torn
        b = Bank(self.path)
        b.add({"n": 6})
        self.assertEqual(self.lines(), [{"n": 6}])
        self.assertEqual(len(self.side.read_text(encoding="utf-8").splitlines()), 4)            # appended, the first kept

    def test_a_torn_tail_that_changed_since_the_load_is_left_alone(self):
        self.write(b'{"n": 1}\n{"n": 2')
        b = Bank(self.path)
        with self.path.open("ab") as f:
            f.write(b', "more": 1}\n')                                                          # someone else finished it
        b.add({"n": 3})
        self.assertEqual([e["n"] for e in self.lines()], [1, 2, 3])                             # nothing cut
        self.assertFalse(self.side.exists())

    def test_a_clean_file_loads_and_appends_as_before(self):
        data = b'{"n": 1}\n{"n": 2}\n'
        self.write(data)
        b = Bank(self.path)
        self.assertEqual(([e["n"] for e in b.entries], b.torn), ([1, 2], None))
        b.add({"a": 1})
        want = NL.replace("\n", os.linesep).encode("utf-8")                                     # the text mode's newline
        self.assertEqual(self.path.read_bytes(), data + want)                                   # pass: one appended line
        self.assertFalse(self.side.exists())
        missing = Path(self.tmp.name) / "new bank.jsonl"
        b = Bank(missing)
        self.assertEqual((b.entries, b.torn), ([], None))
        b.add({"a": 1})
        self.assertEqual(missing.read_bytes(), want)

    def test_an_add_onto_a_whole_last_line_with_no_newline_starts_its_own_line(self):
        self.write(b'{"n": 1}\n{"n": 2}')
        b = Bank(self.path)
        self.assertIsNone(b.torn)
        b.add({"n": 3})
        self.assertEqual([e["n"] for e in self.lines()], [1, 2, 3])                             # flag: not glued on
        self.assertEqual([e["n"] for e in Bank(self.path).entries], [1, 2, 3])
        self.assertFalse(self.side.exists())


if __name__ == "__main__":
    unittest.main()
