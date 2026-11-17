"""One job: its files, its windows, its notes, its bank and its working copy (plan 5.1, 11.1).

A job is any markdown document. It lives in a folder in jobs/ with a job.toml that names the document, the frozen
original, how the document marks its sections, which sections are in scope, the voice samples and where the
verdicts on who wrote which line come from. jobs/README.md lists every field.

    SectionStyle   how a document marks its sections: a heading level, or a regex (the default is the numbered
                   "## N." headings older jobs use). Every module that cuts a document into sections asks it
    job_name()     the job a command works on: the --job it was given, else [job] default in config.toml
    setup()        5.1 step 0: the working copy, with every AI example line swapped
                   for an {{example N.N}} placeholder, and locks.md listing the swaps
                   and every window's locks (5.1 step 3)
    windows()      5.1 step 1: windows of just under 400 words, cut at paragraph
                   breaks, from the frozen original (experiment 6's cutter)
    verdicts()     question 2's list (`your lines.md`), frozen, so a new chat session
                   can't move a line from yours to Claude's mid-job
    notes_for()    5.1 step 2: Sonnet's notes per paragraph, now with a keep: line for
                   hedges and deliberate repeats (v5), cached by paragraph text
    Bank           every round's text and runs (jobs/<job>/bank.jsonl); resume reads it
    assemble()     the working copy from every window's best text
"""
import hashlib
import json
import re
import shutil
import tomllib
from pathlib import Path

from .settings import SANDBOX
from .texts import AI_EXAMPLE_LINES, COMMENT_RE, prose_only, strip_markdown
from .util import describe

JOBS = SANDBOX / "jobs"
WINDOW_WORDS = 395                 # experiment 3's cutter, which every experiment since has used
NOTES_VERSION = "v5|"              # v5 (2026-11-08): the keep: line for hedges and repeats (5.3's fix list)


class JobError(RuntimeError):
    """A job that can't be found or whose job.toml can't be read: the message says what to fix."""


# ---------------------------------------------------------------------- sections
NUMBERED_PATTERN = r"^## (\d+)\.\s"       # the default: "## 3. A title", the section's id is its number
HEADING_LINE = re.compile(r"^(#{1,6})\s", re.M)


class SectionStyle:
    """How a document marks its sections, and the one place that cuts a document into them.

    Give it a heading `level` (3 for "### Part one") or a regex `pattern` (multiline, matched at the start of the
    heading line). With neither, it reads the numbered "## N." headings (NUMBERED_PATTERN), which every job made
    before job.toml had these fields uses.

    A section's id is an int. With a pattern whose first group captures digits, it's that number (the numbered
    style's "## 3." is section 3, so an old job's bank, ledger and markers keep their ids). Otherwise it's the
    section's place among the document's section headings, counting from 1. Window ids (`s2w0`) and marker labels
    (`s2w0 p3`) use that id.

    A section runs from its heading to the start of the next section, or to the next heading at its own level or
    above when that comes first. The numbered default keeps the two rules older code had: texts() runs each section
    to the next numbered one and ends the last at the first "## " heading after it that isn't numbered, and spans()
    ends every section at the next "## " heading of any kind. The one change: older code took the first unnumbered
    "## " anywhere for the last section's end, so one sitting before the last section emptied it."""

    def __init__(self, pattern=None, level=None):
        if pattern and level:
            raise JobError("job.toml: give section_pattern or section_level, not both")
        if level is not None:
            level = int(level)
            if not 1 <= level <= 6:
                raise JobError(f"job.toml: section_level is a heading level from 1 to 6, not {level}")
            pattern = r"^#{%d}[ \t]+\S" % level
        self.level = level
        self.source = pattern or NUMBERED_PATTERN
        self.numbered = self.source == NUMBERED_PATTERN
        try:
            self.regex = re.compile(self.source, re.M)
        except re.error as e:
            raise JobError(f"job.toml: section_pattern {self.source!r} isn't a valid regex ({e})") from None

    def __repr__(self):
        return f"SectionStyle(level={self.level})" if self.level else f"SectionStyle(pattern={self.source!r})"

    def heads(self, md):
        """[(id, start, heading level or 0)] for every section heading in `md`, in document order."""
        out = []
        for k, m in enumerate(self.regex.finditer(md)):
            g = m.group(1) if self.regex.groups else None
            sid = int(g) if g and g.isdigit() else k + 1
            line_start = md.rfind("\n", 0, m.start()) + 1
            lv = HEADING_LINE.match(md, line_start)
            out.append((sid, line_start, len(lv.group(1)) if lv else 0))
        return out

    def ids(self, md):
        """Every section id in `md`, sorted."""
        return sorted({sid for sid, _, _ in self.heads(md)})

    def _ends(self, md, heads):
        """Each section's end under the general rule: the next section's start, or the next heading at its own
        level or above, whichever comes first."""
        others = [(m.start(), len(m.group(1))) for m in HEADING_LINE.finditer(md)]
        out = []
        for i, (_, start, lv) in enumerate(heads):
            end = heads[i + 1][1] if i + 1 < len(heads) else len(md)
            if lv:
                end = min([a for a, olv in others if start < a < end and olv <= lv] + [end])
            out.append(end)
        return out

    def texts(self, md, ids):
        """{id: the section's raw text, subsections included} for the sections in `ids`, in document order."""
        ids = set(ids)
        heads = self.heads(md)
        if self.numbered:
            tail = re.compile(r"^## (?!\d)", re.M).search(md, heads[-1][1]) if heads else None
            ends = [heads[i + 1][1] if i + 1 < len(heads) else (tail.start() if tail else len(md))
                    for i in range(len(heads))]
        else:
            ends = self._ends(md, heads)
        return {sid: md[start:end] for (sid, start, _), end in zip(heads, ends) if sid in ids}

    def spans(self, md, ids):
        """{id: (start, end)} for the sections in `ids`, as the write-back and the sync cut them."""
        ids = set(ids)
        heads = self.heads(md)
        if self.numbered:
            hs = [m.start() for m in re.finditer(r"^## ", md, flags=re.M)]
            return {sid: (start, min([h for h in hs if h > start], default=len(md)))
                    for sid, start, _ in heads if sid in ids}
        return {sid: (start, end) for (sid, start, _), end in zip(heads, self._ends(md, heads)) if sid in ids}

    def titles(self, md):
        """{id: the heading line as the windows show it (marks stripped)}."""
        out = {}
        for sid, start, _ in self.heads(md):
            line = md[start:md.find("\n", start) if "\n" in md[start:] else len(md)]
            out.setdefault(sid, strip_markdown(line).strip())
        return out


NUMBERED = SectionStyle()
LIST_LINE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")


def prose_scope(md, examples, style=None, skip_lists=True):
    """(in scope, {section: why it's left out}) for `md`. In scope: a section with at least one line that isn't a
    heading, a comment or an example line (the section's own heading line never counts, even when the pattern
    matches a line that isn't a markdown heading). With `skip_lists`, a section with a list line outside a comment
    is still being drafted (notes to yourself, 2026-11-09), so it stays out until they're gone."""
    style = style or NUMBERED
    out, why = [], {}
    for n, raw in style.texts(md, style.ids(md)).items():
        body = COMMENT_RE.sub("", raw)
        body = body.split("\n", 1)[1] if "\n" in body else ""
        if skip_lists and any(LIST_LINE.match(ln) for ln in body.splitlines()):
            why[n] = "notes outside a comment (list lines), so it's still being drafted"
            continue
        lines = [ln.strip() for ln in body.splitlines()]
        if any(ln and not ln.startswith("#") and ln not in examples for ln in lines):
            out.append(n)
        else:
            why[n] = "no prose yet, only headings, comments or examples"
    return sorted(out), why


def style_of(job):
    """A job's SectionStyle, or the numbered default for a stand-in job that has none (tests pass bare namespaces)."""
    return getattr(job, "style", None) or NUMBERED


def job_name(name=None, config=None):
    """The job a command works on: `name` when given, else [job] default in config.toml. There's no built-in
    default, so a command with neither says how to name one."""
    if name:
        return name
    if config is None:
        from .settings import load_config
        try:
            config = load_config()
        except OSError:
            config = {}
    default = str((config.get("job") or {}).get("default") or "").strip()
    if not default:
        raise JobError("no job named: pass --job NAME (a folder in jobs/), or set default = \"NAME\" under [job]"
                       " in config.toml")
    return default


NOTES_TEMPLATE = """Below are numbered paragraphs from <<DESCRIPTION>>. For EACH paragraph, write the notes its writer might have jotted down before writing it: 3 to 6 short bullet points, in order, each a plain statement of one point the paragraph makes. Rules:
- Put any name, number, quotation or citation the paragraph contains into the notes, in double quotes, exactly as written. For a citation, add what the cited source is when you know it (a paper, a technical report, a blog post), so the writer can refer to it truthfully.
- Where there's a joke, say in a few plain words what the joke does (what it mocks, exaggerates or compares), without quoting its wording and without listing its images. The writer finds their own image.
- Otherwise don't reuse the paragraph's own phrasing. The notes say what to say, never how to say it.
- Content only: no instructions, reminders or advice to the writer.
- If the paragraph hedges a claim (a word like "probably", "once", "maybe", "about", "might", "I think", "reportedly") or repeats a word on purpose for effect (like "round, and round, and round"), end its notes with one more bullet that starts with keep: and lists each such hedge and repeat exactly as written, in double quotes, separated by commas. A paragraph with neither gets no keep bullet.

Reply in exactly this plain format, nothing else:
### 1
- first note
- second note
### 2
- ...

<<PARAS>>"""
NOTES_PROMPT = describe(NOTES_TEMPLATE)       # with the default description; notes_for fills the job's own


PROTECT_VERSION = "protect-v1|"   # its own cache key, so the notes cache (NOTES_VERSION) never has to be re-paid
PROTECT_TEMPLATE = """Each numbered line below was written by a human author in <<DESCRIPTION>>. Another writer is going to reword some of these lines, and has to keep the author's voice.

For EACH line, name 1 to 3 words from it that carry the author's own voice: an unusual, slangy, coined or vivid word choice (like "gobsmacked" or "jury-rigged"), the word a rewrite would most likely lose. Copy each word exactly as it appears in the line, one word or one hyphenated compound each. Never pick a name, a number, anything inside quotation marks, or a common everyday word.

Reply in exactly this plain format, nothing else:
### 1
word, word
### 2
word

<<LINES>>"""
PROTECT_PROMPT = describe(PROTECT_TEMPLATE)
# words a protected-word pick may never be (the fallback skips them too): function words and everyday filler
COMMON = {"the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "it", "is", "was", "that", "this", "i", "you",
          "he", "she", "they", "we", "for", "with", "as", "at", "by", "be", "so", "if", "then", "not", "no", "just",
          "really", "very", "what", "which", "who", "when", "your", "my", "their", "its", "it's", "are", "were", "have",
          "has", "had", "do", "does", "did", "can", "will", "would", "could", "should", "about", "from", "there",
          "here", "all", "some", "any", "more", "most", "one", "like", "get", "got", "make", "made", "say", "said",
          "thing", "things", "way", "know", "think", "even", "also", "only", "because", "though", "these", "those"}


class NotesError(RuntimeError):
    """Sonnet's notes reply didn't parse. The run loop treats it like a failed Claude call: retry, then hold."""


def word_in(word, text):
    """`word` stands in `text` as a whole word, character for character (a capital first letter may differ, so a
    word that opens a sentence in one and sits mid-sentence in the other still counts)."""
    if not word:
        return True
    pat = "[" + re.escape(word[0].lower()) + re.escape(word[0].upper()) + "]" + re.escape(word[1:])
    return bool(re.search(r"(?<![\w'-])" + pat + r"(?![\w'-])", text))


def protected_fallback(text, cap=3):
    """Plan 5.1's code fallback when Sonnet names no protected word: the rarest content words of the line by general
    English frequency (wordfreq's zipf scale), outside quotes, names and numbers, at most `cap`. Words under zipf 4
    count as rare; when none is that rare, the single rarest word stands in, so a line always has one."""
    from wordfreq import zipf_frequency
    bare = re.sub(r"\"[^\"]*\"|\([^)]*\)", " ", text)
    words = []
    for m in re.finditer(r"[A-Za-z][A-Za-z'-]*[A-Za-z]", bare):
        w = m.group(0)
        if w.lower() in COMMON or len(w) < 3:
            continue
        if w[0].isupper() and m.start() > 0 and not re.search(r"[.!?]\s*$", bare[:m.start()]):
            continue                                   # a name mid-sentence
        if w not in words:
            words.append(w)
    if not words:
        return []
    ranked = sorted(words, key=lambda w: (zipf_frequency(w.lower(), "en"), words.index(w)))
    rare = [w for w in ranked if zipf_frequency(w.lower(), "en") < 4.0][:cap]
    return rare or ranked[:1]


def clean_protected(words, text, cap=3):
    """Sonnet's picks, kept only when they stand in the line as written and aren't names, numbers or quoted."""
    from engine.draws import quoted
    inside = " ".join(quoted(text))
    out = []
    for w in words:
        w = w.strip().strip('"').strip("'").strip(",.;:").strip()
        if not w or len(w.split()) > 1 or w.lower() in COMMON or any(ch.isdigit() for ch in w):
            continue
        if not word_in(w, text) or (inside and word_in(w, inside)) or w in out:
            continue
        if w[0].isupper() and not text.lstrip().startswith(w):
            continue                                       # a name; it's kept anyway (names stay, plan 5.3)
        out.append(w)
    return out[:cap]


def cut_windows(md, sections, examples=None, style=None):
    """5.1 step 1 on a document's text: [{wid, section, index, text}] in document order, windows of just under
    WINDOW_WORDS cut at paragraph breaks (experiment 6's cutter). `style` is the job's SectionStyle (the numbered
    default when None)."""
    out = []
    secs = (style or NUMBERED).texts(md, sections)
    for n in sections:
        if n not in secs:
            continue
        paras = [p for p in re.split(r"\n\s*\n", prose_only(strip_markdown(secs[n]), examples)) if p.strip()]
        windows, cur, words = [], [], 0
        for p in paras:
            w = len(p.split())
            if cur and words + w > WINDOW_WORDS:
                windows.append(cur)
                cur, words = [], 0
            cur.append(p)
            words += w
        if cur:
            windows.append(cur)
        for i, w in enumerate(windows):
            out.append({"wid": f"s{n}w{i}", "section": n, "index": i, "text": "\n\n".join(w)})
    return out


def sha(t):
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]


def flat(runs_para):
    return " ".join(s for rr in runs_para for s in rr)


def window_text_of(current, heading):
    return "\n\n".join(([heading] if heading else []) + [flat(p) for p in current])


class Job:
    """One job folder. `name` is a folder in jobs/ (job_name() resolves None from config.toml); `root` is a job folder
    anywhere, for a copy or a test. jobs/README.md documents every job.toml field read here."""

    def __init__(self, name=None, root=None, bind_rules=True):
        """`bind_rules` (the default, for the command line's one job a process): the job's rules are the ones in use
        for the whole process once it loads, its own file or, with none, the base rules, so a job loaded after
        another never reads the first one's file. False leaves the rules in use alone; read the job's own with
        `with job.rules_in_use():` (a long-lived process reading several jobs does this)."""
        self.dir = Path(root) if root else JOBS / job_name(name)
        toml = self.dir / "job.toml"
        if not toml.exists():
            raise JobError(f"no job.toml in {self.dir}: make the job folder first (jobs/README.md says how)")
        try:
            self.cfg = tomllib.loads(toml.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise JobError(f"{toml} doesn't parse: {e}") from None
        j = self.cfg.get("job")
        if not isinstance(j, dict):
            raise JobError(f"{toml} has no [job] table")
        self.name = j.get("name") or self.dir.name
        self.description = str(j.get("description") or "").strip() or None    # util.describe's default when unset
        self.style = SectionStyle(pattern=j.get("section_pattern") or None, level=j.get("section_level"))
        self.original = self.dir / j.get("original", "original.md")
        self.working = self.dir / j.get("working", "working.md")
        self._scope = j.get("sections", "all")
        if not (self._scope == "all" or isinstance(self._scope, list)):
            raise JobError(f"{toml}: sections is a list of section ids or \"all\", not {self._scope!r}")
        self._sections = None
        # the voice samples: the job's own file (relative to the job folder) or SAMPLES_FILE; voice_samples picks
        # some of them by number, and with none listed every sample counts
        self.samples_file = self.path(j["samples"]) if j.get("samples") else None
        # the job's own house rules file (relative to the job folder), loaded over the defaults and carried on the job
        # as self.rules (an engine/checks.py Rules, or None for the base rules); bind_rules puts them in use
        self.rules_file = self.path(j["rules"]) if j.get("rules") else None
        self.rules = None
        if self.rules_file:
            if not self.rules_file.exists():
                raise JobError(f"{toml}: rules names {self.rules_file}, which isn't there")
            from . import checks
            try:
                self.rules = checks.load_rules(self.rules_file)
            except ValueError as e:
                raise JobError(f"{toml}: the rules file doesn't load: {e}") from None
        picked = j.get("voice_samples")
        self.voice_samples = [f"sample {n}" for n in picked] if picked is not None else None
        # who wrote which line: "scan" (the sync step runs the chat-log scan on new sentences, the old way) or
        # "file" (your_lines is the only source; a sentence it doesn't list is the machine's, and no file at all
        # means every line is the machine's)
        self.verdict_source = str(j.get("verdicts", "scan"))
        if self.verdict_source not in ("scan", "file"):
            raise JobError(f"{toml}: verdicts is \"scan\" or \"file\", not {self.verdict_source!r}")
        self.your_lines = self.dir / j.get("your_lines", "your lines.md")
        self.locks = self.dir / "locks.md"
        self.notes_cache = self.dir / "notes cache.json"
        self.bank_path = self.dir / "bank.jsonl"
        self.couldnt_crack = self.dir / "couldnt crack.md"
        # the sync step writes the example lines; else texts.AI_EXAMPLE_LINES
        self.examples_file = self.dir / j.get("example_lines", "example lines.txt")
        # lines that stay exactly as written, like headings and templates (one per line); none by default
        self.protected_file = self.dir / j.get("protected_lines", "protected lines.txt")
        # the document the write-back writes into (5.6): `document` is relative to the job folder; `real_file`, the
        # older name, is relative to the repo root. Either may be absolute
        if j.get("document"):
            self.real_file = self.path(j["document"]).resolve()
        elif j.get("real_file"):
            self.real_file = (SANDBOX / j["real_file"]).resolve()
        else:
            self.real_file = None
        # a section with list lines outside a comment counts as still being drafted and stays out of scope
        self.skip_list_sections = bool(j.get("skip_list_sections", True))
        self.branch = str(j.get("branch") or "").strip() or None
        self._night_sections = j.get("night_sections")
        self._windows = None
        if bind_rules:                    # last, so a job that fails to load never changes the rules in use
            from . import checks
            checks.use_job_rules(self.rules)

    def rules_in_use(self):
        """A `with` block that reads this job's rules (its own file, or the base rules) in the current thread only,
        and puts back whatever was in use when it ends (engine/checks.py scoped)."""
        from . import checks
        return checks.scoped(self.rules)

    def path(self, name):
        """A path named in job.toml: relative to the job folder unless it's absolute."""
        q = Path(name)
        return q if q.is_absolute() else self.dir / q

    @property
    def document(self):
        """The document the job writes back into (the same as real_file), or None."""
        return self.real_file

    @document.setter
    def document(self, value):
        self.real_file = value

    @property
    def sections(self):
        """The section ids in scope, a list. job.toml's `sections = "all"` is every section of the frozen original
        with a line of prose in it (a line that isn't a heading, a comment or an example line)."""
        if self._sections is None:
            if self._scope == "all":
                self._sections = self.prose_sections()
            else:
                self._sections = [int(x) for x in self._scope]
        return self._sections

    @sections.setter
    def sections(self, value):
        self._sections = list(value)

    def prose_sections(self, md=None):
        """Every section of `md` (the frozen original by default) that prose_scope() puts in scope."""
        if md is None:
            if not self.original.exists():
                return []
            md = self.original.read_text(encoding="utf-8")
        return prose_scope(md, self.example_lines(), self.style, skip_lists=self.skip_list_sections)[0]

    @property
    def night_sections(self):
        """The sections the night's paragraph pass works on: job.toml's night_sections, else the whole scope."""
        return [int(x) for x in self._night_sections] if self._night_sections else list(self.sections)

    def summary(self):
        """What the job knows, for the night and morning steps: its name, folder, document, scope, the night's
        sections, its branch (None when job.toml names none) and its description."""
        return {"name": self.name, "dir": self.dir, "document": self.real_file, "sections": list(self.sections),
                "night_sections": self.night_sections, "branch": self.branch, "description": self.description,
                "section_style": repr(self.style)}

    def samples(self):
        """[("sample N", text)] for the voice: the job's samples file or SAMPLES_FILE, read the way texts.load_samples
        reads it; a job file that reader can't parse (no voice_audit.py) is read as plain markdown, where each
        heading opens a sample and a file with no heading is one sample. Narrowed to voice_samples when listed."""
        from .texts import load_samples
        got = load_samples(samples=self.samples_file) if self.samples_file else load_samples()
        if not got and self.samples_file and self.samples_file.exists():
            got = plain_samples(self.samples_file.read_text(encoding="utf-8"))
        if self.voice_samples is None:
            return got
        want = set(self.voice_samples)
        return [(k, t) for k, t in got if k in want]

    def voice_header(self, words=130):
        """The samples block the writers see: the first `words` words of each voice sample, in voice_samples' order
        when listed, else the file's."""
        samples = dict(self.samples())
        keys = self.voice_samples if self.voice_samples is not None else list(samples)
        return "\n\n---\n\n".join(" ".join(samples[k].split()[:words]) for k in keys if k in samples)

    def protected_lines(self):
        """Lines the loop never rewrites (protected_lines in job.toml), stripped of bold marks: a set."""
        if not self.protected_file.exists():
            return set()
        return {ln.strip().replace("**", "") for ln in self.protected_file.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")}

    def section_titles(self):
        """{id: the section's heading as the windows show it} from the frozen original."""
        return self.style.titles(self.original.read_text(encoding="utf-8"))

    def example_lines(self):
        """The AI example lines that come out of the prose (question 1): the job's own list when the sync step has
        written one, else texts.AI_EXAMPLE_LINES (empty)."""
        if self.examples_file.exists():
            return {ln.strip() for ln in self.examples_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
        return set(AI_EXAMPLE_LINES)

    # ------------------------------------------------------------------ 5.1 step 0
    def placeholder_base(self):
        """The original with every AI example line swapped for {{example S.N}}, numbered by section, and the swaps."""
        out, swaps, section, n = [], [], None, 0
        examples = self.example_lines()
        md = self.original.read_text(encoding="utf-8")
        starts = {start: sid for sid, start, _ in self.style.heads(md)}
        at = 0
        for line in md.split("\n"):
            if at in starts:
                section, n = starts[at], 0
            at += len(line) + 1
            if line.strip() in examples:
                n += 1
                ph = f"{{{{example {section}.{n}}}}}"
                swaps.append((ph, line.strip()))
                out.append(line.replace(line.strip(), ph))
            else:
                out.append(line)
        return "\n".join(out), swaps

    def setup(self, verdicts=None, run_info=None):
        """Make the working copy (only if it isn't there: it's the loop's output) and rewrite locks.md."""
        base, swaps = self.placeholder_base()
        if not self.working.exists():
            self.working.write_text(base, encoding="utf-8")
        lines = [f"# Locks for {self.name}", "",
                 "Written by `engine/run.py` at job setup (plan 5.1 steps 0 and 3). It's derived, so the loop rewrites it.", "",
                 "## Example placeholders", "",
                 "Each AI example line in the working copy is a placeholder. The wrap-up (5.6) puts them back from the"
                 " original, character for character, and anything sent for scoring leaves them out.", "",
                 "| Placeholder | Original line |", "| --- | --- |"]
        lines += [f"| `{ph}` | {orig.replace('|', '/')} |" for ph, orig in swaps]
        if run_info:
            lines += ["", "## Per window", "",
                      "Hard locks never change: the heading, quoted text, citations and numbers. Templates stay as"
                      " written. Claude runs get regrown, keeping their must-keeps. Your lines stay as written while"
                      " Pangram passes them; a flagged one is regrown keeping its protected words (plan 5.1)."]
            for wid, info in run_info.items():
                lines += ["", f"### {wid}", ""]
                for k in ("heading", "quotes", "citations", "numbers"):
                    if info.get(k):
                        lines.append(f"- {k}: " + "; ".join(map(str, info[k])))
                lines.append(f"- your lines (with edited or both lines Pangram did not flag at the start): {info['yours']}")
                for t in info.get("templates", []):
                    lines.append(f"- template (locked): {t}")
                for a in info.get("author", []):
                    if a["regrown"] or a["asked"]:
                        words = ", ".join(a["protected"]) or "none"
                        lines.append(f"- your line p{a['para'] + 1}.{a['run'] + 1} (protected words: {words}"
                                     f"{'' if a['asked'] else ', by code'}{'; regrown' if a['regrown'] else ''}):"
                                     f" {a['text'][:90]}")
                for r in info.get("runs", []):
                    mk = r["must"]
                    keep = ", ".join(mk["hedges"] + [f"{w} x{n}" for w, n in mk["repeats"]]) or "none"
                    lines.append(f"- regrown run p{r['para'] + 1}.{r['run'] + 1} ({r['sents']} sentence(s), must-keeps: {keep}): {r['text'][:90]}")
        self.locks.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return swaps

    # ------------------------------------------------------------------ 5.1 step 1
    def windows(self):
        """[{wid, section, index, text}] in guide order. The same cut as experiment 6's window_text()."""
        if self._windows is None:
            self._windows = cut_windows(self.original.read_text(encoding="utf-8"), self.sections, self.example_lines(),
                                        self.style)
        return self._windows

    # ------------------------------------------------------------------ question 2, frozen
    def verdicts(self):
        """{sentence: verdict} from `your lines.md` (bold marks stripped to match the windows). With verdicts =
        "file" in job.toml, a missing file is no verdicts: every line is the machine's."""
        out = {}
        if self.verdict_source == "file" and not self.your_lines.exists():
            return out
        for line in self.your_lines.read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 4 or cells[0] in ("Verdict", "---") or cells[0].startswith("-"):
                continue
            v, sentence = cells[0], cells[-1].replace("**", "")
            if v in ("yours", "yours?", "claude", "edited", "both", "loop"):    # "loop": the sync step's (Mistral's)
                out[sentence] = v
        return out

    def verdict_map(self):
        return VerdictMap(self.verdicts())

    def headings(self):
        """Every heading line in scope, as the windows show it (subsection headings like "A Subsection Name" turn
        into one-line paragraphs once the marks are stripped, and stay locked), the section headings included even
        when the section pattern matches a line that isn't a markdown heading, and the job's protected lines. A
        heading the sentence splitter cuts in two ("Done? Mostly") is listed whole and in its pieces, so every piece
        stays locked."""
        from engine.draws import sents
        out = set()
        md = self.original.read_text(encoding="utf-8")
        for raw in self.style.texts(md, self.sections).values():
            for line in raw.splitlines():
                if line.startswith("#"):
                    out.add(strip_markdown(line).strip())
        titles = self.style.titles(md)
        out |= {t for n, t in titles.items() if n in set(self.sections)}
        out |= {piece for h in list(out) for piece in sents(h)}
        return out | self.protected_lines()

    def author_paragraphs(self, verdicts, exclude_text=""):
        """All-yours paragraphs of 50+ words in the in-scope sections (experiment 6's worked examples)."""
        from engine.draws import YOURS, sents
        out = []
        md = self.original.read_text(encoding="utf-8")
        for raw in self.style.texts(md, self.sections).values():
            for p in re.split(r"\n\s*\n", prose_only(raw, self.example_lines())):
                p = p.strip()
                if not p or p[0] in "#-<" or len(p.split()) < 50 or p in exclude_text:
                    continue
                vs = [verdicts.get(s.replace("**", ""), next((v for k, v in verdicts.items() if s.replace("**", "") in k), "?"))
                      for s in sents(p)]
                if all(v in YOURS or v == "?" for v in vs) and sum(v in YOURS for v in vs) >= len(vs) - 1:
                    out.append(p)
        return out

    # ------------------------------------------------------------------ 5.1 step 2
    def notes_for(self, claude, settings, conn, paras, role):
        """Sonnet's notes per paragraph, cached by paragraph text under NOTES_VERSION. `claude` is the capped
        caller, so a cap stops this like any other call."""
        cache = json.loads(self.notes_cache.read_text(encoding="utf-8")) if self.notes_cache.exists() else {}
        todo = list(dict.fromkeys(p for p in paras if sha(NOTES_VERSION + p) not in cache))
        for i in range(0, len(todo), 12):            # a dozen paragraphs a call keeps each reply short
            chunk = todo[i:i + 12]
            listing = "\n\n".join(f"{k + 1}. {p}" for k, p in enumerate(chunk))
            prompt = describe(NOTES_TEMPLATE, self.description).replace("<<PARAS>>", listing)
            out, _ = claude(settings, conn, prompt, role=role, model="sonnet",
                            max_budget_usd=0.6)
            blocks = re.split(r"^###\s*(\d+)\s*$", out, flags=re.M)
            got = {int(blocks[k]): [ln[2:].strip() for ln in blocks[k + 1].splitlines() if ln.startswith("- ")]
                   for k in range(1, len(blocks) - 1, 2)}
            missing = [k + 1 for k in range(len(chunk)) if not got.get(k + 1)]
            if missing:
                raise NotesError(f"notes missing for paragraphs {missing}; Sonnet replied:\n{out[:600]}")
            for k, p in enumerate(chunk):
                cache[sha(NOTES_VERSION + p)] = got[k + 1]
            # written whole to a temporary file and moved into place, so a crash mid-write can't leave broken JSON
            tmp = self.notes_cache.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
            shutil.move(str(tmp), str(self.notes_cache))
        return [cache[sha(NOTES_VERSION + p)] for p in paras]

    def have_notes(self, paras):
        cache = json.loads(self.notes_cache.read_text(encoding="utf-8")) if self.notes_cache.exists() else {}
        return all(sha(NOTES_VERSION + p) in cache for p in paras)

    def _cache(self):
        return json.loads(self.notes_cache.read_text(encoding="utf-8")) if self.notes_cache.exists() else {}

    def _save_cache(self, cache):
        tmp = self.notes_cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        shutil.move(str(tmp), str(self.notes_cache))

    def protected_cached(self, text):
        """The protected words already known for one of your lines ({"words": [...], "from": "sonnet"|"code"}),
        or None."""
        return self._cache().get(sha(PROTECT_VERSION + text))

    def protected_for(self, claude, settings, conn, lines, role):
        """Plan 5.1: 1 to 3 protected voice words per line of yours, named by Sonnet in one call per batch and cached
        by the line's text (its own key, so the notes cache stays as it is). A line Sonnet gives nothing usable for
        gets the code fallback (rarest words). Returns {line: [words]}."""
        cache = self._cache()
        todo = list(dict.fromkeys(t for t in lines if sha(PROTECT_VERSION + t) not in cache))
        for i in range(0, len(todo), 20):
            chunk = todo[i:i + 20]
            listing = "\n\n".join(f"{k + 1}. {t}" for k, t in enumerate(chunk))
            prompt = describe(PROTECT_TEMPLATE, self.description).replace("<<LINES>>", listing)
            out, _ = claude(settings, conn, prompt, role=role, model="sonnet",
                            max_budget_usd=0.3)
            blocks = re.split(r"^###\s*(\d+)\s*$", out, flags=re.M)
            got = {int(blocks[k]): blocks[k + 1].strip() for k in range(1, len(blocks) - 1, 2)}
            if not got:
                raise NotesError(f"protected words missing; Sonnet replied:\n{out[:400]}")
            for k, t in enumerate(chunk):
                words = clean_protected(re.split(r"[,\n]", got.get(k + 1, "")), t)
                cache[sha(PROTECT_VERSION + t)] = ({"words": words, "from": "sonnet"} if words
                                                   else {"words": protected_fallback(t), "from": "code"})
            self._save_cache(cache)
        return {t: cache[sha(PROTECT_VERSION + t)]["words"] for t in lines}

    # ------------------------------------------------------------------ the working copy
    def assemble(self, best_by_window, write=True, keep_others=False):
        """The working copy: the placeholder base with every window's paragraphs swapped for its best text.
        `best_by_window` maps wid -> (original paragraphs, current paragraphs). Bold marks come back where the
        bolded words survive. `keep_others` (a --window or --smoke run, which only knows some windows): the other
        windows keep what the working copy already has, instead of going back to the original."""
        md, _ = self.placeholder_base()
        base = re.split(r"(\n\s*\n)", md)
        blocks = list(base)
        if keep_others and self.working.exists():
            have = re.split(r"(\n\s*\n)", self.working.read_text(encoding="utf-8"))
            if len(have) == len(base):
                blocks = have
        index = {}
        for i, b in enumerate(base):
            key = strip_markdown(COMMENT_RE.sub("", b)).strip()
            if key:
                index.setdefault(key, i)
        for orig_paras, cur_paras in best_by_window.values():
            for o, c in zip(orig_paras, cur_paras):
                if o not in index:
                    continue
                i = index[o]
                new = base[i] if o == c else c
                for bold in re.findall(r"\*\*(.+?)\*\*", base[i]):
                    if bold in new and f"**{bold}**" not in new:
                        new = new.replace(bold, f"**{bold}**", 1)
                blocks[i] = new
        text = "".join(blocks)
        if write:
            self.working.write_text(text, encoding="utf-8")
        return text


def plain_samples(text):
    """A plain markdown samples file as [("sample N", text)]: each heading opens a sample, numbered from 1 in file
    order; text before the first heading, or a file with no heading, is a sample of its own."""
    out, cur = [], []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            if "\n".join(cur).strip():
                out.append("\n".join(cur).strip())
            cur = []
            continue
        cur.append(line)
    if "\n".join(cur).strip():
        out.append("\n".join(cur).strip())
    return [(f"sample {i + 1}", t) for i, t in enumerate(out)]


def split_author_runs(runs, locked, verdict_of=None):
    """Plan 5.1 (2026-11-09: your lines are fair game). Every run that isn't Claude's is cut into one-sentence runs,
    and each gets an `author` mark unless it's locked (a heading or a template), so a flagged line of yours is
    regrown alone and keeps its own 1 to 3 protected words; a three-line stretch as one run would carry up to nine
    and freeze (experiment 5). Claude's runs stay as build_runs grouped them. Returns (new runs, index) where
    index[pi][old ri] is the list of new run indices that run became."""
    out, index = [], []
    for p in runs:
        seq, idx = [], []
        for r in p:
            if r["claude"] or len(r["sents"]) == 1:
                pieces = [r["sents"]]
            else:
                pieces = [[s] for s in r["sents"]]
            got = []
            for sents_ in pieces:
                nr = dict(r, sents=list(sents_))
                nr["author"] = (not r["claude"]) and not locked(sents_[0])
                if nr["author"] and verdict_of:
                    nr["verdict"] = verdict_of(sents_[0])
                got.append(len(seq))
                seq.append(nr)
            idx.append(got)
        out.append(seq)
        index.append(idx)
    return out, index


def reshape_runs(entry_runs, old_runs, index):
    """A banked text's runs in the old shape (M1 kept your stretches whole) put into the new one-sentence shape.
    Returns (runs, remap) with remap {(pi, old ri): (pi, new ri)}, or (entry_runs, None) when the entry already has
    the new shape, or (None, None) when it can't be mapped. M1 never changed a run that wasn't Claude's, so a split
    run's banked text is its original sentences; anything else means the entry can't be read in this shape."""
    if entry_runs is None or len(entry_runs) != len(old_runs):
        return (entry_runs, None) if entry_runs is None else (None, None)
    new_lens = [sum(len(x) for x in idx) for idx in index]
    if [len(p) for p in entry_runs] == new_lens:
        return entry_runs, None
    if [len(p) for p in entry_runs] != [len(p) for p in old_runs]:
        return None, None
    out, remap = [], {}
    for pi, (ep, op, idx) in enumerate(zip(entry_runs, old_runs, index)):
        seq = []
        for ri, (er, orun, new_is) in enumerate(zip(ep, op, idx)):
            remap[(pi, ri)] = (pi, new_is[0])
            if len(new_is) == 1:
                seq.append(er)
            elif er == orun["sents"]:
                seq += [[s] for s in er]
            else:
                return None, None
        out.append(seq)
    return out, remap


RANK = {"claude": 0, "loop": 0, "edited": 1, "both": 2, "yours?": 3, "yours": 4}
FL_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")      # find_lines' own sentence split


class VerdictMap(dict):
    """{sentence: verdict}, looked up the way build_runs needs it. find_lines splits after "et al." and the windows
    don't, so a window sentence can span two listed ones: then each piece is looked up and the most Claude-ward
    verdict wins (a piece too short for find_lines to list doesn't count)."""

    def get(self, s, default=None):
        v = super().get(s)
        if v:
            return v
        found = []
        for piece in FL_SPLIT.split(s):
            pv = super().get(piece) or next((val for k, val in self.items() if piece in k), None)
            if pv:
                found.append(pv)
            elif len(piece.split()) >= 5:
                return default
        return min(found, key=RANK.get) if found else default


class Bank:
    """Every round's text and runs, one JSON line each (plan 5.4 step 3). Kinds:
        start     round 0: the original window's score and its flagged segments
        seed      an experiment's best banked text, imported once
        round     a round of a pass (ai is null when Pangram couldn't score it)
        cut       the drop rule cut a run (5.4)
        pass_end  a pass finished, with its state
        rescore   an unscored round's (or cut's) text scored later
        drop_no   the drop rule's judge wouldn't let a run go (so a resumed pass doesn't ask again)
    The whole-text round adds whole, context, ship and whole_end (engine/run.py).

    A last line that isn't whole JSON (a process ended in the middle of an append: the night run's deadline kill) is
    left out of the load and kept as `torn`, its raw text; the load never writes. The first add after such a load cuts
    the file back to the end of its last whole line and appends the torn text, under a line with the time, to
    `<bank name>.torn` beside it, so the new entry starts on its own line. A bad line anywhere else still raises: that
    isn't a kill's work. An add onto a whole last line with no newline after it puts the newline in first.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.entries = []
        self.torn = None              # the raw text of a torn last line left out of the load, else None
        self._torn_bytes = None       # those bytes, and where they start: the first add cuts the file there
        self._cut_at = None
        if self.path.exists():
            data = self.path.read_bytes()
            rows, at = [], 0
            for piece in data.split(b"\n"):
                if piece.strip():
                    rows.append((at, piece))
                at += len(piece) + 1
            for i, (start, piece) in enumerate(rows):
                try:
                    self.entries.append(json.loads(piece.decode("utf-8")))
                except ValueError:            # json.JSONDecodeError and UnicodeDecodeError both are
                    if i != len(rows) - 1:
                        raise
                    self.torn = piece.decode("utf-8", errors="replace").rstrip("\r")
                    self._torn_bytes, self._cut_at = data[start:], start

    def _cut_torn(self):
        """Cut the torn last line the load left out, once, keeping its text in the `.torn` file. When the file no
        longer ends with the bytes the load saw (something else wrote to it since), it's left as it is."""
        import time
        at, tail = self._cut_at, self._torn_bytes
        self._cut_at = self._torn_bytes = None
        with self.path.open("rb") as f:
            f.seek(at)
            if f.read() != tail:
                return
        with self.path.with_name(self.path.name + ".torn").open("a", encoding="utf-8") as f:
            f.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} the torn last line cut from {self.path.name}:\n")
            f.write(self.torn + "\n")
        with self.path.open("r+b") as f:
            f.truncate(at)

    def add(self, entry):
        if self._cut_at is not None:
            self._cut_torn()
        self.entries.append(entry)
        open_line = False
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as f:
                f.seek(-1, 2)
                open_line = f.read(1) != b"\n"
        with self.path.open("a", encoding="utf-8") as f:
            if open_line:
                f.write("\n")
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
        return entry

    def position(self, entry):
        """Where an entry sits in the bank (by identity, since two entries can hold equal data)."""
        return next((i for i, e in enumerate(self.entries) if e is entry), -1)

    def of(self, wid, kind=None):
        return [e for e in self.entries if e.get("window") == wid and (kind is None or e.get("kind") == kind)]

    def start(self, wid):
        s = self.of(wid, "start")
        return s[-1] if s else None

    def scores(self, wid):
        """{text: (ai, flagged_segs, humanized)} for every scored text of the window."""
        out = {}
        for e in self.of(wid):
            if e.get("ai") is not None and e.get("text") is not None:
                out[e["text"]] = (e["ai"], e.get("flagged_segs", []), e.get("humanized", False))
        return out

    def best(self, wid, ok=None):
        """The scored entry with runs and the lowest AI share (the latest on a tie); humanized texts never count,
        and neither does an entry `ok(entry)` turns down (the run loop's must-keep check)."""
        cands = [(i, e) for i, e in enumerate(self.entries)
                 if e.get("window") == wid and e.get("runs") is not None and e.get("ai") is not None
                 and not e.get("humanized") and (ok is None or ok(e))]
        if not cands:
            return None
        return min(cands, key=lambda ie: (ie[1]["ai"], -ie[0]))[1]

    def latest(self, wid):
        e = [x for x in self.of(wid) if x.get("runs") is not None]
        return e[-1] if e else None

    def g(self, wid):
        return max([e.get("g", 0) for e in self.of(wid)] or [0])

    def passes(self, wid):
        return max([e.get("pass", 0) for e in self.of(wid)] or [0])

    def open_pass(self, wid):
        """(pass number, its round entries) when the window's last pass has no pass_end, else None."""
        p = self.passes(wid)
        if not p:
            return None
        ents = [e for e in self.of(wid) if e.get("pass") == p]
        if any(e.get("kind") == "pass_end" for e in ents):
            return None
        return p, [e for e in ents if e.get("kind") == "round"]

    def unscored(self, wid):
        """Rounds and drop-rule cuts banked without a score (Pangram was capped), whose text has none anywhere in the
        bank. A cut counts too (M1 review, finding 9): unscored, it could never become the window's best."""
        scored = self.scores(wid)
        return [e for e in self.of(wid) if e.get("kind") in ("round", "cut") and e.get("ai") is None
                and e.get("text") is not None and e.get("text") not in scored]

    def pass_ends(self, wid):
        return self.of(wid, "pass_end")

    def reshape(self, wid, old_runs, index):
        """Put every banked entry of the window into the one-sentence shape for your lines (split_author_runs), in
        memory only: the file keeps what was written, and every load does this again. Run indices in a round's
        changes and attempts move with their runs. Returns how many entries changed shape and how many couldn't."""
        moved = lost = 0
        for e in self.of(wid):
            if e.get("runs") is None:
                continue
            runs, remap = reshape_runs(e["runs"], old_runs, index)
            if runs is None:
                e["runs_unreadable"] = e.pop("runs")       # Bank.best and resume skip an entry with no runs
                lost += 1
                continue
            if remap is None:
                continue
            e["runs"] = runs
            for c in e.get("changes", []):
                if "run" in c and (c["para"], c["run"]) in remap:
                    c["run"] = remap[(c["para"], c["run"])][1]
            if e.get("attempts"):
                e["attempts"] = {"{}.{}".format(*remap.get(tuple(map(int, k.split("."))), tuple(map(int, k.split("."))))): n
                                 for k, n in e["attempts"].items()}
            moved += 1
        return moved, lost


def best_seed(job_dir, banks, wid, model, arm=None, untagged=None):
    """The lowest-scoring experiment entry for a window: right model, runs present, scored, and not experiment 7's
    arm 1 (it copied Claude's sentences). `arm` (the writer arms, 2026-11-09) is the loop's arm name ("fewshot",
    "notes"): an entry that names a writer arm has to name the same one. Experiment 7's `arm` is a number (its
    three prompt shapes), not a writer arm, and is read as before. An entry with no window id counts for `untagged`
    (job.toml's [seed] untagged_window; the first experiments banked one window and named none), and for no
    window when that's None."""
    best = None
    for name in banks:
        p = Path(job_dir) / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("model") != model or not e.get("runs") or e.get("ai") is None:
                continue
            if e.get("window", untagged) != wid:
                continue
            if name.startswith("exp07") and e.get("arm") == 1:
                continue
            if arm and isinstance(e.get("arm"), str) and e["arm"] != arm:
                continue
            if best is None or e["ai"] < best[1]["ai"]:
                best = (name, e)
    return best


def align(runs, seeded):
    """Map a seeded text's paragraphs onto this job's runs. Returns (current runs in this job's shape, exact) or
    (None, False) when they can't be lined up.

    The experiment that banked the text may have split runs differently (the narrowed notation lock), or regrown a
    line `your lines.md` calls yours (experiment 6 read find_lines live, and the chat logs had drifted since the
    list was made). Your lines and templates are locked here, so every one of their sentences goes back in as
    written: found verbatim, it anchors the paragraph; missing, it's restored and whatever stood in its place
    goes. A Claude run standing as written anchors too. The text between two anchors goes to the Claude runs
    between them: all of it to a single run, or one seeded run each when the counts match. `exact` is True when
    the result is the seeded text itself, so its score carries over."""
    from engine.draws import sents
    if len(runs) != len(seeded):
        return None, False
    out, exact = [], True
    for p_runs, p_seed in zip(runs, seeded):
        seed_runs = [" ".join(r) for r in p_seed if r]
        P = " ".join(seed_runs)
        cur = [r["sents"][:] if not r["claude"] else None for r in p_runs]
        pending, pos = [], 0        # (run index, is_claude) still waiting for their stretch

        def fill(gap):
            nonlocal exact
            gap = gap.strip()
            claude = [j for j, c in pending if c]
            locked = [j for j, c in pending if not c]
            if locked:
                exact = False                # a locked sentence went missing: restored, its stand-in dropped
            if not claude:
                return True
            if not locked and len(claude) == 1:
                cur[claude[0]] = sents(gap) if gap else []
                return True
            pieces = [s for s in seed_runs if s and s in gap]
            if len(pieces) == len(pending) and " ".join(pieces) == gap:
                for (j, c), piece in zip(pending, pieces):
                    if c:
                        cur[j] = sents(piece) or [piece]
                return True
            return False

        for j, r in enumerate(p_runs):
            if r["claude"]:
                o = " ".join(r["sents"])
                if not pending and P[pos:].lstrip().startswith(o):
                    cur[j], pos = r["sents"][:], P.index(o, pos) + len(o)
                else:
                    pending.append((j, True))
                continue
            for s in r["sents"]:
                k = P.find(s, pos)
                if k < 0:
                    if (j, False) not in pending:
                        pending.append((j, False))
                    continue
                if not fill(P[pos:k]):
                    return None, False
                pos, pending = k + len(s), []
        if not fill(P[pos:]):
            return None, False
        if any(c is None for c in cur):
            return None, False
        if flat(cur) != P:
            exact = False
        out.append(cur)
    return out, exact


class PoolQueue:
    """Candidate pools drawn while a cap held (7.3: "candidates that still need a Claude check wait in a queue"),
    keyed by the run's whole context, so a pool only gets judged in exactly the context it was drawn for."""

    def __init__(self, path):
        self.path = Path(path)
        self.pools = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.pools, ensure_ascii=False), encoding="utf-8")
        shutil.move(str(tmp), str(self.path))

    def get(self, key):
        return self.pools.get(key)

    def put(self, key, pool, ts=None):
        if ts is not None:
            pool["queued_at"] = pool.get("queued_at") or ts
        self.pools[key] = pool
        self.save()

    def prune(self, before):
        """Drop pools queued before `before`: their context has moved on, so they'd never be judged."""
        old = [k for k, p in self.pools.items() if p.get("queued_at", before) < before]
        for k in old:
            del self.pools[k]
        if old:
            self.save()
        return len(old)

    def take(self, key):
        p = self.pools.pop(key, None)
        if p is not None:
            self.save()
        return p


def merge_pools(a, b, show):
    """Two pools drawn for the same context, as one: every passed candidate ranked by Open Pangram, the stats added."""
    if not a:
        return b
    seen, ranked = set(), []
    for c, s in sorted(list(a["ranked"]) + list(b["ranked"]), key=lambda cs: cs[1]):
        if c not in seen:
            seen.add(c)
            ranked.append([c, s])
    st = dict(a["stats"])
    for k in ("draws", "returned", "costume", "passed"):
        st[k] = a["stats"].get(k, 0) + b["stats"].get(k, 0)
    why = dict(a["stats"].get("why", {}))
    for k, n in b["stats"].get("why", {}).items():
        why[k] = why.get(k, 0) + n
    st["why"] = why
    origin = dict(b.get("origin", {}))
    origin.update(a.get("origin", {}))
    return {"shown": [c for c, _ in ranked[:show]], "ranked": ranked, "raw": a.get("raw", 0) + b.get("raw", 0),
            "rejected": (a.get("rejected", []) + b.get("rejected", []))[:4],
            "tally": ", ".join(f"{k} {n}" for k, n in sorted(why.items(), key=lambda kv: -kv[1])),
            "stats": st, "origin": origin}
