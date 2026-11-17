"""The paragraph pass (2026-11-14, at the author's request: rewrite the sections whole, paragraph by paragraph, with
the base model or one of the adapters, and put the good voice quirks back afterwards).

The run loop regrows one run at a time inside Claude's paragraphs and judges each pick against Claude's sentence, which
on the all-Claude sections 4, 5 and 6 bought 0.91 / 0.75 / 0.60 in two run-days and $35 of Pangram, with the rest of
the paragraph anchoring every pick to Claude's shape. This pass writes each paragraph whole, from the same cached notes
Sonnet wrote for the loop, with no Claude sentence in the prompt at all:

    1. Each window paragraph with Claude's runs and no line of yours is a unit. A locked sentence (a template, a
       heading) stays as an anchor: every run of Claude sentences between anchors is a span regrown (since 2026-11-14 pm;
the first version took only the last span and held a paragraph with a line of yours whole), with the
       standing text before them as the prefix both writer arms already take. A paragraph that ends on an anchor, or
       holds a line of yours, is left to the run loop and listed.
    2. Both arms draw the span from the notes: the adapter (<notes> ... <target_text>, flush) and the plain few-shot
       writer (your samples, the worked examples, Notes / Paragraph). Draws are free and banked in their own file
       (`regrow bank.jsonl`), never the loop's run-shaped bank.
    3. Code gates every draw (gate below): no copy of the original or the standing text, length 0.6 to 1.5, every
       quote and citation of the original kept (spliced exact where the writer got it nearly right), names, numbers
       and the notes' keep-words kept, no new number or name, no markup, dash, banned word, assistant tell, Reddit
       furniture, invented sighting, punctuation slip, hard AI tic or says-it-twice, en-GB spelling turned American
       (never rejected), and the section's swear and filler budgets. Open Pangram ranks what passes.
    4. Sonnet fact-checks a pick before it goes into an assembly (REGROW_FACT_PROMPT, the loop's prompt plus the
       slips of 2026-11-14): a pick that states something the original and the notes don't support is skipped, and
       so is a pick nobody checked, unless the run says --no-facts. Sonnet writes nothing. The reviewer's
       `review.json` (engine/review.py) vetoes candidates and holds hand fixes, which every assembly carries.
    5. The section is assembled as posted with the picks in and scored once on Pangram. A "humanized" verdict never
       counts as a best. Spans that sit in a flagged segment get the next ranked candidate (or fresh draws) in the
       next round; spans Pangram passed are kept. The section's best assembly is remembered across rounds.

The first scoring is the census: both arm sets (all-adapter, all-few-shot) on one section, so the humanized flag and
the arms' pass rates are read before the rest is paid for. Nothing here touches the working copy, the loop's bank or
the document: the picks go to `regrow picks.json` and a report in analytics/, for the author to judge.

    python -m engine.regrow --job NAME --section 2,3 --census        # draw and gate, tally the reject reasons, score nothing
    python -m engine.regrow --job NAME --section 2,3 --max-usd 10    # the census, then the rounds, under the ceiling
    python -m engine.regrow --job NAME --section 2,3 --arms notes    # one arm only
    python -m engine.regrow --job NAME --section 2 --final --max-usd 2        # score the served text once (vetoes out, hand fixes in)
    python -m engine.regrow --job NAME --section 3 --arm-test --max-usd 1.7   # adapter against few-shot on the same spans
    python -m engine.regrow --job NAME --section 2,3 --arms base2 --only-stuck    # a second base model on the stuck spans

--job may be left out when config.toml names a default job ([job] default); --section defaults to the job's scope.

The arm test (2026-11-14) asks whether Pangram's "humanized" call belongs to the adapter or to Mistral text in
general: over the units where both Mistral arms have a candidate that passes, it builds one assembly from each arm,
the rest of the section standing, and scores the two. A test scoring is never a best and never served.

Every draw and scoring carries the sha of the span it was made for (`regrow origs.json` speaks for the older entries),
so after a re-sync shifts the labels no pick is spliced into a different span.
"""
import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

from . import checks as _checks
from .checks import (connective_problem, house_words, invented_sighting,
                     measured_swear_rates, must_kept, names_missing, names_required, punctuation_problem,
                     reddit_tell, restated_pairs, section_counts, section_filler_problem, section_swear_problem,
                     split_sentences, splice_citations, splice_quotes, strip_footnote_marks, to_us)
from . import review as rv
from .job import NOTES_VERSION, Bank, Job, flat, sha, style_of
from .settings import Settings
from .util import describe, report_file
from .wrapup import as_posted, flagged_segments, layout, norm, restore_examples

ARMS = ("notes", "fewshot", "olmo", "base2")   # olmo (2026-11-14 evening): Olmo 3.1 32B Instruct-SFT, chat template
MISTRAL_ARMS = ("notes", "fewshot")             # the two arms on the loop's server, and the CLI's default
SOLO_ARMS = ("olmo", "base2")                   # each runs on its own server, so never with another arm
OLMO_MODEL = "olmo31-32b-sft"
BASE2_MODEL = "olmo3-32b-iq3"   # base2: the few-shot prompt completed by a second base model (regrow.base2_model)
SEED_OFFSET = {"notes": 0, "fewshot": 500, "olmo": 800, "base2": 300}
EXAMPLE_SEP = "\n\n---\n\n"
COPY_RATIO = 0.85            # a draw this close to the original (or the standing text) is a copy, not a rewrite
MIN_RATIO, MAX_RATIO = 0.6, 1.5
SEED_BASE = 31000
SENT_BREAK = (r"(?<!\bal\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<!\betc\.)(?<=[.!?])\s+"
              r"|(?<=\.[\"')])\s+|(?<=[!?][\"')])\s+(?=[\"'(A-Z])")   # case-blind after `."`; `?" she said` is a continuation
ORIGS_FILE = "regrow origs.json"   # {label: orig sha} for the bank's entries from before they carried one (2026-11-14)
# the cache key's version for the pass's own fact prompt: verdicts made under the loop's FACT_PROMPT aren't reused
FACTS_VERSION = "regrow facts 2026-11-14 v2"
# capitalized everyday words that aren't names of people, works or companies, so a new one is no new name
# ("like a Christmas tree" drew 73 rejects on one unit, 2026-11-14)
EVERYDAY_CAPS = frozenset({
    "Christmas", "Halloween", "Easter",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
    "December",
    "God", "Jesus", "Christ"})

# the pass's fact check (the nightly loop keeps draws.FACT_PROMPT): the same contract, plus the four kinds of slip
# the paragraph pass made on 2026-11-14, each with an example of its own kind
REGROW_FACT_PROMPT = """A rewrite of one stretch of an essay has to say only what the original stretch says. List every problem of these kinds in the rewrite:

1. A factual statement, number, name, example or technical claim that is wrong, changed, or not supported by the original stretch and the notes.
2. A claim, judgment or closing line the original doesn't make. Example: the original says "Let's start with the easiest mistake, which is the one where the recipe skips the resting time." and the rewrite adds "This one is so common, and so annoying, that it's practically a rite of passage." Another: an invented closer such as "But let's take a closer look..."
3. A pointer to another part of the essay that changes what it points at. Example: the original says "As with the salt rule from section 2, every cook picks these habits up early" and the rewrite says "You know, from the salt rule we covered in section 2.", which now says the habits come from the salt rule.
4. A quoted example that loses its quote marks or gets run into the narration. Example: the original says so "Stir. Wait." and "Taste it. Add salt." are really one routine, and the rewrite says Stir. Wait. Taste it. Add salt. That's all there is to it.
5. A sentence that changes what the thing being described is. Example: the original says And at the cheap end you get "winging it," where the cook skips the recipe, and the rewrite says The cheapest way to plan a meal is: winging it, which turns a kind of mistake into a way of planning a meal.

Looser wording, a different joke that claims nothing new, or a dropped point do not count here.

Original stretch:
<<ORIGINAL>>

Notes for the paragraph:
<<NOTES>>

Rewrite:
<<NEW>>

Reply with NONE when there is no problem, or one short line per problem."""


def orig_key(orig):
    """The sha a bank entry carries for the unit's original span, so a pick never lands in a span it wasn't made for."""
    return sha(" ".join(orig.split()))


def fact_key(orig, text):
    """The cache key of a strict verdict (REGROW_FACT_PROMPT, versioned)."""
    return sha(FACTS_VERSION + "\x1f" + orig + "\x1f" + text)


def loose_key(orig, text):
    """The cache key of a loose verdict (draws.FACT_PROMPT), as the pass wrote it before 2026-11-14 evening."""
    return sha(orig + "\x1f" + text)


FACT_GATES = ("loose", "strict")


# ---------------------------------------------------------------------- the units
def spans_of(w, pi):
    """The spans paragraph `pi` of a layout window regrows: [(start, end)] run index ranges, one per maximal run of
    Claude sentences, in order. A line of yours and a locked sentence are anchors that stay; the Claude text around
    them is regrown (the author's hold-and-regrow-the-surroundings theory, 2026-11-14). {"why": ...} when there is
    nothing of Claude's."""
    runs = w["runs"][pi]
    spans, start = [], None
    for ri, r in enumerate(runs):
        if r["claude"] and not r.get("author"):
            if start is None:
                start = ri
        elif start is not None:
            spans.append((start, ri))
            start = None
    if start is not None:
        spans.append((start, len(runs)))
    if not spans:
        return {"why": "nothing of Claude's"}
    return spans


def span_of(w, pi):
    """The last span only, as the first version of the pass had it (its labels are the paragraph's `pN`)."""
    sp = spans_of(w, pi)
    if isinstance(sp, dict):
        return sp
    start, _ = sp[-1]
    return {"start": start, "prefix_runs": list(range(start))}


def units(lay, sections, standing_runs, standing_text=None):
    """One unit per regrowable span: label (`s6w0 p1` for a paragraph's last span, `s6w0 p1 s1`, `s1 s2`... for the
    spans before it, so the first version's banked draws keep their labels), plabel (the paragraph's), wid, pi,
    section, start and end (run indices), the paragraph's original text (the notes key), the original span (Claude's,
    for the facts), the standing span (what the working copy holds now), the prefix and suffix (standing text before
    and after the span), the paragraph's standing runs and its standing text. `standing_runs[plabel]` is the
    paragraph's standing text split into its runs (fixpass.gather), or None when it doesn't split (the original runs
    stand in); `standing_text[plabel]` is the paragraph as the working copy holds it."""
    out, skipped = [], []
    standing_text = standing_text or {}
    for wid, w in lay.items():
        if w["section"] not in sections:
            continue
        for pi, p in enumerate(w["original"]):
            plabel = f"{wid} p{pi + 1}"
            sp = spans_of(w, pi)
            if isinstance(sp, dict):
                if any(r["claude"] or r.get("author") for r in w["runs"][pi]):
                    skipped.append((plabel, sp["why"]))
                continue
            orig_runs = [" ".join(r) for r in p]
            cur_runs = standing_runs.get(plabel) or orig_runs
            if len(cur_runs) != len(orig_runs):
                cur_runs = orig_runs
            whole = standing_text.get(plabel) or " ".join(x for x in cur_runs if x)
            for k, (start, end) in enumerate(sp):
                label = plabel if k == len(sp) - 1 else f"{plabel} s{k + 1}"
                whole_span = start == 0 and end == len(cur_runs)
                out.append({"label": label, "plabel": plabel, "wid": wid, "pi": pi, "section": w["section"],
                            "start": start, "end": end, "para": flat(p),
                            "orig": " ".join(x for x in orig_runs[start:end] if x),
                            "standing": whole if whole_span else " ".join(x for x in cur_runs[start:end] if x),
                            "prefix": " ".join(x for x in cur_runs[:start] if x),
                            "suffix": " ".join(x for x in cur_runs[end:] if x),
                            "runs": list(cur_runs), "para_standing": whole})
    return out, skipped


# ---------------------------------------------------------------------- the gate
def keep_words(notes):
    for n in notes:
        if re.match(r"^\s*keep\s*:", n, re.I):
            return [q for q in re.findall(r'"([^"]+)"', n) if q.strip()]
    return []


def quoted(s):
    return re.findall(r"\"[^\"]+\"|\([^)]*\d{4}\)", s)


def numbers(s):
    return set(re.findall(r"\d+(?:[.,/]\d+)*", s))


def cap_ratio(t):
    """The share of sentences that open on a capital (the adapter's Reddit training wrote lowercase openers)."""
    starts = []
    for s_ in re.split(SENT_BREAK, t):
        m = re.match(r"[\"'(]*([A-Za-z])", s_)       # a piece that opens on a digit or a bracket isn't a sentence start
        if m:
            starts.append(m.group(1))
    return sum(c.isupper() for c in starts) / max(1, len(starts))


def new_names(cand, known):
    """Capitalized words mid-sentence that the window, the samples and the original run don't know. A sentence's
    first word is an opener, not a name, and a sentence that opens after a closing quote mark counts as one too
    (draws.names() misses that and called "Which" a name). Plurals, possessives and a hyphenated word whose first
    part is known pass, as names_ok has it."""
    norm_ = lambda n: re.sub(r"(?:'s|s)$", "", n.lower())  # noqa: E731
    have = {norm_(n) for n in known}
    out = []
    for s_ in split_sentences(cand):
        for m in re.finditer(r"\b[A-Z][a-zA-Z0-9-]+", s_):
            n = m.group(0)
            before = s_[:m.start()].rstrip()
            # an opener: the sentence's first word, the first word inside a quote mark or a bracket, or the first word
            # after one closes (the census's "new name: Or" was `...smelled like") Or, "How far...`), or after a colon
            if not before or before[-1] in "\"'()“”‘’:.?!" or before.endswith(" -"):
                continue
            if n in {"I", "I'm", "I'll", "I've", "I'd"}:
                continue
            if re.sub(r"'s$", "", n) in EVERYDAY_CAPS:      # the raw word: norm_ would make "Christmas" "christma"
                continue
            if n == "Section" and re.match(r"Section \d", s_[m.start():]):     # a document's own section numbers
                continue
            if norm_(n) not in have and norm_(n.split("-")[0]) not in have:
                out.append(n)
    return out


def normalize(cand, orig):
    """A draw on its way to the gate: whitespace squashed, en-GB spelling turned American, the original's quotes and
    citations put back exact where the writer got them nearly right, footnote marks out."""
    c = " ".join(cand.split())
    c = to_us(c)
    c = splice_quotes(c, quoted(orig))
    c = quote_stops(c)
    c = splice_citations(c, orig)
    return strip_footnote_marks(c, orig).strip()


def quote_stops(c):
    """A spliced quote closes on its own stop, and the writer's own stop then sits outside it (`ChatGPT.". Next`,
    `"the words,", "it,"`, `"hard,". This`). The outer stop goes; when it was a period after an inner comma, the comma
    becomes the period, American style."""
    c = re.sub(r',"\.', '."', c)
    return re.sub(r'([.!?,])"[.,]', r'\1"', c)


def gate(cand, *, orig, standing, notes, prefix, after, known, tics, rest=None, rates=None, multiplier=None,
         filler_cap=None, filler_words=None, names_fn=None, markup_re=None, tell_re=None, costume_re=None,
         keeps=True):
    """The reject reason for one normalized candidate span, or None. Every check is code; the order puts the cheap
    and certain ones first, as the loop's filter stack does."""
    from engine.draws import ASSISTANT_TELL, COSTUME_OPENER, MARKUP, run_names
    markup_re, tell_re, costume_re = markup_re or MARKUP, tell_re or ASSISTANT_TELL, costume_re or COSTUME_OPENER
    if not cand.strip():
        return "empty"
    w_orig, w = len(orig.split()), len(cand.split())
    if not MIN_RATIO * w_orig <= w <= max(MAX_RATIO * w_orig, 14):
        return f"length {w / max(1, w_orig):.2f}"
    for ref in (orig, standing):
        if ref and difflib.SequenceMatcher(None, ref.split(), cand.split()).ratio() > COPY_RATIO:
            return "copied the original"
    if house_words(cand):
        return "dash or house word"
    if "�" in cand or re.search(r"[½¼¾⅓⅔⅛̀-ͯ]", cand):
        return "bad glyph"
    if markup_re.search(cand) and not markup_re.search(orig):
        return "markup"
    if reddit_tell(cand):
        return "reddit tell"
    if tell_re.search(cand):
        return "assistant tell"
    if costume_re.match(cand):
        return "assistant costume"
    if cand.startswith('"') and not orig.startswith('"'):
        return "invented dialogue"
    for q in quoted(orig):
        if not must_kept(q.strip('"').strip(",.;:"), cand, orig):
            return f"quote lost: {q[:30]}"
    miss = names_missing(cand, names_required(orig, notes))
    if miss:
        return "name lost: " + ", ".join(miss)
    if not numbers(orig) <= numbers(cand):
        return "number lost: " + ", ".join(sorted(numbers(orig) - numbers(cand)))
    if not numbers(cand) <= numbers(orig):
        return "new number: " + ", ".join(sorted(numbers(cand) - numbers(orig)))
    nn = new_names(cand, set(known) | run_names(orig))
    if nn:
        return "new name: " + ", ".join(nn)
    for k in (keep_words(notes) if keeps else ()):      # a Claude span's hedges are Claude's; the pass doesn't hold them
        if not re.search(r"\b" + re.escape(k) + r"\b", cand, re.I):
            return f"keep lost: {k}"
    if invented_sighting(cand, orig):
        return "invented sighting"
    p = punctuation_problem(cand, orig)
    if p:
        return f"punctuation: {p}"
    if tics(cand)[0]:
        return "hard tic"
    if cap_ratio(cand) < 0.8:
        return "lowercase register"
    if restated_pairs(split_sentences(cand)) and not restated_pairs(split_sentences(orig)):
        return "says it twice"
    cp = connective_problem(cand, prefix, after)
    if cp:
        return cp
    if rest and rates:
        sp = section_swear_problem(cand, orig, rest, rates, multiplier)
        if sp:
            return sp
        fp = section_filler_problem(cand, orig, rest, filler_cap, filler_words)
        if fp:
            return fp
    return None


# ---------------------------------------------------------------------- the prompts
def adapter_prompt(notes, prefix):
    """The adapter's training format, flush after the prefix (a trailing space ends the paragraph at once)."""
    return "<notes>\n" + "\n".join(f"- {n}" for n in notes) + "\n</notes>\n<target_text>\n" + prefix


def fewshot_prompt(header, examples_block, notes, prefix):
    from engine.draws import fmt_example
    p = "\n\n---\n\n".join([header, examples_block, fmt_example(notes, prefix)])
    return p + (" " if prefix else "")


def seed_for(unit, arm, k):
    m = re.match(r"s(\d+)w(\d+)", unit["wid"])
    return SEED_BASE + ((int(m.group(1)) * 100 + int(m.group(2)) * 10 + unit["pi"]) * 1000 + SEED_OFFSET.get(arm, 500) + k)


def olmo_prompt(header, notes, prefix, suffix, description=None):
    """One chat turn for an instruct writer: the author's voice samples, the paragraph's notes, the text before and
    after the span, and the ask. The answer is the span alone. `description` is the job's line about the document."""
    parts = [describe("You are ghost-writing one paragraph of <<DESCRIPTION>>. It has to read as the author wrote it."
                      " Here is how the author writes (samples of their own non-fiction):", description), "", header, "",
             "Notes for the paragraph. Every quote, number, name and citation in them must appear exactly as given;"
             " a note starting with keep: lists words the author uses, which are optional.", ""]
    parts += [f"- {n}" for n in notes]
    if prefix:
        parts += ["", "The paragraph so far (do not repeat it):", "", prefix]
    if suffix:
        parts += ["", "What follows your text in the same paragraph (do not repeat it, and lead into it):", "", suffix]
    ask = "Write the rest of the paragraph" if prefix else "Write the paragraph"
    parts += ["", f"{ask} from the notes, in the author's voice, as plain prose: no markdown, no list, no heading, no"
              " em dashes (a spaced hyphen is the author's pause), no preamble and no closing comment. Output the prose only."]
    return "\n".join(parts)


def olmo_answer(out):
    """The prose out of an instruct answer: a preamble line ending in a colon goes, so does a wrapping pair of quotes,
    and the paragraphs are joined (the gate's length rule judges what's left)."""
    lines = [ln.strip() for ln in out.strip().split("\n") if ln.strip()]
    if len(lines) > 1 and lines[0].endswith(":") and len(lines[0].split()) <= 12:
        lines = lines[1:]
    text = " ".join(lines).strip()
    if len(text) > 2 and text[0] == text[-1] == '"' and text.count('"') % 2 == 0:
        text = text[1:-1].strip()
    return text


# ---------------------------------------------------------------------- the pass
class RegrowPass:
    # defaults for a pass built without __init__ (the tests): no review, no hand fixes, the fact check required
    review = {"vetoes": (), "repairs": (), "for_you": (), "stuck": ()}
    fixes: dict = {}
    stuck_entries: dict = {}      # label -> the reviewer's stuck entry for the unit (pins nothing, changes no pick)
    stuck_parts: tuple = ()       # labels whose stuck entry speaks for a part of its original (review.is_part)
    readonly = False              # True: load_origs never writes the sidecar (pass_paragraphs(readonly=True))
    no_sidecar = 0                # with readonly and no sidecar: the bank's entries with no orig sha, left out
    stuck_unmatched: tuple = ()   # labels of the stuck entries that speak for no unit at all
    review_unmatched: tuple = ()  # (kind, label) of the vetoes, repairs and hand fixes whose original no unit holds
    lost_pins: dict = {}          # old span sha -> the text a repair or hand fix pinned there before you rewrote it
    repairs: dict = {}            # label -> the repair that stands for the unit (resolve_repairs)
    pins: dict = {}               # label -> the text a unit always carries: its hand fix, or its standing repair
    anchor: dict = {}             # section -> the scoring the repairs were read against
    bases: dict = {}              # section -> its review base (the newest scoring a review entry names), or None
    stale_repairs: tuple = ()
    sidecar: dict = {}
    all_origs: dict = {}
    sha_label: dict = {}          # orig sha -> the one label that holds that span now (shas held twice left out)
    orig_text: dict = {}
    section_of: dict = {}
    no_facts, facts_warned, fact_check, claude = False, False, True, None
    tests: list = []              # the arm test's scorings, apart from `scores` so no best or round ever reads them
    arm_tests: dict = {}          # section -> what the arm test found (report_md's "Arm test" part)
    base2_runs: dict = {}         # section -> what the base2 step found (report_md's "Second base model" part)
    test_runs: dict = {}          # label -> what --test-scoring did with that scoring (report_md's "Test scorings")
    only = None                   # --only-stuck: the labels the census draws for, or None for every unit
    stuck: dict = {}              # --only-stuck: section -> its stuck labels, read before the census
    example_parts: tuple = ()     # the worked examples one by one, so the base2 prompt can drop some to fit
    base2_prompts: dict = {}
    trim_warned = False
    fact_gate = "loose"
    raw_forms: dict = {}          # a text after quote_stops -> the banked texts it came from (their verdicts count)
    waiting: dict = {}            # label -> why, units the assembly being built lost to a missing fact check
    unscored: dict = {}           # section -> units waiting on a check, for the sections the run left unscored
    dry_why = None                # why the run stopped scoring (the fact check off or out of money), or None

    def __init__(self, *, job, bank, settings, conn, claude=None, pangram=None, server=None, op=None, purpose="experiment",
                 max_usd=None, max_claude_usd=None, arms=ARMS, log=print, no_facts=False):
        cfg = settings.config.get("regrow", {}) if settings is not None else {}
        self.job, self.bank, self.settings, self.conn = job, bank, settings, conn
        self.claude, self.pg, self.srv, self.op = claude, pangram, server, op
        self.purpose, self.log = purpose, log
        self.arms = tuple(a for a in arms if a in ARMS) or ARMS
        self.draws = int(cfg.get("draws", 8))
        self.more = int(cfg.get("more_draws", 8))
        self.rounds = int(cfg.get("rounds", 4))
        self.max_usd = float(max_usd if max_usd is not None else cfg.get("max_pangram_usd", 10.0))
        self.max_claude_usd = float(max_claude_usd if max_claude_usd is not None else cfg.get("max_claude_usd", 5.0))
        self.fact_check = bool(cfg.get("fact_check", True))
        # regrow.fact_gate: "loose" stops wrong facts only (small slips are the repair step's); "strict" asks with
        # REGROW_FACT_PROMPT. Either way an unchecked candidate can't be picked without --no-facts
        self.fact_gate = str(cfg.get("fact_gate", "loose"))
        if self.fact_gate not in FACT_GATES:
            raise ValueError(f"regrow.fact_gate is {self.fact_gate!r}; it takes {' or '.join(FACT_GATES)}")
        self.waiting, self.unscored, self.dry_why = {}, {}, None
        self.no_facts = bool(no_facts)    # --no-facts: an unchecked candidate may be picked; otherwise it can't be
        self.no_checked = {}              # label -> why the unit had no fact-checked candidate (the report counts them)
        self.review = rv.load(job.dir)
        self.noise = float(cfg.get("noise", 0.05))
        # the swear and filler budgets: the rules file's, or an older config's [loop] keys with a warning
        self.swear_mult, self.filler_cap, self.filler_words = _checks.budgets(
            settings.config if settings is not None else None, warn=lambda m: print(m, flush=True))
        self.spent = self.claude_spent = 0.0
        self.rbank = Bank(job.dir / "regrow bank.jsonl")
        self.units, self.skipped, self.standing = [], [], {}
        self.notes, self.pools, self.facts = {}, {}, {}
        self.scores = []              # every scoring this run: {section, label, ai, humanized, flagged, picks}
        self.best = {}                # section -> the lowest non-humanized scoring, carried over from the bank too
        self.header = self.examples_block = None
        self.tics = None
        self.rates = None
        self.known = set()
        self.swaps = []
        self.tests, self.arm_tests, self.base2_runs, self.base2_prompts = [], {}, {}, {}
        self.test_runs = {}

    # ---- setup
    def setup(self, sections, vm=None):
        from engine.draws import make_tics
        from engine.draws import names
        vm = self.load_units(sections, vm)
        cache = self.load_notes()
        self.tics = make_tics()
        self.rates = measured_swear_rates(vm)
        self.header = self.job.voice_header()
        examples = self.job.author_paragraphs(vm)
        ex_notes = [cache.get(sha(NOTES_VERSION + p)) for p in examples]
        if any(n is None for n in ex_notes):
            if self.claude is None:
                raise RuntimeError("the worked examples have no cached notes and there is no Claude caller")
            ex_notes = self.job.notes_for(self.claude, self.settings, self.conn, examples, role="examples notes")
        from engine.draws import fmt_example
        self.example_parts = tuple(fmt_example(n, p) for n, p in zip(ex_notes, examples))
        self.examples_block = EXAMPLE_SEP.join(self.example_parts)
        self.known = set()
        for u in self.units:
            self.known |= names(u["para"])
        self.known |= names(self.header) | names(self.examples_block)
        self.load_bank()
        self.spent_before = self.spent_today()       # logged; the ceiling is this run's own spend
        self.regate()

    def load_notes(self):
        """Each unit's notes from the job's cache, asking Sonnet for the missing ones when there is a Claude caller.
        Returns the cache."""
        cache = self.job._cache()
        missing = [u["para"] for u in self.units if sha(NOTES_VERSION + u["para"]) not in cache]
        if missing:
            if self.claude is None:
                raise RuntimeError(f"{len(missing)} paragraph(s) have no cached notes and there is no Claude caller")
            self.job.notes_for(self.claude, self.settings, self.conn, missing, role="regrow notes")
            cache = self.job._cache()
        for u in self.units:
            self.notes[u["label"]] = cache[sha(NOTES_VERSION + u["para"])]
        return cache

    def load_units(self, sections, vm=None):
        """The layout, the standing text and the units, with no model, no notes and no paid call."""
        from .fixpass import gather
        vm = vm if vm is not None else self.job.verdict_map()
        self.lay = layout(self.job, self.bank, vm)
        _, self.swaps = self.job.placeholder_base()
        work = self.job.working.read_text(encoding="utf-8") if self.job.working.exists() else None
        rows = gather(self.job, self.bank, work, sections, lay=self.lay) if work else []
        standing_runs = {r["label"]: r["runs"] for r in rows}
        self.standing = {r["label"]: r["text"] for r in rows}
        self.units, self.skipped = units(self.lay, sections, standing_runs, self.standing)
        self.index_units()
        return vm

    def index_units(self):
        """Every unit of every section in the layout (the job's whole scope, whatever this run's sections are), for the
        checks that look across sections: each label's original span, its sha and its section, and the reviewer's
        hand fixes by label."""
        every, _ = units(self.lay, {w["section"] for w in self.lay.values()}, {})
        self.orig_text = {u["label"]: u["orig"] for u in every}
        self.all_origs = {u["label"]: orig_key(u["orig"]) for u in every}
        held = {}
        for label, key in self.all_origs.items():
            held.setdefault(key, []).append(label)
        self.sha_label = {key: labels[0] for key, labels in held.items() if len(labels) == 1}
        self.section_of = {u["label"]: u["section"] for u in every}
        self.fixes = {}
        for u in every:
            fx = rv.hand_fix(self.review, u["orig"])
            if fx is not None:
                self.fixes[u["label"]] = fx
        self.repairs = {}                 # resolved against the bank in load_bank
        self.pins = {label: fx["text"] for label, fx in self.fixes.items()}
        # a stuck entry is its own kind: never a pin, a veto or a repair, so nothing the pass draws or picks moves
        # an entry also speaks for a unit whose original is a run of its sentences (what is left after the author
        # rewrote one of them and the sync made it an anchor); an exact match wins
        self.stuck_entries, parts = {}, []
        for u in every:
            st, part = rv.stuck_match(self.review, u["orig"])
            if st is not None:
                self.stuck_entries[u["label"]] = st
                if part:
                    parts.append(u["label"])
        self.stuck_parts = tuple(parts)
        have = {rv.squash(o) for o in self.orig_text.values()}
        self.review_unmatched = tuple((kind, r.get("label") or "?") for key, kind in
                                      (("vetoes", "veto"), ("repairs", "repair"), ("for_you", "hand fix"))
                                      for r in self.review.get(key, ()) if rv.squash(r.get("orig")) not in have)
        used = {id(e) for e in self.stuck_entries.values()}
        self.stuck_unmatched = tuple(e.get("label") or "?" for e in self.review.get("stuck") or () if id(e) not in used)

    def load_origs(self):
        """The sidecar for the bank's entries from before they carried an orig sha: {label: orig sha}, written once
        from every unit as it stands when the file doesn't exist (no sync had happened since the pass's first run, so
        those labels are the ones the old entries were made under). Never rewritten after that."""
        path = self.job.dir / ORIGS_FILE
        if path.exists():
            self.sidecar = json.loads(path.read_text(encoding="utf-8"))
            return
        if self.readonly:
            # nothing written: with no sidecar, an entry with no orig sha of its own can't be placed, so it's left out
            # (fits() finds no sha for it) and counted, and only the entries that carry their own sha are read
            self.sidecar = {}
            self.no_sidecar = sum(1 for e in self.rbank.entries
                                  if e.get("kind") in ("draw", "score") and e.get("orig_sha") is None
                                  and not e.get("origs"))
            return
        self.sidecar = dict(self.all_origs)
        path.write_text(json.dumps(self.sidecar, indent=1, sort_keys=True), encoding="utf-8")
        self.log(f"wrote {ORIGS_FILE}: {len(self.sidecar)} label(s), for the bank's entries that carry no orig sha")

    def fits(self, label, stored):
        """True when an entry made for `label` (with its orig sha, or None for an old entry, which the sidecar then
        speaks for) belongs to the unit that holds the label now."""
        cur = self.all_origs.get(label)
        if cur is None:
            return False
        return (stored if stored is not None else self.sidecar.get(label)) == cur

    def where(self, label, stored):
        """The label of the unit an entry made for `label` belongs to now, or None. Its own label when that unit still
        holds the span (fits); else the one unit whose span has the entry's orig sha (its own, or the sidecar's for an
        old entry), so a re-cut at a sync that only moves a span to another label (a paragraph moving into the window
        before it renumbers every paragraph after it) takes the entry along. The sha check is the same either way,
        so an entry never lands on a span it wasn't made for; a sha two units share follows neither."""
        if self.fits(label, stored):
            return label
        key = stored if stored is not None else self.sidecar.get(label)
        return self.sha_label.get(key) if key else None

    def load_bank(self):
        """The pass's own bank read back in: draws, facts, scorings and each section's best. Reads only (the sidecar
        is written the first time). A draw or a pick made for a span that has since changed is dropped, and a scoring
        that lost a pick that way is never a best; one made for a span that only moved to another label follows it
        (where)."""
        self.load_origs()
        self.raw_forms = {}
        for e in self.rbank.entries:                  # earlier runs' draws and facts come back for free
            if e.get("kind") == "draw":
                at = self.where(e["label"], e.get("orig_sha"))
                if at is None:
                    continue                          # dropped before "newest wins", so it never hides a valid entry
                if at != e["label"]:                  # the span moved to another label at a sync: in memory only
                    e = dict(e, label=at, orig_sha=self.all_origs[at], was_label=e["label"])
                fixed = quote_stops(e["text"])
                if fixed != e["text"]:
                    self.note_raw(e["text"])
                    e = dict(e, text=fixed, raw=e["text"])   # in memory only; the bank file stays as it is
                self.pools.setdefault((e["label"], e["arm"]), {})[e["k"]] = e
            elif e.get("kind") == "fact":
                self.facts[e["key"]] = e["facts"]
            elif e.get("kind") == "score" and e.get("ai") is not None:
                e = self.span_picks(self.fit_picks(e))
                for text in e.get("picks", {}).values():
                    self.note_raw(text)
                if e.get("test"):                     # an arm test's scoring: reported, never a best, never served
                    self.tests = self.tests + [e]
                    continue
                self.scores.append(e)                 # so a relaunch resumes the rounds instead of paying for round 1 again
        self.find_lost_pins()
        for e in self.scores:
            self.consider(e)
        self.resolve_repairs()
        if self.repairs:                              # a scoring that holds a repaired pick's `from` is no best now
            self.best = {}
            for e in self.scores:
                self.consider(e)

    def note_raw(self, text):
        """Remember a banked text whose stop slip quote_stops fixes, so its fact verdict is found for the fixed text."""
        fixed = quote_stops(text)
        if fixed != text:
            self.raw_forms.setdefault(fixed, set()).add(text)

    def review_base(self, n):
        """Section `n`'s review base: of the section's clean (not humanized), non-stale scorings that some review
        entry of the section names in its `scoring` (the label that was served when the entry was written), the one
        banked last. None when no entry names a usable one (older entries carry no `scoring`). Stuck entries never
        count here, even one with a stray `scoring`: they say nothing about what was served."""
        sections_of = {}
        for label, orig in self.orig_text.items():
            sections_of.setdefault(rv.squash(orig), set()).add(self.section_of.get(label))
        named = {r["scoring"] for kind in ("vetoes", "repairs", "for_you") for r in self.review.get(kind, ())
                 if r.get("scoring") and n in sections_of.get(rv.squash(r.get("orig")), ())}
        usable = [s for s in self.scores if s["section"] == n and s["label"] in named
                  and not s.get("humanized") and not s.get("stale") and not s.get("test")]
        return usable[-1] if usable else None

    def resolve_repairs(self):
        """Which of the reviewer's repairs stand: one stands when the unit's pick in its section's anchor is the
        repair's `from`, or already its text. The anchor is the section's review base (review_base), so a veto never
        moves it; with no base it is the best before any repair was read, else the lowest clean scoring. A repair for
        a unit whose pick has moved on is stale: logged, listed in the report, never spliced onto different text. A
        hand fix for the same unit wins over a repair."""
        self.anchor, self.stale_repairs, self.bases = {}, [], {}
        for n in set(self.section_of.values()):
            self.bases[n] = self.review_base(n)
            clean = [s for s in self.scores if s["section"] == n and not s.get("humanized")]
            self.anchor[n] = (self.bases[n] or self.best.get(n)
                              or (min(clean, key=lambda s: s["ai"]) if clean else None))
        for r in self.review.get("repairs", ()):
            for label, orig in self.orig_text.items():
                if rv.squash(orig) != rv.squash(r.get("orig")) or not r.get("text"):
                    continue
                if label in self.fixes:
                    self.log(f"repair for {label} set aside: the unit has a hand fix")
                    continue
                base = self.anchor.get(self.section_of[label])
                pick = (base or {}).get("picks", {}).get(label)
                if pick is not None and rv.repair_fits(r, pick):
                    self.repairs[label] = r
                    self.pins[label] = r["text"]
                    continue
                self.stale_repairs.append(r.get("label") or label)
                self.log(f"repair for {r.get('label') or label} is stale: the unit's pick"
                         + (" is no longer its `from`" if pick is not None else " is gone") + "; not applied")

    def fit_picks(self, sc):
        """A banked scoring with each pick under the label of the unit that holds its span now (where: a pick whose
        span only moved to another label of the same section follows it, `moved` says from where), and the picks
        whose span is gone dropped (`stale` names them). A pick whose span is gone because the text it served (the
        pick, or the repair of it the reviewer's file holds) is now, word for word, a unit's original of the same
        section is `absorbed` instead, not stale: you rewrote another sentence of its paragraph, so the sync froze
        the paragraph as the document had it, pass text included. That unit standing on its original serves the very
        text the pick served, so the scoring still holds for it, and nothing lands on any span. Its `origs` and
        `span_flagged` follow the picks. A pick made for a span of a paragraph you rewrote or removed (its sha is in
        the layout's `gone_spans`: the paragraph is in no window now) is `gone`: dropped, spliced nowhere, and not
        stale, so the scoring keeps its place (as a best, an anchor or a review base) with every other pick; your
        text stands there and the section's text is no longer the scored one, which `scored` says. Any other pick
        whose span is gone is stale. In memory only; the bank file stays as it is."""
        origs, flags = sc.get("origs") or {}, sc.get("span_flagged") or {}
        picks, new_origs, new_flags, moved, stale, absorbed, gone, held = {}, {}, {}, {}, [], {}, {}, {}
        gone_spans = getattr(getattr(self, "bank", None), "gone_spans", None) or {}
        for k, v in sc.get("picks", {}).items():
            at = self.where(k, origs.get(k))
            if at is not None and at != k and self.section_of.get(at) != sc.get("section"):
                at = None                             # a span never follows into another section's scoring
            if at is None:
                key = origs.get(k) if origs.get(k) is not None else self.sidecar.get(k)
                served = [v] + [r["text"] for r in self.review.get("repairs", ()) if r.get("text")
                                and rv.repair_fits(r, v)]
                now = next((self.sha_label.get(orig_key(x)) for x in served if self.sha_label.get(orig_key(x))), None)
                if now is not None and self.section_of.get(now) == sc.get("section") and now not in picks:
                    absorbed[k] = now
                    if key:
                        held[key] = v
                    continue
                if key in gone_spans:              # you rewrote or removed its paragraph: your text stands there
                    gone[k] = gone_spans[key]
                    held[key] = v
                    continue
            if at is None or at in picks:
                stale.append(k)
                continue
            picks[at], new_origs[at] = v, self.all_origs.get(at)
            if k in flags:
                new_flags[at] = flags[k]
            if at != k:
                moved[k] = at
        if not moved and not stale and not absorbed and not gone:
            return sc
        out = dict(sc, picks=picks, origs=new_origs, span_flagged=new_flags)
        if moved:
            out["moved"] = moved
        if absorbed:
            out["absorbed"] = absorbed
        if gone:
            out["gone"] = gone
        if held:
            out["held"] = held                     # old span sha -> the pick it held there (eligible's lost pins)
        if stale:
            out["stale"] = sorted(stale)
        return out

    def find_lost_pins(self):
        """The pins your edit took away: {old span sha: pinned text} for every repair (its `text`) and hand fix whose
        original span is one the layout lists as gone (a paragraph you rewrote or removed), when some banked scoring
        of the section held exactly that text there, so the pin was served. eligible() still asks for it, or a
        scoring that was out only because it lacked the pin would come back in and the section would jump to it (a
        rehearsal of 2026-11-15: section 6 to an older round with other picks, when the loop's sentence beside a
        repaired pick was edited)."""
        gone = getattr(getattr(self, "bank", None), "gone_spans", None) or {}
        lost = {}
        for kind in ("repairs", "for_you"):
            for r in self.review.get(kind, ()):
                key = orig_key(r.get("orig") or "")
                if key in gone and r.get("text") and any(
                        rv.squash((s.get("held") or {}).get(key)) == rv.squash(r["text"]) for s in self.scores):
                    lost[key] = r["text"]
        self.lost_pins = lost
        return lost

    def eligible(self, sc):
        """Whether a scoring may be a section's best: none of its picks went stale, none is vetoed, and every pinned
        unit of its section (a hand fix, a standing repair) holds its pinned text, so a scoring that still holds a
        repair's `from` is out. A pin your edit took away (find_lost_pins) is asked of the pick the scoring held at
        that old span."""
        if sc.get("stale"):
            return False
        sections = getattr(getattr(self, "bank", None), "gone_sections", None) or {}
        for key, text in self.lost_pins.items():
            if sections.get(key) == sc["section"] and rv.squash((sc.get("held") or {}).get(key)) != rv.squash(text):
                return False
        for label, text in sc["picks"].items():
            if rv.veto_why(self.review, self.orig_text.get(label, ""), text) is not None:
                return False
        for label, text in self.pins.items():
            if self.section_of.get(label) != sc["section"]:
                continue
            if rv.squash(sc["picks"].get(label)) != rv.squash(text):
                return False
        return True

    def consider(self, sc):
        """The scoring becomes its section's best when it's eligible, not humanized, not an arm test's and lower than
        the best so far."""
        n = sc["section"]
        if sc.get("test"):
            return
        if not sc.get("humanized") and self.eligible(sc) and (n not in self.best or sc["ai"] < self.best[n]["ai"]):
            self.best[n] = sc

    def clean_picks(self, n, picks):
        """Section `n`'s picks with the vetoed ones out (that span goes back to its standing text), the stop slip
        fixed and every pinned text of the section in (its hand fixes, and its standing repairs in place of `from`)."""
        out = {}
        for label, text in picks.items():
            if rv.veto_why(self.review, self.orig_text.get(label, ""), text) is None:
                out[label] = quote_stops(text)
        for label, text in self.pins.items():
            if self.section_of.get(label) == n:
                out[label] = text
        return out

    def served(self, n):
        """(the scoring section `n` is served from, its picks with clean_picks() applied), or None when the section
        has no clean scoring: section_pick() without the text, so the report can read it without assembling."""
        base = self.best.get(n)
        if base is None and (self.bases.get(n) is not None
                             or any(self.section_of.get(label) == n for label in self.repairs)):
            base = self.anchor.get(n)            # the review base, or the scoring the repairs were read against
        if base is None:
            clean = [s for s in self.scores if s["section"] == n and not s.get("humanized")]
            if not clean:
                return None
            base = min(clean, key=lambda s: s["ai"])
        return base, self.clean_picks(n, base["picks"])

    def section_pick(self, n):
        """What the pass puts in section `n`: {"picks", "text", "scoring", "scored"}, or None when the section has no
        clean (not humanized) scoring at all. The picks are the best eligible scoring's; with none eligible, the
        review base's (the scoring the reviewer's entries name), or the anchor's when the section has repairs, else
        the lowest clean scoring's; always with clean_picks() applied (vetoes out, repairs and hand fixes in, the stop
        slip fixed). When that changed nothing, the text is the scoring's own; when it did, `scored` says whether
        that exact text was ever scored (a --final scoring, say). Factuality comes before score: no vetoed text is
        ever in it."""
        got = self.served(n)
        if got is None:
            return None
        base, picks = got
        text = self.section_text(n, picks)
        if picks == base["picks"] and base is self.best.get(n):
            if base.get("text_sha"):
                scored = base["text_sha"] == sha(text)
            else:                       # an older scoring carries no text sha: its word count stands in
                scored = base.get("words") in (None, len(text.split()))
            return {"picks": picks, "text": text, "scoring": base, "scored": scored}
        hit = next((s for s in self.scores if s["section"] == n and s.get("text_sha") == sha(text)), None)
        return {"picks": picks, "text": text, "scoring": hit or base, "scored": hit is not None}

    def span_picks(self, sc):
        """A banked scoring with its picks as spans. The first version banked whole-paragraph picks (the standing
        prefix and the span); those lose the prefix here, so the round logic's kept picks and skip sets compare like
        with like."""
        if sc.get("spans"):
            return sc
        by = {u["label"]: u for u in self.units}
        picks = {}
        for label, text in sc.get("picks", {}).items():
            u = by.get(label)
            if u is not None and u["prefix"] and text.startswith(u["prefix"]):
                text = text[len(u["prefix"]):].strip()
            picks[label] = text
        return dict(sc, picks=picks, spans=True)

    def spent_today(self):
        """What the paragraph pass has already put on Pangram today, from the calls table, so a relaunch shares the
        day's ceiling instead of starting a fresh one."""
        if self.conn is None:
            return 0.0
        midnight = time.mktime(time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d"))
        row = self.conn.execute("SELECT COALESCE(SUM(usd), 0) FROM pangram_calls WHERE ts >= ? AND usd > 0"
                                " AND label LIKE 'regrow %'", (midnight,)).fetchone()
        return float(row[0] if row else 0.0)

    def regate(self):
        """Every banked draw judged again by the current gate. A verdict that changed is banked as a fresh entry under
        the same label, arm and k (the bank is append-only; the newest entry wins on reload), with the Open Pangram
        score when it now passes. So a gate fix reaches the draws already paid for in model time, and nothing is
        drawn twice."""
        by_label = {u["label"]: u for u in self.units}
        changed = 0
        for (label, arm), pool in self.pools.items():
            u = by_label.get(label)
            if u is None:
                continue
            rest = self.section_rest(u, {})
            for k, e in list(pool.items()):
                why = self.judge(u, e["text"], rest)
                if why == e.get("why"):
                    continue
                e2 = dict(e, why=why, regated=time.strftime("%Y-%m-%d %H:%M"))
                e2.setdefault("orig_sha", orig_key(u["orig"]))   # it fits this unit, or it wouldn't be in the pool
                e2.pop("op", None)
                if why is None and self.op is not None:
                    e2["op"] = float(self.op.score([e2["text"]])[0]["score"])
                self.rbank.add({f: v for f, v in e2.items() if f != "raw"})    # `raw` lives in memory only
                pool[k] = e2
                changed += 1
        if changed:
            self.log(f"regate: {changed} banked draw(s) changed verdict under the current gate")

    def judge(self, unit, cand, rest):
        """The gate's verdict on one candidate span for the unit, with the reviewer's vetoes first."""
        v = rv.veto_why(self.review, unit["orig"], cand)
        if v is not None:
            return f"vetoed: {v}"
        return gate(cand, orig=unit["orig"], standing=unit["standing"], notes=self.notes[unit["label"]],
                    prefix=unit["prefix"], after=unit["suffix"], known=self.known, tics=self.tics, rest=rest,
                    rates=self.rates, multiplier=self.swear_mult, filler_cap=self.filler_cap,
                    filler_words=self.filler_words, keeps=False)

    # ---- drawing
    def prompt_for(self, unit, arm, n_predict=0):
        notes = self.notes[unit["label"]]
        if arm == "notes":
            return adapter_prompt(notes, unit["prefix"])
        if arm == "olmo":
            return olmo_prompt(self.header, notes, unit["prefix"], unit["suffix"],
                               description=getattr(getattr(self, "job", None), "description", None))
        if arm == "base2":
            return self.base2_prompt(unit, notes, n_predict)
        return fewshot_prompt(self.header, self.examples_block, notes, unit["prefix"])

    def base2_prompt(self, unit, notes, n_predict):
        """The few-shot prompt for the second base model, cut to fit its server's context: when the prompt and the
        answer's tokens don't fit one slot (LlamaServer.ctx), worked examples are dropped from the front until they
        do. Logged once; cached per unit, since only the seed changes between its draws."""
        if unit["label"] in self.base2_prompts:
            return self.base2_prompts[unit["label"]]
        parts = list(self.example_parts) or [self.examples_block]
        ctx = getattr(self.srv, "ctx", None)
        drop = 0
        while True:
            p = fewshot_prompt(self.header, EXAMPLE_SEP.join(parts[drop:]), notes, unit["prefix"])
            if not ctx or drop >= len(parts) or len(self.srv.tokenize(p)) + n_predict <= ctx:
                break
            drop += 1
        if drop and not self.trim_warned:
            self.trim_warned = True
            self.log(f"base2: the few-shot prompt is over {ctx} tokens for {unit['label']}, so the first {drop} worked"
                     f" example(s) of {len(parts)} were dropped (logged once; other units may drop a different number)")
        self.base2_prompts = {**self.base2_prompts, unit["label"]: p}
        return p

    def section_rest(self, unit, picks):
        """section_counts of the unit's section as posted with the current picks in and this unit's span left out."""
        try:
            text = self.section_text(unit["section"], picks)
        except Exception:  # noqa: BLE001
            return None
        counts = section_counts(text, self.filler_words)
        span = section_counts(picks.get(unit["label"], unit["standing"]), self.filler_words)
        return tuple(a - b for a, b in zip(counts, span))

    def draw(self, unit, arm, k, picks):
        """One draw of the unit's span on an arm, normalized and gated, banked as a "draw" entry. Returns the entry."""
        from engine.local_models import ArmServer
        key = (unit["label"], arm)
        have = self.pools.setdefault(key, {})
        if k in have:
            return have[k]
        n_tok = int(len(unit["orig"].split()) * 2.4) + 60
        if arm == "olmo":
            out, _ = self.srv.chat(self.prompt_for(unit, arm), max_tokens=n_tok, temperature=1.0, min_p=0.05,
                                   seed=seed_for(unit, arm, k))
            raw = olmo_answer(out)
        else:          # notes, fewshot and base2: the completion call (a server with no LoRA gets no scale)
            srv = ArmServer(self.srv, arm)
            out, _ = srv.complete(self.prompt_for(unit, arm, n_tok), n_predict=n_tok, temperature=1.0, min_p=0.05,
                                  seed=seed_for(unit, arm, k), stop=["\n"])
            raw = out.split("\n")[0]
        cand = normalize(raw, unit["orig"])
        why = self.judge(unit, cand, self.section_rest(unit, picks))
        e = {"kind": "draw", "label": unit["label"], "arm": arm, "k": k, "text": cand, "why": why,
             "orig_sha": orig_key(unit["orig"]), "time": time.strftime("%Y-%m-%d %H:%M")}
        if why is None and self.op is not None:
            e["op"] = float(self.op.score([cand])[0]["score"])
        self.rbank.add(e)
        have[k] = e
        return e

    def fill(self, unit, arm, n, picks):
        """Up to `n` draws banked for the unit on the arm (the ones already banked count)."""
        have = self.pools.setdefault((unit["label"], arm), {})
        k = 0
        while len(have) < n:
            while k in have:
                k += 1
            self.draw(unit, arm, k, picks)

    def passed(self, unit, arm=None):
        """The unit's passing candidates, best Open Pangram score first, on one arm or both."""
        out = []
        for a in (ARMS if arm is None else (arm,)):
            out += [e for e in self.pools.get((unit["label"], a), {}).values() if e["why"] is None]
        return sorted(out, key=lambda e: (e.get("op", 1.0), e["k"]))

    def tally(self, arm):
        why = {}
        scope = {u["label"] for u in self.units if self.only is None or u["label"] in self.only}
        for (label, a), pool in self.pools.items():
            if a != arm or label not in scope:              # the units in scope, not every section's banked draws
                continue
            for e in pool.values():
                why[e["why"] or "passed"] = why.get(e["why"] or "passed", 0) + 1
        return why

    # ---- the fact check
    def facts_ok(self, unit, e):
        """Sonnet's fact check on one candidate, cached in the bank. True when nothing is wrong, False when something
        is, None when the candidate went unchecked (the check off, no Claude caller, or out of money). A cached
        verdict counts even then, found under the text as it stands or as it was banked before the stop-slip fix.
        The gate (regrow.fact_gate) picks the prompt and the key: "loose" asks with draws.FACT_PROMPT under the
        old unversioned key, so the verdicts banked before 2026-11-14 evening are reused; "strict" asks with
        REGROW_FACT_PROMPT under FACTS_VERSION. The two never read each other's verdicts."""
        strict = self.fact_gate == "strict"
        keyf = fact_key if strict else loose_key
        forms = [e["text"]] + sorted(({e.get("raw")} | self.raw_forms.get(e["text"], set())) - {None, e["text"]})
        for t in forms:
            k_ = keyf(unit["orig"], t)
            if k_ in self.facts:
                return not self.facts[k_]
        key = keyf(unit["orig"], e["text"])
        if not self.fact_check or self.claude is None or self.claude_spent >= self.max_claude_usd:
            self.warn_unchecked()
            return None
        if strict:
            prompt = REGROW_FACT_PROMPT
        else:
            from engine.draws import FACT_PROMPT as prompt
        notes = self.notes[unit["label"]]
        text, meta = self.claude(self.settings, self.conn, prompt.replace("<<ORIGINAL>>", unit["orig"])
                                 .replace("<<NOTES>>", "\n".join(f"- {n}" for n in notes))
                                 .replace("<<NEW>>", e["text"]), role="regrow facts", model="sonnet",
                                 max_budget_usd=0.2)
        self.claude_spent += float((meta or {}).get("cost_usd") or 0.0)
        facts = "" if text.strip().upper().startswith("NONE") else text.strip()
        self.facts[key] = facts
        entry = {"kind": "fact", "key": key, "label": unit["label"], "arm": e.get("arm"), "k": e.get("k"),
                 "facts": facts}
        if strict:
            entry["version"] = FACTS_VERSION
        self.rbank.add(entry)
        return not facts

    def unchecked_why(self):
        if not self.fact_check:
            return "the fact check is off"
        if self.claude is None:
            return "there is no Claude caller"
        return "the fact check ran out of money"

    def warn_unchecked(self):
        """One log line, the first time a candidate goes unchecked."""
        if self.facts_warned:
            return
        self.facts_warned = True
        if self.no_facts:
            self.log("--no-facts: candidates go in on the code gates alone")
        else:
            self.log(f"{self.unchecked_why()} (${self.claude_spent:.2f} of ${self.max_claude_usd:.2f}): a candidate"
                     " with no fact check can't be picked, so those units stay standing (--no-facts lets them in)")

    def fact_passes(self, unit, e):
        """Whether a candidate may go in: Sonnet passed it, or nobody checked it and the run said --no-facts."""
        ok = self.facts_ok(unit, e)
        return ok or (ok is None and self.no_facts)

    # ---- assembly and scoring
    def section_text(self, n, picks):
        best = {}
        for wid, w in self.lay.items():
            if w["section"] != n:
                continue
            origs = [flat(p) for p in w["original"]]
            curs = [self.para_text(wid, pi, picks, self.standing.get(f"{wid} p{pi + 1}", o)) for pi, o in enumerate(origs)]
            best[wid] = (origs, curs)
        md = self.job.assemble(best, write=False, keep_others=True)
        return as_posted(restore_examples(md, self.swaps), [n], style=style_of(self.job))[n]

    def para_text(self, wid, pi, picks, default):
        """The paragraph with its picked spans in and its anchors (your lines, locked sentences) and unpicked spans
        standing; `default` when no span of it is picked."""
        us = [u for u in self.units if u["wid"] == wid and u["pi"] == pi]
        if not any(u["label"] in picks for u in us):
            return default
        runs = list(us[0]["runs"])
        for u in sorted(us, key=lambda u: -u["start"]):
            if u["label"] in picks:
                runs[u["start"]:u["end"]] = [picks[u["label"]]]
        return " ".join(x for x in runs if x)

    def pick_text(self, unit, e):
        """The candidate span's text (picks hold spans; the paragraph is assembled around them)."""
        return e["text"]

    def score_section(self, n, picks, label, arm_set, test=False):
        """One section as posted with `picks` in, scored once. Returns the scoring, or None at the pass's ceiling. A
        `test` scoring (the arm test) is marked so, goes in `tests` instead of `scores` and is never considered."""
        from .pangram import CapReached, summarize
        text = self.section_text(n, picks)
        hit = self.pg.cached(text)
        if hit is None:
            _, usd = self.pg.estimate(text)
            if self.spent + usd > self.max_usd + 1e-9:
                self.log(f"the pass's Pangram ceiling (${self.max_usd:.2f}) would be crossed by {label}; not scored")
                return None
        try:
            resp, cached = self.pg.score(text, self.purpose, f"regrow {label}")
        except CapReached as e:
            self.log(f"Pangram said no to {label}: {e}")
            return None
        if not cached:
            self.spent += self.pg.estimate(text)[1]
        s = summarize(resp)
        segs = [[w.get("text", "").strip(), (w.get("ai_assistance_score") or 0) > 0.5] for w in resp.get("windows", [])]
        fl = flagged_segments(segs)
        flagged = {}
        for u in self.units:
            if u["section"] != n or u["label"] not in picks:
                continue
            sents = split_sentences(picks[u["label"]])
            flagged[u["label"]] = any(norm(x)[:40] in f for x in sents for f in fl)
        mine = {k: v for k, v in picks.items() if k.startswith(f"s{n}w")}
        sc = {"kind": "score", "section": n, "label": label, "arm_set": arm_set, "ai": s["fraction_ai"],
              "humanized": bool(s["humanized"]), "max_humanizer": s.get("max_humanizer_score"),
              "flagged": sum(1 for _, f in segs if f), "segs": segs, "words": len(text.split()),
              "picks": mine, "origs": {k: self.all_origs.get(k) for k in mine}, "text_sha": sha(text),
              "span_flagged": flagged, "spans": True, "cached": cached, "time": time.strftime("%Y-%m-%d %H:%M")}
        if test:
            sc["test"] = True
        self.rbank.add(sc)
        if test:
            self.tests = self.tests + [sc]
        else:
            self.scores.append(sc)
            self.consider(sc)
        self.log(f"{label}: section {n} {sc['ai']:.2f} ({sc['flagged']} flagged"
                 f"{', HUMANIZED' if sc['humanized'] else ''}{', cached' if cached else ''}); Pangram ${self.spent:.2f}")
        return sc

    def standing_score(self, n):
        """The section's standing as-posted score from the Pangram cache (the loop scored it), or None."""
        from .pangram import summarize
        hit = self.pg.cached(self.section_text(n, {}))
        return None if hit is None else summarize(hit)["fraction_ai"]

    # ---- the rounds
    def choose(self, unit, arm=None, skip=()):
        """The best-ranked passing candidate on the arm (or either) that passes the fact check and isn't in `skip`
        (compared after quote_stops). An unchecked candidate is passed over unless the run said --no-facts; when that
        leaves the unit with nothing, `no_checked` says why. Whenever an unchecked candidate was passed over (so the
        missing check shaped the pick, or left none), `waiting` holds the unit and the assembly isn't scored."""
        unchecked = False
        for e in self.passed(unit, arm):
            if quote_stops(e["text"]) in skip:
                continue
            if rv.veto_why(self.review, unit["orig"], e["text"]) is not None:   # vetoed after it was banked
                continue
            ok = self.facts_ok(unit, e)
            if ok or (ok is None and self.no_facts):
                self.no_checked.pop(unit["label"], None)
                if unchecked:
                    self.waiting = {**self.waiting, unit["label"]: self.unchecked_why()}
                return e
            unchecked = unchecked or ok is None
        if unchecked:
            self.no_checked[unit["label"]] = self.unchecked_why()
            self.waiting = {**self.waiting, unit["label"]: self.unchecked_why()}
        return None

    def keepable(self, unit, text):
        """Whether an earlier scoring's pick may stay in the next round: not vetoed, and through the current fact
        check (a pick checked under the other gate is checked again). A pick that can't stay only because nobody
        could check it goes in `waiting`, so the round isn't scored without it."""
        if rv.veto_why(self.review, unit["orig"], text) is not None:
            return False
        ok = self.facts_ok(unit, {"text": quote_stops(text), "raw": text})
        if ok is None and not self.no_facts:
            self.waiting = {**self.waiting, unit["label"]: self.unchecked_why()}
        return bool(ok) or (ok is None and self.no_facts)

    def dry_stop(self, n, sections=()):
        """True when section `n`'s assembly lost a unit to a missing fact check (the check off or out of money, and
        no --no-facts), so it must not be scored: the run then scores nothing more. Records every section in
        `sections` that has units waiting, logs one line, and sets `dry_why`."""
        counts = {}
        for label in self.waiting:
            m = self.section_of.get(label)
            counts[m] = counts.get(m, 0) + 1
        if not counts.get(n):
            return False
        for m in [n] + [m for m in sections if m != n]:
            if counts.get(m):
                self.unscored = {**self.unscored, m: counts[m]}
        self.dry_why = self.unchecked_why()
        self.log(f"{self.dry_why}: " + ", ".join(f"section {m} has {c} unit(s) waiting on a fact check"
                                                for m, c in sorted(self.unscored.items()))
                 + "; an assembly without them isn't scored, so this run scores nothing more")
        return True

    def final(self, n):
        """--final: section `n` as section_pick() serves it (best eligible picks, vetoes out, hand fixes in), scored
        once as posted when that exact text isn't a banked scoring yet. Returns the scoring, or None (nothing to
        serve, or the ceiling)."""
        sp = self.section_pick(n)
        if sp is None:
            self.log(f"section {n}: no clean scoring to build a final text from")
            return None
        if sp["scored"]:
            self.log(f"section {n}: its text is already scored ({sp['scoring']['label']}, {sp['scoring']['ai']:.2f})")
            return sp["scoring"]
        sc = self.score_section(n, sp["picks"], f"s{n} final", "final")
        if sc is not None and sc["humanized"]:
            self.log(f"s{n} final came back humanized: it is served as scored, but it isn't a best")
        return sc

    def census(self, sections):
        """Draw and gate every unit on every arm, nothing scored. Returns the reject tallies per arm. A pinned unit (a
        hand fix, a standing repair) is never drawn for, and under --only-stuck only the stuck units are. A unit that
        passes nothing on this run's arms gets more draws (another arm's old passes don't count)."""
        for u in self.units:
            if u["label"] in self.pins or (self.only is not None and u["label"] not in self.only):
                continue
            for arm in self.arms:
                self.fill(u, arm, self.draws, {})
            if not any(self.passed(u, a) for a in self.arms):
                for arm in self.arms:
                    self.fill(u, arm, self.draws + self.more, {})
        return {arm: self.tally(arm) for arm in self.arms}

    def assembly(self, arm, skip=None):
        """{label: span text} with every unit's best candidate on the arm in, and every pinned text (hand fixes,
        standing repairs); a unit with none stays standing. `waiting` starts over for it."""
        picks, used = {}, {}
        self.waiting = {}
        for u in self.units:
            pin = self.pins.get(u["label"])
            if pin is not None:
                picks[u["label"]] = pin
                continue
            e = self.choose(u, arm, (skip or {}).get(u["label"], ()))
            if e is not None:
                picks[u["label"]] = self.pick_text(u, e)
                used[u["label"]] = e
        return picks, used

    def run(self, sections, census_only=False, only_stuck=False):
        """The census, then the rounds. With `only_stuck` the census draws for the stuck units alone (stuck_units);
        on the base2 arm the run then ends on one base2 assembly per section (base2_step) instead of the rounds."""
        if only_stuck:
            self.stuck = {n: self.stuck_units(n) for n in sections}
            self.only = {label for labels in self.stuck.values() for label in labels}
            self.log("--only-stuck: " + ", ".join(f"section {n} {len(v)}" for n, v in self.stuck.items())
                     + " stuck unit(s)")
        tallies = self.census(sections)
        for arm in self.arms:
            t = tallies[arm]
            self.log(f"census, {arm}: " + ", ".join(f"{k} {n}" for k, n in sorted(t.items(), key=lambda kv: -kv[1])))
        n_pass = {arm: sum(1 for u in self.units if self.passed(u, arm)) for arm in self.arms}
        self.log(f"units with a passing candidate: " + ", ".join(f"{a} {n_pass[a]} of {len(self.units)}" for a in self.arms))
        if census_only:
            return tallies
        if only_stuck and self.arms == ("base2",):
            for n in sections:
                if self.base2_step(n) is False:
                    break
            return tallies
        first = sorted(sections, key=lambda n: len(self.section_text(n, {}).split()))[0]
        # round 1: each arm set on the cheapest section first, the humanized flag read before the rest is paid for
        used = {}
        done = {sc["label"] for sc in self.scores}    # a relaunch's banked scorings
        for arm in self.arms:
            if f"s{first} all-{arm}" in done:
                continue
            picks, used[arm] = self.assembly(arm)
            if not any(k.startswith(f"s{first}w") for k in picks):
                continue
            if self.dry_stop(first, sections):
                return tallies
            self.score_section(first, picks, f"s{first} all-{arm}", arm)
        firsts = [s for s in self.scores if s["section"] == first]
        if firsts and all(s["humanized"] for s in firsts):
            self.log(f"section {first} came back humanized on every arm set; the rest isn't scored")
            return tallies
        for arm in self.arms:
            picks, _ = self.assembly(arm)
            for n in sections:
                if n == first or f"s{n} all-{arm}" in done:
                    continue
                if self.dry_stop(n, sections):
                    return tallies
                if self.score_section(n, picks, f"s{n} all-{arm}", arm) is None:
                    return tallies
        # later rounds: per unit, keep the pick Pangram passed in the better arm set, re-pick the flagged ones
        skip = {}
        prev = [int(sc["label"].rsplit(" ", 1)[-1]) for sc in self.scores if " round " in sc["label"]]
        base = max(prev, default=1)                     # a relaunch's rounds carry on the numbering
        for r in range(base + 1, base + self.rounds):
            picks, changed = {}, False
            self.waiting = {}
            for n in sections:
                sec_scores = [s for s in self.scores if s["section"] == n]
                if not sec_scores:
                    continue
                # a humanized scoring's unflagged picks stay in play, after the clean scorings' (the author, 2026-11-14:
                # under the residue theory the flag is the mix's, not the picks')
                order = sorted(sec_scores, key=lambda s: (s["humanized"], s["ai"]))
                for u in self.units:
                    if u["section"] != n:
                        continue
                    pin = self.pins.get(u["label"])
                    if pin is not None:                 # a hand fix or a standing repair: never drawn for or re-picked
                        picks[u["label"]] = pin
                        if not any(rv.squash(s["picks"].get(u["label"])) == rv.squash(pin) for s in sec_scores):
                            changed = True
                        continue
                    # a kept pick is never vetoed and passes the fact check as it stands now
                    kept = next((s["picks"][u["label"]] for s in order
                                 if u["label"] in s["picks"] and not s["span_flagged"].get(u["label"])
                                 and self.keepable(u, s["picks"][u["label"]])), None)
                    if kept is not None:
                        picks[u["label"]] = quote_stops(kept)
                        changed = changed or picks[u["label"]] != kept
                        continue
                    tried = {quote_stops(s["picks"][u["label"]]) for s in self.scores if u["label"] in s["picks"]}
                    skip.setdefault(u["label"], set()).update(tried)
                    e = self.choose(u, None, skip.get(u["label"], ()))
                    if e is None and u["label"] not in self.no_checked:     # more draws can't buy a fact check
                        for arm in self.arms:
                            have = len(self.pools.get((u["label"], arm), {}))
                            self.fill(u, arm, have + self.more, picks)
                        e = self.choose(u, None, skip.get(u["label"], ()))
                    if e is not None and quote_stops(self.pick_text(u, e)) not in tried:
                        picks[u["label"]] = self.pick_text(u, e)
                        changed = True
            if not changed:
                self.log(f"round {r}: nothing new to try")
                break
            for n in sections:
                sec_picks = {k: v for k, v in picks.items() if k.startswith(f"s{n}w")}
                if not sec_picks:
                    continue
                prev = [s for s in self.scores if s["section"] == n and s["picks"] == sec_picks]
                if prev:
                    continue
                if self.dry_stop(n, sections):
                    return tallies
                if self.score_section(n, sec_picks, f"s{n} round {r}", "mixed") is None:
                    return tallies
        return tallies

    # ---- the stuck units and the second base model
    def stuck_bold(self, u, entry=None):
        """The sentences of unit `u`'s standing span that are, word for word, a sentence of its original span
        (whitespace squashed, bold marks out), in order and squashed, and only those the paragraph as the working
        copy holds it still has (a unit whose paragraph doesn't split into its runs stands on the original runs, so
        its standing span alone can't say what the document holds). With the stuck `entry` the unit matched, only the
        sentences of the entry's original too: a span that grew past its entry at a sync holds the loop's lines,
        which are no sentence of Claude's. Split with the pass's own sentence splitter."""
        def key(t):
            return rv.squash(t.replace("**", ""))
        orig = {key(x) for x in split_sentences(u["orig"])}
        if entry is not None:
            orig &= {key(x) for x in split_sentences(entry.get("orig") or "")}
        para = {key(x) for x in split_sentences(u.get("para_standing") or u["standing"])}
        out = []
        for x in split_sentences(u["standing"]):
            k = key(x)
            if k and k in orig and k in para and k not in out:
                out.append(k)
        return out

    def stuck_state(self, n, picks):
        """What the write-back does with each of section `n`'s units that has a stuck entry, given the served
        `picks`: {"bold": {label: [sentences]}, "moot": [...], "picked": [...], "hand_fix": [...]}. A unit with a hand
        fix goes to the hand fix, one with a pick in `picks` to the pick, one whose standing span holds no sentence of
        its original word for word is moot (the loop already rewrote it), and the rest go in bold."""
        out = {"bold": {}, "moot": [], "picked": [], "hand_fix": []}
        for u in self.units:
            if u["section"] != n or u["label"] not in self.stuck_entries:
                continue
            if u["label"] in self.fixes:
                out["hand_fix"].append(u["label"])
            elif u["label"] in picks:
                out["picked"].append(u["label"])
            else:
                bold = self.stuck_bold(u, self.stuck_entries[u["label"]])
                if bold:
                    out["bold"][u["label"]] = bold
                else:
                    out["moot"].append(u["label"])
        return out

    def stuck_units(self, n):
        """Section `n`'s stuck units, as labels: not pinned, and with no pick in what the pass serves
        (section_pick(n)["picks"]), so no Mistral candidate ever got through the gate and the fact check, or every
        one that did was flagged out or vetoed. With nothing served yet, every unpinned unit is stuck."""
        sp = self.section_pick(n)
        served = sp["picks"] if sp is not None else {}
        return [u["label"] for u in self.units
                if u["section"] == n and u["label"] not in self.pins and u["label"] not in served]

    def scored_once(self, n, picks, label, arm_set, test=False):
        """score_section, unless a banked scoring with this label already holds exactly this text: that one comes
        back and nothing is called (a relaunch of the arm test or the base2 step pays nothing twice)."""
        text_sha = sha(self.section_text(n, picks))
        for s in (self.tests if test else self.scores):
            if s["section"] == n and s["label"] == label and s.get("text_sha") == text_sha:
                self.log(f"{label}: already scored ({s['ai']:.2f}), not scored again")
                return s
        return self.score_section(n, picks, label, arm_set, test=test)

    def base2_step(self, n):
        """--arms base2 --only-stuck: section `n` as the pass serves it (its picks, pins in) with the best base2
        candidate in each stuck unit that has one, scored once as posted. Real, complete text, so it can be the
        section's best. Returns the scoring, None when there was nothing to score, False at the ceiling or when a
        stuck unit waits on a fact check the run can't pay for (then nothing more is scored)."""
        stuck = self.stuck.get(n)
        if stuck is None:
            stuck = self.stuck_units(n)
        before = self.best.get(n)
        sp = self.section_pick(n)
        picks = dict(sp["picks"]) if sp is not None else self.clean_picks(n, {})
        by = {u["label"]: u for u in self.units}
        got = {}
        self.waiting = {}
        for label in stuck:
            e = self.choose(by[label], "base2")
            if e is not None:
                got[label] = self.pick_text(by[label], e)
        row = {"stuck": list(stuck), "got": sorted(got), "before": before, "scoring": None}
        self.base2_runs = {**self.base2_runs, n: row}
        if self.dry_stop(n, [n]):
            return False
        if not got:
            self.log(f"section {n}: none of its {len(stuck)} stuck unit(s) has a base2 candidate through the gate and"
                     " the fact check; nothing scored")
            return None
        sc = self.scored_once(n, dict(picks, **got), f"s{n} base2", "base2")
        row["scoring"] = sc
        return False if sc is None else sc

    # ---- the arm test
    def arm_test_units(self, n):
        """Section `n`'s test units: not pinned, and both Mistral arms have a candidate that passed the gate, isn't
        vetoed and passes the fact check. Returns {label: {"notes": entry, "fewshot": entry}}."""
        out = {}
        for u in self.units:
            if u["section"] != n or u["label"] in self.pins:
                continue
            a = self.choose(u, "notes")
            b = self.choose(u, "fewshot") if a is not None else None      # no fact check spent when notes has none
            if a is not None and b is not None:
                out[u["label"]] = {"notes": a, "fewshot": b}
        return out

    def arm_test(self, sections, min_units=4):
        """--arm-test: per section, one assembly with each test unit's best "notes" candidate and one with its best
        "fewshot" candidate, every other unit standing (no other pick, no pin), so the two differ only by arm. Both
        are scored as posted, or neither when the ceiling can't cover both. Nothing is drawn. The scorings are
        marked `test` and are never a best or served. When a unit of a section waits on a fact check the run can't
        pay for, that section and the rest aren't scored."""
        for i, n in enumerate(sections):
            self.waiting = {}
            pairs = self.arm_test_units(n)
            row = {"units": sorted(pairs), "notes": None, "fewshot": None, "why": None}
            self.arm_tests = {**self.arm_tests, n: row}
            if self.dry_stop(n, [n]):
                row["why"] = f"{self.dry_why}, {self.unscored[n]} unit(s) waiting on a check"
                for m in sections[i + 1:]:
                    gone = f"not reached, the run stopped scoring at section {n}"
                    self.arm_tests = {**self.arm_tests, m: {"units": [], "notes": None, "fewshot": None, "why": gone}}
                break
            if len(pairs) < min_units:
                row["why"] = f"{len(pairs)} test unit(s), under the minimum of {min_units}"
                self.log(f"arm test, section {n}: skipped, {row['why']}")
                continue
            sets = {arm: {label: self.pick_text(None, p[arm]) for label, p in pairs.items()} for arm in MISTRAL_ARMS}
            cost, seen = 0.0, set()
            for arm, picks in sets.items():
                text = self.section_text(n, picks)
                label = f"s{n} armtest {arm}"
                if text in seen or self.pg.cached(text) is not None or any(
                        s["label"] == label and s.get("text_sha") == sha(text) for s in self.tests):
                    continue
                seen.add(text)
                cost += self.pg.estimate(text)[1]
            if self.spent + cost > self.max_usd + 1e-9:
                row["why"] = (f"the two scorings would cost ${cost:.2f} and the ceiling has"
                              f" ${max(0.0, self.max_usd - self.spent):.2f} left, so neither was scored")
                self.log(f"arm test, section {n}: {row['why']}")
                continue
            for arm, picks in sets.items():
                row[arm] = self.scored_once(n, picks, f"s{n} armtest {arm}", "armtest", test=True)
            if row["notes"] is None or row["fewshot"] is None:
                row["why"] = "Pangram refused one of the two scorings, so the pair is incomplete"
                self.log(f"arm test, section {n}: {row['why']}")
            self.log(f"arm test, section {n}: {len(pairs)} unit(s); " + "; ".join(
                f"{arm} {row[arm]['ai']:.2f}" + (" humanized" if row[arm]["humanized"] else "")
                for arm in MISTRAL_ARMS if row[arm] is not None))

    # ---- the repaired test scoring
    def repaired_picks(self, sc):
        """--test-scoring: banked scoring `sc`'s picks with the vetoed ones out, the stop slip fixed, every repair in
        the reviewer's file applied whose `orig` is the unit's and whose `from` fits this scoring's pick (whatever the
        section's review base or anchor is, so not clean_picks(), which reads the repairs against the anchor), and
        the section's hand fixes in last, as every assembly has them. Returns (picks, {label: {"applied": [...],
        "unfit": [...], "vetoed": bool, "hand_fix": bool}}) for the units that had a review entry; `applied` and
        `unfit` hold the repairs' `moves` (or their labels)."""
        n, picks, log = sc["section"], {}, {}
        for label, text in sc["picks"].items():
            orig = self.orig_text.get(label, "")
            if rv.veto_why(self.review, orig, text) is not None:
                log[label] = {"applied": [], "unfit": [], "vetoed": True, "hand_fix": False}
                continue
            picks[label] = quote_stops(text)
        for label in sorted({x for x in self.orig_text if self.section_of.get(x) == n}):
            mine = [r for r in self.review.get("repairs", ()) if r.get("text")
                    and rv.squash(r.get("orig")) == rv.squash(self.orig_text[label])]
            if not mine:
                continue
            row = log.setdefault(label, {"applied": [], "unfit": [], "vetoed": False, "hand_fix": False})
            hit = rv.repair(self.review, self.orig_text[label], picks[label]) if label in picks else None
            for r in mine:
                (row["applied"] if r is hit else row["unfit"]).append(r.get("moves") or r.get("label") or "a repair")
            if hit is not None:
                picks[label] = hit["text"]
        for label, fx in self.fixes.items():
            if self.section_of.get(label) == n:
                picks[label] = fx["text"]
                log.setdefault(label, {"applied": [], "unfit": [], "vetoed": False, "hand_fix": False})["hand_fix"] = True
        return picks, log

    def test_scoring(self, label):
        """--test-scoring: the banked scoring called `label`, rebuilt by repaired_picks() and scored once as posted
        as a test scoring, "<label> repaired test": marked `test`, never a best, never served, and through
        scored_once() so a relaunch pays nothing twice. Draws nothing. Returns the scoring, None at the ceiling (or
        when Pangram said no), and raises ValueError when no banked scoring has that label."""
        sc = next((x for x in reversed(self.scores) if x["label"] == label), None)
        if sc is None:
            n = test_scoring_section(label)          # name that section's scorings, or every one when it has none
            have = sorted({x["label"] for x in self.scores if x["section"] == n}) or sorted(
                {x["label"] for x in self.scores})
            raise ValueError(f"no banked scoring is called {label!r}" + (
                (f"; the bank has {', '.join(have[-8:])}" + (" and more" if len(have) > 8 else "")) if have else
                "; the bank has no scoring at all"))
        n = sc["section"]
        picks, log = self.repaired_picks(sc)
        for unit, row in sorted(log.items()):
            bits = []
            if row["vetoed"]:
                bits.append("its pick is vetoed, so it stands")
            if row["applied"]:
                bits.append("applied " + "; ".join(row["applied"]))
            if row["unfit"]:
                bits.append("no fitting `from` for " + "; ".join(row["unfit"]))
            if row["hand_fix"]:
                bits.append("carries its hand fix")
            self.log(f"{label} repaired test, {unit}: " + ", ".join(bits))
        test = self.scored_once(n, picks, f"{label} repaired test", "repaired test", test=True)
        self.test_runs = {**self.test_runs, label: {"section": n, "from": sc, "log": log, "scoring": test}}
        return test

    # ---- outputs
    def best_picks(self):
        """{label: text} from each section's best scoring, for regrow picks.json."""
        out = {}
        by = {u["label"]: u for u in self.units}
        for n, s in self.best.items():
            for label, text in s["picks"].items():
                u = by.get(label)
                para = self.para_text(u["wid"], u["pi"], s["picks"], u["para_standing"]) if u else text
                out[label] = {"text": text, "section": n, "ai": s["ai"], "label": s["label"],
                              "flagged": s["span_flagged"].get(label), "paragraph": para}
        return out

    def write_picks(self):
        path = self.job.dir / "regrow picks.json"
        path.write_text(json.dumps(self.best_picks(), indent=1, ensure_ascii=False), encoding="utf-8")
        return path


def pass_paragraphs(job, bank, sections=None, vm=None, readonly=False):
    """What the pass's best scoring of each section puts in the document, read from its bank and never written to it
    (the write-back, engine/writeback.py). Returns {"paragraphs": {"s6w2 p5": {"text", "section", "round", "ai",
    "flagged", "for_you", "repaired", "picked", "stuck"}}, "sections": {6: {"ai", "label", "text", "scored"}},
    "moot": [labels], "unmatched": [labels], "unmatched_review": [[kind, label]]}: every paragraph with a picked span, assembled around its anchors, `flagged` when Pangram flagged
    one of its picked spans in that scoring, `for_you` None or the reviewer's note {"went_wrong", "about", "point",
    "fact"} when one of its spans is a hand fix, `repaired` None or the `moves` of its repaired spans joined with "; "
    (normal text, no note); and each section as posted. Factuality comes before score (section_pick): no vetoed text
    is ever returned, and when the section has no eligible scoring the text is built anyway and `scored` is False
    unless that exact text was scored.

    `stuck` (2026-11-15) is a list in unit order of {"bold": [sentences], "why", "about", "point", "fact"}, one per
    unit of the paragraph with a stuck entry in the reviewer's file, no hand fix and no pick in what the pass serves:
    the sentences of its standing span that are Claude's original word for word, for the write-back to put in bold
    under a prompt for the author. A paragraph with a stuck unit and no pick at all still gets a row, its text the
    paragraph as served, with `picked` False (every other row has it True), `round` and `ai` None: the write-back
    then keeps the source it would name without the pass. A stuck unit whose standing span holds none of its
    original's sentences (the loop already rewrote it) is moot: listed under "moot", nothing in bold. An entry may
    also speak for a part of its original (review.stuck_match), and one that speaks for no unit is listed under
    "unmatched" by its label, for the morning digest, and so is a veto, repair or hand fix whose original no unit
    holds now, under "unmatched_review". A row with a
    hand fix carries no `stuck` (the hand fix wins, whole in bold). The first call writes the sidecar
    `regrow origs.json` when it's missing; the bank is never written.

    `readonly` (for a reader such as the MCP server's list_jobs): nothing is written at all. With no sidecar, the
    bank's entries that carry no orig sha of their own are left out, and their count is under "left_out" (0
    otherwise; the key is only there on a read-only read); the rest reads as it would."""
    sections = list(sections) if sections is not None else list(job.sections)
    rp = RegrowPass(job=job, bank=bank, settings=None, conn=None, log=lambda *_: None)
    rp.readonly = bool(readonly)
    rp.load_units(sections, vm)
    rp.load_bank()
    out = {"paragraphs": {}, "sections": {}, "moot": [], "unmatched": list(rp.stuck_unmatched),
           "unmatched_review": [list(x) for x in rp.review_unmatched]}
    if readonly:
        out["left_out"] = rp.no_sidecar
    for n in sorted(sections):
        sp = rp.section_pick(n)
        if sp is None:
            continue
        picks, s = sp["picks"], sp["scoring"]
        out["sections"][n] = {"ai": s["ai"], "label": s["label"], "text": sp["text"], "scored": sp["scored"]}
        for u in rp.units:
            if u["section"] != n or u["label"] not in picks:
                continue
            row = out["paragraphs"].setdefault(u["plabel"], {
                "text": rp.para_text(u["wid"], u["pi"], picks, u["para_standing"]), "section": n, "round": s["label"],
                "ai": s["ai"], "flagged": False, "for_you": None, "repaired": None, "picked": True, "stuck": []})
            if s["picks"].get(u["label"]) == picks[u["label"]] and (s.get("span_flagged") or {}).get(u["label"]):
                row["flagged"] = True
            fx = rp.fixes.get(u["label"])
            if fx is not None and row["for_you"] is None:
                row["for_you"] = rv.note(fx)
            r = rp.repairs.get(u["label"])
            if r is not None and fx is None:
                moves = r.get("moves") or "repaired"
                row["repaired"] = moves if row["repaired"] is None else row["repaired"] + "; " + moves
        state = rp.stuck_state(n, picks)
        out["moot"] += state["moot"]
        for u in rp.units:
            bold = state["bold"].get(u["label"])
            if u["section"] != n or bold is None:
                continue
            row = out["paragraphs"].setdefault(u["plabel"], {
                "text": rp.para_text(u["wid"], u["pi"], picks, u["para_standing"]), "section": n, "round": None,
                "ai": None, "flagged": False, "for_you": None, "repaired": None, "picked": False, "stuck": []})
            if row["for_you"] is None:
                row["stuck"].append(dict(rv.stuck_note(rp.stuck_entries[u["label"]]), bold=bold))
    return out


def audit(text):
    """The Delta line for a text against your samples (engine/tics/voice_audit.py, in process), or "" with no
    SAMPLES_FILE set or too few samples to judge by."""
    from .tics.voice_audit import audit as voice_delta
    try:
        return voice_delta(text)
    except Exception:  # noqa: BLE001
        return ""


def review_lines(rp):
    """The report's lines on the reviewer's file: what it holds, what each stuck entry comes to (in bold for the
    author, moot, picked or under a hand fix), and any entry whose original matches no unit now."""
    vetoes, fixes = rp.review.get("vetoes", ()), rp.review.get("for_you", ())
    repairs, stuck = rp.review.get("repairs", ()), rp.review.get("stuck") or ()
    if not vetoes and not fixes and not repairs and not stuck:
        return ["No review file entries (no vetoes, no repairs, no hand fixes, no stuck units)."]
    have = {rv.squash(o) for o in rp.orig_text.values()}
    L = [f"{len(vetoes)} veto(es), {len(repairs)} repair(s), {len(fixes)} hand fix(es) and {len(stuck)} stuck"
         f" unit(s) in {rv.FILE}; {len(rp.repairs)} unit(s) carry a repair and {len(rp.fixes)} a hand fix."]
    for n, b in sorted((rp.bases or {}).items()):
        if b is not None:
            L.append(f"- section {n}: the review base is {b['label']} ({b['ai']:.2f}), the newest scoring an entry"
                     " names; the repairs are read against it")
    for label in rp.stale_repairs:
        L.append(f"- the repair for {label} is stale (the unit's pick is no longer its `from`), so it isn't applied")
    for n in sorted({u["section"] for u in rp.units}):
        got = rp.served(n)
        for label, para in sorted(((got or (None,))[0] or {}).get("gone", {}).items()):
            L.append(f"- section {n}: {got[0]['label']} had a pick for {label}, whose paragraph you rewrote or removed"
                     f" (\"{' '.join(para.split()[:8])} ...\"); it's dropped and your text stands there")
    if stuck:
        for n in sorted({u["section"] for u in rp.units}):
            got = rp.served(n)
            st = rp.stuck_state(n, got[1] if got is not None else {})
            for label, bold in st["bold"].items():
                L.append(f"- stuck, {label}: {len(bold)} sentence(s) of the original in bold for you to rewrite"
                         + ("" if got is not None else " (once the section is served)"))
            for label in rp.stuck_parts:
                if rp.section_of.get(label) == n:
                    L.append(f"- stuck, {label}: the entry for {rp.stuck_entries[label].get('label') or '?'} speaks for"
                             " it by overlap (a sentence was rewritten by hand, or the loop's line joined the span),"
                             " under its old prompt; only the sentences both originals share go in bold")
            for label in st["moot"]:
                L.append(f"- stuck, {label}: moot, the loop already rewrote every sentence of its original, so nothing"
                         " goes in bold")
            for label in st["picked"]:
                L.append(f"- stuck, {label}: the pass serves a pick for it now, so nothing goes in bold")
            for label in st["hand_fix"]:
                L.append(f"- stuck, {label}: it has a hand fix, which goes in bold whole instead")
    for kind, rows in (("veto", vetoes), ("repair", repairs), ("hand fix", fixes)):
        for r in rows:
            if rv.squash(r.get("orig")) not in have:
                L.append(f"- the {kind} for {r.get('label') or '?'} matches no unit's original now")
    for label in rp.stuck_unmatched:              # a stuck entry counts as matched when it speaks for a part, too
        L.append(f"- the stuck entry for {label} matches no unit now: write a new one or remove it")
    return L


ARM_NAMES = {"notes": "the adapter arm", "fewshot": "the few-shot arm"}


def median_op(entries):
    """The median Open Pangram score of some draws and how many had one, as (median or None, count)."""
    ops = sorted(e["op"] for e in entries if e.get("op") is not None)
    return (ops[len(ops) // 2] if ops else None), len(ops)


def compare(what, a, b, noise):
    """One clause of the verdict sentence: which arm scored lower on `what`, or that the two are within the noise."""
    if a is None or b is None:
        return f"on {what} there is nothing to compare (Pangram gave no score for one arm)"
    if abs(a - b) <= noise:
        return f"on {what} the two are within the noise ({a:.2f} and {b:.2f})"
    lo, hi = (("notes", a), ("fewshot", b)) if a < b else (("fewshot", b), ("notes", a))
    return f"on {what} {ARM_NAMES[lo[0]]} scored lower ({lo[1]:.2f} against {hi[1]:.2f})"


def arm_test_lines(rp):
    """The report's "Arm test" part: per section and arm the test units, the words of Mistral text in the assembly,
    the AI share, the humanized verdict, the highest humanizer score and the test spans Pangram flagged, then which arm
    scored lower."""
    noise = getattr(rp, "noise", 0.05)
    L = ["Each section's test units carry their best candidate from one arm, and every other unit stands as the working"
         " copy holds it (no other pick, no pin), so the two assemblies differ only by arm.", ""]
    for n, row in sorted(rp.arm_tests.items()):
        if row["notes"] is None and row["fewshot"] is None:
            L.append(f"- section {n}: not scored, {row['why']}.")
            continue
        L += [f"**Section {n}**" + (f" ({row['why']})" if row["why"] else ""), "",
              "| Arm | Test units | Mistral words | AI share | Humanized | Highest humanizer score"
              " | Test spans flagged |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for arm in MISTRAL_ARMS:
            sc = row[arm]
            if sc is None:
                L.append(f"| {arm} | {len(row['units'])} | | not scored | | | |")
                continue
            words = sum(len(sc["picks"][x].split()) for x in row["units"] if x in sc["picks"])
            flagged = sum(1 for x in row["units"] if (sc.get("span_flagged") or {}).get(x))
            hum = sc.get("max_humanizer")
            L.append(f"| {arm} | {len(row['units'])} | {words} | {sc['ai']:.2f} |"
                     f" {'yes' if sc['humanized'] else 'no'} | {'' if hum is None else f'{hum:.2f}'} |"
                     f" {flagged} of {len(row['units'])} |")
        a, b = row["notes"], row["fewshot"]
        if a is not None and b is not None:
            s = (compare("AI share", a["ai"], b["ai"], noise) + "; "
                 + compare("the humanizer score", a.get("max_humanizer"), b.get("max_humanizer"), noise))
            L += ["", s[0].upper() + s[1:] + f" (the noise is {noise:.2f})."]
        L.append("")
    return L


def base2_lines(rp):
    """The report's "Second base model" part: per section the stuck units, how many got a base2 candidate, the Open
    Pangram medians of the passing draws on base2 and on the two Mistral arms, and the section's score with the base2
    spans in against its best before."""
    L = []
    for n, row in sorted(rp.base2_runs.items()):
        sec = [u for u in rp.units if u["section"] == n]
        stuck = [u for u in sec if u["label"] in row["stuck"]]
        meds = []
        for arm in ("base2",) + MISTRAL_ARMS:
            m, k = median_op([e for u in sec for e in rp.passed(u, arm)])
            ms, ks = median_op([e for u in stuck for e in rp.passed(u, arm)])
            meds.append(f"{arm} {'none' if m is None else f'{m:.2f}'} ({k} draws"
                        + (f"; {ms:.2f} on the stuck units, {ks} draws" if ms is not None and arm != "base2" else "")
                        + ")")
        line = (f"- section {n}: {len(row['stuck'])} unit(s) stuck, {len(row['got'])} got a base2 candidate through the"
                f" gate and the fact check. Open Pangram median of the passing draws on the section: " + ", ".join(meds)
                + ".")
        sc, before = row["scoring"], row["before"]
        if sc is None:
            line += " Not scored" + (" (no base2 candidate)." if not row["got"] else " (the ceiling).")
        else:
            hum = sc.get("max_humanizer")
            flagged = sum(1 for x in row["got"] if (sc.get("span_flagged") or {}).get(x))
            line += (f" With the base2 spans in the section scored {sc['ai']:.2f} against its best before of "
                     + ("none" if before is None else f"{before['ai']:.2f} ({before['label']})")
                     + f"; {'humanized' if sc['humanized'] else 'not humanized'}, highest humanizer score "
                     + ("none" if hum is None else f"{hum:.2f}") + f"; {flagged} of {len(row['got'])} base2 span(s)"
                     " flagged.")
        L.append(line)
    return L


def test_scoring_lines(rp):
    """The report's "Test scorings" part, beside the arm test's: per --test-scoring run the scoring it started from,
    the repairs it applied and the ones with no fitting `from`, and the test scoring's AI share against the
    scoring's own."""
    L = ["Each row is a banked scoring rebuilt with the vetoed picks out, the stop slip fixed, the hand fixes in and"
         " every repair whose `from` fits its pick applied, scored once as a test: never a best, never served.", "",
         "| Test scoring | Section | From | Repairs applied | No fitting `from` | AI share | Humanized"
         " | Highest humanizer score |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, row in rp.test_runs.items():
        applied = sum(len(x["applied"]) for x in row["log"].values())
        unfit = sum(len(x["unfit"]) for x in row["log"].values())
        sc, base = row["scoring"], row["from"]
        frm = f"{base['label']} ({base['ai']:.2f})"
        if sc is None:
            L.append(f"| {label} repaired test | {row['section']} | {frm} | {applied} | {unfit} | not scored | | |")
            continue
        hum = sc.get("max_humanizer")
        L.append(f"| {sc['label']} | {row['section']} | {frm} | {applied} | {unfit} | {sc['ai']:.2f} |"
                 f" {'yes' if sc['humanized'] else 'no'} | {'' if hum is None else f'{hum:.2f}'} |")
    return L


def report_md(rp, sections, tallies, when, census_only, final=False, arm_test=False, test_scoring=False):
    L = [f"# The paragraph pass, {when}" + (" (census only)" if census_only else " (final)" if final
                                             else " (arm test)" if arm_test
                                             else " (test scoring)" if test_scoring else ""), ""]
    L.append(f"Job `{rp.job.name}`, writer arms {', '.join(rp.arms)}, purpose {rp.purpose}. Pangram ${rp.spent:.2f} of "
             f"${rp.max_usd:.2f}; Claude (the fact check) about ${rp.claude_spent:.2f} of ${rp.max_claude_usd:.2f}.")
    L += ["", "## Units", "", f"{len(rp.units)} paragraph span(s) regrown; {len(rp.skipped)} left to the run loop:", ""]
    for label, why in rp.skipped:
        L.append(f"- {label}: {why}")
    L += ["", "## Review and fact check", ""] + review_lines(rp)
    if rp.no_checked:
        by = {}
        for why in rp.no_checked.values():
            by[why] = by.get(why, 0) + 1
        L.append("Units with no fact-checked candidate, so they stayed standing: "
                 + ", ".join(f"{n} because {why}" for why, n in sorted(by.items())) + ".")
    elif rp.no_facts:
        L.append("Run with --no-facts: candidates went in on the code gates alone.")
    L.append(f"Fact gate: {rp.fact_gate} ("
             + ("REGROW_FACT_PROMPT, the slips too" if rp.fact_gate == "strict" else "FACT_PROMPT, wrong facts only")
             + ").")
    if rp.unscored:
        L.append(f"Scoring stopped because {rp.dry_why}: " + ", ".join(
            f"section {n} left unscored with {c} unit(s) waiting on a fact check"
            for n, c in sorted(rp.unscored.items())) + ". The run made no scoring after that.")
    L += ["", "## Census", ""]
    for arm in rp.arms:
        t = tallies.get(arm, {})
        total = sum(t.values())
        L.append(f"**{arm}:** {total} draws, " + ", ".join(f"{k} {n}" for k, n in sorted(t.items(), key=lambda kv: -kv[1])))
    for arm in rp.arms:
        ops = sorted(e["op"] for u in rp.units for e in rp.passed(u, arm) if e.get("op") is not None)
        if ops:
            under = sum(1 for x in ops if x <= 0.3)
            L.append(f"**{arm}, Open Pangram on the passing draws:** median {ops[len(ops) // 2]:.2f}, {under} of {len(ops)}"
                     f" at or under 0.30, worst {ops[-1]:.2f}")
    L.append("")
    L.append("| Unit | " + " | ".join(f"{a} passed / drawn" for a in rp.arms) + " |")
    L.append("| --- | " + " | ".join("---" for _ in rp.arms) + " |")
    for u in rp.units:
        cells = [f"{len(rp.passed(u, a))} / {len(rp.pools.get((u['label'], a), {}))}" for a in rp.arms]
        L.append(f"| {u['label']} | " + " | ".join(cells) + " |")
    if rp.arm_tests:
        L += ["", "## Arm test", ""] + arm_test_lines(rp)
    if rp.test_runs:
        L += ["", "## Test scorings", ""] + test_scoring_lines(rp)
    if rp.base2_runs:
        L += ["", "## Second base model", ""] + base2_lines(rp)
    if not census_only and not arm_test and not test_scoring:
        L += ["", "## Scores", "", "| Section | Standing | " + " | ".join(f"all-{a}" for a in rp.arms) + " | Later rounds | Best |",
              "| --- | --- | " + " | ".join("---" for _ in rp.arms) + " | --- | --- |"]
        for n in sections:
            st = rp.standing_score(n)
            cells = []
            for a in rp.arms:
                s = next((x for x in rp.scores if x["section"] == n and x["arm_set"] == a), None)
                cells.append("" if s is None else f"{s['ai']:.2f}" + (" humanized" if s["humanized"] else ""))
            later = [x for x in rp.scores if x["section"] == n and x["arm_set"] in ("mixed", "final")]
            lt = ", ".join(f"{x['ai']:.2f}" + (" humanized" if x["humanized"] else "") for x in later)
            b = rp.best.get(n)
            L.append(f"| {n} | {'' if st is None else f'{st:.2f}'} | " + " | ".join(cells) + f" | {lt} | "
                     + ("" if b is None else f"{b['ai']:.2f} ({b['label']})") + " |")
        L += ["", "What the write-back gets (vetoes out, hand fixes in; \"not scored\" until `--final` scores it):", ""]
        for n in sections:
            sp = rp.section_pick(n)
            L.append(f"- section {n}: " + ("nothing (no clean scoring)" if sp is None else
                     f"{sp['scoring']['label']}, {sp['scoring']['ai']:.2f}" + ("" if sp["scored"] else ", not scored")))
        L += ["", "Voice audit (Burrows' Delta, lower is closer to the author; the author's own chunks sit around 0.96):", ""]
        for n in sections:
            b = rp.best.get(n)
            if b is None:
                continue
            L.append(f"- section {n}: standing {audit(rp.section_text(n, {}))}")
            L.append(f"- section {n}: best {audit(rp.section_text(n, b['picks']))}")
        L += ["", "## Side by side", "", "Original on the left (Claude's, as the sync froze it), the best assembly's text on the right."
              " A unit the best assembly left standing shows the standing text.", ""]
        picks = rp.best_picks()
        seen = set()
        for u in rp.units:
            if u["plabel"] in seen:
                continue
            seen.add(u["plabel"])
            us = [x for x in rp.units if x["plabel"] == u["plabel"]]
            tags, now = [], None
            for x in us:
                p = picks.get(x["label"])
                if p is None:
                    continue
                now = p["paragraph"]
                e = next((d for pool_key, pool in rp.pools.items() if pool_key[0] == x["label"]
                          for d in pool.values() if d["text"] == p["text"]), None)
                part = x["label"][len(x["plabel"]):].strip() or (
                    "whole" if x["start"] == 0 and x["end"] == len(x["runs"]) else "tail")
                tags.append(f"{part}: " + (f"{e['arm']}, Open Pangram {e.get('op', 0):.2f}" if e else "pick")
                            + (", flagged" if p.get("flagged") else ""))
            tag = f" ({'; '.join(tags)})" if tags else ""
            L += [f"### {u['plabel']}{tag}", "", f"**Original:** {u['para']}", "",
                  f"**Now:** {now if now is not None else u['para_standing']}", ""]
    return "\n".join(L) + "\n"


def test_scoring_section(label):
    """The section a scoring's label names ("s4 round 3" is section 4), or None."""
    m = re.match(r"\s*s(\d+)(?:\s|$)", label or "")
    return int(m.group(1)) if m else None


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="the paragraph pass: whole Claude paragraphs rewritten from their notes")
    ap.add_argument("--job", help="the job, a folder in jobs/ (default: [job] default in config.toml)")
    ap.add_argument("--job-dir", help="a job folder anywhere, instead of --job")
    ap.add_argument("--section", help="sections, comma-separated (default: the job's scope)")
    ap.add_argument("--arms", default=",".join(MISTRAL_ARMS),
                    help="writer arms, comma-separated: notes and fewshot share the loop's server; olmo and base2"
                         " each run on their own server, so on their own (base2: the few-shot prompt on"
                         " regrow.base2_model)")
    ap.add_argument("--census", action="store_true", help="draw and gate only; score nothing")
    ap.add_argument("--purpose", default="experiment", choices=["loop", "experiment"])
    ap.add_argument("--max-usd", type=float, help="the most this run may spend on Pangram")
    ap.add_argument("--max-claude-usd", type=float, help="the most the fact check may spend")
    ap.add_argument("--draws", type=int, help="draws per arm per unit in the census")
    ap.add_argument("--rounds", type=int, help="scoring rounds after the arm sets")
    ap.add_argument("--no-facts", action="store_true",
                    help="skip Sonnet's fact check and let unchecked candidates in (without it they can't be picked)")
    ap.add_argument("--final", action="store_true",
                    help="score each section's served text (best eligible picks, vetoes out, hand fixes in) once,"
                         " when it isn't scored yet; draws nothing and starts no server")
    ap.add_argument("--arm-test", action="store_true",
                    help="score each section twice, once with the adapter's and once with the few-shot arm's best"
                         " candidate in every unit both arms have one for, the rest standing; draws nothing, starts no"
                         " server, and the two scorings are never a best or served")
    ap.add_argument("--arm-test-min", type=int, default=4,
                    help="the fewest test units a section needs for the arm test")
    ap.add_argument("--test-scoring", metavar="LABEL",
                    help='rebuild the banked scoring LABEL ("s4 round 3") with the vetoed picks out, the hand fixes'
                         " in and every repair whose `from` fits its pick applied, and score that once as a test"
                         " scoring; on its own, its section from the label, draws nothing and starts no server")
    ap.add_argument("--only-stuck", action="store_true",
                    help="draw only for the stuck units (not pinned, no pick in what the pass serves). With --arms"
                         " base2 the run then scores one assembly per section, the served picks plus the base2 spans,"
                         " and stops; with the Mistral arms the normal rounds follow")
    a = ap.parse_args(argv)
    if a.final and a.census:
        ap.error("--final and --census don't go together")
    if a.arm_test and (a.final or a.census or a.only_stuck):
        ap.error("--arm-test goes on its own (not with --final, --census or --only-stuck)")
    if a.final and a.only_stuck:
        ap.error("--final draws nothing, so --only-stuck has nothing to narrow")
    test_section = None
    if a.test_scoring is not None:
        if a.final or a.census or a.arm_test or a.only_stuck:
            ap.error("--test-scoring goes on its own (not with --final, --census, --arm-test or --only-stuck)")
        test_section = test_scoring_section(a.test_scoring)
        if test_section is None:
            ap.error(f'--test-scoring takes a scoring\'s label such as "s4 round 3"; {a.test_scoring!r} names no section')
        if a.section and [int(x) for x in a.section.split(",")] != [test_section]:
            ap.error(f"--test-scoring {a.test_scoring!r} is section {test_section}'s; leave --section out")
    arms = [x.strip() for x in a.arms.split(",")]
    for solo in SOLO_ARMS:
        if solo in arms and len(arms) > 1:
            ap.error(f"the {solo} arm runs on its own server: --arms {solo} on its own")
    if a.arm_test:
        arms = list(MISTRAL_ARMS)

    from . import db
    from .claude_call import run as claude_run
    from .local_models import LlamaServer
    from .open_pangram import OpenPangram
    from .pangram import Pangram
    from .main_copy import load_job
    settings = Settings()
    conn = db.connect(settings.db_path)
    job = load_job(a.job, a.job_dir, settings)
    sections = ([test_section] if test_section is not None
                else [int(x) for x in a.section.split(",")] if a.section else list(job.sections))
    bank = Bank(job.bank_path)
    rcfg = settings.config.get("regrow", {})
    if arms == ["olmo"]:
        model = rcfg.get("olmo_model", OLMO_MODEL)
    elif arms == ["base2"]:
        model = rcfg.get("base2_model", BASE2_MODEL)
    else:
        model = rcfg.get("model") or settings.config["loop"]["model"]
    srv = None if a.arm_test or test_section is not None else LlamaServer(model, settings=settings)
    rp = RegrowPass(job=job, bank=bank, settings=settings, conn=conn, claude=None if a.no_facts else claude_run,
                    pangram=Pangram(settings, conn), server=srv, op=None, purpose=a.purpose, max_usd=a.max_usd,
                    max_claude_usd=a.max_claude_usd, arms=arms, no_facts=a.no_facts)
    if a.draws:
        rp.draws = a.draws
    if a.rounds:
        rp.rounds = a.rounds
    log_path = settings.runs_dir / job.name / "regrow.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(line):
        stamp = time.strftime("%Y-%m-%d %H:%M")
        print(f"{stamp} {line}", flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{stamp} {line}\n")
    rp.log = log
    if a.final:
        rp.load_units(sections)
        rp.load_bank()
        log(f"--final: {len(rp.units)} unit(s) in sections {', '.join(map(str, sections))}; ceiling ${rp.max_usd:.2f}")
        for n in sections:
            rp.final(n)
        tallies = {arm: rp.tally(arm) for arm in rp.arms}
    elif test_section is not None:
        rp.load_units(sections)        # no notes, no fact check, no server: the scoring's picks were checked already
        rp.load_bank()
        log(f"--test-scoring {a.test_scoring}: section {test_section}; ceiling ${rp.max_usd:.2f}")
        try:
            rp.test_scoring(a.test_scoring)
        except ValueError as e:
            log(f"--test-scoring refused: {e}")
            return 2
        tallies = {arm: rp.tally(arm) for arm in rp.arms}
    elif a.arm_test:
        rp.load_units(sections)        # no regate: the banked verdicts stand, and no server is needed
        rp.load_notes()                # the fact check reads them
        rp.load_bank()
        log(f"--arm-test: {len(rp.units)} unit(s) in sections {', '.join(map(str, sections))}; at least"
            f" {a.arm_test_min} test unit(s) a section; ceiling ${rp.max_usd:.2f}")
        rp.arm_test(sections, a.arm_test_min)
        tallies = {arm: rp.tally(arm) for arm in rp.arms}
    else:
        rp.setup(sections)
        log(f"{len(rp.units)} unit(s) in sections {', '.join(map(str, sections))}; {len(rp.skipped)} left to the loop;"
            f" arms {', '.join(rp.arms)}; ceiling ${rp.max_usd:.2f} for this run (${rp.spent_before:.2f} already on"
            f" Pangram today), purpose {rp.purpose}")
        print(f"starting {srv.name}...", flush=True)
        srv.start()
        try:
            rp.op = OpenPangram("roberta")
            tallies = rp.run(sections, census_only=a.census, only_stuck=a.only_stuck)
        finally:
            srv.stop()
    when = time.strftime("%Y-%m-%d %H%M")
    tag = (" (census)" if a.census else " (final)" if a.final else " (arm test)" if a.arm_test
           else " (test scoring)" if test_section is not None else "")
    out = report_file(f"regrow {when}{tag}.md")      # makes analytics/ first, as a fresh clone has none
    out.write_text(report_md(rp, sections, tallies, when, a.census, final=a.final, arm_test=a.arm_test,
                             test_scoring=test_section is not None), encoding="utf-8")
    if not a.census and not a.arm_test and test_section is None:   # a test changes no best: the picks file stays
        rp.write_picks()
    log(f"done: Pangram ${rp.spent:.2f}, Claude about ${rp.claude_spent:.2f}; report {out.name}"
        + ("" if not rp.best else "; best " + ", ".join(f"s{n} {s['ai']:.2f}" for n, s in sorted(rp.best.items()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
