"""Draws, filters and the judge: the per-run machinery the loop, the paragraph pass and the repair step share.

A run is a stretch of consecutive machine-written sentences inside a paragraph. draw_pool asks the local base model
for candidates (free), filter_candidates throws out the ones that break a rule (length, dashes, must-keeps, new
numbers or names, markup, swearing, assistant tells, AI tells), Open Pangram ranks what's left, and judge_pool has
Sonnet pick one, a blind head-to-head check it against the text standing there, and a fact check look for anything
that isn't so. regrow_run is the two in a row.

The text helpers the engine uses everywhere (sents, quoted, numbers, names, split_paragraphs) live here too, and so
does make_tics, the AI-tells scan as a function. These came from the research experiments (3 to 6) and were moved
into the engine as they were.
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from engine.claude_call import run as claude_run
from engine.checks import must_kept, splice_citations, strip_footnote_marks, trim_continuation
from engine.find_lines import classify
from engine.settings import path_setting
from engine.util import describe

YOURS = ("yours", "yours?")
N_CANDS = 48                   # per draw; a second draw follows when the first gives nothing usable
SHOW = 8                       # candidates the judge sees

JUDGE_TEMPLATE = """Pick the best rewrite of a stretch of <<DESCRIPTION>>.

Text right before it:
<<BEFORE>>

Original stretch:
<<ORIGINAL>>

The notes it has to cover (the whole paragraph's notes; this stretch covers the part the text before it hasn't):
<<NOTES>>

Text that comes right after it, unchanged:
<<AFTER>>

Candidates (a [scanner: ...] note lists AI-writing tells a pattern scan found in that candidate; treat each as a mark against it):
<<CANDS>>

<<VOICE>><<JOKE>>This text is by a human who writes the way a person talks. Reject the assistant voice: openers like "here's the thing", "frankly", "at the end of the day", "I like to think of it as", neat summarizing closers, tidy triads, whimsical extended metaphors, and anything that reads like a chatbot being charming. The author's calmer register is accepted too: a candidate that is plainer, less sweary or less performed than the original is not worse for that, as long as it still reads as the same person talking; judge it on whether it says the thing the way a person would, not on how much it performs.<<DANGLE>>

Pick the candidate that (1) covers every point in the notes that the original stretch covers, with no fact, definition or step dropped (a point the text before or after already makes counts as covered), and keeps any joke alive<<FUNNY>>, (2) adds no facts, names, numbers, examples or claims beyond the notes and the original (something the notes state is fine to say), (3) reads naturally between the text before and the text after, (4) sounds like a person talking<<QUALITY>>. If none clears all of these, answer NONE.

Reply with only JSON: {"pick": <number or null>, "drops": ["any fact, definition, number or step from the notes that the pick leaves out; a weaker joke is not a drop<<PUNCH>>"], "why": "short reason"}"""
JUDGE_PROMPT = describe(JUDGE_TEMPLATE)        # the default description; judge_pool fills the job's own

H2H_TEMPLATE = """Two versions of the same stretch of <<DESCRIPTION>>. Both make the same point.

Text right before it:
<<BEFORE>>

Text right after it:
<<AFTER>>

A: <<A>>

B: <<B>>

Which is the better piece of writing for this text: funnier, tighter, more natural as speech, and the better fit between the text before and after?<<JOKE>> A plainer or less sweary version is not worse for being calmer; judge how it reads as a person talking, not how much it performs. If they're about as good as each other, say so. Reply with only A, B or SAME."""
H2H_PROMPT = describe(H2H_TEMPLATE)

# the adapter arm's dangling opener (2026-11-09): the notes-to-text adapter writes from the notes alone, so a run that
# opens a paragraph can start mid-thought ("Because it never checks...", "So it guesses") as if answering a question
# the paragraph never asked. The judge gets this line only for a run that opens its paragraph.
DANGLE_LINE = (" A candidate that opens the paragraph with \"Because\", \"No, but\", \"Instead\", \"So\" or another word"
               " that answers something has to answer something in the text before the paragraph; if nothing before it"
               " sets that up, the candidate is out.")

FACT_PROMPT = """A rewrite of one stretch of an essay has to say only what the original and the notes support. List every factual statement, number, name, example or technical claim in the rewrite that is wrong, changed, or not supported by the original stretch and the notes. Looser wording, a different joke, or a dropped point do not count here, only things the rewrite says that aren't so.

Original stretch:
<<ORIGINAL>>

Notes for the paragraph:
<<NOTES>>

Rewrite:
<<NEW>>

Reply with NONE, or one line per problem."""

ABBREV = re.compile(r"\b(?:et al|e\.g|i\.e|vs|etc|Dr|Mr|Mrs|Ms|St|No)\.$")

# first words that anchor register without steering content ("In" steers toward "In 2023" or "In 99% of cases")
SAFE_OPENERS = {"well", "so", "then", "and", "but", "or", "now", "anyway", "also", "basically", "honestly", "look",
                "okay", "ok", "yeah", "still", "plus", "which", "because", "except", "meanwhile", "again", "no", "yes"}
PRONOUNS = {"it's", "it", "that's", "that", "they're", "they", "you", "we", "this", "there's", "these", "those"}
# chatbot openers and asides a base model still produces now and then; a candidate with one is out. "The thing is," came
# off the list on 2026-11-10: it's the author's own opener, and the loop had been rejecting every draw that put it
# back (experiment 10's first run)
ASSISTANT_TELL = re.compile(
    r"\b(here's the thing|at the end of the day|let's be honest|to be fair|frankly|simply put|"
    r"in a nutshell|it's worth noting|needless to say|i like to think of it as|think of it (?:like|as)|"
    r"you see,|picture this|long story short|make no mistake|i'll try my best|follow your instructions|as requested|"
    r"the paragraph will be|based on the samples provided)\b", re.I)   # the last five: Olmo SFT's preamble, 2026-11-14
# experiment 7's counter (plan 6.9): a base model in a chat-shaped template starts "Sure, here's a revised version"
COSTUME_OPENER = re.compile(r"^\W*(sure\b|here's\b|here is\b|certainly\b|i've\b|i have revised\b|okay,|ok,|alright,|got it\b)", re.I)
HEDGE_WORD = r"\b{}\b"                   # a hedge must-keep matches as a whole word
# Gemma base writes HTML and Markdown into prose; the notes adapter writes single-asterisk italics ("*like this*",
# 2026-11-09). The document uses those itself now and then, so the filter only rejects a candidate whose original run
# has none, as for the rest of this list
MARKUP = re.compile(r"<[^>]+>|\[[^\]]+\]|\*\*|__|~~|^#+\s|(?<!\*)\*[^\s*][^*\n]{0,40}\*(?!\*)", re.M)
SWEAR = re.compile(r"\b(fuck\w*|shit\w*|damn\w*|ass|arse|hell|bloody|crap\w*|bastard\w*|piss\w*|bollocks)\b", re.I)


# ---------------------------------------------------------------------- text helpers (from experiments 4, 5 and 6)
def split_paragraphs(window):
    """[[(sentence, locked)], ...] per paragraph; a short numbered heading is one locked sentence. A heading in
    another style ("Part one") stays a one-sentence paragraph, and the caller's `notation` (the job's headings)
    locks it, as it does subsection headings."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", window) if p.strip()]
    out = []
    for p in paras:
        if re.match(r"^\d+\.\s", p) and len(p.split()) < 12:
            out.append([(p, True)])          # the heading: locked
            continue
        sents = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", p) if s.strip()]
        out.append([(s, False) for s in sents])
    return out


def quoted(s):
    """Double-quoted spans and "(..., year)" citations, as written."""
    return re.findall(r"\"[^\"]+\"|\([^)]*\d{4}\)", s)


def numbers(s):
    """The numbers a text carries, as strings."""
    return set(re.findall(r"\d+(?:[.,/]\d+)*", s))


def names(s):
    """Capitalized words that aren't sentence-initial: a rough proper-noun set."""
    toks = re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-zA-Z0-9-]+", s.strip())
    return {t for t in toks if t not in {"I", "I'm", "I'll", "I've", "I'd"}}


def fmt_example(notes, para):
    """One worked example for the writer's prompt: the notes, then the paragraph written from them."""
    return "Notes:\n" + "\n".join(f"- {n}" for n in notes) + "\n\nParagraph:\n" + para


def sents(p):
    """Sentence split that doesn't break inside double quotes or after an abbreviation ("et al. (2025)"). It does
    break before a sentence that opens with a digit ("a new rule. 3.5 is down to 1.2" is two sentences; 2026-11-10,
    from experiment 10's first run, where a one-word insertion read as a sentence split), which puts it in step with
    sentence_ends below."""
    out, start, in_quote = [], 0, False
    for m in re.finditer(r"[.!?][\"')]*\s+(?=[A-Z\"'(\d])", p):
        before = p[start:m.start() + 1]
        in_quote = (p[:m.end()].count('"') % 2) == 1
        if in_quote or ABBREV.search(before.rstrip()[-8:]):
            continue
        out.append(p[start:m.end()].strip())
        start = m.end()
    tail = p[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def sentence_ends(text):
    """End offsets of complete sentences, with the same quote and abbreviation rules as sents()."""
    ends = []
    for m in re.finditer(r"[.!?][\"')]*(?=\s|$)", text):
        if (text[:m.end()].count('"') % 2) == 1:          # inside an open quote
            continue
        if ABBREV.search(text[:m.start() + 1].rstrip()[-8:]):
            continue
        ends.append(m.end())
    return ends


def first_n_sentences(text, n, long=False):
    """One sentence for a one-sentence run (two when the original is long); n-1 to n+1 for longer runs."""
    text = text.split("\n")[0]
    ends = sentence_ends(text)
    lo, hi = (1, 2 if long else 1) if n == 1 else (n - 1, n + 1)
    if len(ends) < lo:
        return None
    return text[:ends[min(len(ends), hi) - 1]].strip()


def names_ok(candidate, known):
    """No capitalized word the window doesn't already have. Plurals and possessives of a known name are fine."""
    norm = lambda n: re.sub(r"(?:'s|s)$", "", n.lower())  # noqa: E731
    have = {norm(n) for n in known}
    # a hyphenated word whose first part is known ("AI-written" with "AI") is no new name (2026-11-09)
    return all(norm(n) in have or norm(n.split("-")[0]) in have for n in names(candidate))


def run_names(orig_text):
    """Every capitalized word of a run's own original, sentence openers included. names() skips a word that opens a
    sentence, so a surname that opens its sentence never counted as known, and a draw that put it mid-sentence was
    thrown out as a new name (the section 1 diagnosis: 261 such rejects on one sentence)."""
    return set(re.findall(r"\b[A-Z][a-zA-Z0-9-]+", orig_text))


def long_quote(text):
    """The first quotation of 6+ words, as a match, or None."""
    m = re.search(r"\"([^\"]+)\"", text)
    return m if m and len(m.group(1).split()) >= 6 else None


def short_title(text):
    """The first quotation in a run when it's a title of 3 to 5 words in title case ("Lack of Specificity and
    Detail"), with its quote marks, else None. Only the first quotation counts, since the model's lead-in stops at
    the first quote mark it writes."""
    m = re.search(r"\"([^\"]+)\"", text)
    if not m:
        return None
    ws = m.group(1).split()
    big = [w for w in ws if len(w) > 3]
    if not 3 <= len(ws) <= 5 or not ws[0][:1].isupper() or not big:
        return None
    return m.group(0) if sum(1 for w in big if w[:1].isupper()) * 2 >= len(big) else None


def is_notation(s):
    """A pattern template rather than prose: "X and X and X." or a sentence that opens with its quoted example."""
    words = s.split()
    placeholders = sum(1 for w in words if re.fullmatch(r"[XYZ][.,]?", w))
    return placeholders >= 2 or s.startswith('"')


# The built-in AI-tells check, for when no scanner is set (TICS_DIR in engine/settings.py). It is a small stand-in,
# written for this repo: a few stock phrases that count as hard hits, and two patterns that come back as check labels
# for the judge to see. A real scanner finds far more; set TICS_DIR to use one.
BUILTIN_HARD = re.compile(
    r"\b(?:delv(?:e|es|ed|ing)|tapestry|a testament to|in the realm of|it's important to note|navigat\w+ the complexit\w+|"
    r"a symphony of|a dance of|unspoken (?:bond|understanding)|sent shivers down)\b", re.I)
BUILTIN_CHECKS = (
    ("not X but Y", re.compile(r"\bnot (?:just|only|merely)\b[^.!?]{1,60}\bbut\b", re.I)),
    ("a line that sums up its own point", re.compile(r"\b(?:and that's|which is) (?:the|what) (?:point|beauty)\b", re.I)),
)


def builtin_tics(t):
    """The built-in check (see BUILTIN_HARD): (hard hit count, check labels) for one candidate."""
    return (len(BUILTIN_HARD.findall(t)), sorted({label for label, rx in BUILTIN_CHECKS if rx.search(t)}))


def make_tics(tics_dir=None):
    """The AI-tells scan as a function: (hard hit count, check labels) for one candidate.

    With a scanner folder (`tics_dir`, else TICS_DIR from engine/settings.py) it loads tic_scan.py and
    tic_patterns.py from there and runs tic_scan's AI rules (every rule but HOUSE_ONLY). With none set it returns
    builtin_tics, the small built-in stand-in above."""
    tics_dir = tics_dir or path_setting("TICS_DIR")
    if not tics_dir:
        return builtin_tics
    sys.path.insert(0, str(tics_dir))
    import tic_scan  # noqa: E402
    from tic_patterns import HOUSE_ONLY  # noqa: E402
    # the house-only rules come out, every other rule stays
    ai_rules = [r for r in tic_scan.compile_rules() if r["label"] not in HOUSE_ONLY]

    def tics(t):
        hits = tic_scan.scan_line(t, ai_rules, None)
        # the number filter already settles 4s and 7s, so that label isn't a warning here
        return (sum(1 for _, lvl, _ in hits if lvl == "hard"),
                sorted({lab for _, lvl, lab in hits if lvl != "hard" and not lab.startswith("4/7")}))
    return tics


def build_runs(window, flagged0, verdicts=None, notation=None):
    """(who, heading, body, runs) for a window: which sentences get regrown, grouped into runs.

    Every Claude sentence gets regrown. The residue test (plan, 2026-11-08) showed three Claude sentences left in
    the window push the whole window to 1.00, and without them your lines plus Mistral's fill score 0.00.
    "edited" and "both" lines (yours and Claude's hands on them) are regrown only while Pangram flags them
    (`flagged0`, the window's round 0 segments), which is the order you set (Claude's lines first, these last).
    Notation ("X and X and X.") and your lines stay as they are. A sentence carrying a long quote is regrown as its
    lead-in only, and the quote is spliced back in by code.

    Phase 2 (engine/run.py): `verdicts` ({sentence: verdict}, frozen from `your lines.md`) stands in for the live
    find_lines scan, and `notation` for is_notation (narrowed to engine.checks.is_template)."""
    coarse = verdicts if verdicts is not None else {s: v for s, v, _, _ in classify(window)[0]}
    notation = notation or is_notation

    class Who(dict):
        """find_lines' verdicts, looked up for a finer sentence split: a piece takes its containing sentence's verdict."""
        def get(self, s, default=None):
            return coarse.get(s) or next((v for k, v in coarse.items() if s in k), default)

    who = Who()
    paras = split_paragraphs(window)                     # [[(sentence, locked)], ...]
    heading = paras[0][0][0] if paras[0][0][1] else None
    body = [sents(" ".join(s for s, _ in p)) for p in paras if not p[0][1]]
    runs = []
    for p in body:
        seq = []
        for s in p:
            v = who.get(s)
            in_flag0 = any(s[:40] in f for f in flagged0)
            is_claude = (v not in YOURS and not notation(s)
                         and (v not in ("edited", "both") or in_flag0))
            # a sentence carrying quoted material is a run of its own, so its must-keeps don't sink its neighbors
            solo = bool(quoted(s))
            if seq and seq[-1]["claude"] == is_claude and not solo and not seq[-1].get("solo"):
                seq[-1]["sents"].append(s)
            else:
                seq.append({"sents": [s], "claude": is_claude, "solo": solo})
        runs.append(seq)
    return who, heading, body, runs


def is_costume(raw):
    """The draw's first line opens like an assistant ("Sure, here's...") or carries an assistant tell."""
    first = raw.split("\n")[0].strip()
    return bool(COSTUME_OPENER.search(first) or ASSISTANT_TELL.search(first))


def filter_candidates(raw, *, orig_text, must, known_names, swears_elsewhere, swear_budget, tics, why,
                      extra_must=None, extra_checks=None):
    """Experiment 6's filter stack over raw candidates. Returns (passed, up to four rejected examples) and counts
    each reject's reason in `why`. `extra_must` (experiment 8) is a list of hedge words that have to survive as
    whole words, counted as its own reason so the report can tell it apart from the quote must-keeps.
    `extra_checks` (Phase 2, engine/checks.py) are callables that take a candidate and return a reject reason
    or None; they run after the hedge check, so the stack's order is otherwise exp06's."""
    words = orig_text.split()
    cands, rejected = [], []

    def rej(reason, c):
        why[reason] = why.get(reason, 0) + 1
        if len(rejected) < 4:
            rejected.append(f"[{reason}] {c[:140]}")

    for c in dict.fromkeys(re.sub(r"\s{2,}", " ", c).strip() for c in raw):
        if not (0.5 * len(words) <= len(c.split()) <= max(1.8 * len(words), 14)):
            rej("length", c)
        elif "—" in c or "--" in c:
            rej("dash", c)
        elif "�" in c or re.search(r"[½¼¾⅓⅔⅛̀-ͯ]", c):
            rej("bad glyph", c)
        elif c == orig_text.strip():
            rej("identical", c)
        elif any(not must_kept(m, c, orig_text) for m in must):
            rej("must-keep", c)
        elif extra_must and any(not re.search(HEDGE_WORD.format(re.escape(h)), c, re.I) for h in extra_must):
            rej("must-keep hedge", c)
        elif extra_checks and (reason := next((r for r in (f(c) for f in extra_checks) if r), None)):
            rej(reason, c)
        elif not numbers(c) <= numbers(orig_text):
            rej("new number", c)
        elif not names_ok(c, set(known_names) | run_names(orig_text)):
            rej("new name", c)
        elif MARKUP.search(c) and not MARKUP.search(orig_text):
            rej("markup", c)
        elif c.startswith('"') and not orig_text.startswith('"'):
            rej("invented dialogue", c)             # '"I want the payoff already!"' for a plain line
        elif len(SWEAR.findall(c)) > len(SWEAR.findall(orig_text)) + 1 \
                or len(SWEAR.findall(c)) + swears_elsewhere > swear_budget:
            rej("swearing", c)
        elif ASSISTANT_TELL.search(c):
            rej("assistant tell", c)
        elif tics(c)[0]:
            rej("hard tic", c)
        else:
            cands.append(c)
    return cands, rejected


def regrow_run(*, srv, op, settings, conn, tics, prompt, orig_text, n_sent, cur_text, is_original, notes, before,
               after, known_names, swears_elsewhere, swear_budget, digit_ban, pi, ri, rnd, rounds, role, prompt_for=None,
               extra_must=None, claude=None, extra_checks=None, splice_long_quote=True, normalize=None):
    """Draw candidates for one run from `prompt`, filter, rank, judge, head-to-head and fact check.

    `prompt` ends where the model continues (the paragraph's text so far, plus a space when there is any).
    `orig_text` is Claude's original run, `cur_text` the text standing there now, `is_original` whether they're
    the same run. Returns (new_text_or_None, change, stats): `change` is the report entry exp06 writes, `stats`
    holds the draws asked for, the reject tally, the assistant-costume count and the fact check outcome.

    Experiment 8's hooks, both off by default: `prompt_for(k)` gives draw k its own prompt (varied samples in
    front), and `extra_must` adds hedge words as must-keeps. stats then also carries the judge's pick, the fact
    check's text when the pick failed it, and the draw number the pick came from.

    Phase 2's hooks (engine/run.py), all off by default: `claude` stands in for engine.claude_call.run,
    `extra_checks` adds filters, `splice_long_quote` and `normalize` handle prose that quotes patterns. The body
    is draw_pool then judge_pool, split on 2026-11-08 so the loop can draw while a Claude cap holds."""
    pool = draw_pool(srv=srv, op=op, tics=tics, prompt=prompt, orig_text=orig_text, n_sent=n_sent,
                     is_original=is_original, known_names=known_names, swears_elsewhere=swears_elsewhere,
                     swear_budget=swear_budget, digit_ban=digit_ban, pi=pi, ri=ri, rnd=rnd, rounds=rounds,
                     prompt_for=prompt_for, extra_must=extra_must, extra_checks=extra_checks,
                     splice_long_quote=splice_long_quote, normalize=normalize)
    if not pool["shown"]:
        return None, {"para": pi, "orig": orig_text, "result": f"no candidate passed ({pool['raw']} drawn: {pool['tally']})",
                      "examples": pool["rejected"]}, pool["stats"]
    return judge_pool(pool, settings=settings, conn=conn, tics=tics, orig_text=orig_text, cur_text=cur_text,
                      is_original=is_original, notes=notes, before=before, after=after, pi=pi, ri=ri, rnd=rnd,
                      rounds=rounds, role=role, claude=claude)


def draw_pool(*, srv, op, tics, prompt, orig_text, n_sent, is_original, known_names, swears_elsewhere, swear_budget,
              digit_ban, pi, ri, rnd, rounds, prompt_for=None, extra_must=None, extra_checks=None,
              splice_long_quote=True, normalize=None, k_offset=0, next_sent="", no_starter=False):
    """The free part of regrow_run: draws, the filter stack and Open Pangram's ranking. Costs no Claude, so the run
    loop can keep doing it while a Claude cap holds, and bank the pool for when the cap lifts (plan 7.3).
    Returns {"shown": the best SHOW by Open Pangram, "ranked": every passed candidate with its score, "raw",
    "rejected", "tally", "stats", "origin"}. `splice_long_quote=False` skips the long-quote lead-in splice (a
    sentence quoting patterns keeps its quotes as must-keeps instead); `normalize` maps each draw before the
    filters (the quote splice); `k_offset` moves the draw seeds on for extra pool draws. `no_starter` (the adapter
    arm, 2026-11-09) turns off the one-word starter and the last-round ladder: both come from Claude's sentence, which
    the adapter never sees, and a starter's trailing space makes the adapter end the paragraph at once (its prompt
    ends flush with the text so far)."""
    stats = {"draws": 0, "returned": 0, "costume": 0, "why": {}, "passed": 0, "facts": None, "accepted": False,
             "pick": None, "facts_text": None, "source_k": None}
    origin = {}                       # candidate text -> the first draw that produced it
    words = orig_text.split()
    n_tok = int(len(words) * 2.2) + 40
    # a long quote: the model writes only the lead-in, and the quote and what follows it are spliced back.
    # A citation like (OpenAI, 2023) is spliced into the lead-in too, since the model never writes one verbatim.
    lq = long_quote(orig_text) if n_sent == 1 and splice_long_quote else None
    # where the spliced tail starts and what joins it on (M2, 2026-11-09): a quote that closes the sentence inside
    # parentheses ('... it said ("we fixed it, honest").') takes its "(" along and joins with a space, since one run
    # lost the "(" and kept the ")"; every other quote joins with ": " as before, and the punctuation check on the
    # finished candidate catches what that breaks
    joiner, tail_at = ": ", lq.start() if lq else None
    if lq:
        head = orig_text[:lq.start()]
        if head.count("(") > head.count(")") and re.fullmatch(r"\)[.!?]?\s*", orig_text[lq.end():]):
            joiner, tail_at = " ", head.rindex("(")
    quote_tail = orig_text[tail_at:] if lq else None
    # a short quoted title (3 to 5 words, 2026-11-09): the model writes up to it, code puts it in exactly, and the
    # model goes on from there to the sentence's end, so none of Claude's words after the title are kept
    title = short_title(orig_text) if n_sent == 1 and splice_long_quote and not lq else None
    title_head = orig_text[:orig_text.index(title)] if title else ""
    orig_lead = orig_text[:tail_at].strip() if lq else ""
    citation = next((m for m in quoted(orig_text) if m.startswith("(")), None)
    must = [m.strip('"').strip(",.;:") for m in quoted(orig_text)
            if not m.startswith("(") or not quote_tail]
    # a one-word starter anchors the register (a question keeps its question word) and carries no
    # watermark to speak of; words like "In" steer content instead, so they're left out
    first = words[0].strip(",.;:")
    starter = words[0] + " " if (first.lower() in SAFE_OPENERS or orig_text.rstrip().endswith("?")) else ""
    # a second word that's only a pronoun ("Well, it's") keeps the sentence answering the same question
    if starter and len(words) > 2 and words[1].lower() in PRONOUNS:
        starter += words[1] + " "
    # the starter ladder: a run still carrying its Claude text in the last round keeps its first four
    # words, which pins the sentence to the point the judge kept finding dropped (HALO-style glue)
    if rnd == rounds and is_original and not quote_tail and len(words) > 8:
        starter = " ".join(words[:4]) + " "
    if no_starter:
        starter = ""
    # no digits in the original means no digits in the candidate: ban them at the sampler (6.8's "ban").
    # Never ban digits the model has good reason to write (the notes say "GPT-4" and "2023"): with
    # them banned, the sampler collapsed onto combining-strikethrough marks (runs 9 and 10).
    ban = digit_ban if not numbers(orig_text) else None

    def finish(c):
        """Every draw on its way to the filters: the caller's normalize, then footnote marks out, citations put back
        exact, and a run-on into the next run's sentence cut back (the section 1 diagnosis, 2026-11-09)."""
        c = normalize(c) if normalize else c
        c = splice_citations(strip_footnote_marks(c, orig_text), orig_text)
        return trim_continuation(c, n_sent, next_sent)

    def one(k):
        """(candidate or None, the raw first line the model wrote)."""
        seed = 26000 + rnd * 100000 + (pi * 10 + ri) * 1000 + k
        prompt_k = prompt_for(k) if prompt_for else prompt
        if quote_tail:
            # the lead-in ends at the colon before the quote; stopping at a quote mark cut titles in two
            text, _ = srv.complete(prompt_k + starter, n_predict=60, temperature=1.0, min_p=0.05, seed=seed,
                                   stop=["\n", ":"] + (["("] if joiner == " " else []), ban=ban)
            # a full stop comes off too, or the splice writes "tell.: " (M2)
            lead = (starter + text.split("\n")[0]).strip().rstrip(" :,;.")
            # a lead-in that never reached its colon ran to the token budget and got cut mid-word
            if len(lead.split()) < 4 or len(lead.split()) > 2 * len(orig_lead.split()) + 6 \
                    or lead.count('"') % 2 or (joiner == " " and lead.count("(") != lead.count(")")) or "~~" in lead:
                return None, starter + text
            if citation and citation not in lead:
                year = re.search(r"\d{4}", citation)
                # a lead that already names the source and the year is its own citation
                if not (year and year.group(0) in lead and "OpenAI" in lead):
                    lead += " " + citation
            return lead + joiner + quote_tail, starter + text
        if title:
            text, _ = srv.complete(prompt_k + starter, n_predict=int(len(title_head.split()) * 2.2) + 20,
                                   temperature=1.0, min_p=0.05, seed=seed, stop=["\n", '"'], ban=ban)
            lead = (starter + text.split("\n")[0]).rstrip()
            if len(lead.split()) < 3 or len(lead.split()) > 2 * len(title_head.split()) + 6 or '"' in lead \
                    or "~~" in lead or sentence_ends(lead):
                return None, starter + text
            mid = lead + " " + title
            rest, _ = srv.complete(prompt_k + mid, n_predict=n_tok, temperature=1.0, min_p=0.05, seed=seed + 50000,
                                   stop=["\n"], ban=ban)
            return first_n_sentences(mid + rest, n_sent, long=len(words) >= 30), starter + text + rest
        text, _ = srv.complete(prompt_k + starter, n_predict=n_tok, temperature=1.0, min_p=0.05, seed=seed,
                               stop=["\n"], ban=ban)
        return first_n_sentences(starter + text, n_sent, long=len(words) >= 30), starter + text

    raw = []
    for attempt in range(2):          # a second draw when the first gives nothing usable
        with ThreadPoolExecutor(2) as pool:
            ks = range(k_offset + attempt * N_CANDS, k_offset + (attempt + 1) * N_CANDS)
            got = list(pool.map(one, ks))
        for k, (c, _) in zip(ks, got):
            if c:
                origin.setdefault(re.sub(r"\s{2,}", " ", finish(c)).strip(), k)
        stats["draws"] += len(got)
        # the costume counter looks at every draw's raw first line, before any filter can hide it
        stats["costume"] += sum(1 for _, r in got if is_costume(r))
        raw += [finish(c) for c, _ in got if c]
        if any(0.5 * len(words) <= len(c.split()) <= max(1.8 * len(words), 14) for c in raw):
            break
    stats["returned"] = len(raw)
    cands, rejected = filter_candidates(raw, orig_text=orig_text, must=must, known_names=known_names,
                                        swears_elsewhere=swears_elsewhere, swear_budget=swear_budget, tics=tics,
                                        why=stats["why"], extra_must=extra_must, extra_checks=extra_checks)
    why = stats["why"]
    stats["passed"] = len(cands)
    tally = ", ".join(f"{k} {n}" for k, n in sorted(why.items(), key=lambda kv: -kv[1]))
    ranked = sorted(zip(cands, op.score(cands)), key=lambda kv: kv[1]["score"]) if cands else []
    return {"shown": [c for c, _ in ranked[:SHOW]], "ranked": [(c, s["score"]) for c, s in ranked], "raw": len(raw),
            "rejected": rejected, "tally": tally, "stats": stats, "origin": origin}


# The joke rule (2026-11-09): regrowth traded jokes for score (a one-line punchline came back as a plain explanation of
# the same point), so a stretch that ends on a joke, a punchline or a dig needs a pick that ends on one too. Sonnet judges whether the original ends on one; a lost punchline is a drop, which
# rejects the pick in every round, the settle round too. A notes line "joke cut: ..." says the joke went on purpose.
JOKE_BLOCK = ("The stretch as first written:\n<<FIRST>>\nIf it ends on a joke, a punchline or a dig, the pick has to end"
              " on one too: its own joke, not those words. A pick that turns the punchline into a plain explanation of"
              " the same point has dropped it. A stretch that ends on a plain point needs no joke.<<HINTS>>\n\n")
JOKE_DROP = ', and a punchline turned into a plain explanation is one: write "punchline"'
JOKE_H2H = (" If one version ends on a joke, a punchline or a dig and the other makes the same point as a plain"
            " explanation, the one with the joke is the better piece of writing.")
JOKE_NOTE = re.compile(r"\b(?:jokes?|joking|snark\w*|punchline\w*|dig|jab|quip\w*|gag|sarcas\w*|mock\w*|deadpan|"
                       r"zinger|teas(?:e|es|ing))\b", re.I)
JOKE_CUT = re.compile(r"^\s*joke cut\s*:", re.I)


def joke_notes(notes):
    """The paragraph's notes that describe a joke (Sonnet writes "Joke: ..." or "The joke ..." lines, and a snark
    line now and then)."""
    return [n for n in notes or [] if JOKE_NOTE.search(n) and not JOKE_CUT.match(n)]


def joke_cut(notes):
    """True when the notes say the joke was cut on purpose (a "joke cut:" line)."""
    return any(JOKE_CUT.match(n) for n in notes or [])


def joke_block(orig_text, notes):
    """The judge's joke question for one run: empty when the notes say the joke was cut on purpose."""
    if joke_cut(notes):
        return ""
    hints = joke_notes(notes)
    return (JOKE_BLOCK.replace("<<FIRST>>", orig_text)
            .replace("<<HINTS>>", (" The paragraph's notes mention: " + " / ".join(hints)) if hints else ""))


VOICE_BLOCK = ("The author wrote this stretch themselves. Their own words are the voice reference:\n<<AUTHOR>>\n\nThe pick"
               " must read as written by that same person (their word choices, rhythm and attitude), and keep these"
               " words of theirs exactly as written: <<WORDS>>.\n\n")


def judge_pool(pool, *, settings, conn, tics, orig_text, cur_text, is_original, notes, before, after, pi, ri, rnd,
               rounds, role, claude=None, voice=None, opens_para=False, description=None):
    """The paid part of regrow_run: the judge, the blind head-to-head and the fact check over a pool's shown
    candidates. `claude` replaces engine.claude_call.run (the run loop passes its capped caller; the rehearsal its
    fake). Returns (new_text_or_None, change, stats) as regrow_run does. `voice` (plan 5.1, a line of yours
    being regrown) is its protected words: the judge then reads your original (`orig_text`) as the voice reference,
    even when the text standing there now is an earlier regrowth. `opens_para` (the adapter arm, 2026-11-09): the
    run opens its paragraph, so the judge also throws out a candidate whose opener answers nothing (DANGLE_LINE).
    `description` is the job's one line about the document, for the prompts (util.describe's default when None)."""
    claude_run_ = claude or claude_run
    stats, origin, cands = pool["stats"], pool["origin"], pool["ranked"]
    # a re-rolled replacement is judged against the text it would replace, not against Claude's
    # original (pass 4 of section 3 kept a bad replacement because the candidates lost to a line
    # that was no longer in the text); facts are still checked against the original
    compare_text = cur_text if cur_text != orig_text else orig_text
    shown = pool["shown"]
    # the last round drops "as good as the original": under option 1 the Claude line goes regardless,
    # so the judge picks the best candidate that holds the points, the facts and the flow
    last = rnd == rounds
    voice_block = "" if voice is None else (VOICE_BLOCK.replace("<<AUTHOR>>", orig_text)
                                            .replace("<<WORDS>>", ", ".join(f'"{w}"' for w in voice) or "(none)"))
    cut = joke_cut(notes)
    verdict, _ = claude_run_(settings, conn, describe(JUDGE_TEMPLATE, description)
                            .replace("<<VOICE>>", voice_block)
                            .replace("<<JOKE>>", joke_block(orig_text, notes))
                            .replace("<<DANGLE>>", DANGLE_LINE if opens_para else "")
                            .replace("<<PUNCH>>", "" if cut else JOKE_DROP)
                            .replace("<<FUNNY>>", "" if last else " and at least as funny")
                            .replace("<<QUALITY>>", "" if last else ", (5) is at least as good as the original as writing")
                            .replace("<<BEFORE>>", before or "(start of the section)")
                            .replace("<<ORIGINAL>>", compare_text)
                            .replace("<<NOTES>>", "\n".join(f"- {n}" for n in notes))
                            .replace("<<AFTER>>", after or "(end of paragraph)")
                            .replace("<<CANDS>>", "\n".join(
                                f"{i + 1}. {c}" + (f"   [scanner: {', '.join(tics(c)[1])}]" if tics(c)[1] else "")
                                for i, c in enumerate(shown))),
                            role=f"{role} judge", model="sonnet", max_budget_usd=0.3)
    m = re.search(r"\{.*\}", verdict, re.S)
    try:
        v = json.loads(m.group(0)) if m else {"pick": None, "why": "unparsable"}
    except json.JSONDecodeError:     # a quote inside "why" or "drops": fall back to the pick alone
        mp = re.search(r'"pick"\s*:\s*(\d+|null)', verdict)
        v = {"pick": int(mp.group(1)) if mp and mp.group(1) != "null" else None,
             "drops": [], "why": "(judge's JSON didn't parse)"}
    pick = v.get("pick")
    if v.get("drops") and isinstance(pick, int):
        pick, v["why"] = None, f"pick dropped: {'; '.join(map(str, v['drops']))}"
    if isinstance(pick, int) and 1 <= pick <= len(shown):
        new = shown[pick - 1]
        stats["pick"], stats["source_k"] = new, origin.get(new)
        # head-to-head gate: the pick has to beat the original as writing, in a blind A/B
        swap = (pi + ri + rnd) % 2 == 1
        pair = (new, compare_text) if swap else (compare_text, new)
        h2h, _ = claude_run_(settings, conn, describe(H2H_TEMPLATE, description)
                            .replace("<<JOKE>>", "" if cut else JOKE_H2H)
                            .replace("<<BEFORE>>", before or "(start of the section)")
                            .replace("<<AFTER>>", after or "(end of paragraph)")
                            .replace("<<A>>", pair[0]).replace("<<B>>", pair[1]),
                            role=f"{role} h2h", model="sonnet", max_budget_usd=0.2)
        m2 = re.search(r"\bSAME\b|\b[AB]\b", h2h.strip().upper())
        letter = m2.group(0) if m2 else None
        pick_wins = letter == "SAME" or letter == ("A" if swap else "B")
        # try for better in the early rounds; in the last round the Claude line goes regardless
        # (option 1, the watermark), so the best candidate on offer replaces it
        settled = not pick_wins and rnd == rounds
        # the fact check (plan 5.3), separate from the judge: it only looks for things that aren't so
        facts = ""
        if pick_wins or settled:
            facts, _ = claude_run_(settings, conn, FACT_PROMPT
                                  .replace("<<ORIGINAL>>", orig_text)
                                  .replace("<<NOTES>>", "\n".join(f"- {n}" for n in notes))
                                  .replace("<<NEW>>", new), role=f"{role} facts", model="sonnet",
                                  max_budget_usd=0.2)
            facts = "" if facts.strip().upper().startswith("NONE") else facts.strip()
            stats["facts"] = "fail" if facts else "pass"
        if facts:
            stats["facts_text"] = facts
            return None, {"para": pi, "orig": orig_text, "result": f"pick failed the fact check ({len(cands)} passed filters)",
                          "why": f"{facts[:300]} | pick was: {new}"}, stats
        if pick_wins or settled:
            stats["accepted"] = True
            return new, {"para": pi, "orig": orig_text, "new": new,
                         "result": f"replaced ({len(cands)} passed filters, head-to-head: {letter}"
                                   f"{', settled in the last round' if settled else ''})",
                         "why": v.get("why", "")}, stats
        return None, {"para": pi, "orig": orig_text, "result": f"pick lost the head-to-head ({len(cands)} passed filters)",
                      "why": f"judge's pick was: {new}"}, stats
    kept = "the original" if is_original else "the current text (regrown earlier)"
    return None, {"para": pi, "orig": orig_text, "result": f"judge kept {kept} ({len(cands)} passed filters)",
                  "why": v.get("why", "")}, stats
