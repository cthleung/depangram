"""The restore pass (plan 6.9; built 2026-11-10 from experiments 10 and 10b).

The loop's regrown paragraphs score human and sometimes drop a fact, lose a point, go flat or lose the swearing.
Experiments 10 and 10b settled the shape: line edits by an editor don't move Pangram at this scale (what moves it is
a broken sentence and Pangram's own segmentation), the careful prompt fixed the editors' execution (zero gate rejects,
no gloss, no similes, bite restored with the author's own idioms), and the remaining failures were the director's.
So: **Opus writes the punch list** (judgment), **careful-prompt Sonnet applies it** (clerical), and **Opus is the
editor on a retry** when every Sonnet draw fails the gate or the checker reports a broken sentence. Scope is facts and
idiom-level bite. Joke restoration stays off (`restore.jokes = false`) until the author's blind read of
`analytics/exp10b jokes.md` says the restored jokes read as theirs.

    python -m engine.restore                         # every regrown paragraph in scope, no Pangram, no write
    python -m engine.restore --dry                   # the same, named as a dry run for the author to read
    python -m engine.restore --section 1 --write     # apply, score each section as posted, keep it within the noise
    python -m engine.restore --job NAME --para "s1w1 p2"     # one paragraph

What's in scope: paragraphs in the shipped picks that the loop regrew (any arm), never an untouched author
paragraph; runs the flavor pass or an earlier restore locked are left byte for byte (a paragraph whose every run is
locked is skipped); a paragraph under `min_words` is skipped, since at the 30% cap a restore on it is a rewrite by
another name.

The gate is code, never a model: experiment 10's word-level diff (insertions of at most `max_words`, substitutions
of at most two words, deletions of at most two, at most `max_change` of the words changed, sentence count unchanged,
every sentence still there in order with 70% of its words intact, locked sentences byte for byte), then the filter
stack (house words, names, numbers, quotes, keep-words, markup, assistant tells, Reddit tells, punctuation, the
connective cap, the section swear budget, restatement). A checker call (Sonnet) then says which items were applied
and whether any sentence was rewritten past a line edit or broken.

Scoring and keeping follow the fix pass: each touched section is built in memory, scored as posted before and after,
and kept only when its AI share rose by no more than `noise`. A kept section goes into the real guide, the working
copy and the bank as a "restore" entry per window, which ships like a fix and locks the restored runs the way
flavor inserts do (engine/run.py's lock reads both kinds). Only this module and the loop call Pangram; the director,
the editor and the checker never see a score.
"""
import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

from . import checks as _checks
from .checks import (QUOTE, connectives_in, house_words, invented_sighting, measured_swear_rates,
                     names_missing, names_required, new_items, punctuation_problem, reddit_tell, restated_pairs,
                     section_counts, section_filler_problem, section_swear_problem, swear_counts, tokens, words_changed)
from .fixpass import FixPass, _sents, gather
from .job import NOTES_VERSION, Bank, Job, flat, sha, style_of, window_text_of
from .settings import Settings
from .util import describe, report_file
from .wrapup import as_posted, layout, norm

RESTORE_VERSION = "restore-v1|"
LABELS = ("fact", "point", "bite", "flat", "joke")
MAX_ITEMS = 6
MAX_SUB_OLD = 2          # words one substitution may replace
MAX_SUB_NEW = 14         # and what it may replace them with (a swap plus an insertion reads as one difflib op)
MAX_DELETE = 2
MIN_INTACT = 0.70        # of each input sentence's words still there, in order
BUDGET_USD = {"director": 0.60, "editor": 0.30, "check": 0.10, "retry": 1.00}
CAPPED = re.compile(r"session limit|usage limit|hit your|limit reached", re.I)
# words the author stresses in CAPS (the voice guide: IS, REALLY, ONLY, CAN); the name check lets these through
STRESS_WORDS = {"really", "is", "not", "only", "can", "does", "do", "did", "all", "every", "never", "always", "actually",
                "literally", "very", "that", "this", "so", "too", "much", "just", "exactly", "are", "was", "will", "no",
                "yes", "any", "nothing", "everything", "way", "what", "how", "why"}

SYSTEM_DIRECTOR_TEMPLATE = ("You direct a restore pass inside an automated pipeline for <<DESCRIPTION>>. You write "
                            "punch lists only, and you output exactly what the user asks for, with no preamble.")
SYSTEM_EDITOR_TEMPLATE = ("You are the restoring editor inside an automated pipeline for <<DESCRIPTION>>. You make line "
                          "edits only, in the author's own idiom, and you output exactly what the user asks for, with no "
                          "preamble.")
# with the default description; RestorePass fills the job's own
SYSTEM_DIRECTOR = describe(SYSTEM_DIRECTOR_TEMPLATE)
SYSTEM_EDITOR = describe(SYSTEM_EDITOR_TEMPLATE)
SYSTEM_CHECK = "You are a careful checker inside an automated pipeline. Reply with the JSON asked for and nothing else."

DIRECTOR_TEMPLATE = """You direct a restore pass on one paragraph of <<DESCRIPTION>>. A small language model rewrote the paragraph from the notes below so it reads as the author's own; the rewrite is what ships. Rewrites like this sometimes drop a fact, lose a point, go flat or lose the swearing. An editor will apply your list to the REWRITE as line edits: a word, a short clause of at most <<MAX_INSERT>> words, or a one-word swap per item. The editor never sees the original, so every item has to stand on its own.

Original paragraph, sentence by sentence, with who wrote each (the author, or Claude, the AI that drafted the essay with them):
<<ORIGINAL>>

Rewrite as it ships:
<<REWRITE>>

The notes the rewrite was written from:
<<NOTES>>

Swearing room in this section: <<SWEAR_ROOM>>

Write a numbered punch list, at most <<MAX>> items, one line each, each starting with one of these labels:
- fact: a name, number, citation or verbatim phrase the rewrite dropped, in double quotes exactly as the original or the notes have it, then what it does in its sentence (say what the word is doing, "the hedge on the claim about the price", and let the editor place it; never say where it sits). A fact is the only thing you may quote. Quotation marks, a filler word and the author's own opener are not facts.
- point: a point the original makes that the rewrite lost, in your own words.
- bite: where a swear or a harder word belongs and what it does there (the contempt in the claim about the instructions), never which word and never where it sits. Only when the swearing room allows one.
- flat: a sentence that went limp, by position, and what it should carry, described and never written.<<JOKE_LABEL>>
Then one last line starting "keep:" that names what must not change: a phrase or two of the rewrite, and the sentences that are the author's own.

Rules: never write a sentence or a clause for the editor to paste. Never ask for a sentence to be split, merged, moved or rewritten; the editor can't do any of those, and the item burns a draw. Never ask for a first-hand experience, a sighting or any claim the original doesn't make ("I've tested it myself" is a made-up fact, and the gate throws it out). A rewrite sentence that is identical to an [author] sentence of the original is locked, so no item may touch it. A weaker joke is not an item.<<JOKE_RULE>> If the rewrite lost nothing worth restoring, write "1. nothing" and then the keep: line. Output the list only."""
DIRECTOR_PROMPT = describe(DIRECTOR_TEMPLATE)

JOKE_LABEL = ("\n- joke: a joke the rewrite lost: name only its target, the thing the joke is at the expense of. Never "
              "describe its shape, its wording or its mechanism, and never quote or paraphrase a [Claude] sentence: an "
              "editor pastes a described shape in as prose.")
JOKE_RULE = ""
NO_JOKE_RULE = " Jokes are off for this pass: a lost joke is not an item, so leave it."

EDITOR_PROMPT = """A small language model rewrote one paragraph of the author's essay so it reads as the author's own, and the rewrite is what ships. The rewrite dropped a few things, and a director listed them in the punch list below. Your job is to put them back as line edits, in the author's voice, without rewriting what the small model wrote.

THE AUTHOR'S VOICE, from their own style guide:

<<VOICE>>

TWO SAMPLES OF THE AUTHOR'S WRITING, for the register. Copy the habits, never the wording:

<<SAMPLES>>

THE AUTHOR'S OWN IDIOMS, pulled from their posts by frequency. These are the only phrases you may add for bite:

<<IDIOMS>>

THE RULES:
1. Apply each punch-list item by inserting a word or a short clause (at most <<MAX_INSERT>> words) or by swapping a single word. Keep every existing sentence's wording otherwise.
2. Never merge, split or reorder sentences. The sentence count stays what it is.
3. Restore bite ONLY with an idiom or a swear from the list above, or with the item's own named content (the thing the director says was lost). Never a simile, an image or a comparison of your own. If you catch yourself writing "like a ..." or "as if ...", delete it.
4. Add no gloss, no explanation and no new fact. Nothing in brackets that explains a term. No "honestly" tacked onto a clause for bite: the author hardly ever writes it, and a draw that brings one in past the section's cap is thrown out.
5. A sentence the keep: line names as the author's own stays byte for byte.
6. Put a word where the sentence takes it, not where the item happens to mention it; read the sentence back and make sure it still runs as speech.
7. If an item can't be done inside these rules, skip it and name it on the last line.

Paragraph:
<<PARAGRAPH>>

Punch list:
<<LIST>>

Output the edited paragraph on its own, with no preamble and no quotation marks around it. Then one last line in this exact form: "skipped: 3, 5 (a few words why)" or "skipped: none"."""

CHECK_PROMPT = """A punch list was applied to a paragraph as line edits (a word or a short clause changed per item). Compare the paragraph before and after and say, for each numbered item, whether the edit does what the item asks (applied), does part of it (partly) or doesn't (ignored). Then name any sentence of the "before" text that came out rewritten beyond a line edit. Then name any sentence of the "after" text that the edit broke: a word in the wrong place, a clause left dangling, grammar that no longer runs ("is really fucking just annoying"). Then, for each item that was applied or partly applied, copy the exact words the editor inserted or changed for it, as they stand in the "after" text (a few words, verbatim).

Before:
<<BEFORE>>

Punch list:
<<LIST>>

After:
<<AFTER>>

Reply with only JSON: {"applied": [item numbers], "partly": [item numbers], "ignored": [item numbers], "rewritten": ["the before sentence, if any"], "broken": ["the after sentence, if any"], "words": {"item number": "the exact words"}, "note": "one short line"}"""


# ---------------------------------------------------------------------- text helpers
def paras_of_text(t):
    return [p.strip() for p in re.split(r"\n\s*\n", t.strip()) if p.strip()]


def straighten(t):
    """Curly quotes and apostrophes to straight ones, an ellipsis glyph to three dots, double spaces collapsed. An
    instruct model writes curly quotes by habit, and the guide uses straight ones."""
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("…", "...")
    return re.sub(r"[ \t]{2,}", " ", t).strip()


def clean_output(t):
    """The editor's reply as a paragraph: code fences and a wrapping pair of quotes off, newlines inside joined."""
    t = straighten(t)
    t = re.sub(r"^```\w*\n|\n```$", "", t).strip()
    if len(t) > 2 and t[0] == '"' and t[-1] == '"' and t.count('"') == 2:
        t = t[1:-1].strip()
    return re.sub(r"\s*\n\s*", " ", t).strip()


def word_tokens(t):
    return [x for x in tokens(t) if re.match(r"\w", x)]


def keep_words(notes):
    for n in notes:
        if n.lower().startswith("keep:"):
            return re.findall(r'"([^"]+)"', n)
    return []


SKIPPED = re.compile(r"^\W*skipped(?:\s+items?)?\s*:\s*(.*)$", re.I)


def split_skipped(raw, max_item=99):
    """(paragraph, skipped line or None, skipped item numbers). The last non-empty line of the reply is the editor's
    skipped line when it starts with "skipped:"; it comes off before clean_output joins the lines. Item numbers are read
    up to the first bracket, so a reason like "(12 words)" never counts as an item."""
    lines = straighten(raw or "").strip().splitlines()
    skipped = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            m = SKIPPED.match(lines[i].strip())
            if m:
                skipped = lines[i].strip()
                del lines[i]
            break
    nums = []
    if skipped:
        body = SKIPPED.match(skipped).group(1)
        head = re.split(r"[(\[]", body, 1)[0]
        if not re.match(r"^\s*none\b", head, re.I):
            nums = [int(x) for x in re.findall(r"\d+", head) if 1 <= int(x) <= max_item]
    return clean_output("\n".join(lines)), skipped, nums


def changed_spans(before, after):
    """The after-side words of every stretch the word-level diff changed."""
    a, b = tokens(before), tokens(after)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        new = re.sub(r"\s+([^\w\s])", r"\1", " ".join(b[j1:j2])) or "(cut)"
        old = re.sub(r"\s+([^\w\s])", r"\1", " ".join(a[i1:i2]))
        out.append(f"{old} -> {new}" if old and op == "replace" else new)
    return out


def bold_changes(before, after):
    """`after` with every changed stretch in bold, for the report."""
    a = tokens(before)
    spans = [(m.start(), m.end()) for m in re.finditer(r"[\w’']+(?:[-’'][\w’']+)*|[^\w\s]", after)]
    b = [after[s:e] for s, e in spans]
    marks = [False] * len(b)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "equal":
            for j in range(j1, j2):
                marks[j] = True
    out, pos, open_ = [], 0, False
    for j, (s, e) in enumerate(spans):
        if marks[j] and not open_:
            out.append(after[pos:s] + "**")
            open_, pos = True, s
        elif not marks[j] and open_:
            out.append(after[pos:spans[j - 1][1]] + "**")
            open_, pos = False, spans[j - 1][1]
    out.append(after[pos:] + ("**" if open_ else ""))
    return "".join(out)


# ---------------------------------------------------------------------- the punch list
def parse_punch_list(text, labels=LABELS, max_items=MAX_ITEMS):
    """[{"n", "kind", "text"}] plus the keep line, from the director's plain text. Items are numbered lines that
    start with one of `labels`; "nothing" means an empty list; the list is cut at `max_items`. Returns (items, keep).
    An item is renumbered in list order, so the editor and the checker count the same way."""
    items, keep = [], ""
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-*").strip()
        if not line:
            continue
        m = re.match(r"^(?:\d+[.)]\s*)?keep\s*:\s*(.*)$", line, re.I)
        if m:
            keep = m.group(1).strip()
            continue
        m = re.match(r"^(\d+)[.)]\s*(.*)$", line)
        if not m:
            continue
        body = m.group(2).strip()
        if re.match(r"^nothing\b", body, re.I):
            continue
        k = re.match(r"^(" + "|".join(labels) + r")\s*:\s*(.*)$", body, re.I)
        if not k:
            continue
        items.append({"n": len(items) + 1, "kind": k.group(1).lower(), "text": k.group(2).strip()})
    return items[:max_items], keep


def fact_strings(items):
    """The quoted facts on the fact: items, straight-quoted: the first quoted string of each item (the director quotes
    the text being replaced too, "3.5" after "version 3.5")."""
    out = []
    for it in items:
        if it["kind"] == "fact":
            qs = [q.strip() for q in re.findall(r'"([^"]+)"', straighten(it["text"])) if q.strip()]
            out += qs[:1]
    return out


def fact_present(fact, text):
    low, f = straighten(text).lower(), fact.lower()
    if f in low:
        return True
    # "(OpenAI, 2023)" counts when its name and year both stand; "section 2.3.1" when the number does
    m = re.match(r"\(?([^,()]+?),?\s*(\d{4}[a-z]?)\)?$", f)
    if m and m.group(1).strip() in low and m.group(2) in low:
        return True
    return False


def format_list(items):
    return "\n".join(f"{it['n']}. {it['kind']}: {it['text']}" for it in items) or "(empty)"


# ---------------------------------------------------------------------- the diff gate (code only)
def diff_problem(before, after, max_insert=12, max_change=0.30):
    """The first rule the word-level diff breaks, or None: insertions of at most `max_insert` words, substitutions of
    at most MAX_SUB_OLD words for at most MAX_SUB_NEW, deletions of at most MAX_DELETE, and in all at most
    `max_change` of the input's words changed."""
    a, b = word_tokens(before), word_tokens(after)
    if a == b:
        return None                    # punctuation only: the filters below the diff judge it
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    changed = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        old, new = i2 - i1, j2 - j1
        changed += max(old, new)
        if op == "insert" and new > max_insert:
            return f"an insertion of {new} words (the limit is {max_insert})"
        if op == "delete" and old > MAX_DELETE:
            return f"a deletion of {old} words (the limit is {MAX_DELETE})"
        if op == "replace" and (old > MAX_SUB_OLD or new > MAX_SUB_NEW):
            return f"a rewrite of {old} words into {new}"
    if changed > max_change * len(a):
        return f"{changed} of {len(a)} words changed (the cap is {max_change:.0%})"
    return None


def sentence_problem(before, after):
    """Sentence count unchanged, and every input sentence still there in order with at least MIN_INTACT of its
    words intact."""
    sa, sb = _sents(before), _sents(after)
    if len(sa) != len(sb):
        return f"{len(sa)} sentences became {len(sb)}"
    for i, (x, y) in enumerate(zip(sa, sb), 1):
        tx, ty = word_tokens(x), word_tokens(y)
        kept = sum(m.size for m in difflib.SequenceMatcher(None, tx, ty, autojunk=False).get_matching_blocks())
        if tx and kept / len(tx) < MIN_INTACT:
            return f"sentence {i} rewritten ({kept} of {len(tx)} words kept)"
    return None


def _names(text):
    from engine.draws import names
    return names(text)


def _numbers(text):
    from engine.draws import numbers
    return numbers(text)


def gate(before, after, *, original="", facts=(), notes=(), locked=(), rest=None, rates=None, multiplier=None,
         max_insert=12, max_change=0.30, filler_cap=None, filler_words=None):
    """Why `after` can't stand as a line-edited `before`, or None. `before` is the regrown paragraph, `original`
    the paragraph it replaced (names and numbers from either are no additions), `facts` the punch list's quoted
    facts (allowed additions), `notes` the paragraph's notes (keep-words), `locked` sentences that must stay byte for
    byte (the author's own standing unchanged in the rewrite, and runs an earlier pass locked), and `rest`, `rates`,
    `multiplier` the section's swear budget as checks.section_swear_problem takes it, and `filler_cap`, `filler_words`
    the filler-tag budget (checks.section_filler_problem) on the same `rest`."""
    from engine.draws import ASSISTANT_TELL, COSTUME_OPENER, MARKUP
    after = straighten(after)
    if not after:
        return "empty"
    if after == straighten(before):
        return "no change"
    if COSTUME_OPENER.search(after.split("\n")[0]):
        return "assistant costume"
    why = diff_problem(before, after, max_insert, max_change)
    if why:
        return why
    why = sentence_problem(before, after)
    if why:
        return why
    for s in locked:
        if s and s not in after:
            return "touched a locked sentence"
    if MARKUP.search(after) and not MARKUP.search(before):
        return "markup"
    # a tell the original paragraph carries is the author's own phrase, so only a tell new to both counts
    if new_items(before + " " + original, after, ASSISTANT_TELL.findall):
        return "assistant tell"
    if reddit_tell(after) and not reddit_tell(before):
        return "reddit tell"
    # a first-person sighting neither the rewrite nor the original has ("I've tested it myself", the dry run of
    # 2026-11-10) is a made-up claim, as it is in the loop's own stack
    seen = invented_sighting(after, before + " " + original)
    if seen:
        return f"adds a first-person sighting: {seen}"
    # every capitalized word of the regrown text, the original and the listed facts is a known name, sentence
    # openers included (an insertion can move an opener mid-sentence, where names() then sees it)
    allowed_text = before + " " + original + " " + " ".join(facts)
    known = set(re.findall(r"\b[A-Z][a-zA-Z0-9-]+", allowed_text))
    # a word stressed in CAPS ("REALLY", "IS") is the author's spoken emphasis (the voice guide's CAPS stress), not a
    # name (the dry run of 2026-11-10 threw out "REALLY does help"): a plain stress word, or a word the paragraph
    # already has in lowercase, passes; an acronym the paragraph doesn't know ("GPT") is still a new name
    low_words = set(re.findall(r"[a-z']+", allowed_text.lower()))
    new_names = {n for n in _names(after) - known
                 if not (re.fullmatch(r"[A-Z]{2,}", n) and (n.lower() in STRESS_WORDS or n.lower() in low_words))}
    if new_names:
        return "adds a name: " + ", ".join(sorted(new_names))
    if not _numbers(after) <= _numbers(allowed_text):
        return "adds a number: " + ", ".join(sorted(_numbers(after) - _numbers(allowed_text)))
    if not _numbers(before) <= _numbers(after):
        return "number lost: " + ", ".join(sorted(_numbers(before) - _numbers(after)))
    miss = names_missing(after, names_required(before, notes))
    if miss:
        return "name lost: " + ", ".join(miss)
    for q in QUOTE.findall(before):
        if len(q.split()) >= 3 and q not in after:
            return f"quote lost: {q[:30]}"
    low_b, low_a = before.lower(), after.lower()
    for k in keep_words(notes):
        if k.lower() in low_b and k.lower() not in low_a:
            return f"keep lost: {k}"
    slip = punctuation_problem(after, before)
    if slip:
        return "punctuation: " + slip
    if new_items(before, after, connectives_in) and len(connectives_in(after)) > _checks.CONNECTIVE_CAP:
        return "connective glue"
    if new_items(before, after, house_words):
        return "an em dash or a banned word"
    # a restatement the edit made: a neighbouring pair that now says the same thing where the input's didn't
    had = {(i, j) for i, j, _ in restated_pairs(_sents(before))}
    if any((i, j) not in had for i, j, _ in restated_pairs(_sents(after))):
        return "says it twice"
    swear = section_swear_problem(after, before, rest, rates, multiplier) if rest and rates else None
    if swear:
        return swear
    return section_filler_problem(after, before, rest, filler_cap, filler_words) if rest else None


# ---------------------------------------------------------------------- the runs under the paragraph
def _spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r"[\w’']+(?:[-’'][\w’']+)*|[^\w\s]", text)]


def split_runs(before_runs, after):
    """The edited paragraph cut back into its runs, or None when it won't cut cleanly. `before_runs` are the run
    texts the paragraph was made of ("" for a cut run); `after` is the edited paragraph. Each run boundary is a
    token index in the joined input, carried through the word-level diff to the edited text; an insertion at a
    boundary goes with the run that follows it. The cut is accepted only when the runs join back to the edited
    text and every run keeps its sentence count."""
    before = " ".join(t for t in before_runs if t)
    a_sp, b_sp = _spans(before), _spans(after)
    a = [before[s:e] for s, e in a_sp]
    b = [after[s:e] for s, e in b_sp]
    # run boundaries as token indices in `a`
    bounds, n = [], 0
    for t in before_runs:
        n += len(_spans(t)) if t else 0
        bounds.append(n)
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()

    def to_after(p):
        if p >= len(a):
            return len(b)
        for op, i1, i2, j1, j2 in ops:
            if i1 <= p < i2 or (op != "equal" and p == i1):
                return j1 + (p - i1) if op == "equal" else j1
            if p == i2 and op == "equal":
                return j2
        return len(b)
    out, start = [], 0
    for t, p in zip(before_runs, bounds):
        if not t:
            out.append("")
            continue
        end = to_after(p)
        if end <= start:
            return None
        s_char = b_sp[start][0]
        e_char = b_sp[end - 1][1]
        out.append(after[s_char:e_char].strip())
        start = end
    if norm(" ".join(x for x in out if x)) != norm(after):
        return None
    for old, new in zip(before_runs, out):
        if old and len(_sents(old)) != len(_sents(new)):
            return None
    return out


def locked_inserts(bank, wid):
    """{(pi, ri): (the run's sentences before the pass, the run's sentences as the pass left it)} from the bank's
    flavor entries (engine/flavor.py: the insert appended) and restore entries (this module: the run line-edited).
    engine/run.py's lock reads this, so a run standing at the pass's text is never a target, and a run standing at
    the text the pass started from gets the pass's text back when the window resumes."""
    out = {}
    for e in bank.of(wid):
        kind = e.get("kind")
        runs = e.get("runs") or []
        if kind == "flavor":
            for ins in e.get("inserts") or []:
                pi, ri, sent = ins.get("para"), ins.get("run"), ins.get("sentence")
                if pi is None or ri is None or not sent or pi >= len(runs) or ri >= len(runs[pi]):
                    continue
                r = runs[pi][ri]
                if r and r[-1] == sent:
                    out[(pi, ri)] = (r[:-1], list(r))
        elif kind == "restore":
            for rs in e.get("restores") or []:
                pi, ri = rs.get("para"), rs.get("run")
                before, after = rs.get("before"), rs.get("after")
                if pi is None or ri is None or not before or not after:
                    continue
                out[(pi, ri)] = (list(before), list(after))
    return out


# ---------------------------------------------------------------------- the director's inputs
def provenance(w, pi, run_texts):
    """[(tag, sentence)] for the original paragraph's sentences, each tagged author or Claude from the loop's run
    flags, plus the author sentences that stand unchanged in the regrown paragraph (locked)."""
    out, locked = [], []
    for ri, r in enumerate(w["runs"][pi]):
        orig = " ".join(w["original"][pi][ri])
        tag = "author" if r.get("author") or not r.get("claude") else "Claude"
        for s in w["original"][pi][ri]:
            out.append((tag, s))
        if tag == "author" and ri < len(run_texts) and norm(run_texts[ri]) == norm(orig):
            locked += list(w["original"][pi][ri])
    return out, locked


def swear_room(section_text, para, rates, multiplier=None, filler_words=None):
    """(rest, a line for the director): the section's swear counts with this paragraph left out, and how many more
    swears the author's rate allows the section as posted. `multiplier` and `filler_words` default to the rules in
    use (engine/checks.py)."""
    multiplier = _checks.SWEAR_MULTIPLIER if multiplier is None else multiplier
    counts = section_counts(section_text, filler_words)
    w, s, st = counts[:3]
    pc = section_counts(para, filler_words)
    rest = tuple(a - b for a, b in zip(counts, pc)) if norm(para) in norm(section_text) else counts
    room_all = int(rates["all"] * w * multiplier) - s
    room_strong = int(rates["strong"] * w * multiplier) - st
    line = (f"the section as posted carries {s} swears ({st} of the fuck family) in {w} words; the author's own rate "
            f"allows {max(0, room_all)} more swear(s) in all and {max(0, room_strong)} more of the fuck family")
    if room_all <= 0:
        line += ". No room: write no bite: item that asks for a swear"
    return rest, line


class GiveUp(Exception):
    """Claude is capped and stayed capped, or the pass's own Claude ceiling is reached."""


# ---------------------------------------------------------------------- one pass
class RestorePass(FixPass):
    """The fix pass's scoring, apply and bookkeeping with the restore pass's directing, editing and checking on top."""

    def __init__(self, *, job, bank, settings, conn, claude=None, pangram=None, budget=None, purpose="loop",
                 max_usd=None, claude_usd=None, jokes=None, log=print, wait_seconds=600):
        cfg = settings.config.get("restore", {}) if settings is not None else {}
        super().__init__(job=job, bank=bank, settings=settings, conn=conn, claude=claude, pangram=pangram, budget=budget,
                         model=cfg.get("editor_model", "sonnet"), noise=cfg.get("noise", 0.05),
                         max_usd=max_usd if max_usd is not None else cfg.get("max_pangram_usd", 1.0), purpose=purpose,
                         log=log)
        self.director_model = str(cfg.get("director_model", "opus"))
        self.editor_model = str(cfg.get("editor_model", "sonnet"))
        self.retry_model = str(cfg.get("retry_model", "opus"))
        self.jokes = bool(cfg.get("jokes", False)) if jokes is None else bool(jokes)
        self.max_words = int(cfg.get("max_words", 12))
        self.max_change = float(cfg.get("max_change", 0.30))
        self.min_words = int(cfg.get("min_words", 25))
        self.draws = int(cfg.get("draws", 2))
        self.claude_usd = float(claude_usd if claude_usd is not None else cfg.get("max_claude_usd", 3.0))
        self.wait_seconds = wait_seconds
        self.claude_spent = 0.0
        self.cache_path = (settings.runs_dir / job.name / "restore cache.json") if settings is not None else None
        self.idioms_path = (settings.runs_dir / job.name / "idioms cache.json") if settings is not None else None
        self.voice = self.samples = self.idiom_text = None
        self.swear_rates = None
        self.swear_mult, self.filler_cap, self.filler_words = None, None, None      # setup() reads the budgets
        self.labels = tuple(x for x in LABELS if x != "joke" or self.jokes)

    # ---- the fixed parts
    def setup(self, vm=None):
        from .idioms import cached_idioms, idiom_block, voice_material
        vm = vm if vm is not None else self.job.verdict_map()
        self.swear_rates = measured_swear_rates(vm)
        self.swear_mult, self.filler_cap, self.filler_words = _checks.budgets(self.settings.config, warn=self.log)
        self.voice, self.samples = voice_material()
        self.idiom_text = idiom_block(cached_idioms(self.idioms_path, log=self.log))

    # ---- Claude with the caps in view
    def call(self, prompt, role, model, budget_usd, system):
        """One call, the prompt always on stdin. A capped call waits `wait_seconds` and tries twice more; the loop's
        own capper (a ClaudeCapReached) and the pass's Claude ceiling both end the pass with GiveUp."""
        from .budget import ClaudeCapReached
        from .claude_call import ClaudeCallError
        if self.claude_spent >= self.claude_usd:
            raise GiveUp(f"the pass's own Claude ceiling (${self.claude_usd:.2f}) is reached")
        for attempt in range(3):
            try:
                if self.budget is not None:
                    self.budget.claude_check(f"restore {role}", budget_usd)
                text, meta = self.claude(self.settings, self.conn, prompt, role=f"restore {role}", model=model,
                                         max_budget_usd=budget_usd, system=system, stdin_over=0)
                self.claude_spent += float(meta.get("cost_usd") or 0.0)
                return text, meta
            except ClaudeCapReached as e:
                raise GiveUp(f"the capper said no: {e}") from None
            except ClaudeCallError as e:
                if not CAPPED.search(str(e)):
                    raise
                if attempt == 2:
                    raise GiveUp(f"Claude capped and stayed capped: {str(e)[:160]}") from None
                self.log(f"  Claude capped ({str(e)[:80]}); waiting {self.wait_seconds // 60} minutes ({attempt + 1} of 3)")
                time.sleep(self.wait_seconds)

    # ---- scope
    def scope(self, rows, lay, real_text):
        """{label: target} for every row the pass may work on; the others get `fixable=False` and a reason."""
        out, section_text, rests = {}, {}, {}
        for row in rows:
            if not row["fixable"]:
                continue
            w, pi, wid = lay[row["wid"]], row["pi"], row["wid"]
            if norm(row["text"]) == norm(row["original"]):
                row.update(fixable=False, why="the loop didn't regrow this paragraph")
                continue
            if len(row["text"].split()) < self.min_words:
                row.update(fixable=False, why=f"under {self.min_words} words: at the {self.max_change:.0%} cap a restore"
                                              " is a rewrite by another name")
                continue
            locks = locked_inserts(self.bank, wid)
            cur_lists = self.window_runs(lay, rows, {}, wid)[pi]
            locked_runs = [ri for ri, t in enumerate(row["runs"]) if t and (pi, ri) in locks and cur_lists[ri] == locks[(pi, ri)][1]]
            if locked_runs and all(not t or ri in locked_runs for ri, t in enumerate(row["runs"])):
                row.update(fixable=False, why="every run is locked by the flavor pass or an earlier restore")
                continue
            prov, locked = provenance(w, pi, row["runs"])
            for ri in locked_runs:
                locked += list(cur_lists[ri])
            n = row["section"]
            if n not in section_text:
                section_text[n] = as_posted(real_text, [n], style=style_of(self.job)).get(n, "")
            rest, room = swear_room(section_text[n], row["text"], self.swear_rates, self.swear_mult, getattr(self, 'filler_words', None))
            notes = self.notes_for(w, pi)
            out[row["label"]] = {"label": row["label"], "wid": wid, "pi": pi, "section": n, "original": row["original"],
                                 "regrown": row["text"], "runs": list(row["runs"]), "run_lists": cur_lists, "notes": notes,
                                 "provenance": prov, "locked": locked, "locked_runs": locked_runs, "rest": rest,
                                 "swear_room": room}
        return out

    def notes_for(self, w, pi):
        para = " ".join(s for r in w["original"][pi] for s in r)
        return self.job._cache().get(sha(NOTES_VERSION + para)) or []

    # ---- the director
    def director_key(self, t):
        return sha(f"{RESTORE_VERSION}director|{self.director_model}|jokes={int(self.jokes)}|{NOTES_VERSION}|"
                   f"{t['original']}\x1f{t['regrown']}")

    def direct(self, t):
        """The punch list for one paragraph: Opus, cached by the sha of (original, regrown, notes version). Sets
        t["punch_text"], t["items"], t["keep"], t["facts"]."""
        cache = self._cache()
        key = self.director_key(t)
        if key not in cache:
            prompt = (describe(DIRECTOR_TEMPLATE, self.description()).replace("<<ORIGINAL>>", "\n".join(f"{i + 1}. [{tag}] {x}" for i, (tag, x) in enumerate(t["provenance"])))
                      .replace("<<REWRITE>>", t["regrown"])
                      .replace("<<NOTES>>", "\n".join(f"- {x}" for x in t["notes"]) or "(none)")
                      .replace("<<SWEAR_ROOM>>", t["swear_room"]).replace("<<MAX>>", str(MAX_ITEMS))
                      .replace("<<MAX_INSERT>>", str(self.max_words))
                      .replace("<<JOKE_LABEL>>", JOKE_LABEL if self.jokes else "")
                      .replace("<<JOKE_RULE>>", JOKE_RULE if self.jokes else NO_JOKE_RULE))
            text, meta = self.call(prompt, "director", self.director_model, BUDGET_USD["director"],
                                   describe(SYSTEM_DIRECTOR_TEMPLATE, self.description()))
            cache[key] = {"text": text, "cost_usd": meta.get("cost_usd")}
            self._save(cache)
        t["punch_text"] = cache[key]["text"]
        t["items"], t["keep"] = parse_punch_list(t["punch_text"], self.labels)
        t["facts"] = fact_strings(t["items"])
        return t["items"]

    # ---- the editor and the checker
    def edit_prompt(self, t):
        return (EDITOR_PROMPT.replace("<<VOICE>>", self.voice).replace("<<SAMPLES>>", self.samples)
                .replace("<<IDIOMS>>", self.idiom_text).replace("<<MAX_INSERT>>", str(self.max_words))
                .replace("<<PARAGRAPH>>", t["regrown"]).replace("<<LIST>>", format_list(t["items"])))

    def gate(self, t, cand):
        why = gate(t["regrown"], cand, original=t["original"], facts=t["facts"], notes=t["notes"], locked=t["locked"],
                   rest=t["rest"], rates=self.swear_rates, multiplier=self.swear_mult, max_insert=self.max_words,
                   max_change=self.max_change, filler_cap=getattr(self, 'filler_cap', None),
                   filler_words=getattr(self, 'filler_words', None))
        if why is None and split_runs(t["runs"], cand) is None:
            why = "the edit doesn't cut back into the paragraph's runs"
        return why

    def check(self, t, after):
        """Code checks every fact: item; Sonnet says which items were applied, whether a sentence was rewritten past a
        line edit, and whether one came out broken. Cached by the before and after texts."""
        cache = self._cache()
        key = sha(f"{RESTORE_VERSION}check|{t['regrown']}\x1f{after}\x1f{format_list(t['items'])}")
        present = [f for f in t["facts"] if fact_present(f, after)]
        out = {"facts_present": len(present), "facts_total": len(t["facts"]),
               "facts_missing": [f for f in t["facts"] if f not in present],
               "applied": [], "partly": [], "ignored": [], "rewritten": [], "broken": [], "words": {}, "note": ""}
        if key not in cache:
            text, _ = self.call(CHECK_PROMPT.replace("<<BEFORE>>", t["regrown"]).replace("<<AFTER>>", after)
                                .replace("<<LIST>>", format_list(t["items"])), "check", "sonnet", BUDGET_USD["check"],
                                SYSTEM_CHECK)
            cache[key] = text
            self._save(cache)
        m = re.search(r"\{.*\}", cache[key] or "", re.S)
        try:
            v = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            v = {"note": "(the check's JSON didn't parse)"}
        for k in ("applied", "partly", "ignored", "rewritten", "broken"):
            val = v.get(k) or []
            out[k] = [x for x in val if isinstance(x, (int, str))] if isinstance(val, list) else []
        words = v.get("words") or {}
        if isinstance(words, dict):
            for k, w in words.items():
                w = straighten(str(w))
                out["words"][str(k)] = w if w and w.lower() in straighten(after).lower() else f"(not found verbatim) {w}"
        out["note"] = str(v.get("note") or "")
        return out

    def draw(self, t, arm, k):
        """One editor draw through the gate: (candidate, draw record). `arm` is "editor" (Sonnet) or "retry" (Opus)."""
        model = self.editor_model if arm == "editor" else self.retry_model
        cache = self._cache()
        key = sha(f"{RESTORE_VERSION}draw|{arm}|{model}|{k}|{t['regrown']}\x1f{format_list(t['items'])}")
        if key not in cache:
            t0 = time.time()
            raw, meta = self.call(self.edit_prompt(t), "editor" if arm == "editor" else "editor retry", model,
                                  BUDGET_USD[arm], describe(SYSTEM_EDITOR_TEMPLATE, self.description()))
            cache[key] = {"raw": raw, "cost_usd": meta.get("cost_usd"), "seconds": round(time.time() - t0, 1)}
            self._save(cache)
        raw = cache[key]["raw"]
        cand, skipped_line, skipped = split_skipped(raw, len(t["items"]))
        why = self.gate(t, cand)
        if why == "no change" and skipped and len(skipped) >= len(t["items"]):
            why = "declined every item (skipped: " + ", ".join(map(str, skipped)) + ")"
        rec = {"arm": arm, "draw": k + 1, "text": cand, "why": why, "skipped_line": skipped_line, "skipped": skipped,
               "cost_usd": cache[key].get("cost_usd"), "words_changed": words_changed(t["regrown"], cand)}
        return (cand if why is None else None), rec

    def edit(self, t):
        """Sonnet draws until one passes the gate and the checker reports no broken or rewritten sentence; when every
        draw fails the gate, or the checker reports a break, Opus draws once. Sets t["picked"], t["arm"],
        t["draws"], t["check"], t["why"]."""
        t.update(picked=None, arm=None, draws=[], check=None, why=None)
        if not t["items"]:
            t["why"] = "nothing on the list"
            return None
        need_retry, broke = True, None
        for k in range(self.draws):
            cand, rec = self.draw(t, "editor", k)
            t["draws"].append(rec)
            if rec["why"] and rec["why"].startswith("declined"):
                t["why"] = rec["why"]
                return None
            if cand is None:
                continue
            chk = self.check(t, cand)
            rec["check"] = chk
            if chk["broken"] or chk["rewritten"]:
                broke = chk
                break
            t.update(picked=cand, arm="editor", check=chk)
            need_retry = False
            break
        if need_retry:
            reason = "the checker reported a broken sentence" if broke else "every Sonnet draw failed the gate"
            self.log(f"  {t['label']}: {reason}; Opus draws once")
            cand, rec = self.draw(t, "retry", 0)
            t["draws"].append(rec)
            if cand is not None:
                chk = self.check(t, cand)
                rec["check"] = chk
                if chk["broken"] or chk["rewritten"]:
                    t["why"] = f"{reason}, and Opus's draw did too ({'; '.join(map(str, chk['broken'] + chk['rewritten']))[:120]})"
                else:
                    t.update(picked=cand, arm="retry", check=chk)
            else:
                t["why"] = f"{reason}, and Opus's draw failed the gate ({rec['why']})"
        return t["picked"]

    # ---- one pass
    def propose(self, rows, lay, real_text):
        """Direct, edit and check every paragraph in scope. Returns (results in the fix pass's shape, {label: target}),
        so apply_sections and window_runs work unchanged."""
        targets = self.scope(rows, lay, real_text)
        results = {}
        for label, t in targets.items():
            try:
                items = self.direct(t)
                self.log(f"  {label}: {len(items)} item(s)" + (f", {len(t['facts'])} fact string(s)" if t["facts"] else ""))
                picked = self.edit(t)
            except GiveUp as e:
                t.update(picked=None, why=f"stopped: {e}")
                self.log(f"  {label}: stopped ({e})")
                for other in targets.values():
                    if "picked" not in other:
                        other.update(picked=None, arm=None, draws=[], check=None, why="not reached: " + str(e)[:80])
                break
            if not picked:
                self.log(f"  {label}: no restore ({t['why']})")
                results[label] = {"runs": list(t["runs"]), "accepted": [],
                                  "rejected": [{"sentence": d["text"], "reason": d["why"] or "checker", "category": "restore",
                                                "by": d["arm"]} for d in t["draws"]]}
                continue
            new_runs = split_runs(t["runs"], picked)
            acc = []
            for ri, (old, new) in enumerate(zip(t["runs"], new_runs)):
                if old and norm(old) != norm(new):
                    acc.append({"run": ri, "para": t["pi"], "sentence": old, "replacement": new, "category": "restore",
                                "by": t["arm"], "why": format_list(t["items"])})
            results[label] = {"runs": new_runs, "accepted": acc,
                              "rejected": [{"sentence": d["text"], "reason": d["why"] or "checker", "category": "restore",
                                            "by": d["arm"]} for d in t["draws"] if d["text"] != picked]}
            self.log(f"  {label}: restored by the {t['arm']} arm, {words_changed(t['regrown'], picked)} token(s) changed")
        return results, targets

    def restored_runs(self, lay, rows, results, wid):
        """The window's full runs with each restored run's sentence list replaced (the run shape holds)."""
        runs = self.window_runs(lay, rows, results, wid)
        for row in rows:
            if row["wid"] != wid:
                continue
            for a in results.get(row["label"], {}).get("accepted", []):
                runs[row["pi"]][a["run"]] = _sents(a["replacement"]) or [a["replacement"]]
        return runs

    def bank_entries(self, lay, rows, results, targets, sections_out, kept, stamp):
        """One "restore" entry per window in a kept section; returns {wid: (original flats, new flats)} for assemble."""
        best = {}
        for wid in sorted({r["wid"] for r in rows if r["section"] in kept}):
            restores = []
            for r in rows:
                if r["wid"] != wid:
                    continue
                t = targets.get(r["label"])
                for a in results.get(r["label"], {}).get("accepted", []):
                    restores.append({"para": r["pi"], "run": a["run"], "before": list(t["run_lists"][a["run"]]),
                                     "after": _sents(a["replacement"]) or [a["replacement"]], "arm": t["arm"],
                                     "items": [f"{it['kind']}: {it['text']}" for it in t["items"]],
                                     "diff": changed_spans(a["sentence"], a["replacement"]),
                                     "check": {k: t["check"].get(k) for k in ("applied", "partly", "ignored", "note")} if t.get("check") else None})
            if not restores:
                continue
            runs = self.restored_runs(lay, rows, results, wid)
            w = lay[wid]
            n = w["section"]
            self.bank.add({"kind": "restore", "window": wid, "time": stamp, "text": window_text_of(runs, w["heading"]),
                           "runs": runs, "ai": None, "section": n, "arm": "restore",
                           "section_ai": sections_out[n]["after"], "section_ai_before": sections_out[n]["before"],
                           "restores": restores})
            best[wid] = ([flat(p) for p in w["original"]], [flat(p) for p in runs])
        return best


# ---------------------------------------------------------------------- the report
def _clean(text):
    """Free text from a model, with any house word blanked, before it goes into a report in the repo."""
    out = str(text or "")
    for h in house_words(out):
        out = out.replace(h, "[house word]")
    return out


def report_md(when, dry, rows, results, targets, sections_out, rp, claude_spend=None):
    L = [f"# Restore pass, {when}{' (dry run)' if dry else ''}", "",
         f"Director `{rp.director_model}` writes the punch list, editor `{rp.editor_model}` applies it as line edits "
         f"(insertions of at most {rp.max_words} words, at most {rp.max_change:.0%} of a paragraph changed), "
         f"`{rp.retry_model}` edits on a retry; a section keeps its restores when its AI share as posted rises by no "
         f"more than {rp.noise:.2f}. Jokes {'on' if rp.jokes else 'off'}; paragraphs under {rp.min_words} words skipped."
         f" Pangram purpose `{rp.purpose}`.", ""]
    if sections_out:
        L += ["## Sections", "", "| Section | Before | After | Result |", "| --- | --- | --- | --- |"]
        for n, s in sorted(sections_out.items()):
            fmt = lambda v: "not scored" if v is None else f"{v:.2f}"  # noqa: E731
            L.append(f"| {n} | {fmt(s.get('before'))} | {fmt(s.get('after'))} | {s.get('result', '')} |")
        L.append("")
    done = [t for t in targets.values() if t.get("picked")]
    L += [f"## Restored ({len(done)})", ""]
    for t in done:
        L += [f"### {t['label']} (section {t['section']}, {t['arm']} arm)", "", "Punch list:", ""]
        L += [f"- {_clean(it['kind'])}: {_clean(it['text'])}" for it in t["items"]]
        if t.get("keep"):
            L.append(f"- keep: {_clean(t['keep'])}")
        L += ["", "Accepted text, changes in bold:", "", _clean(bold_changes(t["regrown"], t["picked"])), ""]
        chk = t.get("check") or {}
        L += ["Checker: applied " + (", ".join(map(str, chk.get("applied", []))) or "none")
              + (", partly " + ", ".join(map(str, chk["partly"])) if chk.get("partly") else "")
              + (", ignored " + ", ".join(map(str, chk["ignored"])) if chk.get("ignored") else "")
              + (f"; facts present {chk['facts_present']} of {chk['facts_total']}" if chk.get("facts_total") else "")
              + (f". {_clean(chk['note'])}" if chk.get("note") else "")]
        skipped = next((d["skipped_line"] for d in reversed(t["draws"]) if d["text"] == t["picked"] and d["skipped_line"]), None)
        if skipped:
            L.append(f"Editor's skip line: {_clean(skipped)}")
        rej = [d for d in t["draws"] if d["text"] != t["picked"]]
        if rej:
            L.append("Earlier draws: " + "; ".join(f"{d['arm']} {d['draw']} ({_clean(d['why'] or 'checker: ' + '; '.join(map(str, (d.get('check') or {}).get('broken', []) + (d.get('check') or {}).get('rewritten', []))))})" for d in rej))
        L.append("")
    missed = [t for t in targets.values() if not t.get("picked")]
    L += [f"## Not restored ({len(missed)})", ""]
    for t in missed:
        L += [f"### {t['label']} (section {t['section']})", "", f"Why: {_clean(t.get('why') or '')}", ""]
        if t.get("items"):
            L += ["Punch list:", ""] + [f"- {_clean(it['kind'])}: {_clean(it['text'])}" for it in t["items"]] + [""]
        for d in t.get("draws", []):
            chk = d.get("check") or {}
            L += [f"- {d['arm']} draw {d['draw']}: {_clean(d['why'] or 'passed the gate')}"
                  + (f"; checker: broken {chk['broken']}" if chk.get("broken") else "")
                  + (f"; rewritten {chk['rewritten']}" if chk.get("rewritten") else ""),
                  f"  - text: {_clean(d['text'])}"]
        L.append("")
    skipped = [r for r in rows if not r["fixable"] and r.get("asked", True)]
    if skipped:
        L += [f"## Not looked at ({len(skipped)})", ""]
        L += [f"- {r['label']}: {r['why']}" for r in skipped]
    if claude_spend:
        L += ["", "## Claude spend today (every run of the pass, this one included)", "", "| Role | Model | Calls | USD |", "| --- | --- | --- | --- |"]
        L += [f"| {r['role']} | {r['model']} | {r['calls']} | {r['usd']:.2f} |" for r in claude_spend]
        L.append(f"| all | | {sum(r['calls'] for r in claude_spend)} | {sum(r['usd'] for r in claude_spend):.2f} |")
    return "\n".join(L) + "\n"


def claude_spend_since(conn, t_start):
    """The restore roles' Claude calls since `t_start` (the report passes the start of the day, so a rerun that paid
    nothing still shows what the day's runs cost)."""
    rows = conn.execute("SELECT role, model, count(*), coalesce(sum(cost_usd), 0) FROM claude_calls WHERE role LIKE "
                        "'%restore %' AND ts >= ? GROUP BY role, model", (t_start,)).fetchall()
    return [{"role": r[0], "model": r[1], "calls": r[2], "usd": round(r[3], 2)} for r in rows]


# ---------------------------------------------------------------------- the wrap-up hook (engine/run.py's finish)
def run_for_loop(loop):
    """The loop's wrap-up call, behind `restore.run_at_wrapup` (default false). Skipped in --smoke and --window
    runs; a placeholder at zero spend in a rehearsal; else the pass over the sections in scope through Loop.claude
    and the loop's Pangram client, with the real guide written only when the loop's write_back is on. Sets
    loop.restore_result for the digest."""
    cfg = loop.settings.config.get("restore", {})
    if not cfg.get("run_at_wrapup", False):
        loop.restore_result = {"skipped": "run_at_wrapup is off"}
        return loop.restore_result
    if loop.args.smoke or loop.args.window or not loop.whole_sections():
        loop.restore_result = {"skipped": "a smoke or partial run"}
        return loop.restore_result
    if loop.rehearsal:
        loop.log("restore pass: rehearsal, the call is a placeholder at zero spend")
        loop.restore_result = {"placeholder": True, "skipped": "rehearsal placeholder"}
        return loop.restore_result
    from .budget import CapReached
    from .claude_call import ClaudeCallError
    from .pangram import CapReached as ClientCap
    real = loop.real_file
    secs = loop.whole_sections()
    rp = RestorePass(job=loop.job, bank=loop.bank, settings=loop.settings, conn=loop.conn, claude=loop.claude, pangram=loop.pg,
                     budget=loop.budget, purpose=getattr(loop, "purpose", "loop"), log=loop.log)
    raw0 = real.read_bytes().decode("utf-8")
    lay = layout(loop.job, loop.bank)
    rows = gather(loop.job, loop.bank, raw0, secs, lay=lay)
    rp.setup()
    try:
        results, targets = rp.propose(rows, lay, raw0)
    except (CapReached, ClaudeCallError) as e:
        loop.restore_result = {"skipped": f"stopped: {str(e)[:120]}"}
        return loop.restore_result
    touched = sorted({r["section"] for r in rows if results.get(r["label"], {}).get("accepted")})
    sections_out, kept = {}, []
    if touched:
        raw, sections_out, kept = rp.apply_sections(raw0, rows, results, touched, (ClientCap, CapReached))
        if kept:
            stamp = loop.clock.stamp()
            if loop.write_back_on and real.read_bytes().decode("utf-8") == raw0:
                real.write_bytes(raw.encode("utf-8"))
            best = rp.bank_entries(lay, rows, results, targets, sections_out, kept, stamp)
            loop.job.assemble(best, keep_others=True)
    when = loop.clock.stamp()
    out = report_file(f"restore {when.replace(':', '')}.md")
    out.write_text(report_md(when, False, rows, results, targets, sections_out, rp), encoding="utf-8")
    loop.restore_result = {"restored": sum(1 for t in targets.values() if t.get("picked")), "looked_at": len(targets),
                           "sections": {n: s.get("result") for n, s in sections_out.items()}, "report": out.name,
                           "pangram_usd": round(rp.spent, 2), "claude_usd": round(rp.claude_spent, 2)}
    loop.log(f"restore pass: {loop.restore_result['restored']} of {len(targets)} paragraph(s) restored; "
             + ("; ".join(f"section {n} {s}" for n, s in loop.restore_result["sections"].items()) or "no section touched")
             + f"; report {out.name}")
    return loop.restore_result


# ---------------------------------------------------------------------- main
def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="the restore pass: Opus directs, careful Sonnet edits, code gates (plan 6.9)")
    ap.add_argument("--job", help="the job, a folder in jobs/ (default: [job] default in config.toml)")
    ap.add_argument("--real", help="the document (default: job.toml's document)")
    ap.add_argument("--section", help="only these sections, comma-separated (default: the job's scope)")
    ap.add_argument("--para", help="only these paragraphs or windows, comma-separated ('s1w1 p2,s3w0')")
    ap.add_argument("--write", action="store_true", help="apply and re-score; without this nothing is written or scored")
    ap.add_argument("--dry", action="store_true", help="direct, edit and check only: no Pangram, no write (the default; names the report a dry run)")
    ap.add_argument("--purpose", default="loop", choices=["loop", "experiment"],
                    help="the Pangram purpose for the re-score (experiment when the day's loop cap is spent)")
    ap.add_argument("--max-usd", type=float, help="the most this pass may spend on Pangram")
    ap.add_argument("--claude-usd", type=float, help="the most this pass may spend on Claude (default: restore.max_claude_usd)")
    ap.add_argument("--jokes", action="store_true", help="joke: items on, against the config (for a test read)")
    ap.add_argument("--report", help="where the report goes (default: analytics/restore <date> [<time>] [(dry run)].md)")
    a = ap.parse_args(argv)
    if a.write and a.dry:
        ap.error("--write and --dry don't go together")

    from . import db
    from .budget import Budget, CapReached
    from .claude_call import ClaudeCallError, run as claude_run
    from .pangram import CapReached as ClientCap, Pangram
    from .schedule import Clock, Schedule
    settings = Settings()
    conn = db.connect(settings.db_path)
    job = Job(a.job)
    real = Path(a.real) if a.real else job.real_file
    sections = [int(x) for x in a.section.split(",")] if a.section else list(job.sections)
    only = {x.strip() for x in a.para.split(",")} if a.para else None
    bank = Bank(job.bank_path)
    clock = Clock()
    budget = Budget(settings, conn, clock, Schedule(settings.config))
    day_start = time.mktime(time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d"))
    rp = RestorePass(job=job, bank=bank, settings=settings, conn=conn, claude=claude_run,
                     pangram=Pangram(settings, conn) if a.write else None, budget=budget, purpose=a.purpose,
                     max_usd=a.max_usd, claude_usd=a.claude_usd, jokes=True if a.jokes else None)
    raw0 = real.read_bytes().decode("utf-8")
    lay = layout(job, bank)
    rows = gather(job, bank, raw0, sections, lay=lay)
    if only:
        for r in rows:
            r["asked"] = r["label"] in only or r["wid"] in only
    rp.setup()
    todo = [r for r in rows if r["fixable"] and r.get("asked", True)]
    print(f"{len(todo)} paragraph(s) to look at (sections {', '.join(map(str, sections))}); jokes {'on' if rp.jokes else 'off'}")
    try:
        results, targets = rp.propose(todo, lay, raw0)
    except (CapReached, ClaudeCallError) as e:
        print(f"stopped: {e}")
        return 1
    touched = sorted({r["section"] for r in rows if results.get(r["label"], {}).get("accepted")})
    sections_out = {}
    if a.write and touched:
        raw, sections_out, kept = rp.apply_sections(raw0, rows, results, touched, (ClientCap, CapReached))
        if kept:
            if real.read_bytes().decode("utf-8") != raw0:
                print("the real guide changed while the pass ran; nothing written")
                return 1
            real.write_bytes(raw.encode("utf-8"))
            best = rp.bank_entries(lay, rows, results, targets, sections_out, kept, clock.stamp())
            job.assemble(best, keep_others=True)
    elif touched:
        for n in touched:
            _, rep = rp.section_text(raw0, rows, results, n)
            sections_out[n] = {"result": f"dry run: {len(rep['replaced'])} paragraph(s) would change", "before": None}
    dry = not a.write
    when = time.strftime("%Y-%m-%d") if dry else time.strftime("%Y-%m-%d %H%M")
    out = (report_file(Path(a.report).name, Path(a.report).parent) if a.report
           else report_file(f"restore {when}{' (dry run)' if dry else ''}.md"))
    out.write_text(report_md(when, dry, rows, results, targets, sections_out, rp, claude_spend_since(conn, day_start)),
                   encoding="utf-8")
    n_done = sum(1 for t in targets.values() if t.get("picked"))
    line = (f"{clock.stamp()} restore pass{' (dry run)' if dry else ''}: {len(targets)} paragraph(s) looked at,"
            f" {n_done} restored; "
            + ("; ".join(f"section {n} {s.get('result', '')}" for n, s in sorted(sections_out.items())) or "no section touched")
            + f"; Pangram ${rp.spent:.2f}; Claude ${rp.claude_spent:.2f}; report {out.name}")
    logp = settings.runs_dir / job.name / "restore.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    with logp.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
