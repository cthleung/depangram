"""The Phase 2 fix list (plan 5.3), as filters that plug into the filter stack in engine/draws.py.

From experiments 7 and 8:
  - hedges on a claim ("probably", "once", "I think") and deliberate repeats
    ("round, and round, and round") are must-keeps for the run that holds them.
    Code finds them in the original run, so a must-keep never depends on Sonnet
    remembering to write it down; the notes carry them too (a "keep:" line), for
    the writer's sake, and any short phrase on that line joins the list.
  - a candidate that adds a first-person sighting the original doesn't have
    ("I've seen", "hit me with", "I asked it", "I've tested it") is out; "I've got you" is an idiom, not a
    sighting.
  - the notation lock narrows to real templates: two or more X/Y/Z placeholders
    outside quotation marks, or a sentence that opens with its quoted example.
    A prose sentence that quotes patterns regrows, with its quotes as must-keeps;
    a quote the writer got nearly right (case, punctuation) is spliced back exact.

Everything here is plain code: no model, no Pangram, no prompt.

The wording choices (the banned words, dashes, the spelling locale, the swear list and its budget, the filler tags,
the connective openers and the forum furniture) come from the house rules file: rules.toml at the repo root, or the
file RULES_FILE names, read by engine/tics/house_rules.py. checks.SWEARS, checks.FILLER_TAGS, checks.CONNECTIVE_CAP
and the rest read the rules in use: the base (use_rules), a loaded job's own file over it, or a scope's
(scoped(rules), for one block of code in one thread).
"""
import contextvars
import re
from contextlib import contextmanager

from engine.tics import house_rules as _house_rules_file

PLACEHOLDER = re.compile(r"^[XYZ][.,;:!?\"')]*$")
QUOTE = re.compile(r"\"[^\"]+\"")

# hedge phrases, matched as whole words; the run's original must have one for it to become a must-keep
HEDGES = ["probably", "maybe", "perhaps", "possibly", "reportedly", "apparently", "supposedly", "likely",
          "might", "I think", "I suspect", "I guess", "I reckon", "kind of", "sort of", "pretty much",
          "once", "about", "roughly", "around", "nearly", "almost", "up to"]
# phrases on the list that are only a hedge in some uses
NOT_A_HEDGE = {
    "once": re.compile(r"\bonce\s+(?:you|we|I|it|they|he|she|the|that|this|people|someone|somebody)\b", re.I),
    "sort of": re.compile(r"\bsort of (?:thing|stuff)\b", re.I),
    "kind of": re.compile(r"\bkind of (?:thing|stuff)\b", re.I),
}
NEEDS_NUMBER = {"about", "roughly", "around", "nearly", "almost", "up to"}     # only before a number
STOP = {"the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "it", "is", "was", "that", "this", "i",
        "you", "he", "she", "they", "we", "for", "with", "as", "at", "by", "be", "so", "if", "then", "not", "no"}
SIGHTING = re.compile(
    r"\b(?:I(?:'ve| have)?\s+(?:just\s+|actually\s+|literally\s+|personally\s+|once\s+|even\s+)?"
    r"(?:seen|saw|watched|caught|asked|tried|tested|read|had|got(?!\s+you\b)|remember|noticed|found|met|heard|made it|"
    r"told it|gave it|fed it)|"
    r"(?:hit|showed|told|gave|sent|served|handed)\s+me\b|my (?:own )?(?:chats?|prompts?|experience)|"
    r"in my experience|the other day|last (?:week|night|month)|"
    r"I(?:'ve| have)?\s+\w+\s+(?:it|this|that|one|them)\s+myself\b)", re.I)


def strip_quotes(s):
    return QUOTE.sub(" ", s)


def is_template(s):
    """A pattern template rather than prose: two or more X/Y/Z placeholders outside quotation marks
    ("X and X and X."), or a sentence that opens with its quoted example ('"Before X..." is the classic one.')."""
    if s.lstrip().startswith('"'):
        return True
    return sum(1 for w in strip_quotes(s).split() if PLACEHOLDER.match(w)) >= 2


def pattern_quotes(s):
    """Quoted spans that hold a placeholder: "every X, every Y, every Z"."""
    return [q for q in QUOTE.findall(s) if any(PLACEHOLDER.match(w) for w in q.strip('"').split())]


def _word_re(phrase):
    return re.compile(r"(?<![\w'])" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?![\w'])", re.I)


def hedges_in(text):
    """The hedge phrases a run carries, outside quotation marks."""
    t = strip_quotes(text)
    out = []
    for h in HEDGES:
        for m in _word_re(h).finditer(t):
            if h in NOT_A_HEDGE and NOT_A_HEDGE[h].match(t, m.start()):
                continue
            if h in NEEDS_NUMBER and not re.match(r"\s*(?:\d|a (?:third|quarter|dozen))", t[m.end():]):
                continue
            if h == "kind of" and re.search(r"\b(?:the|that|this|what|a|some)\s+$", t[:m.start()], re.I):
                continue
            out.append(h)
            break
    return out


def repeats_in(text, span=10, times=3):
    """Deliberate repeats outside quotes: a content word said `times` or more within `span` words
    ("round, and round, and round"). Returns [(word, count)]."""
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", strip_quotes(text))]
    out = {}
    for i, w in enumerate(words):
        if w in STOP or len(w) < 3:
            continue
        n = words[i:i + span].count(w)
        if n >= times:
            out[w] = max(out.get(w, 0), n)
    return sorted(out.items())


def keep_phrases(notes):
    """The quoted phrases on a notes "keep:" line (the v5 notes prompt asks for one)."""
    out = []
    for n in notes:
        if re.match(r"^\s*keep\s*:", n, re.I):
            out += [q.strip('"').strip() for q in QUOTE.findall(n)]
    return [p for p in out if p]


def must_keeps(orig_text, notes=()):
    """{"hedges": [...], "repeats": [(word, count)]} for one run: what code finds in the original run, plus what the
    notes' keep: line names that the run's original really has. The keep line adds repeats, and hedges only from
    HEDGES (where Sonnet can overrule the "once you've" test): on the smoke test it also listed "usually" and
    "basically", which threw out 30 of 48 draws, and experiment 5 showed a long must-keep list freezes a sentence."""
    hedges = hedges_in(orig_text)
    repeats = dict(repeats_in(orig_text))
    bare = strip_quotes(orig_text)
    known = {h.lower() for h in HEDGES}
    for p in keep_phrases(notes):
        rep = repeats_in(p, span=12, times=2)
        if rep:
            for w, n in rep:
                have = len(_word_re(w).findall(bare))
                if have >= 2:
                    repeats[w] = max(repeats.get(w, 0), min(n, have))
        elif p.lower() in known and _word_re(p).search(bare) and p.lower() not in (h.lower() for h in hedges):
            hedges.append(p)
    return {"hedges": hedges, "repeats": sorted(repeats.items())}


def repeat_ok(candidate, repeats):
    bare = strip_quotes(candidate)
    return all(len(_word_re(w).findall(bare)) >= n for w, n in repeats)


def invented_sighting(candidate, orig_text):
    """The first first-person sighting in the candidate that the original run doesn't have, or None."""
    have = {re.sub(r"\s+", " ", m.group(0).lower()) for m in SIGHTING.finditer(orig_text)}
    for m in SIGHTING.finditer(candidate):
        if re.sub(r"\s+", " ", m.group(0).lower()) not in have:
            return m.group(0)
    return None


def split_sentences(text):
    from engine.draws import sents
    return sents(text)


def restates_in_context(candidate, orig_text, before="", after=""):
    """A candidate that says twice what one sentence already said, against its neighbours in the paragraph (the
    sentence before it and the one after), when the original run in the same place doesn't."""
    def pairs(t):
        ss = ([before] if before else []) + split_sentences(t) + ([after] if after else [])
        return restated_pairs(ss)
    return bool(pairs(candidate)) and not pairs(orig_text)


def extra_checks_for(orig_text, repeats, notes=(), before="", after=""):
    """Callables for draws.filter_candidates(extra_checks=...): each takes a candidate and returns a reject
    reason or None. (Hedges go through filter_candidates' own `extra_must`.) `notes` are the paragraph's notes (for
    the names a run has to keep); `before` and `after` the neighbouring sentences in its paragraph (for the
    restatement check). The candidate they see is the finished one: quotes spliced, the long-quote tail on."""
    required = names_required(orig_text, notes)
    checks = [lambda c: "invented sighting" if invented_sighting(c, orig_text) else None,
              lambda c: punctuation_problem(c, orig_text),
              lambda c: "lost a name" if names_missing(c, required) else None,
              lambda c: "says it twice" if restates_in_context(c, orig_text, before, after) else None]
    if repeats:
        checks.append(lambda c: None if repeat_ok(c, repeats) else "must-keep repeat")
    return checks


def _norm(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def splice_quotes(candidate, quotes):
    """Put each must-keep quote back exactly where the writer quoted it nearly right (same words, different case or
    punctuation). `quotes` are the original's quoted spans with their marks."""
    out = candidate
    for q in quotes:
        if q in out:
            continue
        want = _norm(q)
        for m in QUOTE.finditer(out):
            if _norm(m.group(0)) == want:
                out = out[:m.start()] + q + out[m.end():]
                break
    return out


# ---------------------------------------------------------------------- the first run-day's checks
def punctuation_problem(candidate, orig_text=""):
    """The first punctuation slip in a candidate that its original doesn't have, or None (M2: an accepted run
    ended `the old line.: "we fixed it, honest").`). A sentence end straight before a colon, an
    odd number of double quotes, unmatched parentheses (or a ")" before any "("), and a run that ends on a colon."""
    def problems(t):
        out = []
        if re.search(r"[.?!]:", t):
            out.append("stray colon after a full stop")
        if t.count('"') % 2:
            out.append("unmatched quote")
        depth = 0
        for ch in t:
            depth += (ch == "(") - (ch == ")")
            if depth < 0:
                break
        if depth < 0:
            out.append("stray closing parenthesis")
        elif depth > 0:
            out.append("unclosed parenthesis")
        if t.rstrip().endswith(":"):
            out.append("ends on a colon")
        return out
    have = set(problems(orig_text))
    return next((p for p in problems(candidate) if p not in have), None)


# leading words a multiword name can't start with when it opens a sentence ("Then Jane Doe")
NAME_SKIP = {"the", "then", "so", "and", "but", "in", "on", "now", "when", "if", "this", "that", "it", "a", "an",
             "well", "look", "back", "even", "also", "which", "where", "while", "after", "before", "since", "as",
             "at", "for", "from", "with", "by", "of", "to", "or", "yes", "no", "oh", "plus", "still", "just",
             "each", "every", "some", "any", "all", "most", "my", "your", "our", "his", "her", "their", "its",
             "these", "those", "there", "here", "what", "why", "how", "who", "one", "two", "both"}


def multiword_names(text):
    """Capitalized names of two or more words ("Jane Doe", "Big Example Guide"), outside quotation marks,
    with a capitalized sentence opener taken off the front."""
    bare = strip_quotes(text)
    out = []
    for m in re.finditer(r"[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)+", bare):
        toks = m.group(0).split()
        while toks and toks[0].lower() in NAME_SKIP:
            toks = toks[1:]
        if len(toks) >= 2:
            out.append(" ".join(toks))
    return out


def citations(text):
    """"Surname et al." and "(Name, year)" citations, as written."""
    return re.findall(r"\b[A-Z][\w'’-]+ et al\.", text) + re.findall(r"\([A-Z][^()]*?\d{4}[a-z]?\)", text)


def names_required(orig_text, notes=()):
    """Plan 5.3's names, counted (M2: a run dropped a person's full name and the fact check passed it). Every multiword
    name and every citation in the original run has to stay, unless the notes leave it out (the notes prompt
    quotes every name the paragraph makes a point of, so a name it skipped is decoration) or list it on a
    "droppable:" line."""
    notes_text = " ".join(notes).lower()
    drop = " ".join(n for n in notes if re.match(r"^\s*droppable\s*:", n, re.I)).lower()
    out = []
    for nm in multiword_names(orig_text) + citations(orig_text):
        key = nm.lower().rstrip(".")
        bare = re.sub(r"\s+et al$", "", key).strip("()")
        if notes and bare.split(",")[0] not in notes_text:
            continue
        if bare.split(",")[0] in drop:
            continue
        if nm not in out:
            out.append(nm)
    return out


def names_missing(candidate, required):
    """The required names the candidate lost. A "(Name, year)" citation counts as kept when its name and its year
    both stand in the candidate ("the OpenAI, 2023 paper"), the same rule draw_pool's citation splice uses."""
    low = candidate.lower()
    out = []
    for nm in required:
        if nm.lower().rstrip(".") in low:
            continue
        m = re.match(r"\(([^,()]+?),?\s*(\d{4}[a-z]?)\)$", nm)
        if m and m.group(1).lower() in low and m.group(2) in candidate:
            continue
        # "Smith et al." stays when its surname does ("the 2025 Smith paper")
        m = re.match(r"([A-Z][\w'’-]+) et al\.$", nm)
        if m and re.search(r"(?<![\w])" + re.escape(m.group(1)) + r"(?![\w])", candidate):
            continue
        out.append(nm)
    return out


RESTATE_STOP = STOP | {"are", "were", "been", "being", "has", "have", "had", "do", "does", "did", "don't", "doesn't",
                       "didn't", "hasn't", "haven't", "isn't", "aren't", "wasn't", "can't", "won't", "it's", "that's",
                       "there", "their", "them", "its", "his", "her", "my", "your", "our", "what", "which", "who",
                       "when", "where", "how", "all", "any", "some", "more", "most", "just", "really", "actually",
                       "course", "very", "even", "also", "about", "into", "out", "up", "down", "than", "like",
                       "one", "much", "get", "got", "make", "thing", "things", "way", "these", "those", "would",
                       "could", "should", "will", "can", "may", "might", "fucking", "fuck", "shit", "damn", "hell"}


def _stem(w):
    w = w.lower().replace("’", "'")
    w = re.sub(r"'s$", "", w)
    for suf in ("ing", "ed", "es", "s", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def content_stems(sentence):
    words = re.findall(r"[A-Za-z][A-Za-z'’]+", strip_quotes(sentence))
    return {_stem(w) for w in words if w.lower() not in RESTATE_STOP and len(w) >= 3}


RESTATE_MIN = 0.6        # a known restatement scored 0.83; ordinary neighbouring pairs in a test bank topped out
                         # at 0.50
RESTATE_WORDS = 4        # a sentence with fewer content words than this is too short to judge


def restatement(a, b):
    """How much of the shorter of two neighbouring sentences the other already says: the share of its content
    stems (lightly stemmed, quotes out) found in the other. 0 when the shorter has fewer than RESTATE_WORDS."""
    sa, sb = content_stems(a), content_stems(b)
    short, other = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    if len(short) < RESTATE_WORDS:
        return 0.0
    return len(short & other) / len(short)


def restated_pairs(sentences, threshold=RESTATE_MIN):
    """[(i, i + 1, score)] for neighbouring sentences where one says what the other already did."""
    return [(i, i + 1, s) for i in range(len(sentences) - 1)
            if (s := restatement(sentences[i], sentences[i + 1])) >= threshold]


# the drop rule's "carries a fact" test (plan 5.4), erring toward not cutting
FACTY = re.compile(r"\d|\"|\bet al\b|\(\w[^)]*\d{4}\)|\b(?:is called|are called|means|meaning|stands for|defined|"
                   r"is a|is an|are a|refers to|known as|according to|study|paper|research|found|measured)\b", re.I)


def carries_fact(run_text, names_in_run=()):
    """True when a run holds a number, a quote, a citation, a definition or a study, or names something: the drop
    rule can't cut it."""
    return bool(FACTY.search(run_text)) or bool(names_in_run)



# ---------------------------------------------------------------------- the house rules in use
# The rules in use are a Rules object (below), read each time a function runs. A process has a base, which use_rules
# sets (the default file, or a test's), and a job's own rules over it: a Job loaded with its rules file sets them,
# and a Job loaded without one clears them, so one job's file never carries over to the next job loaded. That is the
# command line's way, one job a process. scoped(rules) puts a job's rules in use for one block of code in the
# current thread only (a contextvars.ContextVar), so a long-lived process (the MCP server) reads each job with its
# own rules and leaves nothing behind when the block ends. checks.SWEARS, checks.FILLER_TAGS and the other names in
# RULE_NAMES read the rules in use at the moment they're read; code that copied one at import (from .checks import
# FILLER_TAGS) keeps the value it copied.
RULE_NAMES = ("HOUSE_RULES", "RULES_PATH", "SWEARS", "STRONG_SWEARS", "SWEAR_MULTIPLIER", "FILLER_TAGS", "FILLER_CAP",
              "SPELLING", "CONNECTIVE_OPENERS", "CONNECTIVE_RE", "CONNECTIVE_CAP", "REDDIT_TELL")


class Rules:
    """One rules dict (shaped like engine/tics/house_rules.py DEFAULTS) and what the checks need from it, compiled
    once. RULES_PATH is the file it came from, or None for the built-in defaults."""

    def __init__(self, rules, path=None):
        self.HOUSE_RULES, self.RULES_PATH = rules, path
        sw = rules["swears"]
        self.SWEARS = _house_rules_file.word_regex(sw["words"], sw["except"])
        self.STRONG_SWEARS = _house_rules_file.word_regex(sw["strong"], sw["except"])
        self.SWEAR_MULTIPLIER = float(sw["multiplier"])
        self.FILLER_TAGS = tuple(rules["filler"]["words"])
        self.FILLER_CAP = int(rules["filler"]["cap"])
        self.SPELLING = rules["locale"]["spelling"]
        self.CONNECTIVE_OPENERS = tuple(rules["connectives"]["openers"])
        self.CONNECTIVE_RE = re.compile(r"(?:^|(?<=[.!?][\"')]\s)|(?<=[.!?]\s))(?:"
                                        + "|".join(re.escape(o) for o in self.CONNECTIVE_OPENERS) + r")(?![\w])",
                                        re.I | re.M) if self.CONNECTIVE_OPENERS else re.compile(r"(?!)")
        self.CONNECTIVE_CAP = int(rules["connectives"]["cap"])
        pats = rules["furniture"]["patterns"]
        self.REDDIT_TELL = re.compile("|".join(pats), re.M) if pats else re.compile(r"(?!)")
        self.house = None            # house_patterns' compiled regexes, made on first use


_base = None                         # use_rules: the default file, or the one a test or a caller set
_job = None                          # a loaded job's own rules over the base (Job with bind_rules), or None
_scoped = contextvars.ContextVar("depangram_rules", default=None)


def _r():
    """The rules in use: a scope's, else a loaded job's, else the base."""
    return _scoped.get() or _job or _base


def __getattr__(name):
    """checks.SWEARS and the other RULE_NAMES, read from the rules in use."""
    if name in RULE_NAMES:
        return getattr(_r(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_rules(path=None):
    """A rules file (a job's own, say) over the defaults, as a Rules object, without putting it in use. With no path,
    the default file (RULES_FILE, else rules.toml at the repo root). A file that doesn't validate raises ValueError."""
    path = path or _house_rules_file.default_path()
    return Rules(_house_rules_file.load_rules(path), path)


def use_rules(path=None):
    """Load a rules file over the defaults and make it the base in use for the whole process, clearing any job's
    rules over it. With no path, the default file. Returns the rules dict."""
    global _base, _job
    _base, _job = load_rules(path), None
    return _base.HOUSE_RULES


def use_job_rules(rules):
    """A job's rules (a Rules, or None for none) over the base for the whole process: what a Job loaded with
    bind_rules sets, so the command line, one job a process, reads that job's file everywhere."""
    global _job
    _job = rules


@contextmanager
def scoped(rules):
    """`rules` (a Rules, or None for the base alone) in use for the block, in the current thread only. Whatever was
    in use before comes back when the block ends, however it ends. Yields the Rules in use."""
    token = _scoped.set(rules if rules is not None else _base)
    try:
        yield _r()
    finally:
        _scoped.reset(token)


def house_rules():
    """The rules dict in use."""
    return _r().HOUSE_RULES


OLD_BUDGET_KEYS = ("swear_multiplier", "filler_cap", "filler_words")


def budgets(config=None, warn=None):
    """(swear multiplier, filler cap, filler words) for a run: the rules file's ([swears] multiplier, [filler] cap
    and words). An older config.toml that still sets them under [loop] wins, with a one-line warning through `warn`
    (default: standard error), so an old setup keeps its numbers until they move to the rules file."""
    loop = (config or {}).get("loop", {}) or {}
    r = _r()
    mult, cap, words = r.SWEAR_MULTIPLIER, r.FILLER_CAP, r.FILLER_TAGS
    old = [k for k in OLD_BUDGET_KEYS if k in loop]
    if old:
        import sys
        msg = (f"config.toml [loop] still sets {', '.join(old)}; they win for now, but they belong in the rules file"
               " (rules.toml: [swears] multiplier, [filler] cap and words)")
        (warn or (lambda m: print(m, file=sys.stderr)))(msg)
        mult = float(loop.get("swear_multiplier", mult))
        cap = int(loop.get("filler_cap", cap))
        words = tuple(loop.get("filler_words", words))
    return mult, cap, words


use_rules()


# ---------------------------------------------------------------------- swearing
# What the section budget and the fix pass count as a swear, from the rules file. "strong" is counted again on its
# own, because a writer model that trades a mild swear for a strong one keeps the total flat while the tone climbs.


def swear_counts(text):
    """(words, swears, strong swears) in a text."""
    return len(text.split()), len(_r().SWEARS.findall(text)), len(_r().STRONG_SWEARS.findall(text))


def author_swear_rates(lines, samples=()):
    """Swears per word in your own writing, for the per-section budget: {"all", "strong", "sources"}. Each source
    (your lines in the document, your samples) is measured on its own and the higher rate counts. Pooled, samples
    that swear less than the document would drag the rate under what your own sections carry, and the budget would
    flag the original text."""
    sources = {}
    for name, texts in (("your lines", lines), ("samples", samples)):
        w = s = st = 0
        for t in texts:
            a, b, c = swear_counts(t)
            w, s, st = w + a, s + b, st + c
        if w:
            sources[name] = {"words": w, "swears": s, "strong": st}
    if not sources:
        return {"all": 0.0, "strong": 0.0, "sources": {}}
    return {"all": max(v["swears"] / v["words"] for v in sources.values()),
            "strong": max(v["strong"] / v["words"] for v in sources.values()), "sources": sources}


def section_swear_problem(candidate, orig_text, rest, rates, multiplier=None):
    """Plan 5.3's swearing filter, for the whole section: a reject reason when the candidate carries more swears
    than its original run (all of them, or the strong ones) and the section as posted, with the candidate in, would
    then sit above your own rate times `multiplier`. `rest` is (words, swears, strong) of the section with this
    run left out. A candidate that swears no more than the run it replaces always passes, so neither your own text
    nor a creep that's already in can make a run unreplaceable. `multiplier` defaults to the rules file's."""
    if not rates or not rest:
        return None
    multiplier = _r().SWEAR_MULTIPLIER if multiplier is None else multiplier
    cw, cs, cst = swear_counts(candidate)
    _, os_, ost = swear_counts(orig_text)
    words = rest[0] + cw
    if cst > ost and rest[2] + cst > rates["strong"] * words * multiplier:
        return "section swearing"
    if cs > os_ and rest[1] + cs > rates["all"] * words * multiplier:
        return "section swearing"
    return None


# ---------------------------------------------------------------------- the filler-tag budget
# A filler tag tacked onto a clause ("it works, honestly") is a habit a model editor brings in. It's capped per
# section as posted, like the swear budget: a candidate that brings in more of them than the run it replaces is out
# when the section would then hold more than the cap. Your own stay, since a candidate that adds none always passes.
# The tags and the cap come from the rules file (FILLER_TAGS, FILLER_CAP).


def filler_count(text, words=None):
    """How many filler tags a text carries, outside quotation marks. `words` defaults to FILLER_TAGS."""
    words = _r().FILLER_TAGS if words is None else words
    bare = strip_quotes(text)
    return sum(len(re.findall(r"\b" + re.escape(w) + r"\b", bare, re.I)) for w in words)


def section_counts(text, words=None):
    """(words, swears, strong swears, filler tags) of a text: swear_counts plus the filler-tag count, the shape the
    section budgets take as `rest`."""
    return swear_counts(text) + (filler_count(text, words),)


def section_filler_problem(candidate, orig_text, rest, cap=None, words=None):
    """A reject reason when the candidate carries more filler tags than the run it replaces and the section as
    posted, with the candidate in, would then hold more than `cap`. `rest` is section_counts of the section with
    this run left out; an older three-item rest (no filler count) passes everything. `cap` and `words` default to
    the rules file's (FILLER_CAP, FILLER_TAGS)."""
    cap = _r().FILLER_CAP if cap is None else cap
    words = _r().FILLER_TAGS if words is None else words
    if not rest or len(rest) < 4 or not words:
        return None
    cf, of = filler_count(candidate, words), filler_count(orig_text, words)
    if cf > of and rest[3] + cf > cap:
        found = [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", strip_quotes(candidate), re.I)]
        return f"filler tag ({', '.join(found)})"
    return None


# en-GB spellings and their en-US forms, as (pattern, replacement): spelling facts, so they stay in code. Whether the
# text keeps American spelling is the rules file's [locale] spelling. With "american" the fix pass proposes these
# itself and refuses a replacement that brings one in; with "any" both functions below do nothing.
_GB_US = [
    (r"\b(organ|real|recogn|apolog|emphas|summar|priorit|critic|minim|maxim|categor|character|memor|normal|optim|"
     r"standard|util|visual|human|special|general|final|author|legal|modern|popular|stabil|subsid|harmon|"
     r"weapon|monopol|jeopard|patron|sympath|theor|scrutin|fantas|dramat|familiar|capital|central|commercial|"
     r"global|ideal|internal|neutral|person|revolution|social|vandal|equal)is(e|ed|es|ing|ation|ations|er|ers)\b",
     r"\1iz\2"),
    (r"\banalys(e|ed|es|ing)\b", r"analyz\1"),
    (r"\bparalys(e|ed|es|ing)\b", r"paralyz\1"),
    (r"\b(col|fav|flav|behavi|hum|neighb|lab|harb|hon|rum|vap|arm|vig|endeav|rig)our(s|ed|ing|ite|ites|able|ful|less)?\b",
     r"\1or\2"),
    (r"\bcentre(s|d)?\b", r"center\1"), (r"\bmetre(s)?\b", r"meter\1"), (r"\btheatre(s)?\b", r"theater\1"),
    (r"\bfibre(s)?\b", r"fiber\1"), (r"\bcalibre\b", "caliber"),
    (r"\bgrey(s|er|est|ish)?\b", r"gray\1"),
    (r"\b(trav|lab|mod|sign|canc|fu|quarr|mar|tunn|jew|dial)ell(ed|ing|er|ers)\b", r"\1el\2"),
    (r"\bpractis(e|ed|es|ing)\b", r"practic\1"),
    (r"\bdefence(s)?\b", r"defense\1"), (r"\boffence(s)?\b", r"offense\1"),
    (r"\bcatalogue(s|d)?\b", r"catalog\1"), (r"\banalogue(s)?\b", r"analog\1"),
    (r"\bjudgement(s)?\b", r"judgment\1"), (r"\bprogramme(s)?\b", r"program\1"),
    (r"\blearnt\b", "learned"), (r"\bspelt\b", "spelled"), (r"\bdreamt\b", "dreamed"), (r"\bburnt\b", "burned"),
    (r"\bwhilst\b", "while"), (r"\bamongst\b", "among"), (r"\bmum(s)?\b", r"mom\1"), (r"\bmaths\b", "math"),
    (r"\bsceptic(s|al|ism)?\b", r"skeptic\1"), (r"\btyre(s)?\b", r"tire\1"), (r"\baluminium\b", "aluminum"),
    (r"\bcheque(s)?\b", r"check\1"), (r"\bplough(s|ed|ing)?\b", r"plow\1"), (r"\bmould(s|ed|ing|y)?\b", r"mold\1"),
]
GB_US = [(re.compile(p, re.I), r) for p, r in _GB_US]


def gb_spellings(text):
    """The en-GB spellings in a text, as written ([] when the rules leave spelling alone)."""
    if _r().SPELLING != "american":
        return []
    return [m.group(0) for rx, _ in GB_US for m in rx.finditer(text)]


def to_us(text):
    """The text with every en-GB spelling on the list turned American, keeping a capital first letter (the text as
    it is when the rules leave spelling alone)."""
    if _r().SPELLING != "american":
        return text
    for rx, rep in GB_US:
        def one(m, rep=rep):
            new = m.expand(rep)
            return new[0].upper() + new[1:] if m.group(0)[0].isupper() else new
        text = rx.sub(one, text)
    return text


def house_patterns():
    """The house rules' regexes, in the order house_words reports them: em and en dashes, the banned words (the
    hard and the check list as one regex), a "--". The dash ones only when the rules ban dashes. The scanner
    (engine/tics/tic_patterns.py) reads the same banned words from the same file, so there's one list."""
    r = _r()
    if r.house is None:
        h = r.HOUSE_RULES["house"]
        out = [re.compile("[\u2014\u2013]")] if h["dashes"] else []
        pat = _house_rules_file.word_pattern(list(h["banned_words"]) + list(h["banned_words_check"]))
        if pat:
            out.append(re.compile(pat, re.I))
        if h["dashes"]:
            out.append(re.compile(r"--"))
        r.house = out
    return r.house


_house_rules = house_patterns          # the old name, for older callers


def banned_words():
    """The banned words as words: the hard and the check list with their wildcards off, for a prompt to name."""
    h = _r().HOUSE_RULES["house"]
    return _house_rules_file.stems(list(h["banned_words"]) + list(h["banned_words_check"]))


def house_words(text):
    """Em dashes, en dashes, a "--" and the banned words in a text, as written."""
    return [m.group(0) for rx in _house_rules() for m in rx.finditer(text)]


def new_items(old, new, find):
    """What `find(new)` has that `find(old)` doesn't, counted (a word said twice where it was said once counts)."""
    have = [x.lower() for x in find(old)]
    out = []
    for x in find(new):
        if x.lower() in have:
            have.remove(x.lower())
        else:
            out.append(x)
    return out


def tokens(text):
    """Word tokens for the fix pass's diff: words (with their apostrophes and hyphens) and each punctuation mark."""
    return re.findall(r"[\w’']+(?:[-’'][\w’']+)*|[^\w\s]", text)


def words_changed(a, b):
    """Tokens changed between two sentences, word level: for every stretch that differs, the longer side counts."""
    import difflib
    sm = difflib.SequenceMatcher(None, tokens(a), tokens(b), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def measured_swear_rates(verdicts, samples_path=None):
    """author_swear_rates from your lines in the job (`your lines.md`'s "yours" verdicts) and the clean samples in
    the tagged samples file (the ones tagged as AI-assisted left out). The samples come from `samples_path`, else
    SAMPLES_TAGGED (engine/settings.py), read with engine/tics/voice_audit.py; with none set, or a file that won't
    read, the rate is your lines' alone."""
    from engine.settings import path_setting
    from engine.tics import voice_audit
    lines = [s for s, v in verdicts.items() if v in ("yours", "yours?")]
    samples = []
    path = samples_path or path_setting("SAMPLES_TAGGED")
    if path:
        try:
            kept, _ = voice_audit.load_samples(str(path))
            samples = ["\n".join(ln for _, ln in s["lines"]) for s in kept]
        except (OSError, ValueError):
            samples = []
    return author_swear_rates(lines, samples)


# ---------------------------------------------------------------------- the citation sentence's fixes
# From a diagnosis run: one citation sentence never got a candidate through, and most rejects were
# mechanical (footnote marks read as new numbers, "Smith et al. 2025" read as losing "(2025)",
# draws that pull in the next run's sentence read as saying it twice).
def _nums(s):
    return set(re.findall(r"\d+(?:[.,/]\d+)*", s))


def strip_footnote_marks(candidate, orig_text):
    """The candidate with Mistral's footnote marks taken out: a bare 1 or 2 digit number the original doesn't have,
    glued to a word or punctuation ("readers.1", "paragraphs2,"), opening the candidate before a capital ("1 In
    fact"), or standing between a closing bracket or quote and punctuation ("(2025) 2, who"). Numbers the original
    has, and digits inside a longer number ("2.3.1", "3/4"), stay."""
    keep = _nums(orig_text)

    def drop(m):
        return m.group(0) if m.group("n") in keep else ""
    c = re.sub(r"(?<![\d.,/])(?<!\d[.,/])(?<=[A-Za-z.,;:!?)\"'’”])(?P<n>\d{1,2})(?=[\s.,;:!?)\"]|$)", drop, candidate)
    c = re.sub(r"(?<=[)\"”])\s(?P<n>\d{1,2})(?=[,.;:]|\s)", drop, c)
    c = re.sub(r"^(?P<n>\d{1,2})\s+(?=[A-Z])", drop, c)
    c = re.sub(r"\s+([,.;:!?])", r"\1", c) if c != candidate else c
    return re.sub(r"\s{2,}", " ", c).strip()


def _citing_name(must, orig_text):
    """For a "(2025)" or "(OpenAI, 2023)" must-keep: (name, year), the name being the one inside the brackets or
    the surname just before them in the original ("Smith et al. (2025)"). None when it isn't a citation."""
    m = re.fullmatch(r"\(\s*(?:([^(),]+?),?\s+)?(\d{4}[a-z]?)\s*\)", must.strip())
    if not m:
        return None
    name, year = m.group(1), m.group(2)
    if not name:
        i = orig_text.find(must.strip())
        mm = re.search(r"([A-Z][\w'’-]+)(?:\s+et al\.?|\s+and\s+[A-Z][\w'’-]+)?\s*$", orig_text[:i] if i >= 0 else "")
        name = mm.group(1) if mm else None
    return (name, year) if name else None


def citation_kept(must, candidate, orig_text):
    """A citation must-keep counts as kept when its citing name and its year both stand in the candidate:
    "Smith et al. 2025" and "the 2025 Smith paper" keep "(2025)", as "the OpenAI, 2023 paper" keeps
    "(OpenAI, 2023)" (names_missing's rule). The year on its own doesn't."""
    got = _citing_name(must, orig_text)
    if not got:
        return False
    name, year = got
    return bool(re.search(r"(?<!\d)" + year + r"(?!\d)", candidate)) and \
        bool(re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", candidate, re.I))


def must_kept(must, candidate, orig_text):
    """One quote or citation must-keep, as the filter stack reads it."""
    return must.lower() in candidate.lower() or citation_kept(must, candidate, orig_text)


def splice_citations(candidate, orig_text):
    """Every "Surname et al. (year)" citation of the original put back exactly where the writer wrote it nearly
    right ("Smith et al. 2025", "Smith et al., (2025)", "Smith et al (2025)")."""
    out = candidate
    for m in re.finditer(r"\b([A-Z][\w'’-]+) et al\. \((\d{4}[a-z]?)\)", orig_text):
        exact, name, year = m.group(0), m.group(1), m.group(2)
        if exact in out:
            continue
        out = re.sub(re.escape(name) + r" et al\.?,?\s*\(?" + year + r"\)?", lambda _: exact, out, count=1)
    return out


# ---------------------------------------------------------------------- the writer arm's checks
# A writer model trained on forum comments writes their register, and two things come with it: explainer glue at
# the front of sentences ("Of course", "In other words", "we can see that") and the forum's own furniture (a
# zero-width-space entity, an "Edit:" paragraph, "TL;DR", "OP"). A writer may use the openers now and then, so
# they're capped per paragraph rather than banned; the furniture is out. Both lists, and the cap, come from the
# rules file (CONNECTIVE_OPENERS, CONNECTIVE_CAP, REDDIT_TELL above).
EG_RE = re.compile(r"(?<=\w[,;:(]\s)e\.g\.|(?<=\w\s)e\.g\.", re.I)


def connectives_in(text):
    """The connective openers a text carries: the listed phrases opening a sentence, and "e.g." mid-sentence."""
    bare = strip_quotes(text)
    return [m.group(0) for m in _r().CONNECTIVE_RE.finditer(bare)] + [m.group(0) for m in EG_RE.finditer(bare)]


def connective_problem(candidate, before="", after="", cap=None):
    """A reject reason when the paragraph with the candidate in ("before" and "after" are the rest of it) would hold
    more than `cap` connective openers and the candidate brought at least one of them. A paragraph already over the
    cap from earlier run-days doesn't freeze a candidate that adds none. `cap` defaults to CONNECTIVE_CAP."""
    cap = _r().CONNECTIVE_CAP if cap is None else cap
    mine = connectives_in(candidate)
    if not mine:
        return None
    total = len(mine) + len(connectives_in(before)) + len(connectives_in(after))
    return f"connective glue ({', '.join(mine)})" if total > cap else None


def reddit_tell(text):
    """The first piece of forum furniture in a text (by default a zero-width entity, an "Edit:" paragraph, TL;DR,
    r/sub, OP, a vote; the rules file's [furniture] patterns), or None."""
    m = _r().REDDIT_TELL.search(text)
    return m.group(0) if m else None


def trim_continuation(candidate, n_sent, next_sent):
    """A one-sentence run's draw that ran on into the next run's sentence ("... in the drawer. Which, as it turns
    out, ..."), cut back to its own sentence, so the says-it-twice check doesn't throw the whole draw away. Only when the
    extra sentence restates the next one or opens with its first three words."""
    if n_sent != 1 or not next_sent:
        return candidate
    ss = split_sentences(candidate)
    if len(ss) != 2:
        return candidate
    head = lambda s: [w.lower().strip(",.;:") for w in s.split()[:3]]  # noqa: E731
    if restatement(ss[1], next_sent) >= RESTATE_MIN or head(ss[1]) == head(next_sent):
        return ss[0]
    return candidate
