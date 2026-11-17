"""The reflection (plan 9 and 6.8): one Sonnet call at the wrap-up that reads the day and writes to the
author. The author asked for a reflection in the digest: what's working, where the writing and the document stand,
and where to improve next. The loop runs unattended, so at 16:00 this call writes it.

    python -m engine.reflect                      # today's run-day, live (one Sonnet call, at most --max-usd)
    python -m engine.reflect --day 2026-11-09     # a run-day by date
    python -m engine.reflect --dry                # gather only: the rendered input goes to the runs folder to read
    python -m engine.reflect --tag "by hand"      # the file is analytics/reflections/<day> (by hand).md

What it gathers, from the bank, the ledger and the files, never from Pangram: the day's digest tables (windows, as
posted, efficiency and arms, spend, caps, cuts), each section as posted now and as the day began (the shipped
picks assembled, examples in, comments out, the way score_section builds it), a side by side of every paragraph
that changed today with the arm that wrote it and what the fix pass or the flavor pass touched, the couldn't-crack
report with the reject reasons per stuck run, the previous reflection, the decision log's last entries, and which
sections the author has drafted in the real guide. One Sonnet call, role "loop reflect", through Loop.claude so the
capper and the ledger see it. Tools off.

**The Goodhart guard (plan 6.8).** The notes writer, the judge, the fix pass and the flavor pass never see Pangram's
verdicts, because an agent that sees the detector chases the score. This is the one Claude role that sees the
scores, and it only talks to the author: its output goes to the digest and to analytics/reflections/<day>.md and
nowhere else. It is never cached where another role reads, never put into a prompt of the writing roles, and never
parsed by code into a parameter. Nothing it writes ships in the guide.

**The house check (code).** The prompt carries the rules and the code checks the answer: em and en dashes become a
spaced hyphen; the banned words, the rules file's [reflection] words and invented-number digits (a number that is
in the material it was given is quoted, fine), the banned phrases ("the kind of", the "it's not X, it's Y" frame,
"here's the thing", "the real X is Y", "landed" for impact), a rule-of-three list, a heading beyond the five, a
bullet list, and a length far outside 350 to 600 words all count as hits. A draft with hits gets one retry with
the hits named; a hit on the retry drops its sentence; what then still fails goes to the digest as "the reflection
didn't pass the house check today" with the draft kept as analytics/reflections/<day> (draft).md. The banned
words, the [reflection] words and the digits come from the house rules file at run time (rules.toml, through
engine/checks.py) and the rule of three from
the engine's own scanner (engine/tics/tic_patterns.py), so this file never spells them.

In --smoke the reflection is skipped, and in a rehearsal it is a canned placeholder at zero spend (Loop.claude is
never called, no file is written).
"""
import argparse
import datetime as dt
import re
import sys
from pathlib import Path

from . import checks
from .checks import banned_words, house_words
from .job import flat, style_of
from .settings import SANDBOX, Settings
from .util import describe
from .wrapup import as_posted, norm, restore_examples

HEADINGS = ["What's working", "The writing", "The guide", "Where to push next"]
OPTIONAL_HEADING = "Questions for you"
WORDS_ASKED = (350, 600)
WORDS_HARD = (250, 750)                # outside these the draft is a hit; inside, the length is Sonnet's call
COULDNT_CAP_WORDS = 2500
DIGEST_CUT = "## Couldn't crack so far"
REFLECTIONS_DIR = SANDBOX / "analytics" / "reflections"
DASHES = "[\u2014\u2013]"              # em and en dash, as escapes so this file carries neither

SYSTEM_TEMPLATE = ("You write a short end-of-day reflection to the author of <<DESCRIPTION>>, inside an automated"
                  " pipeline. Output only the reflection in markdown, nothing before it and nothing after it.")
SYSTEM = describe(SYSTEM_TEMPLATE)            # with the default description; the calls fill the job's own

PLACEHOLDER = ("### What's working\n\nRehearsal: the reflection is a placeholder here, so nothing was spent and no"
               " file was written.\n\n### The writing\n\nNothing to read in a rehearsal.\n\n### The guide\n\n"
               "Nothing to read in a rehearsal.\n\n### Where to push next\n\n1. Run a real day.")

# The rules Sonnet is told, word for word what the gate below checks. <<BANNED_WORDS>> is filled at run time from
# the house rules file (banned_word_stems), and <<REFLECTION_RULES>> from its [reflection] section
# (reflection_rule_lines), so every word and digit a rule names comes from the file.
RULES = """How to write it. You're writing to the author: the person who wrote the document and runs this loop. The register is the one the digests use: plain, casual, specific, first person, like talking to a friend. Short paragraphs. No corporate words, no bullet points, no padding, no headings other than the five below, and don't recite what the tables already say unless you're making a point with it. Say what you'd actually say to them.

House rules, checked by code after you answer (a draft that breaks one gets a single retry, then the sentence is cut):
- No em dash or en dash anywhere. A pause or a cut is a spaced hyphen: "for - example".
- Never a word from this list or any word built on one of these stems, as a metaphor or at all: <<BANNED_WORDS>>.
<<REFLECTION_RULES>>- None of these: "the kind of", "the sort of", "the way that", "here's the thing", "the real X is Y", the "it's not X, it's Y" frame or any "not X, but Y" contrast, "landed" or "lands" for impact.
- No rule-of-three lists (three adjectives or three nouns in a row). Two, or one.
- Quoting a line of the document is fine, in double quotes; the rules above don't apply inside a quote.

Length: 350 to 600 words in all.

Headings, exactly these, each on its own line starting with ### and in this order:
### What's working
### The writing
### The guide
### Where to push next
### Questions for you

What goes under each:
- What's working: what moved today and why, with the paragraphs below as evidence (name the window and paragraph, quote a few words of it).
- The writing: a read of the shipped prose. Where it reads like the author, where it has gone flat or calmer or glued together (connective openers, "e.g.", explainer glue), jokes that got lost, slips worth a hand fix.
- The guide: the document as a whole. Which sections are done, which are stuck and on what, how it reads end to end, and what the author's drafting of the other sections means for the loop.
- Where to push next: two to four concrete changes, each with the evidence that suggests it and a rough cost in Pangram or Claude dollars, most worth doing first. Number them 1., 2., and so on, one short paragraph each.
- Questions for you: only when a real decision is the author's to make. Otherwise leave this heading out entirely."""

CONTEXT_TEMPLATE = ("""What this loop is. It takes the author's document (<<DESCRIPTION>>, with citations) and pushes the Pangram AI share of its in-scope sections toward zero while keeping the author's points, facts and voice. Claude (Sonnet) writes notes on each paragraph and judges candidates; a local Mistral 24B base model writes the sentences, in two arms: "notes", an adapter trained on the author's own writing that writes a paragraph from the notes alone (calmer, fewer swears, faithful), and "fewshot", the plain base with worked examples in front (closer to the document's performed register, needs more draws). Code holds every filter and every cap. The score is Pangram's AI share, 0.00 human to 1.00 AI, per window (a stretch of a section the loop works on) and per section as posted (what a reader would paste). The fix pass makes few-word fixes on shipped text; the flavor pass adds one sentence of the author's bite to a paragraph the adapter wrote. "Your lines" are sentences the author wrote by hand; Claude's lines are the ones the loop regrows.

The author's own habits are not slips: trailing hedges, asides in parentheses, one stressed word in CAPS, swearing. The author has accepted the adapter's calmer register, so calmer alone is no fault; flag it only where a joke, a swear or a point was lost.

You are the one Claude role that sees Pangram's scores, and you talk only to the author. Nothing you write goes into the document or into another prompt.""")
CONTEXT = describe(CONTEXT_TEMPLATE)          # with the default description; render_prompt fills the job's own


def banned_word_stems():
    """The banned words and stems from the house rules file (engine/checks.py banned_words), wildcards off."""
    words = []
    for w in banned_words():
        if w not in words:
            words.append(w)
    return words



# ---------------------------------------------------------------------- the day record
def parse_stamp(s):
    try:
        return dt.datetime.strptime((s or "").strip()[:16], "%Y-%m-%d %H:%M").timestamp()
    except ValueError:
        return None


def pick_entry(entries, wid, shape, until=None, ok=None):
    """What the working copy ships for a window at a time: the newest ship, fix or flavor entry with the window's
    run shape (and passing `ok`) when it was banked after the best, else the best scored entry (the lowest AI share,
    humanized texts never). `until` is an epoch time: entries stamped after it don't count. None when the bank has
    nothing for the window yet."""
    cands = []
    for i, e in enumerate(entries):
        if e.get("window") != wid or e.get("runs") is None or [len(p) for p in e["runs"]] != shape:
            continue
        ts = parse_stamp(e.get("time"))
        if until is not None and ts is not None and ts > until:
            continue
        if ok is not None and not ok(e):
            continue
        cands.append((i, e))
    scored = [(i, e) for i, e in cands if e.get("ai") is not None and not e.get("humanized")]
    best = min(scored, key=lambda ie: (ie[1]["ai"], -ie[0])) if scored else None
    ships = [(i, e) for i, e in cands if e.get("kind") in ("ship", "fix", "flavor")]
    if ships and (best is None or ships[-1][0] > best[0]):
        return ships[-1][1]
    return best[1] if best else None


def assemble_sections(job, originals, picks, sections):
    """{section: text as posted} from each window's original paragraphs and its picked paragraphs, the way
    Loop.score_section builds the text Pangram scores: the placeholder base with the picks swapped in, the example
    lines put back, comments and marks out."""
    best = {wid: ([flat(p) for p in originals[wid]], [flat(p) for p in picks[wid]]) for wid in picks}
    _, swaps = job.placeholder_base()
    md = job.assemble(best, write=False)
    return as_posted(restore_examples(md, swaps), sections, style_of(job))


def in_span(ts, span):
    return ts is not None and span[0] - span[0] % 60 <= ts <= span[1]


def changed_paragraphs(entries, windows, start_picks, now_picks, span):
    """One row per paragraph whose shipped text differs between the day's start and now: label, section, before,
    after, whose lines it holds, the arms whose accepted draws wrote into it today, and the fix or flavor passes
    that touched it today. `windows` is {wid: {"section", "runs"}}, `span` the day's (start, end) epoch times."""
    rows = []
    for wid, w in windows.items():
        arms, passes = {}, {}
        for e in entries:
            if e.get("window") != wid or not in_span(parse_stamp(e.get("time")), span):
                continue
            if e.get("kind") in ("round", "context"):
                for ch in e.get("changes") or []:
                    if ch.get("new") and "para" in ch:
                        arms.setdefault(ch["para"], set()).add(ch.get("arm") or "fewshot")
            elif e.get("kind") == "fix":
                for fx in e.get("fixes") or []:
                    passes.setdefault(fx.get("para"), set()).add("fix pass")
            elif e.get("kind") == "flavor":
                for ins in e.get("inserts") or []:
                    passes.setdefault(ins.get("para"), set()).add("flavor pass")
        for pi, (before, after) in enumerate(zip(start_picks[wid], now_picks[wid])):
            b, a = norm(flat(before)), norm(flat(after))
            if b == a:
                continue
            runs = w["runs"][pi] if pi < len(w["runs"]) else []
            yours = sum(1 for r in runs if r.get("author"))
            rows.append({"label": f"{wid} p{pi + 1}", "section": w["section"], "before": b, "after": a,
                         "yours": yours, "arms": sorted(arms.get(pi, ())), "passes": sorted(passes.get(pi, ()))})
    return rows


def pass_notes(entries, span):
    """What the fix pass and the flavor pass did today, as lines."""
    out = []
    for e in entries:
        if not in_span(parse_stamp(e.get("time")), span):
            continue
        if e.get("kind") == "fix":
            for fx in e.get("fixes") or []:
                out.append(f"- fix pass, {e.get('window')} p{(fx.get('para') or 0) + 1} ({fx.get('category')}): "
                           f"\"{fx.get('sentence')}\" -> \"{fx.get('replacement')}\" ({fx.get('why', '')})")
        elif e.get("kind") == "flavor":
            for ins in e.get("inserts") or []:
                out.append(f"- flavor pass, {e.get('window')} p{(ins.get('para') or 0) + 1}: added"
                           f" \"{ins.get('text') or ins.get('sentence', '')}\"")
    return out


def previous_reflection(out_dir=REFLECTIONS_DIR, before_day=None):
    """(name, text) of the newest reflection file, drafts left out, and a file of `before_day` or later left out
    too (the day's own reflection isn't its previous one). None when there is none."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return None
    files = sorted(p for p in out_dir.glob("*.md") if "(draft)" not in p.name)
    if before_day:
        files = [p for p in files if p.name[:10] < before_day]
    if not files:
        return None
    p = files[-1]
    return p.stem, p.read_text(encoding="utf-8")


def decision_log_entries(plan_path=SANDBOX / "plan.md", n=4, chars=700):
    """The first `n` bullets of plan.md's decision log (the newest come first), each cut to `chars`."""
    try:
        text = Path(plan_path).read_text(encoding="utf-8")
    except OSError:
        return []
    m = re.search(r"^## 14\. Decision log\s*$", text, re.M)
    if not m:
        return []
    body = text[m.end():]
    end = re.search(r"^## \d+\. ", body, re.M)
    body = body[:end.start()] if end else body
    out = []
    for line in body.splitlines():
        if line.startswith("- ") and len(out) < n:
            out.append(line[2:].strip()[:chars] + ("..." if len(line) - 2 > chars else ""))
    return out


def guide_state(real_file, in_scope, style=None):
    """Which sections the real guide holds now and how long each is, so the reflection knows what the author has
    drafted beyond the loop's scope. `style` is the job's section style (engine/job.py style_of); every section it
    finds in the file is listed, and the ones in `in_scope` (the job's sections) are marked."""
    try:
        raw = Path(real_file).read_bytes().decode("utf-8")
    except (OSError, TypeError, ValueError):
        return []
    style = style or style_of(None)
    posted = as_posted(raw, style.ids(raw.replace("\r\n", "\n")), style)
    return [(n, len(posted.get(n, "").split()), n in in_scope) for n in sorted(posted)]


def cut_digest(report):
    """The digest's tables (everything before the couldn't-crack section and the timeline), a reflection section
    left out when the digest already carries one."""
    i = report.find(DIGEST_CUT)
    body = report if i < 0 else report[:i]
    j = body.find("## Reflection")
    if j >= 0:
        k = body.find("\n## ", j + 3)
        body = body[:j] + (body[k + 1:] if k >= 0 else "")
    return body.strip()


def day_span_from_ledger(conn, day, now):
    """(start, end) of a run-day from the ledger's run_days table, else midnight of `day` to `now`."""
    row = None
    if conn is not None:
        try:
            row = conn.execute("SELECT started, ended FROM run_days WHERE day = ?", (day,)).fetchone()
        except Exception:  # noqa: BLE001
            row = None
    midnight = dt.datetime.strptime(day, "%Y-%m-%d").timestamp()
    if row and row[0]:
        return float(row[0]), float(row[1]) if row[1] else now
    return midnight, now


def record_from_loop(loop):
    """The day record from a running loop at its wrap-up: the digest as it stands, the loop's shipped picks and the
    bank's picks as the day began."""
    from engine import discord
    day = getattr(loop, "day", None) or loop.schedule.day_key(loop.clock.now())
    now = loop.clock.now()
    start = getattr(loop.budget, "run_day_start", None) or getattr(loop, "smoke_t0", None) or now - 14 * 3600
    secs = loop.whole_sections()
    wss = [ws for ws in loop.windows if ws.w["section"] in secs and getattr(ws, "ready", False)]
    windows = {ws.wid: {"section": ws.w["section"], "runs": ws.runs} for ws in wss}
    originals = {ws.wid: ws.original for ws in wss}
    now_picks = {ws.wid: loop.pick_runs(ws) for ws in wss}
    start_picks = {}
    for ws in wss:
        e = pick_entry(loop.bank.entries, ws.wid, [len(p) for p in ws.runs], until=start)
        start_picks[ws.wid] = e["runs"] if e else [[r[:] for r in p] for p in ws.original]
    return {"day": day, "stopped": loop.stop_reason or "crash", "model": loop.model,
            "arms": f"primary {loop.primary_arm}, fallback {loop.fallback_arm or 'none'}",
            "digest": cut_digest(discord.digest_report(loop)), "sections": secs,
            "now": assemble_sections(loop.job, originals, now_picks, secs) if wss else {},
            "start": assemble_sections(loop.job, originals, start_picks, secs) if wss else {},
            "changed": changed_paragraphs(loop.bank.entries, windows, start_picks, now_picks, (start, now)),
            "passes": pass_notes(loop.bank.entries, (start, now)),
            "couldnt": discord.couldnt_crack_report(loop),
            "previous": previous_reflection(before_day=day),
            "decisions": decision_log_entries(),
            "guide": guide_state(loop.real_file, loop.job.sections, style_of(loop.job)) if loop.real_file else [],
            "description": getattr(loop.job, "description", None),
            "guide_scan": guide_scan_record(getattr(loop, "guide_scan", None), getattr(loop, "sections_combined", None))}


def guide_scan_record(gs, comb):
    """{ai, words, flagged, skipped, sections_combined} from the loop's result (or a bank `guide` entry) and the
    sections combined (the loop's dict, or the entry's number); None when the day has neither (plan 9)."""
    if not gs and not comb:
        return None
    gs = gs or {}
    c = comb.get("ai") if isinstance(comb, dict) else comb
    return {"ai": gs.get("ai"), "words": gs.get("words"), "flagged": gs.get("flagged"),
            "skipped": "a placeholder in a rehearsal" if gs.get("placeholder") else gs.get("skipped"),
            "sections_combined": c if c is not None else gs.get("sections_combined")}


def guide_scan_from_bank(entries, day):
    """The day's last `guide` entry, as guide_scan_record reads it."""
    hits = [e for e in entries if e.get("kind") == "guide" and e.get("day") == day]
    return guide_scan_record(hits[-1], None) if hits else None


def record_from_files(day, settings, job, bank, conn, digests_dir=None, now=None):
    """The day record without a loop: the digest file for the day, the bank's shipped picks now and as the day
    began (the ledger's run_days table says when the day began), the couldn't-crack file. "Today" runs from the
    day's start to `now`, so a fix pass or flavor pass run after the loop stopped counts as today's."""
    from .wrapup import layout
    now = now or dt.datetime.now().timestamp()
    start, _ = day_span_from_ledger(conn, day, now)
    end = now
    lay = layout(job, bank)
    windows = {wid: {"section": w["section"], "runs": w["runs"]} for wid, w in lay.items()}
    originals = {wid: w["original"] for wid, w in lay.items()}
    now_picks, start_picks = {}, {}
    for wid, w in lay.items():
        shape = [len(p) for p in w["original"]]
        e = pick_entry(bank.entries, wid, shape)
        now_picks[wid] = e["runs"] if e else [[r[:] for r in p] for p in w["original"]]
        e0 = pick_entry(bank.entries, wid, shape, until=start)
        start_picks[wid] = e0["runs"] if e0 else [[r[:] for r in p] for p in w["original"]]
    dpath = Path(digests_dir or SANDBOX / "digests") / f"{day} digest.md"
    digest = cut_digest(dpath.read_text(encoding="utf-8")) if dpath.exists() else f"(no digest file for {day})"
    lc = settings.config.get("loop", {}) if settings is not None else {}
    return {"day": day, "stopped": "(read from the files after the day)", "model": lc.get("model", "?"),
            "arms": f"primary {lc.get('primary_arm', 'fewshot')}, fallback {lc.get('fallback_arm') or 'none'}",
            "digest": digest, "sections": list(job.sections),
            "now": assemble_sections(job, originals, now_picks, job.sections),
            "start": assemble_sections(job, originals, start_picks, job.sections),
            "changed": changed_paragraphs(bank.entries, windows, start_picks, now_picks, (start, end)),
            "passes": pass_notes(bank.entries, (start, end)),
            "couldnt": job.couldnt_crack.read_text(encoding="utf-8") if job.couldnt_crack.exists() else "(none yet)",
            "previous": previous_reflection(before_day=day),
            "decisions": decision_log_entries(),
            "guide": guide_state(job.real_file, job.sections, style_of(job)) if job.real_file else [],
            "description": getattr(job, "description", None),
            "guide_scan": guide_scan_from_bank(bank.entries, day)}


# ---------------------------------------------------------------------- the prompt
def cap_words(text, n):
    """The first `n` words of a text, line breaks kept, with a note on what was cut."""
    total = len(text.split())
    if total <= n:
        return text
    count, out = 0, []
    for line in text.splitlines():
        words = line.split()
        if count + len(words) > n:
            out.append(" ".join(words[:n - count]))
            break
        out.append(line)
        count += len(words)
    return "\n".join(out).rstrip() + f"\n\n[cut here: {total - n} more words in the file]"


def render_material(rec):
    """Everything Sonnet reads about the day, as one markdown document (the rules go in a separate part)."""
    L = [f"# The day: {rec['day']}", "",
         f"Stopped: {rec['stopped']}. Writer model `{rec['model']}` ({rec['arms']}). Sections in scope:"
         f" {', '.join(map(str, rec['sections']))}.", "",
         "## The day's digest tables", "", rec["digest"], ""]
    L += ["## Each section as posted now (what a reader pastes)", ""]
    for n in rec["sections"]:
        if n in rec["now"]:
            L += [f"### Section {n}", "", rec["now"][n], ""]
    L += ["## Paragraphs that changed today (as the day began, then now)", ""]
    if rec["changed"]:
        for r in rec["changed"]:
            who = f"; holds {r['yours']} line(s) of yours" if r["yours"] else ""
            wrote = (f"written by the {' and '.join(r['arms'])} arm{'s' if len(r['arms']) > 1 else ''}" if r["arms"]
                     else "no accepted draw banked today (a whole-text pick or an earlier best)")
            touched = ("; " + " and ".join(r["passes"]) + " touched it") if r["passes"] else ""
            L += [f"### {r['label']} (section {r['section']}; {wrote}{who}{touched})", "",
                  f"Before: {r['before']}", "", f"After: {r['after']}", ""]
    else:
        L += ["No paragraph's shipped text changed today.", ""]
    same = [n for n in rec["sections"] if n in rec["now"] and n in rec["start"] and rec["now"][n] == rec["start"][n]]
    if same:
        L += [f"Sections {', '.join(map(str, same))} read now exactly as they did when the day began.", ""]
    L += ["## The fix pass and the flavor pass today", ""]
    L += rec["passes"] or ["Neither ran today, or neither changed anything."]
    L += ["", "## Couldn't crack (every stuck run, with what was tried)", "",
          cap_words(rec["couldnt"], COULDNT_CAP_WORDS), ""]
    if rec.get("previous"):
        name, text = rec["previous"]
        L += [f"## The previous reflection ({name})", "",
              "Don't repeat it. Say what moved since it, and where it was wrong.", "", text, ""]
    else:
        L += ["## The previous reflection", "", "None yet: this is the first.", ""]
    L += ["## The decision log's last entries (newest first)", ""]
    L += [f"- {d}" for d in rec["decisions"]] or ["(none read)"]
    L += ["", "## The rest of the guide", ""]
    if rec["guide"]:
        in_scope = [f"{n} ({w} words)" for n, w, s in rec["guide"] if s]
        others = [f"{n} ({w} words)" for n, w, s in rec["guide"] if not s]
        L.append(f"The real guide holds sections {', '.join(str(n) for n, _, _ in rec['guide'])}. In the loop's scope:"
                 f" {', '.join(in_scope) or 'none'}. Drafted by the author outside the loop's scope so far:"
                 f" {', '.join(others) or 'none yet'}. The sync step (python -m engine.sync) brings a finished section"
                 " into scope when the author hands the file over.")
    else:
        L.append("The real guide couldn't be read.")
    gs = rec.get("guide_scan")
    if gs:
        # plan 9: the whole guide's number and the sections combined, so the reflection can say what the
        # whole guide does, not only the sections in scope
        if gs.get("ai") is not None:
            whole = (f"Scored whole as it stands today, every section as a reader would paste the file:"
                     f" {gs['ai']:.2f} AI share over {gs.get('words') or 0:,} words"
                     + (f" ({gs['flagged']} segments flagged)" if gs.get("flagged") is not None else "") + ".")
        else:
            whole = f"The whole guide as it stands wasn't scored today ({gs.get('skipped') or 'no scoring'})."
        comb = gs.get("sections_combined")
        if comb is not None:
            whole += f" The sections in scope combined, weighted by words: {comb:.2f}."
        L.append(whole)
    return "\n".join(L)


def render_prompt(rec, material=None):
    material = material if material is not None else render_material(rec)
    rules = (RULES.replace("<<BANNED_WORDS>>", ", ".join(banned_word_stems()))
             .replace("<<REFLECTION_RULES>>", "".join(f"{line}\n" for line in reflection_rule_lines())))
    context = describe(CONTEXT_TEMPLATE, rec.get("description"))
    return f"{context}\n\n{rules}\n\nThe material follows. Write the reflection after it.\n\n{material}"


def system_prompt(rec):
    """The system prompt with the job's description (the record's "description", else the default)."""
    return describe(SYSTEM_TEMPLATE, rec.get("description"))


# ---------------------------------------------------------------------- the house check
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?:%|st|nd|rd|th)?(?![\w])")


def _spelled():
    """{number word: its digits} for the one-word numbers: one to nineteen, the tens, their ordinals and plurals."""
    ones = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
            " seventeen eighteen nineteen").split()
    tens = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
            "ninety": 90, "hundred": 100, "thousand": 1000}
    irregular = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                 "nine": "ninth", "twelve": "twelfth"}
    out = {}
    for word, n in list(zip(ones, range(20))) + list(tens.items()):
        out[word] = str(n)
        out[word + "s"] = str(n)
        ordinal = irregular.get(word) or (word[:-1] + "ieth" if word.endswith("y") else word + "th")
        out[ordinal] = str(n)
    out.pop("seconds", None)
    out.pop("second", None)                 # "a second" is mostly time, not an ordinal
    return out


SPELLED = _spelled()
SPELLED_RE = re.compile(r"\b(" + "|".join(sorted(SPELLED, key=len, reverse=True)) + r")\b", re.I)
QUOTE_SPAN = re.compile(r"\"[^\"\n]{1,400}\"")
HEADING_LINE = re.compile(r"^\s*#{1,6}\s*(.*?)\s*$")
BULLET_LINE = re.compile(r"^\s*[-*•]\s+")
PHRASES = [
    ("the kind of", re.compile(r"\bthe (?:kind|sort|type) (?:of|that|who|where)\b|\bthe way that\b", re.I)),
    ("here's the thing", re.compile(r"\bhere's the thing\b", re.I)),
    ("the real X is Y", re.compile(r"\bthe real (?:\w+ ){1,3}(?:is|was|isn't|wasn't)\b", re.I)),
    ("it's not X, it's Y", re.compile(
        r"\b(?:it's|it is|this is|that's|that is|they're|that was|it was|the problem is|the issue is) not "
        r"(?:about |just |only |that |a |the |so much )?[^.;:\n]{1,60}?[,;] (?:it's|it is|but|they're|that's|so much as)\b"
        r"|\bnot (?:just |only |so much )?[^.;,\n]{1,40}, but\b", re.I)),
    ("landed for impact", re.compile(r"\bland(?:ed|s)\b", re.I)),
]
_TICS = None


# The rule of three, from the engine's scanner (engine/tics/tic_patterns.py) so there's one list, with the same
# patterns here for a scanner that lacks the label.
OWN_TICS = {
    "rule of three": [r"\b[a-z]+, [a-z]+,? (?:and|or) [a-z]+\b(?!-)",
                      r"(?:-|,) [a-z]+, [a-z]+, (?:[a-z]+ )?[a-z]+(?=[.!?\"])"],
}
TIC_LABELS = {"rule of three": "rule of three"}


def reflection_rules():
    """The rules file's [reflection] section, as the rules in use hold it (engine/checks.py use_rules)."""
    return checks.house_rules()["reflection"]


def reflection_rule_lines():
    """The prompt's lines for the [reflection] rules: one per banned word, one for the invented-number digits."""
    r = reflection_rules()
    lines = [f'- Never "{w}" in any form.' for w in checks._house_rules_file.stems(r["banned_words"])]
    digits = r["invented_digits"]
    if digits:
        lines.append("- Numbers: quote only numbers that appear in the material below, written as they are there."
                     " Never make a figure up for texture, and never one with "
                     + " or ".join(f"a {d}" for d in digits) + " in it.")
    return lines


def _tic_rules():
    """[(rule, regex)]: the rule of three, from the engine's scanner when it has that label, else from OWN_TICS;
    then each of the [reflection] banned words as its own rule, named by the word."""
    global _TICS
    if _TICS is None:
        try:
            from engine.tics.tic_patterns import RULES as SCAN
        except Exception:  # noqa: BLE001
            SCAN = []
        rules = []
        for name, own in OWN_TICS.items():
            found = [re.compile(r["regex"], 0 if r.get("case") else re.I) for r in SCAN
                     if r["label"] == TIC_LABELS[name]]
            rules += [(name, rx) for rx in (found or [re.compile(p, re.I) for p in own])]
        _TICS = rules
    words = reflection_rules()["banned_words"]
    hf = checks._house_rules_file
    return _TICS + [(stem, hf.word_regex([w])) for w, stem in zip(words, hf.stems(words))]


def fix_dashes(text):
    """Em and en dashes and a "--" become a spaced hyphen, the house way (a mechanical fix, never a retry)."""
    text = re.sub(r"\s*" + DASHES + r"\s*", " - ", text)
    text = re.sub(r"\s*--\s*", " - ", text)
    return re.sub(r" {2,}", " ", text)


def _forms(tok):
    """A number token's comparable forms: as written (commas out, % and ordinals off) and as a float string."""
    tok = re.sub(r"(?:%|st|nd|rd|th)$", "", tok).replace(",", "")
    out = {tok}
    try:
        out.add(repr(float(tok)))
    except ValueError:
        pass
    return out


def allowed_numbers(material):
    """Every number in the material: quoting one of these is counted, fine."""
    out = set()
    for m in NUMBER.finditer(material):
        out |= _forms(m.group(0))
    for w, d in SPELLED.items():
        if re.search(r"\b" + w + r"\b", material, re.I):
            out.add(d)
    return out


def sentence_hits(sentence, allowed):
    """The house rules a sentence breaks (quoted spans don't count), as (rule, match) pairs."""
    body = QUOTE_SPAN.sub('""', sentence)
    hits = [("banned word", w) for w in house_words(body) if not re.fullmatch(DASHES + r"|--", w)]
    for rule, rx in _tic_rules():
        hits += [(rule, m.group(0)) for m in rx.finditer(body)]
    for rule, rx in PHRASES:
        hits += [(rule, m.group(0)) for m in rx.finditer(body)]
    digits = reflection_rules()["invented_digits"]
    if not digits:
        return hits
    rule = "invented number with " + " or ".join(f"a {d}" for d in digits)
    for m in NUMBER.finditer(body):
        forms = _forms(m.group(0))
        if any(d in f for f in forms for d in digits) and not forms & allowed:
            hits.append((rule, m.group(0)))
    for m in SPELLED_RE.finditer(body):
        value = SPELLED[m.group(1).lower()]
        if any(d in value for d in digits) and value not in allowed:
            hits.append((rule, m.group(0)))
    return hits


def _sents(text):
    from engine.draws import sents
    return sents(text)


def body_words(text):
    return [w for line in text.splitlines() if not HEADING_LINE.match(line) for w in line.split()]


def gate(text, allowed, strict_length=True):
    """Every hit in a draft: {"rule", "match", "sentence"} per rule broken, plus heading, bullet and length hits
    (those carry no sentence). An empty list means the draft passes."""
    hits, found = [], []
    for line in text.splitlines():
        m = HEADING_LINE.match(line)
        if m:
            found.append(m.group(1).strip().rstrip(":"))
            continue
        if BULLET_LINE.match(line):
            hits.append({"rule": "bullet list", "match": line.strip()[:60], "sentence": None})
    core = [h for h in found if h in HEADINGS]
    if core != HEADINGS:
        missing = [h for h in HEADINGS if h not in found]
        hits.append({"rule": "headings", "match": ("missing: " + ", ".join(missing)) if missing else "out of order",
                     "sentence": None})
    for h in found:
        if h not in HEADINGS and h != OPTIONAL_HEADING:
            hits.append({"rule": "headings", "match": f"a heading beyond the five: {h}", "sentence": None})
    n = len(body_words(text))
    if strict_length and not (WORDS_HARD[0] <= n <= WORDS_HARD[1]):
        hits.append({"rule": "length", "match": f"{n} words (asked for {WORDS_ASKED[0]} to {WORDS_ASKED[1]})",
                     "sentence": None})
    for line in text.splitlines():
        if HEADING_LINE.match(line) or not line.strip():
            continue
        for s in _sents(line):
            for rule, match in sentence_hits(s, allowed):
                hits.append({"rule": rule, "match": match, "sentence": s})
    return hits


def drop_sentences(text, hits):
    """The draft with every sentence a hit names taken out. Headings stay; a line left empty goes."""
    bad = {h["sentence"] for h in hits if h.get("sentence")}
    if not bad:
        return text, []
    out, dropped = [], []
    for line in text.splitlines():
        if HEADING_LINE.match(line) or not line.strip():
            out.append(line)
            continue
        m = re.match(r"^(\s*(?:\d+\.\s+)?)(.*)$", line)
        lead, rest = m.group(1), m.group(2)
        keep = []
        for s in _sents(rest):
            if s in bad:
                dropped.append(s)
            else:
                keep.append(s)
        if keep:
            out.append(lead + " ".join(keep))
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"
    return text, dropped


def describe_hits(hits):
    return "; ".join(f"{h['rule']}: \"{h['match']}\"" + (f" in \"{h['sentence'][:90]}\"" if h.get("sentence") else "")
                     for h in hits)


# ---------------------------------------------------------------------- the call
def reflect(rec, claude_fn, model="sonnet", log=print):
    """The reflection for a day record: one call, a retry with the hits named when the first draft breaks a rule,
    the offending sentences cut when the retry does too. `claude_fn(prompt)` returns (text, meta) and may raise the
    caller's cap or call errors. Returns {"ok", "text", "draft", "hits", "attempts", "cost", "model", "dropped"}."""
    material = render_material(rec)
    allowed = allowed_numbers(material)
    prompt = render_prompt(rec, material)
    cost, attempts = 0.0, 0
    text, meta = claude_fn(prompt)
    attempts += 1
    cost += float((meta or {}).get("cost_usd") or 0.0)
    text = fix_dashes((text or "").strip())
    hits = gate(text, allowed)
    if hits:
        log(f"reflection: the first draft broke the house rules ({describe_hits(hits)[:300]}); one retry")
        retry = (prompt + "\n\nYour first draft broke these house rules: " + describe_hits(hits)
                 + ". Write the whole reflection again, keeping what was right, without them.")
        text, meta = claude_fn(retry)
        attempts += 1
        cost += float((meta or {}).get("cost_usd") or 0.0)
        text = fix_dashes((text or "").strip())
        hits = gate(text, allowed)
    dropped = []
    if hits:
        text, dropped = drop_sentences(text, hits)
        hits = gate(text, allowed, strict_length=False)
        if dropped:
            log(f"reflection: {len(dropped)} sentence(s) cut by the house check")
    ok = not hits
    if not ok:
        log(f"reflection: didn't pass the house check ({describe_hits(hits)[:300]})")
    return {"ok": ok, "text": text if ok else None, "draft": None if ok else text, "hits": hits,
            "attempts": attempts, "cost": round(cost, 4), "model": model, "dropped": dropped}


def write_reflection(result, day, out_dir=REFLECTIONS_DIR, tag=None, when=None):
    """The reflection to analytics/reflections/<day>.md (or "<day> (<tag>).md"), one header line on top; a failed
    one to "<day> (draft).md" with the hits under it. Returns the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    when = when or dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    name = f"{day} ({tag})" if tag else day
    cut = f", {len(result['dropped'])} sentence(s) cut by the house check" if result.get("dropped") else ""
    head = (f"Written {when} by `python -m engine.reflect`, model {result['model']}, about ${result['cost']:.2f} of"
            f" Claude, {result['attempts']} draft(s){cut}.")
    if result["ok"]:
        p = out_dir / f"{name}.md"
        p.write_text(f"# Reflection, {name}\n\n{head}\n\n{result['text'].strip()}\n", encoding="utf-8")
    else:
        p = out_dir / f"{name} (draft).md"
        caught = "\n".join(f"- {h['rule']}: \"{h['match']}\"" + (f" in \"{h['sentence']}\"" if h.get("sentence") else "")
                           for h in result["hits"])
        p.write_text(f"# Reflection, {name} (draft: didn't pass the house check)\n\n{head}\n\n"
                     f"{(result['draft'] or '').strip()}\n\n## What the house check caught\n\n{caught}\n",
                     encoding="utf-8")
    return p


def section_of(text, heading):
    """The body under one ### heading of a reflection."""
    on, out = False, []
    for line in text.splitlines():
        m = HEADING_LINE.match(line)
        if m:
            on = m.group(1).strip().rstrip(":") == heading
            continue
        if on:
            out.append(line)
    return "\n".join(out).strip()


def card_snippet(text, room):
    """For the Discord card: the first paragraph of What's working and the first line of Where to push next, cut at
    a sentence end to fit `room` characters."""
    working = section_of(text, HEADINGS[0])
    push = section_of(text, HEADINGS[3])
    first = next((p.strip() for p in re.split(r"\n\s*\n", working) if p.strip()), "")
    line = next((ln.strip() for ln in push.splitlines() if ln.strip()), "")
    parts = (["Reflection: " + first] if first else []) + (["Push next: " + line] if line else [])
    out = "\n".join(parts)
    if len(out) <= room:
        return out
    cut = out[:max(0, room)]
    end = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("."))
    return cut[:end + 1] if end > 40 else cut.rstrip()


def run_for_loop(loop):
    """The wrap-up's call (engine/run.py's finish): sets loop.reflection. Skipped in --smoke and for a partial
    (--window) run, a placeholder at zero spend in a rehearsal, else one call through Loop.claude."""
    from engine.budget import ClaudeCapReached
    from engine.claude_call import ClaudeCallError
    if loop.args.smoke:
        loop.reflection = {"skipped": "a smoke test"}
        return loop.reflection
    if loop.args.window or not loop.whole_sections():
        loop.reflection = {"skipped": "a partial run only knows some windows, so the sections can't be read whole"}
        return loop.reflection
    if loop.rehearsal:
        # the gather runs for real in a rehearsal (the one chance to see it work before a real wrap-up) and its
        # input goes to the rehearsal folder to read; the call itself is the placeholder, at zero spend
        try:
            rec = record_from_loop(loop)
            out = loop.settings.runs_dir / loop.job.name / f"reflect input {rec['day']}.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(render_prompt(rec), encoding="utf-8")
            loop.log(f"reflection: rehearsal, the gathered input is {out.name} ({len(rec['changed'])} changed"
                     f" paragraph(s)); the call is a placeholder")
        except Exception as e:  # noqa: BLE001
            loop.log(f"reflection: REHEARSAL GATHER FAILED ({type(e).__name__}: {str(e)[:160]}); a real wrap-up"
                     " would skip the reflection here")
        loop.reflection = {"ok": True, "text": PLACEHOLDER, "cost": 0.0, "model": "rehearsal", "attempts": 0,
                           "dropped": [], "placeholder": True}
        return loop.reflection
    try:
        rec = record_from_loop(loop)
    except Exception as e:  # noqa: BLE001
        why = f"the gather failed ({type(e).__name__}: {str(e)[:120]})"
        loop.log(f"reflection: {why}")
        loop.reflection = {"skipped": why}
        return loop.reflection
    cfg = loop.settings.config.get("reflect", {})
    model = str(cfg.get("model", "sonnet"))
    max_usd = float(cfg.get("max_usd", 0.60))

    def call(prompt):
        return loop.claude(loop.settings, loop.conn, prompt, "reflect", model=model, max_budget_usd=max_usd,
                           system=system_prompt(rec), timeout=900)
    try:
        result = reflect(rec, call, model=model, log=loop.log)
    except ClaudeCapReached as e:
        loop.on_cap(e)
        loop.reflection = {"skipped": f"Claude is capped ({e.check}: {e.detail})"}
        return loop.reflection
    except ClaudeCallError as e:
        loop.reflection = {"skipped": f"the Claude call failed ({str(e)[:120]})"}
        return loop.reflection
    out_dir = getattr(loop, "reflections_dir", None) or REFLECTIONS_DIR
    path = write_reflection(result, rec["day"], out_dir=out_dir, when=loop.clock.stamp())
    result["path"] = str(path)
    loop.log(f"reflection {'written' if result['ok'] else 'did not pass the house check; draft kept'}:"
             f" {path.name} (${result['cost']:.2f}, {result['attempts']} draft(s))")
    loop.reflection = result
    return result


# ---------------------------------------------------------------------- by hand
def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="the end-of-day reflection for the author (plan 9, 6.8)")
    ap.add_argument("--job", default=None, help="the job's name (default: the job engine/job.py picks)")
    ap.add_argument("--day", help="the run-day, YYYY-MM-DD (default today)")
    ap.add_argument("--dry", action="store_true", help="gather only: write the rendered input to the runs folder")
    ap.add_argument("--max-usd", type=float, default=None,
                    help="the call's ceiling (default config's reflect.max_usd, 0.60)")
    ap.add_argument("--model", help="default config's reflect.model (sonnet)")
    ap.add_argument("--tag", help="a word for the file name: analytics/reflections/<day> (<tag>).md")
    a = ap.parse_args(argv)

    from . import db
    from .budget import Budget, ClaudeCapReached
    from .claude_call import ClaudeCallError, run as claude_run
    from .job import Bank, Job
    from .schedule import Clock, Schedule
    settings = Settings()
    cfg = settings.config.get("reflect", {})
    model = a.model or str(cfg.get("model", "sonnet"))
    max_usd = float(a.max_usd if a.max_usd is not None else cfg.get("max_usd", 0.60))
    day = a.day or dt.date.today().isoformat()
    conn = db.connect(settings.db_path)
    job = Job(a.job)
    bank = Bank(job.bank_path)
    rec = record_from_files(day, settings, job, bank, conn)
    if a.dry:
        prompt = render_prompt(rec)
        out = settings.runs_dir / job.name / f"reflect input {day}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(prompt, encoding="utf-8")
        print(f"dry run: the input is {len(prompt.split())} words, {len(rec['changed'])} changed paragraph(s),"
              f" {len(rec['passes'])} pass line(s), previous reflection"
              f" {rec['previous'][0] if rec['previous'] else 'none'}; written to {out}")
        return 0
    clock = Clock()
    budget = Budget(settings, conn, clock, Schedule(settings.config))

    def call(prompt):
        budget.claude_check("reflect", max_usd)
        return claude_run(settings, conn, prompt, role="loop reflect", model=model, system=system_prompt(rec),
                          max_budget_usd=max_usd, timeout=900)
    try:
        result = reflect(rec, call, model=model)
    except ClaudeCapReached as e:
        print(f"the Claude capper said no: {e}")
        return 1
    except ClaudeCallError as e:
        print(f"the Claude call failed: {str(e)[:300]}")
        return 1
    path = write_reflection(result, day, tag=a.tag)
    print(f"{'written' if result['ok'] else 'did not pass the house check; draft kept'}: {path}"
          f" (${result['cost']:.2f}, {result['attempts']} draft(s), {len(result['dropped'])} sentence(s) cut)")
    print()
    print((result["text"] or result["draft"] or "").strip())
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
