"""The repair step (the author's rules of 2026-11-14: factuality comes before the score, and repair comes first).

    python -m engine.repair                 # read, repair, check, write review.json, score what changed
    python -m engine.repair --dry           # list the spans it would read; no Claude, no Pangram, no file
    python -m engine.repair --section 5     # some sections only

For each section in scope it takes what the paragraph pass serves (RegrowPass.section_pick) and reads every picked
span that isn't pinned (no standing repair, no hand fix) and hasn't been read in this exact form. Per span:

  1. Opus reads the rewrite against the original span and the paragraph's notes for six kinds of slip (the five of
     REGROW_FACT_PROMPT, word for word, and a dropped point that matters) and answers CLEAN, UNFIXABLE: <reason>, or a
     JSON list of small moves: a cut (one sentence of the rewrite out), a swap (up to three words of one sentence
     changed) or a splice (one sentence of the original back in, word for word).
  2. Code decides (apply_moves). Each move names its sentences exactly (a cut, or the place of a splice, may name a
     run of sentences in a row; a swap names one); a swap goes through the fix pass's gate
     (engine/fixpass.py: the word-level diff, no new number, name, swear, dash or house word, the must-keeps); at
     most two splices; and a result made mostly of the original's own sentences is no repair (a span of twelve words
     or fewer may go back whole). Anything else fails the whole proposal. An answer that fails here, or doesn't
     parse, gets one more try: the same prompt with the answer and the reason quoted (RETRY_NOTE).
  3. A second Opus call checks the repaired span against the original for the same six kinds; it must answer NONE.

The outcomes go in the reviewer's file (engine/review.py), appended, with nothing there dropped or reordered: a
repair; a veto when the span failed, so the pass picks something else; or, once the same original span has failed
on `stuck_after` different picks, a hand fix for the author that is the original span word for word, with a prompt
built from the notes. A clean span is only logged. Every read goes in `repair log.jsonl` in the job folder, keyed
by the original span, the pick and REPAIR_VERSION, so a span is paid for once; a read made in propose mode is
replayed for free by the next apply run. The money ceiling is checked before each span's read, and a read always
gets its check, so a run can pass the ceiling by one call at most.

Every entry names the scoring that was served when its span was read (`scoring`), and the pass reads the section's
repairs against the newest scoring its entries name (RegrowPass.review_base), so a veto written beside a repair never
makes it stale. A veto can still change what is served in one case: a section with no pins (no repair, no hand fix)
whose vetoed scoring was its best falls back to its next eligible scoring, whose picks nobody has read. So after
writing, the step loads the pass again and reads any served span it hasn't read, up to READ_ROUNDS times; in every
other case that second look finds nothing. Then every section whose served text changed is scored once as posted
(RegrowPass.final), under the step's own Pangram ceiling, purpose "loop".

config.toml, [repair]: mode ("apply", "propose" or "off"), model, max_words_changed, max_splices,
max_original_share, stuck_after, call_usd, max_claude_usd, max_pangram_usd.
"""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

from . import review as rv
from .checks import split_sentences
from .job import NOTES_VERSION, Bank, Job, sha
from .regrow import ORIGS_FILE, REGROW_FACT_PROMPT, RegrowPass, orig_key

REPAIR_VERSION = "repair 2026-11-14 v1"
LOG = "repair log.jsonl"
READ_ROUNDS = 3
SHORT_SPAN = 12           # words: an original span this short may go back to its original whole
MODES = ("apply", "propose", "off")
DEFAULTS = {"mode": "apply", "model": "opus", "max_words_changed": 3, "max_splices": 2, "max_original_share": 0.6,
            "stuck_after": 2, "call_usd": 0.5, "max_claude_usd": 3.0, "max_pangram_usd": 3.0}
MOVES = {"cut": ("sentence",), "swap": ("sentence", "to"), "splice": ("after", "sentence")}

# the pass's own five kinds of slip, word for word with their examples, and a sixth for what a rewrite drops
KINDS = "\n".join(re.findall(r"^[1-5]\. .*$", REGROW_FACT_PROMPT, re.M)) + """
6. A dropped point that matters: a fact, a named step in a numbered sequence (the original says "Step two is where..." and the rewrite drops it, so the steps skip from one to three), or an observation the text after it depends on.

A dropped joke, a different joke that claims nothing new, or looser wording is not a slip."""

READ_PROMPT = """A rewrite of one stretch of an essay has to say what the original stretch says, and keep every point of it that matters. Read the rewrite against the original and the notes for these kinds of slip:

<<KINDS>>

Original stretch:
<<ORIGINAL>>

Notes for the paragraph:
<<NOTES>>

Rewrite:
<<NEW>>

If the rewrite has none of these slips, reply CLEAN and nothing else.

If it has one, mend it with the fewest small moves, keeping the rewrite's own words everywhere else. Every move names sentences of the rewrite exactly as they stand above, punctuation included, never as an earlier move left them. The moves:
- cut: take out one sentence of the rewrite that the original doesn't have.
- swap: change up to <<N>> words in one sentence of the rewrite; it stays one sentence.
- splice: put one sentence of the original back in, word for word, after a sentence of the rewrite, or at the start ("after": "START"). At most <<SPLICES>> splices.

Reply with only a JSON list of moves, each with a short "why", and nothing else:
[{"move": "cut", "sentence": "<one sentence of the rewrite, exactly>", "why": "..."},
 {"move": "swap", "sentence": "<one sentence of the rewrite, exactly>", "to": "<that sentence with up to <<N>> words changed>", "why": "..."},
 {"move": "splice", "after": "<one sentence of the rewrite, exactly, or START>", "sentence": "<one sentence of the original, exactly>", "why": "..."}]

If small moves can't make the rewrite say what the original says, reply UNFIXABLE: and the reason, on one line."""

RETRY_NOTE = """

Your last answer couldn't be used. It was:
<<ANSWER>>

The code that applies the moves said: <<WHY>>

Answer again in the same form. Copy every "sentence" and "after" from the rewrite above exactly as it stands, punctuation included: one sentence, or several sentences in a row exactly as they run. A swap names exactly one sentence."""

CHECK_PROMPT = """A rewrite of one stretch of an essay has to say what the original stretch says, and keep every point of it that matters. List every problem of these kinds in the rewrite:

<<KINDS>>

Original stretch:
<<ORIGINAL>>

Notes for the paragraph:
<<NOTES>>

Rewrite:
<<NEW>>

Reply with NONE when there is no problem, or one short line per problem."""


def squash(t):
    return rv.squash(t)


def config(settings):
    """The [repair] block with the defaults in code under it."""
    cfg = dict(DEFAULTS)
    cfg.update((settings.config.get("repair") if settings is not None else None) or {})
    return cfg


def span_key(orig, pick):
    """The log's key for one read: the original span, the pick and the step's version."""
    return sha(REPAIR_VERSION + "\x1f" + squash(orig) + "\x1f" + squash(pick))


def fill(prompt, orig, notes, new, cfg):
    return (prompt.replace("<<KINDS>>", KINDS).replace("<<ORIGINAL>>", orig)
            .replace("<<NOTES>>", "\n".join(f"- {n}" for n in notes) or "(no notes)").replace("<<NEW>>", new)
            .replace("<<N>>", str(cfg["max_words_changed"])).replace("<<SPLICES>>", str(cfg["max_splices"])))


def parse_answer(text):
    """The read's answer as ("clean", None), ("unfixable", reason), ("moves", [move, ...]) or ("bad", reason). A
    JSON list may sit in a code fence; anything else around it is a bad answer."""
    t = (text or "").strip()
    if t.upper().startswith("CLEAN"):
        return "clean", None
    if t.upper().startswith("UNFIXABLE"):
        return "unfixable", squash(t.split(":", 1)[1]) if ":" in t else "no reason given"
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", t, re.S)
    if m:
        t = m.group(1)
    try:
        moves = json.loads(t)
    except ValueError:
        return "bad", "the answer isn't CLEAN, UNFIXABLE or a list of moves"
    if not isinstance(moves, list) or not moves or not all(isinstance(x, dict) for x in moves):
        return "bad", "the answer isn't CLEAN, UNFIXABLE or a list of moves"
    return "moves", moves


def sentences(text):
    return [squash(s) for s in split_sentences(squash(text))]


def apply_moves(moves, pick, orig, notes=(), max_words=3, max_splices=2, max_share=0.6):
    """The pick with the moves applied, as (text, None), or (None, the reason the proposal fails). Every `sentence`
    and `after` names sentences of the pick as served, never as an earlier move left it; each sentence takes one cut
    or one swap at most. A cut's `sentence` and a splice's `after` may name one sentence or a run of sentences in a
    row, whitespace squashed (the splitter cuts "Not. Even. Close." into three): a cut on a run cuts every sentence of
    it, and a splice after a run goes in after its last sentence. A swap names exactly one sentence, since its word
    limit is per sentence. A name found at more than one place fails."""
    from .fixpass import gate as fix_gate, run_keeps
    sents, osents = sentences(pick), sentences(orig)
    if " ".join(sents) != squash(pick):
        return None, "the rewrite doesn't split cleanly into its sentences"

    def found(name):
        """(first, last) of the one place `name` stands in the pick as a sentence or a run of them, or (None, why)."""
        want = squash(name)
        hits = [(i, j) for i in range(len(sents)) for j in range(i, len(sents)) if " ".join(sents[i:j + 1]) == want]
        if len(hits) != 1:
            return None, ("names a sentence the rewrite has twice" if hits else "names no sentence of the rewrite")
        return hits[0], None
    body, after, start, touched, splices = list(sents), [[] for _ in sents], [], set(), 0
    for m in moves:
        kind = m.get("move")
        if kind not in MOVES:
            return None, f"not a move the step makes ({kind})"
        need = MOVES[kind]
        if set(m) - set(need) - {"move", "why"} or any(not isinstance(m.get(k), str) or not m[k].strip() for k in need):
            return None, f"a {kind} with fields it doesn't take or without the ones it needs"
        if kind in ("cut", "swap"):
            at, why = found(m["sentence"])
            if why:
                return None, f"a {kind} {why}"
            i, j = at
            if kind == "swap" and j != i:
                return None, f"a swap names {j - i + 1} sentences, and a swap changes one"
            if touched & set(range(i, j + 1)):
                return None, "two moves on one sentence"
            touched |= set(range(i, j + 1))
            if kind == "cut":
                if any(sents[k] in osents for k in range(i, j + 1)):
                    return None, "a cut takes out a sentence the original has"
                for k in range(i, j + 1):
                    body[k] = None
                continue
            _, _, new_sent, why = fix_gate({"sentence": sents[i], "replacement": m["to"], "category": "fact"},
                                           [squash(pick)], [run_keeps(orig, list(notes))], orig, max_words)
            if why:
                return None, f"a swap {why}"
            body[i] = squash(new_sent)
            continue
        s = squash(m["sentence"])
        if s not in osents:
            return None, "a splice puts in a sentence the original doesn't have word for word"
        if s in sents:
            return None, "a splice puts in a sentence the rewrite already has"
        splices += 1
        if splices > max_splices:
            return None, f"more than {max_splices} splices"
        if squash(m["after"]).upper() == "START":
            start.append(s)
            continue
        at, why = found(m["after"])
        if why:
            return None, f"a splice {why}"
        after[at[1]].append(s)
    out = start + [x for i in range(len(sents)) for x in ([body[i]] if body[i] else []) + after[i]]
    text = " ".join(out)
    if not text:
        return None, "the moves leave nothing"
    if text == squash(pick):
        return None, "the moves change nothing"
    share = sum(len(s.split()) for s in out if s in osents) / len(text.split())
    if len(orig.split()) > SHORT_SPAN and share > max_share:
        return None, f"the original's own sentences make up {share:.0%} of the result, so it's no repair"
    return text, None


def moves_line(moves):
    """The moves as one readable line: cut: "..."; swap: "..." to "..."; splice: "..."."""
    bits = []
    for m in moves:
        s = squash(m.get("sentence"))
        bits.append(f'swap: "{s}" to "{squash(m.get("to"))}"' if m.get("move") == "swap" else f'{m.get("move")}: "{s}"')
    return "; ".join(bits)


def carries_fact(t):
    """A note that holds a number, a quote, a name or a citation."""
    from engine.draws import names
    return bool(re.search(r"\d|[\"“”]|\bet al\.", t) or names(t))


def author_prompt(notes, orig):
    """{"about", "point", "fact"} for a span that goes to the author: from the notes (a keep: note lists the author's
    words, not a point, so it stays out), or from the original's first sentence when there are none."""
    pts = [n for n in notes if not re.match(r"^\s*keep\s*:", n, re.I)]
    if pts:
        return {"about": pts[0], "point": " ".join(pts), "fact": " ".join(n for n in pts if carries_fact(n))}
    first = (sentences(orig) or [squash(orig)])[0]
    return {"about": first, "point": first, "fact": first if carries_fact(first) else ""}


def notes_of(job, para):
    """The paragraph's cached notes (the job's notes cache, as RegrowPass.setup reads it), or [] when it has none."""
    cache = job._cache() if hasattr(job, "_cache") else {}
    return list(cache.get(sha(NOTES_VERSION + para)) or [])


# ---------------------------------------------------------------------- the files
def read_log(job_dir):
    path = Path(job_dir) / LOG
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def append_log(job_dir, row):
    with (Path(job_dir) / LOG).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review(job_dir, add):
    """Append `add` ({list name: [entries]}) to the reviewer's file: read as it is now, every entry there kept in its
    place, written back with indent 1, its line endings and its last newline as they were."""
    path = Path(job_dir) / rv.FILE
    raw = path.read_bytes().decode("utf-8") if path.exists() else ""
    data = json.loads(raw) if raw.strip() else {}
    for k in ("vetoes", "repairs", "for_you"):
        if not isinstance(data.get(k), list):
            data[k] = list(data.get(k) or [])
    for k, rows in add.items():
        data[k].extend(rows)
    text = json.dumps(data, indent=1, ensure_ascii=False) + ("\n" if raw.endswith("\n") or not raw else "")
    if "\r\n" in raw:
        text = text.replace("\n", "\r\n")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    tmp.replace(path)


# ---------------------------------------------------------------------- the pass, read only
def load_pass(job, settings, conn, sections, dry=False, pangram=None, max_usd=None, log=print):
    """The paragraph pass loaded the way regrow.main does for --final: units and bank, no model, no draw. In a dry run
    the sidecar (`regrow origs.json`) is read, or built in memory when it's missing, and never written."""
    rp = RegrowPass(job=job, bank=Bank(job.bank_path), settings=settings, conn=conn, pangram=pangram, purpose="loop",
                    max_usd=max_usd, log=log)
    if dry:
        def in_memory():
            path = job.dir / ORIGS_FILE
            rp.sidecar = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(rp.all_origs)
        rp.load_origs = in_memory
    rp.load_units(sections)
    rp.load_bank()
    return rp


def served(rp, sections):
    out = {}
    for n in sections:
        sp = rp.section_pick(n)
        out[n] = sp["text"] if sp else None
    return out


def spans(rp, job, sections):
    """Every span the pass serves that isn't pinned: label, section, the original span, the pick, the notes."""
    by = {u["label"]: u for u in rp.units}
    out = []
    for n in sections:
        sp = rp.section_pick(n)
        if sp is None:
            continue
        for u in rp.units:
            label = u["label"]
            if u["section"] != n or label not in sp["picks"] or label in rp.pins:
                continue
            orig = rp.orig_text.get(label, u["orig"])
            out.append({"label": label, "section": n, "orig": orig, "pick": sp["picks"][label],
                        "notes": notes_of(job, by[label]["para"]), "key": span_key(orig, sp["picks"][label]),
                        "scoring": sp["scoring"]["label"]})
    return out


# ---------------------------------------------------------------------- one read
def judge(answer, s, cfg):
    """One answer of the read as (kind, moves, text, why): "clean"; "unfixable" with its reason; "bad" (it didn't
    parse) or "gate" (the code gate said no) with the reason, the two a retry can mend; or "moves" with the repaired
    text."""
    kind, val = parse_answer(answer)
    if kind == "clean":
        return "clean", None, None, None
    if kind == "unfixable":
        return "unfixable", None, None, f"unfixable: {val}"
    if kind == "bad":
        return "bad", None, None, val
    text, why = apply_moves(val, s["pick"], s["orig"], s["notes"], int(cfg["max_words_changed"]),
                            int(cfg["max_splices"]), float(cfg["max_original_share"]))
    if why:
        return "gate", val, None, f"the gate said no: {why}"
    return "moves", val, text, None


def read_span(s, claude, settings, conn, cfg):
    """The read, the gate and the check for one span. Returns (log row, Claude dollars). A Claude error goes up, and
    nothing is logged for that span. When the first answer can't be used (it didn't parse, or the gate said no), Opus
    is asked once more, with its answer and the gate's reason quoted, under the same per-call ceiling; only when that
    answer fails too does the span fail. The row keeps both reads (`read`, `retry_read`), with `retried`, the first
    answer's reason (`first_why`) and its moves (`first_moves`). A check that still finds a slip isn't retried."""
    usd = 0.0
    prompt = fill(READ_PROMPT, s["orig"], s["notes"], s["pick"], cfg)
    answer, meta = claude(settings, conn, prompt, role="repair read", model=cfg["model"],
                          max_budget_usd=float(cfg["call_usd"]))
    usd += float((meta or {}).get("cost_usd") or 0.0)
    row = {"outcome": "failed", "text": None, "moves": None, "moves_line": None, "why": None,
           "read": (answer or "")[:4000], "check": None, "retried": False}
    kind, val, text, why = judge(answer, s, cfg)
    if kind in ("bad", "gate"):
        quoted = json.dumps(val, ensure_ascii=False) if val is not None else (answer or "").strip()[:2000]
        note = RETRY_NOTE.replace("<<ANSWER>>", quoted).replace("<<WHY>>", why)
        again, meta = claude(settings, conn, prompt + note, role="repair retry", model=cfg["model"],
                             max_budget_usd=float(cfg["call_usd"]))
        usd += float((meta or {}).get("cost_usd") or 0.0)
        row.update(retried=True, first_why=why, first_moves=val, retry_read=(again or "")[:4000])
        kind, val, text, why = judge(again, s, cfg)
        if why:
            why += " (on the second try)"
    row["moves"] = val
    if kind == "clean":
        row["outcome"] = "clean"
    elif kind != "moves":
        row["why"] = why
    else:
        check, meta = claude(settings, conn, fill(CHECK_PROMPT, s["orig"], s["notes"], text, cfg),
                             role="repair check", model=cfg["model"], max_budget_usd=float(cfg["call_usd"]))
        usd += float((meta or {}).get("cost_usd") or 0.0)
        row["check"] = (check or "")[:4000]
        if (check or "").strip().upper().startswith("NONE"):
            row.update(outcome="repaired", text=text, moves_line=moves_line(val))
        else:
            first = next((ln for ln in (check or "").splitlines() if ln.strip()), "no reason given")
            row["why"] = "the check still found: " + squash(first).lstrip("- ")
    row["usd"] = usd
    return row, usd


def entry_for(s, row, log_rows, cfg, today):
    """(outcome, list name, entry) for the reviewer's file, or (outcome, None, None) for a clean span. Every entry
    names the scoring that was served when it was read (`scoring`), which makes that scoring the section's review
    base, so the repair is read against what the step read."""
    if row["outcome"] == "clean":
        return "clean", None, None
    at = s.get("scoring")
    if row["outcome"] == "repaired":
        return "repaired", "repairs", {"label": s["label"], "orig": s["orig"], "from": s["pick"], "text": row["text"],
                                       "moves": row["moves_line"], "by": cfg["model"], "when": today, "scoring": at}
    o = orig_key(s["orig"])
    failed = {squash(r.get("from")) for r in log_rows if r.get("outcome") == "failed" and r.get("orig_sha") == o}
    if len(failed) >= int(cfg["stuck_after"]):
        ask = author_prompt(s["notes"], s["orig"])
        return "for you", "for_you", {"label": s["label"], "orig": s["orig"], "text": s["orig"],
                                      "went_wrong": row["why"], **ask, "when": today, "scoring": at}
    return "vetoed", "vetoes", {"label": s["label"], "orig": s["orig"], "text": s["pick"], "why": row["why"],
                                "scoring": at}


# ---------------------------------------------------------------------- the step
def run(job, settings, conn, mode=None, sections=None, claude=None, pangram=None, today=None, log=print):
    """The repair step. `mode` None reads [repair] mode; "dry" lists the spans it would read and calls nothing. Returns
    the result dict the report and the morning's digest are built from."""
    cfg = config(settings)
    mode = mode or cfg["mode"]
    if mode not in MODES + ("dry",):
        raise ValueError(f"[repair] mode is {mode!r}; it takes apply, propose or off")
    today = today or dt.date.today().isoformat()
    sections = sorted(sections if sections is not None else job.sections)
    res = {"mode": mode, "spans": [], "counts": {}, "seen": 0, "to_read": [], "unread": 0, "unread_why": None,
           "claude_usd": 0.0, "max_claude_usd": float(cfg["max_claude_usd"]), "pangram_usd": 0.0, "rescored": {},
           "stale": [], "written": False}
    if mode == "off":
        return res
    dry = mode == "dry"
    if not dry and claude is None:
        from . import db
        from .claude_call import run as claude
        conn = conn or db.connect(settings.db_path)
    rp = load_pass(job, settings, conn, sections, dry=dry, log=log)
    before, stale_before = served(rp, sections), set(rp.stale_repairs)
    log_rows = read_log(job.dir)
    unread, stop = set(), False
    for rnd in range(READ_ROUNDS):
        applied = {r["key"] for r in log_rows if r.get("mode") == "apply"}
        proposed = {r["key"]: r for r in log_rows if r.get("mode") == "propose"}
        todo = []
        for s in spans(rp, job, sections):
            if s["key"] in applied or (mode == "propose" and s["key"] in proposed):
                res["seen"] += rnd == 0
                continue
            todo.append(s)
        if dry:
            res["to_read"] = [{"label": s["label"], "section": s["section"], "before": s["pick"],
                               "free": s["key"] in proposed} for s in todo]
            return res
        add = {"vetoes": [], "repairs": [], "for_you": []}
        for s in todo:
            replay = proposed.get(s["key"]) if mode == "apply" else None
            if replay is not None:
                row = {k: replay.get(k) for k in ("outcome", "text", "moves", "moves_line", "why", "read", "check",
                                                  "retried", "first_why", "first_moves", "retry_read")}
                row["usd"] = 0.0
            elif stop or res["claude_usd"] >= res["max_claude_usd"]:
                unread.add(s["key"])          # a set: a later round meets the same span again
                res["unread_why"] = res["unread_why"] or "the Claude ceiling"
                continue
            else:
                try:
                    row, usd = read_span(s, claude, settings, conn, cfg)
                except Exception as e:  # noqa: BLE001
                    from .util import redact
                    stop = True
                    unread.add(s["key"])
                    res["unread_why"] = f"Claude said no: {type(e).__name__}: {redact(e)[:200]}"
                    continue
                res["claude_usd"] += usd
                unread.discard(s["key"])
            row.update({"key": s["key"], "version": REPAIR_VERSION, "when": today, "mode": mode, "label": s["label"],
                        "section": s["section"], "orig_sha": orig_key(s["orig"]), "from": s["pick"]})
            append_log(job.dir, row)
            log_rows.append(row)
            outcome, where, entry = entry_for(s, row, log_rows, cfg, today)
            if where and mode == "apply":
                add[where].append(entry)
            res["counts"][outcome] = res["counts"].get(outcome, 0) + 1
            res["spans"].append({"label": s["label"], "section": s["section"], "outcome": outcome, "before": s["pick"],
                                 "after": row["text"], "moves": row["moves_line"], "why": row["why"],
                                 "replayed": replay is not None})
            log(f"repair: {s['label']} {outcome}" + (f" ({row['why']})" if row["why"] else ""))
        res["unread"] = len(unread)
        if mode != "apply" or not any(add.values()):
            break
        write_review(job.dir, add)
        res["written"] = True
        rp = load_pass(job, settings, conn, sections, log=log)      # a veto can move a pinless section's best
    if not res["written"]:
        return res
    if pangram is None:
        from . import db
        from .pangram import Pangram
        pangram = Pangram(settings, conn or db.connect(settings.db_path))
    fin = load_pass(job, settings, conn, sections, pangram=pangram, max_usd=float(cfg["max_pangram_usd"]), log=log)
    res["stale"] = sorted(set(fin.stale_repairs) - stale_before)     # the reviewer's repairs too, not only this run's
    for n in sections:
        sp = fin.section_pick(n)
        if (sp["text"] if sp else None) == before.get(n):
            continue
        sc = fin.final(n)
        res["rescored"][n] = None if sc is None else {"ai": sc["ai"], "label": sc["label"],
                                                      "humanized": bool(sc.get("humanized"))}
    res["pangram_usd"] = fin.spent
    return res


def has_news(res):
    """Whether the result holds anything for the digest beyond clean reads."""
    if not res:
        return False
    return bool(res.get("to_read") or res.get("unread") or res.get("rescored")
                or any(s["outcome"] != "clean" for s in res.get("spans", ())))


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def report_lines(res):
    """The step's report: the counts and the spend, then each span that wasn't clean (a repair as a before line and an
    after line), the spans left unread and the sections scored again."""
    mode = res.get("mode")
    if mode == "off":
        return ["The repair step is off ([repair] mode = \"off\")."]
    if mode == "dry":
        todo = res.get("to_read") or []
        seen = f", {res['seen']} read before" if res.get("seen") else ""
        L = [f"A dry run: {plural(len(todo), 'span')} to read{seen}; nothing was called or written."]
        L += [f"- {t['label']} (section {t['section']})" + (", free: a proposal to replay" if t.get("free") else "")
              + f": {squash(t['before'])}" for t in todo]
        return L
    c = res.get("counts") or {}
    would = "would be " if mode == "propose" else ""
    L = [f"{plural(len(res.get('spans') or []), 'span')} read: {c.get('clean', 0)} clean, {c.get('repaired', 0)}"
         f" {would}repaired, {c.get('vetoed', 0)} {would}vetoed, {c.get('for you', 0)} {would}sent to you;"
         f" {res.get('seen', 0)} read before. Claude ${res.get('claude_usd', 0):.2f} of"
         f" ${res.get('max_claude_usd', 0):.2f}, Pangram ${res.get('pangram_usd', 0):.2f}."]
    if mode == "propose":
        L.append("Proposals only ([repair] mode = \"propose\"): nothing was written to review.json.")
    if res.get("unread"):
        L.append(f"{plural(res['unread'], 'span')} went unread ({res['unread_why']}); they ship as the pass's fact"
                 " check left them.")
    for s in res.get("spans") or []:
        if s["outcome"] == "repaired":
            L += [f"- {s['label']}, {would}repaired: {s['moves']}", f"  before: {squash(s['before'])}",
                  f"  after: {squash(s['after'])}"]
        elif s["outcome"] == "vetoed":
            L += [f"- {s['label']}, {would}vetoed, so the pass picks again: {s['why']}",
                  f"  before: {squash(s['before'])}"]
        elif s["outcome"] == "for you":
            L += [f"- {s['label']}, {would}sent to you to rewrite by hand (the original goes in, in bold): {s['why']}",
                  f"  before: {squash(s['before'])}"]
    for n, sc in sorted((res.get("rescored") or {}).items()):
        if sc is None:
            L.append(f"Section {n} scored again as served: not scored (the ceiling, or nothing to serve)")
        else:
            L.append(f"Section {n} scored again as served: {sc['ai']:.2f} ({sc['label']})"
                     + (", humanized, so it isn't a best" if sc["humanized"] else ""))
    if res.get("stale"):
        L.append("Repairs that went stale this run (their section's review base moved to a scoring where the unit's"
                 " pick is no longer the repair's `from`): " + ", ".join(res["stale"]))
    return L


# ---------------------------------------------------------------------- --reread, the smoke test of the real call
def reread(job, settings, conn, labels, claude=None, log=print):
    """--reread: the served spans named in `labels` read again though the log has them, in propose mode only, under
    the normal ceilings ([repair] call_usd for each call, max_claude_usd for the run). Nothing is written: not
    review.json, not the repair log, not the pass's sidecar (the pass is loaded as a dry run loads it). A label a
    standing repair pins is read from the pick that repair mended (its `from`, against its `orig`), and the report
    shows the repair beside what Opus says; a label a hand fix pins, or one the pass doesn't serve, is listed with the
    reason. Returns
    {"spans", "missing", "unread", "claude_usd", "max_claude_usd"}; each span holds the log row the read would have
    made and each Claude call's role and cost."""
    cfg = config(settings)
    if claude is None:
        from . import db
        from .claude_call import run as claude
        conn = conn or db.connect(settings.db_path)
    sections = sorted({int(label.split("w")[0][1:]) for label in labels})
    rp = load_pass(job, settings, conn, sections, dry=True, log=log)
    by = {s["label"]: s for s in spans(rp, job, sections)}
    units = {u["label"]: u for u in rp.units}
    res = {"spans": [], "missing": [], "unread": [], "claude_usd": 0.0, "max_claude_usd": float(cfg["max_claude_usd"])}
    stop = None
    for label in labels:
        s = by.get(label)
        rep = (rp.repairs or {}).get(label)
        if s is None and rep and label in units:
            # a standing repair pins the span, so the step never reads it again; read the pick it mended (its `from`)
            # against its `orig`, the read the log holds, and show the repair that stands beside what Opus says
            u = units[label]
            s = {"label": label, "section": u["section"], "orig": rep["orig"], "pick": rep["from"],
                 "notes": notes_of(job, u["para"]), "key": span_key(rep["orig"], rep["from"]),
                 "scoring": rep.get("scoring"), "standing": rep}
        if s is None:
            why = ("pinned by a hand fix for you, which has no pick to read" if label in rp.pins
                   else "not in what the pass serves" if label in units else "no such span in those sections")
            res["missing"].append((label, why))
            continue
        if stop or res["claude_usd"] >= res["max_claude_usd"]:
            res["unread"].append((label, stop or "the Claude ceiling"))
            continue
        calls = []

        def counted(*a, **k):
            text, meta = claude(*a, **k)
            calls.append((k.get("role"), float((meta or {}).get("cost_usd") or 0.0)))
            return text, meta
        try:
            row, usd = read_span(s, counted, settings, conn, cfg)
        except Exception as e:  # noqa: BLE001
            from .util import redact
            stop = f"Claude said no: {type(e).__name__}: {redact(e)[:200]}"
            res["unread"].append((label, stop))
            continue
        res["claude_usd"] += usd
        res["spans"].append({"label": label, "section": s["section"], "pick": s["pick"], "row": row, "calls": calls,
                             "standing": s.get("standing")})
        log(f"repair --reread: {label} {row['outcome']} (${usd:.2f})")
    return res


def reread_lines(res):
    """--reread's report: per span what Opus answered, the moves, what the code gate and the check said, what the
    step would do, and what each call cost."""
    L = [f"--reread, propose mode, nothing written: {plural(len(res['spans']), 'span')} read, Claude"
         f" ${res['claude_usd']:.2f} of ${res['max_claude_usd']:.2f}."]
    for s in res["spans"]:
        row = s["row"]
        cost = ", ".join(f"{role} ${usd:.2f}" for role, usd in s["calls"])
        L += ["", f"{s['label']} (section {s['section']}): {cost}", f"  the pick: {squash(s['pick'])}"]
        if s.get("standing"):
            st = s["standing"]
            L += [f"  the repair that stands (by {st.get('by')}, {st.get('when')}): {squash(st.get('moves'))}",
                  f"  its text: {squash(st.get('text'))}"]
        L.append(f"  Opus read: {squash(row.get('read'))[:1200]}")
        if row.get("retried"):
            if row.get("first_moves"):
                L.append(f"  its moves: {moves_line(row['first_moves'])}")
            L += [f"  not usable, so it was asked once more: {row.get('first_why')}",
                  f"  Opus read again: {squash(row.get('retry_read'))[:1200]}"]
        if row["outcome"] == "clean":
            L.append("  clean: the step would log it and write nothing")
            continue
        if row.get("moves"):
            L.append(f"  moves: {moves_line(row['moves'])}")
            gate = row.get("why") or ""
            L.append(f"  the code gate: {gate[len('the gate said no: '):]}, so no" if gate.startswith("the gate said no")
                     else "  the code gate: let them through")
        if row.get("check") is not None:
            L.append(f"  the check: {squash(row['check'])[:600]}")
        if row["outcome"] == "repaired":
            L += ["  the step would repair it:", f"  after: {squash(row['text'])}"]
        else:
            L.append(f"  the step would veto the pick (or send it to you once it's stuck): {row.get('why')}")
    for label, why in res["missing"]:
        L.append(f"- {label}: not read, {why}")
    for label, why in res["unread"]:
        L.append(f"- {label}: not read, {why}")
    return L


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="the repair step: small edits for the factual slips in the paragraph"
                                             " pass's picks, written to review.json")
    ap.add_argument("--job", help="the job, a folder in jobs/ (default: [job] default in config.toml)")
    ap.add_argument("--section", help="sections, comma-separated (default: the job's scope)")
    ap.add_argument("--dry", action="store_true", help="list the spans it would read: no Claude, no Pangram, no file")
    ap.add_argument("--mode", choices=["apply", "propose"], help="override [repair] mode for this run")
    ap.add_argument("--reread", metavar="LABELS",
                    help='read these served spans again though the log has them ("s5w0 p3,s6w2 p5"), in propose mode'
                         " only: prints what Opus said and each call's cost, and writes nothing")
    a = ap.parse_args(argv)
    if a.reread is not None:
        if a.mode == "apply":
            ap.error("--reread runs in propose mode only: it never writes review.json or the repair log")
        if a.dry or a.section:
            ap.error("--reread goes on its own (its labels name the sections)")
        labels = [x.strip() for x in a.reread.split(",") if x.strip()]
        if not labels or any(not re.fullmatch(r"s\d+w\d+ p\d+", x) for x in labels):
            ap.error('--reread takes labels like "s5w0 p3,s6w2 p5"')
    from .settings import Settings
    settings = Settings()
    job = Job(a.job)
    if a.reread is not None:
        print("\n".join(reread_lines(reread(job, settings, None, labels))))
        return 0
    sections = [int(x) for x in a.section.split(",")] if a.section else None
    res = run(job, settings, None, mode="dry" if a.dry else a.mode, sections=sections)
    print("\n".join(report_lines(res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
