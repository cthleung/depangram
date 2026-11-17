"""The write-back with the master kept (the author's review scheme, 2026-11-14).

    python -m engine.writeback --job NAME                  # dry run: what would change in the document, nothing written
    python -m engine.writeback --job NAME --write          # do it
    python -m engine.writeback --job NAME --section 2,3    # some sections only (default: the job's scope)

--job may be left out when config.toml names a default job ([job] default).

Every paragraph in scope that the job knows (the frozen original's text, a version the loop banked, or a text an
earlier write-back put there) becomes the newest machine text: the loop's working copy, with the paragraph pass's
best picks in where the pass has scored the section (engine/regrow.py). The paragraph goes under a comment that
holds the frozen original's paragraph, so the master stays in the file for fact checking:

    <!-- depangram s2w1 p3 · written 2026-01-05 21:10 · the paragraph pass (s2 round 3)
    the original (frozen at the sync of 2026-01-02):
    The first thing to check is the kettle, ...
    -->

    Start with the kettle, because it's the thing that breaks first. ...

One comment per paragraph, since a section already holds research and INSERT comments and a comment can't sit
inside another. Under each section heading a note says when the section was last written and, where it holds, which
Pangram scoring this exact text has.

A section that has left the job's scope (a sync takes a section out while it holds notes outside a comment) can
still hold the loop's text from an earlier write-back. A full run marks those paragraphs too, with their frozen
originals, and changes none of their words.

A rerun rewrites the paragraph under a marker when there's newer text, and moves that marker's time; a paragraph
that didn't change keeps its marker byte for byte. A paragraph the job doesn't know is yours (an edit by hand since)
and stays as it is, marker and all. Every text written goes in the job's ledger (`written back.jsonl`), which is
how the sync step (engine/sync.py) and the next write-back know the pass's paragraphs as the machine's.

A paragraph the reviewer hand-corrected for a factual slip (the pass's row carries `for_you`, the author's rule of
2026-11-14: facts before the score) goes in whole in bold, and its marker says FOR YOU TO REWRITE and holds the
prompt for the author as the human writer: what went wrong, what to write about, the point to make and the fact.

A paragraph with a stuck unit (the pass's row carries `stuck`, from a stuck entry in the reviewer's file, 2026-11-15:
a span the machine couldn't rewrite, so the author writes it) keeps its words and gets only those sentences in bold,
the ones still Claude's original word for word, with the original's own bold words back outside them. Its marker says
FOR YOU TO REWRITE and holds one prompt group per stuck unit: why it's the author's, what to write about, the point
and the fact (a paragraph with two or more groups opens each with a line naming its bold text by its first words).
It gets the marker and the bold even when its text is still the original. A hand fix on the same paragraph wins.

A paragraph whose pick was mended with small edits (the pass's row carries `repaired`, from the reviewer's file or
the repair step, engine/repair.py) ships as normal text, and its marker says so on the line under its first one:
`repaired: cut: "..."; splice: "..."`. A changed repaired line rewrites the marker; the same one changes no byte.

Nothing here calls Pangram or Claude: the score in a section's note is looked up in the local cache and is only
printed when the section as posted is exactly the text that scoring paid for.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

from .job import Bank, Job, flat, style_of
from .texts import COMMENT_RE, strip_markdown
from .wrapup import (BLOCK_SPLIT, LEDGER, as_posted, is_prose_block, layout, norm, paragraph_history, rebold,
                     recognized, run_versions, section_spans)

MARK_RE = re.compile(r"\A<!-- depangram (s\d+w\d+ p\d+) · ")
HEAD_RE = re.compile(r"\A<!-- depangram section (\d+) · ")
STRIP_RE = re.compile(r"<!-- depangram\b.*?-->\s*", re.S)
FOR_YOU = "FOR YOU TO REWRITE"
HAND = "hand-corrected by the reviewer"
REPAIRED = "repaired: "
# the prompt for the author in a for-you marker: (the line's name, the key in the row's for_you dict)
PROMPT = (("what went wrong", "went_wrong"), ("write about", "about"), ("the point", "point"), ("the fact", "fact"))
# the prompt for a stuck unit's sentences in bold: (the line's name, the key in the row's stuck group)
STUCK_PROMPT = (("why it's yours", "why"), ("write about", "about"), ("the point", "point"), ("the fact", "fact"))
STUCK_GROUP = 'the bold starting "'         # opens each group when a paragraph has more than one


def strip_markers(md):
    """The document without the write-back's comments (the commented originals and the section notes)."""
    return STRIP_RE.sub("", md)


def safe(text):
    """A paragraph that can sit inside a comment: its own comments out, and nothing left that closes one."""
    return COMMENT_RE.sub("", text).replace("-->", "--&gt;").replace("<!--", "&lt;!--").strip()


def is_marker(block):
    return bool(MARK_RE.match(block.lstrip())) and not COMMENT_RE.sub("", block).strip()


def prompt_lines(for_you):
    """The for-you prompt as marker lines, each value made safe and kept to one line (a blank line would split the
    marker into two blocks); a line with no value is left out."""
    out = []
    for name, key in PROMPT:
        value = " ".join(safe(str((for_you or {}).get(key) or "")).split())
        if value:
            out.append(f"{name}: {value}")
    return out


def first_words(text, n=6):
    """The first `n` words of a text, made safe for a marker line, with "..." when there were more."""
    words = " ".join(safe(text).split()).split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def stuck_prompt_lines(groups):
    """A stuck paragraph's prompt as marker lines: per group, the lines of STUCK_PROMPT that have a value, each made
    safe and kept to one line; with more than one group, each opens with a line naming its bold text by its first
    words."""
    out = []
    for g in groups or ():
        if len(groups) > 1:
            out.append(f'{STUCK_GROUP}{first_words(" ".join(g.get("bold") or ()))}":')
        for name, key in STUCK_PROMPT:
            value = " ".join(safe(str(g.get(key) or "")).split())
            if value:
                out.append(f"{name}: {value}")
    return out


def stuck_body(plain, sentences, old_block):
    """`plain` (a paragraph's text, no bold marks) with each of its sentences that is one of `sentences` in bold,
    and the bold words of `old_block` (the frozen original) back where rebold() would put them, unless that place is
    inside or across a bold sentence, so no bold mark is ever nested or broken. Sentences next to each other go in
    one bold run. Matched by sentence (the pass's own splitter), never by substring, so a short sentence never bolds
    part of a longer one. Returns (the body, the squashed sentences found), or (None, set()) when none is there."""
    from .checks import split_sentences
    want = {norm(x) for x in sentences}
    ranges, found, pos = [], set(), 0
    for sent in split_sentences(plain):
        i = plain.find(sent, pos)
        if i < 0:
            continue
        pos = i + len(sent)
        if norm(sent) not in want:
            continue
        found.add(norm(sent))
        if ranges and not plain[ranges[-1][1]:i].strip():
            ranges[-1] = (ranges[-1][0], pos)
        else:
            ranges.append((i, pos))
    if not ranges:
        return None, set()
    marks = []
    for bold in re.findall(r"\*\*(.+?)\*\*", old_block or ""):
        i = plain.find(bold)
        j = i + len(bold)
        if i < 0 or any(a < j and i < b for a, b in ranges + marks):
            continue
        marks.append((i, j))
    out, last = [], 0
    for a, b in sorted(ranges + marks):
        out += [plain[last:a], "**", plain[a:b], "**"]
        last = b
    out.append(plain[last:])
    return "".join(out), found


def is_for_you(block):
    """A marker that puts its paragraph in bold for the author to rewrite."""
    return is_marker(block) and FOR_YOU in block.lstrip().split("\n", 1)[0]


def is_stuck_marker(block):
    """A for-you marker over a stuck paragraph (some of its sentences in bold), not over a hand fix: a hand fix's
    marker always names the reviewer as its source on its first line."""
    return is_for_you(block) and HAND not in block.lstrip().split("\n", 1)[0]


def marker_prompt(block):
    """The prompt lines a for-you marker holds, a hand fix's or a stuck paragraph's (empty for any other marker)."""
    names = tuple(f"{name}: " for name, _ in PROMPT + STUCK_PROMPT) + (STUCK_GROUP,)
    return [ln for ln in block.strip().split("\n")[1:] if ln.startswith(names)] if is_for_you(block) else []


def marker_repaired(block):
    """The `repaired:` line a marker holds right under its first line, or None."""
    lines = block.strip().split("\n")
    return lines[1] if len(lines) > 1 and lines[1].startswith(REPAIRED) else None


def marker(p, stamp, sync_date=None):
    bits = [f"depangram {p['label']}", f"written {stamp}", p["source"]]
    if p.get("flagged"):
        bits.append("Pangram flagged a span the pass wrote here")
    lines = []
    if p.get("repaired"):                  # the reviewer's or the repair step's small edits on this paragraph's pick
        lines.append(REPAIRED + " ".join(safe(str(p["repaired"])).split()))
    if p.get("for_you"):
        bits.append(FOR_YOU)
        lines += prompt_lines(p["for_you"])
    elif p.get("stuck"):                   # a stuck unit's sentences in bold, one prompt group per unit
        bits.append(FOR_YOU)
        lines += stuck_prompt_lines(p["stuck"])
    origin = "the original" + (f" (frozen at the sync of {sync_date})" if sync_date else "") + ":"
    return "\n".join(["<!-- " + " · ".join(bits), *lines, origin, safe(p["orig_block"]), "-->"])


def last_sync(job_dir):
    """The date of the job's last sync, from the backups it left ("... (before sync 2026-11-12).md")."""
    dates = [m.group(1) for f in Path(job_dir).iterdir() for m in [re.search(r"\(before sync (\d{4}-\d\d-\d\d)\)", f.name)] if m]
    return max(dates) if dates else None


def frozen_blocks(job):
    """{normalized plain text: the block as the frozen original has it, bold marks in, comments out}."""
    out = {}
    for blk in BLOCK_SPLIT.split(job.original.read_text(encoding="utf-8").replace("\r\n", "\n")):
        k = norm(strip_markdown(blk))
        if k:
            out.setdefault(k, COMMENT_RE.sub("", blk).strip())
    return out


def paragraphs(job, bank, lay=None, passed=None, sections=None):
    """One row per window paragraph the loop may write: its label and section, the versions and whole texts it's
    known by, the frozen original (plain and as a block), the newest text and where that came from. `passed` is
    engine.regrow.pass_paragraphs()["paragraphs"]; a paragraph in it takes the pass's text, the rest the working
    copy's. A pass row with `for_you` (the reviewer's hand-corrected text and the prompt for the author) carries it
    through, and so does one with `repaired` (a line saying which small edits mended its pick) and one with `stuck`
    (its stuck units' sentences for bold, each group with its prompt). A pass row with `picked` False (a paragraph
    with a stuck unit and no pick) only lends its `stuck`: the text and the source stay what they'd be without the
    pass. A row passed without `picked` counts as picked."""
    lay = lay or layout(job, bank)
    base_md, swaps = job.placeholder_base()
    base = BLOCK_SPLIT.split(base_md)
    work = BLOCK_SPLIT.split(job.working.read_text(encoding="utf-8")) if job.working.exists() else base
    if len(work) != len(base):
        raise RuntimeError("the working copy's blocks don't line up with the original's; let the loop rewrite it")
    index = {}
    for i, blk in enumerate(base):
        k = norm(strip_markdown(COMMENT_RE.sub("", blk)))
        if k:
            index.setdefault(k, i)
    frozen = frozen_blocks(job)
    history = paragraph_history(bank)
    out = []
    for wid, w in lay.items():
        if sections is not None and w["section"] not in sections:
            continue
        versions = run_versions(bank, wid, w["original"])
        for pi, p in enumerate(w["original"]):
            if not any(r["claude"] or r.get("author") for r in w["runs"][pi]):
                continue                       # a subsection heading or a template: locked, never written
            label, orig = f"{wid} p{pi + 1}", norm(flat(p))
            i = index.get(orig)
            new, source, flagged = (norm(strip_markdown(work[i])) if i is not None else orig), "the loop", None
            hit, for_you, repaired = (passed or {}).get(label), None, None
            if hit and hit.get("picked", True):
                new, source, flagged = norm(hit["text"]), f"the paragraph pass ({hit['round']})", bool(hit["flagged"])
                for_you, repaired = hit.get("for_you") or None, hit.get("repaired") or None
                if for_you:
                    source = HAND
            stuck = []
            for g in (hit or {}).get("stuck") or ():
                bold = []
                for x in g.get("bold") or ():
                    for ph, line in swaps:
                        x = x.replace(ph, line)
                    bold.append(x)
                stuck.append(dict(g, bold=bold))
            for ph, line in swaps:
                new = new.replace(ph, line)
            out.append({"label": label, "section": w["section"], "versions": versions[pi],
                        "known": history.get(orig, set()), "orig": orig, "orig_block": frozen.get(orig, orig),
                        "new": new, "source": source, "flagged": flagged, "for_you": for_you, "repaired": repaired,
                        "stuck": stuck or None})
    return out


def set_aside(job, bank, scope):
    """Rows for paragraphs of sections that have left the job's scope (a sync takes a section out when list lines
    put it back in drafting). The loop works them no more, but the document can still hold its text from an earlier
    write-back, and that text gets its commented original like any other. `new` is None: whatever known text the
    document holds stays, and a paragraph at its original or edited by hand is left alone."""
    frozen = frozen_blocks(job)
    history = paragraph_history(bank)
    out, seen = [], set()
    for e in bank.entries:
        m = re.match(r"s(\d+)w\d+", e.get("window") or "")
        if e.get("kind") != "start" or not m or not e.get("text") or int(m.group(1)) in scope:
            continue
        paras = [x for x in re.split(r"\n\s*\n", e["text"]) if x.strip()]
        if paras and re.match(r"^\d+\.\s", paras[0]) and len(paras[0].split()) < 12:
            paras = paras[1:]                  # the section heading isn't a paragraph
        for pi, para in enumerate(paras):
            orig = norm(para)
            if orig in seen or len(history.get(orig, ())) < 2:
                continue                       # the loop never wrote another version of it
            seen.add(orig)
            out.append({"label": f"{e['window'].split('~')[0]} p{pi + 1}", "section": int(m.group(1)), "versions": [],
                        "known": history[orig], "orig": orig, "orig_block": frozen.get(orig, orig), "new": None,
                        "source": "the loop, at an earlier write-back (the section is out of the job's scope for now)",
                        "flagged": None})
    return out


def write_marked(real_text, sections, paras, stamp, frozen_keys=(), sync_date=None, notes=None, style=None):
    """The real document with the newest text in and each changed paragraph under its commented original. Returns
    (new text, report). `notes` is {section: a sentence for that section's note}; `frozen_keys` the frozen original's
    block keys, so a block that's simply the original (a template, a lead-in the loop never writes) isn't called
    yours. `style` is the job's SectionStyle (the numbered default when None). Line endings come back as they were.

    A row with `for_you` goes in whole in bold under a FOR YOU marker that holds the prompt; it stays byte for byte
    while its text and its prompt do, and a row that stops being for-you goes back to plain text with the original's
    bold words. A row with `stuck` (and no `for_you`, which wins) gets only its stuck sentences in bold, those its text
    holds, under a FOR YOU marker with one prompt group per unit, the same rules holding; a row whose text holds none
    of them is written as if it had no `stuck`. A row with no newer text (`new` None) bolds the text the document
    holds."""
    crlf = "\r\n" in real_text
    text = real_text.replace("\r\n", "\n")
    report = {"replaced": [], "marked": [], "same": [], "restored": [], "yours": [], "skipped": [], "missing": [],
              "ledger": [], "live": {}, "for_you": [], "prompts": {}, "repaired": [], "stuck": []}
    used, out, last = set(), [], 0
    spans = section_spans(text, sections, style)
    for n in sorted(spans, key=lambda k: spans[k][0]):
        a, b = spans[n]
        out.append(text[last:a])
        blocks = BLOCK_SPLIT.split(text[a:b])
        touched = live = bold = 0
        for i in range(0, len(blocks), 2):
            block = blocks[i]
            body = COMMENT_RE.sub("", block).strip()
            if not body or body.startswith("#"):
                continue                       # a comment-only block (a marker too) or a heading
            key = norm(strip_markdown(block))
            has_mark = i >= 2 and is_marker(blocks[i - 2])
            hit = next((k for k, p in enumerate(paras) if k not in used and p["section"] == n
                        and recognized(key, p["versions"], p["known"])), None)
            if hit is None:
                if has_mark or (is_prose_block(block) and key not in frozen_keys):
                    report["yours"].append((n, key[:90], has_mark))
                    live += has_mark
                continue
            used.add(hit)
            p = paras[hit]
            fy = p.get("for_you") if p["new"] is not None else None
            if fy:                             # in bold for the author: the bold marks inside it come out first
                new = norm(p["new"].replace("**", ""))
            else:
                new = norm(p["new"]) if p["new"] is not None else key    # a set-aside paragraph keeps the text it has
            stuck_txt, groups = None, None
            if not fy and p.get("stuck"):      # only its stuck sentences in bold, the ones this text still holds
                plain = norm(new.replace("**", ""))
                stuck_txt, found = stuck_body(plain, [x for g in p["stuck"] for x in g.get("bold") or ()],
                                              p["orig_block"])
                if stuck_txt is not None:
                    new = plain
                    groups = [g for g in p["stuck"] if any(norm(x) in found for x in g.get("bold") or ())]
            if "<!--" in block:
                report["skipped"].append(p["label"])           # a comment inside the paragraph's block: not ours to move
                live += has_mark
                continue
            lead, tail = block[:len(block) - len(block.lstrip())], block[len(block.rstrip()):]
            was_fy = has_mark and is_for_you(blocks[i - 2])
            if new == p["orig"] and not fy and stuck_txt is None:    # the newest text is the original: no marker
                if has_mark:
                    blocks[i - 2] = blocks[i - 1] = ""
                    touched += 1
                if key != new or was_fy:       # was in bold for you: the bold comes out with the marker
                    blocks[i] = lead + p["orig_block"] + tail
                    report["restored"].append(p["label"])
                    touched += 1
                else:
                    report["same"].append(p["label"])
                continue
            live += 1
            if stuck_txt is not None:
                shown = dict(p, stuck=groups)
                if p["source"] == "the loop" and new == p["orig"]:
                    shown["source"] = "the original"           # nothing has rewritten it yet
                mark = marker(shown, stamp, sync_date)
            else:
                mark = marker(dict(p, stuck=None), stamp, sync_date)
            if p.get("repaired") and p["new"] is not None:
                report["repaired"].append(p["label"])
            # the marker's repaired line still says what the row says (a row with no newer text keeps what it has)
            same_repair = p["new"] is None or (has_mark and marker_repaired(blocks[i - 2]) == marker_repaired(mark))
            if fy:
                bold += 1
                report["for_you"].append(p["label"])
                report["prompts"][p["label"]] = fy
                body_new = f"**{new}**"
                if (was_fy and block.strip() == body_new and marker_prompt(blocks[i - 2]) == marker_prompt(mark)
                        and same_repair):
                    report["same"].append(p["label"])
                    continue
            elif stuck_txt is not None:
                bold += 1
                report["for_you"].append(p["label"])
                report["stuck"].append(p["label"])
                # `about` joined, as a hand fix's reads, for the morning digest; the groups for the report's lines
                abouts = [" ".join(str(g.get("about") or "").split()) for g in groups]
                report["prompts"][p["label"]] = {"about": "; ".join(a for a in abouts if a), "stuck": groups}
                body_new = stuck_txt
                if (was_fy and block.strip() == body_new and marker_prompt(blocks[i - 2]) == marker_prompt(mark)
                        and same_repair):
                    report["same"].append(p["label"])
                    continue
            else:
                # a row with no newer text keeps its block and marker, a hand fix's too; a stuck paragraph's bold
                # comes out once the row stops carrying `stuck`, newer text or not
                was_stuck = was_fy and is_stuck_marker(blocks[i - 2])
                if key == new and has_mark and (not was_fy or (p["new"] is None and not was_stuck)) and same_repair:
                    report["same"].append(p["label"])
                    continue
                # out of bold (it was for-you): only the original's bold words come back
                body_new = (block.strip() if key == new and not was_fy
                            else rebold(new, p["orig_block"] + ("" if was_fy else "\n" + block)))
            if has_mark:
                blocks[i - 2] = mark
                blocks[i] = lead + body_new + tail
            else:
                blocks[i] = lead + mark + "\n\n" + body_new + tail
            report["marked" if key == new else "replaced"].append(p["label"])
            touched += 1
            if new not in p["known"]:
                row = {"when": stamp, "label": p["label"], "orig": p["orig"], "text": new, "source": p["source"]}
                if fy:
                    row["for_you"] = fy
                report["ledger"].append(row)
        head_i = next((i for i in range(0, len(blocks), 2) if HEAD_RE.match(blocks[i].lstrip())), None)
        if live and (touched or head_i is None):
            line = (f"<!-- depangram section {n} · written back {stamp} · {live} paragraph(s) here sit under their"
                    f" commented originals" + (f" · {bold} in bold for you to rewrite by hand" if bold else "")
                    + (f" · {notes[n]}" if notes and notes.get(n) else "") + " -->")
            if head_i is not None:
                blocks[head_i] = line
            else:
                blocks[0] = blocks[0].rstrip() + "\n\n" + line + blocks[0][len(blocks[0].rstrip()):]
        elif not live and head_i is not None:
            blocks[head_i] = ""
            if head_i + 1 < len(blocks):
                blocks[head_i + 1] = ""
        report["live"][n] = live
        out.append("".join(blocks))
        last = b
    out.append(text[last:])
    report["missing"] = [p["label"] for k, p in enumerate(paras)
                         if k not in used and p["section"] in spans and p["new"] is not None]
    new_text = "".join(out)
    return (new_text.replace("\n", "\r\n") if crlf else new_text), report


def section_notes(new_text, passed, pangram=None, style=None):
    """{section: the sentence for its note}. The score is claimed only when the section as posted is the text the
    pass's best scoring paid for and the local Pangram cache still holds that answer."""
    from .pangram import summarize
    notes = {}
    posted = as_posted(new_text, list(passed["sections"]), style=style)
    for n, s in passed["sections"].items():
        hit = pangram.cached(posted.get(n, "")) if pangram is not None and posted.get(n) == s["text"] else None
        if hit is not None and abs((summarize(hit)["fraction_ai"] or 0) - s["ai"]) < 1e-6:
            notes[n] = f"as posted, this is the text Pangram scored {s['ai']:.2f} ({s['label']})"
        else:
            notes[n] = (f"the pass's best assembly scored {s['ai']:.2f} ({s['label']}); this section isn't exactly"
                        f" that text, so it has no score of its own yet")
    return notes


def report_lines(r):
    L = [f"{len(r['replaced'])} paragraph(s) get newer text, {len(r['marked'])} already had it and get their"
         f" commented original, {len(r['same'])} stay as they are."]
    if r["replaced"]:
        L.append("Newer text: " + ", ".join(r["replaced"]))
    if r["marked"]:
        L.append("Marked only: " + ", ".join(r["marked"]))
    if r.get("for_you"):
        L.append("In bold for you to rewrite by hand: " + ", ".join(r["for_you"]))
        for label in r["for_you"]:
            pr = (r.get("prompts") or {}).get(label, {})
            if pr.get("stuck"):                # a stuck paragraph: its sentences in bold, one prompt per unit
                for g in pr["stuck"]:
                    parts = [f"{name}: {' '.join(str(g.get(key) or '').split())}" for name, key in STUCK_PROMPT
                             if str(g.get(key) or "").strip()]
                    L.append(f'- {label}, the bold starting "{first_words(" ".join(g.get("bold") or ()))}"'
                             + (": " + "; ".join(parts) if parts else ""))
                continue
            about = " ".join(str(pr.get("about") or "").split())
            if about:
                L.append(f"- {label}, write about: {about}")
    if r.get("repaired"):
        L.append("Mended with small edits, the marker says which: " + ", ".join(r["repaired"]))
    if r["restored"]:
        L.append("Back at the original: " + ", ".join(r["restored"]))
    for label in r.get("stuck_unmatched") or ():
        L.append(f"The stuck entry for {label} matches no unit now: write a new one or remove it")
    for kind, label in r.get("review_unmatched") or ():
        L.append(f"The {kind} for {label} in review.json matches no unit now (its paragraph was rewritten or"
                 f" removed): remove it")
    if r["skipped"]:
        L.append("Left alone, a comment sits inside the paragraph: " + ", ".join(r["skipped"]))
    if r["missing"]:
        L.append("Not found in the document (edited or moved since, so left alone): " + ", ".join(r["missing"]))
    for n, t, under in r["yours"]:
        L.append(f"Section {n}, yours, left as it is" + (" (edited by hand under a marker)" if under else "") + f": {t}")
    return L


def run(job, write=True, stamp=None, real=None, sections=None, settings=None, passed=None, pangram=None,
        after_sync=False):
    """The write-back, for the command line and the morning submission (engine/morning.py). `sections` None is a
    full run: the job's scope, and the sections out of scope that still hold the loop's text. `passed` stands in for
    engine.regrow.pass_paragraphs() and `pangram` for the cache client (tests pass both). `after_sync`: a sync just
    started the working copy again from the new original, so a row the working copy has at its original keeps the
    known text the document holds instead of going back to the original (the loop writes its text into the working copy
    at its next stop, not before). A working copy that is exactly the original, as a sync leaves it, counts the same.
    Returns (report, notes, changed); `changed` says the document's text differs, written or, with write=False, what a
    write would change."""
    real = Path(real) if real else job.real_file
    if real is None:
        from .job import JobError
        raise JobError(f"the job {job.name} names no document: set document in its job.toml, or pass the file")
    full = sections is None
    sections = list(job.sections) if full else list(sections)
    bank = Bank(job.bank_path)
    lay = layout(job, bank)
    if passed is None:
        from .regrow import pass_paragraphs
        passed = pass_paragraphs(job, bank, sections, readonly=not write)   # a dry run writes nothing, sidecar included
    paras = paragraphs(job, bank, lay, passed["paragraphs"], sections)
    if after_sync or not job.working.exists() or job.working.read_text(encoding="utf-8") == job.placeholder_base()[0]:
        for p in paras:
            if p["source"] == "the loop" and p["new"] == p["orig"]:
                p["new"] = None
    if full:                                   # a full run also marks what an earlier write-back left in sections out of scope
        aside = set_aside(job, bank, set(job.sections))
        paras += aside
        sections = sorted(set(sections) | {p["section"] for p in aside})
    raw = real.read_bytes().decode("utf-8")
    stamp = stamp or time.strftime("%Y-%m-%d %H:%M")
    kw = {"frozen_keys": set(frozen_blocks(job)), "sync_date": last_sync(job.dir), "style": style_of(job)}
    first, _ = write_marked(raw, sections, paras, stamp, **kw)
    if pangram is None and passed["sections"]:
        from . import db
        from .pangram import Pangram
        from .settings import Settings
        settings = settings or Settings()
        pangram = Pangram(settings, db.connect(settings.db_path))
    notes = section_notes(first, passed, pangram, style=style_of(job))
    new, report = write_marked(raw, sections, paras, stamp, notes=notes, **kw)
    changed = new != raw
    if write and changed:
        real.write_bytes(new.encode("utf-8"))
        if report["ledger"]:
            with (job.dir / LEDGER).open("a", encoding="utf-8") as f:
                for row in report["ledger"]:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["written"] = bool(write and changed)
    report["stuck_unmatched"] = list(passed.get("unmatched") or [])   # entries in review.json that speak for no unit
    report["review_unmatched"] = [tuple(x) for x in passed.get("unmatched_review") or []]
    return report, notes, changed


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="the newest machine text into the document, each paragraph under its"
                                             " commented original")
    ap.add_argument("--job", help="the job, a folder in jobs/ (default: [job] default in config.toml)")
    ap.add_argument("--real", help="the document (default: job.toml's document)")
    ap.add_argument("--section", help="sections, comma-separated (default: the job's scope)")
    ap.add_argument("--write", action="store_true", help="write it; without this it's a dry run")
    a = ap.parse_args(argv)
    job = Job(a.job)
    real = Path(a.real) if a.real else job.real_file
    sections = [int(x) for x in a.section.split(",")] if a.section else None
    report, notes, changed = run(job, write=a.write, real=real, sections=sections)
    print("\n".join(report_lines(report)))
    for n, note in sorted(notes.items()):
        print(f"Section {n}: {note}")
    if report["written"]:
        print(f"written: {real}")
    else:
        print("dry run" if changed else "nothing to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
