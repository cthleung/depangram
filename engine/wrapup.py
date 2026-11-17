"""The wrap-up (plan 5.6) and the text handling the whole-text round shares with it.

    as_posted()      a document's in-scope sections as a reader would paste them: example lines back in, comments,
                     heading marks and bold marks out (the scratch scorer of 2026-11-08, score_sections.py)
    layout()         every window's runs as the run loop builds them, without the loop (no paid call)
    run_versions()   every text each run has stood as: its original, every banked version, and "" for a cut,
                     including the versions banked under a window the sync retired, for the same original runs
    match_paragraph  whether a paragraph is made of those versions, run by run (so a paragraph the loop wrote,
                     or one where a single run was put back by hand, is still recognized as the loop's)
    write_back()     the loop's paragraphs into the real document: prose blocks only, inside the sections in
                     scope, every other byte as it was (comments, the other sections, your own new prose)

The functions that cut a document into sections take `style`, the job's SectionStyle (engine/job.py); None is the
numbered "## N." default.

    python -m engine.wrapup --job NAME              # dry run: what the write-back would change in the document
    python -m engine.wrapup --job NAME --write      # do it

The CLI takes the loop's text from the working copy (the loop writes it at every stop), so it ships exactly what
the loop would. Only the run loop calls Pangram; this module never does.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from .job import NUMBERED, Bank, Job, flat, sha, split_author_runs, style_of
from .texts import COMMENT_RE, strip_markdown

BLOCK_SPLIT = re.compile(r"(\n\s*\n)")
LEDGER = "written back.jsonl"      # every paragraph text engine/writeback.py has put in the real guide, in the job folder


def ledger(job_dir):
    """The write-back's ledger: [{"when", "label", "orig", "text", "source"}], oldest first. `orig` and `text` are
    normalized plain text. The paragraph pass's text lives nowhere in the loop's bank, so this is how the sync step
    and a later write-back know a paragraph in the real guide as the machine's."""
    path = Path(job_dir) / LEDGER
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def norm(t):
    return " ".join(t.split())


def restore_examples(md, swaps):
    """The working copy with every {{example N.N}} placeholder put back as the original line."""
    for ph, line in swaps:
        md = md.replace(ph, line)
    return md


def as_posted(md, sections, style=None):
    """{section: the text a reader would paste}: comments, heading marks and bold marks out, example lines in. Line
    endings go to LF first, so a file read as bytes (CRLF on Windows) scores as the same text the cache holds."""
    md = md.replace("\r\n", "\n")
    return {n: strip_markdown(raw).strip() for n, raw in (style or NUMBERED).texts(md, sections).items()}


def guide_as_posted(md):
    """The whole document as a reader would paste it today, every section included (the title and the intro, the
    sections in scope, the stubs and drafts outside it, any closing matter): comments, heading marks and bold marks
    out, line endings to LF first. The loop's wrap-up and engine/guide_scan.py both build the text here, so the
    same document is one text to the Pangram cache whichever of them scores it."""
    return strip_markdown(md.replace("\r\n", "\n")).strip()


def flagged_segments(segs):
    """Flagged segment texts, plus each pair of flagged neighbors joined (the run loop's flag_list)."""
    segs = [(norm(t), bool(f)) for t, f in segs]
    return [t for t, f in segs if f] + [a + " " + b for (a, fa), (b, fb) in zip(segs, segs[1:]) if fa and fb]


def window_runs(text, segs, vm, locked, bank, wid):
    """One window's runs as the loop builds them, with the bank's entries for `wid` put into that shape: (heading,
    runs, original runs)."""
    from engine.draws import build_runs
    who, heading, body, runs0 = build_runs(text, flagged_segments(segs or []), verdicts=vm, notation=locked)
    runs, index = split_author_runs(runs0, locked, verdict_of=who.get)
    bank.reshape(wid, runs0, index)
    return heading, runs, [[r["sents"][:] for r in p] for p in runs]


def layout(job, bank, vm=None):
    """{wid: {"section", "heading", "runs", "original"}}: the runs the loop builds for each window (your lines one
    sentence a run), with the bank's entries put into that shape. Needs no model and no paid call.

    It also leaves `bank.retired_versions` (2026-11-15), which run_versions() reads: for every window the sync
    retired (its start entry's id is no window now: its text changed, or the cut moved), the runs its start text
    builds, keyed by each paragraph's original runs, with every version that window banked for them. A paragraph
    whose window was only re-cut keeps the same original runs, so the loop's text there still splits into its runs
    and is still known as the loop's. The key is the runs' exact sentences, so a version never crosses to a run it
    wasn't written for.

    And `bank.gone_spans` (2026-11-15, round 4): {the sha of a Claude span of a retired window's paragraph (as the
    paragraph pass keys its picks, engine/regrow.py's orig_key): that paragraph's original text}, for every such
    paragraph that no window holds now. A paragraph's original text only changes when a sync takes it as your edit
    (or you deleted it), so a pick made for one of these spans is gone because you rewrote or removed its
    paragraph, and the pass drops it from its scoring without making the scoring stale. In memory only."""
    from .checks import is_template
    vm = vm if vm is not None else job.verdict_map()
    heads = job.headings()               # the section headings and the job's protected lines too
    locked = lambda s: is_template(s) or s in heads  # noqa: E731
    out = {}
    for w in job.windows():
        st = bank.start(w["wid"])
        heading, runs, original = window_runs(w["text"], st["segs"] if st else [], vm, locked, bank, w["wid"])
        out[w["wid"]] = {"section": w["section"], "heading": heading, "runs": runs, "original": original}
    retired, gone, gone_sections = {}, {}, {}
    bank.retired_versions = {}                 # read by run_versions below, so the current windows' own come first
    now = {norm(flat(p)) for w in out.values() for p in w["original"]}
    for wid in dict.fromkeys(e.get("window") for e in bank.entries if e.get("kind") == "start"):
        st = bank.start(wid) if wid else None
        if not wid or wid in out or not st or not st.get("text"):
            continue
        _, runs, original = window_runs(st["text"], st.get("segs"), vm, locked, bank, wid)
        for p, rp in zip(original, runs):
            para = norm(flat(p))
            if para in now:
                continue
            span = []
            for r in rp + [{"claude": False}]:           # maximal runs of Claude's, as the pass's spans_of cuts them
                if r["claude"] and not r.get("author"):
                    span.append(" ".join(r["sents"]))
                elif span:
                    key = sha(" ".join(" ".join(span).split()))
                    gone.setdefault(key, para)
                    m = re.match(r"s(\d+)w", wid)
                    if m:
                        gone_sections.setdefault(key, int(m.group(1)))
                    span = []
        for p, vers in zip(original, run_versions(bank, wid, original)):
            key = tuple(tuple(r) for r in p)
            slot = retired.setdefault(key, [set() for _ in p])
            for have, more in zip(slot, vers):
                have |= more
    bank.retired_versions = retired
    bank.gone_spans = gone
    bank.gone_sections = gone_sections         # each gone span's section, from its retired window's id
    return out


def run_versions(bank, wid, original):
    """[[set of normalized texts, per run], per paragraph]: what each run has ever stood as. "" when a cut took it.
    A paragraph whose original runs a retired window had too (bank.retired_versions, left by layout()) also takes
    what that window banked for them."""
    out = [[{norm(" ".join(r))} for r in p] for p in original]
    for pi, p in enumerate(original):
        for have, more in zip(out[pi], (getattr(bank, "retired_versions", None) or {}).get(
                tuple(tuple(r) for r in p), ())):
            have |= more
    shape = [len(p) for p in original]
    for e in bank.of(wid):
        runs = e.get("runs")
        if runs is None or [len(p) for p in runs] != shape:
            continue
        for pi, p in enumerate(runs):
            for ri, r in enumerate(p):
                out[pi][ri].add(norm(" ".join(r)))
    return out


def paragraph_history(bank):
    """{normalized original paragraph: every normalized text the loop banked for it}, across every window id in the
    bank, a window the sync step set aside included (its start entry still holds the window's text). So a
    paragraph the loop wrote stays recognizable after a hand edit elsewhere in its window gave the window a new
    history. The write-back's ledger is in it too."""
    out = {}
    starts = {e["window"]: e["text"] for e in bank.entries if e.get("kind") == "start" and e.get("window") and e.get("text")}
    for wid, text in starts.items():
        paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        if paras and re.match(r"^\d+\.\s", paras[0]) and len(paras[0].split()) < 12:
            paras = paras[1:]                  # the section heading isn't a paragraph of runs
        keys = [norm(p) for p in paras]
        for k in keys:
            out.setdefault(k, {k})
        for e in bank.of(wid):
            runs = e.get("runs") if e.get("runs") is not None else e.get("runs_unreadable")
            if runs and len(runs) == len(keys):
                for k, p in zip(keys, runs):
                    out[k].add(norm(" ".join(s for r in p for s in r)))
    for e in ledger(bank.path.parent):         # what the write-back put in the guide (the paragraph pass's text too)
        out.setdefault(e["orig"], {e["orig"]}).add(e["text"])
    return out


def recognized(key, versions, known=()):
    """A paragraph is the loop's or the original's: made of its runs' versions, or a whole text banked for it."""
    return key in known or (bool(versions) and match_paragraph(key, versions) is not None)


def match_paragraph(text, versions):
    """The version picked for each run when `text` is exactly those versions joined by spaces, else None. Longer
    versions are tried first, and a run may be empty (a cut)."""
    text = norm(text)
    memo = {}

    def go(ri, pos):
        if (ri, pos) in memo:
            return memo[(ri, pos)]
        if ri == len(versions):
            res = [] if pos >= len(text) else None
            memo[(ri, pos)] = res
            return res
        res = None
        for v in sorted(versions[ri], key=len, reverse=True):
            if not v:
                rest = go(ri + 1, pos)
            elif text.startswith(v, pos) and (pos + len(v) == len(text) or text[pos + len(v)] == " "):
                rest = go(ri + 1, pos + len(v) + 1)
            else:
                continue
            if rest is not None:
                res = [v] + rest
                break
        memo[(ri, pos)] = res
        return res
    return go(0, 0)


def rebold(new, old_block):
    """Bold marks come back where the bolded words survive (as Job.assemble does)."""
    for bold in re.findall(r"\*\*(.+?)\*\*", old_block):
        if bold in new and f"**{bold}**" not in new:
            new = new.replace(bold, f"**{bold}**", 1)
    return new


def section_spans(raw, sections, style=None):
    """{section: (start, end)} for the sections in `sections`, cut as `style` cuts them (SectionStyle.spans; the
    numbered default ends a section at the next "## " heading of any kind)."""
    return (style or NUMBERED).spans(raw, sections)


def is_prose_block(block):
    """A block a reader reads as prose: not a heading, not only comments, not only quoted example lines."""
    body = COMMENT_RE.sub("", block).strip()
    if not body or body.startswith("#"):
        return False
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    return not all(ln.startswith('"') and ln.endswith('"') for ln in lines)


def write_back(real_text, sections, paragraphs, style=None):
    """The real document with the loop's paragraphs in. `paragraphs` is a list of (label, versions, new plain text,
    known texts): a prose block inside a section in scope that `recognized` gives to one becomes the new text,
    with its bold marks put back. Every other byte stays: comments, headings, example lines, the other sections, and
    any prose the loop doesn't recognize (your edits since, or text it never had). Returns (new text, report)."""
    report = {"replaced": [], "same": [], "unrecognized": [], "missing": []}
    used = set()
    out, last = [], 0
    spans = section_spans(real_text, sections, style)
    for n in sorted(spans, key=lambda k: spans[k][0]):
        a, b = spans[n]
        out.append(real_text[last:a])
        blocks = BLOCK_SPLIT.split(real_text[a:b])
        for i, block in enumerate(blocks):
            body = COMMENT_RE.sub("", block).strip()
            if i % 2 or not body or body.startswith("#"):
                continue                       # a separator, a comment-only block or a heading: never looked up
            key = norm(strip_markdown(block))
            hit = next((k for k, para in enumerate(paragraphs)
                        if k not in used and recognized(key, para[1], para[3] if len(para) > 3 else ())), None)
            # a block that opens and closes on a quote mark looks like an example line, but when it's one of the
            # loop's paragraphs it's prose (one at the 2026-11-12 sync opened on a quoted phrase and closed on a
            # quoted word); the windows never hold an example line, so the match settles it
            if not is_prose_block(block) and hit is None:
                continue
            if hit is None:
                report["unrecognized"].append((n, key[:90]))
                continue
            used.add(hit)
            label, new = paragraphs[hit][0], paragraphs[hit][2]
            if norm(new) == key:
                report["same"].append(label)
                continue
            lead = block[:len(block) - len(block.lstrip())]
            tail = block[len(block.rstrip()):]
            blocks[i] = lead + rebold(new, block) + tail
            report["replaced"].append(label)
        out.append("".join(blocks))
        last = b
    out.append(real_text[last:])
    report["missing"] = [paragraphs[k][0] for k in range(len(paragraphs)) if k not in used]
    return "".join(out), report


def paragraphs_from_working(job, lay, bank):
    """(label, versions, new plain text, known) for every window paragraph, the new text read from the working copy (the
    loop writes it at every stop). The working copy's blocks sit where the placeholder base's blocks sit, since
    Job.assemble swaps whole blocks."""
    base_md, _ = job.placeholder_base()
    base = BLOCK_SPLIT.split(base_md)
    work = BLOCK_SPLIT.split(job.working.read_text(encoding="utf-8")) if job.working.exists() else base
    if len(work) != len(base):
        raise RuntimeError("the working copy's blocks don't line up with the original's; let the loop rewrite it")
    index = {}
    for i, blk in enumerate(base):
        k = norm(strip_markdown(COMMENT_RE.sub("", blk)))
        if k:
            index.setdefault(k, i)
    out = []
    history = paragraph_history(bank)
    for wid, w in lay.items():
        versions = run_versions(bank, wid, w["original"])
        for pi, p in enumerate(w["original"]):
            if not any(r["claude"] or r.get("author") for r in w["runs"][pi]):
                continue                       # a subsection heading or a template: locked, never written
            i = index.get(norm(flat(p)))
            new = norm(strip_markdown(work[i])) if i is not None else norm(flat(p))
            out.append((f"{wid} p{pi + 1}", versions[pi], new, history.get(norm(flat(p)), set())))
    return out


def report_lines(report):
    L = [f"{len(report['replaced'])} paragraph(s) replaced, {len(report['same'])} already as the loop has them."]
    if report["replaced"]:
        L.append("Replaced: " + ", ".join(report["replaced"]))
    if report["missing"]:
        L.append("Not found in the document (edited or moved since, so left alone): " + ", ".join(report["missing"]))
    for n, t in report["unrecognized"]:
        L.append(f"Section {n}, left as it is (not text the loop knows): {t}")
    return L


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="write the loop's text back into the document (plan 5.6)")
    ap.add_argument("--job", help="the job, a folder in jobs/ (default: [job] default in config.toml)")
    ap.add_argument("--real", help="the document (default: job.toml's document)")
    ap.add_argument("--write", action="store_true", help="write it; without this it's a dry run")
    a = ap.parse_args(argv)
    job = Job(a.job)
    real = Path(a.real) if a.real else job.real_file
    bank = Bank(job.bank_path)
    lay = layout(job, bank)
    paras = paragraphs_from_working(job, lay, bank)
    raw = real.read_bytes().decode("utf-8")
    new, report = write_back(raw, job.sections, paras, style=style_of(job))
    print("\n".join(report_lines(report)))
    if a.write and new != raw:
        real.write_bytes(new.encode("utf-8"))
        print(f"written: {real}")
    else:
        print("dry run" if new != raw else "nothing to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
