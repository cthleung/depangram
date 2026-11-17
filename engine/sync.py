"""The sync step (plan 12): take the full document you hand over into the job.

    python -m engine.sync --job NAME               # dry run: what would change, nothing written
    python -m engine.sync --job NAME --write       # do it (backs up what it replaces, in the job folder)
    python -m engine.sync --job-dir COPY --source FILE --write    # on a copy, for testing

--job may be left out when config.toml names a default job ([job] default). Sections are cut the way the job's
job.toml says (engine/job.py's SectionStyle), and a section's id is its number in a numbered "## N." document, else
its place in the document.

What it does, in order:
  1. The new frozen original is your file, with one exception. A paragraph in a section the loop already works on
     that the loop recognizes as its own (every run is its original, a banked regrowth, or cut, as an earlier
     write-back put it there; or a text in the write-back's ledger, which is how the paragraph pass's text is
     known) goes back to the frozen original's text. The write-back's marker comments
     (engine/writeback.py: the commented originals and the section notes) come off first. Without that, the
     loop's lines would come in as new text, the windows would stop matching their bank history, and the chat-log
     scan would call Mistral's lines yours. A paragraph it doesn't recognize is your edit, and your text stands.
     Under a FOR YOU TO REWRITE marker, bold is the machine's flag and never your emphasis (2026-11-15): a
     recognized paragraph there goes back to the frozen original's block even when its words are the original's,
     and one that comes in as your edit is frozen with every bold mark out and the old original's own bold words
     back where it still has them. Bold you type anywhere else stays yours.
  2. Scope widens to every section with prose in it: a line that isn't a heading, a comment or an example line.
     A section that's only comments stays out.
  3. Example lines: quoted-only lines in the new sections join the job's list (`example lines.txt`), so they're
     swapped for placeholders and stay out of the prose. Listed in the report, so you can move any back.
  4. `your lines.md`: every verdict already frozen stays as it is. Sentences new to the job are looked up once:
     a sentence the bank shows the loop wrote gets "loop" (regrown like Claude's, no protected words), the rest
     go through the chat-log scan (engine/find_lines.py) once, and the result is frozen again. A job with
     verdicts = "file" skips the scan: a sentence the file doesn't list stays the machine's.
  5. The bank: a window whose text didn't change keeps its history under its (possibly new) id, keyed by text.
     A window whose text changed has its history kept in the file under a retired id, and starts at round 0.
  6. The working copy starts again from the new original, with one exception (2026-11-15): every block whose
     original is the same block it was (same words, same placeholders) keeps the text the old working copy had
     there, which is what the loop shipped at its last stop (Job.assemble of each window's shipped entry). Without
     that, every paragraph would stand on Claude's original until the loop's next stop, and the loop isn't on a
     schedule: the paragraph pass would assemble its mixed paragraphs around the original, and sections nobody
     touched would change in the guide and lose their scores. job.toml's sections and seed windows are updated,
     and locks.md is rewritten.

Nothing here calls Pangram or Claude. The chat-log scan reads local files only.
"""
import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

from .job import LIST_LINE, NUMBERED, Bank, Job, cut_windows, job_name, prose_scope  # noqa: F401
from .texts import COMMENT_RE, strip_markdown
from .wrapup import (BLOCK_SPLIT, is_prose_block, layout, ledger, norm, paragraph_history, rebold, recognized,
                     run_versions)
from .writeback import MARK_RE, is_for_you, strip_markers

QUOTED_LINE = re.compile(r'^"[^"]+"$')


def numbered_sections(md, style=None):
    """Every section id in `md`, sorted (the name is older than SectionStyle: any style's ids come back)."""
    return (style or NUMBERED).ids(md)


def prose_sections(md, examples, style=None, skip_lists=True):
    """(in scope, {section: why it's left out}): engine.job.prose_scope, the one rule the sync and a job's
    `sections = "all"` share."""
    return prose_scope(md, examples, style, skip_lists=skip_lists)


def new_example_lines(md, sections, known, style=None):
    """Quoted-only lines (outside comments) in `sections` that aren't in `known` yet."""
    out = []
    for raw in (style or NUMBERED).texts(md, sections).values():
        for ln in COMMENT_RE.sub("", raw).splitlines():
            ln = ln.strip()
            if QUOTED_LINE.match(ln) and ln not in known and ln not in out:
                out.append(ln)
    return out


def carry_working(old_base, old_work, new_base):
    """The new working copy: `new_base` (the placeholder base of the new original) with every block whose text is,
    exactly, a block of `old_base` (the old original's placeholder base; compared without the whitespace around it)
    carrying the block the old working copy `old_work` has in that place. A block that sits twice in the old base,
    or in a working copy whose blocks don't line up with its base, carries nothing. Returns (text, how many blocks
    carried)."""
    ob, ow = BLOCK_SPLIT.split(old_base), BLOCK_SPLIT.split(old_work)
    if len(ob) != len(ow):
        return new_base, 0
    seen, have = set(), {}
    for i in range(0, len(ob), 2):
        k = ob[i].strip()
        if k in seen:
            have.pop(k, None)
            continue
        seen.add(k)
        if ow[i].strip() != k:
            have[k] = ow[i].strip()
    nb = BLOCK_SPLIT.split(new_base)
    carried = 0
    for i in range(0, len(nb), 2):
        k = nb[i].strip()
        if k in have:
            lead, tail = nb[i][:len(nb[i]) - len(nb[i].lstrip())], nb[i][len(nb[i].rstrip()):]
            nb[i] = lead + have[k] + tail
            carried += 1
    return "".join(nb), carried


def marker_original(block):
    """The frozen original a write-back marker holds (the lines after "the original ...:"), normalized plain text,
    or None."""
    lines = block.strip().split("\n")
    at = next((i for i, ln in enumerate(lines) if ln.startswith("the original")), None)
    if at is None:
        return None
    body = [ln for ln in lines[at + 1:] if ln.strip() != "-->"]
    return norm(strip_markdown("\n".join(body))) or None


def for_you_paragraphs(md):
    """{the plain text (normalized, bold marks out) of every paragraph that sits right under one of the write-back's
    FOR YOU TO REWRITE markers in `md`: (the label that marker names, the original it holds)}, read before the
    markers come off. The original finds the old frozen block even when the label is an older cut's (a marker that
    didn't change keeps the label it was written under)."""
    blocks = BLOCK_SPLIT.split(md)
    out = {}
    for i in range(0, len(blocks) - 2, 2):
        if is_for_you(blocks[i]):
            m = MARK_RE.match(blocks[i].lstrip())
            out[norm(strip_markdown(blocks[i + 2]))] = (m.group(1) if m else None, marker_original(blocks[i]))
    return out


def rebuild_original(real, old_md, worked, lay, bank, bolded=None, style=None):
    """The new frozen original: `real`, with every prose block of the worked sections that the loop recognizes as
    its own put back to the frozen original's block. `bolded` is for_you_paragraphs(): the plain texts that sat under
    a FOR YOU marker, with the label each marker named. Such a block goes back even when its words are the
    original's, and one that is your edit loses every bold mark, then gets the old original block's own bold words
    back where it still has them (the marker's label says which block), so the write-back's bold never stays.
    `style` is the job's SectionStyle (the numbered default when None). Returns (text, restored labels, your edits
    as (section, text))."""
    bolded = bolded or {}
    old_blocks = {}
    for blk in BLOCK_SPLIT.split(old_md):
        k = norm(strip_markdown(COMMENT_RE.sub("", blk)))
        if k:
            old_blocks.setdefault(k, blk)
    history = paragraph_history(bank)
    paras, covered = [], set()
    for wid, w in lay.items():
        versions = run_versions(bank, wid, w["original"])
        for pi, p in enumerate(w["original"]):
            if any(r["claude"] or r.get("author") for r in w["runs"][pi]):
                orig = norm(" ".join(s for r in p for s in r))
                paras.append((f"{wid} p{pi + 1}", versions[pi], orig, history.get(orig, set())))
                covered.add(orig)
    # paragraphs of windows an earlier sync set aside: still the loop's text when the real guide has them
    for orig, known in history.items():
        if orig not in covered:
            paras.append((f"set aside: {orig[:40]}", [], orig, known))
    by_label = {label: orig for label, _, orig, _ in paras}
    restored, edits, used = [], [], set()
    spans = (style or NUMBERED).spans(real, worked)
    out, last = [], 0
    for n, (a, b) in sorted(spans.items(), key=lambda kv: kv[1][0]):
        out.append(real[last:a])
        blocks = BLOCK_SPLIT.split(real[a:b])
        for i, blk in enumerate(blocks):
            if i % 2 or not is_prose_block(blk):
                continue
            key = norm(strip_markdown(blk))
            hit = next((k for k, (_, versions, _, known) in enumerate(paras)
                        if k not in used and recognized(key, versions, known)), None)
            if hit is None:
                if key in bolded:              # your edit of a paragraph in bold for you: the machine's bold comes out
                    lead, tail = blk[:len(blk) - len(blk.lstrip())], blk[len(blk.rstrip()):]
                    label, held = bolded[key]
                    old = old_blocks.get(held) or old_blocks.get(by_label.get(label))
                    plain = blk.strip().replace("**", "")
                    blocks[i] = lead + (rebold(plain, old) if old else plain) + tail
                edits.append((n, key))
                continue
            used.add(hit)
            label, orig = paras[hit][0], paras[hit][2]
            if (key != orig or key in bolded) and orig in old_blocks:
                lead = blk[:len(blk) - len(blk.lstrip())]
                tail = blk[len(blk.rstrip()):]
                blocks[i] = lead + old_blocks[orig].strip() + tail
                restored.append(label)
        out.append("".join(blocks))
        last = b
    out.append(real[last:])
    return "".join(out), restored, edits


def loop_sentences(bank):
    """Every sentence the loop has banked (originals too: a sentence already in a window has a verdict, so only the
    loop's own lines are ever looked up here), and every sentence the write-back put in the guide (the paragraph
    pass's text is Mistral's as well, and it's in no bank entry of the loop's)."""
    from engine.draws import sents
    out = set()
    for e in bank.entries:
        for p in e.get("runs") or []:
            for r in p:
                for s in r:
                    out.add(norm(s))
    for e in ledger(bank.path.parent):
        out.update(norm(s) for s in sents(e["text"]))
    return out


def sync(job_dir, source, write=False, classify=None, today=None):
    """Returns a report dict. `classify` stands in for engine.find_lines.classify (tests pass a stub)."""
    from engine.draws import sents
    today = today or dt.date.today().isoformat()
    job = Job(root=job_dir)
    bank = Bank(job.bank_path)
    old_md = job.original.read_text(encoding="utf-8")
    old_base = job.placeholder_base()[0]       # read before the new original is written, for carry_working
    old_work = job.working.read_text(encoding="utf-8") if job.working.exists() else None
    real = Path(source).read_bytes().decode("utf-8").replace("\r\n", "\n")
    bolded = for_you_paragraphs(real)          # their bold is the write-back's, read before the markers come off
    real = strip_markers(real)                 # the write-back's commented originals and section notes aren't the guide
    old_md_n = old_md.replace("\r\n", "\n")
    vm = job.verdict_map()
    lay = layout(job, bank, vm)
    old_windows = {w["wid"]: w["text"] for w in job.windows()}
    # 1. the new original (the frozen copy keeps your file's line endings)
    # the sections the loop has ever worked, the ones out of scope today included: a section can leave the scope with
    # the loop's text still in the document, and without this the next sync would take that text for yours and
    # freeze it as the original
    style = job.style
    worked = set(job.sections) | {int(m.group(1)) for e in bank.entries
                                  for m in [re.match(r"s(\d+)w\d+", e.get("window") or "")] if m}
    new_md, restored, edits = rebuild_original(real, old_md_n, worked, lay, bank, bolded, style=style)
    edits = [(n, t) for n, t in edits if n in job.sections]     # out of scope, every line the loop never had is simply yours
    # 3. example lines, before the scope (an example line isn't prose)
    known = job.example_lines()
    candidates = numbered_sections(new_md, style)
    added_examples = new_example_lines(new_md, [n for n in candidates if n not in job.sections], known, style)
    # quotes the old original already kept as prose (a famous human quote the author cites) stay prose
    old_quoted = {ln.strip() for ln in old_md_n.splitlines() if QUOTED_LINE.match(ln.strip())} - known
    added_examples = [ln for ln in added_examples if ln not in old_quoted]
    examples = known | set(added_examples)
    # 2. scope
    sections, left_out = prose_sections(new_md, examples, style, skip_lists=job.skip_list_sections)
    new_windows = cut_windows(new_md, sections, examples, style)
    # 5. bank: history follows the window text
    by_text = {}
    for w in new_windows:
        by_text.setdefault(w["text"], w["wid"])
    remap, kept, retired = {}, [], []
    for wid, text in old_windows.items():
        if text in by_text:
            remap[wid] = by_text[text]
            kept.append((wid, by_text[text]))
        else:
            remap[wid] = f"{wid}~before-sync-{today}"
            retired.append(wid)
    fresh = [w["wid"] for w in new_windows if w["wid"] not in {n for _, n in kept}]
    # 4. verdicts for sentences new to the job
    loop_said = loop_sentences(bank)
    # a sentence you typed into a paragraph that sat under a FOR YOU marker is yours: it's in no original and the
    # loop never wrote it, and the chat-log scan can't know it (you typed it in the guide). Short ones too, or the
    # layout would read a verdict-less sentence as Claude's and the pass would regrow your line
    old_sents = {norm(x) for blk in BLOCK_SPLIT.split(strip_markdown(COMMENT_RE.sub("", old_md_n))) for x in sents(blk)}
    typed = {x for _, key in edits if key in bolded for x in sents(key)
             if norm(x) not in old_sents and norm(x) not in loop_said}
    new_sents = []
    heads = ({strip_markdown(ln).strip() for ln in new_md.splitlines() if ln.startswith("#")}
             | set(style.titles(new_md).values()) | job.protected_lines())
    for w in new_windows:
        for p in w["text"].split("\n\n"):
            if p.strip() in heads:
                continue                       # a heading is locked; it never needs a verdict
            for s in sents(p):
                if vm.get(s) is None and (len(s.split()) >= 5 or s in typed) and s not in new_sents:
                    new_sents.append(s)
    verdicts_new = {}
    to_scan = []
    for s in new_sents:
        if norm(s) in loop_said:
            verdicts_new[s] = ("loop", "-", "-")
        elif s in typed:
            verdicts_new[s] = ("yours", "-", "-")
        elif job.verdict_source == "scan":
            to_scan.append(s)
        # verdicts = "file": a sentence the file doesn't list is the machine's, so it needs no row
    if to_scan:
        if classify is None:
            from .find_lines import classify as classify_real
            classify = lambda text: classify_real(text)[0]  # noqa: E731
        for s, v, u, c in classify("\n\n".join(to_scan)):
            verdicts_new[s] = (v, f"{u:.0%}", f"{c:.0%}")
    counts = {}
    for v, _, _ in verdicts_new.values():
        counts[v] = counts.get(v, 0) + 1
    seed_windows = [w for w in job.cfg.get("seed", {}).get("windows", []) if remap.get(w, w) == w]
    report = {"source": str(source), "sections_before": job.sections, "sections_after": sections,
              "left_out": left_out, "restored": restored, "your_edits": edits, "kept": kept, "retired": retired,
              "fresh": fresh, "examples_added": added_examples, "new_verdicts": counts,
              "seed_windows": seed_windows, "written": False, "carried": 0}
    if not write:
        return report
    # backups, then the job folder
    stamp = today
    for f in (job.original, job.your_lines, job.bank_path, job.dir / "job.toml"):
        if f.exists():
            shutil.copy2(f, f.with_name(f"{f.stem} (before sync {stamp}){f.suffix}"))
    crlf = b"\r\n" in Path(source).read_bytes()
    job.original.write_bytes((new_md.replace("\n", "\r\n") if crlf else new_md).encode("utf-8"))
    job.examples_file.write_text("\n".join(sorted(examples)) + "\n", encoding="utf-8")
    rows = [f"| {v} | {u} | {c} | - | {s.replace('|', '/')} |" for s, (v, u, c) in verdicts_new.items()]
    if rows:
        with job.your_lines.open("a", encoding="utf-8") as f:
            f.write("\n".join(["", f"## Added at the sync of {stamp}", "",
                               "New to the job. \"loop\" means the bank shows the loop wrote it (regrown like Claude's"
                               " lines, no protected words). \"yours\" with no percentages is a sentence you"
                               " typed under a FOR YOU marker. The rest came from the chat-log scan, once.", "",
                               "| Verdict | In your messages | In Claude's output | Pangram | Sentence |",
                               "| --- | --- | --- | --- | --- |", *rows]) + "\n")
    with job.bank_path.open("w", encoding="utf-8") as f:
        for e in bank.entries:
            e = dict(e)
            if "runs_unreadable" in e:
                e["runs"] = e.pop("runs_unreadable")     # as the file had it; the next load sorts it out again
            if e.get("window") in remap:
                e["window"] = remap[e["window"]]
            if e.get("kind") == "whole_end":
                continue                       # its picks name the old windows; the next whole-text round is due
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    toml = (job.dir / "job.toml").read_text(encoding="utf-8")
    toml = re.sub(r"(?m)^sections = \[[^\]]*\]", "sections = [" + ", ".join(map(str, sections)) + "]", toml)
    if "[seed]" in toml:
        toml = re.sub(r"(?m)^windows = \[[^\]]*\]", "windows = [" + ", ".join(f'"{w}"' for w in seed_windows) + "]",
                      toml)
    (job.dir / "job.toml").write_text(toml, encoding="utf-8")
    if job.working.exists():
        job.working.unlink()
    fresh_job = Job(root=job_dir)
    if old_work is not None:                   # the loop's shipped text stays where the original didn't move
        text, report["carried"] = carry_working(old_base, old_work, fresh_job.placeholder_base()[0])
        fresh_job.working.write_text(text, encoding="utf-8")
    fresh_job.setup()
    report["written"] = True
    return report


def report_text(r):
    L = [f"Sync from {r['source']}{'' if r['written'] else ' (dry run: nothing written)'}",
         f"Sections in scope: {r['sections_before']} -> {r['sections_after']}"]
    for n, why in sorted(r["left_out"].items()):
        L.append(f"Section {n} left out: {why}")
    L.append(f"Windows kept with their bank history: {len(r['kept'])}"
             + (" (" + ", ".join(f"{a}->{b}" if a != b else a for a, b in r["kept"]) + ")" if r["kept"] else ""))
    if r["retired"]:
        L.append(f"Windows whose text changed, history set aside, starting at round 0: {', '.join(r['retired'])}")
    L.append(f"New windows at round 0: {len(r['fresh'])}" + (f" ({', '.join(r['fresh'])})" if r["fresh"] else ""))
    L.append(f"Paragraphs of the loop's put back to the frozen original's text: {len(r['restored'])}")
    for n, t in r["your_edits"]:
        L.append(f"- your edit, kept as you wrote it (section {n}): {t[:100]}")
    if r["examples_added"]:
        L.append("Example lines added (out of the prose, swapped for placeholders); say if one is a quote that should"
                 " stay in:")
        L += [f"- {x}" for x in r["examples_added"]]
    L.append("New verdicts: " + (", ".join(f"{k} {v}" for k, v in sorted(r["new_verdicts"].items())) or "none"))
    L.append(f"Seed windows kept: {r['seed_windows']}")
    if r.get("written"):
        L.append(f"Working copy: {r.get('carried', 0)} block(s) keep the loop's text, the rest start at the new original")
    return "\n".join(L)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="bring the full document into the job (plan 12, the sync step)")
    ap.add_argument("--job", help="the job, a folder in jobs/ (default: [job] default in config.toml)")
    ap.add_argument("--job-dir", help="a job folder to work on (a copy, for testing); default jobs/<job>")
    ap.add_argument("--source", help="the document to take in; default job.toml's document")
    ap.add_argument("--write", action="store_true", help="write it; without this it's a dry run")
    a = ap.parse_args(argv)
    job = Job(root=a.job_dir) if a.job_dir else Job(job_name(a.job))
    if a.source is None and job.real_file is None:
        ap.error("the job names no document: pass --source FILE or set document in its job.toml")
    source = Path(a.source) if a.source else job.real_file
    print(report_text(sync(job.dir, source, write=a.write)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
