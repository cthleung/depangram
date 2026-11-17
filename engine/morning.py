"""The morning submission: the newest write-back committed on the work branch, published when publishing is on,
with a digest sent through the notifier.

    python -m engine.morning          # do it
    python -m engine.morning --dry    # say what it would do: no file written, no git, no post

The git checkout is the repo holding the job's document, or [git] repo_root in config.toml. With [git] publish off
(the default) the submission commits on the work branch and stops there: no pull, no push, no publish. With it on,
the publish branch ([git] publish_branch, default main) holds your master copy of the document, as
engine/main_copy.py describes.

In order:
  1. Where the two copies of the document stand (engine/main_copy.py). Uncommitted changes in this branch's copy
     stop it: they aren't the morning's to commit. A checkout on the publish branch, or on another branch than
     [git] work_branch when that's set, stops it too. With publishing on: the publish branch ahead (you edited it)
     brings your copy over and the sync step takes it in; both moved: the published copy wins (the write-back is
     rebuilt from the banks anyway), then the sync. The pulled copy and what the sync wrote are committed first, so
     the published copy is in this branch's history and the publish below can go forward from it.
     Before any pull, the paragraph pass's sidecar `regrow origs.json` is made sure of (a read-only load of the
     pass writes it when it's missing): a sync shifts the labels, and a sidecar written after one records them
     shifted. When it can't be written and a sync is due, the submission stops before the pull.
     This step is take_main, and main_first runs it on its own: the night run (engine/night.py) calls it before the
     paragraph pass and hands the result to submit(early=...), so the submission later finds nothing to pull and
     keeps the night's pull, sync and commits as its own.
  2. The repair step (engine/repair.py), unless [repair] mode is "off": small edits for the factual slips in what
     the paragraph pass serves, written to review.json. An error in it goes in the digest and never stops the rest.
  3. The write-back, a full run (engine/writeback.py's run). Straight after a sync it keeps the loop's text the
     document holds, since the sync started the working copy again from the original.
  4. The document and the job files the steps wrote (the ledger, the reviewer's file, the repair log, the paragraph
     pass's bank, picks and sidecar, and after a sync the files it rewrote and its backups) are committed on this
     branch and nothing else. Job files outside the repo, or ones git ignores, stay out of the commit. When [git]
     push is on the branch is pushed, and when [git] publish is on the document is published. Nothing is forced: a
     refused publish or a failed push is reported and waits for the next morning.
  5. The digest, through the notifier (a short message and the full report as a file) and in digests/. A failed
     post never undoes the submission. After the post, one line goes at the end of the saved file and in the log:
     "Posted on Discord: yes", or "no" and the reason. The night run's own lines, when it passes them, open the
     digest.

config.toml: [git] (repo_root, remote, work_branch, publish_branch, publish, push; see engine/main_copy.py's
git_config), and [morning]: post (default true), at_digest (default true: the loop runs this right after its own
digest), take_main (default true; false stops step 1 before any pull when the publish branch moved, so an edit
there waits untouched). [repair] mode turns step 2 off.
"""
import argparse
import datetime as dt
import sys
import time
from pathlib import Path

from . import main_copy
from .job import Bank, Job
from .main_copy import NoRepo, Refused, git, git_config, repo_for
from .settings import SANDBOX
from .wrapup import LEDGER

# the paragraph pass's and the repair step's files in the job folder, committed with the morning's write-back
PASS_FILES = ("review.json", "regrow origs.json", "repair log.jsonl", "regrow bank.jsonl", "regrow picks.json")

# [morning] take_main = false: a moved publish branch stops the submission before any pull
TAKE_MAIN_OFF = ("{pb} has an edit of yours ({state}), and taking an edit on {pb} in is switched off ([morning]"
                 " take_main = false). Nothing was pulled, synced, written, committed or published: your edit on {pb}"
                 " is untouched and stays as it is, and the bold and the comments stay in the document one more day")
NOT_PUBLISHED = "not published"     # the state when publishing is off: there's no published copy to compare


class Stop(Exception):
    """A reason to stop the submission before it pulls, said as it is in the digest."""


def ensure_origs(job, log=print):
    """The paragraph pass's sidecar `regrow origs.json`, written by a read-only load of the pass when it's missing and
    the pass has a bank to speak for (with no bank there is no older entry to read under a label). True when written."""
    from .regrow import ORIGS_FILE, RegrowPass
    if (job.dir / ORIGS_FILE).exists() or not (job.dir / "regrow bank.jsonl").exists():
        return False
    rp = RegrowPass(job=job, bank=Bank(job.bank_path), settings=None, conn=None, log=log)
    rp.load_units(list(job.sections))
    rp.load_bank()
    return True


def sync_paths(job, today):
    """Every file the sync step writes in the job folder, the backups it makes today included."""
    backed_up = [job.original, job.your_lines, job.bank_path, job.dir / "job.toml"]
    backups = [f.with_name(f"{f.stem} (before sync {today}){f.suffix}") for f in backed_up]
    return backed_up + backups + [job.examples_file, job.working, job.locks]


def inside(repo, path):
    """`path` relative to `repo` as a posix string, or None when it's outside the repo."""
    try:
        return Path(path).resolve().relative_to(Path(repo).resolve()).as_posix()
    except ValueError:
        return None


def commit(repo, rels, message):
    """Commit exactly these paths on the current branch (anything else staged stays staged and out of it). Paths
    outside the repo (None) and paths git ignores are left out. Returns the short commit id, or None when none of
    them changed."""
    rels = sorted({r for r in rels if r and ((repo / r).exists() or git("ls-files", "--", r, cwd=repo))})
    ignored = set(git("check-ignore", "--no-index", "--", *rels, cwd=repo, check=False).splitlines()) if rels else set()
    rels = [r for r in rels if r not in ignored or git("ls-files", "--", r, cwd=repo)]
    if not rels or not git("status", "--porcelain", "--", *rels, cwd=repo):
        return None
    git("add", "--", *rels, cwd=repo)
    git("commit", "-q", "-m", message, "--only", "--", *rels, cwd=repo)
    return git("rev-parse", "--short", "HEAD", cwd=repo)


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def repair_counts(res):
    """The repair step's outcomes as a short phrase ("1 repaired, 2 vetoed"), or "" when it wrote nothing."""
    c = (res or {}).get("counts") or {}
    if not (res or {}).get("written"):
        return ""
    return ", ".join(f"{c[k]} {k if k != 'for you' else 'sent to you'}" for k in ("repaired", "vetoed", "for you")
                     if c.get(k))


def commit_message(today, report, synced, repairs="", name="", pb="main"):
    bits = [plural(len(report["replaced"]), "paragraph") + " newer"]
    if report.get("for_you"):
        bits.append(f"{len(report['for_you'])} for you")
    if repairs:
        bits.append(f"repair step: {repairs}")
    if synced:
        bits.append(f"your edit on {pb} synced in first")
    return f"{name or 'depangram'}: the morning submission of {today} ({', '.join(bits)})"


def sections_of(labels):
    return {int(label.split("w")[0][1:]) for label in labels}


def nothing_happened(r):
    return not (r["stopped"] or r["changed"] or r["pulled"] or r["commits"] or r["published"] or r["problems"]
                or r.get("repair_news") or (r.get("night") or {}).get("lines"))


def clean(e):
    """An error's text with keys, tokens and URLs out (git's stderr can name the remote)."""
    from .util import redact
    return redact(e)[:300]


def digest(r):
    """The report: one line when nothing happened, else the full text for the file."""
    from .writeback import report_lines
    dry = " (dry run)" if r["dry"] else ""
    lost = [f"the stuck entry for {label} matches no unit now: write a new one or remove it"
            for label in (r["report"] or {}).get("stuck_unmatched") or ()]
    if nothing_happened(r):
        k = len((r["report"] or {}).get("for_you") or [])
        tail = (f" nothing to pull, and {r.get('publish_branch', 'main')} already has this branch's copy."
                if r.get("publishing", True)
                else " publishing is off, so there's nothing to pull or publish.")
        return (f"The morning submission of {r['day']}{dry}: nothing new to write back,{tail}"
                + (f" {plural(k, 'paragraph')} still in bold for you to rewrite." if k else "")
                + "".join(f" {x[0].upper()}{x[1:]}." for x in lost))
    L = [f"# The morning submission, {r['stamp']}{dry}", ""]
    if (r.get("night") or {}).get("lines"):
        L += ["## The night run", "", *r["night"]["lines"], ""]
    if r["stopped"]:
        L.append(f"Stopped: {r['stopped']}")
        L.append(f"Committed on {r['branch']} before it stopped, not pushed or published: {', '.join(r['commits'])}"
                 if r["commits"] else "Nothing was committed, pushed or published.")
        return "\n".join(L) + "\n"
    rep = r["report"] or {}
    for_you = rep.get("for_you") or []
    L.append(f"{plural(len(rep.get('replaced') or []), 'paragraph')} took newer text"
             + (f", {len(for_you)} in bold for you to rewrite by hand" if for_you else "") + ".")
    if r.get("publishing", True):
        pb = r.get("publish_branch", "main")
        L += ["", f"{pb[:1].upper()}{pb[1:]}'s copy and this branch's at the start: {r['state']}."
              + (f" {r['pulled']}" if r["pulled"] else "")]
    else:
        L += ["", "Publishing is off ([git] publish), so there's no published copy to compare or pull."]
    if for_you:
        L += ["", "## For you to rewrite by hand", "",
              "Each is in bold in the document, under a comment that holds the prompt and the original."]
        for label in for_you:
            about = " ".join(str((rep.get("prompts") or {}).get(label, {}).get("about") or "").split())
            L.append(f"- {label}" + (f": {about}" if about else ""))
    written = sections_of(rep.get("replaced", []) + rep.get("marked", []) + rep.get("restored", []) + for_you)
    if written or r["notes"]:
        L += ["", "## Sections written", ""]
        for n in sorted(written | set(r["notes"])):
            L.append(f"- Section {n}" + ("" if n in written else " (nothing new)")
                     + (f": {r['notes'][n]}" if r["notes"].get(n) else ""))
    if r.get("repair_lines") or lost:
        L += ["", "## Repairs", "", *(r.get("repair_lines") or []), *[f"- {x}" for x in lost]]
    if rep:
        L += ["", "## The write-back", "", *report_lines(dict(rep, for_you=[], stuck_unmatched=[]))]
    if r["sync"]:
        L += ["", "## What the sync took in", "", *r["sync"]]
    L += ["", "## Git", ""]
    if r["dry"]:
        L.append("A dry run: nothing committed, pushed or published.")
    else:
        L.append(f"- On {r['branch']}: " + (", ".join(r["commits"]) if r["commits"] else "no new commit")
                 + (" (pushed)" if r["pushed"] else ""))
        L.append(f"- On {r.get('publish_branch', 'main')}: {r['published'] or r.get('publish_note') or 'not published'}")
    L += [f"- {p}" for p in r["problems"]]
    return "\n".join(L) + "\n"


def short_message(r):
    night = (r.get("night") or {}).get("short")
    head = f"**depangram morning submission, {r['day']}:** " + (f"the night: {night}; " if night else "")
    if r["stopped"]:
        return f"{head}stopped: {r['stopped']}"[:1900]
    rep = r["report"] or {}
    k = len(rep.get("for_you") or [])
    return (f"{head}{plural(len(rep.get('replaced') or []), 'paragraph')}"
            f" newer" + (f", {k} in bold for you to rewrite" if k else "")
            + (f"; repair step: {repair_counts(r.get('repair'))}" if repair_counts(r.get("repair")) else "")
            + (f"; on {r.get('publish_branch', 'main')} as {r['published']}" if r["published"]
               else f"; {r['publish_note'] or 'not on ' + r.get('publish_branch', 'main')}")
            + (f"; {plural(len(r['problems']), 'problem')}, see the report" if r["problems"] else "") + ".")


def new_result(today, stamp, dry):
    return {"day": today, "stamp": stamp, "dry": dry, "rel": None, "branch": None, "stopped": None, "state": None,
            "pulled": None, "sync": None, "report": None, "notes": {}, "changed": False, "commits": [], "pushed": None,
            "published": None, "publish_note": None, "problems": [], "posted": None, "posted_line": None,
            "digest": None, "repair": None, "repair_lines": None, "repair_news": False, "night": None,
            "synced": False, "job": None, "publish_branch": "main", "publishing": True}


def take_main(job, r, repo, dry, *, sync_fn, origs_fn, today, log, remote, branch, allow=True, publishing=True,
              work_branch=""):
    """Step 1, filling `r` as it goes (so a failure part way keeps what was pulled and committed): where the two copies
    stand, the refusals, the sidecar made sure of, and when the publish branch (`branch`) moved the pull, the sync
    and the commit of what the sync wrote. Returns the job (loaded again after a sync) and whether a sync ran; sets
    r["stopped"] and returns when it refuses. A state already in `r` (the night's earlier call) stays: it is the
    state at the start. With `allow` False ([morning] take_main = false) a moved publish branch stops it before the
    sidecar and the pull: nothing is written. With `publishing` False there's no published copy: nothing is
    fetched or pulled, and only the branch and the uncommitted changes are checked."""
    from . import sync as sync_step
    repo = Path(repo).resolve()
    real = job.real_file
    r["publish_branch"], r["publishing"] = branch, publishing
    if real is None:
        r["stopped"] = "the job names no document (document in job.toml), so there's nothing to commit"
        return job, False

    def rel_of(p):
        return inside(repo, p)

    r["rel"] = rel = rel_of(real)
    if rel is None:
        r["stopped"] = f"the job's document ({real}) isn't inside the repo the submission commits in ({repo})"
        return job, False
    r["branch"] = here = git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo)
    if here == "HEAD" or (publishing and here == branch):
        r["stopped"] = f"this checkout is on {here}, and the morning submission only commits on a work branch"
        return job, False
    if work_branch and here != work_branch:
        r["stopped"] = (f"this checkout is on {here}, not the work branch {work_branch} ([night] branch, [git]"
                        " work_branch or the job's branch); nothing was committed")
        return job, False
    if not publishing:
        r["state"] = r["state"] or NOT_PUBLISHED
        head_blob = main_copy.blob_at("HEAD", rel, repo)
        work_blob = git("hash-object", "--", rel, cwd=repo) if (repo / rel).exists() else None
        if work_blob != head_blob:
            r["stopped"] = ("this branch's copy of the document has changes that aren't committed; they aren't the"
                            " morning's to commit, so commit or drop them first")
            return job, False
        if not dry:                        # the sidecar, as on a publishing morning, for a hand sync later
            try:
                if origs_fn(job, log=log):
                    log("morning: wrote the paragraph pass's regrow origs.json")
            except Exception as e:  # noqa: BLE001
                r["problems"].append(f"the paragraph pass's regrow origs.json couldn't be written"
                                     f" ({type(e).__name__}: {clean(e)})")
        return job, False
    st = main_copy.status(repo, rel, remote=remote, branch=branch)
    r["state"] = r["state"] or st["state"]
    if st["dirty"]:
        r["stopped"] = ("this branch's copy of the document has changes that aren't committed; they aren't the"
                        " morning's to commit, so commit or drop them first")
        return job, False
    due = st["state"] in ("main ahead", "diverged")
    if due and not allow:
        r["stopped"] = TAKE_MAIN_OFF.format(state=st["state"], pb=branch)
        return job, False
    if not dry:
        try:
            if origs_fn(job, log=log):
                log("morning: wrote the paragraph pass's regrow origs.json before any sync")
        except Exception as e:  # noqa: BLE001
            why = f"the paragraph pass's regrow origs.json couldn't be written ({type(e).__name__}: {clean(e)})"
            if due:
                raise Stop(why + ", and a sync is due: one now would shift the labels it has to record") from e
            r["problems"].append(why)
    if not due:
        return job, False
    both = st["state"] == "diverged"
    if dry:
        r["pulled"] = (f"It would pull {branch}'s copy" + (f" (both moved, so {branch}'s wins)" if both else "")
                       + " and sync it in first; the write-back below is this branch's copy as it stands.")
        return job, False
    main_copy.pull(repo, rel, take_main=both, remote=remote, branch=branch)
    r["pulled"] = f"{branch[:1].upper()}{branch[1:]}'s copy was pulled" + (f" (both moved, so {branch}'s won)"
                                                                       if both else "")
    log(f"morning: {branch}'s copy pulled ({st['state']}); the sync takes it in")
    rep = sync_fn(job.dir, real, write=True, today=today)
    r["pulled"] += " and synced in."
    r["sync"] = (r["sync"] or []) + sync_step.report_text(rep).splitlines()
    job = Job(root=job.dir)
    sid = commit(repo, [rel] + [rel_of(p) for p in sync_paths(job, today)],
                 f"{job.name}: {branch}'s copy taken in for the morning submission of {today}")
    if sid:
        r["commits"].append(sid)
    return job, True


def error_stop(r, e, dry):
    """The stop text for an error the steps didn't expect."""
    text = f"an error, {type(e).__name__}: {clean(e)}"
    if r["pulled"] and not r["commits"] and not dry:
        text += (f". The {r.get('publish_branch', 'main')} copy sits in this branch's document, not committed, so the next"
                 " morning stops too until it's committed or dropped")
    return text


def main_first(job, repo=None, dry=False, *, settings=None, sync_fn=None, origs_fn=None, today=None, stamp=None,
               log=print, remote=None, branch=None):
    """Step 1 on its own, for the night run (engine/night.py): when the publish branch's copy moved, pull it and sync
    it in before the paragraph pass, so the pass works on your newest text; the same refusals and the same commits
    as the submission's own step 1. Returns a result dict whose "job" is the job to carry on with and "synced" says
    a sync ran; hand it to submit(early=...), which then finds nothing to pull and keeps what this one did.
    `settings` gives [git] and [morning] take_main (true when it's missing). `repo` defaults to repo_for(job)."""
    from . import sync as sync_step
    today = today or dt.date.today().isoformat()
    r = new_result(today, stamp or time.strftime("%Y-%m-%d %H:%M"), dry)
    r["job"] = job
    g = git_config(settings)
    remote, branch = remote or g["remote"], branch or g["publish_branch"]
    r["publish_branch"], r["publishing"] = branch, g["publish"]
    try:
        repo = repo_for(job, settings, repo)
        allow = bool(((settings.config.get("morning") if settings is not None else None) or {}).get("take_main", True))
        r["job"], r["synced"] = take_main(job, r, repo, dry, sync_fn=sync_fn or sync_step.sync,
                                          origs_fn=origs_fn or ensure_origs, today=today, log=log, remote=remote,
                                          branch=branch, allow=allow, publishing=g["publish"],
                                          work_branch=work_branch_of(job, settings))
    except (Stop, NoRepo) as e:
        r["stopped"] = str(e)
    except Exception as e:  # noqa: BLE001
        r["stopped"] = error_stop(r, e, dry)
    return r


def work_branch_of(job, settings):
    """The branch the engine commits on: [night] branch (older configs), else [git] work_branch, else the job's
    branch in job.toml, else "" (any branch but the publish branch)."""
    cfg = (getattr(settings, "config", None) or {}) if settings is not None else {}
    return (str((cfg.get("night") or {}).get("branch") or "").strip() or git_config(settings)["work_branch"]
            or str(getattr(job, "branch", None) or "").strip())


def submit(job, repo=None, settings=None, dry=False, *, writeback=None, sync_fn=None, post=None, repair_fn=None,
           origs_fn=None, today=None, stamp=None, digests=None, log=print, remote=None, branch=None,
           early=None, night=None, repair_mode=None):
    """The whole submission. `writeback`, `sync_fn`, `post`, `repair_fn` and `origs_fn` stand in for
    engine.writeback.run, engine.sync.sync, engine.discord.post, engine.repair.run and ensure_origs (tests pass
    fakes). `early` is main_first's result when the night ran step 1 before the pass: its pull, sync and commits
    count as this submission's, and the state it found is the state at the start. `night` is {"lines": [...],
    "short": "..."}: the night run's own lines, put at the top of the digest, and its phrase for the short message.
    `repair_mode` overrides [repair] mode for this run ("off" on a night past your Claude ceiling). `repo` defaults
    to repo_for(job); `remote` and `branch` (the publish branch) to [git]. Returns a result dict; its "digest" is
    the text posted and saved, and "posted_line" says whether the notifier took it."""
    from . import discord
    from . import sync as sync_step
    from . import writeback as wb
    if settings is None:
        from .settings import Settings
        settings = Settings()
    cfg = settings.config.get("morning", {})
    writeback, sync_fn, post = writeback or wb.run, sync_fn or sync_step.sync, post or discord.post
    origs_fn = origs_fn or ensure_origs
    repair_mode = repair_mode or (settings.config.get("repair") or {}).get("mode", "apply")
    today = today or dt.date.today().isoformat()
    stamp = stamp or time.strftime("%Y-%m-%d %H:%M")
    g = git_config(settings)
    remote, branch = remote or g["remote"], branch or g["publish_branch"]

    def rel_of(p):
        return inside(repo, p)

    r = new_result(today, stamp, dry)
    r["night"] = night
    r["publish_branch"], r["publishing"] = branch, g["publish"]
    synced = False
    if early:
        job = early.get("job") or job
        synced = bool(early.get("synced"))
        r["state"], r["pulled"] = early.get("state"), early.get("pulled")
        r["sync"], r["commits"] = list(early.get("sync") or []) or None, list(early.get("commits") or [])
    try:
        repo = repo_for(job, settings, repo)
        job, again = take_main(job, r, repo, dry, sync_fn=sync_fn, origs_fn=origs_fn, today=today, log=log,
                               remote=remote, branch=branch, allow=bool(cfg.get("take_main", True)),
                               publishing=g["publish"], work_branch=work_branch_of(job, settings))
        synced = synced or again
        real = job.real_file
        rel, here = r["rel"], r["branch"]
        if not r["stopped"]:
            if repair_mode != "off":
                try:                           # imported here, so a broken import is a problem in the digest too
                    from . import repair as repair_step
                    r["repair"] = (repair_fn or repair_step.run)(job, settings, None, mode="dry" if dry else None,
                                                                 today=today, log=log)
                    r["repair_lines"] = repair_step.report_lines(r["repair"])
                    r["repair_news"] = repair_step.has_news(r["repair"])
                except Exception as e:  # noqa: BLE001
                    r["problems"].append(f"the repair step failed, and the write-back went on without it:"
                                         f" {type(e).__name__}: {clean(e)}")
            report, notes, changed = writeback(job, write=not dry, stamp=stamp, real=real, after_sync=synced)
            r["report"], r["notes"], r["changed"] = report, notes, changed
            log(f"morning: the write-back {'would change' if dry else 'changed'} the document" if changed
                else "morning: the write-back has nothing new")
            if not dry:
                message = commit_message(today, report, synced, repair_counts(r["repair"]), job.name, branch)
                paths = ([rel] + ([rel_of(job.dir / LEDGER)] if report.get("ledger") else [])
                         + [rel_of(job.dir / f) for f in PASS_FILES])
                sid = commit(repo, paths, message)
                if sid:
                    r["commits"].append(sid)
                if r["commits"] and g["push"]:
                    try:
                        git("push", remote, here, cwd=repo)
                        r["pushed"] = here
                    except RuntimeError as e:
                        r["problems"].append(f"the push of {here} failed, so it waits for the next morning: {clean(e)}")
                if g["publish"]:
                    try:
                        out = main_copy.publish(repo, rel, message=message if sid else None, remote=remote,
                                                branch=branch)
                        if out.startswith("published"):
                            r["published"] = git("rev-parse", "--short", f"{remote}/{branch}", cwd=repo)
                        else:
                            r["publish_note"] = out
                    except Refused as e:
                        r["problems"].append(f"the publish to {branch} was refused: {clean(e)}")
                    except RuntimeError as e:
                        r["problems"].append(f"the publish to {branch} failed: {clean(e)}")
                else:
                    r["publish_note"] = "not published ([git] publish is off)"
    except (Stop, NoRepo) as e:
        r["stopped"] = str(e)
    except Exception as e:  # noqa: BLE001
        r["stopped"] = error_stop(r, e, dry)
    r["synced"] = synced
    r["digest"] = text = digest(r)
    label = posted_label(settings)
    if dry:
        r["posted_line"] = f"{label}: no, a dry run"
        return r
    one_line = nothing_happened(r)
    name = f"{today} morning submission.md"
    saved = None
    try:
        ddir = Path(digests) if digests else SANDBOX / "digests"
        ddir.mkdir(parents=True, exist_ok=True)
        saved = ddir / name
        saved.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    except OSError as e:
        saved = None
        log(f"morning: the digest wasn't saved: {type(e).__name__}")
    why = "[morning] post is off"
    if notifier_of(settings) == "none":
        why = "the notifier is none (NOTIFIER)"
    elif cfg.get("post", True):
        try:
            got = post(settings, text) if one_line else post(settings, short_message(r), file_name=name,
                                                             file_text=text)
            r["posted"] = bool(got)
            why = ("Discord didn't take it" if notifier_of(settings) == "discord" else "the notifier didn't take it")
            why += " (the post came back empty)"
        except Exception as e:  # noqa: BLE001
            r["posted"] = False
            why = f"the post failed ({type(e).__name__})"     # the class only: a requests error carries the webhook URL
            log(f"morning: the digest post failed: {type(e).__name__}")
    r["posted_line"] = f"{label}: yes" if r["posted"] else f"{label}: no, {why}"
    if saved is not None:
        try:
            with saved.open("a", encoding="utf-8") as f:
                f.write("\n" + r["posted_line"] + "\n")
        except OSError as e:
            log(f"morning: the posted line wasn't added to the saved digest: {type(e).__name__}")
    log(f"morning: {r['posted_line']}")
    log(f"morning: {r['stopped'] or 'done'}" + (f", on main as {r['published']}" if r["published"] else ""))
    return r


def notifier_of(settings):
    """The notifier's name, "discord" when settings can't say (a test double with a config only)."""
    from . import discord
    try:
        return discord.notifier(settings)
    except Exception:  # noqa: BLE001
        return "discord"


def posted_label(settings):
    """The start of the digest's last line: "Posted on Discord", or where the file notifier wrote it."""
    kind = notifier_of(settings)
    return {"discord": "Posted on Discord", "file": "Written to the notifications folder"}.get(kind, "Posted")


def run_for_loop(loop, crashed=None, submit_fn=None):
    """The loop's call, right after its own digest. Off with [morning] at_digest = false; skipped when the day
    crashed and in a rehearsal, a smoke test or a --window run; a loop built without a run's settings (a test double)
    never submits. Sets loop.morning_result."""
    if not getattr(loop, "morning_on", False):
        loop.morning_result = {"skipped": "at_digest is off"}
    elif crashed:
        loop.morning_result = {"skipped": "the day crashed"}
    elif getattr(loop, "rehearsal", False) or loop.args.smoke or getattr(loop.args, "window", None):
        loop.morning_result = {"skipped": "a rehearsal, smoke or partial run"}
    else:
        loop.morning_result = (submit_fn or submit)(loop.job, settings=loop.settings, log=loop.log)
    return loop.morning_result


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="the morning submission: the write-back committed on the work branch,"
                                             " published when [git] publish is on, and sent as a digest")
    ap.add_argument("--job", help="the job (a folder in jobs/); default [job] default in config.toml")
    ap.add_argument("--job-dir", help="a job folder anywhere, instead of --job")
    ap.add_argument("--dry", action="store_true", help="say what it would do: no file, no git, no post")
    a = ap.parse_args(argv)
    from .job import JobError
    from .main_copy import load_job
    from .settings import Settings
    settings = Settings()
    try:
        job = load_job(a.job, a.job_dir, settings)
    except JobError as e:
        print(f"the morning submission can't start: {e}")
        return 1
    r = submit(job, settings=settings, dry=a.dry)
    print(r["digest"])
    print(r["posted_line"])
    return 1 if r["stopped"] or r["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
