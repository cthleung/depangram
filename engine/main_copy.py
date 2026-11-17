"""The published copy of a job's document, and the work branch the engine writes on.

The document lives in a git repo (the folder holding it, or [git] repo_root in config.toml). The engine commits its
write-backs on the work branch, and, when [git] publish is on, puts the document on the publish branch ([git]
publish_branch, default main) as its master copy, which you edit there. Publishing is off by default.

    python -m engine.main_copy              # where the two copies stand, nothing changed
    python -m engine.main_copy pull         # the publish branch's copy into this branch's file; then engine.sync
    python -m engine.main_copy publish      # this branch's committed copy onto the publish branch: one commit

The round trip: you edit the document on the publish branch. `pull` brings your copy here, the sync step takes it
into the job, the loop and the paragraph pass work, `python -m engine.writeback --write` puts the newest text in
under the commented originals, the file is committed on this branch (so it goes through the commit hooks), and
`publish` puts that committed file on the publish branch.

Neither step switches branches or touches any other file. `publish` builds its commit on top of the remote's
publish branch with git's plumbing and pushes it without force, so that branch only ever moves forward. It refuses
when the published copy is a version this branch has never had (you edited it since the last pull: pull and sync
first), and when this branch's file has changes that aren't committed. `pull` refuses when this branch's file has
uncommitted changes, and when both copies moved (this branch holds a write-back the publish branch never got AND
the publish branch holds an edit of yours): then `pull --take-main` takes yours, since the write-back can be run
again and your edit can't.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .settings import SANDBOX


class Refused(RuntimeError):
    pass


def git(*args, cwd, env=None, check=True):
    r = subprocess.run(["git", *args], cwd=str(cwd), env=env, capture_output=True, text=True, encoding="utf-8")
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args[:3])}...: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


GIT_DEFAULTS = {"repo_root": "", "remote": "origin", "work_branch": "", "publish_branch": "main", "publish": False,
                "push": None}


def git_config(settings):
    """[git] from config.toml over the defaults: repo_root ("" means the repo holding the job's document), remote,
    work_branch ("" means any branch but the publish branch), publish_branch, publish (off by default) and push
    (pushing the work branch; it follows publish when unset). An older config's [morning] publish counts when [git]
    has none."""
    cfg = (getattr(settings, "config", None) or {}) if settings is not None else {}
    g = dict(GIT_DEFAULTS)
    old = cfg.get("morning") or {}
    if "publish" in old:
        g["publish"] = bool(old["publish"])
    g.update(cfg.get("git") or {})
    g["publish"] = bool(g["publish"])
    g["push"] = g["publish"] if g.get("push") is None else bool(g["push"])
    for k in ("repo_root", "remote", "work_branch", "publish_branch"):
        g[k] = str(g.get(k) or "").strip()
    g["remote"] = g["remote"] or "origin"
    g["publish_branch"] = g["publish_branch"] or "main"
    return g


class NoRepo(RuntimeError):
    """The job's document isn't in a git repo the engine can find: the message says what to set."""


def repo_for(job, settings=None, repo=None):
    """The git checkout to commit in: `repo` when given, else [git] repo_root (relative to the depangram folder),
    else the top of the repo holding the job's document."""
    if repo:
        return Path(repo).resolve()
    root = git_config(settings)["repo_root"]
    if root:
        p = Path(root).expanduser()
        return (p if p.is_absolute() else SANDBOX / p).resolve()
    doc = getattr(job, "real_file", None)
    if doc is None:
        raise NoRepo("the job names no document (document in job.toml), so there's no repo to commit in")
    folder = Path(doc).resolve().parent
    if not folder.is_dir():
        raise NoRepo(f"the job's document folder isn't there: {folder}")
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(folder), capture_output=True, text=True,
                       encoding="utf-8")
    if r.returncode or not r.stdout.strip():
        raise NoRepo(f"the job's document isn't in a git repo ({folder}); run git init there, or set [git]"
                     " repo_root in config.toml")
    return Path(r.stdout.strip()).resolve()


def load_job(name=None, job_dir=None, settings=None):
    """The job a command works on: the folder `job_dir`, else jobs/<name>, else [job] default in config.toml."""
    from .job import Job, job_name
    if job_dir:
        return Job(root=job_dir)
    return Job(job_name(name, getattr(settings, "config", None)))


def blob_at(ref, rel, cwd):
    """The file's blob id at a ref, or None when the ref doesn't hold it."""
    out = git("rev-parse", "--verify", "-q", f"{ref}:{rel}", cwd=cwd, check=False)
    return out or None


def versions(ref, rel, cwd):
    """Every blob id the file has had in the history of `ref`."""
    commits = git("log", "--format=%H", ref, "--", rel, cwd=cwd).split()
    return {b for b in (blob_at(c, rel, cwd) for c in commits) if b}


def status(repo, rel, remote="origin", branch="main", fetch=True):
    """Where the two copies stand: "same", "main behind" (this branch has newer text to publish), "main ahead"
    (main has an edit to pull), "diverged" (both moved) or "main has none"; and whether this branch's file has
    uncommitted changes."""
    if fetch:
        git("fetch", remote, branch, cwd=repo)
    ref = f"{remote}/{branch}"
    main_blob, head_blob = blob_at(ref, rel, repo), blob_at("HEAD", rel, repo)
    work_blob = git("hash-object", "--", rel, cwd=repo) if (Path(repo) / rel).exists() else None
    if main_blob is None:
        state = "main has none"
    elif main_blob == head_blob:
        state = "same"
    elif main_blob in versions("HEAD", rel, repo):
        state = "main behind"
    elif head_blob in versions(ref, rel, repo):
        state = "main ahead"
    else:
        state = "diverged"
    return {"state": state, "dirty": work_blob != head_blob, "main": main_blob, "head": head_blob, "ref": ref}


def pull(repo, rel, take_main=False, **kw):
    st = status(repo, rel, **kw)
    if st["dirty"]:
        raise Refused("this branch's file has changes that aren't committed; commit or drop them first")
    if st["state"] in ("same", "main behind", "main has none"):
        return f"nothing to pull ({st['state']})"
    if st["state"] == "diverged" and not take_main:
        raise Refused("both copies moved: this branch holds text main never got, and main holds an edit this branch"
                      " never had. `pull --take-main` takes main's (the write-back can be run again)")
    git("restore", f"--source={st['ref']}", "--worktree", "--", rel, cwd=repo)
    return "main's copy is in this branch's file now (not committed). Next: python -m engine.sync"


def publish(repo, rel, message=None, remote="origin", branch="main", **kw):
    st = status(repo, rel, remote=remote, branch=branch, **kw)
    if st["dirty"]:
        raise Refused("this branch's file has changes that aren't committed; commit it here first, then publish")
    if st["state"] == "same":
        return "nothing to publish: main already has this copy"
    if st["state"] in ("main ahead", "diverged"):
        raise Refused("main's copy is a version this branch has never had (an edit on main since the last pull);"
                      " pull and sync first")
    head = git("rev-parse", "--short", "HEAD", cwd=repo)
    here = git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo)
    message = message or f"{rel}: the copy from {here} ({head})"
    mode = (git("ls-tree", st["ref"], "--", rel, cwd=repo).split() or ["100644"])[0]
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(d) / "index"))
        git("read-tree", st["ref"], cwd=repo, env=env)
        git("update-index", "--add", "--cacheinfo", f"{mode},{st['head']},{rel}", cwd=repo, env=env)
        tree = git("write-tree", cwd=repo, env=env)
    commit = git("commit-tree", tree, "-p", st["ref"], "-m", message, cwd=repo)
    git("push", remote, f"{commit}:refs/heads/{branch}", cwd=repo)
    git("fetch", remote, branch, cwd=repo)
    local = git("rev-parse", "--verify", "-q", f"refs/heads/{branch}", cwd=repo, check=False)
    if local and here != branch and not subprocess.run(["git", "merge-base", "--is-ancestor", local, commit],
                                                        cwd=str(repo)).returncode:
        git("update-ref", f"refs/heads/{branch}", commit, local, cwd=repo)      # the local main follows, forward only
    return f"published to {remote}/{branch} as {commit[:7]}: {message}"


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="the job's document between the publish branch (the master) and this"
                                             " branch")
    ap.add_argument("action", nargs="?", default="status", choices=["status", "pull", "publish"])
    ap.add_argument("--job", help="the job (a folder in jobs/); default [job] default in config.toml")
    ap.add_argument("--job-dir", help="a job folder anywhere, instead of --job")
    ap.add_argument("--take-main", action="store_true", help="pull: take the published copy when both moved")
    ap.add_argument("--message", help="publish: the commit message on the publish branch")
    a = ap.parse_args(argv)
    from .job import JobError
    from .settings import Settings
    settings = Settings()
    g = git_config(settings)
    try:
        job = load_job(a.job, a.job_dir, settings)
        repo = repo_for(job, settings)
        if job.real_file is None:
            raise NoRepo("the job names no document (document in job.toml)")
        try:
            rel = Path(job.real_file).resolve().relative_to(repo).as_posix()
        except ValueError:
            raise NoRepo(f"the job's document ({job.real_file}) sits outside the repo ({repo}); move it into the"
                         " repo, or set [git] repo_root to the repo that holds it") from None
    except (JobError, NoRepo) as e:
        print(f"refused: {e}")
        return 1
    kw = {"remote": g["remote"], "branch": g["publish_branch"]}
    try:
        if a.action == "status":
            st = status(repo, rel, **kw)
            print(f"{rel}: {st['state']}" + ("; this branch's file has uncommitted changes" if st["dirty"] else ""))
        elif a.action == "pull":
            print(pull(repo, rel, take_main=a.take_main, **kw))
        else:
            print(publish(repo, rel, message=a.message, **kw))
    except Refused as e:
        print(f"refused: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
