"""The loop's questions on Discord and the author's answers (plan 9.1, GitHub issue #8).

The loop never waits on an answer. Where it asks, the code's default goes ahead (a drop-rule cut happens), and an
answer that comes later can only do one of three things: put a cut back (the veto), rest a window for the rest of
the run-day, or stop the day the way the STOP file does. An answer can never raise a cap, change a target or
spend money, and nothing in this module calls Pangram or Claude.

How the author answers:
    a cut question     reply with the number ("2"), or react: thumbs-up for 1, a cross for the veto, a number emoji
    stop the day       a plain message "stop" in the channel, after today's start post
    rest a window      a plain message "rest s3w0" (or "retire s3w0"), after today's start post

The questions and each run-day's start post live in one JSON file next to the pool queue (11.2), so a veto
answered after the evening's digest still reaches the next run-day. engine/run.py's hooks call in here only when
the bot path is on; with no bot token none of this runs.
"""
import json
import re
from pathlib import Path

from engine import discord

KEEP_SECONDS = 2 * 86400            # an unanswered question closes on its default after two days
STOP_WORDS = {"stop", "stop the day", "stop today"}
REST = re.compile(r"^\s*(?:rest|retire)\s+(s\d+w\d+)\b", re.I)
CUT_OPTIONS = ["yes", "keep it going (the line goes back in)"]
CUT_VETO = 2


class Asked:
    """The open questions (one dict each: id, kind, window, para, run, text, day, ts, options, default, veto, and
    answer once there is one) and the message id of each run-day's start post."""

    def __init__(self, path):
        self.path = Path(path)
        self.questions, self.day_posts = [], {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.questions, self.day_posts = data.get("questions", []), data.get("day_posts", {})
            except (ValueError, OSError):
                pass                    # a broken file means no open questions: every default stands

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"questions": self.questions, "day_posts": self.day_posts}, ensure_ascii=False,
                                  indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, q):
        self.questions.append(q)
        self.save()
        return q

    def open(self):
        return [q for q in self.questions if q.get("answer") is None and not q.get("closed")]

    def prune(self, now):
        """Questions past KEEP_SECONDS close on their default; closed ones that old leave the file."""
        changed = False
        for q in self.open():
            if now - q.get("ts", now) > KEEP_SECONDS:
                q["closed"] = "no answer in time, so the default stood"
                changed = True
        keep = [q for q in self.questions if now - q.get("ts", now) <= 2 * KEEP_SECONDS]
        if len(keep) != len(self.questions):
            self.questions, changed = keep, True
        if len(self.day_posts) > 3:                 # only today's start post is ever read; keep a few for looking
            self.day_posts = {d: self.day_posts[d] for d in sorted(self.day_posts)[-3:]}
            changed = True
        if changed:
            self.save()


def commands(settings, replies, since_id):
    """The author's plain messages after today's start post: whether one says stop, and the windows they asked to
    rest. Only messages after `since_id` count, so yesterday's "stop" can't stop today."""
    stop, rest = False, []
    for m in replies:
        if int(m["id"]) <= int(since_id) or not discord.is_author(settings, m):
            continue
        text = (m.get("content") or "").strip()
        if text.lower().rstrip(".!") in STOP_WORDS:
            stop = True
        r = REST.match(text)
        if r and r.group(1).lower() not in rest:
            rest.append(r.group(1).lower())
    return stop, rest


def read(settings, asked, day_post_id=None, session=None):
    """One read of the channel: ({question id: option number}, stop, [windows to rest]). One Get Channel Messages
    covers every open question and today's commands; reactions are read per question only when it has no text
    answer."""
    opens = asked.open()
    ids = [int(q["id"]) for q in opens] + ([int(day_post_id)] if day_post_id else [])
    if not ids:
        return {}, False, []
    replies = discord.replies_since(settings, min(ids), session=session)
    every = sorted(int(q["id"]) for q in asked.questions)
    got = {}
    for q in opens:
        nxt = next((i for i in every if i > int(q["id"])), None)
        n = discord.answer_to(settings, q["id"], len(q["options"]), veto=q.get("veto"), replies=replies,
                              until_id=nxt, session=session)
        if n is not None:
            got[q["id"]] = n
    stop, rest = commands(settings, replies, day_post_id) if day_post_id else (False, [])
    return got, stop, rest
