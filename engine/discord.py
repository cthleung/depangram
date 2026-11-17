"""Notifications: Discord, a folder of files, or nothing, picked by NOTIFIER in .env (or [notify] notifier in
config.toml): "discord" (the default), "file" or "none".

On Discord, messages over the 2,000-character limit go up as an attached markdown file with a short summary on top.
There are two ways out (plan 9.1). With DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID in .env, posts go through
Discord's REST API as the bot, so every post has a message id the author can answer, and the loop can read the
replies and reactions (replies_since, reactions, ask, answer_to). Without them the webhook path
(DISCORD_WEBHOOK_URL) posts one way, and the reading side returns nothing.

With NOTIFIER=file, post() appends each message to posts.md in NOTIFY_DIR (default: notifications/ in RUNS_DIR)
and writes an attached report next to it under its own name. With NOTIFIER=none, post() drops the message. Either
way the bot path is off, so the loop asks nothing and reads nothing.

The run loop's reports (the hourly post, the digest, the couldn't-crack report) are built in engine/digest.py; the
names are kept here as aliases."""
import datetime as dt
import json
import re
from pathlib import Path
from urllib.parse import quote

import requests

from .digest import (_ai, _hm, arm_lines, as_posted_lines, couldnt_crack_report, crack_lines,  # noqa: F401
                     digest_card, digest_report, efficiency_lines, efficiency_rows, hourly_text, is_yours,
                     model_lines, open_questions_line, reflection_lines, restore_lines, spend_lines, tiny_lines,
                     whole_flag_lines, window_rows)
from .settings import choice, setting_path

LIMIT = 2000
API = "https://discord.com/api/v10"
TIMEOUT = 15                    # seconds; a slow Discord reads as "no answer", never as a stall
# Discord asks bots for a User-Agent of this shape; nothing in it names the author
AGENT = "DiscordBot (https://github.com/discord/discord-api-docs, 1.0)"
# the reactions that answer a question, built from code points so the source stays plain ASCII
VS16, KEYCAP = chr(0xFE0F), chr(0x20E3)
THUMBS_UP = chr(0x1F44D)
CROSS = chr(0x274C)
CROSSES = {CROSS, chr(0x2716) + VS16, chr(0x2716), chr(0x274E)}
KEYCAPS = {f"{d}{VS16}{KEYCAP}": d for d in range(1, 10)} | {f"{d}{KEYCAP}": d for d in range(1, 10)}
_said = set()                   # what the reading side has already said once in the log


NOTIFIERS = ("discord", "file", "none")


def notifier(settings):
    """The notifier, discord, file or none: NOTIFIER in .env or the environment, else [notify] notifier, else discord."""
    return choice(settings, "NOTIFIER", "notify", "notifier", "discord", NOTIFIERS)


def where(settings):
    """Where a post goes, for the logs and the digests' "Posted ..." line."""
    return {"discord": "on Discord", "file": f"to {setting_path('NOTIFY_DIR', settings)}",
            "none": "nowhere (NOTIFIER is none)"}[notifier(settings)]


def bot_on(settings):
    """The bot path is on when the notifier is Discord and both the bot token and the channel id are in .env (plan
    9.1)."""
    return (notifier(settings) == "discord"
            and bool(settings.get("DISCORD_BOT_TOKEN") and settings.get("DISCORD_CHANNEL_ID")))


def say_once(key, text):
    if key not in _said:
        _said.add(key)
        print(text, flush=True)


def _headers(settings):
    # the token goes in this header and nowhere else: never a log line, never an error message
    return {"Authorization": f"Bot {settings.get('DISCORD_BOT_TOKEN')}", "User-Agent": AGENT}


def bot_post(settings, content, file_name=None, file_text=None, session=None, reply_to=None):
    """One message through the bot (Create Message). Returns its message id, or None. The 2,000-character cut and
    the attached markdown file work the way the webhook's do; mentions are switched off so a quoted line can't
    ping anybody."""
    return _bot_send(settings, content, file_name, file_text, session, reply_to)[0]


def _bot_send(settings, content, file_name=None, file_text=None, session=None, reply_to=None):
    """bot_post's work: (message id or None, why it failed). The why is the HTTP status, or the error's class name
    for a connection error, never the request (it carries the token)."""
    http = session or requests
    url = f"{API}/channels/{settings.get('DISCORD_CHANNEL_ID')}/messages"
    payload = {"content": content[:LIMIT], "allowed_mentions": {"parse": []}}
    if reply_to:
        payload["message_reference"] = {"message_id": str(reply_to), "fail_if_not_exists": False}
    try:
        if file_text is not None:
            files = {"files[0]": (file_name or "report.md", file_text.encode("utf-8"), "text/markdown")}
            r = http.post(url, headers=_headers(settings), data={"payload_json": json.dumps(payload)}, files=files,
                          timeout=30)
        else:
            r = http.post(url, headers=_headers(settings), json=payload, timeout=30)
    except Exception as e:  # noqa: BLE001  (the class only: a requests error can carry the request)
        print(f"discord bot post failed: {type(e).__name__}", flush=True)
        return None, type(e).__name__
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    try:
        return (str(r.json().get("id") or "") or None), None
    except ValueError:
        return None, "a reply with no message in it"


def send(settings, content, file_name=None, file_text=None, session=None):
    """The bot first, then the webhook when the bot's post fails (an HTTP error, a 401 or 403 from a bad token, a
    connection error) and DISCORD_WEBHOOK_URL is set, so alerts and the digest reach you on a bad token. Returns
    (message id or None, went out or not). A post that went through the webhook has no id, so it can't be
    answered. The fallback is said once a day in the log, with the status, never the token."""
    mid, why = _bot_send(settings, content, file_name, file_text, session)
    if mid:
        return mid, True
    if not settings.get("DISCORD_WEBHOOK_URL"):
        return None, False
    say_once(f"fallback {dt.date.today().isoformat()}",
             f"discord: the bot's post failed ({why}), so posts go through the webhook today, with no message id to"
             " answer; check DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID with python -m engine.discord_check")
    return None, _webhook_post(settings, content, file_name, file_text, session)


def _get(settings, path, params=None, session=None):
    """A GET on the channel through the bot. Anything but a 200 (a 429 included) is None: no answer, the default."""
    http = session or requests
    try:
        r = http.get(f"{API}/channels/{settings.get('DISCORD_CHANNEL_ID')}{path}", headers=_headers(settings),
                     params=params, timeout=TIMEOUT)
    except Exception as e:  # noqa: BLE001
        say_once(f"get {type(e).__name__}", f"discord read failed: {type(e).__name__} (read as no answer)")
        return None
    if r.status_code != 200:
        say_once(f"status {r.status_code}", f"discord read got HTTP {r.status_code} (read as no answer)")
        return None
    try:
        return r.json()
    except ValueError:
        return None


def _slim(m):
    a = m.get("author") or {}
    ref = m.get("message_reference") or {}
    return {"id": str(m.get("id")), "author_id": str(a.get("id", "")), "bot": bool(a.get("bot")),
            "content": m.get("content") or "", "reply_to": str(ref["message_id"]) if ref.get("message_id") else None,
            "time": m.get("timestamp", "")}


OFF = ("discord: the bot path is off (no DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID), so replies and reactions read"
       " as none; the webhook path posts as before")


def replies_since(settings, message_id, session=None, pages=5):
    """Every message in the channel after `message_id`, oldest first, as small dicts (id, author_id, bot, content,
    reply_to, time). Get Channel Messages hands back newest first whichever way it pages, so this sorts by id
    itself and pages on the highest id seen. Empty on the webhook path, said once in the log."""
    if not bot_on(settings):
        say_once("off", OFF)
        return []
    if not message_id:                          # a post that went through the webhook has no id to read after
        return []
    out, after = {}, str(message_id)
    for _ in range(pages):
        got = _get(settings, "/messages", {"after": after, "limit": 100}, session)
        if not got:
            break
        for m in got:
            out[str(m.get("id"))] = _slim(m)
        after = str(max(int(m.get("id")) for m in got))
        if len(got) < 100:
            break
    return sorted(out.values(), key=lambda m: int(m["id"]))


def reactions(settings, message_id, session=None):
    """{emoji: [{"id", "bot"}, ...]} for one message. The message object only carries counts, so the users behind
    each emoji come from Get Reactions. Empty on the webhook path."""
    if not bot_on(settings):
        say_once("off", OFF)
        return {}
    if not message_id:
        return {}
    m = _get(settings, f"/messages/{message_id}", None, session)
    out = {}
    for r in (m or {}).get("reactions") or []:
        em = r.get("emoji") or {}
        name = f"{em['name']}:{em['id']}" if em.get("id") else em.get("name")
        if not name:
            continue
        users = _get(settings, f"/messages/{message_id}/reactions/{quote(name)}", {"limit": 100}, session) or []
        out[name] = [{"id": str(u.get("id", "")), "bot": bool(u.get("bot"))} for u in users]
    return out


def is_author(settings, user):
    """Only the author's answers count: their Discord user id when DISCORD_AUTHOR_ID is set, else anybody who isn't
    a bot (the bot's own posts and any other bot's)."""
    aid = settings.get("DISCORD_AUTHOR_ID")
    uid = str(user.get("author_id") or user.get("id") or "")
    return uid == str(aid) if aid else not user.get("bot")


def parse_answer(text, n_options):
    """A bare number, or a number at the start of a reply ("2", "2.", "#2 keep it"), as the option number; else
    None."""
    m = re.match(r"\s*#?(\d+)(?!\d)", text or "")
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= n_options else None


def reaction_answer(emoji, n_options, veto=None):
    """A reaction as an option number: thumbs-up is option 1 (yes), a cross is the veto option, a number emoji is
    that option."""
    if emoji.startswith(THUMBS_UP):
        return 1
    if emoji in CROSSES:
        return veto
    n = KEYCAPS.get(emoji)
    return n if n and n <= n_options else None


def ask(settings, question, options, default=1, veto=None, session=None):
    """Post a question with numbered options (1., 2., 3.) through the bot. Returns its message id, or None on the
    webhook path (a question nobody can answer is only noise) or when the post failed. When the bot's post fails
    and the webhook is set, the question goes out through the webhook so you still see it, and this returns None:
    with no id it can't be answered, so the loop treats it as unasked and the default applies."""
    if not bot_on(settings):
        return None
    lines = [f"**depangram asks:** {question}"]
    lines += [f"{i}. {o}{' (default)' if i == default else ''}" for i, o in enumerate(options, 1)]
    how = f"{THUMBS_UP} for 1" + (f", {CROSS} for {veto}" if veto else "") + ", or a number emoji"
    lines.append(f"Reply with the number, or react ({how}). No answer means the default.")
    text = "\n".join(lines)
    if len(text) > LIMIT:                      # the options and the how-to always fit; the question gives way
        over = len(text) - LIMIT + 3
        text = "\n".join([lines[0][:max(0, len(lines[0]) - over)] + "...", *lines[1:]])
    return send(settings, text, session=session)[0]


def answer_to(settings, message_id, n_options, veto=None, replies=None, until_id=None, session=None):
    """The author's answer to one question, as its option number (1 is the first option), or None: unanswered means
    the code's default. A Discord reply to the question counts first, the latest one that parses; then a plain
    message posted after the question and before the next one (`until_id`), since on a phone a bare "2" is the
    natural answer; then the author's reactions on it, when they point at one option only."""
    if not bot_on(settings) or not message_id:
        return None
    if replies is None:
        replies = replies_since(settings, message_id, session=session)
    mine = [m for m in replies if is_author(settings, m)]
    mid = int(message_id)
    direct = [m for m in mine if m.get("reply_to") == str(message_id)]
    loose = [m for m in mine if not m.get("reply_to") and int(m["id"]) > mid
             and (until_id is None or int(m["id"]) < int(until_id))]
    for group in (direct, loose):
        for m in reversed(group):
            n = parse_answer(m.get("content"), n_options)
            if n is not None:
                return n
    picked = set()
    for emoji, users in reactions(settings, message_id, session=session).items():
        if any(is_author(settings, u) for u in users):
            n = reaction_answer(emoji, n_options, veto)
            if n is not None:
                picked.add(n)
    return picked.pop() if len(picked) == 1 else None


def post(settings, content, file_name=None, file_text=None, session=None):
    """One post, through the notifier. On Discord the bot path returns the new message's id (a non-empty string, so
    it still reads as true), the webhook path True; False when nothing went out. The webhook request is the one it
    always was. On the bot path a failed post falls back to the webhook (send), and returns True when that went
    out. NOTIFIER=file writes it (file_post) and returns True; NOTIFIER=none drops it and returns True, since
    nothing failed."""
    kind = notifier(settings)
    if kind == "none":
        return True
    if kind == "file":
        return file_post(settings, content, file_name, file_text)
    if bot_on(settings):
        mid, went = send(settings, content, file_name=file_name, file_text=file_text, session=session)
        return mid or went
    return _webhook_post(settings, content, file_name, file_text, session)


def _webhook_post(settings, content, file_name=None, file_text=None, session=None):
    url = settings.get("DISCORD_WEBHOOK_URL")
    if not url:
        return False
    http = session or requests
    payload = {"username": "depangram", "content": content[:LIMIT]}
    if file_text is not None:
        files = {"file": (file_name or "report.md", file_text.encode("utf-8"), "text/markdown")}
        r = http.post(url, data={"payload_json": json.dumps(payload)}, files=files, timeout=30)
    else:
        r = http.post(url, json=payload, timeout=30)
    return r.status_code in (200, 204)


def file_post(settings, content, file_name=None, file_text=None, folder=None):
    """NOTIFIER=file: the message appended to posts.md in NOTIFY_DIR under a time stamp, and an attached report
    written next to it under its own name (a later report with the same name replaces it). True when written."""
    d = Path(folder) if folder else setting_path("NOTIFY_DIR", settings)
    d.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with (d / "posts.md").open("a", encoding="utf-8", newline="\n") as f:
        f.write(f"===== {stamp}{' + ' + file_name if file_name else ''}\n{content}\n")
    if file_text is not None:
        name = Path(file_name or "report.md").name
        (d / name).write_text(file_text, encoding="utf-8", newline="\n")
    return True
