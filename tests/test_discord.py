"""Tests for the two-way Discord (plan 9.1, GitHub issue #8): the bot path and the webhook path, replies and
reactions over a faked Discord REST API, the numbered questions and how an answer is read, and the loop's hooks
(a veto puts a cut back for good, "rest s3w0" rests a window for the run-day, "stop" stops the day like the STOP
file, and with no bot token nothing happens at all). Nothing here talks to Discord: `requests` is swapped for
FakeDiscord, which keeps a channel in memory and answers the endpoints the module calls. Run from
the repo root:

    python -m unittest discover tests
"""
import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from engine import answers, discord
from engine.job import Bank, window_text_of
from engine.run import Loop, Notifier, WindowState
from engine.schedule import CompressedClock
from engine.settings import Settings

TOKEN = "BOT.TOKEN.SECRET"
CHANNEL = "555"
BOT_ID, AUTHOR_ID, STRANGER_ID = "900", "100", "200"
HOOK = "https://discord.com/api/webhooks/1/HOOKSECRET"


class Resp:
    def __init__(self, status, body=None):
        self.status_code, self.body = status, body

    def json(self):
        if self.body is None:
            raise ValueError("no body")
        return self.body


class FakeDiscord:
    """A channel in memory behind the four endpoints the module uses: Create Message, Get Channel Messages (newest
    first, as Discord answers it), Get Channel Message and Get Reactions. `fail` makes every call answer with that
    status (a 429, say)."""

    def __init__(self):
        self.messages, self.reacts, self.calls = {}, {}, []
        self.next_id, self.fail, self.bot_fail = 1000, None, None

    def add(self, author, content, reply_to=None, bot=False, mid=None):
        mid = str(mid or self.next_id)
        self.next_id = max(self.next_id, int(mid)) + 1
        m = {"id": mid, "author": {"id": author, "bot": bot}, "content": content, "timestamp": "t"}
        if reply_to:
            m["message_reference"] = {"message_id": str(reply_to)}
        self.messages[mid] = m
        return mid

    def react(self, mid, emoji, user, bot=False):
        self.reacts.setdefault(mid, {}).setdefault(emoji, []).append({"id": user, "bot": bot})

    def post(self, url, headers=None, json=None, data=None, files=None, timeout=None):
        self.calls.append(("POST", url, headers, json, data, files))
        if self.fail:
            return Resp(self.fail)
        if "/webhooks/" in url:
            return Resp(204)
        if self.bot_fail:
            return Resp(self.bot_fail)
        body = json if json is not None else __import__("json").loads(data["payload_json"])
        ref = (body.get("message_reference") or {}).get("message_id")
        mid = self.add(BOT_ID, body["content"], reply_to=ref, bot=True)
        return Resp(200, self.messages[mid])

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(("GET", url, headers, params))
        if self.fail:
            return Resp(self.fail)
        path = url.split(f"/channels/{CHANNEL}", 1)[1]
        parts = path.strip("/").split("/")
        if parts == ["messages"]:
            after = int(params["after"])
            got = sorted((m for m in self.messages.values() if int(m["id"]) > after), key=lambda m: int(m["id"]))
            got = got[:params.get("limit", 50)]
            return Resp(200, list(reversed(got)))          # newest first, whichever way it pages
        if len(parts) == 2:
            m = dict(self.messages[parts[1]])
            m["reactions"] = [{"count": len(u), "emoji": {"id": None, "name": e}}
                              for e, u in self.reacts.get(parts[1], {}).items()]
            return Resp(200, m)
        if len(parts) == 4 and parts[2] == "reactions":
            from urllib.parse import unquote
            return Resp(200, self.reacts.get(parts[1], {}).get(unquote(parts[3]), []))
        return Resp(404)

    def requests_count(self):
        return len(self.calls)


def bot_settings(author=None, runs_dir=None, webhook=True, **cfg):
    env = {"DISCORD_BOT_TOKEN": TOKEN, "DISCORD_CHANNEL_ID": CHANNEL}
    if webhook:
        env["DISCORD_WEBHOOK_URL"] = HOOK
    if author:
        env["DISCORD_AUTHOR_ID"] = author
    if runs_dir:
        env["RUNS_DIR"] = str(runs_dir)
    return Settings(env=env, config={"discord": {"hourly": True, **cfg}})


def hook_settings(runs_dir=None):
    env = {"DISCORD_WEBHOOK_URL": HOOK}
    if runs_dir:
        env["RUNS_DIR"] = str(runs_dir)
    return Settings(env=env, config={"discord": {"hourly": True}})


class Refuse:
    """`requests` for the no-token tests: any call at all is a failure."""

    def post(self, *a, **k):
        raise AssertionError("no request should go out")

    get = post


# ---------------------------------------------------------------------- the module: two paths out
class Paths(unittest.TestCase):
    def test_the_webhook_path_is_the_request_it_always_was(self):
        fake = FakeDiscord()
        self.assertIs(discord.post(hook_settings(), "hello", session=fake), True)
        kind, url, headers, body, data, files = fake.calls[0]
        self.assertEqual((kind, url, headers, body), ("POST", HOOK, None, {"username": "depangram", "content": "hello"}))
        discord.post(hook_settings(), "card", file_name="d.md", file_text="report", session=fake)
        self.assertIn("file", fake.calls[1][5])
        self.assertEqual(json.loads(fake.calls[1][4]["payload_json"])["content"], "card")

    def test_the_bot_path_posts_as_the_bot_and_hands_back_the_id(self):
        fake = FakeDiscord()
        mid = discord.post(bot_settings(), "x" * 2500, session=fake)
        self.assertEqual(mid, "1000")
        kind, url, headers, body, _, _ = fake.calls[0]
        self.assertEqual(url, f"{discord.API}/channels/{CHANNEL}/messages")
        self.assertEqual(headers["Authorization"], f"Bot {TOKEN}")
        self.assertEqual(len(body["content"]), discord.LIMIT)          # the 2,000-character cut, as on the webhook
        self.assertEqual(body["allowed_mentions"], {"parse": []})
        self.assertNotIn(HOOK, url)

    def test_the_bot_path_attaches_the_report(self):
        fake = FakeDiscord()
        discord.post(bot_settings(), "card", file_name="day digest.md", file_text="# report", session=fake)
        files, data = fake.calls[0][5], fake.calls[0][4]
        self.assertEqual(files["files[0]"][0], "day digest.md")
        self.assertEqual(files["files[0]"][1], b"# report")
        self.assertEqual(json.loads(data["payload_json"])["content"], "card")

    def test_a_failed_bot_post_with_no_webhook_is_false_and_prints_no_token(self):
        fake = FakeDiscord()
        fake.fail = 500
        self.assertIs(discord.post(bot_settings(webhook=False), "hi", session=fake), False)

        class Boom:
            def post(self, *a, **k):
                raise ConnectionError(f"failed with header Bot {TOKEN}")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertIs(discord.post(bot_settings(webhook=False), "hi", session=Boom()), False)
        self.assertNotIn(TOKEN, out.getvalue())
        self.assertIn("ConnectionError", out.getvalue())


class Fallback(unittest.TestCase):
    """A failed bot post goes through the webhook, so alerts and the digest reach you on a bad token."""

    def setUp(self):
        for k in [k for k in discord._said if k.startswith("fallback")]:
            discord._said.discard(k)

    def test_a_401_falls_back_to_the_webhook_and_says_so_once_a_day(self):
        fake = FakeDiscord()
        fake.bot_fail = 401
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertIs(discord.post(bot_settings(), "alert one", session=fake), True)
            self.assertIs(discord.post(bot_settings(), "card", file_name="d.md", file_text="report", session=fake),
                          True)
        hooks = [c for c in fake.calls if c[1] == HOOK]
        self.assertEqual(len(hooks), 2)
        self.assertEqual(hooks[0][3], {"username": "depangram", "content": "alert one"})
        self.assertIn("file", hooks[1][5])                                    # the report still attached
        log = out.getvalue()
        self.assertEqual(log.count("posts go through the webhook today"), 1)
        self.assertIn("HTTP 401", log)
        self.assertNotIn(TOKEN, log)
        self.assertNotIn(HOOK, log)

    def test_a_403_or_a_connection_error_falls_back_too(self):
        fake = FakeDiscord()
        fake.bot_fail = 403
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(discord.post(bot_settings(), "hi", session=fake), True)

        class DownThenHook(FakeDiscord):
            def post(self, url, **k):
                if "/webhooks/" not in url:
                    raise ConnectionError(f"no route, header Bot {TOKEN}")
                return super().post(url, **k)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertIs(discord.post(bot_settings(), "hi", session=DownThenHook()), True)
        self.assertNotIn(TOKEN, out.getvalue())

    def test_a_good_bot_post_never_touches_the_webhook(self):
        fake = FakeDiscord()
        self.assertEqual(discord.post(bot_settings(), "hi", session=fake), "1000")
        self.assertFalse([c for c in fake.calls if c[1] == HOOK])

    def test_ask_on_the_fallback_returns_none_so_the_question_is_unasked(self):
        fake = FakeDiscord()
        fake.bot_fail = 401
        with contextlib.redirect_stdout(io.StringIO()):
            mid = discord.ask(bot_settings(), "cut s3w0 p1.2?", ["yes", "keep it going"], veto=2, session=fake)
        self.assertIsNone(mid)
        hook = [c for c in fake.calls if c[1] == HOOK]
        self.assertEqual(len(hook), 1)                                         # you still see it
        self.assertIn("1. yes (default)", hook[0][3]["content"])

    def test_the_read_functions_are_empty_for_a_none_id_and_ask_nothing(self):
        fake = FakeDiscord()
        self.assertEqual(discord.replies_since(bot_settings(), None, session=fake), [])
        self.assertEqual(discord.reactions(bot_settings(), None, session=fake), {})
        self.assertIsNone(discord.answer_to(bot_settings(), None, 2, veto=2, session=fake))
        self.assertEqual(fake.calls, [])

    def test_the_webhook_path_reads_nothing_and_says_so_once(self):
        discord._said.discard("off")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(discord, "requests", Refuse()):
            self.assertEqual(discord.replies_since(hook_settings(), "1"), [])
            self.assertEqual(discord.reactions(hook_settings(), "1"), {})
            self.assertIsNone(discord.ask(hook_settings(), "q?", ["a", "b"]))
            self.assertIsNone(discord.answer_to(hook_settings(), "1", 2))
        self.assertEqual(out.getvalue().count("the bot path is off"), 1)


# ---------------------------------------------------------------------- replies and reactions
class Reading(unittest.TestCase):
    def test_replies_come_back_oldest_first_and_page_on_the_highest_id(self):
        fake = FakeDiscord()
        q = fake.add(BOT_ID, "question", bot=True)
        for i in range(150):
            fake.add(AUTHOR_ID, f"m{i}")
        got = discord.replies_since(bot_settings(), q, session=fake)
        self.assertEqual(len(got), 150)
        self.assertEqual([int(m["id"]) for m in got], sorted(int(m["id"]) for m in got))
        self.assertEqual(got[0]["content"], "m0")
        gets = [c for c in fake.calls if c[0] == "GET"]
        self.assertEqual(len(gets), 2)
        self.assertEqual(gets[1][3]["after"], str(int(q) + 100))

    def test_a_reply_carries_what_it_answers(self):
        fake = FakeDiscord()
        q = fake.add(BOT_ID, "question", bot=True)
        fake.add(AUTHOR_ID, "2", reply_to=q)
        m = discord.replies_since(bot_settings(), q, session=fake)[0]
        self.assertEqual((m["reply_to"], m["author_id"], m["bot"]), (q, AUTHOR_ID, False))

    def test_reactions_name_the_users_behind_each_emoji(self):
        fake = FakeDiscord()
        q = fake.add(BOT_ID, "question", bot=True)
        fake.react(q, discord.THUMBS_UP, AUTHOR_ID)
        fake.react(q, discord.CROSS, BOT_ID, bot=True)
        got = discord.reactions(bot_settings(), q, session=fake)
        self.assertEqual(got[discord.THUMBS_UP], [{"id": AUTHOR_ID, "bot": False}])
        self.assertTrue(any("%" in c[1] for c in fake.calls if c[1].count("/reactions/")))   # URL-encoded

    def test_a_429_reads_as_no_answer(self):
        fake = FakeDiscord()
        q = fake.add(BOT_ID, "question", bot=True)
        fake.add(AUTHOR_ID, "2", reply_to=q)
        fake.fail = 429
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(discord.replies_since(bot_settings(), q, session=fake), [])
            self.assertIsNone(discord.answer_to(bot_settings(), q, 2, veto=2, session=fake))


# ---------------------------------------------------------------------- questions and answers
class Questions(unittest.TestCase):
    def setUp(self):
        self.fake, self.s = FakeDiscord(), bot_settings()

    def ask(self):
        return discord.ask(self.s, "cut s3w0 p2.1?", ["yes", "keep it going"], default=1, veto=2, session=self.fake)

    def test_ask_posts_numbered_options_and_the_default(self):
        mid = self.ask()
        text = self.fake.messages[mid]["content"]
        self.assertIn("1. yes (default)\n2. keep it going\n", text)
        self.assertIn("No answer means the default", text)

    def test_a_long_question_gives_way_to_the_options(self):
        mid = discord.ask(self.s, "x" * 3000, ["yes", "keep it going"], session=self.fake)
        text = self.fake.messages[mid]["content"]
        self.assertLessEqual(len(text), discord.LIMIT)
        self.assertIn("2. keep it going", text)

    def test_numbers_parse_bare_or_at_the_start(self):
        self.assertEqual(discord.parse_answer("2", 2), 2)
        self.assertEqual(discord.parse_answer(" 2. keep it, it's funny", 2), 2)
        self.assertEqual(discord.parse_answer("#1", 2), 1)
        self.assertIsNone(discord.parse_answer("12", 2))
        self.assertIsNone(discord.parse_answer("3", 2))
        self.assertIsNone(discord.parse_answer("keep 2", 2))

    def test_a_reply_to_the_question_answers_it(self):
        q = self.ask()
        self.fake.add(AUTHOR_ID, "2 keep it", reply_to=q)
        self.assertEqual(discord.answer_to(self.s, q, 2, veto=2, session=self.fake), 2)

    def test_the_latest_reply_wins(self):
        q = self.ask()
        self.fake.add(AUTHOR_ID, "2", reply_to=q)
        self.fake.add(AUTHOR_ID, "1 actually", reply_to=q)
        self.assertEqual(discord.answer_to(self.s, q, 2, session=self.fake), 1)

    def test_a_plain_number_counts_for_the_last_question_before_it(self):
        q1 = self.ask()
        q2 = self.ask()
        self.fake.add(AUTHOR_ID, "2")
        self.assertIsNone(discord.answer_to(self.s, q1, 2, until_id=q2, session=self.fake))
        self.assertEqual(discord.answer_to(self.s, q2, 2, session=self.fake), 2)

    def test_reactions_answer(self):
        q = self.ask()
        self.fake.react(q, discord.THUMBS_UP + chr(0x1F3FD), AUTHOR_ID)        # a skin tone is still a thumbs-up
        self.assertEqual(discord.answer_to(self.s, q, 2, veto=2, session=self.fake), 1)
        q2 = self.ask()
        self.fake.react(q2, discord.CROSS, AUTHOR_ID)
        self.assertEqual(discord.answer_to(self.s, q2, 2, veto=2, session=self.fake), 2)
        q3 = self.ask()
        self.fake.react(q3, "2" + discord.VS16 + discord.KEYCAP, AUTHOR_ID)
        self.assertEqual(discord.answer_to(self.s, q3, 2, session=self.fake), 2)

    def test_two_different_reactions_are_no_answer(self):
        q = self.ask()
        self.fake.react(q, discord.THUMBS_UP, AUTHOR_ID)
        self.fake.react(q, discord.CROSS, AUTHOR_ID)
        self.assertIsNone(discord.answer_to(self.s, q, 2, veto=2, session=self.fake))

    def test_only_the_author_counts(self):
        q = self.ask()
        self.fake.add(BOT_ID, "2", reply_to=q, bot=True)                         # a bot: never
        self.assertIsNone(discord.answer_to(self.s, q, 2, session=self.fake))
        self.fake.add(STRANGER_ID, "2", reply_to=q)
        self.assertEqual(discord.answer_to(self.s, q, 2, session=self.fake), 2)  # no author id: anybody not a bot
        s = bot_settings(author=AUTHOR_ID)
        self.assertIsNone(discord.answer_to(s, q, 2, session=self.fake))         # with it: only the author
        self.fake.react(q, discord.CROSS, STRANGER_ID)
        self.assertIsNone(discord.answer_to(s, q, 2, veto=2, session=self.fake))
        self.fake.add(AUTHOR_ID, "2", reply_to=q)
        self.assertEqual(discord.answer_to(s, q, 2, session=self.fake), 2)

    def test_no_answer_is_none(self):
        q = self.ask()
        self.fake.add(AUTHOR_ID, "nice one")
        self.assertIsNone(discord.answer_to(self.s, q, 2, veto=2, session=self.fake))


class AskedFile(unittest.TestCase):
    def test_questions_survive_a_restart_and_close_on_their_default_after_two_days(self):
        d = Path(tempfile.mkdtemp())
        a = answers.Asked(d / "q.json")
        a.add({"id": "1000", "kind": "cut", "ts": 0.0, "options": ["yes", "no"]})
        a.day_posts["2026-11-11"] = "999"
        a.save()
        b = answers.Asked(d / "q.json")
        self.assertEqual([q["id"] for q in b.open()], ["1000"])
        self.assertEqual(b.day_posts, {"2026-11-11": "999"})
        b.prune(3 * 86400)
        self.assertEqual(b.open(), [])

    def test_a_broken_file_means_no_open_questions(self):
        d = Path(tempfile.mkdtemp())
        (d / "q.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(answers.Asked(d / "q.json").open(), [])

    def test_stop_and_rest_count_only_after_the_day_post(self):
        fake = FakeDiscord()
        fake.add(AUTHOR_ID, "stop")                                  # yesterday's
        day = fake.add(BOT_ID, "run-day started", bot=True)
        fake.add(AUTHOR_ID, "rest S3W0 please")
        reps = discord.replies_since(bot_settings(), "0", session=fake)
        self.assertEqual(answers.commands(bot_settings(), reps, day), (False, ["s3w0"]))
        fake.add(AUTHOR_ID, "Stop.")
        reps = discord.replies_since(bot_settings(), "0", session=fake)
        self.assertEqual(answers.commands(bot_settings(), reps, day)[0], True)
        fake.add(BOT_ID, "stop", bot=True)
        self.assertEqual(answers.commands(bot_settings(), reps[:1], day), (False, []))


# ---------------------------------------------------------------------- the loop's hooks
A, B, C = "Keep this line.", "this one goes on a bit, honestly.", "The last paragraph."
HEAD = "Heading"
R0 = [[[A], [B]], [[C]]]
R_CUT = [[[A], []], [[C]]]
R_CUT_LATER = [[["A regrown line."], []], [["The last paragraph, regrown."]]]


def make_ws(wid="s3w0"):
    ws = WindowState({"wid": wid, "text": "", "section": 3, "index": 0})
    ws.runs = [[{"sents": r[:], "claude": True} for r in p] for p in R0]
    ws.original = [[r[:] for r in p] for p in R0]
    ws.current = [[r[:] for r in p] for p in R_CUT]
    ws.heading, ws.must = HEAD, {(pi, ri): {"hedges": [], "repeats": []} for pi, p in enumerate(R0) for ri in range(len(p))}
    ws.scored_runs, ws.forced, ws.ready = None, set(), True
    ws.last_text = window_text_of(R_CUT, HEAD)
    return ws


def make_bank(d, later=False):
    b = Bank(Path(d) / "bank.jsonl")
    b.add({"kind": "start", "window": "s3w0", "time": "2026-11-11 02:10", "ai": 0.8, "segs": [], "text": "orig"})
    b.add({"kind": "round", "window": "s3w0", "time": "2026-11-11 03:00", "pass": 1, "ai": 0.5,
           "text": window_text_of(R0, HEAD), "runs": R0})
    b.add({"kind": "cut", "window": "s3w0", "time": "2026-11-11 03:30", "pass": 1, "para": 0, "run": 1,
           "cut_text": B, "ai": 0.3, "text": window_text_of(R_CUT, HEAD), "runs": R_CUT})
    b.add({"kind": "pass_end", "window": "s3w0", "time": "2026-11-11 03:30", "pass": 1, "day": "2026-11-11"})
    if later:
        b.add({"kind": "round", "window": "s3w0", "time": "2026-11-11 05:00", "pass": 2, "ai": 0.2,
               "text": window_text_of(R_CUT_LATER, HEAD), "runs": R_CUT_LATER})
    return b


def make_loop(d, settings, fake=None, later=False, rehearsal=False):
    lp = Loop.__new__(Loop)
    lp.bank, lp.noise, lp.repeat_must, lp.timeline = make_bank(d, later), 0.05, False, []
    lp.day, lp.log = "2026-11-11", lambda m: lp.timeline.append(m)
    lp.clock = types.SimpleNamespace(stamp=lambda *a: "2026-11-11 09:25", now=lambda: 1000.0)
    lp.settings, lp.rehearsal = settings, rehearsal
    lp.args = types.SimpleNamespace(smoke=False, now=False)
    lp.notifier = types.SimpleNamespace(live=True, send=lambda *a, **k: None)
    lp.job = types.SimpleNamespace(name="garden-notes")
    lp.windows, lp.cuts, lp.stop_reason, lp.stop_file = [make_ws()], [], None, None
    lp.cuts.append({"window": "s3w0", "para": 0, "run": 1, "text": B, "why": "no fact", "time": "2026-11-11 03:30"})
    return lp


class LoopHooks(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.fake = FakeDiscord()
        self.patch = mock.patch.object(discord, "requests", self.fake)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def bot_loop(self, **k):
        lp = make_loop(self.d, bot_settings(runs_dir=self.d / "runs"), **k)
        lp.day_post_id = self.fake.add(BOT_ID, "run-day started", bot=True)
        lp.note_day_post()
        return lp

    def test_with_no_bot_token_every_hook_is_a_no_op(self):
        lp = make_loop(self.d, hook_settings(runs_dir=self.d / "runs"))
        lp.day_post_id = "999"
        before = len(lp.bank.entries)
        with mock.patch.object(discord, "requests", Refuse()):
            self.assertFalse(lp.asks_on())
            lp.note_day_post()
            lp.ask_cut(lp.windows[0], lp.cuts[0])
            lp.ask_open_cuts()
            lp.read_answers(force=True)
        self.assertEqual(len(lp.bank.entries), before)
        self.assertIsNone(lp.asked)
        self.assertFalse((self.d / "runs").exists())
        self.assertEqual((lp.stop_reason, lp.discord_stop, lp.timeline), (None, False, []))

    def test_a_rehearsal_or_a_smoke_test_never_asks(self):
        lp = make_loop(self.d, bot_settings(runs_dir=self.d / "runs"), rehearsal=True)
        self.assertFalse(lp.asks_on())
        lp = make_loop(self.d, bot_settings(runs_dir=self.d / "runs"))
        lp.args.smoke = True
        self.assertFalse(lp.asks_on())
        lp.args.smoke, lp.notifier.live = False, False
        self.assertFalse(lp.asks_on())
        lp.notifier.live, lp.settings = True, bot_settings(runs_dir=self.d / "runs", questions=False)
        self.assertFalse(lp.asks_on())

    def test_a_cut_asks_and_a_two_puts_the_line_back(self):
        lp = self.bot_loop()
        ws = lp.windows[0]
        ws.rested_today, ws.state = True, "done"            # the cut left it at target, resting
        self.assertEqual(lp.best(ws)["kind"], "cut")
        lp.ask_cut(ws, lp.cuts[0])
        q = lp.cuts[0]["question"]
        self.assertIn("cut s3w0 p1.2?", self.fake.messages[q]["content"])
        self.assertEqual(lp.asked.open()[0]["veto"], 2)
        self.fake.add(AUTHOR_ID, "2", reply_to=q)
        lp.read_answers(force=True)
        self.assertTrue(lp.cuts[0]["vetoed"])
        self.assertEqual(lp.bank.of("s3w0", "veto")[0]["cut_text"], B)
        self.assertEqual(ws.vetoed, {(0, 1)})
        self.assertEqual(ws.current[0][1], [B])              # back in, from the best banked text that keeps it
        self.assertEqual(lp.best(ws)["kind"], "round")
        self.assertEqual(lp.ship_entry(ws)["runs"][0][1], [B])
        self.assertFalse(ws.rested_today)                    # the window carries on
        self.assertEqual(lp.asked.open(), [])

    def test_a_one_or_a_thumbs_up_keeps_the_cut(self):
        lp = self.bot_loop()
        ws = lp.windows[0]
        lp.ask_cut(ws, lp.cuts[0])
        self.fake.react(lp.cuts[0]["question"], discord.THUMBS_UP, AUTHOR_ID)
        lp.read_answers(force=True)
        self.assertFalse(lp.cuts[0].get("vetoed"))
        self.assertEqual(lp.bank.of("s3w0", "veto"), [])
        self.assertEqual(ws.current[0][1], [])
        self.assertEqual(lp.asked.open(), [])

    def test_no_answer_leaves_everything_as_it_was(self):
        lp = self.bot_loop()
        ws = lp.windows[0]
        lp.ask_cut(ws, lp.cuts[0])
        before = len(lp.bank.entries)
        lp.read_answers(force=True)
        self.assertEqual(len(lp.bank.entries), before)
        self.assertEqual(ws.current[0][1], [])
        self.assertEqual(len(lp.asked.open()), 1)

    def test_reads_wait_for_the_poll_interval(self):
        lp = self.bot_loop()
        lp.read_answers(force=True)
        n = self.fake.requests_count()
        lp.read_answers()                                     # the clock hasn't moved: no read
        self.assertEqual(self.fake.requests_count(), n)

    def test_a_late_veto_after_work_on_the_cut_text_still_never_ships_the_cut(self):
        lp = self.bot_loop(later=True)
        ws = lp.windows[0]
        self.assertEqual(lp.best(ws)["ai"], 0.2)
        lp.ask_cut(ws, lp.cuts[0])
        self.fake.add(AUTHOR_ID, "2")                         # a plain 2 after the question counts for it
        lp.read_answers(force=True)
        self.assertEqual(lp.best(ws)["ai"], 0.5)              # every entry holding the cut is turned down
        self.assertEqual(lp.ship_entry(ws)["runs"][0][1], [B])

    def test_a_veto_survives_a_restart_and_the_drop_rule_leaves_the_run_alone(self):
        lp = self.bot_loop()
        lp.ask_cut(lp.windows[0], lp.cuts[0])
        self.fake.add(AUTHOR_ID, "2", reply_to=lp.cuts[0]["question"])
        lp.read_answers(force=True)
        # a new process on the same bank: the window is readied from scratch
        lp2 = make_loop(self.d, bot_settings(runs_dir=self.d / "runs"))
        lp2.bank = Bank(lp.bank.path)
        ws = make_ws()
        ws.ready, ws.current = False, [[r[:] for r in p] for p in R_CUT]
        lp2.windows = [ws]
        lp2.load_calls(ws)
        self.assertEqual(ws.vetoed, {(0, 1)})
        ws.current = [[r[:] for r in p] for p in R_CUT]
        lp2.demote(ws)
        self.assertEqual(ws.current[0][1], [B])
        self.assertTrue(any("you put this cut back on Discord" in t for t in lp2.timeline))
        # the drop rule: the run qualifies on every count code checks, and Claude is never asked
        ws.who = {s: "claude" for p in R0 for r in p for s in r}
        ws.attempts, ws.completed, ws.pass_no, ws.notes = {(0, 1): 6}, 1, 2, [[], []]   # only the vetoed run was tried
        lp2.blocked = lambda: False
        lp2.must_stop = lambda: False
        lp2.claude = mock.Mock(side_effect=AssertionError("the drop rule asked Claude about a vetoed run"))
        lp2.drop_rule(ws)
        self.assertIn((0, 1, "you said keep it going on Discord"), ws.couldnt_cut)
        lp2.claude.assert_not_called()

    def test_rest_rests_the_window_for_today_and_a_restart_keeps_it(self):
        lp = self.bot_loop()
        self.fake.add(AUTHOR_ID, "rest s3w0")
        lp.read_answers(force=True)
        ws = lp.windows[0]
        self.assertTrue(ws.rested_today and ws.rest_call)
        self.assertNotIn(ws, lp.order())
        fresh = make_ws()
        lp.load_calls(fresh)
        self.assertTrue(fresh.rested_today)
        lp.day = "2026-11-12"                                 # tomorrow: the rest is over
        fresh = make_ws()
        lp.load_calls(fresh)
        self.assertFalse(fresh.rested_today)

    def test_stop_stops_the_day_like_the_stop_file(self):
        lp = self.bot_loop()
        lp.schedule = types.SimpleNamespace(bounds=lambda now: {"stop": 1e12})
        self.assertFalse(lp.hard_stop())
        self.fake.add(AUTHOR_ID, "stop")
        lp.read_answers(force=True)
        self.assertTrue(lp.discord_stop)
        self.assertTrue(lp.stop_reason.startswith("the STOP"))   # run_with_restarts skips the wind-down wait for it
        self.assertTrue(lp.must_stop())
        self.assertTrue(lp.hard_stop())

    def test_a_stop_before_today_s_start_post_doesnt_count(self):
        self.fake.add(AUTHOR_ID, "stop")
        lp = self.bot_loop()
        lp.read_answers(force=True)
        self.assertFalse(lp.discord_stop)
        self.assertIsNone(lp.stop_reason)

    def test_an_error_reading_costs_only_the_answer(self):
        lp = self.bot_loop()
        lp.ask_cut(lp.windows[0], lp.cuts[0])
        with mock.patch.object(answers, "read", side_effect=RuntimeError(f"Bot {TOKEN}")):
            lp.read_answers(force=True)
        self.assertTrue(any("the defaults stand" in t for t in lp.timeline))
        self.assertFalse(any(TOKEN in t for t in lp.timeline))

    def test_a_cut_question_that_went_out_through_the_webhook_is_unasked(self):
        lp = self.bot_loop()
        self.fake.bot_fail = 401
        for k in [k for k in discord._said if k.startswith("fallback")]:
            discord._said.discard(k)
        with contextlib.redirect_stdout(io.StringIO()):
            lp.ask_cut(lp.windows[0], lp.cuts[0])
        self.assertIsNone(lp.cuts[0].get("question"))
        self.assertEqual(lp.asked.open(), [])
        self.assertTrue([c for c in self.fake.calls if c[1] == HOOK])
        before = len(lp.bank.entries)
        lp.read_answers(force=True)
        self.assertEqual(len(lp.bank.entries), before)                  # the cut stands: the default
        self.assertEqual(lp.windows[0].current[0][1], [])

    def test_the_digest_asks_for_a_cut_that_has_no_question(self):
        lp = self.bot_loop()
        lp.ask_open_cuts()
        self.assertTrue(lp.cuts[0].get("question"))
        n = len(self.fake.messages)
        lp.ask_open_cuts()                                    # asked once only
        self.assertEqual(len(self.fake.messages), n)

    def test_a_late_answer_read_at_the_day_s_start_reaches_the_window_through_resume(self):
        # the usual late answer: yesterday's cut, answered overnight, read before any window is readied
        lp = self.bot_loop()
        lp.ask_cut(lp.windows[0], lp.cuts[0])
        self.fake.add(AUTHOR_ID, "2", reply_to=lp.cuts[0]["question"])
        ws = lp.windows[0]
        ws.ready = False
        lp.read_answers(force=True)
        self.assertEqual(ws.vetoed, set())                    # not ready: the bank holds the veto for prepare
        self.assertEqual(len(lp.bank.of("s3w0", "veto")), 1)
        lp.stuck_days = 3
        ws.current = None
        lp.load_calls(ws)
        lp.resume(ws, lp.bank.start("s3w0"))
        self.assertEqual(ws.current[0][1], [B])
        self.assertEqual(ws.ai, 0.5)                          # the banked text that keeps the line, with its score

    def test_a_stop_lands_inside_a_pass_without_touching_any_text(self):
        lp = self.bot_loop()
        lp.ask_cut(lp.windows[0], lp.cuts[0])
        self.fake.add(AUTHOR_ID, "2", reply_to=lp.cuts[0]["question"])
        self.fake.add(AUTHOR_ID, "stop")
        self.assertTrue(lp.must_stop())                       # what do_round asks before each run
        self.assertEqual(lp.stop_reason, "the STOP reply on Discord")
        self.assertEqual(lp.bank.of("s3w0", "veto"), [])      # the veto waits for the next full read
        self.assertEqual(lp.windows[0].current[0][1], [])

    def test_the_stop_read_waits_for_the_poll_interval(self):
        lp = self.bot_loop()
        lp.schedule = types.SimpleNamespace(bounds=lambda now: {"stop": 1e12})
        self.assertFalse(lp.hard_stop())
        self.fake.add(AUTHOR_ID, "stop")
        self.assertFalse(lp.hard_stop())                      # read a moment ago: not again until poll_seconds pass
        lp.next_stop_poll = 0.0
        self.assertTrue(lp.hard_stop())

    def test_restore_section_s_ship_map_keeps_a_vetoed_line(self):
        from engine.restore_section import ship_map
        lp = self.bot_loop()
        lp.ask_cut(lp.windows[0], lp.cuts[0])
        self.fake.add(AUTHOR_ID, "2", reply_to=lp.cuts[0]["question"])
        lp.read_answers(force=True)
        lay = {"s3w0": {"section": 3, "runs": make_ws().runs, "original": [[r[:] for r in p] for p in R0],
                        "heading": HEAD}}
        _, shipped = ship_map(Bank(lp.bank.path), lay)["s3w0"]
        self.assertIn(B, shipped[0])


class DigestLines(unittest.TestCase):
    def test_the_card_and_the_report_say_a_cut_went_back(self):
        lp = types.SimpleNamespace(cuts=[{"window": "s3w0", "para": 0, "run": 1, "text": B, "why": "no fact",
                                          "time": "t", "vetoed": True, "question": "1000"}],
                                   asked=None)
        self.assertEqual(discord.open_questions_line(lp), "")
        lp.asked = types.SimpleNamespace(open=lambda: [{"id": "1001"}])
        self.assertIn("Open questions on Discord: 1", discord.open_questions_line(lp))
        ws = make_ws()
        ws.rest_call, ws.state = True, "working"
        rows = discord.window_rows(types.SimpleNamespace(windows=[ws], bank=make_bank(tempfile.mkdtemp()),
                                                   best=lambda w: None))
        self.assertIn("(rested on your word)", rows[0]["state"])


class NotifierIds(unittest.TestCase):
    def test_send_hands_back_the_bot_path_s_id_and_none_on_the_webhook(self):
        log = Path(tempfile.mkdtemp()) / "d.log"
        n = Notifier(Settings(env={}, config={}), CompressedClock(0.0), log)
        with mock.patch.object(discord, "post", return_value="1234"):
            self.assertEqual(n.send("hi"), "1234")
        with mock.patch.object(discord, "post", return_value=True):
            self.assertIsNone(n.send("hi"))
        n.live = False
        self.assertIsNone(n.send("hi"))


class Notifiers(unittest.TestCase):
    """NOTIFIER: discord (the default), file (posts written to a folder) or none (dropped). With file or none the bot
    path is off whatever tokens are set, so the loop asks and reads nothing, and nothing reaches the network."""

    def settings(self, kind, d=None, **env):
        env = {"DISCORD_BOT_TOKEN": TOKEN, "DISCORD_CHANNEL_ID": CHANNEL, "DISCORD_WEBHOOK_URL": HOOK, **env}
        if kind:
            env["NOTIFIER"] = kind
        if d:
            env["NOTIFY_DIR"] = str(d)
        return Settings(env=env, config={})

    def test_discord_is_the_default_and_the_config_can_pick_another(self):
        self.assertEqual(discord.notifier(self.settings(None)), "discord")
        self.assertTrue(discord.bot_on(self.settings(None)))
        s = Settings(env={}, config={"notify": {"notifier": "file"}})
        self.assertEqual(discord.notifier(s), "file")
        s = Settings(env={"NOTIFIER": "none"}, config={"notify": {"notifier": "file"}})
        self.assertEqual(discord.notifier(s), "none")                                    # the environment wins
        with self.assertRaises(ValueError):
            discord.notifier(self.settings("slack"))

    def test_file_writes_the_post_and_the_report_and_never_touches_discord(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(discord, "requests", Refuse()):
            s = self.settings("file", d)
            self.assertFalse(discord.bot_on(s))                                           # flag: tokens set, still off
            self.assertIs(discord.post(s, "the card", file_name="day digest.md", file_text="# the report\n"), True)
            self.assertIs(discord.post(s, "an alert"), True)
            log = (Path(d) / "posts.md").read_text(encoding="utf-8")
            self.assertIn("+ day digest.md\nthe card\n", log)
            self.assertIn("an alert", log)
            self.assertEqual((Path(d) / "day digest.md").read_text(encoding="utf-8"), "# the report\n")
            self.assertIsNone(discord.ask(s, "q?", ["a", "b"]))
            self.assertEqual(discord.replies_since(s, "1"), [])
            self.assertIn("to " + d, discord.where(s))

    def test_none_drops_the_post_and_reports_it_taken(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(discord, "requests", Refuse()):
            s = self.settings("none", d)
            self.assertIs(discord.post(s, "dropped", file_name="x.md", file_text="x"), True)
            self.assertEqual(list(Path(d).iterdir()), [])                                # pass: nothing written
            self.assertFalse(discord.bot_on(s))

    def test_the_loop_s_notifier_goes_through_the_choice(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(discord, "requests", Refuse()):
            s = self.settings("file", d)
            n = Notifier(s, CompressedClock(0), Path(d) / "discord.log", live=True)
            self.assertIsNone(n.send("hourly"))
            self.assertIn("hourly", (Path(d) / "posts.md").read_text(encoding="utf-8"))

    def test_the_digest_builders_are_still_reachable_here(self):
        from engine import digest
        for name in ("digest_report", "digest_card", "couldnt_crack_report", "hourly_text", "efficiency_lines",
                     "model_lines", "spend_lines"):
            self.assertIs(getattr(discord, name), getattr(digest, name))

    def test_the_check_says_discord_is_off_under_another_notifier(self):
        from engine import discord_check
        out = []
        with mock.patch.object(discord, "requests", Refuse()):
            self.assertEqual(discord_check.main([], settings=self.settings("none"), out=out.append), 0)
        self.assertIn("the notifier is none", out[0])


class Check(unittest.TestCase):
    def test_with_no_token_it_says_so_posts_nothing_and_exits_0(self):
        from engine import discord_check
        out = []
        with mock.patch.object(discord, "requests", Refuse()):
            self.assertEqual(discord_check.main([], settings=hook_settings(), out=out.append), 0)
        self.assertIn("the bot path is off", out[0])
        self.assertIn("webhook path is on", out[1])
        self.assertFalse(any(HOOK in o for o in out))

    def test_with_a_token_it_posts_one_prefixed_message_and_reads_it_back(self):
        from engine import discord_check
        fake, out = FakeDiscord(), []
        with mock.patch.object(discord, "requests", fake):
            self.assertEqual(discord_check.main([], settings=bot_settings(), out=out.append), 0)
            first = fake.messages["1000"]["content"]
            self.assertTrue(first.startswith("[discord check]"))
            self.assertEqual(len(fake.messages), 1)                 # one message, no more
            self.assertTrue(any("read back: yes" in o for o in out))
            fake.add(AUTHOR_ID, "2", reply_to="1000")
            out.clear()
            discord_check.main(["--read", "1000"], settings=bot_settings(), out=out.append)
        self.assertTrue(any("option 2" in o for o in out))
        self.assertFalse(any(TOKEN in o for o in out))


if __name__ == "__main__":
    unittest.main()
