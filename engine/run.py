"""The run loop, M1: one job, every window, on the schedule, under every cap (plan 12).

What Task Scheduler will start at 2:00 (M5). With no director yet (M4), the code makes the default picks:
windows in order of their score (worst first), 6 rounds a pass, the stuck rule and the drop rule. Each round
runs experiment 6's method through engine/draws.py (draws, the filter stack with 5.3's Phase 2 fix
list, Open Pangram's ranking, the judge, the head-to-head, the fact check), then Pangram scores the window.

    python -m engine.run                      # the real run-day (waits for 2:00, stops at 16:00)
    python -m engine.run --rehearse           # the whole day on a compressed clock, every paid call faked
    python -m engine.run --rehearse --scenario full-week
    python -m engine.run --smoke --window s1w0   # one window, one round, real models, capped small

Only this script calls Pangram, and no Pangram score, label or segment ever goes into a Claude prompt. Sonnet
writes notes and verdicts only; every word that ships is the local model's (or yours, or the original's).
Every cap lives in config.toml and is checked in engine/budget.py before each paid call.
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
import traceback
from pathlib import Path

import requests

from engine import db
from engine import discord
from engine.budget import WINDOW_SHARE, Budget, ClaudeCapReached, LoopPangram, PangramCapReached, kind_of
from engine.checks import (carries_fact, connective_problem, extra_checks_for, invented_sighting, is_template,
                           must_keeps, names_missing, names_required, pattern_quotes, punctuation_problem, reddit_tell,
                           repeat_ok, restated_pairs, section_counts, section_filler_problem, splice_quotes)
from engine import checks as _checks
from engine.checks import measured_swear_rates, section_swear_problem, swear_counts
from engine.claude_call import ClaudeCallError, run as claude_call_run
from engine.job import (Bank, Job, NotesError, PoolQueue, align, best_seed, flat, merge_pools, protected_fallback,
                        job_name, split_author_runs, style_of, window_text_of, word_in)
from engine.pangram import CapReached as PangramClientCap, PangramError, summarize
from engine.schedule import Clock, CompressedClock, Schedule, parse_clock
from engine.section_best import runs_for, section_best, section_standing
from engine.settings import SANDBOX, Settings
from engine.util import describe, redact  # noqa: F401  (redact stays importable from here)
from engine.wrapup import as_posted, flagged_segments, norm, restore_examples, write_back

FALLBACK_USD = {"judge": 0.3, "h2h": 0.2, "facts": 0.2, "notes": 0.6, "drop": 0.3, "probe": 0.05, "protect": 0.3}
SHIP_KINDS = ("ship", "fix", "flavor", "restore")       # bank entries that ship like a whole-text pick (ship_entry)
NAMES_NOT_FACTS = {"AI", "LLM", "LLMs", "OK", "TL", "DR"}


def start_time(schedule, now, ignore_schedule=False):
    """When a start at `now` should begin work (M1 review, finding 1): straight away inside the night or day
    phase; otherwise the next 2:00, which is tomorrow's once today's last attempt has passed."""
    if ignore_schedule or schedule.phase(now) in ("night", "day"):
        return now
    return schedule.next_start(now)


def now_stop_time(schedule, now):
    """--now still stops (M1 review, finding 7): at today's stop time, or tomorrow's when started after it."""
    stop = schedule.bounds(now)["stop"]
    return stop if now < stop else schedule.bounds(now + 86400)["stop"]


def protected_missing(candidate, words):
    """The protected words of a line of yours (plan 5.1) that a rewrite lost, in order."""
    return [w for w in words if not word_in(w, candidate)]


def failing_runs(must, current, original, repeats=True):
    """The replaced runs that break a must-keep (a hedge or a repeat lost, a sighting added, a protected word of
    yours or a number lost, a name lost) or slip on punctuation, as (pi, ri), plus any replaced run in a pair of
    neighbouring sentences that says the same thing twice when the original paragraph doesn't (M2's checks).
    `repeats=False` (repeat_must_keep off) leaves a lost repeat to the judge."""
    out = []
    for (pi, ri), mk in must.items():
        cur = " ".join(current[pi][ri])
        orig = " ".join(original[pi][ri])
        if not cur or cur == orig:
            continue
        missing = [h for h in mk["hedges"] if not re.search(r"\b{}\b".format(re.escape(h)), cur, re.I)]
        if missing or (repeats and not repeat_ok(cur, mk["repeats"])) or invented_sighting(cur, orig) \
                or protected_missing(cur, mk.get("protected") or []) \
                or not set(mk.get("numbers") or []) <= set(re.findall(r"\d+(?:[.,/]\d+)*", cur)) \
                or punctuation_problem(cur, orig) or names_missing(cur, mk.get("names") or []):
            out.append((pi, ri))
    for pi in sorted({pi for pi, _ in must}):
        if pi >= len(current):
            continue
        owner = [(ri, s) for ri, r in enumerate(current[pi]) for s in r]
        if not owner or restated_pairs([s for r in original[pi] for s in r]):
            continue
        for i, j, _ in restated_pairs([s for _, s in owner]):
            for ri in sorted({owner[i][0], owner[j][0]}):
                if (pi, ri) in must and current[pi][ri] != original[pi][ri] and (pi, ri) not in out:
                    out.append((pi, ri))
    return out

DROP_TEMPLATE = """A paragraph from <<DESCRIPTION>>, and the same paragraph with one stretch cut out.

Text right before the paragraph:
<<BEFORE>>

The paragraph as it stands:
<<PARA>>

The paragraph with the stretch cut:
<<CUT>>

The stretch that was cut:
<<RUN>>

The notes the paragraph has to cover:
<<NOTES>>

Does the cut paragraph (1) still cover every point in the notes, apart from a joke, flourish or transition the cut stretch carried, with no fact, definition, number or step lost, (2) read naturally, with nothing left pointing at the missing stretch, (3) sound like a person talking?

Reply with only JSON: {"pass": true or false, "lost": ["any fact, definition, number or step the cut loses"], "why": "short reason"}"""
DROP_PROMPT = describe(DROP_TEMPLATE)


# ---------------------------------------------------------------------- the rules, as plain functions
def stuck_rule(changed, after_round, streak):
    """Plan 5.6: rounds 1 to `after_round` always run; after that, `streak` rounds in a row with no change in
    the window's score or text end the pass. `changed` is one bool per finished round of the pass."""
    n = len(changed)
    return n >= after_round + streak and not any(changed[-streak:])


def score_is_fresh(ai, scored_runs, current):
    """The window's standing score belongs to the text it stands at (M1 review, finding 14). A round whose scoring
    was capped leaves the new text unscored with the old score still on the window; a later round that changes
    nothing must score that text, not bank it with the old text's score."""
    return ai is not None and scored_runs is not None and scored_runs == current


def resume_base(bank, wid, op):
    """Where an open pass picks up (M1 review, finding 11): its last round, or a drop-rule cut banked after that
    round. A crash in end_pass after a cut and before the pass_end entry would otherwise lose the cut and pay
    Sonnet again for the same verdict."""
    last = op[1][-1]
    cuts = [e for e in bank.of(wid, "cut") if e.get("pass") == op[0] and e.get("runs") is not None
            and bank.position(e) > bank.position(last)]
    return cuts[-1] if cuts else last


def drop_verdicts(bank, wid, pass_no):
    """The drop rule's verdicts already banked in this pass (finding 11): the paragraphs it cut, and the runs the
    judge wouldn't let go, with its reason. A resumed pass reads them instead of asking again."""
    ents = [e for e in bank.of(wid) if e.get("pass") == pass_no and e.get("kind") in ("cut", "drop_no")]
    return ({e["para"] for e in ents if e["kind"] == "cut"},
            {(e["para"], e["run"]): e.get("why", "") for e in ents if e["kind"] == "drop_no"})


def drop_eligible(runs, current, original, attempts, rounds_done, verdict_of, names_fn):
    """Plan 5.4's drop rule, the parts code decides. A run may be cut when it's Claude-written (every sentence's
    verdict is "claude"), still stands as Claude wrote it after the pass (every route failed), was tried in
    every round of the pass, and carries no fact (no number, quote, citation, definition, study or name), and
    the paragraph keeps something once it's gone. Returns (cuttable, couldnt_cut) as lists of (pi, ri, why)."""
    cut, keep = [], []
    for pi, p in enumerate(runs):
        for ri, r in enumerate(p):
            if not r["claude"] or current[pi][ri] != original[pi][ri]:
                continue
            if attempts.get((pi, ri), 0) < max(rounds_done, 1):
                continue
            if any(verdict_of(s) != "claude" for s in original[pi][ri]):
                continue
            text = " ".join(original[pi][ri])
            if carries_fact(text, names_fn(text) - NAMES_NOT_FACTS):
                keep.append((pi, ri, "carries a fact"))
            elif not any(flat([current[pi][x]]) for x in range(len(p)) if x != ri):
                keep.append((pi, ri, "it's the whole paragraph"))
            else:
                cut.append((pi, ri, "no fact, every route failed"))
    return cut, keep


def flag_list(segs):
    """Flagged segment texts, plus each pair of neighbors joined: Pangram splits after "et al.", so a sentence can
    span two flagged segments and the `s[:40] in segment` test would miss it."""
    segs = [(t.strip(), bool(f)) for t, f in segs]
    return [t for t, f in segs if f] + [a + " " + b for (a, fa), (b, fb) in zip(segs, segs[1:]) if fa and fb]


def segs_of(resp):
    return [[w.get("text", "").strip(), (w.get("ai_assistance_score") or 0) > 0.5] for w in resp.get("windows", [])]


# ---------------------------------------------------------------------- one window's state
class WindowState:
    def __init__(self, w):
        self.w, self.wid, self.text = w, w["wid"], w["text"]
        self.ready = False
        self.state = "new"
        self.pass_open = False
        self.pass_no = 0
        self.completed = 0
        self.changed = []
        self.settle_next = False
        self.attempts = {}
        self.forced = set()
        self.history = {}            # (pi, ri) -> list of result strings, every pass
        self.tiny_misses = {}        # (pi, ri) -> tries today with no accepted change (the tiny-line rule)
        self.tiny_skipped = {}       # (pi, ri) -> the run's text when the rule skipped it, for the digest
        self.g = 0
        self.ai = None
        self.segs = []
        self.flagged = []
        self.scored_runs = None
        self.last_text = None
        self.last_was_last = False
        self.passes_today = 0
        self.rounds_today = 0
        self.rested_today = False
        self.retired = False
        self.start_of_day = None
        self.couldnt_cut = []
        self.vetoed = set()          # (pi, ri) of cuts you put back on Discord (plan 9.1): never cut again, never shipped cut
        self.rest_call = False       # you rested this window for the run-day on Discord (plan 9.1)

    def claude_text_left(self):
        return any(r["claude"] and self.current[pi][ri] == self.original[pi][ri]
                   for pi, p in enumerate(self.runs) for ri, r in enumerate(p))

    def n_flagged(self):
        return sum(1 for _, f in self.segs if f)


# ---------------------------------------------------------------------- the loop
class Loop:
    # defaults for a Loop built without __init__ (the tests): the 2026-11-10 calls read from config in __init__
    target, protected_cap, repeat_must, rest_at_target = None, 3, True, True
    noise, floor = 0.05, 0.0        # the section-best rule's noise floor; the reserve left after its release
    # two-way Discord (plan 9.1, issue #8): the open questions file, the next read, a stop said on Discord, and the
    # run-day's start post. All of it stays None or off with no bot token, so the loop runs as it always has
    asked, next_poll, next_stop_poll, discord_stop, day_post_id = None, 0.0, 0.0, False, None

    def __init__(self, *, settings, conn, clock, schedule, budget, notifier, job, pg, claude_raw, srv_factory,
                 op_factory, args, rehearsal=False, stop_file=None):
        self.settings, self.conn, self.clock, self.schedule, self.budget = settings, conn, clock, schedule, budget
        self.notifier, self.job, self.pg, self.claude_raw = notifier, job, pg, claude_raw
        self.srv_factory, self.op_factory = srv_factory, op_factory
        self.args, self.rehearsal = args, rehearsal
        self.stop_file = Path(stop_file) if stop_file else None
        lc = settings.config["loop"]
        self.model = lc["model"]
        # the writer arms (6.9, Phase 3; 2026-11-09): "notes" is the adapter writing from the notes alone, "fewshot"
        # the plain base with the worked examples in front. Both run on the one server, told apart by the LoRA scale.
        self.primary_arm = str(lc.get("primary_arm", "fewshot") or "fewshot")
        self.fallback_arm = str(lc.get("fallback_arm", "") or "") or None
        self.seed_model = str(lc.get("seed_model", "") or "") or self.model
        self.rewrite_cleared = bool(lc.get("rewrite_cleared", False))
        self.arms = ("fewshot",)             # what the server can serve, known once it's up
        self.rounds = args.rounds or int(lc["rounds_per_window"])
        self.after_round, self.streak = int(lc["stuck_after_round"]), int(lc["stuck_streak"])
        self.stuck_days = int(lc["stuck_days"])
        self.pool_draws = int(lc["pool_draws"])
        self.wait = 60 * float(lc["wait_minutes"])
        self.target = float(lc["target_ai_share"]) if str(lc.get("target_ai_share", "")).strip() else None
        # the author's calls of 2026-11-10, after a night where the filters killed 6,400 draws and 8 got through:
        self.protected_cap = int(lc.get("protected_words", 3))      # protected words per line of yours (plan 5.1)
        self.repeat_must = bool(lc.get("repeat_must_keep", True))   # a deliberate repeat: a hard filter, or a judge note
        self.rest_at_target = bool(lc.get("rest_at_target", True))  # a window under the target rests, Claude text or not
        self.purpose = "experiment" if args.smoke else "loop"
        self.bank = Bank(job.bank_path)
        # pools churn all through local work, so they live with the runtime files (11.2), not in the repo
        qdir = settings.runs_dir / job.name
        qdir.mkdir(parents=True, exist_ok=True)
        self.queue = PoolQueue(qdir / "pool queue.json")
        self.claude_block = None
        self.pangram_block = None
        self.alerted = set()
        self.timeline = []
        self.cap_events = []
        self.cuts = []
        self.stop_reason = None
        self.srv = self.op = None
        self.tics = None
        self.windows = []
        self.errors_in_row = 0
        self.smoke_t0 = clock.now()
        self.next_hourly = None
        self.worked_this_hour = {}
        self.pools_used = self.pool_draws_used = 0
        # the tiny-line rule (2026-11-10): a run under tiny_words words with tiny_tries tries today and no accepted
        # change is skipped for the rest of the run-day ("They're everywhere:" took 27 tries on 2026-11-09)
        self.tiny_words = int(lc.get("tiny_words", 6))
        self.tiny_tries = int(lc.get("tiny_tries", 3))
        # finding 2: transient errors are retried, then the door is held, and the day carries on
        self.retry_tries = int(lc.get("retry_tries", 3))
        self.retry_waits = [float(x) for x in lc.get("retry_wait_seconds", [30, 90])] or [30.0]
        self.error_block = 60 * float(lc.get("error_block_minutes", 10))
        self.now_stop = now_stop_time(schedule, clock.now())
        # a --now run gets the schedule's wind-down too (stop minus last attempt), so the day-end whole-text round
        # and the wrap-up fit before its stop
        sb = schedule.bounds(clock.now())
        self.wind_down = max(0.0, sb["stop"] - sb["last_attempt"])
        self.rounds_done = 0
        self.entered = False
        # M3 (plan 5.6): the whole-text round and the wrap-up
        self.reserve = float(lc.get("whole_text_reserve_usd", 0) or 0)
        # once the day's whole-text round is done, the reserve goes back to the window rounds down to this floor,
        # one section's scoring for a day-end round (2026-11-10 evening: five idle hours with $2 unspent)
        self.floor = min(self.reserve, float(lc.get("whole_text_floor_usd", 0) or 0))
        # a section never ships below its best known as-posted text; "worse" means by more than this (the fix
        # pass's noise floor, the same number)
        self.noise = float(settings.config.get("fixpass", {}).get("noise", 0.05))
        self.whole_rounds = int(lc.get("whole_text_rounds", 3))
        self.write_back_on = bool(lc.get("write_back", False))
        self.real_file = job.real_file
        self.whole_snapshot = None          # {wid: text} the last whole-text round ended on
        self.whole_log = []                 # one row per whole-text scoring, for the digest
        self.as_posted_first = None         # {section: ai} at today's first whole-text scoring
        self.as_posted_now = None           # {section: {"ai", "flagged", "segs"}} after the last one
        self.writeback_report = None
        self.final_scan = None
        # plan 9 (2026-11-10): two numbers for the author at the wrap-up, never a target for anything
        self.guide_scan_on = bool(lc.get("guide_scan", False))   # the whole guide scored once as it stands
        self.guide_scan = None              # engine/guide_scan.py's result, for the digest, the card and the reflection
        self.sections_combined = None       # the final scan's sections combined, weighted by words (free)
        self.reflection = None              # engine/reflect.py's result at the wrap-up (plan 9), for the digest
        self.restore_result = None          # engine/restore.py's result at the wrap-up (plan 6.9), for the digest
        # the morning submission right after the digest (engine/morning.py), behind [morning] at_digest
        self.morning_on = bool(settings.config.get("morning", {}).get("at_digest", True))
        self.morning_result = None          # its result, or why it was skipped
        self.whole_mode = False             # set while the whole-text round readies a window: its scores may use the reserve
        self.whole_wait_until = 0.0         # a whole-text round that couldn't start isn't tried again before this

    # ------------------------------------------------------------------ logging and posts
    def log(self, msg):
        line = f"{self.clock.stamp()} {msg}"
        self.timeline.append(line)
        print(line, flush=True)

    def alert(self, key, text):
        """An instant alert, once per key per run-day."""
        if key in self.alerted:
            return
        self.alerted.add(key)
        self.log(f"ALERT {text}")
        self.notifier.send(f"**depangram alert:** {text}")

    def tick(self):
        """The hourly post, when the clock has passed the hour."""
        now = self.clock.now()
        if self.next_hourly is None:
            self.next_hourly = (int(now // 3600) + 1) * 3600
        if now >= self.next_hourly:
            while self.next_hourly <= now:
                self.next_hourly += 3600
            if self.settings.config.get("discord", {}).get("hourly", True):
                self.notifier.send(discord.hourly_text(self))
            self.worked_this_hour = {}

    # ------------------------------------------------------------------ two-way Discord (plan 9.1, issue #8)
    def asks_on(self):
        """The loop asks and reads only on a real run with the bot path on: never in a rehearsal or a smoke test,
        never with no bot token, and not with [discord] questions = false. Off, every hook below returns at once
        and the loop runs exactly as it did before."""
        try:
            return (not getattr(self, "rehearsal", False) and not getattr(self.args, "smoke", False)
                    and bool(getattr(self.notifier, "live", False)) and discord.bot_on(self.settings)
                    and bool(self.settings.config.get("discord", {}).get("questions", True)))
        except Exception:  # noqa: BLE001
            return False

    def asked_file(self):
        if self.asked is None:
            from engine.answers import Asked
            self.asked = Asked(self.settings.runs_dir / self.job.name / "discord questions.json")
        return self.asked

    def note_day_post(self):
        """Today's start post's id: your plain "stop" and "rest s3w0" count only after it."""
        if not self.asks_on() or not self.day_post_id:
            return
        try:
            self.asked_file().day_posts[self.day] = self.day_post_id
            self.asked_file().save()
        except Exception as e:  # noqa: BLE001
            self.log(f"discord: today's start post wasn't noted ({type(e).__name__})")

    def ask_cut(self, ws, cut):
        """The drop rule's question (5.4, 9.1): "cut it? 1. yes (default) 2. keep it going". The cut has already
        happened, since no answer means the default; a 2 later puts the line back (veto_cut)."""
        if not self.asks_on():
            return
        try:
            from engine.answers import CUT_OPTIONS, CUT_VETO
            where = f"{cut['window']} p{cut['para'] + 1}.{cut['run'] + 1}"
            q = f"cut {where}? \"{cut['text'][:600]}\" (the judge: {cut['why'][:300]})"
            mid = discord.ask(self.settings, q, CUT_OPTIONS, default=1, veto=CUT_VETO)
            if not mid:
                self.log(f"discord: the cut question for {where} has no message id (it didn't post, or went out"
                         " through the webhook), so the cut stands; the digest asks again")
                return
            cut["question"] = mid
            self.asked_file().add({"id": mid, "kind": "cut", "window": cut["window"], "para": cut["para"],
                                   "run": cut["run"], "text": cut["text"], "day": getattr(self, "day", None),
                                   "ts": self.clock.now(), "time": self.clock.stamp(), "options": CUT_OPTIONS,
                                   "default": 1, "veto": CUT_VETO})
        except Exception as e:  # noqa: BLE001
            self.log(f"discord: the cut question didn't post ({type(e).__name__})")

    def ask_open_cuts(self):
        """At the digest: today's cuts with no question yet (a post that failed) get theirs now."""
        if not self.asks_on():
            return
        for c in self.cuts:
            if not c.get("question") and not c.get("vetoed"):
                self.ask_cut(None, c)

    def read_answers(self, force=False):
        """Your answers on Discord (plan 9.1): before each pass, every poll_seconds while idle, and at the digest.
        A 2 (or a cross) on a cut puts the line back; "rest s3w0" rests a window for the run-day; "stop" stops the
        day like the STOP file. Nothing an answer does raises a cap, changes a target or spends money, and an
        error here only costs the answer: the defaults stand."""
        if not self.asks_on():
            return
        now = self.clock.now()
        if not force and now < self.next_poll:
            return
        self.next_poll = now + float(self.settings.config.get("discord", {}).get("poll_seconds", 60))
        try:
            from engine import answers
            asked = self.asked_file()
            asked.prune(now)
            got, stop, rest = answers.read(self.settings, asked, asked.day_posts.get(getattr(self, "day", None)))
            for q in asked.open():
                n = got.get(q["id"])
                if n is None:
                    continue
                q["answer"], q["answered"] = n, self.clock.stamp()
                if q.get("kind") == "cut" and n == q.get("veto"):
                    self.veto_cut(q)
                else:
                    self.log(f"discord: you answered {n} on {q.get('window')} p{q.get('para', 0) + 1}."
                             f"{q.get('run', 0) + 1}'s cut, so it stands")
            asked.save()
            for wid in rest:
                self.rest_window(wid)
            if stop and not self.discord_stop:
                self.discord_stop = True
                self.log("discord: you said stop, so the day stops as if the STOP file were there")
                if not self.stop_reason:
                    self.stop_reason = "the STOP reply on Discord"
        except Exception as e:  # noqa: BLE001
            self.log(f"discord: reading your answers failed ({type(e).__name__}); the defaults stand")

    def heard_stop(self):
        """A "stop" on Discord, also read between the full reads: must_stop and hard_stop ask here, at most once
        every poll_seconds, so a stop lands inside a pass or the whole-text round within a minute or so, near enough
        the STOP file. This reads the stop only; a veto or a rest waits for the next full read, between passes,
        where changing a window's text is safe."""
        if self.discord_stop:
            return True
        if not self.asks_on():
            return False
        now = self.clock.now()
        if now < self.next_stop_poll:
            return False
        self.next_stop_poll = now + float(self.settings.config.get("discord", {}).get("poll_seconds", 60))
        try:
            from engine import answers
            day_post = self.asked_file().day_posts.get(getattr(self, "day", None))
            if not day_post:
                return False
            stop, _ = answers.commands(self.settings, discord.replies_since(self.settings, day_post), day_post)
        except Exception:  # noqa: BLE001  (a failed read is no stop: the STOP file still works)
            return False
        if stop:
            self.discord_stop = True
            self.log("discord: you said stop, so the day stops as if the STOP file were there")
        return self.discord_stop

    def veto_key(self, ws, para, run, text):
        """Where a vetoed cut sits in today's runs: the run whose original is the cut text, else (para, run)."""
        key = " ".join((text or "").split())
        for pi, p in enumerate(getattr(ws, "original", None) or []):
            for ri, r in enumerate(p):
                if key and " ".join(" ".join(r).split()) == key:
                    return pi, ri
        return para, run

    def veto_cut(self, q):
        """Your "keep it going" on a cut: banked as a veto entry, so a restart, the next run-day and every pick
        (failing, demote, restart_from_best, the ships) keep the line in, and the drop rule never cuts it again.
        The window carries on: a window that rested after the cut works again. No Pangram, no Claude: the text
        the window goes back to is a banked one, and an unscored one waits for the normal doors."""
        self.bank.add({"kind": "veto", "window": q["window"], "day": getattr(self, "day", None),
                       "time": self.clock.stamp(), "para": q["para"], "run": q["run"], "cut_text": q["text"],
                       "question": q["id"]})
        for c in self.cuts:
            if c["window"] == q["window"] and (c["para"], c["run"], c["text"]) == (q["para"], q["run"], q["text"]):
                c["vetoed"] = True
        self.log(f"discord: you said keep it going, so {q['window']} p{q['para'] + 1}.{q['run'] + 1} goes back in:"
                 f" \"{q['text'][:80]}\"")
        ws = next((w for w in self.windows if w.wid == q["window"]), None)
        if ws is None or not ws.ready:
            return                               # prepare() reads the veto from the bank (load_calls)
        pi, ri = self.veto_key(ws, q["para"], q["run"], q["text"])
        ws.vetoed.add((pi, ri))
        if not ws.pass_open:
            self.restart_from_best(ws)           # the best banked text that keeps the line, else the line put back
        if pi < len(ws.current) and ri < len(ws.current[pi]) and not ws.current[pi][ri]:
            ws.current[pi][ri] = ws.original[pi][ri][:]
            ws.last_text = window_text_of(ws.current, ws.heading)
            hit = self.latest_score(ws.wid, ws.last_text)
            ws.ai, ws.segs = hit if hit else (None, [])
            ws.scored_runs = json.loads(json.dumps(ws.current)) if hit else None
            ws.flagged = flag_list(ws.segs)
        ws.couldnt_cut = [c for c in ws.couldnt_cut if (c[0], c[1]) != (pi, ri)]
        if ws.rested_today and not ws.rest_call:
            ws.rested_today, ws.state = False, "working"

    def rest_window(self, wid):
        """Your "rest s3w0": the window sits out the rest of the run-day (banked, so a restart keeps it). An open
        pass stays open and picks up on the next run-day, the way a stop mid-pass does."""
        ws = next((w for w in self.windows if w.wid == wid), None)
        if ws is None or ws.rest_call:
            return
        ws.rested_today, ws.rest_call = True, True
        self.bank.add({"kind": "rest_call", "window": wid, "day": getattr(self, "day", None),
                       "time": self.clock.stamp()})
        self.log(f"discord: you said rest {wid}, so it sits out the rest of today")

    def load_calls(self, ws):
        """Your Discord answers already in the bank, for a window being readied: every veto (for good) and a rest
        for today."""
        for e in self.bank.of(ws.wid, "veto"):
            ws.vetoed.add(self.veto_key(ws, e.get("para", 0), e.get("run", 0), e.get("cut_text")))
        if any(e.get("day") == getattr(self, "day", None) for e in self.bank.of(ws.wid, "rest_call")):
            ws.rested_today, ws.rest_call = True, True

    # ------------------------------------------------------------------ stops and blocks
    def must_stop(self):
        if self.stop_reason:
            return True
        now = self.clock.now()
        if self.stop_file and self.stop_file.exists():
            self.stop_reason = "the STOP file"
        elif self.heard_stop():
            self.stop_reason = "the STOP reply on Discord"
        elif not self.args.now and not self.schedule.may_start_attempt(now):
            self.stop_reason = f"{self.schedule.phase(now)}: no new attempts after {self.settings.config['schedule']['last_attempt']}"
        elif self.args.now and now >= self.now_stop - getattr(self, "wind_down", 0.0):
            self.stop_reason = (f"a --now run's wind-down: no new attempts after"
                                f" {self.clock.stamp(self.now_stop - getattr(self, 'wind_down', 0.0))}")
        elif self.args.smoke and self.smoke_spent_out():
            self.stop_reason = "the smoke test's spend limit"
        return bool(self.stop_reason)

    def hard_stop(self):
        """The whole-text round's own stop (must_stop has already said yes by the day-end trigger): the STOP file,
        or five minutes before the day's stop time. A "stop" on Discord counts as the STOP file (plan 9.1)."""
        if (self.stop_file and self.stop_file.exists()) or self.heard_stop():
            return True
        now = self.clock.now()
        end = self.now_stop if self.args.now else self.schedule.bounds(now)["stop"]
        return now >= end - 300

    def smoke_spent_out(self):
        lc = self.settings.config["loop"]
        pg = self.budget.pangram_spent(self.smoke_t0, "experiment")
        cl = self.budget.loop_dollars(self.smoke_t0)
        return pg >= self.smoke_limit("pangram") or cl >= self.smoke_limit("claude")

    def smoke_limit(self, which):
        """--smoke's spend limit: --smoke-pangram-usd / --smoke-claude-usd when given, else config.toml's."""
        given = getattr(self.args, f"smoke_{which}_usd", None)
        return float(given) if given is not None else float(self.settings.config["loop"][f"smoke_{which}_usd"])

    def blocked(self):
        now = self.clock.now()
        if self.claude_block and now >= (self.claude_block.until or 0):
            self.log(f"{self.claude_block.check} has lifted")
            self.claude_block = None
        if self.pangram_block and now >= (self.pangram_block.until or 0):
            self.log(f"{self.pangram_block.check} has lifted")
            self.pangram_block = None
        return self.claude_block or self.pangram_block

    def on_cap(self, e):
        now = self.clock.now()
        if e.until is None or e.until <= now:
            e.until = self.schedule.bounds(now)["stop"] + 1
        if isinstance(e, ClaudeCapReached):
            self.claude_block = e
        else:
            self.pangram_block = e
        until = self.clock.stamp(e.until)
        self.cap_events.append({"time": self.clock.stamp(), "check": e.check, "detail": e.detail, "until": until})
        if self.args.smoke and not self.stop_reason:
            # a smoke test is one round: a cap ends it, or it would sit drawing local pools until 16:00 (2026-11-09)
            self.stop_reason = f"the smoke test stops at a cap ({e.check})"
        snap = self.budget.snapshot()
        extra = ""
        if isinstance(e, ClaudeCapReached):
            extra = (f" Loop share {snap['loop_share']:.1%} of the week, your week {snap['real_total']:.0%},"
                     f" week resets {self.clock.stamp(snap['week_reset'])}.")
        if e.check.endswith(" errors"):           # a hold after failed retries: one alert a day per door is plenty
            self.alert(f"cap {e.check} {self.schedule.day_key(now)}",
                       f"{e.check}: {e.detail}. Tried {self.retry_tries} times; that door is held until {until},"
                       " and the loop carries on with what it can do.")
            return
        self.alert(f"cap {e.check} {self.schedule.day_key(now)} {until}",
                   f"{e.check} said no ({e.detail}). Holds until {until}.{extra}"
                   + (" Claude work waits; local draws carry on." if isinstance(e, ClaudeCapReached)
                      else " Scoring waits; draws bank for later."))

    def retrying(self, what, fn, errors):
        """`fn` up to retry_tries times, waiting retry_wait_seconds between tries (finding 2). The last error goes up
        to the caller, which turns it into a hold."""
        for i in range(self.retry_tries):
            try:
                return fn()
            except errors as err:
                if i + 1 >= self.retry_tries:
                    raise
                wait = self.retry_waits[min(i, len(self.retry_waits) - 1)]
                self.log(f"{what} failed ({redact(f'{type(err).__name__}: {err}')[:140]}); try {i + 2} of"
                         f" {self.retry_tries} in {wait:.0f}s")
                self.clock.sleep(wait)

    def hold(self, door, err):
        """After the retries, hold the door (Claude or Pangram) for error_block_minutes, like a cap that lifts."""
        cls = ClaudeCapReached if door == "Claude" else PangramCapReached
        return cls(f"{door} errors", redact(f"{type(err).__name__}: {err}")[:160],
                   until=self.clock.now() + self.error_block)

    # ------------------------------------------------------------------ the two paid doors
    def claude(self, settings, conn, prompt, role, model="sonnet", max_budget_usd=0.5, **kw):
        """Every Claude call the loop makes comes through here, and the capper runs first."""
        now = self.clock.now()
        if self.claude_block and now < (self.claude_block.until or 0):
            raise self.claude_block
        if self.args.smoke and (self.budget.loop_dollars(self.smoke_t0) + max_budget_usd
                                > self.smoke_limit("claude")):
            # the call's own --max-budget-usd is its worst case, so the smoke test can't pass its limit at all
            raise ClaudeCapReached("smoke test limit", "the smoke test's Claude limit", until=now + 86400)
        def once():
            # the capper runs before every try, so a retry can't sneak past a cap the first try used up
            self.budget.claude_check(kind_of(role), FALLBACK_USD.get(kind_of(role), max_budget_usd))
            try:
                out = self.claude_raw(settings, conn, prompt, role=f"loop {role}", model=model,
                                      max_budget_usd=max_budget_usd, **kw)
                self.errors_in_row = 0
                return out
            except ClaudeCallError as e:     # a timeout comes here too: claude_call logs it at its ceiling
                msg = redact(e)
                self.errors_in_row += 1
                if re.search(r"session limit|usage limit|hit your|limit reached", msg, re.I):
                    raise ClaudeCapReached("Claude session limit", msg[:160],
                                           until=self.session_reset(msg, self.clock.now())) from None
                if re.search(r"\b401\b|unauthori[sz]ed|invalid (?:api|oauth|token)|token (?:has )?expired|"
                             r"please (?:log|sign) in", msg, re.I):
                    self.alert("token", "Claude rejected the login token. Make a new one with `claude setup-token`"
                                        " (plan 3.3).")
                raise
        try:
            return self.retrying(f"Claude {kind_of(role)}", once, (ClaudeCallError,))
        except ClaudeCallError as err:
            raise self.hold("Claude", err) from None

    def session_reset(self, msg, now):
        """When "You've hit your session limit, resets 5am" lifts (7.2): the time in the message, else the 5-hour
        reset the last reading carried, else thirty minutes on."""
        m = re.search(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*([ap]m)?", msg, re.I)
        if m:
            h, mi = int(m.group(1)) % 12 if m.group(3) else int(m.group(1)), int(m.group(2) or 0)
            if m.group(3) and m.group(3).lower() == "pm":
                h += 12
            d = dt.datetime.fromtimestamp(now).replace(hour=h % 24, minute=mi, second=0, microsecond=0)
            ts = d.timestamp() if d.timestamp() > now else d.timestamp() + 86400
            if ts - now <= 6 * 3600:
                return ts
        reset = self.budget.five_hour_util(now)[1]
        return reset if reset and reset > now else now + 1800

    def score(self, text, label, whole=False):
        """Pangram, through the cache first and the caps second. Raises PangramCapReached. The window rounds keep
        whole_text_reserve_usd of the day cap back; the whole-text round (`whole`) may spend it, and passes the
        "window share" hold the reserve puts up."""
        hit = self.pg.cached(text)
        if hit is not None:
            return hit
        whole = whole or getattr(self, "whole_mode", False)
        now = self.clock.now()
        if self.pangram_block and now < (self.pangram_block.until or 0) \
                and not (whole and self.pangram_block.check == WINDOW_SHARE):
            raise self.pangram_block
        _, usd = self.pg.estimate(text)
        if self.args.smoke:
            if self.budget.pangram_spent(self.smoke_t0, "experiment") + usd > self.smoke_limit("pangram"):
                raise PangramCapReached("smoke test limit", "the smoke test's Pangram limit", until=now + 86400)
        def once():
            # before every try: a failed try may have been charged
            self.budget.pangram_check(usd, self.purpose, reserve=0.0 if whole else self.reserve)
            return self.pg.score(text, self.purpose, f"run {label}")[0]
        try:
            resp = self.retrying("Pangram", once, (PangramError, requests.RequestException))
        except PangramClientCap as e:          # engine/pangram.py's own check, the second lock on the door
            raise PangramCapReached("Pangram (client check)", str(e),
                                    until=self.schedule.bounds(now)["midnight"] + 86400) from None
        except (PangramError, requests.RequestException) as err:
            raise self.hold("Pangram", err) from None
        if getattr(self.pg, "model_change", None):
            self.check_model()
        return resp

    def check_model(self):
        """Plan 9's model-change alert. While Pangram's latest version isn't the one config.toml's [pangram] model_ack
        names (and the ledger has seen it change): an ALERT once a run, every banked window score from before the
        change counts as unscored (latest_score), a window already standing on one scores again at its next round,
        the cache ignores the old version's answers (engine/pangram.py), and the whole-text round is due until one
        has run since the change."""
        status = getattr(self.pg, "model_status", None)
        if status is None:
            return None
        st = status()
        if st.get("stale"):
            st["changed_at_text"] = self.clock.stamp(st["changed_at"]) if st.get("changed_at") else "?"
            if not getattr(self, "stale_applied", False):
                self.stale_applied = True
                for ws in self.windows:
                    if ws.ready:                   # its standing score came from the old model
                        ws.ai, ws.scored_runs = None, None
            self.alert(f"pangram model {st['latest']}",
                       f"Pangram answered with model version {st['latest']} (first at {st['changed_at_text']}); the"
                       f" banked scores were made with {st['trusted']}. They count as stale: each window's text is"
                       " scored again before the loop trusts it, and the whole-text round runs again. Once you've"
                       f" looked, set model_ack = \"{st['latest']}\" under [pangram] in config.toml (read at the"
                       " next start).")
        self.model_status = st
        return st

    def trusts(self, entry):
        """A banked score the loop may use: any, unless Pangram's model changed and nobody has acknowledged it, when
        only scores banked since the change count."""
        st = getattr(self, "model_status", None)
        if not st or not st.get("stale") or not st.get("changed_at"):
            return True
        return (entry.get("time") or "") >= self.clock.stamp(st["changed_at"])

    # ------------------------------------------------------------------ setup
    def setup(self):
        from engine.draws import fmt_example
        from engine.draws import make_tics
        self.fmt_example = fmt_example
        self.tics = make_tics()
        self.vm = self.job.verdict_map()
        heads = self.job.headings()
        self.locked = lambda s: is_template(s) or s in heads
        wins = self.job.windows()
        if self.args.window:
            wins = [w for w in wins if w["wid"] in self.args.window.split(",")]
        prev = {w.wid: w for w in self.windows}
        self.windows = [WindowState(w) for w in wins]
        for ws in self.windows:                  # a restart of the day (finding 2) keeps today's counts
            old = prev.get(ws.wid)
            if old:
                ws.passes_today, ws.rounds_today, ws.start_of_day = old.passes_today, old.rounds_today, old.start_of_day
                ws.rested_today, ws.couldnt_cut = old.rested_today, old.couldnt_cut
                ws.rest_call = getattr(old, "rest_call", False)
        self.header = self.job.voice_header()
        self.examples = self.job.author_paragraphs(self.vm)
        # the per-section swear budget (5.3): your own rate, measured once a day
        self.swear_rates = measured_swear_rates(self.vm)
        # the swear and filler budgets (a filler tag is "honestly" tacked onto a clause): the rules file's, or an
        # older config's [loop] keys with a warning
        self.swear_mult, self.filler_cap, self.filler_words = _checks.budgets(self.settings.config, warn=self.log)
        self.examples_block = None
        token = self.settings.config["claude"]
        try:
            made = dt.date.fromisoformat(token["token_made"])
            left = (made + dt.timedelta(days=int(token["token_life_days"])) - dt.date.fromtimestamp(self.clock.now())).days
            if left <= int(token["token_warn_days"]):
                self.alert("token expiry", f"the Claude token runs out in about {left} days. Make a new one with"
                                           " `claude setup-token` (plan 3.3) and update token_made in config.toml.")
        except (KeyError, ValueError):
            pass

    def notes(self, paras, role):
        """job.notes_for through the Claude door, with a reply that won't parse retried like a failed call."""
        try:
            return self.retrying("notes", lambda: self.job.notes_for(self.claude, self.settings, self.conn, paras,
                                                                     role=role), (NotesError,))
        except NotesError as err:
            raise self.hold("Claude", err) from None

    def ensure_examples(self):
        """The worked examples' notes (one Sonnet call, cached). False while Claude is capped."""
        if self.examples_block is not None:
            return True
        try:
            ex_notes = self.notes(self.examples, "examples notes")
        except ClaudeCapReached as e:
            self.on_cap(e)
            return False
        self.example_notes = ex_notes
        self.examples_block = "\n\n---\n\n".join(self.fmt_example(n, p) for n, p in zip(ex_notes, self.examples))

    def examples_without(self, line):
        """The worked examples with any paragraph that holds `line` taken out. The examples are your own all-yours
        paragraphs, so a line of yours being regrown can sit in one, and Mistral then copies it word for word
        (the 2026-11-09 smoke test: 48 of 48 draws identical for five lines of s6w0)."""
        key = " ".join(line.split())
        keep = [(n, p) for n, p in zip(getattr(self, "example_notes", []), self.examples) if key not in " ".join(p.split())]
        if len(keep) == len(self.examples):
            return self.examples_block
        return "\n\n---\n\n".join(self.fmt_example(n, p) for n, p in keep)
        return True

    def prepare(self, ws):
        """Round 0, the runs, the notes, the seed and where to start from. False when a cap holds it up."""
        from engine.draws import SWEAR, build_runs
        from engine.draws import names
        if ws.ready:
            return True
        start = self.bank.start(ws.wid)
        if not start:
            try:
                resp = self.score(ws.text, f"{ws.wid} round 0")
            except PangramCapReached as e:
                self.on_cap(e)
                return False
            s = summarize(resp)
            start = self.bank.add({"kind": "start", "window": ws.wid, "time": self.clock.stamp(), "ai": s["fraction_ai"],
                                   "flagged": sum(1 for _, f in segs_of(resp) if f), "segs": segs_of(resp),
                                   "text": ws.text})
        who, heading, body, runs0 = build_runs(ws.text, flag_list(start["segs"]), verdicts=self.vm, notation=self.locked)
        # your lines go one sentence a run (plan 5.1), and the bank's older entries follow them into that shape
        runs, index = split_author_runs(runs0, self.locked, verdict_of=who.get)
        moved, lost = self.bank.reshape(ws.wid, runs0, index)
        if lost:
            self.log(f"{ws.wid}: {lost} banked text(s) don't fit the one-line-a-run shape for your lines and are skipped")
        ws.who, ws.heading, ws.body, ws.runs = who, heading, body, runs
        ws.original = [[r["sents"][:] for r in p] for p in runs]
        has_claude = any(r["claude"] for p in runs for r in p)
        ws.notes_ready = False
        if has_claude:
            if not self.ensure_examples():
                return False
            try:
                ws.notes = self.notes([" ".join(p) for p in body], f"{ws.wid} notes")
                ws.notes_ready = True
            except ClaudeCapReached as e:
                self.on_cap(e)
                return False
        else:
            # nothing of Claude's to regrow, so no notes yet; a line of yours that Pangram flags fetches them then
            ws.notes = [[] for _ in body]
        ws.names = names(ws.text) | names(self.header) | names(self.examples_block or "")
        ws.swear_budget = len(SWEAR.findall(ws.text)) + 1
        ws.must = {(pi, ri): dict(must_keeps(" ".join(r["sents"]), ws.notes[pi]),
                                  names=names_required(" ".join(r["sents"]), ws.notes[pi]))
                   for pi, p in enumerate(runs) for ri, r in enumerate(p) if r["claude"]}
        self.author_must(ws)
        self.import_seed(ws)
        # unscored texts (a seed rebuilt around your lines, a round banked while Pangram was capped) get scored
        todo = {e["text"]: e for e in self.bank.unscored(ws.wid)[-1:] + self.bank.of(ws.wid, "seed")
                if e.get("text") and e.get("ai") is None and not self.latest_score(ws.wid, e["text"])}
        for ent in todo.values():
            try:
                resp = self.score(ent["text"], f"{ws.wid} rescore")
            except PangramCapReached as err:
                self.on_cap(err)
                return False
            s = summarize(resp)
            self.bank.add({"kind": "rescore", "window": ws.wid, "time": self.clock.stamp(), "day": self.day,
                           "g": ent.get("g", 0), "ai": s["fraction_ai"], "humanized": s["humanized"],
                           "flagged": sum(1 for _, f in segs_of(resp) if f), "segs": segs_of(resp),
                           "text": ent["text"], "runs": ent["runs"]})
            self.log(f"{ws.wid}: scored a banked text that had no score: AI share {s['fraction_ai']:.2f}")
        self.load_calls(ws)                       # your Discord answers in the bank (plan 9.1), before the resume
        self.resume(ws, start)
        if ws.ai is None:
            # a text with no score yet (a must-keep demotion just made it): score it before the window works
            try:
                resp = self.score(ws.last_text, f"{ws.wid} rescore")
            except PangramCapReached as err:
                self.on_cap(err)
                return False
            s = summarize(resp)
            self.bank.add({"kind": "rescore", "window": ws.wid, "time": self.clock.stamp(), "day": self.day,
                           "g": ws.g, "ai": s["fraction_ai"], "humanized": s["humanized"],
                           "flagged": sum(1 for _, f in segs_of(resp) if f), "segs": segs_of(resp),
                           "text": ws.last_text, "runs": json.loads(json.dumps(ws.current))})
            ws.ai, ws.segs, ws.flagged = s["fraction_ai"], segs_of(resp), flag_list(segs_of(resp))
            ws.scored_runs = json.loads(json.dumps(ws.current))
            if ws.start_of_day is None:
                ws.start_of_day = ws.ai
        ws.ready = True
        return True

    def author_must(self, ws, words_by_line=None):
        """Must-keeps for your lines (plan 5.1 and 5.3's capped list): the protected words, the numbers, and the
        quotes and citations draw_pool already holds; no hedges or repeats. The words come from Sonnet's cached pick,
        else from code (rarest words) until Sonnet is asked, which happens when the line is first targeted."""
        from engine.draws import numbers
        for pi, p in enumerate(ws.runs):
            for ri, r in enumerate(p):
                if not r.get("author"):
                    continue
                line = " ".join(r["sents"])
                got = (words_by_line or {}).get(line)
                if got is None:
                    hit = self.job.protected_cached(line)
                    got = hit["words"] if hit else None
                ws.must[(pi, ri)] = {"hedges": [], "repeats": [], "author": True,
                                     "protected": (got if got is not None
                                                   else protected_fallback(line, cap=self.protected_cap))[:self.protected_cap],
                                     "asked": got is not None, "numbers": sorted(numbers(line)),
                                     "names": names_required(line, ws.notes[pi] if ws.notes_ready else ())}

    def ensure_author(self, ws, runs_wanted):
        """Before a line of yours is drawn for: the window's notes (a window with no Claude text never paid for
        them) and Sonnet's protected words for every line of yours in the window, one cached call. False while
        Claude is capped (the round waits, like any Claude call)."""
        try:
            if not ws.notes_ready:
                if not self.ensure_examples():
                    return False
                ws.notes = self.notes([" ".join(p) for p in ws.body], f"{ws.wid} notes")
                ws.notes_ready = True
            lines = [" ".join(ws.runs[pi][ri]["sents"]) for pi, ri in runs_wanted
                     if not ws.must[(pi, ri)].get("asked")]
            if lines:
                every = [" ".join(r["sents"]) for p in ws.runs for r in p if r.get("author")]
                got = self.retrying("protected words", lambda: self.job.protected_for(
                    self.claude, self.settings, self.conn, every, role=f"{ws.wid} protect"), (NotesError,))
                self.author_must(ws, got)
        except NotesError as err:
            self.on_cap(self.hold("Claude", err))
            return False
        except ClaudeCapReached as e:
            self.on_cap(e)
            return False
        return True

    def import_seed(self, ws):
        """M1: the windows job.toml [seed] lists start from the best text the experiments banked."""
        seed_cfg = self.job.cfg.get("seed", {})
        if ws.wid not in seed_cfg.get("windows", []) or self.bank.of(ws.wid, "seed"):
            return
        # the experiment banks were drawn by the few-shot writer on its old base; seed_model names it now that the
        # loop's server is the adapter's (2026-11-09)
        b = best_seed(self.job.dir, seed_cfg.get("banks", []), ws.wid, self.seed_model, arm="fewshot",
                      untagged=seed_cfg.get("untagged_window"))
        if not b:
            return
        name, e = b
        aligned, exact = align(ws.runs, e["runs"])
        if aligned is None:
            self.log(f"{ws.wid}: the seed from {name} doesn't line up with this job's runs; starting from the original")
            self.bank.add({"kind": "seed", "window": ws.wid, "source": name, "failed": True, "time": self.clock.stamp()})
            return
        text = window_text_of(aligned, ws.heading)
        entry = {"kind": "seed", "window": ws.wid, "source": name, "source_time": e.get("time"), "time": self.clock.stamp(),
                 "exact": exact, "text": text, "runs": aligned, "g": 0}
        if exact:
            entry.update({"ai": e["ai"], "flagged": e.get("flagged"),
                          "segs": [[t, True] for t in e.get("flagged_segs", [])]})
        else:
            entry["ai"] = None
            entry["note"] = "your lines put back as written; the rest is the seed's text"
        self.bank.add(entry)
        self.log(f"{ws.wid}: seeded from {name} ({e['ai']:.2f}{'' if exact else ', your lines put back, so it gets a fresh score'})")

    def latest_score(self, wid, text):
        """(ai, segs) of the newest scored bank entry holding exactly this text, or None. The original window put
        back together from its runs differs from the cut window only in spacing ("*together*,  combining"), so it
        takes round 0's score instead of paying for a second one."""
        for e in reversed(self.bank.of(wid)):
            if e.get("text") == text and e.get("ai") is not None and self.trusts(e):
                return e["ai"], e.get("segs") or [[t, True] for t in e.get("flagged_segs", [])]
        ws = next((w for w in getattr(self, "windows", []) if w.wid == wid), None)
        start = self.bank.start(wid)
        if ws is not None and start and self.trusts(start) and getattr(ws, "original", None) \
                and text == window_text_of(ws.original, ws.heading):
            return start["ai"], start["segs"]
        return None

    def resume(self, ws, start):
        """Where the window starts: the open pass's latest round, else the best scored text, else the original.
        Its score is the newest one banked for exactly that text; a text with none waits for Pangram."""
        shape = [len(p) for p in ws.runs]
        ok = lambda e: e and e.get("runs") is not None and [len(p) for p in e["runs"]] == shape  # noqa: E731
        ws.g = self.bank.g(ws.wid)
        ws.pass_no = self.bank.passes(ws.wid)
        op = self.bank.open_pass(ws.wid)
        best = self.base_entry(ws)
        base = None
        if op and op[1] and ok(op[1][-1]):
            base = op[1][-1]
            cut = resume_base(self.bank, ws.wid, op)        # a cut banked after the last round stands (finding 11)
            if cut is not base and ok(cut):
                base = cut
            ws.pass_open = True
            done = [e for e in op[1] if not e.get("interrupted")]
            ws.completed = len(done)
            ws.changed = [e.get("changed", True) for e in done]
            ws.settle_next = bool(op[1][-1].get("settle_next"))
            ws.attempts = {tuple(map(int, k.split("."))): n for k, n in op[1][-1].get("attempts", {}).items()}
            if base is not op[1][-1]:
                self.log(f"{ws.wid}: the drop rule's cut after round {ws.completed} stands")
            self.log(f"{ws.wid}: resuming pass {ws.pass_no} after round {ws.completed}")
        elif ok(best):
            base = best
        ws.current = json.loads(json.dumps(base["runs"])) if base else [[r[:] for r in p] for p in ws.original]
        self.demote(ws)
        if self.apply_flavor(ws):
            self.log(f"{ws.wid}: the flavor pass's sentence(s) carried into the text the window works on")
        ws.last_text = window_text_of(ws.current, ws.heading)
        hit = self.latest_score(ws.wid, ws.last_text)
        if hit:
            ws.ai, ws.segs = hit
            ws.scored_runs = json.loads(json.dumps(ws.current))
        else:
            ws.ai, ws.segs, ws.scored_runs = None, [], None
        ws.flagged = flag_list(ws.segs)
        for e in self.bank.of(ws.wid, "round"):
            for c in e.get("changes", []):
                if "run" in c:
                    ws.history.setdefault((c["para"], c["run"]), []).append(c["result"])
                    # today's tries with no accepted change, for the tiny-line rule (a restart of the day keeps count)
                    if e.get("day") == getattr(self, "day", None) and not str(c.get("result", "")).startswith("queued"):
                        self.tiny_note(ws, c["para"], c["run"], accepted="new" in c)
        days = sorted({e.get("day") for e in self.bank.pass_ends(ws.wid) if e.get("day")})
        best_day = (best or {}).get("day")
        later = [d for d in days if not best_day or d > best_day]
        ws.retired = len(later) >= self.stuck_days and not self.args.smoke
        if ws.start_of_day is None:              # a restart of the day keeps the score the day began with
            ws.start_of_day = ws.ai

    # ------------------------------------------------------------------ one run
    def targets(self, ws, is_last):
        """Which runs this round tries (experiment 6's targeting, with experiment 8's check that a flag is on the
        current text): runs still carrying Claude text, replacements Pangram flags in a paragraph with no Claude
        text left and no long quote, and the must-keep audit's re-rolls."""
        from engine.draws import long_quote
        out = []
        # a run the flavor pass wrote into stays (the tests call this with no loop behind it)
        live = self is not None and getattr(self, "bank", None) is not None
        locked = (lambda pi, ri: self.flavor_locked(ws, pi, ri)) if live else (lambda pi, ri: False)
        for pi, p in enumerate(ws.runs):
            for ri, r in enumerate(p):
                if locked(pi, ri):
                    continue
                if live and self.tiny_skip(ws, pi, ri):
                    continue
                if r.get("author"):
                    # plan 5.1: a line of yours stays as written while Pangram passes it. Flagged, it's regrown like
                    # a replaced Claude run: only in a paragraph with no Claude text left (a flag next to Claude's
                    # residue says nothing about your line) and no long quote (the quote flags the paragraph itself)
                    if not ws.current[pi][ri]:
                        continue
                    para_clean = all(not rr["claude"] or ws.current[pi][x] != ws.original[pi][x] for x, rr in enumerate(p))
                    has_quote = any(long_quote(s) and not is_template(s) for rr in ws.current[pi] for s in rr)
                    flagged = (ws.scored_runs is not None and ws.current[pi][ri] == ws.scored_runs[pi][ri]
                               and any(s[:40] in f for s in ws.current[pi][ri] for f in ws.flagged))
                    if para_clean and flagged and not has_quote:
                        out.append((pi, ri, False))
                    continue
                if not r["claude"]:
                    continue
                if (pi, ri) in ws.forced:
                    out.append((pi, ri, True))
                    continue
                if ws.current[pi][ri] == ws.original[pi][ri]:
                    out.append((pi, ri, False))
                    continue
                if not ws.current[pi][ri]:
                    continue                               # cut by the drop rule
                para_clean = all(not rr["claude"] or ws.current[pi][x] != ws.original[pi][x] for x, rr in enumerate(p))
                has_quote = any(long_quote(s) and not is_template(s) for rr in ws.current[pi] for s in rr)
                run_flagged = (ws.scored_runs is not None and ws.current[pi][ri] == ws.scored_runs[pi][ri]
                               and any(s[:40] in f for s in ws.current[pi][ri] for f in ws.flagged))
                if para_clean and run_flagged and not has_quote:
                    out.append((pi, ri, False))
        # rewrite_cleared (config): a clean window with nothing else to do gets its few-shot-written runs regrown
        # once more by the adapter, for readability
        ws.rewrites = set()
        if not out and live and getattr(self, "rewrite_cleared", False):
            rw = self.rewrite_targets(ws)
            ws.rewrites = {(pi, ri) for pi, ri, _ in rw}
            out += rw
        return out

    # ------------------------------------------------------------------ the tiny-line rule (2026-11-10)
    def tiny_note(self, ws, pi, ri, accepted):
        """Count a try on a run for the tiny-line rule: an accepted change clears the count, a miss adds one."""
        if accepted:
            ws.tiny_misses[(pi, ri)] = 0
        else:
            ws.tiny_misses[(pi, ri)] = ws.tiny_misses.get((pi, ri), 0) + 1

    def tiny_skip(self, ws, pi, ri):
        """True when the run standing at (pi, ri) is under tiny_words words and has had tiny_tries tries today with
        no accepted change: it's skipped for the rest of the run-day, logged once, and listed in the digest."""
        words = getattr(self, "tiny_words", 6)
        tries = getattr(self, "tiny_tries", 3)
        if words <= 0 or tries <= 0:
            return False
        cur = ws.current[pi][ri] if pi < len(ws.current) and ri < len(ws.current[pi]) else None
        text = " ".join(cur) if cur else " ".join(ws.original[pi][ri])
        misses = getattr(ws, "tiny_misses", None) or {}
        if len(text.split()) >= words or misses.get((pi, ri), 0) < tries:
            return False
        skipped = getattr(ws, "tiny_skipped", None)
        if skipped is not None and (pi, ri) not in skipped:
            skipped[(pi, ri)] = text
            self.log(f"{ws.wid} p{pi + 1}.{ri + 1}: skipped as tiny for the rest of the day ({len(text.split())} words, "
                     f"{misses[(pi, ri)]} tries with no change): \"{text[:60]}\"")
        return True

    def flavor_inserts(self, ws):
        """{(pi, ri): (the run's sentences before the pass, the run's sentences as the pass left it)} from the bank's
        flavor entries (engine/flavor.py: the insert appended) and restore entries (engine/restore.py: the run
        line-edited; 2026-11-10, the one lock for both)."""
        from .restore import locked_inserts
        return locked_inserts(self.bank, ws.wid)

    def apply_flavor(self, ws, runs=None):
        """The second part of the lock: a flavor or restore entry banks with no window score, so it is never
        the best a window resumes from, and the pass's text would be missing from ws.current. A run standing at the
        text the pass started from gets the pass's text back, so the rounds work on it, bank it, and never target
        it. Returns the (pi, ri) put back."""
        runs = ws.current if runs is None else runs
        done = []
        for (pi, ri), (base, full) in self.flavor_inserts(ws).items():
            if pi < len(runs) and ri < len(runs[pi]) and runs[pi][ri] == base:
                runs[pi][ri] = list(full)
                done.append((pi, ri))
        return done

    def flavor_locked(self, ws, pi, ri, runs=None):
        """A run whose text (in `runs`, else ws.current) is what a flavor or restore pass left it stays as it is:
        the text was gated and its section scored with it in, and a regrowth from the run's original would drop it."""
        cur = (ws.current if runs is None else runs)[pi][ri]
        got = self.flavor_inserts(ws).get((pi, ri))
        return bool(cur) and got is not None and list(cur) == list(got[1])

    def arm_of(self, ws, pi, ri):
        """The writer arm that wrote the text standing in a run, from the bank's accepted changes: "fewshot" for a
        change banked before the arms existed, None for the original or a text the bank doesn't know."""
        cur = " ".join(ws.current[pi][ri])
        if not cur or ws.current[pi][ri] == ws.original[pi][ri]:
            return None
        for e in reversed(self.bank.of(ws.wid)):
            for c in e.get("changes") or []:
                if c.get("new") == cur:
                    return c.get("arm") or "fewshot"
        return None

    def rewrite_targets(self, ws):
        """rewrite_cleared = true (6.9, Phase 3): in a clean window (scored, no flagged segment, no Claude text
        left), every replaced Claude run the few-shot arm wrote gets one more regrowth by the adapter, judged against
        the text standing there (as good as it, and a win or tie in the head-to-head). Only in the window's first
        pass of the day, never a run the flavor pass locked, and only on the adapter arm (do_round reads
        ws.rewrites). With the flag false (the default) a clean window is left alone."""
        if "notes" not in self.arms or ws.passes_today > 1 or ws.ai is None or ws.n_flagged() or ws.claude_text_left():
            return []
        return [(pi, ri, False) for pi, p in enumerate(ws.runs) for ri, r in enumerate(p)
                if r["claude"] and ws.current[pi][ri] and ws.current[pi][ri] != ws.original[pi][ri]
                and not self.flavor_locked(ws, pi, ri) and self.arm_of(ws, pi, ri) != "notes"]

    def arm_order(self, is_last=False):
        """The arms a run tries, in order: the primary, then the fallback when the server has it (6.9, Phase 3)."""
        arms = getattr(self, "arms", ("fewshot",))
        primary_arm, fallback = getattr(self, "primary_arm", "fewshot"), getattr(self, "fallback_arm", None)
        primary = primary_arm if primary_arm in arms else "fewshot"
        out = [primary]
        if fallback and fallback in arms and fallback != primary:
            out.append(fallback)
        return out

    @staticmethod
    def adapter_prompt(notes, prefix_para):
        """The notes-to-text adapter's prompt (training/hip_notes.yaml): the paragraph's notes, then the paragraph so
        far, flush. No header, no worked examples, no earlier paragraphs: it was trained on one passage at a time,
        and the text before goes to the judge as it does now. No trailing space after the prefix: with one, the
        adapter ends the paragraph at once (9 of 12 draws empty against 0 of 12 without, 2026-11-09)."""
        return "<notes>\n" + "\n".join(f"- {n}" for n in notes) + "\n</notes>\n<target_text>\n" + prefix_para

    def run_args(self, ws, pi, ri, forced, g, is_last, ctx=None, arm=None):
        """Everything one run's draw and judge need. `ctx` (the whole-text round's context round) carries the
        section's text around the window: `before` paragraphs go in front of the writer's prompt and the judge's
        before-text, `after` text extends the judge's after-text, so the regrowth fits its neighbors. `arm` picks
        the writer (6.9, Phase 3): "fewshot" is the few-shot prompt below, "notes" the adapter's (adapter_prompt);
        the default is the loop's primary arm."""
        from engine.draws import SWEAR
        cur, orig = ws.current, ws.original
        arm = arm or self.arm_order()[0]
        author = bool(ws.runs[pi][ri].get("author"))
        mk = ws.must[(pi, ri)]
        # the settle rule and the starter ladder exist for Claude's watermark; your lines have none, so a line of
        # yours is never in a "last" round: the pick has to be as good as yours and win or tie the head-to-head.
        # The adapter arm never sees Claude's sentence either, so its draws are judged the same way (the few-shot
        # fallback settles a Claude run the adapter's pick couldn't, see do_round)
        if author or arm == "notes":
            is_last = False
        before_paras = [flat(cur[x]) for x in range(pi)]
        prefix_para = " ".join(s for rr in cur[pi][:ri] for s in rr)
        after = " ".join(s for rr in cur[pi][ri + 1:] for s in rr)
        if arm == "notes":
            prompt = self.adapter_prompt(ws.notes[pi], prefix_para)
        else:
            prev = [self.fmt_example(ws.notes[x], before_paras[x]) for x in range(pi)]
            ctx_prev = [self.fmt_example(n or ["(the paragraph before this one)"], t) for n, t in (ctx or {}).get("before", [])]
            examples = self.examples_without(" ".join(orig[pi][ri])) if author else self.examples_block
            prompt = "\n\n---\n\n".join([self.header, examples, *ctx_prev, *prev,
                                         self.fmt_example(ws.notes[pi], prefix_para)])
            if prefix_para:
                prompt += " "
        orig_text = " ".join(orig[pi][ri])
        # a forced re-roll is judged against Claude's original: the banked replacement broke a must-keep
        cur_text = orig_text if forced else " ".join(cur[pi][ri])
        pq = pattern_quotes(orig_text)
        swears_elsewhere = sum(len(SWEAR.findall(" ".join(cur[x][y]))) for x in range(len(cur))
                               for y in range(len(cur[x])) if (x, y) != (pi, ri))
        before_text = " ".join([t for _, t in (ctx or {}).get("before", [])] + before_paras + [prefix_para])
        after_text = after
        if ctx:
            nxt = flat(cur[pi + 1]) if pi + 1 < len(cur) else ctx.get("after", "")
            after_text = " ".join(" ".join(x for x in (after, nxt) if x).split()[:80])
        # the restatement check reads the run against the sentences either side of it in the paragraph
        prev_sent = next((rr[-1] for rr in reversed(cur[pi][:ri]) if rr), "")
        next_sent = next((rr[0] for rr in cur[pi][ri + 1:] if rr), "")
        checks = extra_checks_for(orig_text, mk["repeats"] if self.repeat_must else [], notes=ws.notes[pi],
                                  before=prev_sent, after=next_sent)
        # the adapter arm's checks (engine/checks.py, 2026-11-09), on every arm: Reddit furniture is out, and the
        # connective openers are capped over the paragraph the candidate would sit in
        checks.append(lambda c: "reddit tell" if reddit_tell(c) else None)
        checks.append(lambda c: connective_problem(c, prefix_para, after))
        rest = self.section_swear_rest(ws, pi, ri) if getattr(self, "swear_rates", None) else None
        if rest:
            checks.append(lambda c: section_swear_problem(c, orig_text, rest, self.swear_rates, self.swear_mult))
            checks.append(lambda c: section_filler_problem(c, orig_text, rest, getattr(self, 'filler_cap', None),
                                                           getattr(self, 'filler_words', None)))
        if author:
            words, nums = mk.get("protected") or [], set(mk.get("numbers") or [])
            checks.append(lambda c: "protected word" if protected_missing(c, words) else None)
            checks.append(lambda c: "lost a number" if not nums <= set(re.findall(r"\d+(?:[.,/]\d+)*", c)) else None)
        # the arm is in the key: both arms share one server (and one model string), so without it a queued few-shot
        # pool could be replayed as an adapter pick
        parts = [self.model, arm, prompt, orig_text, cur_text, str(is_last), str(forced)]
        if author or ctx:                       # keys of ordinary Claude runs stay as M1 made them (queued pools)
            parts += [str(author), after_text]
        key = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
        return {"key": key, "pi": pi, "ri": ri, "g": g, "is_last": is_last, "forced": forced, "prompt": prompt,
                "orig_text": orig_text, "cur_text": cur_text, "is_original": cur[pi][ri] == orig[pi][ri],
                "n_sent": len(orig[pi][ri]), "notes": self.judge_notes(ws, pi, mk), "author": author, "arm": arm,
                "opens_para": ri == 0 or not prefix_para,
                "protected": mk.get("protected") if author else None,
                "before": " ".join(before_text.split()[-60:]), "after": after_text,
                "swears_elsewhere": swears_elsewhere, "hedges": mk["hedges"] or None,
                "checks": checks, "pattern_quotes": pq}

    def judge_notes(self, ws, pi, mk):
        """The paragraph's notes as the judge sees them. With repeat_must_keep off, each deliberate repeat in the run
        becomes a joke line, so the judge weighs whether the repeat survived instead of the regex refusing every draw
        (2026-11-10: one paragraph whose joke repeats a word lost 1,627 draws to the repeat rule in one night). The writer's
        prompt still gets the plain notes."""
        notes = list(ws.notes[pi])
        if not self.repeat_must:
            for w, n in mk.get("repeats") or []:
                notes.append(f'Joke: the original says "{w}" {n} times in a row on purpose; a pick that keeps the'
                             f' repeat going keeps the joke')
        return notes

    def draw(self, ws, ra, k_offset=0):
        from engine.local_models import ArmServer
        from engine.draws import draw_pool, sents
        if hasattr(self.srv, "hint"):
            self.srv.hint(ra["orig_text"], ws.wid)
        pq = ra["pattern_quotes"]
        arm = ra.get("arm", "fewshot")
        srv = ArmServer(self.srv, arm)          # the arm's LoRA scale and stop strings on every completion
        # draws are free, so a llama-server hiccup just gets retried; one that won't clear goes up to main,
        # which restarts the day (and the server) from the bank (M1 review, finding 2)
        return self.retrying("the local model", lambda: draw_pool(srv=srv, op=self.op, tics=self.tics, prompt=ra["prompt"], orig_text=ra["orig_text"],
                         n_sent=ra["n_sent"], is_original=ra["is_original"] or ra["forced"], known_names=ws.names,
                         swears_elsewhere=ra["swears_elsewhere"], swear_budget=ws.swear_budget,
                         digit_ban=self.digit_ban, pi=ra["pi"], ri=ra["ri"], rnd=ra["g"],
                         rounds=ra["g"] if ra["is_last"] else ra["g"] + 1, extra_must=ra["hedges"],
                         extra_checks=ra["checks"], splice_long_quote=not pq,
                         normalize=(lambda c: splice_quotes(c, pq)) if pq else None, k_offset=k_offset,
                         next_sent=(sents(ra["after"]) or [""])[0] if ra.get("after") else "",
                         no_starter=arm == "notes"),
                             (requests.RequestException,))

    def judge(self, ws, ra, pool):
        from engine.draws import judge_pool
        if hasattr(self.claude_raw, "hint"):
            self.claude_raw.hint(ra["orig_text"], ws.wid)
        return judge_pool(pool, settings=self.settings, conn=self.conn, tics=self.tics, orig_text=ra["orig_text"],
                          cur_text=ra["cur_text"], is_original=ra["is_original"], notes=ra["notes"],
                          before=ra["before"], after=ra["after"], pi=ra["pi"], ri=ra["ri"], rnd=ra["g"],
                          rounds=ra["g"] if ra["is_last"] else ra["g"] + 1, role=ws.wid, claude=self.claude,
                          voice=ra["protected"] if ra["author"] else None,
                          opens_para=bool(ra.get("opens_para")) and ra.get("arm") == "notes",
                          description=getattr(getattr(self, "job", None), "description", None))

    def regrow(self, ws, pi, ri, forced, g, is_last, ctx=None, may_judge=None, arms=None):
        """One run through the writer arms (6.9, Phase 3). The primary arm's pool (queued, else drawn) goes to the
        judge; when it passes no draw, the fallback arm draws and is judged instead. In a settle round the adapter
        arm never settles (run_args), so when its pick didn't go in, the few-shot fallback draws and is judged with
        the settle rule, which is how a Claude run whose adapter picks keep losing still gets regrown. `may_judge()`
        says whether a Claude call may happen now (do_round's cap and smoke checks); when it says no, the arm's pool
        is queued under its key and the run stops at "cap". Returns (new_text_or_None, [one change per arm tried],
        "cap" or None). Each change carries its arm, so the bank and the analytics can tell the arms apart. `arms`
        narrows the arms tried (a rewrite_cleared regrowth runs on the adapter alone)."""
        changes, new, status = [], None, None
        for arm in (arms or self.arm_order(is_last)):
            ra = self.run_args(ws, pi, ri, forced, g, is_last, ctx=ctx, arm=arm)
            pool = self.queue.take(ra["key"])
            if pool is None:
                pool = self.draw(ws, ra)
            else:
                self.pools_used += 1
                self.pool_draws_used += pool["stats"].get("draws", 0)
            change = {"para": pi, "run": ri, "orig": ra["orig_text"], "forced": forced, "arm": arm,
                      "who": "yours" if ra["author"] else "claude",
                      "draws": pool["stats"].get("draws"), "tally": pool.get("tally")}    # for engine/analytics.py
            if ctx:
                change["context"] = True
            if not pool["shown"]:
                change.update({"result": f"no candidate passed ({pool['raw']} drawn: {pool['tally']})",
                               "examples": pool["rejected"]})
                changes.append(change)
                continue                                  # the fallback arm's turn
            if may_judge is not None and not may_judge():
                self.queue.put(ra["key"], pool, ts=self.clock.now())
                change["result"] = "queued for when the cap lifts"
                changes.append(change)
                return None, changes, "cap"
            try:
                self.budget.run_check(FALLBACK_USD)
                new, ch, _ = self.judge(ws, ra, pool)
                change.update({k: v for k, v in ch.items() if k != "para"})
                changes.append(change)
                if new or not (is_last and arm == "notes"):
                    break
                # the settle round, and the adapter's pick didn't go in: the fallback settles the run
            except ClaudeCapReached as e:
                self.on_cap(e)
                self.queue.put(ra["key"], pool, ts=self.clock.now())
                change["result"] = f"queued: {e.check} said no"
                changes.append(change)
                return None, changes, "cap"
            except ClaudeCallError as e:
                change["result"] = f"Claude call failed: {redact(e)[:120]}"
                changes.append(change)
                break
        return new, changes, status

    # ------------------------------------------------------------------ one round
    def do_round(self, ws):
        from engine.draws import sents
        is_last = ws.settle_next or ws.completed + 1 >= self.rounds
        g = ws.g + 1
        targets = self.targets(ws, is_last)
        if not targets:
            return "nothing"
        yours = [(pi, ri) for pi, ri, _ in targets if ws.runs[pi][ri].get("author")]
        if yours and not self.ensure_author(ws, yours):
            return "cap"                       # notes or protected words wait for Claude; nothing banked yet
        changes, interrupted, accepted, accepted_yours, fallbacks = [], None, 0, 0, 0
        for pi, ri, forced in targets:
            if self.must_stop():
                interrupted = "stop"
                break
            self.tick()
            ws.attempts[(pi, ri)] = ws.attempts.get((pi, ri), 0) + 1
            new, tried, status = self.regrow(ws, pi, ri, forced, g, is_last,
                                             may_judge=lambda: not (self.blocked() or (self.args.smoke and self.smoke_spent_out())),
                                             arms=["notes"] if (pi, ri) in getattr(ws, "rewrites", set()) else None)
            fallbacks += len(tried) > 1
            if status == "cap":
                interrupted = interrupted or "cap"
                ws.attempts[(pi, ri)] -= 1
            else:
                self.tiny_note(ws, pi, ri, accepted=bool(new))
            if new:
                if new != " ".join(ws.current[pi][ri]):
                    accepted += 1           # a pick identical to the standing text changes nothing
                    accepted_yours += bool(ws.runs[pi][ri].get("author"))
                ws.current[pi][ri] = sents(new) or [new]
                ws.forced.discard((pi, ri))
            for change in tried:
                ws.history.setdefault((pi, ri), []).append(change["result"])
            changes += tried
        text = window_text_of(ws.current, ws.heading)
        entry = {"kind": "round", "window": ws.wid, "day": self.day, "time": self.clock.stamp(), "pass": ws.pass_no,
                 "round": ws.completed + 1, "g": g, "last": is_last, "changes": changes, "text": text,
                 "arm": self.arm_order()[0], "fallbacks": fallbacks,
                 "runs": json.loads(json.dumps(ws.current)),
                 "replaced": sum(1 for pi, p in enumerate(ws.current) for ri, rr in enumerate(p)
                                 if rr != ws.original[pi][ri]),
                 "yours_replaced": [f"{pi}.{ri}" for pi, p in enumerate(ws.current) for ri, rr in enumerate(p)
                                    if ws.runs[pi][ri].get("author") and rr != ws.original[pi][ri]],
                 "interrupted": bool(interrupted)}
        changed = text != ws.last_text
        # the standing score only carries over when it belongs to this text (M1 review, finding 14): after a round
        # whose scoring a cap or a Pangram hold stopped, the text stands unscored and the next round scores it
        fresh = score_is_fresh(ws.ai, ws.scored_runs, ws.current)
        ai, segs, humanized = (ws.ai, ws.segs, False) if fresh and not changed else (None, None, False)
        if changed or not fresh:
            try:
                resp = self.score(text, f"{ws.wid} g{g}")
                s = summarize(resp)
                ai, segs, humanized = s["fraction_ai"], segs_of(resp), s["humanized"]
            except PangramCapReached as e:
                self.on_cap(e)
                interrupted = interrupted or "cap"
        if ai is not None:
            was_clean = ws.ai is not None and ws.n_flagged() == 0
            changed = changed or ai != ws.ai
            ws.ai, ws.segs, ws.flagged = ai, segs, flag_list(segs)
            ws.scored_runs = json.loads(json.dumps(ws.current))
            if was_clean and ws.n_flagged():
                self.alert(f"flip {ws.wid} g{g}", f"{ws.wid} passed and now flags: AI share {ai:.2f},"
                                                  f" {ws.n_flagged()} segment(s) flagged after round {g}.")
            if humanized:
                self.log(f"{ws.wid}: Pangram marked round {g} as humanized; it can't be the window's best (5.5)")
        entry.update({"ai": ai, "segs": segs, "humanized": humanized,
                      "flagged": sum(1 for _, f in segs) if segs else None, "changed": changed,
                      "attempts": {f"{a}.{b}": n for (a, b), n in ws.attempts.items()},
                      "settle_next": ws.settle_next})
        self.bank.add(entry)
        ws.last_text = text
        ws.g = g
        ws.rounds_today += 1
        self.worked_this_hour[ws.wid] = self.worked_this_hour.get(ws.wid, 0) + 1
        n_runs = len({(c["para"], c["run"]) for c in changes})
        n_yours = len({(c["para"], c["run"]) for c in changes if c.get("who") == "yours"})
        yours_note = f" ({accepted_yours} of {n_yours} lines of yours)" if n_yours else ""
        arms_note = f", fallback arm on {fallbacks}" if fallbacks else ""
        self.log(f"{ws.wid} pass {ws.pass_no} round {ws.completed + 1}{' (settle)' if is_last else ''}:"
                 f" {accepted} of {n_runs} runs replaced{yours_note}{arms_note}, AI share "
                 f"{'unscored' if ai is None else f'{ai:.2f}'}{', interrupted' if interrupted else ''}")
        if interrupted:
            return interrupted
        ws.completed += 1
        self.rounds_done += 1
        ws.changed.append(changed)
        ws.last_was_last = is_last
        return "done"

    # ------------------------------------------------------------------ one pass
    def demote(self, ws):
        """The must-keep audit (5.3's fix list; M1 review, finding 8). A replacement that loses a hedge or a repeat,
        or adds a sighting, fails the filter stack every new candidate has to pass, so it can't ship: the run goes
        back to Claude's original, which the next rounds re-roll like any other Claude text (with the must-keeps in
        the filters), and the window can't count as clean while it stands. s1w0's "again" said once is the case
        that started it."""
        ws.forced = set()
        bad = self.failing(ws, ws.current)
        for pi, ri in bad:
            whose = "your original line" if ws.runs[pi][ri].get("author") else "Claude's original"
            if not ws.current[pi][ri]:
                self.log(f"{ws.wid} p{pi + 1}.{ri + 1}: you put this cut back on Discord, so {whose} goes back in:"
                         f" {' '.join(ws.original[pi][ri])[:80]}")
            else:
                self.log(f"{ws.wid} p{pi + 1}.{ri + 1}: the banked replacement breaks a must-keep, so {whose}"
                         f" goes back in until a regrowth passes: {' '.join(ws.current[pi][ri])[:80]}")
            ws.current[pi][ri] = ws.original[pi][ri][:]
        return bad

    def failing(self, ws, runs):
        """The runs that can't stand in `runs`: failing_runs's must-keep breaks, plus a drop-rule cut you vetoed on
        Discord that's still cut there (plan 9.1). Every place that turns down a banked text for a must-keep turns
        it down for a vetoed cut too, so the cut never ships and a pass never starts from it."""
        bad = failing_runs(ws.must, runs, ws.original, repeats=self.repeat_must)
        for pi, ri in sorted(getattr(ws, "vetoed", ())):
            if pi < len(runs) and ri < len(runs[pi]) and not runs[pi][ri] and (pi, ri) not in bad:
                bad.append((pi, ri))
        return bad

    def base_entry(self, ws):
        """Where a pass starts: the best text that passes the must-keeps, else the best text of any kind, whose
        failing runs demote() then sends back to Claude's original (so s1w0 keeps its four good replacements)."""
        shape = [len(p) for p in ws.runs]
        return self.best(ws) or self.bank.best(ws.wid, ok=lambda e: [len(p) for p in e["runs"]] == shape)

    def best(self, ws):
        """The window's best banked text that still passes the must-keeps (and has this job's run shape). Only this
        one ships in the working copy."""
        shape = [len(p) for p in ws.runs]
        return self.bank.best(ws.wid, ok=lambda e: [len(p) for p in e["runs"]] == shape
                              and not self.failing(ws, e["runs"]))

    def restart_from_best(self, ws):
        """A new pass starts from the window's best banked text (6.7's success bank), not from wherever the last
        pass happened to end."""
        best = self.base_entry(ws)
        if not best:
            return
        runs = json.loads(json.dumps(best["runs"]))
        for pi, ri in self.failing(ws, runs):
            runs[pi][ri] = ws.original[pi][ri][:]
        self.apply_flavor(ws, runs)
        text = window_text_of(runs, ws.heading)
        if text == ws.last_text:
            return
        hit = self.latest_score(ws.wid, text)
        self.log(f"{ws.wid}: the new pass starts from the best banked text ({'unscored' if not hit else f'{hit[0]:.2f}'}),"
                 f" not where the last pass ended ({'unscored' if ws.ai is None else f'{ws.ai:.2f}'})")
        ws.current, ws.last_text = runs, text
        ws.ai, ws.segs = hit if hit else (None, [])
        ws.scored_runs = json.loads(json.dumps(ws.current)) if hit else None
        ws.flagged = flag_list(ws.segs)

    def work(self, ws):
        if not ws.pass_open:
            self.restart_from_best(ws)
            ws.pass_no += 1
            ws.pass_open, ws.completed, ws.changed, ws.settle_next, ws.attempts = True, 0, [], False, {}
            ws.passes_today += 1
            self.log(f"{ws.wid}: pass {ws.pass_no} starts (AI share {'unscored' if ws.ai is None else f'{ws.ai:.2f}'})")
        while True:
            if self.must_stop() or self.blocked():
                return "paused"
            if ws.completed >= self.rounds:
                return self.end_pass(ws, "rounds done")
            r = self.do_round(ws)
            if r == "nothing":
                return self.end_pass(ws, "nothing left to target")
            if r != "done":
                return "paused"
            if self.args.smoke:
                self.stop_reason = "the smoke test ran its round"
                return "paused"
            if ws.last_was_last:
                return self.end_pass(ws, "settle round done" if ws.settle_next else "rounds done")
            if stuck_rule(ws.changed, self.after_round, self.streak):
                if ws.claude_text_left():
                    ws.settle_next = True
                    self.log(f"{ws.wid}: the stuck rule fired with Claude text left, so one settle round runs first")
                else:
                    return self.end_pass(ws, "stuck")

    def at_target(self, ws):
        """The window's standing score is at or under target_ai_share with nothing flagged (question 9)."""
        return (self.target is not None and getattr(ws, "ai", None) is not None and ws.ai <= self.target
                and ws.n_flagged() == 0)

    def end_pass(self, ws, reason):
        self.drop_rule(ws)
        if ws.ai is not None and ws.n_flagged() == 0 and not ws.claude_text_left():
            state = "done" if self.at_target(ws) else "clean"
        elif self.at_target(ws) and self.rest_at_target:
            state = "done"          # under the target with Claude text left: it rests (the author's call, 2026-11-10)
        elif reason in ("stuck", "nothing left to target", "settle round done"):
            state = "stuck"
        else:
            state = "working"
        self.bank.add({"kind": "pass_end", "window": ws.wid, "day": self.day, "time": self.clock.stamp(),
                       "pass": ws.pass_no, "state": state, "reason": reason, "rounds": ws.completed, "ai": None,
                       "arm": self.arm_order()[0],
                       "best": (self.bank.best(ws.wid) or {}).get("ai")})
        ws.pass_open, ws.state = False, state
        # a clean window, or one with nothing left to try, rests until tomorrow; a stuck one comes back with fresh
        # seeds once every window has had its turn (8.1: the long day is for hammering stuck lines)
        ws.rested_today = state in ("clean", "done") or reason == "nothing left to target"
        self.log(f"{ws.wid}: pass {ws.pass_no} ends ({reason}) after {ws.completed} rounds, state {state}")
        return state

    def drop_rule(self, ws):
        """Plan 5.4: cut a Claude run that failed every route and carries no fact, if the judge passes the
        paragraph without it. At most one cut per paragraph per pass; every cut goes in the digest."""
        from engine.draws import names
        cut, keep = drop_eligible(ws.runs, ws.current, ws.original, ws.attempts, ws.completed,
                                  lambda s: ws.who.get(s), names)
        ws.couldnt_cut = keep
        # verdicts this pass already banked (finding 11): a resumed pass reads them instead of paying Sonnet again
        done_paras, said_no = drop_verdicts(self.bank, ws.wid, ws.pass_no)
        for pi, ri, _ in cut:
            if pi in done_paras or self.blocked() or self.must_stop():
                continue
            if (pi, ri) in said_no:
                ws.couldnt_cut.append((pi, ri, f"the judge wouldn't pass the paragraph without it: {said_no[(pi, ri)]}"))
                continue
            if (pi, ri) in getattr(ws, "vetoed", ()):   # you put this cut back on Discord (plan 9.1): it stays for good
                ws.couldnt_cut.append((pi, ri, "you said keep it going on Discord"))
                continue
            para = flat(ws.current[pi])
            trial = [r[:] for r in ws.current[pi]]
            trial[ri] = []
            cut_para = flat(trial)
            run_text = " ".join(ws.current[pi][ri])
            before = " ".join(" ".join(flat(ws.current[x]) for x in range(pi)).split()[-60:])
            prompt = (describe(DROP_TEMPLATE, getattr(getattr(self, "job", None), "description", None))
                      .replace("<<BEFORE>>", before or "(start of the section)").replace("<<PARA>>", para)
                      .replace("<<CUT>>", cut_para).replace("<<RUN>>", run_text)
                      .replace("<<NOTES>>", "\n".join(f"- {n}" for n in ws.notes[pi])))
            try:
                if hasattr(self.claude_raw, "hint"):
                    self.claude_raw.hint(run_text, ws.wid)
                out, _ = self.claude(self.settings, self.conn, prompt, role=f"{ws.wid} drop", model="sonnet",
                                     max_budget_usd=0.3)
            except ClaudeCapReached as e:
                self.on_cap(e)
                return
            except ClaudeCallError:
                continue
            m = re.search(r"\{.*\}", out, re.S)
            try:
                v = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                v = {}
            if v.get("pass") is True and not v.get("lost"):
                ws.current[pi] = trial
                done_paras.add(pi)
                text = window_text_of(ws.current, ws.heading)
                cut_entry = {"window": ws.wid, "para": pi, "run": ri, "text": run_text, "why": v.get("why", ""),
                             "time": self.clock.stamp()}
                self.cuts.append(cut_entry)
                self.log(f"{ws.wid} p{pi + 1}.{ri + 1}: the drop rule cut \"{run_text[:80]}\"")
                ai, segs = None, None
                try:
                    resp = self.score(text, f"{ws.wid} cut")
                    s = summarize(resp)
                    ai, segs = s["fraction_ai"], segs_of(resp)
                    ws.ai, ws.segs, ws.flagged = ai, segs, flag_list(segs)
                    ws.scored_runs = json.loads(json.dumps(ws.current))
                except PangramCapReached as e:
                    self.on_cap(e)
                self.bank.add({"kind": "cut", "window": ws.wid, "day": self.day, "time": self.clock.stamp(),
                               "pass": ws.pass_no, "g": ws.g, "para": pi, "run": ri, "cut_text": run_text,
                               "why": v.get("why", ""), "text": text, "runs": json.loads(json.dumps(ws.current)),
                               "ai": ai, "segs": segs})
                ws.last_text = text
                self.ask_cut(ws, cut_entry)       # the cut stands unless you answer 2 (plan 9.1; no bot token, no-op)
            else:
                ws.couldnt_cut.append((pi, ri, f"the judge wouldn't pass the paragraph without it: {v.get('why', '')}"))
                self.bank.add({"kind": "drop_no", "window": ws.wid, "day": self.day, "time": self.clock.stamp(),
                               "pass": ws.pass_no, "para": pi, "run": ri, "why": v.get("why", "")})

    # ------------------------------------------------------------------ local work while a cap holds
    def local_work(self):
        """Claude or Pangram is capped: draw pools for the runs each window's next round would try, until every
        pool holds pool_draws candidates. One pool at a time, so stops and hourly posts stay prompt."""
        for ws in self.order():
            if not ws.ready:
                continue
            is_last = ws.settle_next or ws.completed + 1 >= self.rounds
            for pi, ri, forced in self.targets(ws, is_last):
                if ws.runs[pi][ri].get("author") and not (ws.notes_ready and ws.must[(pi, ri)].get("asked")):
                    continue                     # a line of yours needs its notes and protected words from Claude
                ra = self.run_args(ws, pi, ri, forced, ws.g + 1, is_last)
                have = self.queue.get(ra["key"])
                drawn = have["stats"]["draws"] if have else 0
                if drawn >= self.pool_draws:
                    continue
                from engine.draws import SHOW
                self.queue.put(ra["key"], merge_pools(have, self.draw(ws, ra, k_offset=drawn), SHOW), ts=self.clock.now())
                return True
        return False

    # ------------------------------------------------------------------ the day
    def order(self):
        """Worst first: windows not yet worked today, then by AI share (unscored counts as worst)."""
        live = [w for w in self.windows if not w.retired and not w.rested_today
                and not (self.rest_at_target and self.at_target(w))]
        return sorted(live, key=lambda w: (w.passes_today if not w.pass_open else -1,
                                           -(w.ai if w.ai is not None else 1.01)))

    def run_day(self):
        """One run-day, or the rest of it: main calls this again after an error, and the bank picks up the work."""
        now = self.clock.now()
        first = not self.entered
        self.entered = True
        if first:
            # pinned once: a restart after midnight (a --now run started in the evening) is still the same run-day,
            # for its bank entries and for the run-day Pangram cap (M2)
            self.day = self.schedule.day_key(now)
            if not self.args.smoke:
                self.budget.run_day_start = now
        if not self.args.smoke:
            self.budget.day_started(self.day)
        if first:
            self.log(f"run-day {self.day} begins ({'rehearsal' if self.rehearsal else 'smoke test' if self.args.smoke else 'real'}),"
                     f" phase {self.schedule.phase(now)}")
        else:
            self.log(f"run-day {self.day} carries on after the error, from the bank (phase {self.schedule.phase(now)})")
        self.setup()
        self.check_model()                       # a model change nobody has acknowledged yet alerts every run-day
        pruned = self.queue.prune(now - 2 * 86400)
        if pruned:
            self.log(f"dropped {pruned} queued pools older than two days")
        if first and self.claude_raw and not self.args.smoke:
            try:
                self.claude(self.settings, self.conn, "Reply with exactly: ok", role="probe",
                            model=self.settings.config["claude"].get("probe_model", "haiku"), max_budget_usd=0.05)
            except ClaudeCapReached as e:
                self.on_cap(e)
            except ClaudeCallError as e:
                self.log(f"the start-up probe failed: {str(e)[:120]}")
        self.srv = self.srv_factory()
        self.srv.start()
        self.digit_ban = self.srv.digit_tokens()
        from engine.local_models import arms_available
        self.arms = arms_available(self.srv)
        if self.primary_arm not in self.arms:
            self.log(f"the server for {self.model} has no adapter, so the {self.primary_arm} arm is off and the few-shot"
                     " writer is the only arm today")
        self.op = self.op_factory()
        b = self.budget.snapshot()
        if first:
            self.day_post_id = self.notifier.send(f"**depangram** run-day {self.day} started: {len(self.windows)} windows, model {self.model},"
                               f" writer arms {' then '.join(self.arm_order())}."
                               f" Pangram ${b['pangram_today']:.2f} of ${b['pangram_day_cap']:.2f} today,"
                               f" your Claude week at {b['real_total']:.0%}.")
            self.note_day_post()
        self.read_answers(force=True)            # late answers to yesterday's questions, before any window starts
        for ws in self.order():                  # round 0 and the notes for every window first, so the order is real
            if self.must_stop() or self.blocked():
                break
            self.prepare(ws)
        while not self.must_stop():
            self.tick()
            self.read_answers()                  # before each pass: your Discord answers (plan 9.1; no token, no-op)
            if self.must_stop():
                break
            if self.blocked():
                # the window rounds have spent their part of the Pangram day: the reserve is the whole-text round's
                # (a round that couldn't start waits wait_minutes before it's tried again, so it can't spin)
                if self.pangram_block and self.pangram_block.check == WINDOW_SHARE and not self.claude_block \
                        and self.clock.now() >= self.whole_wait_until and self.whole_due():
                    self.whole_text_round("the window rounds' share of the Pangram day is spent")
                    continue
                if not self.local_work():
                    self.idle()
                continue
            ws = next((w for w in self.order() if w.ready or self.prepare(w)), None)
            if ws is None:
                if self.blocked():
                    continue
                if self.whole_due():
                    self.whole_text_round()
                self.stop_reason = "every window is done, stuck or resting for today"
                break
            self.work(ws)
        # the day's end (the schedule's wind-down, or a --now run's): one more whole-text round when any round
        # changed text after the last one. Not on the STOP file: that means stop now
        if (self.stop_reason or "").startswith(("wind-down", "a --now run's wind-down")) and self.whole_due():
            self.whole_text_round("the day's end")
        self.log(f"stopping: {self.stop_reason}")

    def stop_servers(self):
        """The llama server and Open Pangram down, so a restart of the day starts them clean."""
        for closer in (lambda: self.srv and self.srv.stop(), lambda: self.op and self.op.unload()):
            try:
                closer()
            except Exception as e:  # noqa: BLE001
                self.log(f"cleanup: {redact(e)}")
        self.srv = self.op = None

    def idle(self):
        """Nothing to do until a cap lifts or the day ends: sleep in short steps, checking the STOP file."""
        now = self.clock.now()
        ends = [x.until for x in (self.claude_block, self.pangram_block) if x and x.until]
        wake = min([now + self.wait] + ends)
        while self.clock.now() < wake and not self.must_stop():
            self.clock.sleep(min(60.0, wake - self.clock.now()))
            self.tick()
            self.read_answers()                  # a "stop" on Discord works while a cap holds, like the STOP file

    # ------------------------------------------------------------------ the whole-text round (M3, plan 5.6)
    def whole_sections(self):
        """Sections in scope whose every window is in this run: a --window run that leaves part of a section out
        can't score that section as posted."""
        have = {ws.wid for ws in self.windows}
        every = self.job.windows()
        return [n for n in self.job.sections
                if any(w["section"] == n for w in every) and all(w["wid"] in have for w in every if w["section"] == n)]

    def section_swear_rest(self, ws, pi, ri):
        """(words, swears, strong swears, filler tags) of ws's section as posted, with ws at the text it stands at now, the other
        loaded windows at what they ship and the rest at the working copy's, and run (pi, ri) left out. None when
        the section can't be assembled."""
        n = ws.w["section"]
        picks = {o.wid: o.current if o is ws else self.pick_runs(o) for o in self.windows
                 if o is ws or (o.w["section"] == n and getattr(o, "ready", False))}
        try:
            _, swaps = self.job.placeholder_base()
            md = self.job.assemble(self.picks_map(picks), write=False, keep_others=True)
            counts = section_counts(as_posted(restore_examples(md, swaps), [n], style=style_of(self.job)).get(n, ""),
                                    self.filler_words)
        except Exception:  # noqa: BLE001
            return None
        run = section_counts(" ".join(ws.current[pi][ri]), self.filler_words)
        return tuple(a - b for a, b in zip(counts, run))

    def ship_entry(self, ws):
        """What the working copy ships for a window: the window rule (window_ship: the newest whole-text pick, or
        the window's best when it's strictly better than what that pick stood on), unless the section rule says
        the section's best known as-posted picks (section_fallback). Plan 5.6, 2026-11-10 evening."""
        e = self.window_ship(ws)
        return self.section_fallback(ws, e) or e

    def window_ship(self, ws):
        """The window rule: the newest whole-text pick (a "ship" entry) when it was banked after the window's best
        and still passes the must-keeps, else the best. A window best banked after the pick overrides it only when
        it's strictly better (past the noise floor) than the window-level score the pick stood on; never on a tie.
        A tie overrode on 2026-11-10 and section 3 shipped 0.29 with a 0.00 version banked."""
        best = self.best(ws)
        shape = [len(p) for p in ws.runs]
        # a "fix" entry (engine/fixpass.py: few-word fixes on the shipped text) ships the same way, and so do a
        # "flavor" entry (engine/flavor.py: one sentence of the author's register into an adapter paragraph) and a
        # "restore" entry (engine/restore.py: a punch list applied as line edits)
        ships = [e for e in self.bank.of(ws.wid) if e.get("kind") in SHIP_KINDS and e.get("runs") is not None
                 and [len(p) for p in e["runs"]] == shape
                 and not self.failing(ws, e["runs"])]
        if not ships:
            return best
        ship = ships[-1]
        if best is None or self.bank.position(ship) > self.bank.position(best):
            return ship
        to_beat = self.ship_window_ai(ws, ship)
        if to_beat is not None and best["ai"] < to_beat - self.noise:
            return best
        return ship

    def ship_window_ai(self, ws, ship):
        """The window-level score a shipped pick stands on: the one stamped on it (`window_ai`), else the newest
        score banked for its text, else the window's best before it was banked (what the pick built on)."""
        if ship.get("window_ai") is not None:
            return ship["window_ai"]
        hit = self.latest_score(ws.wid, ship.get("text")) if ship.get("text") else None
        if hit:
            return hit[0]
        pos = self.bank.position(ship)
        prior = self.bank.best(ws.wid, ok=lambda e: self.bank.position(e) < pos)
        return prior["ai"] if prior else None

    def window_ai_now(self, ws, text):
        """What a pick banked now stands on at window level: its text's own score when the bank has one, else the
        window's best so far."""
        hit = self.latest_score(ws.wid, text)
        if hit:
            return hit[0]
        best = self.bank.best(ws.wid)
        return best["ai"] if best else None

    def section_fallback(self, ws, e):
        """The section rule: when the section's last as-posted scoring is worse than its best known by more than
        the noise floor, and this window's pick in it is what the window rule would ship now, ship the best known
        picks for this window instead (as a stand-in entry; the whole-text round banks the restore for real).
        None when the rule doesn't apply or the best known runs no longer fit the window."""
        n = (getattr(ws, "w", None) or {}).get("section")
        sb = section_best(self.bank, n, self.noise) if n is not None else None
        if not sb:
            return None
        st = section_standing(self.bank, n)
        if st is None or st["index"] == sb["index"] or st["ai"] <= sb["ai"] + self.noise:
            return None
        now = window_text_of(e["runs"], ws.heading) if e else window_text_of(ws.original, ws.heading)
        if st["picks"].get(ws.wid) != now or sb["picks"].get(ws.wid) in (None, now):
            return None
        runs = self.section_best_runs(ws, sb)
        if runs is None:
            return None
        return {"kind": "section_best", "window": ws.wid, "runs": runs, "ai": None, "text": sb["picks"][ws.wid],
                "section": n, "section_ai": sb["ai"], "section_time": sb["time"]}

    def section_best_runs(self, ws, sb):
        """The best known picks' run lists for a window, when they fit it today."""
        runs = runs_for(self.bank, sb, ws.wid, original=(window_text_of(ws.original, ws.heading), ws.original))
        if runs is None or [len(p) for p in runs] != [len(p) for p in ws.runs] \
                or self.failing(ws, runs):
            return None
        return json.loads(json.dumps(runs))

    def pick_runs(self, ws):
        e = self.ship_entry(ws)
        return json.loads(json.dumps(e["runs"])) if e else [[r[:] for r in p] for p in ws.original]

    def picks_map(self, picks):
        return {ws.wid: ([flat(p) for p in ws.original], [flat(p) for p in picks[ws.wid]])
                for ws in self.windows if ws.wid in picks}

    def whole_due(self):
        """A whole-text round is worth running: none has run for today's text yet. Text changed after the last one
        (a round or a cut moved a window's shipped text) means it's due again."""
        if self.args.smoke or not self.whole_sections():
            return False
        st = getattr(self, "model_status", None)
        if st and st.get("stale") and st.get("changed_at"):
            # Pangram's model changed (plan 9): due until every section has been scored since (a capped round banks
            # its whole_end too, so the scorings are what count)
            since = self.clock.stamp(st["changed_at"])
            done = {str(n) for e in self.bank.entries if e.get("kind") == "whole" and (e.get("time") or "") >= since
                    for n in (e.get("sections") or {})}
            if not {str(n) for n in self.whole_sections()} <= done:
                return True
        if any(not ws.ready for ws in self.windows):
            return True
        now = {ws.wid: window_text_of(self.pick_runs(ws), ws.heading) for ws in self.windows}
        last = self.whole_snapshot
        if last is None:
            ent = [e for e in self.bank.entries if e.get("kind") == "whole_end" and e.get("picks")]
            last = ent[-1]["picks"] if ent else None
        return last is None or any(last.get(w) != t for w, t in now.items())

    def whole_targets(self, sec_wss, picks, scores):
        """{wid: [(pi, ri)]}: every run of Claude's or yours that sits in a segment Pangram flags in the section as
        posted. Locked runs (headings, templates) and example lines map to nothing."""
        out = {}
        for n, wss in sec_wss.items():
            fl = flagged_segments(scores[n]["segs"])
            for ws in wss:
                for pi, p in enumerate(ws.runs):
                    for ri, r in enumerate(p):
                        cur = picks[ws.wid][pi][ri]
                        if (r["claude"] or r.get("author")) and cur and any(norm(s)[:40] in f for s in cur for f in fl) \
                                and not (self is not None and self.flavor_locked(ws, pi, ri, runs=picks[ws.wid])) \
                                and not (self is not None and self.tiny_skip(ws, pi, ri)):
                            out.setdefault(ws.wid, []).append((pi, ri))
        return out

    def section_context(self, ws, picks, wss):
        """The section's text around a window, for its context round: the paragraph before the window (with its
        notes when it has some) and the paragraph after it."""
        i = wss.index(ws)
        ctx = {"before": [], "after": ""}
        if i > 0:
            prev = wss[i - 1]
            notes = prev.notes[-1] if getattr(prev, "notes_ready", False) and prev.notes else None
            ctx["before"].append((notes, flat(picks[prev.wid][-1])))
        if i + 1 < len(wss):
            ctx["after"] = flat(picks[wss[i + 1].wid][0])
        return ctx

    def context_round(self, ws, base_runs, targets, ctx, k):
        """One round on the runs the section's as-posted score flags, drawn and judged with the section's text
        around the window (run_args' `ctx`). Runs as an ordinary round would (no settle round), never scores the
        window alone (the section score decides), and banks a "context" entry. Returns (runs, changes, why)."""
        from engine.draws import sents
        saved = (ws.current, ws.scored_runs)
        ws.current = json.loads(json.dumps(base_runs))
        yours = [(pi, ri) for pi, ri in targets if ws.runs[pi][ri].get("author")]
        if yours and not self.ensure_author(ws, yours):
            ws.current, ws.scored_runs = saved
            return base_runs, [], "cap"
        g = ws.g + 1
        changes, why = [], None
        for pi, ri in targets:
            if self.hard_stop():
                why = "stop"
                break
            if self.claude_block and self.clock.now() < (self.claude_block.until or 0):
                why = "cap"
                break
            self.tick()
            new, tried, status = self.regrow(ws, pi, ri, False, g, False, ctx=ctx)
            if status != "cap":
                self.tiny_note(ws, pi, ri, accepted=bool(new))
            if new:
                ws.current[pi][ri] = sents(new) or [new]
            for change in tried:
                ws.history.setdefault((pi, ri), []).append(change["result"])
            changes += tried
            if status == "cap":
                why = "cap"
            if why:
                break
        new_runs = ws.current
        ws.current, ws.scored_runs = saved
        ws.g = g
        self.bank.add({"kind": "context", "window": ws.wid, "day": self.day, "time": self.clock.stamp(), "g": g,
                       "whole": k, "changes": changes, "text": window_text_of(new_runs, ws.heading),
                       "arm": self.arm_order()[0], "runs": json.loads(json.dumps(new_runs)), "ai": None})
        n_new = sum(1 for c in changes if c.get("new"))
        n_runs = len({(c["para"], c["run"]) for c in changes})
        self.log(f"{ws.wid} context round (whole-text pass {k + 1}): {n_new} of {n_runs} flagged runs regrown"
                 f" in the section's context")
        return new_runs, changes, why

    def score_section(self, n, picks, label):
        """One section as posted, scored as one document. None when Pangram is capped (the cap is logged)."""
        _, swaps = self.job.placeholder_base()
        text = as_posted(restore_examples(self.job.assemble(self.picks_map(picks), write=False), swaps), [n],
                         style=style_of(self.job))[n]
        try:
            resp = self.score(text, label, whole=True)
        except PangramCapReached as e:
            self.on_cap(e)
            return None
        s = summarize(resp)
        return {"ai": s["fraction_ai"], "segs": segs_of(resp), "humanized": s["humanized"], "text": text,
                "flagged": sum(1 for _, f in segs_of(resp) if f)}

    def whole_text_round(self, why="every window is done, stuck or resting"):
        """Plan 5.6. Assemble each window's shipped text and score every section in scope as posted (examples in,
        comments and marks out). Then, section by section, worst first: the runs Pangram flags there (yours
        included) get a context round, the section is scored again, and it keeps the new text only when its
        as-posted score falls (otherwise its windows go back to what they had). Sections are scored as separate
        documents, so one section's change can't move another's score, and a cap part-way loses nothing already
        settled. It stops when nothing is flagged that a run can answer for, after two passes in a row where no
        section got better, after whole_text_rounds passes counting the first scoring, or at a cap or the stop.
        Unchanged text costs nothing to rescore (the cache)."""
        secs = self.whole_sections()
        if not secs or self.args.smoke:
            return
        for ws in self.windows:
            if ws.w["section"] not in secs or ws.ready:
                continue
            # a window readied for the whole-text round (after a restart, say) may score out of its reserve
            self.whole_mode = True
            try:
                ok = self.prepare(ws)
            finally:
                self.whole_mode = False
            if not ok:
                self.whole_wait_until = self.clock.now() + self.wait
                self.log(f"the whole-text round waits: {ws.wid} couldn't get ready (a cap); it's tried again in"
                         f" {self.wait / 60:.0f} minutes")
                return False
        self.log(f"the whole-text round starts ({why}): sections {', '.join(map(str, secs))} as posted")
        sec_wss = {n: [ws for ws in self.windows if ws.w["section"] == n] for n in secs}
        picks = {ws.wid: self.pick_runs(ws) for n in secs for ws in sec_wss[n]}
        scores, capped = {}, False
        for n in secs:
            got = self.score_section(n, picks, f"whole s{n} pass 1")
            if got is None:
                capped = True
                break
            scores[n] = got

        def bank_scoring(k, ns, improved, restored=()):
            self.as_posted_now = dict(scores)
            self.whole_log.append({"time": self.clock.stamp(), "k": k, "why": why, "improved": improved,
                                   "restored": list(restored), "sections": {n: scores[n]["ai"] for n in ns}})
            self.bank.add({"kind": "whole", "day": self.day, "time": self.clock.stamp(), "k": k, "why": why,
                           "arm": self.arm_order()[0],
                           "sections": {str(n): {"ai": scores[n]["ai"], "flagged": scores[n]["flagged"],
                                                 "segs": scores[n]["segs"]} for n in ns}, "improved": improved,
                           "restored": list(restored)})
        if scores:
            if self.as_posted_first is None:
                self.as_posted_first = {n: s["ai"] for n, s in scores.items()}
            bank_scoring(1, list(scores), [])
            self.log("as posted: " + ", ".join(f"section {n} {s['ai']:.2f} ({s['flagged']} flagged)"
                                               for n, s in scores.items()))
            restored = self.restore_section_bests(sec_wss, picks, scores)
            if restored:
                bank_scoring(1, restored, [], restored=restored)
        quiet, stop = 0, "Pangram is capped" if capped else None
        arms_used = {}                 # wid -> the arms whose picks this round accepted (for the ship entries)
        for k in range(2, max(1, self.whole_rounds) + 1):
            if stop:
                break
            improved, any_targets = [], False
            for n in sorted(scores, key=lambda x: -scores[x]["ai"]):
                targets = self.whole_targets({n: sec_wss[n]}, picks, scores)
                if not targets:
                    continue
                any_targets = True
                if self.hard_stop():
                    stop = "the day's stop"
                    break
                new_picks = json.loads(json.dumps(picks))
                for ws in sec_wss[n]:
                    if ws.wid in targets and not stop:
                        runs, chs, why_stop = self.context_round(ws, picks[ws.wid], targets[ws.wid],
                                                                 self.section_context(ws, picks, sec_wss[n]), k - 1)
                        arms_used.setdefault(ws.wid, set()).update(c["arm"] for c in chs if c.get("new") and c.get("arm"))
                        new_picks[ws.wid] = runs
                        stop = {"cap": "a Claude cap", "stop": "the day's stop"}.get(why_stop)
                if all(new_picks[ws.wid] == picks[ws.wid] for ws in sec_wss[n]):
                    continue
                got = self.score_section(n, new_picks, f"whole s{n} pass {k}")
                if got is None:
                    stop = "Pangram is capped"
                    break
                old = scores[n]
                if got["ai"] < old["ai"] - 1e-9 and not got["humanized"]:
                    for ws in sec_wss[n]:
                        picks[ws.wid] = new_picks[ws.wid]
                    scores[n] = got
                    improved.append(n)
                    self.log(f"section {n} as posted: {old['ai']:.2f} -> {got['ai']:.2f}; the context round's text stays")
                else:
                    self.log(f"section {n} as posted: {old['ai']:.2f} -> {got['ai']:.2f}, no better; its windows go back"
                             " to what they had")
                bank_scoring(k, [n], [n] if n in improved else [])
            if not any_targets:
                self.log("the whole-text round holds: nothing flagged that a run can answer for")
                break
            quiet = 0 if improved else quiet + 1
            if quiet >= 2:
                self.log("the whole-text round ends: two passes in a row with no section better")
                break
        if stop:
            self.log(f"the whole-text round stops early: {stop}")
        # what each window now ships, when the whole-text round changed it
        for n in secs:
            for ws in sec_wss[n]:
                if picks[ws.wid] != self.pick_runs(ws):
                    text = window_text_of(picks[ws.wid], ws.heading)
                    self.bank.add({"kind": "ship", "window": ws.wid, "day": self.day, "time": self.clock.stamp(),
                                   "arm": self.arm_order()[0], "arms": sorted(arms_used.get(ws.wid, ())),
                                   "text": text, "runs": json.loads(json.dumps(picks[ws.wid])), "ai": None,
                                   "section": n, "section_ai": scores.get(n, {}).get("ai"),
                                   "window_ai": self.window_ai_now(ws, text), "why": "whole-text round"})
        self.whole_snapshot = {ws.wid: window_text_of(self.pick_runs(ws), ws.heading) for ws in self.windows}
        # the standing scores and the run lists go on the whole_end too, so the section-best memory
        # (engine/section_best.py) reads a round off one entry
        self.bank.add({"kind": "whole_end", "day": self.day, "time": self.clock.stamp(),
                       "picks": dict(self.whole_snapshot),
                       "sections": {str(n): {"ai": s["ai"], "flagged": s["flagged"], "segs": s["segs"]}
                                    for n, s in scores.items()},
                       "runs": {wid: json.loads(json.dumps(r)) for wid, r in picks.items()}})
        self.release_reserve()
        return True

    def restore_section_bests(self, sec_wss, picks, scores):
        """Plan 5.6's section-best rule, at the start of a whole-text round: a section whose picks differ from its
        best known as-posted picks has just been scored as it stands; when that score is worse than the best known
        by more than the noise floor, the best known picks come back for its windows (a "ship" entry each, so the
        working copy and the write-back take them), at no Pangram cost, and the round goes on from the banked
        score. Returns the sections restored."""
        restored = []
        for n in list(scores):
            sb = section_best(self.bank, n, self.noise)
            if not sb:
                continue
            wss = sec_wss[n]
            if all(sb["picks"].get(ws.wid) == window_text_of(picks[ws.wid], ws.heading) for ws in wss):
                continue
            if scores[n]["ai"] <= sb["ai"] + self.noise:
                continue
            runs = {ws.wid: self.section_best_runs(ws, sb) for ws in wss}
            if any(r is None for r in runs.values()):
                self.log(f"section {n}: {scores[n]['ai']:.2f} as it stands, {sb['ai']:.2f} best known, but the best"
                         f" known picks no longer fit {', '.join(w for w, r in runs.items() if r is None)}")
                continue
            for ws in wss:
                if runs[ws.wid] == picks[ws.wid]:
                    continue
                picks[ws.wid] = runs[ws.wid]
                text = window_text_of(runs[ws.wid], ws.heading)
                self.bank.add({"kind": "ship", "window": ws.wid, "day": self.day, "time": self.clock.stamp(),
                               "arm": self.arm_order()[0], "arms": [], "text": text,
                               "runs": json.loads(json.dumps(runs[ws.wid])), "ai": None, "section": n,
                               "section_ai": sb["ai"], "window_ai": self.window_ai_now(ws, text),
                               "why": f"section best restored (scored {sb['ai']:.2f} on {sb['time']})"})
            _, swaps = self.job.placeholder_base()
            text = as_posted(restore_examples(self.job.assemble(self.picks_map(picks), write=False), swaps), [n],
                             style=style_of(self.job))[n]
            standing = scores[n]["ai"]
            scores[n] = {"ai": sb["ai"], "segs": sb["segs"], "humanized": False, "text": text,
                         "flagged": sb["flagged"] if sb["flagged"] is not None else sum(1 for _, f in sb["segs"] if f)}
            self.log(f"section {n}: {standing:.2f} as it stands, {sb['ai']:.2f} best known, restored")
            restored.append(n)
        return restored

    def release_reserve(self):
        """Once the day's whole-text round is done, what it didn't spend of whole_text_reserve_usd goes back to the
        window rounds, down to whole_text_floor_usd (one section's scoring, for a day-end round when a later round
        changes text). The window-share hold lifts with it. Runs once a day: after the first release the reserve
        is the floor already."""
        if self.reserve <= self.floor + 1e-9:
            return
        cap = float(self.settings.config["pangram"]["nightly_cap_usd"])
        spent = max(self.budget.pangram_today(), self.budget.pangram_run_day() or 0.0)
        released = max(0.0, (cap - self.floor) - max(spent, cap - self.reserve))
        self.reserve = self.floor
        if released < 0.005:
            self.log("whole-text round done; the reserve is spent, nothing to release to the window rounds")
            return
        if self.pangram_block and self.pangram_block.check == WINDOW_SHARE:
            self.pangram_block = None
        self.log(f"whole-text round done; ${released:.2f} of the reserve released to the window rounds"
                 + (f" (${self.floor:.2f} kept for a day-end round)" if self.floor else ""))

    # ------------------------------------------------------------------ the wrap-up (plan 5.6)
    def wrap_up(self):
        """At the stop: the sections back into the real guide (prose blocks only; write_back in config.toml, else a
        dry run that only reports), then the final as-posted scan of what a reader would now see."""
        secs = self.whole_sections()
        if not secs or self.args.smoke or self.real_file is None or not self.real_file.exists():
            return
        from engine.wrapup import paragraph_history, run_versions
        paras = []
        history = paragraph_history(self.bank)
        for ws in self.windows:
            if ws.w["section"] in secs and not ws.ready:
                try:
                    self.prepare(ws)               # a window today never reached still ships its best
                except Exception as e:  # noqa: BLE001
                    self.log(f"{ws.wid}: not ready for the write-back ({redact(e)[:80]})")
            if ws.w["section"] not in secs or not ws.ready:
                continue
            pick = self.pick_runs(ws)
            versions = run_versions(self.bank, ws.wid, ws.original)
            for pi, p in enumerate(ws.runs):
                if any(r["claude"] or r.get("author") for r in p):
                    paras.append((f"{ws.wid} p{pi + 1}", versions[pi], norm(flat(pick[pi])),
                                  history.get(norm(flat(ws.original[pi])), set())))
        raw = self.real_file.read_bytes().decode("utf-8")       # read right before writing: you may be editing it
        new, report = write_back(raw, secs, paras, style=style_of(self.job))
        report["written"] = bool(self.write_back_on and new != raw)
        if report["written"]:
            self.real_file.write_bytes(new.encode("utf-8"))
        self.writeback_report = report
        self.log(f"write-back {'done' if report['written'] else 'dry run' if new != raw else 'not needed'}:"
                 f" {len(report['replaced'])} paragraph(s) replaced in {self.real_file.name}")
        if report["written"] or new == raw:
            texts = as_posted(new, secs, style=style_of(self.job))     # what a reader sees in the document now
        else:                                      # a dry run: the loop's own text, as it would be posted
            _, swaps = self.job.placeholder_base()
            picks = {ws.wid: self.pick_runs(ws) for ws in self.windows if ws.ready}
            texts = as_posted(restore_examples(self.job.assemble(self.picks_map(picks), write=False), swaps), secs,
                              style=style_of(self.job))
        self.final_scan = {}
        for n in secs:
            try:
                resp = self.score(texts[n], f"as posted s{n}", whole=True)
            except PangramCapReached as e:
                self.on_cap(e)
                self.final_scan[n] = None
                continue
            self.final_scan[n] = summarize(resp)["fraction_ai"]
        self.log("final as-posted scan: " + ", ".join(
            f"section {n} {'not scanned (cap)' if v is None else f'{v:.2f}'}" for n, v in self.final_scan.items()))
        # plan 9 (2026-11-10): the two numbers for the author. The sections combined costs nothing (the final scan's
        # scores, weighted by each section's as-posted words); the guide as it stands is one Pangram call on the
        # whole file as a reader would paste it today, after the final scan, from the day cap. Neither is a target.
        from engine.guide_scan import combined, run_for_loop as guide_scan_for_loop
        self.sections_combined = combined(self.final_scan, {n: len(t.split()) for n, t in texts.items()})
        if self.sections_combined["ai"] is not None:
            self.log(f"sections {', '.join(map(str, self.sections_combined['sections']))} combined:"
                     f" {self.sections_combined['ai']:.2f} (weighted by words)")
        try:
            guide_scan_for_loop(self, new)
        except Exception as e:  # noqa: BLE001
            self.log(f"the guide as it stands wasn't scored: {redact(e)[:160]}")
            self.guide_scan = {"skipped": f"an error ({redact(e)[:120]})", "ai": None}

    # ------------------------------------------------------------------ the end of the day
    def finish(self, crashed=None):
        self.stop_servers()
        # at the digest (plan 9.1): your last answers first, so a vetoed cut is back in before the write-back and
        # the report, then a question for any of today's cuts that didn't get one. No bot token, no-op
        self.read_answers(force=True)
        self.ask_open_cuts()
        try:
            self.write_working_copy()
        except Exception as e:  # noqa: BLE001
            self.log(f"working copy not written: {e}")
        if not crashed:
            try:
                self.wrap_up()
            except Exception as e:  # noqa: BLE001
                self.log(f"the wrap-up didn't finish: {redact(e)[:160]}")
            # the restore pass (engine/restore.py, plan 6.9), after the write-back and before the reflection, behind
            # restore.run_at_wrapup (default false); an error here can't take the digest with it
            try:
                from engine.restore import run_for_loop as restore_for_loop
                restore_for_loop(self)
            except Exception as e:  # noqa: BLE001
                self.log(f"the restore pass didn't run: {redact(e)[:160]}")
                self.restore_result = {"skipped": f"an error ({redact(e)[:120]})"}
        else:
            self.restore_result = {"skipped": "the day crashed"}
        now = self.clock.now()
        b = self.schedule.bounds(now)
        if not self.args.smoke and not crashed:
            rates = self.budget.measure_rates(getattr(self, "day", self.schedule.day_key(now)), b["start"], b["claude_until"])
            if rates:
                self.log("conversion rate refit from the night: " + ", ".join(f"{k} {v:.4f}" for k, v in rates.items()))
        if not crashed:
            # the reflection for the author (plan 9, 6.8): after the day's numbers are final, before the digest goes.
            # It runs inside finish's finally, so an error here can't take the digest with it
            try:
                from engine.reflect import run_for_loop
                run_for_loop(self)
            except Exception as e:  # noqa: BLE001
                self.log(f"the reflection didn't run: {redact(e)[:160]}")
                self.reflection = {"skipped": f"an error ({redact(e)[:120]})"}
        else:
            self.reflection = {"skipped": "the day crashed, so the sections couldn't be read whole"}
        report = discord.digest_report(self, crashed=crashed)
        card = discord.digest_card(self, crashed=crashed)
        if not self.args.smoke and not crashed:
            self.budget.day_finished(getattr(self, "day", self.schedule.day_key(now)))     # after the digest reads the backstop
        tag = " (rehearsal)" if self.rehearsal else " (smoke)" if self.args.smoke else ""
        name = f"{getattr(self, 'day', 'day')} digest{tag}.md"
        ddir = Path(self.args.digests_dir)
        ddir.mkdir(parents=True, exist_ok=True)
        (ddir / name).write_text(report, encoding="utf-8")
        if not self.args.window:               # a --window or --smoke run only sees some windows: the report stays whole
            self.job.couldnt_crack.write_text(discord.couldnt_crack_report(self), encoding="utf-8")
        self.notifier.send(card, file_name=name, file_text=report)
        # the morning submission (engine/morning.py): the write-back committed, published to main and posted with its
        # own digest. Skipped in a rehearsal, a smoke test, a --window run or a crashed day; an error is only logged
        try:
            from engine.morning import run_for_loop as morning_for_loop
            morning_for_loop(self, crashed)
        except Exception as e:  # noqa: BLE001
            self.log(f"the morning submission didn't run: {redact(e)[:160]}")
            self.morning_result = {"skipped": f"an error ({redact(e)[:120]})"}

    def write_working_copy(self):
        """Each window's shipped text (ship_entry: the whole-text round's pick, else the best) into the working
        copy."""
        best = {}
        for ws in self.windows:
            if not getattr(ws, "original", None):
                continue
            e = self.ship_entry(ws) if getattr(ws, "must", None) is not None else None
            if e:
                best[ws.wid] = ([flat(p) for p in ws.original], [flat(p) for p in e["runs"]])
        if best:
            # a --window or --smoke run only knows some windows: the rest keep what the working copy has
            self.job.assemble(best, keep_others=bool(self.args.window))


# ---------------------------------------------------------------------- the notifier
class Notifier:
    """Discord, or a log file. Rehearsal posts carry a [rehearsal] prefix and go to the log unless --discord-live.
    send() returns the message id on the bot path (plan 9.1), else None."""

    def __init__(self, settings, clock, log_path, rehearsal=False, live=True):
        self.settings, self.clock, self.log_path = settings, clock, Path(log_path)
        self.rehearsal, self.live = rehearsal, live
        self.sent = []

    def send(self, content, file_name=None, file_text=None):
        if self.rehearsal:
            content = "[rehearsal] " + content
        self.sent.append((self.clock.stamp(), content, file_name))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"===== {self.clock.stamp()}{' + ' + file_name if file_name else ''}\n{content}\n")
        if self.live:
            # a requests error's text carries the webhook URL, which is the token: print the class only (finding 12)
            try:
                got = discord.post(self.settings, content, file_name=file_name, file_text=file_text)
                if not got:
                    print("discord post failed: the webhook didn't accept it", flush=True)
                # the bot path (plan 9.1) hands back the message id, so a post can be answered; the webhook, True
                return got if isinstance(got, str) else None
            except Exception as e:  # noqa: BLE001
                print(f"discord post failed: {type(e).__name__}", flush=True)
        return None


# ---------------------------------------------------------------------- main
def real_parts(settings):
    from engine.local_models import LlamaServer
    from engine.open_pangram import OpenPangram
    model = settings.config["loop"]["model"]
    return (lambda: LlamaServer(model, settings=settings)), (lambda: OpenPangram("roberta")), claude_call_run


def run_with_restarts(loop, job, schedule, clock, stop_file, a):
    """The day, re-entered after an error (M1 review, finding 2). Only the STOP file, the schedule (or a --now run's
    stop time) or failures_in_row restarts with no round finished between them end it; the bank carries the work
    across a restart. Returns the crash text, or None."""
    lc = loop.settings.config["loop"]
    limit = int(lc.get("failures_in_row", 5))
    failures, rounds_at_failure = 0, None
    while True:
        try:
            job.setup()
            loop.run_day()
            # the run in flight has finished; wait out the wind-down so the digest goes at the stop time
            if not a.now and not (loop.stop_reason or "").startswith("the STOP") \
                    and schedule.phase(clock.now()) == "wind-down":
                stop_at = schedule.bounds(clock.now())["stop"]
                while clock.now() < stop_at and not Path(stop_file).exists():
                    clock.sleep(min(60.0, stop_at - clock.now()))
            return None
        except Exception as e:  # noqa: BLE001
            err = redact(f"{type(e).__name__}: {e}")
            print(redact(traceback.format_exc()), flush=True)
            loop.stop_servers()
            failures = failures + 1 if loop.rounds_done == rounds_at_failure else 1
            rounds_at_failure = loop.rounds_done
            if failures >= limit:
                loop.log(f"CRASH {err} ({failures} restarts in a row with no round finished)")
                loop.alert("crash", f"the loop failed {failures} times in a row and stops for today, at"
                                    f" {clock.stamp()}: {err[:300]}. The bank has every finished round; the next"
                                    " start resumes from it.")
                return err
            if loop.must_stop():
                loop.log(f"ERROR {err}; the day is over anyway ({loop.stop_reason})")
                return None
            loop.log(f"ERROR {err}; the day carries on from the bank in {loop.error_block / 60:.0f} minutes"
                     f" (failure {failures} of {limit} in a row)")
            loop.alert(f"restart {loop.schedule.day_key(clock.now())}",
                       f"the loop hit an error at {clock.stamp()} ({err[:200]}) and carries on from the bank in"
                       f" {loop.error_block / 60:.0f} minutes. It stops for the day after {limit} failures in a row.")
            wake = clock.now() + loop.error_block
            while clock.now() < wake and not loop.must_stop():
                clock.sleep(min(60.0, wake - clock.now()))
                loop.tick()
            if loop.must_stop():
                return None


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="depangram run loop (M1)")
    ap.add_argument("--job", help="the job, a folder in jobs/ (default: [job] default in config.toml)")
    ap.add_argument("--rehearse", action="store_true", help="fake Pangram, Sonnet and the llama server; zero spend")
    ap.add_argument("--scenario", default="default", help="rehearsal scenario (engine/rehearsal.py SCENARIOS)")
    ap.add_argument("--keep", action="store_true", help="rehearsal: keep the last rehearsal's bank and ledger (resume)")
    ap.add_argument("--clock", help="start the clock here ('02:00' or '2026-11-09 01:55'); the rehearsal's default 01:58")
    ap.add_argument("--speed", type=float, default=None, help="real seconds run this many times faster (with --clock)")
    ap.add_argument("--discord-live", action="store_true", help="rehearsal: post to Discord too, with the prefix")
    ap.add_argument("--smoke", action="store_true", help="one window, one round, real models, small limits")
    ap.add_argument("--smoke-pangram-usd", type=float, help="--smoke: this Pangram limit instead of config's")
    ap.add_argument("--smoke-claude-usd", type=float, help="--smoke: this Claude limit instead of config's")
    ap.add_argument("--now", action="store_true", help="ignore the schedule (the smoke test)")
    ap.add_argument("--window", help="only these windows, comma-separated (s1w0,s3w0); --smoke's default is the"
                                     " job's first window")
    ap.add_argument("--rounds", type=int, help="rounds per pass, instead of config's rounds_per_window")
    a = ap.parse_args(argv)
    if (a.clock or a.speed is not None) and not a.rehearse:
        # a compressed clock never sleeps, so on a real run caps would lift in seconds and ledger rows would carry
        # the wrong day (M1 review, finding 3)
        ap.error("--clock and --speed only work with --rehearse")
    if a.smoke:
        a.now = True

    base = Settings()
    a.job = job_name(a.job, base.config)
    if a.rehearse:
        from engine import rehearsal
        rdir = base.runs_dir / "rehearsal"
        env = rehearsal.prepare_dir(base, rdir, a.job, keep=a.keep, live=a.discord_live)
        settings = Settings(env=env, config=base.config)
        sc = rehearsal.SCENARIOS[a.scenario]
        start = parse_clock(a.clock) if a.clock else rehearsal.default_start(sc)
        clock = CompressedClock(start, speed=a.speed if a.speed is not None else 0.0)
        conn = db.connect(settings.db_path)
        fakes = rehearsal.Fakes(clock, conn, sc, rdir)
        clock.on_tick = fakes.events
        job = Job(a.job, root=rdir / "jobs" / a.job)
        if job.real_file is not None:
            job.real_file = rdir / job.real_file.name          # the rehearsal's copy of the real guide
        pg = LoopPangram(settings, conn, clock, session=fakes.pangram_http)
        srv_factory, op_factory, claude_raw = fakes.server, fakes.open_pangram, fakes.claude
        stop_file = rdir / "STOP"
        a.digests_dir = str(rdir / "digests")
        notifier = Notifier(settings, clock, rdir / "discord.log", rehearsal=True, live=a.discord_live)
    else:
        settings = base
        clock = Clock()
        conn = db.connect(settings.db_path)
        job = Job(a.job)
        pg = LoopPangram(settings, conn, clock)
        srv_factory, op_factory, claude_raw = real_parts(settings)
        stop_file = settings.stop_file                  # STOP_FILE, else STOP in RUNS_DIR
        a.digests_dir = str(SANDBOX / "digests")
        notifier = Notifier(settings, clock, settings.runs_dir / "discord.log", live=not a.smoke)
    if a.smoke and not a.window:
        wins = job.windows()
        if not wins:
            print(f"--smoke: the job {job.name} has no window in scope", flush=True)
            return 1
        a.window = wins[0]["wid"]                       # the job's first window
    schedule = Schedule(settings.config)
    budget = Budget(settings, conn, clock, schedule, rehearsal=a.rehearse)
    loop = Loop(settings=settings, conn=conn, clock=clock, schedule=schedule, budget=budget, notifier=notifier,
                job=job, pg=pg, claude_raw=claude_raw, srv_factory=srv_factory, op_factory=op_factory, args=a,
                rehearsal=a.rehearse, stop_file=stop_file)
    if a.rehearse:
        fakes.loop = loop
    if Path(stop_file).exists():
        print(f"the STOP file is there ({stop_file}); delete it to let the loop run", flush=True)
        return 0
    # a start outside the run window waits for the next 2:00, today's or tomorrow's (M1 review, finding 1)
    wake = start_time(schedule, clock.now(), ignore_schedule=a.now)
    if wake > clock.now():
        print(f"waiting for {clock.stamp(wake)}", flush=True)
        while clock.now() < wake:
            if Path(stop_file).exists():
                return 0
            clock.sleep(min(60.0, wake - clock.now()))
    crashed = None
    try:
        crashed = run_with_restarts(loop, job, schedule, clock, stop_file, a)
    finally:
        loop.finish(crashed=crashed)
        try:
            if not a.window:                   # locks.md lists every window; a partial run would cut it down
                job.setup(run_info=lock_info(loop))
        except Exception:  # noqa: BLE001
            pass
    return 1 if crashed else 0


def lock_info(loop):
    """locks.md's per-window part (plan 5.1 step 3), from the runs the loop built."""
    from engine.draws import numbers, quoted
    out = {}
    for ws in loop.windows:
        if not getattr(ws, "runs", None):
            continue
        q = quoted(ws.text)
        out[ws.wid] = {"heading": [ws.heading] if ws.heading else [],
                       "quotes": [x for x in q if x.startswith('"')][:12],
                       "citations": [x for x in q if x.startswith("(")], "numbers": sorted(numbers(ws.text)),
                       "yours": sum(1 for p in ws.runs for r in p if not r["claude"] and not loop.locked(r["sents"][0])),
                       "templates": [r["sents"][0] for p in ws.runs for r in p if not r["claude"] and loop.locked(r["sents"][0])],
                       "author": [{"para": pi, "run": ri, "text": " ".join(r["sents"]),
                                   "protected": ws.must.get((pi, ri), {}).get("protected") or [],
                                   "asked": ws.must.get((pi, ri), {}).get("asked", False),
                                   "regrown": ws.current[pi][ri] != ws.original[pi][ri] if getattr(ws, "current", None) else False}
                                  for pi, p in enumerate(ws.runs) for ri, r in enumerate(p) if r.get("author")],
                       "runs": [{"para": pi, "run": ri, "sents": len(r["sents"]), "text": " ".join(r["sents"]),
                                 "must": ws.must.get((pi, ri), {"hedges": [], "repeats": []})}
                                for pi, p in enumerate(ws.runs) for ri, r in enumerate(p) if r["claude"]]}
    return out


if __name__ == "__main__":
    sys.exit(main())
