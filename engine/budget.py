"""Every cap the loop runs under, checked in code before every paid call (plan 7.1, 7.3, 8.1).

Nothing here is a prompt, and no prompt holds a limit. Each check reads the
ledgers in the one SQLite database and the numbers in config.toml:

Pangram (the pangram_calls ledger, the same one engine/pangram.py writes):
    daily     the loop's spend since midnight stays under [pangram] nightly_cap_usd
    monthly   all spend this calendar month stays under [pangram] monthly_cap_usd

Claude (the claude_calls ledger; every headless call logs its estimated dollars
and the rate_limit_event readings of your week and your 5-hour window):
    ceiling   the real weekly total (the latest reading) plus the next call stays under total_ceiling
    share     the loop's share of the week (its estimated dollars this week times the
              conversion rate) plus the next call stays under loop_share, less any
              overshoot carried from last week
    allowance the loop's remaining share, spread over the run-days left before the
              weekly reset, caps today's use
    five-hour the 5-hour reading plus the next call stays under the phase's cap
              (night_claude_max_5h until claude_until, day_claude_max_5h after)
    backstop  until backstop_run_days real run-days have finished, the loop's estimated
              dollars today stay under first_night_cap_usd

"The next call" is the costly end of recent calls of the same kind (the dearest of
the last expected_cost_calls), so a big call can't sneak past.

A refusal raises ClaudeCapReached or PangramCapReached, naming the check, and says
until when it holds, so the run loop knows whether to wait or to stop asking.
"""
import datetime as dt
import statistics

from .claude_call import ensure_schema
from .pangram import Pangram

WEEK = 7 * 86400
FIVE_HOURS = 5 * 3600
LOOP_ROLE = "loop %"                 # every loop call's role starts with "loop "
# a kind's fallback expected cost before it has history: the call's own --max-budget-usd
RUN_KINDS = ("judge", "h2h", "facts")

RUN_DAYS = """
CREATE TABLE IF NOT EXISTS run_days (
    day TEXT PRIMARY KEY,
    started REAL,
    ended REAL,
    done INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS claude_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    day TEXT NOT NULL,
    kind TEXT NOT NULL,
    points REAL NOT NULL,
    dollars REAL NOT NULL,
    rate REAL NOT NULL
);
"""


WINDOW_SHARE = "Pangram window share"     # the window rounds' part of the day cap; the whole-text round passes it


class CapReached(Exception):
    def __init__(self, check, detail, until=None):
        super().__init__(f"{check}: {detail}")
        self.check, self.detail, self.until = check, detail, until


class ClaudeCapReached(CapReached):
    pass


class PangramCapReached(CapReached):
    pass


def kind_of(role):
    """'loop s1w0 judge' -> 'judge'."""
    return role.split()[-1] if role else "other"


class LoopPangram(Pangram):
    """engine/pangram.py's client with the loop's clock: ledger rows and the cap check both use it,
    so a compressed rehearsal day keeps its own midnight. The HTTP calls and the cache are unchanged."""

    def __init__(self, settings, conn, clock, session=None):
        super().__init__(settings, conn, session=session)
        self.clock = clock

    def check_caps(self, usd, purpose, now=None):
        return super().check_caps(usd, purpose, now=now or self.clock.now())

    def _log(self, sha, words, credits, usd, purpose, label, status, version=None):
        self.conn.execute(
            "INSERT INTO pangram_calls (ts, sha, model, words, credits, usd, purpose, label, status, version)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (self.clock.now(), sha, self.model, words, credits, usd, purpose, label, status, version))
        self.conn.commit()


class Budget:
    def __init__(self, settings, conn, clock, schedule, rehearsal=False):
        self.settings, self.conn, self.clock, self.schedule = settings, conn, clock, schedule
        self.pg = settings.config["pangram"]
        self.cl = settings.config["claude"]
        self.rehearsal = rehearsal
        self.run_day_start = None          # set once by the run loop when its run-day begins (pinned across restarts)
        ensure_schema(conn)
        conn.executescript(RUN_DAYS)
        conn.commit()

    # ------------------------------------------------------------------ Pangram
    def pangram_spent(self, since, purpose=None):
        # every row with a cost, whatever its status (a failed or timed-out task may still be charged)
        q = "SELECT COALESCE(SUM(usd), 0) FROM pangram_calls WHERE ts >= ? AND usd > 0"
        args = [since]
        if purpose:
            q += " AND purpose = ?"
            args.append(purpose)
        return self.conn.execute(q, args).fetchone()[0]

    def month_start(self, now=None):
        d = dt.datetime.fromtimestamp(now or self.clock.now())
        return d.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()

    def pangram_check(self, usd, purpose="loop", reserve=0.0):
        """Raise PangramCapReached if this spend would cross the daily or the monthly cap. `reserve` keeps that much
        of the day cap back: the window rounds pass whole_text_reserve_usd, so the whole-text round (which passes
        nothing) still has money once they've had theirs (5.6). That check is called "Pangram window share", and
        the run loop lets the whole-text round through it."""
        now = self.clock.now()
        b = self.schedule.bounds(now)
        month = self.pangram_spent(self.month_start(now))
        if month + usd > self.pg["monthly_cap_usd"]:
            first = dt.datetime.fromtimestamp(self.month_start(now) + 32 * 86400).replace(day=1, hour=0, minute=0)
            raise PangramCapReached("Pangram monthly", f"${month:.2f} spent this month, cap ${self.pg['monthly_cap_usd']:.2f}",
                                    until=first.timestamp())
        if purpose == "loop":
            cap = float(self.pg["nightly_cap_usd"])
            today = self.pangram_spent(b["midnight"], "loop")
            if today + usd > cap:
                raise PangramCapReached("Pangram daily", f"${today:.2f} spent today, cap ${cap:.2f}",
                                        until=b["midnight"] + 86400)
            # the run-day's own figure (M2): a --now run started in the evening crosses midnight, and the calendar
            # day's count starts again at 00:00, so the spend since this run-day began has to fit the cap as well.
            # It holds for the rest of the run (a run-day is one start of the loop).
            run_day = self.pangram_run_day()
            if run_day is not None and run_day + usd > cap:
                raise PangramCapReached("Pangram run-day", f"${run_day:.2f} spent since this run-day began at"
                                        f" {dt.datetime.fromtimestamp(self.run_day_start):%H:%M}, cap ${cap:.2f}",
                                        until=now + 86400)
            spent = max(today, run_day or 0.0)
            if reserve and spent + usd > cap - reserve:
                raise PangramCapReached(WINDOW_SHARE, f"${spent:.2f} spent; the last ${reserve:.2f} of the"
                                        f" ${cap:.2f} cap is kept for the whole-text round",
                                        until=(b["midnight"] + 86400) if run_day is None or today >= run_day
                                        else now + 86400)

    def pangram_run_day(self):
        """The loop's Pangram spend since this run-day began, or None before the run loop sets the start."""
        if self.run_day_start is None:
            return None
        return self.pangram_spent(self.run_day_start, "loop")

    def pangram_today(self):
        return self.pangram_spent(self.schedule.bounds(self.clock.now())["midnight"], "loop")

    def pangram_month(self):
        return self.pangram_spent(self.month_start())

    # ------------------------------------------------------------------ Claude readings
    def latest_reading(self):
        """(ts, week_util, five_hour_util, week_resets_at, five_hour_resets_at) of the newest call that carried one."""
        return self.conn.execute(
            "SELECT ts, week_util, five_hour_util, week_resets_at, five_hour_resets_at FROM claude_calls"
            " WHERE week_util IS NOT NULL ORDER BY ts DESC LIMIT 1").fetchone()

    def week_window(self, now=None):
        """(start, reset) of the current week, from the latest reading's resetsAt rolled forward past now."""
        now = now or self.clock.now()
        r = self.latest_reading()
        if not r or not r[3]:
            return now - WEEK, now + WEEK          # no reading yet: count everything from the last week
        reset = float(r[3])
        while reset <= now:
            reset += WEEK
        return reset - WEEK, reset

    def loop_dollars(self, since, until=None):
        q = "SELECT COALESCE(SUM(cost_usd), 0) FROM claude_calls WHERE role LIKE ? AND ts >= ?"
        args = [LOOP_ROLE, since]
        if until is not None:
            q += " AND ts < ?"
            args.append(until)
        return self.conn.execute(q, args).fetchone()[0]

    def rate(self, kind="week"):
        """Points per estimated dollar: the median of the last week of measured nights, else the config seed. The
        refit stays within rate_clamp (0.5x to 2x) of the seed: the weekly reading moves in whole points, so one
        quiet night with a single tick could otherwise cut the rate by five and let five times the spend through
        (M1 review, finding 6)."""
        seed = float(self.cl["week_points_per_usd" if kind == "week" else "five_hour_points_per_usd"])
        rows = [r[0] for r in self.conn.execute(
            "SELECT rate FROM claude_rates WHERE kind = ? ORDER BY ts DESC LIMIT 7", (kind,))]
        if not rows:
            return seed
        lo, hi = (float(x) for x in self.cl.get("rate_clamp", [0.5, 2.0]))
        return min(max(statistics.median(rows), lo * seed), hi * seed)

    def loop_share(self, now=None):
        start, _ = self.week_window(now)
        return self.loop_dollars(start) * self.rate("week")

    def share_cap(self, now=None):
        """loop_share, less what last week's loop went over it (7.3: an overshoot comes off next week)."""
        start, _ = self.week_window(now)
        last = self.loop_dollars(start - WEEK, start) * self.rate("week")
        return float(self.cl["loop_share"]) - max(0.0, last - float(self.cl["loop_share"]))

    def real_total(self, now=None):
        """The latest weekly reading, or zero once its week has reset. Never below the loop's own share."""
        now = now or self.clock.now()
        r = self.latest_reading()
        start, _ = self.week_window(now)
        reading = r[1] if r and r[0] >= start and (not r[3] or float(r[3]) > now) else 0.0
        return max(reading or 0.0, self.loop_share(now))

    def five_hour_util(self, now=None):
        """(the latest 5-hour reading, its reset time), or (0, None) once that window has reset."""
        now = now or self.clock.now()
        r = self.latest_reading()
        if not r:
            return 0.0, None
        ts, _, five, _, five_reset = r
        reset = float(five_reset) if five_reset else ts + FIVE_HOURS    # unknown reset: assume the latest possible
        if now >= reset:
            return 0.0, None
        return float(five or 0.0), reset

    def expected_cost(self, kind, fallback):
        n = int(self.cl.get("expected_cost_calls", 20))
        rows = [r[0] for r in self.conn.execute(
            "SELECT cost_usd FROM claude_calls WHERE role LIKE ? AND cost_usd IS NOT NULL ORDER BY ts DESC LIMIT ?",
            (f"loop %{kind}", n))]
        return max(rows) if rows else fallback

    def finished_run_days(self):
        return self.conn.execute("SELECT COUNT(*) FROM run_days WHERE done = 1").fetchone()[0]

    def allowance(self, now=None):
        """Today's points of the week (7.3, tonight's allowance): what's left of the share at the start of
        today, spread over the run-days left before the reset, today included."""
        now = now or self.clock.now()
        b = self.schedule.bounds(now)
        start, reset = self.week_window(now)
        day_from = max(b["midnight"], start)
        before_today = self.loop_dollars(start, day_from) * self.rate("week")
        left = max(0.0, self.share_cap(now) - before_today)
        return left / self.schedule.run_days_between(now, reset)

    def loop_points_today(self, now=None):
        now = now or self.clock.now()
        start, _ = self.week_window(now)
        return self.loop_dollars(max(self.schedule.bounds(now)["midnight"], start)) * self.rate("week")

    def loop_dollars_today(self, now=None):
        now = now or self.clock.now()
        return self.loop_dollars(self.schedule.bounds(now)["midnight"])

    # ------------------------------------------------------------------ the Claude check
    def claude_check(self, kind, fallback_usd, extra_usd=0.0):
        """Raise ClaudeCapReached if a call of this kind (plus `extra_usd` of calls planned after it) could cross a
        cap. Returns the expected dollars."""
        now = self.clock.now()
        b = self.schedule.bounds(now)
        usd = self.expected_cost(kind, fallback_usd) + extra_usd
        pts = usd * self.rate("week")
        start, reset = self.week_window(now)
        total = self.real_total(now)
        if total + pts >= float(self.cl["total_ceiling"]):
            raise ClaudeCapReached("Claude weekly ceiling",
                                   f"your week is at {total:.0%}, ceiling {float(self.cl['total_ceiling']):.0%}", until=reset)
        share, cap = self.loop_share(now), self.share_cap(now)
        if share + pts >= cap:
            raise ClaudeCapReached("Claude weekly share",
                                   f"the loop's share is {share:.1%} of the week, its cap {cap:.0%}", until=reset)
        allow, today = self.allowance(now), self.loop_points_today(now)
        if today + pts > allow:
            raise ClaudeCapReached("Claude allowance today",
                                   f"the loop used {today:.1%} of the week today, today's allowance {allow:.1%}",
                                   until=self.schedule.next_start(b["stop"]))
        five, five_reset = self.five_hour_util(now)
        five_cap = self.schedule.five_hour_cap(now)
        if five + usd * self.rate("five_hour") >= five_cap:
            phase = "night" if now < b["claude_until"] else "day"
            raise ClaudeCapReached(f"Claude 5-hour ({phase})",
                                   f"the 5-hour window is at {five:.0%}, the {phase} cap {five_cap:.0%}",
                                   until=five_reset or now + FIVE_HOURS)
        if self.finished_run_days() < int(self.cl.get("backstop_run_days", 1)):
            spent = self.loop_dollars_today(now)
            if spent + usd > float(self.cl["first_night_cap_usd"]):
                raise ClaudeCapReached("Claude first-run backstop",
                                       f"${spent:.2f} estimated today, backstop ${float(self.cl['first_night_cap_usd']):.2f}",
                                       until=self.schedule.next_start(b["stop"]))
        return usd

    def run_check(self, fallbacks):
        """Before a run's judge: can the judge, the head-to-head and the fact check all still fit?"""
        rest = sum(self.expected_cost(k, fallbacks[k]) for k in RUN_KINDS[1:])
        return self.claude_check("judge", fallbacks["judge"], extra_usd=rest)

    # ------------------------------------------------------------------ run-days and the conversion rate
    def day_started(self, day):
        self.conn.execute("INSERT OR IGNORE INTO run_days (day, started) VALUES (?, ?)", (day, self.clock.now()))
        self.conn.commit()

    def day_finished(self, day):
        self.conn.execute("UPDATE run_days SET ended = ?, done = 1 WHERE day = ?", (self.clock.now(), day))
        self.conn.commit()

    def measure_rates(self, day, since, until):
        """Refit the conversion rates from a quiet stretch (the night phase): reading change over loop dollars.
        Readings move in whole points, so a stretch needs a few points of movement and a dollar of spend to count.
        Returns what it stored."""
        rows = self.conn.execute(
            "SELECT ts, week_util, five_hour_util, five_hour_resets_at, cost_usd, role FROM claude_calls"
            " WHERE ts >= ? AND ts < ? AND week_util IS NOT NULL ORDER BY ts", (since, until)).fetchall()
        out = {}
        if len(rows) < 2:
            return out
        dollars = sum(r[4] or 0 for r in rows[1:] if r[5].startswith("loop "))
        dw = (rows[-1][1] or 0) - (rows[0][1] or 0)
        if dollars >= 1.0 and dw >= 0.02:
            out["week"] = dw / dollars
        # the 5-hour rate only within one window (same reset time)
        same = [r for r in rows if r[3] == rows[-1][3]]
        if len(same) >= 2:
            d5 = sum(r[4] or 0 for r in same[1:] if r[5].startswith("loop "))
            dp = (same[-1][2] or 0) - (same[0][2] or 0)
            if d5 >= 1.0 and dp >= 0.02:
                out["five_hour"] = dp / d5
        for kind, rate in out.items():
            self.conn.execute("INSERT INTO claude_rates (ts, day, kind, points, dollars, rate) VALUES (?,?,?,?,?,?)",
                              (self.clock.now(), day, kind, dw if kind == "week" else dp,
                               dollars if kind == "week" else d5, rate))
        self.conn.commit()
        return out

    # ------------------------------------------------------------------ for the posts
    def snapshot(self):
        now = self.clock.now()
        five, five_reset = self.five_hour_util(now)
        start, reset = self.week_window(now)
        return {
            "pangram_today": self.pangram_today(), "pangram_day_cap": self.pg["nightly_cap_usd"],
            "pangram_run_day": self.pangram_run_day(),
            "pangram_month": self.pangram_month(), "pangram_month_cap": self.pg["monthly_cap_usd"],
            "claude_usd_today": self.loop_dollars_today(now),
            "backstop": (float(self.cl["first_night_cap_usd"])
                         if self.finished_run_days() < int(self.cl.get("backstop_run_days", 1)) else None),
            "loop_points_today": self.loop_points_today(now), "allowance": self.allowance(now),
            "loop_share": self.loop_share(now), "share_cap": self.share_cap(now),
            "real_total": self.real_total(now), "ceiling": float(self.cl["total_ceiling"]),
            "five_hour": five, "five_hour_cap": self.schedule.five_hour_cap(now), "five_hour_reset": five_reset,
            "week_reset": reset, "rate": self.rate("week"),
        }
