"""The clock and the run-day schedule (plan 8.1).

Everything in the loop that asks the time asks a Clock, so the rehearsal can run
a whole 2:00 to 16:00 day on a compressed clock with the same code.

    Clock            real time
    CompressedClock  starts at a chosen time; real seconds run `speed` times
                     faster, and sleep() and advance() move it forward without
                     waiting (the rehearsal's fakes advance it by how long the
                     real thing would take)

Phases of a run-day, from config.toml:

    before      until [schedule] start
    night       start to [claude] claude_until    (5-hour cap: night_claude_max_5h)
    day         claude_until to last_attempt      (5-hour cap: day_claude_max_5h)
    wind-down   last_attempt to stop              (no new attempts; the one in flight finishes)
    after       from stop
"""
import datetime as dt
import threading
import time


def hm(s):
    """'02:00' -> (2, 0)."""
    h, m = s.strip().split(":")
    return int(h), int(m)


class Clock:
    def now(self):
        return time.time()

    def sleep(self, seconds):
        if seconds > 0:
            time.sleep(seconds)

    def advance(self, seconds):
        """Work that takes real time on its own; only the compressed clock moves on it."""

    def stamp(self, ts=None):
        return dt.datetime.fromtimestamp(self.now() if ts is None else ts).strftime("%Y-%m-%d %H:%M")


class CompressedClock(Clock):
    def __init__(self, start_ts, speed=1.0):
        self.t0_real = time.time()
        self.start = start_ts
        self.speed = speed
        self.extra = 0.0
        self.lock = threading.Lock()
        self.on_tick = None            # the rehearsal hangs its clock-timed events here (the STOP file)

    def now(self):
        with self.lock:
            return self.start + (time.time() - self.t0_real) * self.speed + self.extra

    def sleep(self, seconds):
        self.advance(seconds)

    def advance(self, seconds):
        if seconds > 0:
            with self.lock:
                self.extra += seconds
            if self.on_tick:
                self.on_tick()


def parse_clock(value, today=None):
    """--clock value: 'HH:MM' (today), or 'YYYY-MM-DD HH:MM' / 'YYYY-MM-DDTHH:MM'. Returns a timestamp."""
    value = value.strip().replace("T", " ")
    if len(value) <= 5:
        h, m = hm(value)
        d = today or dt.date.today()
        return dt.datetime(d.year, d.month, d.day, h, m).timestamp()
    return dt.datetime.strptime(value, "%Y-%m-%d %H:%M").timestamp()


class Schedule:
    def __init__(self, config):
        s, c = config["schedule"], config["claude"]
        self.start = hm(s["start"])
        self.last_attempt = hm(s["last_attempt"])
        self.stop = hm(s["stop"])
        self.claude_until = hm(c["claude_until"])
        self.night_cap = float(c["night_claude_max_5h"])
        self.day_cap = float(c["day_claude_max_5h"])

    @staticmethod
    def _at(ts, h_m):
        d = dt.datetime.fromtimestamp(ts)
        return d.replace(hour=h_m[0], minute=h_m[1], second=0, microsecond=0).timestamp()

    def bounds(self, ts):
        """The run-day that `ts` falls on, as timestamps."""
        return {"midnight": self._at(ts, (0, 0)), "start": self._at(ts, self.start),
                "claude_until": self._at(ts, self.claude_until), "last_attempt": self._at(ts, self.last_attempt),
                "stop": self._at(ts, self.stop)}

    def phase(self, ts):
        b = self.bounds(ts)
        if ts < b["start"]:
            return "before"
        if ts < b["claude_until"]:
            return "night"
        if ts < b["last_attempt"]:
            return "day"
        if ts < b["stop"]:
            return "wind-down"
        return "after"

    def five_hour_cap(self, ts):
        """How much of Claude's 5-hour window the loop may fill at `ts` (8.1's split)."""
        return self.night_cap if ts < self.bounds(ts)["claude_until"] else self.day_cap

    def may_start_attempt(self, ts):
        return self.phase(ts) in ("night", "day")

    def day_key(self, ts):
        return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

    def next_start(self, ts):
        """The next run-day start at or after ts."""
        b = self.bounds(ts)
        if ts <= b["start"]:
            return b["start"]
        return self._at(ts + 86400, self.start)

    def run_days_between(self, ts, until):
        """Run-days whose start falls in [today's start, until), counting today while today's stop hasn't passed.
        Tonight's allowance (7.3) spreads the loop's weekly share over these."""
        b = self.bounds(ts)
        first = b["start"] if ts < b["stop"] else self._at(ts + 86400, self.start)
        n, t = 0, first
        while t < until:
            n += 1
            t = self._at(t + 86400, self.start)
        return max(n, 1)
