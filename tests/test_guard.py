"""The daytime guard (engine/guard.py): every probe and the server are fakes, the clock is a fake one."""
import datetime as dt
import json
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from engine import guard as gd
from engine.guard import Guard, GuardedModel, GuardedServer, Halted

NOON = dt.datetime(2026, 11, 16, 12, 0)
CFG = {"enabled": True, "hours": "always", "active_minutes": 3, "idle_minutes": 10, "pause_on_fullscreen": True,
       "other_gpu_mb": 1500, "poll_seconds": 15, "low_priority": True, "ignore_processes": ["dwm.exe"]}


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


class Probes:
    """Input happened at `last_input` (clock time); idle() reads the gap. Each call is recorded."""

    def __init__(self, clock, last_input=None):
        self.clock = clock
        self.last_input = clock.t - 3600 if last_input is None else last_input
        self.fs, self.procs, self.names = "", {}, {}
        self.calls, self.fail = [], set()

    def _call(self, name):
        self.calls.append(name)
        if name in self.fail:
            raise OSError(f"{name} probe broke")

    def idle(self):
        self._call("idle")
        return self.clock.t - self.last_input

    def fullscreen(self):
        self._call("fullscreen")
        return self.fs

    def gpu(self):
        self._call("gpu")
        return dict(self.procs)

    def vram(self):
        self._call("vram")
        return (2000.0, 16376.0)

    def name(self, pid):
        return self.names.get(pid, "app.exe")

    def family(self):
        return {1}


class Server:
    def __init__(self):
        self.up, self.starts, self.stops = False, 0, 0
        self.proc = types.SimpleNamespace(pid=999)
        self.order = []
        self.hold = None            # an Event a request waits on, for the request-in-flight test
        self.entered = threading.Event()

    def start(self):
        self.starts += 1
        self.up = True
        self.order.append("start")
        return 1.5

    def stop(self):
        if self.up:
            self.stops += 1
            self.order.append("stop")
        self.up = False

    def complete(self, prompt, **kw):
        assert self.up, "a request reached a stopped server"
        self.entered.set()
        if self.hold is not None:
            self.hold.wait(10)
        self.order.append(f"done {prompt}")
        return f"out {prompt}", {}


def make(cfg=None, *, now=NOON, halt=None, status=None):
    clock = Clock()
    probes = Probes(clock)
    lines = []
    g = Guard(dict(CFG, **(cfg or {})), probes=probes, clock=clock.now, sleep=clock.sleep, now=lambda: now,
              log=lines.append, status_path=status, halt=halt)
    raw = Server()
    return GuardedServer(raw, g), g, raw, probes, clock, lines


class Pausing(unittest.TestCase):
    def test_active_pauses_and_stops_the_server(self):
        with tempfile.TemporaryDirectory() as d:
            srv, g, raw, p, clock, lines = make(status=Path(d) / "guard.json")
            srv.start()
            self.assertEqual((raw.starts, g.state), (1, "running"))
            self.assertEqual(srv.complete("a")[0], "out a")                     # clear: straight through
            clock.t += 20
            p.last_input = clock.t - 40                                          # a keypress 40s ago
            g.poll()
            self.assertEqual((g.state, raw.stops, raw.up), ("paused", 1, False))  # flag
            self.assertIn("guard: paused, active: last input 40s ago; the llama server is stopped", lines[-1])
            st = json.loads((Path(d) / "guard.json").read_text(encoding="utf-8"))
            self.assertEqual((st["state"], st["reason"]), ("paused", "active: last input 40s ago"))
            self.assertEqual(gd.read_status(Path(d) / "guard.json"), "paused: active: last input 40s ago")

    def test_idle_for_idle_minutes_resumes_and_restarts_it(self):
        srv, g, raw, p, clock, lines = make()
        srv.start()
        clock.t += 20
        p.last_input = clock.t - 5
        g.poll()
        self.assertEqual(g.state, "paused")
        left = clock.t - 5
        self.assertEqual(srv.complete("b")[0], "out b")                          # held, then let through
        self.assertEqual((raw.starts, g.state), (2, "running"))
        self.assertGreaterEqual(clock.t - left, 600)                             # ten minutes away, no less
        self.assertLess(clock.t - left, 600 + 2 * 15)                            # and not much more
        self.assertTrue(any(x.startswith("guard: resumed after") and "the llama server is back" in x for x in lines))
        self.assertEqual(raw.order, ["start", "stop", "start", "done b"])

    def test_a_moment_of_input_while_paused_starts_the_count_over(self):
        srv, g, raw, p, clock, _ = make()
        srv.start()
        p.last_input = clock.t
        clock.t += 15
        g.poll()
        t0 = clock.t

        def touch(s):                                       # you touch the mouse again eight minutes in
            clock.t += s
            if 480 <= clock.t - t0 < 495:
                p.last_input = clock.t
        g.sleep = touch
        srv.complete("c")
        self.assertGreaterEqual(clock.t - t0, 480 + 600)                         # ten clear minutes after the touch

    def test_fullscreen_pauses(self):
        srv, g, raw, p, clock, lines = make()
        srv.start()
        p.fs = "fullscreen app (a fullscreen Direct3D game or video, QUNS_RUNNING_D3D_FULL_SCREEN)"
        clock.t += 15
        g.poll()
        self.assertEqual((g.state, raw.up), ("paused", False))                   # flag: no input needed
        self.assertIn("fullscreen app", g.reason)
        p.fs = ""
        srv.complete("d")                                                        # clear for ten minutes, back
        self.assertEqual(raw.starts, 2)

    def test_fullscreen_is_ignored_when_turned_off(self):
        srv, g, raw, p, clock, _ = make({"pause_on_fullscreen": False})
        srv.start()
        p.fs = "fullscreen app (presentation mode, QUNS_PRESENTATION_MODE)"
        clock.t += 15
        g.poll()
        self.assertEqual(g.state, "running")                                     # pass
        self.assertNotIn("fullscreen", p.calls)

    def test_another_gpu_process_over_the_threshold_pauses(self):
        srv, g, raw, p, clock, lines = make()
        srv.start()
        p.names = {4321: "game.exe", 1356: "dwm.exe"}
        p.procs = {999: 9000.0, 1: 1800.0, 1356: 2900.0, 77: 400.0}               # ours, our family, dwm, a small app
        clock.t += 15
        g.poll()
        self.assertEqual(g.state, "running")                                     # pass: none of those count
        p.procs[4321] = 6348.8
        clock.t += 15
        g.poll()
        self.assertEqual(g.state, "paused")                                      # flag
        self.assertEqual(g.reason, "another process holds 6.2 GB (game.exe 4321)")

    def test_a_busy_pc_holds_the_first_start(self):
        srv, g, raw, p, clock, lines = make()
        p.last_input = clock.t - 10
        t0 = clock.t
        self.assertEqual(srv.start(), 1.5)                                       # what the server's start returned
        self.assertGreaterEqual(clock.t - t0, 590)
        self.assertEqual(raw.starts, 1)
        self.assertEqual(lines[0], "guard: the llama server waits to start, active: last input 10s ago")


class Hours(unittest.TestCase):
    def test_parse_and_in_hours(self):
        h = gd.parse_hours("09:00-02:00")
        self.assertTrue(gd.in_hours(h, dt.datetime(2026, 1, 1, 23, 30)))
        self.assertTrue(gd.in_hours(h, dt.datetime(2026, 1, 1, 1, 59)))
        self.assertFalse(gd.in_hours(h, dt.datetime(2026, 1, 1, 3, 0)))
        self.assertTrue(gd.in_hours(gd.parse_hours("09:00-17:00"), dt.datetime(2026, 1, 1, 9, 0)))
        self.assertFalse(gd.in_hours(gd.parse_hours("09:00-17:00"), dt.datetime(2026, 1, 1, 17, 0)))
        self.assertIsNone(gd.parse_hours("always"))
        for bad in ("9-5", "25:00-01:00", "nine"):
            with self.assertRaises(ValueError):
                gd.parse_hours(bad)

    def test_outside_hours_the_guard_stands_down(self):
        srv, g, raw, p, clock, _ = make({"hours": "09:00-02:00"}, now=dt.datetime(2026, 11, 16, 3, 0))
        p.last_input = clock.t - 5                                               # you're right there, at 3am
        srv.start()
        clock.t += 15
        g.poll()
        srv.complete("e")
        self.assertEqual((g.state, raw.starts, raw.stops), ("running", 1, 0))
        self.assertEqual(p.calls, [])                                            # nothing was looked at

    def test_a_paused_run_carries_on_once_the_hours_end(self):
        when = {"now": NOON}
        srv, g, raw, p, clock, _ = make({"hours": "09:00-02:00"})
        g.now = lambda: when["now"]
        srv.start()
        p.last_input = clock.t
        clock.t += 15
        g.poll()
        self.assertEqual(g.state, "paused")
        when["now"] = dt.datetime(2026, 11, 17, 2, 30)                           # past the window's end
        srv.complete("f")
        self.assertEqual((g.state, raw.starts), ("running", 2))


class InFlight(unittest.TestCase):
    def test_the_request_in_flight_finishes_before_the_stop(self):
        srv, g, raw, p, clock, _ = make()
        srv.start()
        raw.hold = threading.Event()
        got = {}
        worker = threading.Thread(target=lambda: got.update(r=srv.complete("g")))
        worker.start()
        self.assertTrue(raw.entered.wait(5))
        p.last_input = clock.t
        clock.t += 15
        pauser = threading.Thread(target=g.poll)
        pauser.start()
        deadline = time.time() + 5
        while g.state != "paused" and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(g.state, "paused")                                       # the gate is closed...
        time.sleep(0.2)
        self.assertEqual(raw.stops, 0)                                            # ...but the server waits
        raw.hold.set()
        worker.join(5)
        pauser.join(5)
        self.assertEqual(got["r"][0], "out g")
        self.assertEqual(raw.order, ["start", "done g", "stop"])


class Deadline(unittest.TestCase):
    def test_a_paused_run_reaches_its_deadline_and_ends_cleanly(self):
        clock_box = {}
        srv, g, raw, p, clock, lines = make(halt=lambda: clock_box["c"].t >= clock_box["end"])
        clock_box.update(c=clock, end=clock.t + 300)
        srv.start()
        p.last_input = clock.t
        clock.t += 15
        g.poll()

        def busy_sleep(s):                                   # you stay at the PC
            clock.t += s
            p.last_input = clock.t
        g.sleep = busy_sleep
        with self.assertRaises(Halted):
            srv.complete("h")
        self.assertGreaterEqual(clock.t, clock_box["end"])
        self.assertLess(clock.t, clock_box["end"] + 20)
        self.assertEqual((raw.starts, raw.up), (1, False))                       # never started again
        self.assertTrue(any("stop time came while paused" in x for x in lines))
        srv.stop()                                                               # the run ends cleanly
        self.assertEqual(g.state, "stopped")

    def test_the_window_loop_ends_its_day_on_a_halt(self):
        from engine.run import run_with_restarts
        from engine.schedule import CompressedClock, Schedule
        from engine.settings import Settings
        from test_run_loop import CONFIG, ts
        events = []

        class FakeLoop:
            settings = Settings(env={}, config=dict(CONFIG, loop={"failures_in_row": 5}))
            rounds_done, error_block, stop_reason = 0, 600.0, None
            schedule = Schedule(CONFIG)
            calls = stopped = 0

            def run_day(self):
                self.calls += 1
                raise Halted("active: last input 5s ago")

            def stop_servers(self):
                self.stopped += 1

            def log(self, m):
                events.append(m)

            def alert(self, key, text):
                events.append("ALERT " + key)

            def must_stop(self):
                self.stop_reason = self.stop_reason or "wind-down: no new attempts after 15:30"
                return True

            def tick(self):
                pass
        lp = FakeLoop()
        clock = CompressedClock(ts(15, 40), speed=0.0)
        got = run_with_restarts(lp, types.SimpleNamespace(setup=lambda: None), lp.schedule, clock, Path("no STOP"),
                                types.SimpleNamespace(now=True))
        self.assertIsNone(got)                                                   # no crash text
        self.assertEqual((lp.calls, lp.stopped), (1, 1))                         # no restart, servers down
        self.assertFalse(any(e.startswith(("ERROR", "CRASH", "ALERT")) for e in events))
        self.assertTrue(events[-1].startswith("stopping: wind-down"))

    def test_a_held_draw_stops_the_run_s_round_without_an_error(self):
        from engine.run import Loop
        lp = Loop.__new__(Loop)
        lp.run_args = lambda *a, **k: {"key": "k"}
        lp.queue = types.SimpleNamespace(take=lambda key: None)

        def draw(ws, ra):
            raise Halted("fullscreen app")
        lp.draw = draw
        self.assertEqual(lp.regrow(None, 0, 0, False, 1, False, arms=["fewshot"]), (None, [], "held"))

    def test_guard_halt_reads_the_stop_file_and_the_clock(self):
        from engine.run import Loop
        from engine.schedule import CompressedClock, Schedule
        from test_run_loop import CONFIG, ts
        with tempfile.TemporaryDirectory() as d:
            lp = Loop.__new__(Loop)
            lp.stop_file, lp.discord_stop, lp.stop_reason = Path(d) / "STOP", False, None
            lp.args = types.SimpleNamespace(now=False)
            lp.schedule = Schedule(CONFIG)
            lp.clock = CompressedClock(ts(12), speed=0.0)
            self.assertFalse(lp.guard_halt())                                    # midday: work on
            lp.clock = CompressedClock(ts(15, 45), speed=0.0)
            self.assertTrue(lp.guard_halt())                                     # past the last attempt
            lp.stop_reason = "wind-down"
            self.assertFalse(lp.guard_halt())                                    # the day-end round still may
            lp.clock = CompressedClock(ts(15, 56), speed=0.0)
            self.assertTrue(lp.guard_halt())                                     # not in the last five minutes
            lp.stop_reason, lp.clock = None, CompressedClock(ts(12), speed=0.0)
            lp.stop_file.write_text("", encoding="utf-8")
            self.assertTrue(lp.guard_halt())


class Priority(unittest.TestCase):
    def server(self, d, config):
        from engine import local_models
        from engine.settings import Settings
        srv = local_models.LlamaServer("mistral-small-24b-base", settings=Settings(env={"RUNS_DIR": d}, config=config))
        got = {}

        def popen(cmd, **kw):
            got.update(kw)
            return types.SimpleNamespace(pid=4242, poll=lambda: 1)          # exits at once: start raises after
        with mock.patch.object(local_models.subprocess, "Popen", popen), \
                mock.patch.object(srv, "_answers", return_value=False), mock.patch.object(srv, "missing", return_value=[]):
            with self.assertRaises(RuntimeError):
                srv.start()
        return got

    def test_the_low_priority_flags_are_passed_when_starting_a_process(self):
        from test_run_loop import CONFIG
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            with mock.patch.object(gd, "WINDOWS", True):
                got = self.server(d, CONFIG)
                self.assertTrue(got["creationflags"] & gd.BELOW_NORMAL_PRIORITY_CLASS)          # flag: on by default
                got = self.server(d, dict(CONFIG, guard={"low_priority": False}))
                self.assertFalse(got["creationflags"] & gd.BELOW_NORMAL_PRIORITY_CLASS)         # pass: turned off
            calls = []
            with mock.patch.object(gd, "WINDOWS", False), \
                    mock.patch.object(gd.os, "getpriority", lambda which, who: 0, create=True), \
                    mock.patch.object(gd.os, "setpriority", lambda *a: calls.append(a), create=True), \
                    mock.patch.object(gd.os, "PRIO_PROCESS", 0, create=True):
                got = self.server(d, CONFIG)
                self.assertFalse(got["creationflags"] & gd.BELOW_NORMAL_PRIORITY_CLASS)
                self.assertEqual(calls, [(0, 4242, gd.NICE)])                   # POSIX: nice 10 just after the start

    def test_the_night_starts_its_children_low_on_windows(self):
        from engine import night
        got = {}
        with mock.patch.object(night, "POSIX", False), mock.patch.object(gd, "WINDOWS", True), \
                mock.patch.object(gd, "low_priority_on", return_value=True), \
                mock.patch.object(night.subprocess, "Popen", lambda cmd, **kw: got.update(kw) or "proc"):
            night.spawn(["x"], None)
        self.assertTrue(got["creationflags"] & gd.BELOW_NORMAL_PRIORITY_CLASS)


class Off(unittest.TestCase):
    def test_disabled_means_no_probe_is_ever_called(self):
        srv, g, raw, p, clock, _ = make({"enabled": False})
        srv.start()
        p.last_input = clock.t
        p.fs, p.procs = "fullscreen app (x, QUNS_BUSY)", {4321: 9000.0}
        for _ in range(3):
            clock.t += 30
            g.poll()
            srv.complete("i")
        self.assertEqual(p.calls, [])
        self.assertEqual((raw.starts, raw.stops), (1, 0))
        plain = Server()
        settings = types.SimpleNamespace(config={"guard": {"enabled": False}}, runs_dir=Path("."))
        self.assertIs(gd.guard_server(plain, settings), plain)                   # not even wrapped

    def test_a_probe_that_raises_counts_as_unknown_and_never_crashes_a_run(self):
        srv, g, raw, p, clock, lines = make()
        p.fail = {"idle", "fullscreen", "gpu"}
        srv.start()                                                             # unknown everything: clear
        for _ in range(3):
            clock.t += 15
            g.poll()
            srv.complete("j")
        self.assertEqual((g.state, raw.stops), ("running", 0))
        self.assertEqual(sum("probe failed" in x for x in lines), 3)            # each said once
        # unknown idle counts as away: a pause for a fullscreen app ends ten clear minutes after it closes
        p.fail = {"idle"}
        p.fs = "fullscreen app (a fullscreen app, QUNS_BUSY)"
        clock.t += 15
        g.poll()
        self.assertEqual(g.state, "paused")
        p.fs = None                                                              # None: the probe can't say
        t0 = clock.t
        srv.complete("k")
        self.assertEqual(raw.starts, 2)
        self.assertGreaterEqual(clock.t - t0, 600)


class Models(unittest.TestCase):
    def test_open_pangram_leaves_the_card_on_a_pause_and_comes_back(self):
        srv, g, raw, p, clock, _ = make()
        moves = []
        inner = types.SimpleNamespace(to=lambda dev: moves.append(dev))
        op = types.SimpleNamespace(device="cuda", model=inner, score=lambda texts: [{"score": 0.1}] * len(texts))
        gop = GuardedModel(op, g)
        srv.start()
        self.assertEqual(gop.score(["a"]), [{"score": 0.1}])
        p.last_input = clock.t
        clock.t += 15
        g.poll()
        self.assertEqual(moves, ["cpu"])
        gop.score(["b"])                                                         # held until clear, then back
        self.assertEqual(moves, ["cpu", "cuda"])
        self.assertEqual(gd.guard_model(Server(), op), op)                       # an unguarded server: as it is


class McpStatus(unittest.TestCase):
    def test_run_status_says_paused_with_the_reason(self):
        from mcp_server.core import App
        with tempfile.TemporaryDirectory() as d:
            app = App(settings=types.SimpleNamespace(runs_dir=Path(d), get=lambda k, default="": default))
            rid = "pass-20261116-020000-a1b2c3"
            folder = app.store().make(rid)
            rec = {"id": rid, "state": "running"}
            self.assertIsNone(app.guard_line(rec))                               # no file yet
            g = Guard(CFG, probes=Probes(Clock()), status_path=folder / "guard.json", log=lambda m: None)
            g.state, g.reason = "paused", "fullscreen app (a fullscreen app, QUNS_BUSY)"
            g.write_status()
            self.assertEqual(app.guard_line(rec), "paused: fullscreen app (a fullscreen app, QUNS_BUSY)")
            self.assertIsNone(app.guard_line(dict(rec, state="finished")))       # only while it runs
            g.clear_s = 300.0                                                    # the PC has cleared since
            g.write_status()
            self.assertEqual(app.guard_line(rec), "paused: fullscreen app (a fullscreen app, QUNS_BUSY) (clear for"
                                                  " 5 min now; the server comes back after 10 min clear, at the"
                                                  " next draw)")
            st = json.loads((folder / "guard.json").read_text(encoding="utf-8"))
            st["updated"] -= 3600
            (folder / "guard.json").write_text(json.dumps(st), encoding="utf-8")
            self.assertIsNone(app.guard_line(rec))                               # an old file says nothing

    def test_the_launch_hands_the_run_its_status_file(self):
        from mcp_server.core import App
        with tempfile.TemporaryDirectory() as d:
            Path(d, "jobs", "j").mkdir(parents=True)
            Path(d, "jobs", "j", "job.toml").write_text("", encoding="utf-8")
            seen = {}

            def spawn(cmd, out, env):
                seen.update(env)
                return types.SimpleNamespace(pid=1)
            app = App(settings=types.SimpleNamespace(runs_dir=Path(d), get=lambda k, default="": default),
                      repo=d, spawn_fn=spawn, began_fn=lambda pid: 1.0)
            out = app.launch(Path(d, "jobs", "j"), types.SimpleNamespace(name="j"), "pass", None, 1, 1.0, None)
            self.assertEqual(Path(seen[gd.STATUS_ENV]), app.store().folder(out["run_id"]) / "guard.json")


class RealProbes(unittest.TestCase):
    def test_the_real_probes_answer_or_say_unknown(self):
        g = Guard(CFG, log=lambda m: None)
        r = g.read()                                                             # never raises
        self.assertTrue(r["idle"] is None or r["idle"] >= 0)
        self.assertTrue(r["fullscreen"] is None or isinstance(r["fullscreen"], str))


if __name__ == "__main__":
    unittest.main()
