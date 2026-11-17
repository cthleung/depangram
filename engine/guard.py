"""The daytime guard (plan 8.1, "Daytime local work"): the local models give the GPU back the moment you're at the PC.

The engine draws thousands of candidates on the llama server, on the same PC you game and watch video on. The guard
sits between the engine and that server. Every request to the server goes through a gate that returns at once while
the PC is clear. When you're back, the guard pauses:

  - you're active: a keypress or a mouse move in the last [guard] active_minutes;
  - something fullscreen is in front (a game, a fullscreen video, presentation mode), with pause_on_fullscreen;
  - another process holds [guard] other_gpu_mb of VRAM or more.

A pause closes the gate, lets the requests in flight finish, then stops the llama server so its VRAM is freed (and
moves Open Pangram off the card, when the run has it there), and waits. A thin watcher thread looks every
poll_seconds, so a pause also lands while the engine is busy elsewhere (a Claude call, a Pangram scoring, a wait on a
cap). Once everything has been clear for [guard] idle_minutes, the next request starts the server again and the work
carries on where it stopped. Every pause and resume goes in the run's log with its reason, and a small status file
says where the guard stands (RUNS_DIR/guard.json, or the path in DEPANGRAM_GUARD_STATUS; the MCP server gives each
run its own).

Nothing is lost: the pass banks every draw as it's made (regrow bank.jsonl), and the window loop's pool in hand
waits in memory across a pause. A pause never moves a deadline. A run that is paused when its stop time comes gives
up the wait (Halted), with the server left stopped; the window loop treats that as the day's stop, and the night
ends a paused pass at pass_until like any other.

Outside [guard] hours ("09:00-02:00", say) the guard stands down: it reads nothing and a paused run carries on at
its next request. [guard] enabled = false turns it off, and no probe is ever called.

[guard] low_priority (on by default) starts the llama server and the engine's own processes at below-normal CPU
priority (BELOW_NORMAL_PRIORITY_CLASS on Windows, nice 10 elsewhere). CPU priority doesn't reach the GPU's own
scheduling: the pause is what keeps a game smooth.

Every probe is injectable (Probes), and a probe that raises counts as unknown and never crashes a run: an unknown
idle time counts as away, an unknown fullscreen state as no, an unknown GPU reading as no other user.
"""
import ctypes
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

WINDOWS = os.name == "nt"
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
NICE = 10
STATUS_ENV = "DEPANGRAM_GUARD_STATUS"
DEFAULTS = {"enabled": True, "hours": "always", "active_minutes": 3, "idle_minutes": 10, "pause_on_fullscreen": True,
            "other_gpu_mb": 1500, "poll_seconds": 15, "low_priority": True, "ignore_processes": ["dwm.exe"]}
MIB = 1024 * 1024


class Halted(Exception):
    """A paused run's stop time came (the guard's `halt` said so): the request never went out, the server stays
    stopped, and the caller ends the work in hand the way it ends any stop."""


# ---------------------------------------------------------------------- settings
def config(source=None):
    """[guard] with the defaults filled in. `source` is a Settings, a config dict, or the [guard] dict itself."""
    cfg = getattr(source, "config", source)
    if isinstance(cfg, dict) and "guard" in cfg:
        cfg = cfg.get("guard")
    elif isinstance(cfg, dict) and not (set(cfg) & set(DEFAULTS)):
        cfg = {}
    out = dict(DEFAULTS)
    out.update(cfg if isinstance(cfg, dict) else {})
    for key in ("active_minutes", "idle_minutes", "other_gpu_mb", "poll_seconds"):
        out[key] = float(out[key])
        if out[key] < 0 or (key == "poll_seconds" and out[key] <= 0):
            raise ValueError(f"[guard] {key} is {out[key]}; it takes a number over zero")
    parse_hours(out["hours"])
    return out


def parse_hours(text):
    """None for "always", else (start, end) in minutes after midnight from "HH:MM-HH:MM". The end may come before
    the start ("09:00-02:00" runs past midnight); a start equal to its end means always."""
    t = str(text or "always").strip().lower()
    if t == "always":
        return None
    m = re.fullmatch(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", t)
    if not m or int(m.group(1)) > 23 or int(m.group(3)) > 23 or int(m.group(2)) > 59 or int(m.group(4)) > 59:
        raise ValueError(f'[guard] hours is {text!r}; it takes "always" or a window like "09:00-02:00"')
    start, end = int(m.group(1)) * 60 + int(m.group(2)), int(m.group(3)) * 60 + int(m.group(4))
    return None if start == end else (start, end)


def in_hours(hours, when):
    """True when `when` (a datetime) falls inside the guard's hours."""
    if hours is None:
        return True
    start, end = hours
    m = when.hour * 60 + when.minute
    return start <= m < end if start < end else (m >= start or m < end)


def status_path(settings=None):
    """Where the guard writes where it stands: DEPANGRAM_GUARD_STATUS, else guard.json in RUNS_DIR."""
    if os.environ.get(STATUS_ENV):
        return Path(os.environ[STATUS_ENV])
    try:
        return Path(settings.runs_dir) / "guard.json"
    except Exception:  # noqa: BLE001  (a settings object with no runs folder: no status file)
        return None


# ---------------------------------------------------------------------- priority
def low_priority_on(source=None):
    """[guard] low_priority; read from config.toml when no settings are given. A config that can't be read counts
    as on."""
    try:
        if source is None:
            from .settings import load_config
            source = load_config()
        return bool(config(source)["low_priority"])
    except Exception:  # noqa: BLE001
        return True


def priority_flags(source=None, low=None):
    """The Popen creationflags that start a process at below-normal priority on Windows, or 0."""
    low = low_priority_on(source) if low is None else low
    return BELOW_NORMAL_PRIORITY_CLASS if WINDOWS and low else 0


def lower_process(pid=None):
    """One process (this one by default) down to below-normal priority: the priority class on Windows, nice 10
    elsewhere (never raised back up when it's nicer already). Returns True when it took."""
    try:
        if WINDOWS:
            import psutil
            psutil.Process(pid or os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            return True
        who = pid or 0
        if os.getpriority(os.PRIO_PROCESS, who) < NICE:
            os.setpriority(os.PRIO_PROCESS, who, NICE)
        return True
    except Exception:  # noqa: BLE001  (a priority we can't set leaves the process as it was)
        return False


def lower_self(source=None):
    """This process to below-normal priority when [guard] low_priority is on: what run, regrow and night do first."""
    return lower_process() if low_priority_on(source) else False


def after_start(proc, source=None, low=None):
    """POSIX: a process just started, niced to 10 (Windows gets it at the start, from priority_flags)."""
    low = low_priority_on(source) if low is None else low
    if low and not WINDOWS and getattr(proc, "pid", None):
        lower_process(proc.pid)


# ---------------------------------------------------------------------- the probes
class _LastInput(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def idle_seconds():
    """Seconds since the last keyboard or mouse input in this session, or None when this system can't say.
    Windows: GetLastInputInfo (its tick count is 32 bits, so the gap is taken modulo 2**32). Linux: xprintidle when
    it's installed and there's an X display. A gamepad doesn't count as input to Windows; a game is caught by the
    fullscreen and GPU probes instead."""
    if WINDOWS:
        info = _LastInput()
        info.cbSize = ctypes.sizeof(info)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            raise OSError("GetLastInputInfo failed")
        tick = ctypes.windll.kernel32.GetTickCount
        tick.restype = ctypes.c_uint32
        return ((tick() - info.dwTime) & 0xFFFFFFFF) / 1000.0
    exe = shutil.which("xprintidle")
    if exe and os.environ.get("DISPLAY"):
        out = subprocess.run([exe], capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip()) / 1000.0
    return None


QUNS = {1: "QUNS_NOT_PRESENT", 2: "QUNS_BUSY", 3: "QUNS_RUNNING_D3D_FULL_SCREEN", 4: "QUNS_PRESENTATION_MODE",
        5: "QUNS_ACCEPTS_NOTIFICATIONS", 6: "QUNS_QUIET_TIME", 7: "QUNS_APP"}
# the states that mean something fullscreen is in front. QUNS_NOT_PRESENT (a locked screen or the screensaver) means
# you're away, so it never counts.
FULLSCREEN = {2: "a fullscreen app", 3: "a fullscreen Direct3D game or video", 4: "presentation mode",
              7: "a fullscreen Store app"}


def fullscreen():
    """What fullscreen thing is in front as a short phrase, "" when nothing is, None when this system can't say.
    Windows: SHQueryUserNotificationState, the call Windows itself uses to hold notifications back during a game,
    a fullscreen video or a presentation. Elsewhere: None."""
    if not WINDOWS:
        return None
    state = ctypes.c_int(0)
    hr = ctypes.windll.shell32.SHQueryUserNotificationState(ctypes.byref(state))
    if hr != 0:
        raise OSError(f"SHQueryUserNotificationState failed ({hr & 0xFFFFFFFF:#x})")
    what = FULLSCREEN.get(state.value)
    return f"fullscreen app ({what}, {QUNS[state.value]})" if what else ""


def _quiet_run(cmd, timeout=10):
    kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if WINDOWS else {}
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


def gpu_memory():
    """(MiB used, MiB total) on the first GPU, from nvidia-smi --query-gpu; None when there's no nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    out = _quiet_run([exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    if out.returncode != 0:
        raise OSError(f"nvidia-smi exited {out.returncode}")
    used, total = (float(x) for x in out.stdout.strip().splitlines()[0].split(","))
    return used, total


def nvidia_compute_apps():
    """{pid: MiB} from nvidia-smi --query-compute-apps=pid,used_memory (our own server excluded later, by the
    guard). Only compute apps show here: a game or a video player that only draws never does, which is why the
    fullscreen probe exists. Under Windows' WDDM driver model every row says "[N/A]", so a row with no number is left
    out, and gpu_processes reads Windows' own counters first. None when there's no nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    out = _quiet_run([exe, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    if out.returncode != 0:
        raise OSError(f"nvidia-smi exited {out.returncode}")
    procs = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                procs[int(parts[0])] = procs.get(int(parts[0]), 0.0) + float(parts[1].split()[0])
            except (ValueError, IndexError):
                continue
    return procs


class _PdhValue(ctypes.Union):
    _fields_ = [("longValue", ctypes.c_long), ("doubleValue", ctypes.c_double), ("largeValue", ctypes.c_longlong),
                ("AnsiStringValue", ctypes.c_char_p), ("WideStringValue", ctypes.c_wchar_p)]


class _PdhFmtValue(ctypes.Structure):
    _fields_ = [("CStatus", ctypes.c_ulong), ("u", _PdhValue)]


class _PdhItem(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _PdhFmtValue)]


PDH_FMT_LARGE = 0x00000400
PDH_MORE_DATA = 0x800007D2
PDH_COUNTER = "\\GPU Process Memory(*)\\Dedicated Usage"


def windows_gpu_processes():
    """{pid: MiB of dedicated VRAM} from Windows' "GPU Process Memory" performance counters (PDH, the numbers Task
    Manager shows). They count graphics and compute apps alike and need no admin rights."""
    pdh = ctypes.WinDLL("pdh")
    query, counter = ctypes.c_void_p(), ctypes.c_void_p()

    def ok(rc, what):
        if rc & 0xFFFFFFFF:
            raise OSError(f"{what} failed ({rc & 0xFFFFFFFF:#x})")

    ok(pdh.PdhOpenQueryW(None, None, ctypes.byref(query)), "PdhOpenQueryW")
    try:
        ok(pdh.PdhAddEnglishCounterW(query, ctypes.c_wchar_p(PDH_COUNTER), None, ctypes.byref(counter)),
           "PdhAddEnglishCounterW")
        ok(pdh.PdhCollectQueryData(query), "PdhCollectQueryData")
        size, count = ctypes.c_ulong(0), ctypes.c_ulong(0)
        rc = pdh.PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None)
        if (rc & 0xFFFFFFFF) != PDH_MORE_DATA:
            ok(rc, "PdhGetFormattedCounterArrayW")
            return {}
        buf = (ctypes.c_byte * size.value)()
        ok(pdh.PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buf),
           "PdhGetFormattedCounterArrayW")
        items = ctypes.cast(buf, ctypes.POINTER(_PdhItem))
        procs = {}
        for i in range(count.value):
            m = re.match(r"pid_(\d+)_", items[i].szName or "")
            if m and items[i].FmtValue.CStatus in (0, 1):
                pid = int(m.group(1))
                procs[pid] = procs.get(pid, 0.0) + items[i].FmtValue.u.largeValue / MIB
        return procs
    finally:
        pdh.PdhCloseQuery(query)


def gpu_processes():
    """{pid: MiB of VRAM} for every process on the GPU, or None when this system can't say: Windows' counters
    first (windows_gpu_processes), then nvidia-smi's compute apps (nvidia_compute_apps)."""
    if WINDOWS:
        try:
            return windows_gpu_processes()
        except OSError:
            pass
    return nvidia_compute_apps()


def process_name(pid):
    """A process's executable name ("dwm.exe"), or "" when it can't be read."""
    try:
        import psutil
        return psutil.Process(pid).name()
    except Exception:  # noqa: BLE001
        return ""


def own_family():
    """This process, the processes that started it (the night, the MCP server and its wrapper) and the processes
    it started (the llama server): their VRAM is the engine's own, never another user's."""
    import psutil
    me = psutil.Process()
    pids = {me.pid}
    for group in (me.parents, lambda: me.children(recursive=True)):
        try:
            pids |= {p.pid for p in group()}
        except psutil.Error:
            pass
    return pids


class Probes:
    """The real probes. A test passes any object with the same five methods."""

    def idle(self):
        return idle_seconds()

    def fullscreen(self):
        return fullscreen()

    def gpu(self):
        return gpu_processes()

    def vram(self):
        return gpu_memory()

    def name(self, pid):
        return process_name(pid)

    def family(self):
        return own_family()


# ---------------------------------------------------------------------- the guard
def ago(seconds):
    s = max(0.0, float(seconds))
    return f"{s:.0f}s" if s < 120 else f"{s / 60:.0f} min"


class Guard:
    """The state machine: "waiting" (the server has never started), "running", "paused" (the guard stopped it),
    "stopped" (the run ended it). poll() looks and pauses; the gate (enter/leave) holds requests while paused and
    starts the server again once everything has been clear for idle_minutes. `clock` and `sleep` are the run's own
    (the window loop passes its clock), `now` gives the wall time the hours are read in, `halt` says when a paused
    run's stop time has come."""

    def __init__(self, cfg=None, *, probes=None, clock=None, sleep=None, now=None, log=None, status_path=None,
                 halt=None):
        self.cfg = config(cfg)
        self.enabled = bool(self.cfg["enabled"])
        self.hours = parse_hours(self.cfg["hours"])
        self.poll_s = float(self.cfg["poll_seconds"])
        self.active_s = 60.0 * float(self.cfg["active_minutes"])
        self.idle_s = 60.0 * float(self.cfg["idle_minutes"])
        self.other_mb = float(self.cfg["other_gpu_mb"])
        self.ignore = {str(n).lower() for n in self.cfg.get("ignore_processes") or []}
        self.probes = probes if probes is not None else Probes()
        self.clock = clock or time.time
        self.sleep = sleep or time.sleep
        self.now = now or dt.datetime.now
        self.log = log or (lambda m: print(m, flush=True))
        self.status_path = Path(status_path) if status_path else None
        self.halt = halt
        self.start_fn = self.stop_fn = None
        self.server = None
        self.parked = []            # (park, unpark) for models the run keeps on the card next to the server
        self.off_card = False       # True while a pause has those models on the CPU
        self.state, self.reason, self.since = "waiting", None, None
        self.last, self.clear_s = None, 0.0
        self.quiet_since = None     # since when no fullscreen app and no other GPU user has been seen
        self.next_poll = 0.0
        self.inflight = 0
        self.failed = None
        self.closed = False
        self.cv = threading.Condition()
        self.step_lock = threading.Lock()
        self.watcher, self.watch_stop = None, threading.Event()
        self.warned = set()
        self.events = []            # (state, reason): every start, pause and resume, in order

    # ------------------------------------------------------------------ looking
    def probe(self, name, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  (a probe that fails is unknown, never a crash)
            if name not in self.warned:
                self.warned.add(name)
                self.log(f"guard: the {name} probe failed ({type(e).__name__}: {str(e)[:120]}); it counts as unknown")
            return None

    def others(self):
        """[(MiB, pid, name)] of the other processes at or over other_gpu_mb, biggest first; None when unknown."""
        procs = self.probes.gpu()
        if procs is None:
            return None
        ours = set(self.probe("family", self.probes.family) or ())
        pid = getattr(getattr(self.server, "proc", None), "pid", None)
        if pid:
            ours.add(pid)
        out = []
        for p, mb in procs.items():
            if p in ours or mb < self.other_mb:
                continue
            name = self.probe("name", lambda p=p: self.probes.name(p)) or ""
            if name.lower() in self.ignore:
                continue
            out.append((mb, p, name))
        return sorted(out, reverse=True)

    def read(self):
        """One look at the PC: {"idle": seconds or None, "fullscreen": a phrase or None, "gpu": others()}."""
        r = {"idle": self.probe("idle", self.probes.idle), "fullscreen": None, "gpu": None}
        if self.cfg["pause_on_fullscreen"]:
            r["fullscreen"] = self.probe("fullscreen", self.probes.fullscreen)
        r["gpu"] = self.probe("gpu", self.others)
        return r

    def why(self, r):
        """Why the PC is yours right now, or None when it's clear. An unknown idle time counts as away."""
        if r["idle"] is not None and r["idle"] < self.active_s:
            return f"active: last input {ago(r['idle'])} ago"
        if r.get("fullscreen"):
            return str(r["fullscreen"])
        if r.get("gpu"):
            mb, pid, name = r["gpu"][0]
            return f"another process holds {mb / 1024:.1f} GB ({name or 'pid'} {pid})" if name else \
                f"another process holds {mb / 1024:.1f} GB (pid {pid})"
        return None

    def vram_text(self):
        if self.standing_down():
            return ""                                 # outside its hours the guard reads nothing at all
        got = self.probe("vram", self.probes.vram)
        return f" (VRAM {got[0] / 1024:.1f} of {got[1] / 1024:.1f} GB in use)" if got else ""

    def standing_down(self):
        return not in_hours(self.hours, self.now())

    # ------------------------------------------------------------------ the state machine
    def poll(self, force=False):
        """One look when one is due (every poll_seconds): pauses a running server when the PC is yours, and keeps
        the clear count while paused. Never starts the server; the next request does. Returns the state."""
        if not self.enabled or self.closed:
            return self.state
        if not self.step_lock.acquire(blocking=False):
            return self.state                         # another thread is looking or pausing
        try:
            now = self.clock()
            if not force and now < self.next_poll:
                return self.state
            self.next_poll = now + self.poll_s
            if self.standing_down():
                self.last, self.clear_s = None, float("inf")
                self.write_status()
                return self.state
            r = self.read()
            self.last = r
            if r.get("fullscreen") or r.get("gpu"):
                self.quiet_since = None
            elif self.quiet_since is None:
                self.quiet_since = now
            why = self.why(r)
            if why:
                self.clear_s = 0.0
            else:
                idle = r["idle"] if r["idle"] is not None else float("inf")
                self.clear_s = min(now - self.quiet_since, idle)
            if self.state == "running" and why:
                self.pause(why)
            else:
                if why and self.state in ("paused", "waiting"):
                    self.reason = why
                self.write_status()                   # every look, so a reader can tell a live file from an old one
            return self.state
        finally:
            self.step_lock.release()

    def ready(self):
        """Clear long enough for the server to come back: idle_minutes with nothing busy, or the guard standing
        down outside its hours."""
        return self.standing_down() or self.clear_s >= self.idle_s

    def pause(self, why):
        """Close the gate, let the requests in flight finish, stop the server, move parked models off the card.
        The caller holds step_lock."""
        with self.cv:
            self.state, self.reason, self.since = "paused", why, self.clock()
            while self.inflight > 0:
                self.cv.wait(1.0)
        try:
            for park, _ in self.parked:
                park()
            self.off_card = bool(self.parked)
        except Exception as e:  # noqa: BLE001
            self.log(f"guard: a model couldn't leave the card ({type(e).__name__}: {str(e)[:120]})")
        if self.stop_fn:
            self.stop_fn()
        self.events.append(("paused", why))
        self.log(f"guard: paused, {why}; the llama server is stopped{self.vram_text()}")
        self.write_status()

    def resume(self, why):
        """Start the server again (the caller holds step_lock) and open the gate."""
        was, held = self.state, (self.clock() - self.since) if self.since is not None else 0.0
        try:
            if self.start_fn:
                self.start_fn()
            if self.off_card:
                for _, unpark in self.parked:
                    unpark()
                self.off_card = False
        except Exception as e:
            self.failed = e
            self.log(f"guard: the llama server didn't come back ({type(e).__name__}: {str(e)[:200]})")
            raise
        with self.cv:
            self.state, self.reason, self.since = "running", None, self.clock()
            self.cv.notify_all()
        self.events.append(("running", why))
        if was == "waiting":
            self.log(f"guard: the llama server starts, {why}{self.vram_text()}")
        else:
            self.log(f"guard: resumed after {ago(held)} paused, {why}; the llama server is back{self.vram_text()}")
        self.write_status()

    def try_resume(self):
        """Start the server again when it's time; True when the gate is open after it."""
        with self.step_lock:
            if self.state == "running":
                return True
            if self.closed:
                raise Halted("the run is ending")
            if self.standing_down():
                self.resume(f"outside the guard's hours ({self.cfg['hours']})")
                return True
            if self.ready():
                self.resume(f"clear for {ago(self.clear_s)}" if self.clear_s != float("inf") else "clear")
                return True
        return False

    def wait_running(self, count=False):
        """Hold until the gate is open (counting one request in flight when `count`). Raises Halted when `halt`
        says the run's stop time has come, and the server's start error when it couldn't come back."""
        while True:
            self.poll()
            with self.cv:
                if self.state == "running":
                    if count:
                        self.inflight += 1
                    return
            if self.failed is not None:
                raise self.failed
            if self.try_resume():
                continue
            if self.halt is not None and self.halt():
                if "halt" not in self.warned:          # once: every queued draw of a pool comes through here
                    self.warned.add("halt")
                    self.log(f"guard: the run's stop time came while paused ({self.reason}); the server stays"
                             " stopped")
                raise Halted(self.reason or "paused")
            self.sleep(self.poll_s)

    # ------------------------------------------------------------------ the gate
    def enter(self):
        """Before a request: returns at once while clear, else holds."""
        if not self.enabled:
            return
        self.wait_running(count=True)

    def leave(self):
        if not self.enabled:
            return
        with self.cv:
            self.inflight = max(0, self.inflight - 1)
            self.cv.notify_all()

    # ------------------------------------------------------------------ start and end
    def begin(self, start_fn, stop_fn):
        """The run's first start of the server: at once when the PC is clear, else held until it has been clear for
        idle_minutes. Starts the watcher. Returns what start_fn returned."""
        self.start_fn, self.stop_fn = start_fn, stop_fn
        if not self.enabled:
            self.state = "running"
            return start_fn()
        got = {}
        self.start_fn = lambda: got.setdefault("first", start_fn())
        self.poll(force=True)
        if self.standing_down() or (self.last is not None and not self.why(self.last)):
            with self.step_lock:
                self.resume("the PC is clear" if not self.standing_down()
                            else f"outside the guard's hours ({self.cfg['hours']})")
        else:
            self.reason = self.why(self.last) if self.last else None
            self.since = self.clock()
            self.events.append(("waiting", self.reason))
            self.log(f"guard: the llama server waits to start, {self.reason}")
            self.write_status()
            self.wait_running()
        self.start_fn = start_fn
        self.watch()
        return got.get("first")

    def watch(self):
        """The watcher thread: a poll every poll_seconds, so a pause lands while the engine is away from the server."""
        if not self.enabled or self.watcher is not None:
            return

        def loop():
            while not self.watch_stop.wait(self.poll_s):
                try:
                    self.poll()
                except Exception as e:  # noqa: BLE001
                    if "watcher" not in self.warned:
                        self.warned.add("watcher")
                        self.log(f"guard: the watcher hit {type(e).__name__}: {str(e)[:160]}")

        self.watcher = threading.Thread(target=loop, name="depangram-guard", daemon=True)
        self.watcher.start()

    def close(self):
        """The run is ending: the watcher stops and never starts the server again. The caller stops the server."""
        self.closed = True
        self.watch_stop.set()
        if self.watcher is not None and self.watcher is not threading.current_thread():
            self.watcher.join(self.poll_s + 5)
        with self.cv:
            self.state = "stopped"
            self.cv.notify_all()
        self.write_status()

    # ------------------------------------------------------------------ the status file
    def status(self):
        return {"state": self.state, "reason": self.reason, "since": self.since, "updated": time.time(),
                "pid": os.getpid(), "clear_seconds": None if self.clear_s == float("inf") else round(self.clear_s),
                "idle_minutes": self.cfg["idle_minutes"], "poll_seconds": self.poll_s}

    def write_status(self):
        if self.status_path is None:
            return
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.status_path.with_name(f"{self.status_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(self.status()), encoding="utf-8")
            os.replace(tmp, self.status_path)
        except OSError:
            pass


def read_status(path, fresh_seconds=None):
    """A guard status file as a short line for a person: "paused: <reason>", "waiting to start: <reason>",
    "running", or None when there's no file, it can't be read, or it's older than `fresh_seconds`."""
    try:
        st = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if fresh_seconds is not None:
        fresh = max(float(fresh_seconds), 3 * float(st.get("poll_seconds") or 0))
        if time.time() - float(st.get("updated") or 0) > fresh:
            return None
    state = st.get("state")
    clear = st.get("clear_seconds")
    if state == "paused" and clear:
        # the reason is the last thing that kept it paused; the PC has been clear since
        return (f"paused: {st.get('reason') or 'the PC was in use'} (clear for {ago(clear)} now; the server comes"
                f" back after {float(st.get('idle_minutes') or 0):g} min clear, at the next draw)")
    if state == "paused":
        return f"paused: {st.get('reason') or 'the PC is in use'}"
    if state == "waiting":
        return f"waiting to start: {st.get('reason') or 'the PC is in use'}"
    return state


# ---------------------------------------------------------------------- the wrappers
class GuardedServer:
    """A llama server behind the guard: every call that reaches the server goes through the gate, and start and stop
    go through the guard. Everything else passes through, so ArmServer, draw_pool and the pass take it where they
    take a LlamaServer."""

    GATED = ("complete", "chat", "tokenize", "adapters", "digit_tokens")

    def __init__(self, srv, guard):
        self.srv, self.guard = srv, guard
        guard.server = srv

    def start(self, *a, **kw):
        return self.guard.begin(lambda: self.srv.start(*a, **kw), self.srv.stop)

    def stop(self):
        try:
            self.guard.close()
        finally:
            self.srv.stop()

    def __getattr__(self, name):
        attr = getattr(self.srv, name)
        if name not in self.GATED or not callable(attr):
            return attr

        def gated(*a, **kw):
            self.guard.enter()
            try:
                return attr(*a, **kw)
            finally:
                self.guard.leave()
        return gated


class GuardedModel:
    """A model the run keeps on the card next to the server (Open Pangram): its score() goes through the gate, and
    a pause moves it to the CPU and back, so the pause frees its VRAM too (the CUDA context, a few hundred MB, stays
    with the process)."""

    def __init__(self, model, guard):
        self.model_obj, self.guard = model, guard
        if str(getattr(model, "device", "")).startswith("cuda") and getattr(model, "model", None) is not None:
            guard.parked.append((self.park, self.unpark))

    def park(self):
        self.model_obj.model.to("cpu")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def unpark(self):
        self.model_obj.model.to(self.model_obj.device)

    def score(self, *a, **kw):
        self.guard.enter()
        try:
            return self.model_obj.score(*a, **kw)
        finally:
            self.guard.leave()

    def __getattr__(self, name):
        return getattr(self.model_obj, name)


def guard_server(srv, settings=None, *, log=None, halt=None, clock=None, sleep=None, probes=None):
    """`srv` behind the guard [guard] describes, or `srv` itself when the guard is off. Only the real runs call this
    (regrow's main and run.py's real_parts): the rehearsal and the tests keep their fakes unguarded."""
    cfg = config(settings)
    if not cfg["enabled"]:
        return srv
    return GuardedServer(srv, Guard(cfg, log=log, halt=halt, clock=clock, sleep=sleep, probes=probes,
                                    status_path=status_path(settings)))


def guard_model(srv, model):
    """`model` behind the same guard as `srv`, when `srv` is guarded; else `model` as it is."""
    g = getattr(srv, "guard", None) if isinstance(srv, GuardedServer) else None
    return GuardedModel(model, g) if g is not None and model is not None else model


def guard_of(srv):
    return srv.guard if isinstance(srv, GuardedServer) else None


if __name__ == "__main__":       # python -m engine.guard: one look with the real probes, for checking a machine
    sys.stdout.reconfigure(encoding="utf-8")
    g = Guard({"enabled": True})
    g.poll(force=True)
    print(json.dumps({"reading": g.last, "why": g.why(g.last) if g.last else None, "vram": g.probe("vram", g.probes.vram)},
                     default=str, indent=1))
