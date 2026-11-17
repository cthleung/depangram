"""The rehearsal (plan 12, M1's acceptance test): a whole run-day on a compressed clock, every paid call faked.

    python -m engine.run --rehearse                       # the default scenario
    python -m engine.run --rehearse --scenario busy-user  # you use Claude at 9:05, so the day cap bites
    python -m engine.run --rehearse --scenario crash      # the llama server goes away at 03:30 for a while
    python -m engine.run --rehearse --scenario crash-loop # ... and never comes back: five restarts end the day

Everything runs in its own folder (C:\\depangram\\runs\\rehearsal\\): its own SQLite ledger, its own copy of the
job folder, its own bank, its own STOP file and a discord.log instead of the channel (--discord-live posts too,
with a [rehearsal] prefix). The real bank, the real ledger and the real working copy are never touched.

What's real: the run loop, the capper, the schedule, the filter stack, the bank, the posts. What's fake:
  - Pangram: a stand-in behind engine/pangram.py's own client (so its cache, ledger and caps run for real),
    charging the real price. It flags sentences Claude wrote and sometimes the line next to one, and passes
    most regrown sentences, so scores trend down as the loop works.
  - Sonnet: canned notes, verdicts and fact checks, with made-up costs and rate_limit readings that climb
    with spend (and with "your" use in some scenarios), written to the ledger like real calls.
  - The llama server: draws made from the original run by seeded edits; a share of them trip each filter.
  - Open Pangram: a hash.
Each fake moves the clock on by about how long the real thing takes.
"""
import datetime as dt
import hashlib
import json
import random
import re
import shutil
import time
from pathlib import Path

import requests

from .checks import hedges_in, repeats_in
from .claude_call import ensure_schema

COSTS = {"judge": 0.035, "h2h": 0.02, "facts": 0.02, "notes": 0.25, "drop": 0.03, "probe": 0.002, "protect": 0.04}
SECONDS = {"judge": 9, "h2h": 5, "facts": 5, "notes": 30, "drop": 6, "probe": 3, "protect": 8}
DRAW_SECONDS = 0.6          # per completion; the real server runs two at a time
PANGRAM_SECONDS = 4

SCENARIOS = {
    # the night's 5-hour cap bites, the window rolls over, Pangram's daily cap bites, local work to 16:00
    "default": {"week_base": 0.30, "reset_days": 3, "week_rate": 0.007, "five_rate": 0.12, "regrown_mod": 3},
    # cheap Claude and a fussy Pangram, so regrown lines keep getting flagged and re-rolled: Pangram's daily cap bites
    "pangram-day": {"week_base": 0.30, "reset_days": 3, "week_rate": 0.007, "five_rate": 0.02, "regrown_mod": 1},
    # the loop starts at 8:58 (a restart in the day), you open Claude at 9:05 and use a big chunk of the 5-hour
    # window, so the day phase's 50% cap bites
    "busy-user": {"clock": "08:58", "week_base": 0.30, "reset_days": 3, "week_rate": 0.007, "five_rate": 0.03,
                  "user": [("09:05", 0.40, 0.02)]},
    # your week is already near the 90% ceiling: Claude stops at the first check, local work all day
    "full-week": {"week_base": 0.88, "reset_days": 3, "week_rate": 0.007, "five_rate": 0.03},
    # the loop spent most of its weekly share on earlier nights, so today's allowance bites
    "low-share": {"week_base": 0.40, "reset_days": 3, "week_rate": 0.007, "five_rate": 0.03, "loop_used_usd": 50.0},
    # the loop's weekly share is already spent on earlier nights: the share check says no at the first call
    "share-spent": {"week_base": 0.60, "reset_days": 3, "week_rate": 0.007, "five_rate": 0.03, "loop_used_usd": 80.0},
    # dear calls: the first-run $20 backstop bites before anything else
    "backstop": {"week_base": 0.20, "reset_days": 1, "week_rate": 0.001, "five_rate": 0.004, "cost_scale": 8.0},
    # the llama server goes away at 03:30 for five minutes: the retries fail, the day restarts from the bank ten
    # minutes later and carries on (M1 review, finding 2)
    "crash": {"week_base": 0.30, "reset_days": 3, "week_rate": 0.007, "five_rate": 0.03, "crash_at": "03:30",
              "crash_minutes": 5},
    # it never comes back: five restarts in a row with no round between them end the day with a crash alert
    # (02:20 since M3: by 03:30 the whole-text round has spent the Pangram day, so nothing draws to hit the crash)
    "crash-loop": {"week_base": 0.30, "reset_days": 3, "week_rate": 0.007, "five_rate": 0.03, "crash_at": "02:20",
                   "crash_minutes": 900},
    # Pangram answers 502 from 03:00 for a few minutes: retries, then a ten-minute hold, then on with the day
    "pangram-502": {"week_base": 0.30, "reset_days": 3, "week_rate": 0.007, "five_rate": 0.03, "pangram_down_at": "03:00",
                    "pangram_down_minutes": 4},
    # the STOP file appears at 06:00
    "stop-file": {"week_base": 0.30, "reset_days": 3, "week_rate": 0.007, "five_rate": 0.03, "stop_at": "06:00"},
    # plan 5.1: Pangram flags some of your own lines (about one in three), so they get notes, protected words and
    # regrowth; one line in five gets no usable protected word from Sonnet and takes the code fallback
    "author-flip": {"week_base": 0.30, "reset_days": 3, "week_rate": 0.007, "five_rate": 0.03, "author_mod": 3},
}


def default_start(scenario=None):
    d = dt.date.today()
    hh, mm = map(int, (scenario or {}).get("clock", "01:58").split(":"))
    return dt.datetime(d.year, d.month, d.day, hh, mm).timestamp()


def h(*parts):
    return int(hashlib.sha1("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[:12], 16)


def prepare_dir(base, rdir, job, keep=False, live=False):
    """A fresh rehearsal folder (or the last one, with --keep) and the env that points the loop at it. `job` is the
    job's name, a folder in jobs/. The job's own files are copied in (the ones job.toml names inside the job folder:
    the original, your lines, the samples, the example and protected lines, the seed banks) with the notes cache."""
    from .job import Job
    from .settings import SANDBOX
    rdir = Path(rdir)
    if rdir.exists() and not keep:
        shutil.rmtree(rdir)
    jdir = rdir / "jobs" / job
    jdir.mkdir(parents=True, exist_ok=True)
    src = SANDBOX / "jobs" / job
    source = Job(root=src)
    if not (jdir / "job.toml").exists():
        files = [src / "job.toml", source.original, source.your_lines, source.notes_cache, source.examples_file,
                 source.protected_file] + ([source.samples_file] if source.samples_file else [])
        files += [src / n for n in source.cfg.get("seed", {}).get("banks", [])]
        for f in files:
            try:
                rel = Path(f).resolve().relative_to(src.resolve())
            except ValueError:
                continue                       # outside the job folder (an absolute samples file): read in place
            if Path(f).exists():
                (jdir / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, jdir / rel)
    real = source.real_file
    if real and not (rdir / real.name).exists():
        # the wrap-up writes into this stand-in, never the real document. It starts as the job's frozen original,
        # since a rehearsal's bank starts empty and wouldn't know the text the real document holds now
        shutil.copy2(Job(root=jdir).original, rdir / real.name)
    env = dict(base.env)
    env.update({"RUNS_DIR": str(rdir), "STOP_FILE": str(rdir / "STOP"), "PANGRAM_API_KEY": "rehearsal-fake",
                "CLAUDE_CODE_OAUTH_TOKEN": "rehearsal-fake"})
    if not live:
        env["DISCORD_WEBHOOK_URL"] = "rehearsal-no-post"
        env["DISCORD_BOT_TOKEN"] = ""              # the bot path (plan 9.1) off too, so nothing can post
    return env


def tomllib_load(path):
    import tomllib
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def at(clock_start, hm):
    d = dt.datetime.fromtimestamp(clock_start)
    hh, mm = map(int, hm.split(":"))
    return d.replace(hour=hh, minute=mm, second=0, microsecond=0).timestamp()


class Fakes:
    def __init__(self, clock, conn, scenario, rdir):
        self.clock, self.conn, self.sc, self.rdir = clock, conn, scenario, Path(rdir)
        self.loop = None
        self.t0 = clock.now()
        # the week's reset, kept in the folder so a --keep run (the next day) sees the same week
        pin = self.rdir / "week reset.txt"
        self.reset = (float(pin.read_text()) if pin.exists()
                      else at(self.t0, "14:00") + 86400 * scenario.get("reset_days", 3))
        pin.write_text(str(self.reset))
        self.five_start = None
        self.user_done = set()
        self.hint_text, self.hint_wid = "", ""
        self.pangram_http = FakePangramHttp(self)
        self.claude = FakeClaude(self)
        ensure_schema(conn)
        if scenario.get("loop_used_usd") and not conn.execute(
                "SELECT COUNT(*) FROM claude_calls WHERE role = 'loop earlier nights'").fetchone()[0]:
            # spend on earlier nights this week, for the allowance scenario
            ts = self.t0 - 2 * 86400
            conn.execute("INSERT INTO claude_calls (ts, role, model, ok, cost_usd, week_util, five_hour_util,"
                         " week_resets_at, five_hour_resets_at) VALUES (?,?,?,?,?,?,?,?,?)",
                         (ts, "loop earlier nights", "sonnet", 1, scenario["loop_used_usd"],
                          scenario["week_base"], 0.1, int(self.reset), int(ts + 3600)))
            conn.commit()

    # --------------------------------------------------------------- factories the loop calls
    def server(self):
        return FakeServer(self)

    def open_pangram(self):
        return FakeOpenPangram(self)

    # --------------------------------------------------------------- shared
    def claude_sentences(self):
        """Sentences that came from Claude in the original windows (anything not yours), for the fake Pangram."""
        if getattr(self, "_claude_sents", None) is None:
            from engine.draws import sents
            vm = self.loop.job.verdict_map()
            out = set()
            for w in self.loop.job.windows():
                for p in w["text"].split("\n\n"):
                    for s in sents(p):
                        if vm.get(s) not in ("yours", "yours?") and not s.endswith("?") and len(s.split()) > 2:
                            out.add(s)
            self._claude_sents = out
        return self._claude_sents

    def author_sentences(self):
        """Your sentences in the original windows, for the author-flip scenario's fake Pangram."""
        if getattr(self, "_author_sents", None) is None:
            from engine.draws import sents
            vm = self.loop.job.verdict_map()
            self._author_sents = {s for w in self.loop.job.windows() for p in w["text"].split("\n\n") for s in sents(p)
                                  if vm.get(s) in ("yours", "yours?") and len(s.split()) > 4}
        return self._author_sents

    def events(self):
        """Scenario events keyed to the clock: the STOP file."""
        now = self.clock.now()
        if self.sc.get("stop_at") and now >= at(self.t0, self.sc["stop_at"]) and not (self.rdir / "STOP").exists() \
                and not (self.rdir / "stop.done").exists():
            (self.rdir / "STOP").write_text("rehearsal", encoding="utf-8")
            (self.rdir / "stop.done").write_text("", encoding="utf-8")


class FakeClaude:
    """Canned Sonnet. Writes each call to claude_calls with a made-up cost and readings that climb with spend."""

    def __init__(self, fakes):
        self.f = fakes

    def hint(self, orig_text, wid):
        self.f.hint_text, self.f.hint_wid = orig_text, wid

    def readings(self, cost):
        f, sc, now = self.f, self.f.sc, self.f.clock.now()
        if f.five_start is None or now >= f.five_start + 5 * 3600:
            f.five_start = now
        spent = self.f.conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM claude_calls WHERE ts >= ?",
                                    (f.t0,)).fetchone()[0] + cost
        in_five = self.f.conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM claude_calls WHERE ts >= ?",
                                      (f.five_start,)).fetchone()[0] + cost
        user_week = user_five = 0.0
        for hm, five_pts, week_pts in sc.get("user", []):
            t = at(f.t0, hm)
            if now >= t:
                user_week += week_pts
                if t >= f.five_start:
                    user_five += five_pts
        week = sc["week_base"] + sc["week_rate"] * spent + user_week
        five = sc["five_rate"] * in_five + user_five
        return int(week * 100) / 100, int(five * 100) / 100, int(f.reset), int(f.five_start + 5 * 3600)

    def __call__(self, settings, conn, prompt, role, model="sonnet", max_budget_usd=0.5, **kw):
        kind = role.split()[-1]
        rng = random.Random(h(prompt, role))
        cost = COSTS.get(kind, 0.02) * self.f.sc.get("cost_scale", 1.0) * (0.8 + 0.4 * rng.random())
        self.f.clock.advance(SECONDS.get(kind, 5))
        text = getattr(self, "_" + kind, self._other)(prompt, rng)
        week, five, wr, fr = self.readings(cost)
        conn.execute(
            "INSERT INTO claude_calls (ts, role, model, ok, seconds, input_tokens, output_tokens, cache_read_tokens,"
            " cache_write_tokens, cost_usd, week_util, five_hour_util, week_resets_at, error, five_hour_resets_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.f.clock.now(), role, model, 1, SECONDS.get(kind, 5), len(prompt) // 4, len(text) // 4, 0, 0, cost,
             week, five, wr, None, fr))
        conn.commit()
        return text, {"cost_usd": cost, "week_util": week, "five_hour_util": five}

    # the canned answers
    def _notes(self, prompt, rng):
        from engine.draws import sents
        listing = prompt.split("\n\n", 1)[1] if "\n\n" in prompt else prompt
        paras = re.split(r"\n\n(?=\d+\. )", listing[listing.rfind("\n\n1. ") + 2:] if "\n\n1. " in listing else listing)
        out = []
        for i, p in enumerate(paras):
            p = re.sub(r"^\d+\.\s", "", p.strip())
            bullets = [" ".join(s.split()[:12]) for s in sents(p)[:5]] or ["a point"]
            keep = [f'"{x}"' for x in hedges_in(p)] + [f'"{w}, and {w}, and {w}"' for w, _ in repeats_in(p)]
            if keep:
                bullets.append("keep: " + ", ".join(keep))
            out.append(f"### {i + 1}\n" + "\n".join(f"- {b}" for b in bullets))
        return "\n".join(out)

    def _orig(self, prompt):
        return self.f.hint_text or (re.search(r"Original stretch:\n(.*?)\n\n", prompt, re.S) or [None, ""])[1]

    def _judge(self, prompt, rng):
        orig = self._orig(prompt)
        n = len(re.findall(r"^\d+\. ", prompt.split("Candidates", 1)[-1], re.M))
        hard = h("hard", orig) % 7 == 0                     # a run no route ever cracks: stuck rule and drop rule fodder
        if hard or n == 0:
            return json.dumps({"pick": None, "drops": [], "why": "none keeps the joke (rehearsal)"})
        if rng.random() < 0.15:
            return json.dumps({"pick": 1, "drops": ["the second point (rehearsal)"], "why": "drops a point"})
        return json.dumps({"pick": 1 + int(rng.random() * min(n, 3)), "drops": [], "why": "holds the points (rehearsal)"})

    def _h2h(self, prompt, rng):
        return "SAME" if rng.random() < 0.6 else rng.choice(["A", "B"])

    def _facts(self, prompt, rng):
        orig = self._orig(prompt)
        if re.search(r"\d", orig) and h("facty", orig) % 5 == 1:
            return "The rewrite changes the number (rehearsal)."
        return "NONE"

    def _drop(self, prompt, rng):
        return json.dumps({"pass": h("drop", self._orig(prompt)) % 3 != 0, "lost": [], "why": "the paragraph holds (rehearsal)"})

    def _protect(self, prompt, rng):
        """One or two of the line's longest words, or nothing for one line in five (the code fallback's case)."""
        listing = prompt.split("\n\n1. ", 1)[1] if "\n\n1. " in prompt else ""
        lines = re.split(r"\n\n(?=\d+\. )", "1. " + listing) if listing else []
        out = []
        for i, line in enumerate(lines):
            line = re.sub(r"^\d+\.\s", "", line.strip())
            words = sorted({w for w in re.findall(r"[a-z][a-z-]{4,}", re.sub(r'"[^"]*"', " ", line))},
                           key=lambda w: (-len(w), w))
            pick = [] if h("protect", line) % 5 == 0 else words[:1 + h("n", line) % 2]
            out.append(f"### {i + 1}\n" + ", ".join(pick))
        return "\n".join(out)

    def _probe(self, prompt, rng):
        return "ok"

    def _other(self, prompt, rng):
        return "NONE"


class FakeServer:
    """Canned Mistral: each draw is the original run with seeded edits; some trip a filter on purpose. It says it
    loads the adapter (has_lora), so the loop passes each arm's LoRA scale, and for one run in four the adapter arm
    (scale 1.0) returns nothing at all, so the few-shot fallback fires (the writer arms, 2026-11-09)."""
    has_lora = True
    SWAPS = {"really": "seriously", "very": "super", "big": "huge", "just": "simply", "basically": "pretty much",
             "shows": "turns", "thing": "deal", "lot": "ton", "pretty": "fairly", "model": "bot", "word": "term"}

    def __init__(self, fakes):
        self.f = fakes
        self.orig = ""

    def start(self):
        self.f.clock.advance(60)
        return 60

    def stop(self):
        pass

    def digit_tokens(self):
        return [15, 16, 17]

    def hint(self, orig_text, wid):
        self.orig = orig_text
        self.f.hint_text, self.f.hint_wid = orig_text, wid

    def _variant(self, text, rng):
        parts = re.split(r"(\"[^\"]*\")", text)          # quoted pieces stay as they are
        out = []
        for i, part in enumerate(parts):
            if i % 2:
                out.append(part)
                continue
            ws = part.split(" ")
            for k, w in enumerate(ws):
                core = w.strip(",.;:!?").lower()
                if core in self.SWAPS and rng.random() < 0.7 and (i or k >= 4):
                    ws[k] = w.lower().replace(core, self.SWAPS[core])
            out.append(" ".join(ws))
        new = "".join(out)
        if new == text and text:
            # nothing to swap: a small change anyway, as a real draw always makes one
            m = re.search(r"[.!?]?$", new)
            new = new[:m.start()] + rng.choice([", really", ", if you ask me", ", more or less", " though"]) + new[m.start():]
        return new

    def complete(self, prompt, n_predict=220, temperature=1.0, min_p=0.05, seed=-1, stop=None, ban=None,
                 lora_scale=None):
        f = self.f
        f.clock.advance(DRAW_SECONDS)
        crash = f.sc.get("crash_at")
        if crash and 0 <= f.clock.now() - at(f.t0, crash) < 60 * f.sc.get("crash_minutes", 5):
            raise requests.ConnectionError("rehearsal: llama-server went away")
        rng = random.Random(seed)
        orig = self.orig
        if lora_scale == 1.0 and h("adapter-dud", orig) % 4 == 0:
            return "", {}                                # the adapter passes no draw for this run: the fallback's case
        if stop and ":" in stop:                         # the long-quote lead-in: up to the colon
            lead = orig.split('"')[0].rstrip(" :")
            text = self._variant(lead, rng)
        else:
            # the first four words stay as written (the longest starter the loop uses); the edits go after them
            ow = orig.split()
            head = " ".join(ow[:4])
            tail = self._variant(" ".join(ow[4:]), rng)
            r = rng.random()
            if r < 0.06:
                tail = tail.replace(" ", " " + chr(0x2014) + " ", 1)       # an em dash, which the filter must catch
            elif r < 0.12:
                tail = tail.rstrip(".") + ", and I've seen it do this."
            elif r < 0.17:
                tail = tail.rstrip(".") + " about 99 times."
            elif r < 0.24:
                for hdg in hedges_in(orig):
                    tail = re.sub(r"\b{}\b\s*".format(re.escape(hdg)), "", tail, count=1, flags=re.I)
                for w, _ in repeats_in(orig):
                    tail = re.sub(r"(,? and {0})+".format(w), "", tail, flags=re.I)
            elif r < 0.28:
                tail = ""
            elif r < 0.33:
                tail = tail.rstrip(".") + ". Honestly, at the end of the day, that's it."
            text = (head + " " + tail).strip()
        # the prompt ends with the starter the loop chose (a prefix of the original): carry on after it
        ow = orig.split()
        for n in range(min(4, len(ow)), 0, -1):
            start = " ".join(ow[:n]) + " "
            if prompt.endswith(start) and text.startswith(start):
                return text[len(start):], {}
        return text, {}


class FakeOpenPangram:
    def __init__(self, fakes):
        self.f = fakes

    def score(self, texts, batch=16):
        self.f.clock.advance(0.05 * len(texts))
        return [{"score": (h("op", t) % 1000) / 1000, "bucket": 0, "chunks": []} for t in texts]

    def unload(self):
        pass


class FakeResp:
    def __init__(self, data, code=200):
        self.status_code, self._d, self.text = code, data, json.dumps(data)

    def json(self):
        return self._d

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakePangramHttp:
    """Answers engine/pangram.py's two requests (create a task, read it back) with a made-up verdict."""

    def __init__(self, fakes):
        self.f = fakes
        self.tasks = {}

    def post(self, url, headers=None, json=None, timeout=None):
        down = self.f.sc.get("pangram_down_at")
        if down and 0 <= self.f.clock.now() - at(self.f.t0, down) < 60 * self.f.sc.get("pangram_down_minutes", 4):
            self.f.clock.advance(PANGRAM_SECONDS)
            return FakeResp({"detail": "Bad Gateway (rehearsal)"}, 502)
        tid = f"t{len(self.tasks) + 1}"
        self.tasks[tid] = json["text"]
        return FakeResp({"task_id": tid})

    def get(self, url, headers=None, timeout=None):
        self.f.clock.advance(PANGRAM_SECONDS)
        if url.endswith("/models"):
            return FakeResp({"models": ["pangram-4.1"]})
        return FakeResp(self.verdict(self.tasks[url.rsplit("/", 1)[1]]))

    def verdict(self, text):
        from engine.draws import sents
        claude = self.f.claude_sentences()
        segs = []
        for p in text.split("\n\n"):
            ss = sents(p) if not re.match(r"^\d+\.\s", p) else [p]
            author_mod = self.f.sc.get("author_mod")
            for s in ss:
                segs.append([s, s in claude or (h("regrown", s) % self.f.sc.get("regrown_mod", 9) == 0
                                                and len(s.split()) > 5)
                             or bool(author_mod and s in self.f.author_sentences() and h("author", s) % author_mod == 0)])
        # spillover: the line next to a Claude line sometimes flags too
        for i, (s, f) in enumerate(segs):
            if not f and any(0 <= j < len(segs) and segs[j][0] in claude for j in (i - 1, i + 1)) and h("spill", s) % 2:
                segs[i][1] = True
        words = sum(len(s.split()) for s, _ in segs) or 1
        ai = sum(len(s.split()) for s, f in segs if f) / words
        return {"stage": "STAGE_SUCCESS", "fraction_ai": ai, "fraction_human": 1 - ai, "fraction_ai_assisted": 0.0,
                "prediction_short": "AI" if ai > 0.5 else ("Mixed" if ai > 0 else "Human"), "version": "rehearsal",
                "windows": [{"text": " " + s, "ai_assistance_score": 0.95 if f else 0.05, "is_humanized": False,
                             "humanizer_score": 0.0} for s, f in segs]}
