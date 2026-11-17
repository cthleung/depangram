"""Pangram client with a ledger, a cache and hard caps (plan 7.1).

Only this module ever calls Pangram, and Claude never gets access to it.
Every call is checked against the caps before it's sent, written to the
ledger, and cached by its exact text, so the same text is never paid twice.

Every answer carries Pangram's model version ("4.1"), which goes in the
ledger's version column. An answer whose version differs from the last one
stored is a model change (plan 9): the run loop alerts on it, and a cached
answer from another version stops counting as a hit, so the text is scored
again by the model Pangram runs now. The change stays unacknowledged (the
banked scores stale) until config.toml's [pangram] model_ack names the new
version.
"""
import hashlib
import json
import math
import time
from datetime import datetime

import requests


class CapReached(Exception):
    pass


class PangramError(Exception):
    pass


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_count(text):
    return len(text.split())


def credits_for(text, words_per_credit=100):
    """One credit per 100 words, rounded up, at least one."""
    return max(1, math.ceil(word_count(text) / words_per_credit))


def ensure_version_column(conn):
    """The ledger's version column (added 2026-11-09), filled in once for the rows already there from their cached
    answers, so "the last version stored" means something from the first call on."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(pangram_calls)")]
    if "version" in cols:
        return
    conn.execute("ALTER TABLE pangram_calls ADD COLUMN version TEXT")
    conn.execute("UPDATE pangram_calls SET version = (SELECT json_extract(k.response_json, '$.version')"
                 " FROM pangram_cache k WHERE k.sha = pangram_calls.sha AND k.model = pangram_calls.model)"
                 " WHERE status = 'ok'")
    conn.commit()


def summarize(resp):
    """The fields the analytics care about, from one Pangram response."""
    windows = resp.get("windows") or []
    scores = [w.get("ai_assistance_score") for w in windows if w.get("ai_assistance_score") is not None]
    worst = max(windows, key=lambda w: w.get("ai_assistance_score") or 0) if windows else {}
    humanizer = [w.get("humanizer_score") for w in windows if w.get("humanizer_score") is not None]
    return {
        "prediction": resp.get("prediction_short") or resp.get("headline"),
        "fraction_ai": resp.get("fraction_ai"),
        "fraction_ai_assisted": resp.get("fraction_ai_assisted"),
        "fraction_human": resp.get("fraction_human"),
        "max_window_score": max(scores) if scores else None,
        "humanized": any(w.get("is_humanized") for w in windows),
        "windows": len(windows),
        "version": resp.get("version"),
        # per window in 4.1: [human, AI-assisted, AI]; kept for the worst window
        "class_probabilities": worst.get("class_probabilities"),
        "calibrated_confidence": worst.get("calibrated_confidence"),
        "max_humanizer_score": max(humanizer) if humanizer else None,
    }


class Pangram:
    def __init__(self, settings, conn, session=None):
        self.settings = settings
        self.conn = conn
        self.cfg = settings.config["pangram"]
        self.key = settings.get("PANGRAM_API_KEY")
        self.model = settings.get("PANGRAM_MODEL", "pangram-4.1")
        self.base = settings.get("PANGRAM_BASE_URL", "https://text.external-api.pangram.com").rstrip("/")
        self.http = session or requests.Session()
        self.model_change = None           # (old, new) once a call answers with a new model version (plan 9)
        ensure_version_column(conn)

    # --- ledger -----------------------------------------------------------
    def spent_usd(self, since_ts, purpose=None):
        # every row that carries a cost counts, whatever its status: a task created and then failed or timed
        # out is logged with its credits, and may well have been charged (M1 review, finding 5)
        q = "SELECT COALESCE(SUM(usd), 0) FROM pangram_calls WHERE ts >= ? AND usd > 0"
        args = [since_ts]
        if purpose:
            q += " AND purpose = ?"
            args.append(purpose)
        return self.conn.execute(q, args).fetchone()[0]

    def check_caps(self, usd, purpose, now=None):
        now = now or time.time()
        dt = datetime.fromtimestamp(now)
        month_start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
        if self.spent_usd(month_start) + usd > self.cfg["monthly_cap_usd"]:
            raise CapReached(f"monthly cap ${self.cfg['monthly_cap_usd']:.2f} would be crossed")
        if purpose == "calibration":
            if self.spent_usd(0, "calibration") + usd > self.cfg["calibration_cap_usd"]:
                raise CapReached(f"calibration cap ${self.cfg['calibration_cap_usd']:.2f} would be crossed")
        if purpose == "loop":
            # the night runs 02:00 to 09:00, so "tonight" starts at 00:00 local
            night_start = dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            if self.spent_usd(night_start, "loop") + usd > self.cfg["nightly_cap_usd"]:
                raise CapReached(f"nightly cap ${self.cfg['nightly_cap_usd']:.2f} would be crossed")

    def _log(self, sha, words, credits, usd, purpose, label, status, version=None):
        self.conn.execute(
            "INSERT INTO pangram_calls (ts, sha, model, words, credits, usd, purpose, label, status, version)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (time.time(), sha, self.model, words, credits, usd, purpose, label, status, version),
        )
        self.conn.commit()

    # --- the model version (plan 9) -----------------------------------------
    def version_runs(self):
        """[(version, first ts, last ts)] for each stretch of calls that answered with one version, oldest first."""
        runs = []
        for v, ts in self.conn.execute("SELECT version, ts FROM pangram_calls WHERE version IS NOT NULL"
                                       " ORDER BY ts, id"):
            if runs and runs[-1][0] == v:
                runs[-1][2] = ts
            else:
                runs.append([v, ts, ts])
        return [tuple(r) for r in runs]

    def last_version(self):
        row = self.conn.execute("SELECT version FROM pangram_calls WHERE version IS NOT NULL"
                                " ORDER BY ts DESC, id DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def model_status(self):
        """{latest, trusted, changed_at, stale}. `trusted` is [pangram] model_ack, else the first version the ledger
        saw. Stale means the ledger has seen the version change and the latest isn't the one you've acknowledged;
        `changed_at` is when the latest version's stretch began."""
        runs = self.version_runs()
        trusted = str(self.cfg.get("model_ack") or "").strip() or (runs[0][0] if runs else None)
        if not runs:
            return {"latest": None, "trusted": trusted, "changed_at": None, "stale": False}
        latest = runs[-1][0]
        stale = len({r[0] for r in runs}) > 1 and latest != trusted
        return {"latest": latest, "trusted": trusted, "changed_at": runs[-1][1] if stale else None, "stale": stale}

    # --- calls ------------------------------------------------------------
    def models(self):
        r = self.http.get(f"{self.base}/models", headers={"x-api-key": self.key}, timeout=30)
        r.raise_for_status()
        return r.json().get("models", [])

    def cached(self, text):
        """The cached answer for exactly this text, unless another model version gave it: once Pangram answers with
        a new version, an old answer is a miss and the text gets scored again."""
        row = self.conn.execute(
            "SELECT response_json FROM pangram_cache WHERE sha = ? AND model = ?",
            (text_sha(text), self.model),
        ).fetchone()
        if not row:
            return None
        resp = json.loads(row[0])
        latest = self.last_version()
        if latest and resp.get("version") and resp["version"] != latest:
            return None
        return resp

    def estimate(self, text):
        credits = credits_for(text, self.cfg["words_per_credit"])
        return credits, credits * self.cfg["usd_per_credit"]

    def score(self, text, purpose, label=""):
        """Score one text. Returns (response, from_cache)."""
        hit = self.cached(text)
        if hit is not None:
            return hit, True
        if not self.key:
            raise PangramError("PANGRAM_API_KEY is empty in .env")
        credits, usd = self.estimate(text)
        self.check_caps(usd, purpose)
        sha = text_sha(text)
        before = self.last_version()
        headers = {"x-api-key": self.key, "Content-Type": "application/json"}
        body = {"text": text, "model": self.model, "public_dashboard_link": False}
        r = self.http.post(f"{self.base}/task", headers=headers, json=body, timeout=60)
        if r.status_code >= 400:
            self._log(sha, word_count(text), 0, 0.0, purpose, label, f"http_{r.status_code}")
            raise PangramError(f"create task failed: {r.status_code} {r.text[:300]}")
        task_id = r.json()["task_id"]
        deadline = time.time() + self.cfg["timeout_seconds"]
        while True:
            g = self.http.get(f"{self.base}/task/{task_id}", headers={"x-api-key": self.key}, timeout=60)
            g.raise_for_status()
            resp = g.json()
            stage = resp.get("stage")
            if stage == "STAGE_SUCCESS":
                break
            if stage == "STAGE_FAILED":
                self._log(sha, word_count(text), credits, usd, purpose, label, "failed")
                raise PangramError(f"task {task_id} failed: {json.dumps(resp)[:300]}")
            if time.time() > deadline:
                self._log(sha, word_count(text), credits, usd, purpose, label, "timeout")
                raise PangramError(f"task {task_id} timed out")
            time.sleep(self.cfg["poll_seconds"])
        version = resp.get("version")
        self._log(sha, word_count(text), credits, usd, purpose, label, "ok", version=version)
        if before and version and version != before:
            self.model_change = (before, version)
        self.conn.execute(
            "INSERT OR REPLACE INTO pangram_cache (sha, model, response_json, created_at) VALUES (?,?,?,?)",
            (sha, self.model, json.dumps(resp), time.time()),
        )
        self.conn.commit()
        return resp, False
