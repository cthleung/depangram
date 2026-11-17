"""Tests for running on any machine and any setup: the path settings and their defaults, the config file and its
documented example, and the Claude API backend behind claude_call.run (a fake client, never a live call). Each
check has a flag case and a pass case. Run from the repo root:

    python -m unittest discover tests
"""
import contextlib
import io
import os
import sqlite3
import tempfile
import tomllib
import types
import unittest
from pathlib import Path
from unittest import mock

from engine import claude_call, settings as st
from engine.settings import Settings

HERE = Path(__file__).resolve().parents[1]


class Paths(unittest.TestCase):
    def test_every_path_has_a_default_with_no_machine_path_in_it(self):
        env = {}
        with mock.patch.dict(os.environ, {}, clear=True):
            for name in ("RUNS_DIR", "WORK_DIR", "MODELS_DIR", "MODELS_FILE", "LLAMA_CPP_DIR", "OPEN_PANGRAM_DIR",
                         "STOP_FILE", "NOTIFY_DIR"):
                got = st.setting_path(name, env)
                self.assertIsInstance(got, Path, name)
                self.assertNotIn("depangram\\runs", str(got))
            self.assertEqual(st.setting_path("RUNS_DIR", env), st.SANDBOX / "runs")
            self.assertEqual(st.setting_path("STOP_FILE", env), st.SANDBOX / "runs" / "STOP")
            self.assertEqual(st.setting_path("WORK_DIR", env), st.USER_HOME / "work")
            self.assertEqual(st.USER_HOME.name, ".depangram")
            self.assertIsNone(st.setting_path("SAMPLES_FILE", env))                          # optional: no default

    def test_a_set_value_wins_and_the_defaults_follow_it(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}, clear=True):
            env = {"RUNS_DIR": d, "MODELS_DIR": str(Path(d) / "m")}
            self.assertEqual(st.setting_path("STOP_FILE", env), Path(d) / "STOP")             # follows RUNS_DIR
            self.assertEqual(st.setting_path("OPEN_PANGRAM_DIR", env), Path(d) / "m" / "open-pangram")
            s = Settings(env=dict(env, STOP_FILE=str(Path(d) / "halt")), config={})
            self.assertEqual(s.stop_file, Path(d) / "halt")
            self.assertEqual(s.runs_dir, Path(d))
            self.assertEqual(s.db_path, Path(d) / "depangram.sqlite")

    def test_tics_dir_defaults_to_the_vendored_scanner_when_it_is_there(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            want = st.SANDBOX / "engine" / "tics"
            self.assertEqual(st.path_setting("TICS_DIR", {}), want if want.is_dir() else None)
            with tempfile.TemporaryDirectory() as d:
                self.assertEqual(st.path_setting("TICS_DIR", {"TICS_DIR": d}), Path(d))       # overridable

    def test_a_caller_s_literal_default_gives_way_to_the_path_setting_s_own(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            s = Settings(env={}, config={})
            self.assertEqual(s.get("STOP_FILE", "an old machine path"), str(st.SANDBOX / "runs" / "STOP"))
            self.assertEqual(s.get("NOT_A_PATH", "kept"), "kept")                             # other names: as asked
            s = Settings(env={"STOP_FILE": "mine"}, config={})
            self.assertEqual(s.get("STOP_FILE", "an old machine path"), "mine")               # a set value wins

    def test_main_copy_refuses_a_document_outside_its_repo(self):
        from engine import main_copy
        with tempfile.TemporaryDirectory() as d:
            job = types.SimpleNamespace(real_file=Path(d) / "elsewhere" / "doc.md")
            out = []
            with mock.patch.object(main_copy, "load_job", lambda *a: job),                     mock.patch.object(main_copy, "repo_for", lambda job, settings: Path(d) / "repo"),                     mock.patch("builtins.print", lambda *a, **k: out.append(" ".join(map(str, a)))):
                self.assertEqual(main_copy.main(["status"]), 1)
        self.assertIn("sits outside the repo", out[0])

    def test_a_settings_double_with_only_get_or_only_config_still_reads(self):
        fake = types.SimpleNamespace(get=lambda k, d=None: d, runs_dir=Path("."))
        self.assertEqual(st.setting_path("MODELS_FILE", fake), st.SANDBOX / "models.toml")
        bare = types.SimpleNamespace(config={"notify": {"notifier": "file"}})
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(st.choice(bare, "NOTIFIER", "notify", "notifier", "discord", ("discord", "file")), "file")


class Config(unittest.TestCase):
    def test_the_example_parses_and_publishing_is_off(self):
        cfg = tomllib.loads((HERE / "config.example.toml").read_text(encoding="utf-8"))
        for section in ("pangram", "claude", "loop", "schedule", "discord", "git", "morning", "night", "regrow",
                        "repair"):
            self.assertIn(section, cfg)
        self.assertIs(cfg["git"]["publish"], False)
        self.assertNotIn("publish", cfg["morning"])
        self.assertNotIn("branch", cfg["night"])
        self.assertEqual(cfg["claude"]["token_made"], "")

    def test_the_example_holds_no_dates_or_private_names(self):
        text = (HERE / "config.example.toml").read_text(encoding="utf-8")
        for bad in ("2026-", "the author", "side/ai-guide", "C:\\", "\u2014"):
            self.assertNotIn(bad, text)

    def test_a_missing_config_falls_back_to_the_example_and_says_so_once(self):
        env = {k: v for k, v in os.environ.items() if k != "DEPANGRAM_CONFIG"}
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(st, "CONFIG_FILE", Path(d) / "config.toml"), mock.patch.object(st, "_told", set()):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                cfg = st.load_config()
                st.load_config()
            self.assertIs(cfg["git"]["publish"], False)
            self.assertEqual(err.getvalue().count("config.example.toml"), 1)                  # once
            (Path(d) / "config.toml").write_text("[git]\npublish = true\n", encoding="utf-8")
            self.assertIs(st.load_config()["git"]["publish"], True)                           # pass: yours wins
            (Path(d) / "other.toml").write_text("[git]\npublish = false\n", encoding="utf-8")
            os.environ["DEPANGRAM_CONFIG"] = str(Path(d) / "other.toml")
            self.assertIs(st.load_config()["git"]["publish"], False)                          # the variable wins

    def test_env_example_lists_every_key_with_no_value(self):
        lines = (HERE / ".env.example").read_text(encoding="utf-8").splitlines()
        pairs = [ln.split("=", 1) for ln in lines if ln and not ln.startswith("#")]
        self.assertTrue(all(v == "" for _, v in pairs))
        names = {k for k, _ in pairs}
        for k in ("PANGRAM_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_BACKEND", "NOTIFIER",
                  "RUNS_DIR", "WORK_DIR", "MODELS_DIR", "MODELS_FILE", "LLAMA_CPP_DIR", "STOP_FILE", "TICS_DIR"):
            self.assertIn(k, names)


# ---------------------------------------------------------------------- the API backend, with a fake client
class FakeClient:
    """anthropic.Anthropic's shape, as much as run_api uses: messages.create(**body) -> a message."""

    def __init__(self, text="the answer", stop="end_turn", usage=None, error=None):
        self.calls, self.text, self.stop, self.error = [], text, stop, error
        self.usage = usage or {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0}
        self.messages = self

    def create(self, **body):
        self.calls.append(body)
        if self.error:
            raise self.error
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="thinking", thinking=""), types.SimpleNamespace(type="text", text=self.text)],
            stop_reason=self.stop, usage=types.SimpleNamespace(**self.usage))


def api_settings(**claude_cfg):
    return Settings(env={"CLAUDE_BACKEND": "api", "ANTHROPIC_API_KEY": "sk-test"}, config={"claude": claude_cfg})


class ApiBackend(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")

    def row(self):
        return self.conn.execute("SELECT role, model, ok, input_tokens, output_tokens, cost_usd, week_util, error"
                                 " FROM claude_calls ORDER BY id DESC").fetchone()

    def test_the_cli_is_the_default_and_the_setting_picks_the_api(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(claude_call.backend(Settings(env={}, config={})), "cli")
            self.assertEqual(claude_call.backend(Settings(env={}, config={"claude": {"backend": "api"}})), "api")
            with self.assertRaises(ValueError):
                claude_call.backend(Settings(env={"CLAUDE_BACKEND": "carrier pigeon"}, config={}))

    def test_a_call_maps_the_model_logs_the_same_row_and_returns_the_same_shape(self):
        fake = FakeClient()
        text, meta = claude_call.run(api_settings(), self.conn, "the prompt", role="loop s1w0 judge", model="sonnet",
                                     max_budget_usd=0.50, client=fake)
        self.assertEqual(text, "the answer")                                                # the text block only
        body = fake.calls[0]
        self.assertEqual(body["model"], "claude-sonnet-5-5")
        self.assertEqual(body["messages"], [{"role": "user", "content": "the prompt"}])
        self.assertEqual(body["system"], claude_call.DEFAULT_SYSTEM)
        self.assertNotIn("temperature", body)
        want = (1000 * 2.00 + 200 * 10.00) / 1e6
        self.assertAlmostEqual(meta["cost_usd"], want)
        self.assertEqual(set(meta), {"usage", "cost_usd", "week_util", "five_hour_util"})
        self.assertEqual(self.row(), ("loop s1w0 judge", "claude-sonnet-5-5", 1, 1000, 200, want, None, None))

    def test_the_output_is_capped_at_what_the_ceiling_buys(self):
        fake = FakeClient()
        claude_call.run(api_settings(), self.conn, "p", role="r", model="opus", max_budget_usd=0.10, client=fake)
        self.assertLessEqual(fake.calls[0]["max_tokens"] * 20.00 / 1e6, 0.10)                 # flag: capped
        claude_call.run(api_settings(), self.conn, "p", role="r", model="haiku", max_budget_usd=0.50, client=fake)
        self.assertEqual(fake.calls[1]["max_tokens"], claude_call.API_MAX_TOKENS)              # pass: room to spare

    def test_refused_before_any_call(self):
        fake = FakeClient()
        for kw in ({"tools": "Read"}, {"max_turns": 3}, {"model": "mystery-model"}, {"max_budget_usd": 0.0001}):
            with self.assertRaises(claude_call.ClaudeCallError):
                claude_call.run(api_settings(), self.conn, "p", role="r", client=fake, **kw)
        self.assertEqual(fake.calls, [])                                                    # nothing sent
        s = api_settings(api_models={"sonnet": "claude-new"}, api_prices={"claude-new": [1, 2, 0, 0]})
        claude_call.run(s, self.conn, "p", role="r", client=fake)
        self.assertEqual(fake.calls[0]["model"], "claude-new")                               # pass: config wins

    def test_a_refusal_a_cut_off_and_an_error_are_failed_rows(self):
        for fake, why in ((FakeClient(stop="refusal"), "refused"), (FakeClient(stop="max_tokens"), "cut off"),
                          (FakeClient(text=""), "no text"), (FakeClient(error=RuntimeError("boom")), "boom")):
            with self.assertRaises(claude_call.ClaudeCallError) as cm:
                claude_call.run(api_settings(), self.conn, "p", role="r", client=fake)
            self.assertIn(why, str(cm.exception))
            self.assertEqual(self.row()[2], 0)
        self.assertEqual(self.row()[5], 0.0)                                                # an error spent nothing

    def test_a_timeout_is_counted_at_its_ceiling(self):
        class APITimeoutError(Exception):
            pass
        with self.assertRaises(claude_call.ClaudeCallError):
            claude_call.run(api_settings(), self.conn, "p", role="loop t", max_budget_usd=0.3,
                            client=FakeClient(error=APITimeoutError("slow")))
        self.assertEqual(self.row()[5], 0.3)

    def test_no_key_or_no_package_says_so_and_logs_a_row(self):
        s = Settings(env={"CLAUDE_BACKEND": "api"}, config={})
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.dict("sys.modules", {"anthropic": types.SimpleNamespace(Anthropic=lambda **kw: None)}):
            with self.assertRaises(claude_call.ClaudeCallError) as cm:
                claude_call.run(s, self.conn, "p", role="r")
        self.assertIn("ANTHROPIC_API_KEY", str(cm.exception))
        with mock.patch.dict("sys.modules", {"anthropic": None}):
            with self.assertRaises(claude_call.ClaudeCallError) as cm:
                claude_call.run(api_settings(), self.conn, "p", role="r")
        self.assertIn("pip install anthropic", str(cm.exception))

    def test_the_cli_path_is_untouched_by_the_api_settings(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(claude_call.subprocess, "run",
                                  side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=1)) as run:
            s = Settings(env={"WORK_DIR": d, "CLAUDE_BACKEND": "cli"}, config={})
            with self.assertRaises(claude_call.ClaudeCallError):
                claude_call.run(s, self.conn, "p", role="r", max_budget_usd=0.2)
        self.assertEqual(run.call_args.kwargs["cwd"], d)
        self.assertEqual(self.row()[5], 0.2)


if __name__ == "__main__":
    unittest.main()
