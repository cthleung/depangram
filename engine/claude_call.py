"""Headless Claude calls for the loop (plan 7.2, 7.3, 10.5).

Two backends behind one run(...), picked by CLAUDE_BACKEND in .env (or [claude] backend in config.toml):

"cli" (the default) runs the Claude Code CLI, `claude -p`, on a subscription. Every call:
  - runs from WORK_DIR (default ~/.depangram/work, outside any repo, so no repo hooks fire and no project
    instructions are loaded)
  - uses CLAUDE_CODE_OAUTH_TOKEN from .env (made with `claude setup-token`) and strips ANTHROPIC_API_KEY, so it
    bills the subscription, not an API account
  - loads no MCP servers (--strict-mcp-config with an empty config), so Claude has no way to reach Pangram
  - replaces the default system prompt with a short one, to keep context small
  - has a --max-budget-usd ceiling and --max-turns
  - streams JSON, so the result, the token usage and the rate_limit_event (your 5-hour and weekly utilization)
    all get logged in claude_calls

"api" calls the Anthropic API with ANTHROPIC_API_KEY through the official `anthropic` package (pip install
anthropic). Text in, text out, one turn, no tools. The short model names the engine uses ("sonnet", "haiku",
"opus") map to API model ids through [claude] api_models, and the dollars come from the usage and [claude]
api_prices, since the API returns no cost. A call can't stop itself at its ceiling the way the CLI does, so the
output is capped at what the ceiling still buys after the prompt. The rows carry no week or 5-hour readings; the
capper's dollar checks still hold, and its percentage checks read the loop's own estimated dollars.

Both write the same claude_calls row and return the same (text, meta).
"""
import json
import math
import os
import subprocess
import time

from .settings import Settings, choice, setting_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS claude_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    model TEXT NOT NULL,
    ok INTEGER NOT NULL,
    seconds REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    cost_usd REAL,
    week_util REAL,
    five_hour_util REAL,
    week_resets_at INTEGER,
    error TEXT,
    five_hour_resets_at INTEGER
);
"""


def ensure_schema(conn):
    """The table, plus the five_hour_resets_at column on databases made before it existed."""
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(claude_calls)")}
    if "five_hour_resets_at" not in cols:
        conn.execute("ALTER TABLE claude_calls ADD COLUMN five_hour_resets_at INTEGER")
        conn.commit()


DEFAULT_SYSTEM = ("You are a careful writing assistant inside an automated pipeline. Follow the user's "
                  "instructions exactly and output only what they ask for, with no preamble.")
BACKENDS = ("cli", "api")

# the API backend's defaults; [claude] api_models and api_prices in config.toml override them
API_MODELS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}
# dollars per million tokens: [input, output, cache read, cache write]. Check them against Anthropic's pricing page.
API_PRICES = {
    "claude-opus-5-5": [4.00, 20.00, 0.20, 5.00],
    "claude-sonnet-5-5": [2.00, 10.00, 0.10, 2.50],
    "claude-haiku-5-5": [0.10, 0.50, 0.01, 0.125],
}
API_MAX_TOKENS = 16000          # the most output one call asks for, before the ceiling cuts it down
CHARS_PER_TOKEN = 3.5           # a rough count for the prompt's cost before the call, on the safe side


class ClaudeCallError(Exception):
    pass


STDIN_OVER = 20000      # characters: a longer prompt goes on stdin (Windows caps a command line near 32k, and the
                        # reflection's prompt runs to about 60k)


def backend(settings):
    """The backend, cli or api: CLAUDE_BACKEND in .env or the environment, else [claude] backend, else "cli"."""
    return choice(settings, "CLAUDE_BACKEND", "claude", "backend", "cli", BACKENDS)


def run(settings, conn, prompt, role, model="sonnet", system=DEFAULT_SYSTEM, max_turns=1,
        max_budget_usd=0.50, timeout=600, tools="", stdin_over=STDIN_OVER, client=None):
    """One call through the chosen backend. tools="" means no built-in tools at all (text in, text out): a bare CLI
    call with the default tools carried about 37,000 tokens of context, so text-only roles must keep tools off. A
    prompt longer than `stdin_over` characters goes to `claude -p` on stdin instead of as the argument. `client`
    stands in for the API backend's anthropic.Anthropic() (tests pass a fake)."""
    if backend(settings) == "api":
        return run_api(settings, conn, prompt, role, model=model, system=system, max_turns=max_turns,
                       max_budget_usd=max_budget_usd, timeout=timeout, tools=tools, client=client)
    return run_cli(settings, conn, prompt, role, model=model, system=system, max_turns=max_turns,
                   max_budget_usd=max_budget_usd, timeout=timeout, tools=tools, stdin_over=stdin_over)


def log_row(conn, t0, role, model, ok, usage=None, cost=None, week=None, five=None, err=None):
    usage, week, five = usage or {}, week or {}, five or {}
    conn.execute(
        "INSERT INTO claude_calls (ts, role, model, ok, seconds, input_tokens, output_tokens, cache_read_tokens,"
        " cache_write_tokens, cost_usd, week_util, five_hour_util, week_resets_at, error,"
        " five_hour_resets_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (t0, role, model, int(ok), time.time() - t0, usage.get("input_tokens"), usage.get("output_tokens"),
         usage.get("cache_read_input_tokens"), usage.get("cache_creation_input_tokens"), cost,
         week.get("utilization"), five.get("utilization"), week.get("resetsAt"), err, five.get("resetsAt")))
    conn.commit()


def log_timeout(conn, t0, role, model, max_budget_usd, timeout):
    # the call may have spent up to its ceiling before it was stopped, so the ledger counts the ceiling (M1 review,
    # finding 2): the capper sums cost_usd, and without a row it would never see that spend
    conn.execute(
        "INSERT INTO claude_calls (ts, role, model, ok, seconds, cost_usd, error) VALUES (?,?,?,?,?,?,?)",
        (t0, role, model, 0, time.time() - t0, max_budget_usd, f"timed out after {timeout}s"))
    conn.commit()


# ---------------------------------------------------------------------- the CLI backend
def run_cli(settings, conn, prompt, role, model="sonnet", system=DEFAULT_SYSTEM, max_turns=1,
            max_budget_usd=0.50, timeout=600, tools="", stdin_over=STDIN_OVER):
    ensure_schema(conn)
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    token = settings.get("CLAUDE_CODE_OAUTH_TOKEN")
    if token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    work = str(setting_path("WORK_DIR", settings))
    os.makedirs(work, exist_ok=True)
    empty_mcp = os.path.join(work, "empty-mcp.json")
    if not os.path.exists(empty_mcp):
        with open(empty_mcp, "w", encoding="utf-8") as f:
            f.write('{"mcpServers": {}}')
    cli = settings.get("CLAUDE_CLI") or "claude"
    piped = len(prompt) > stdin_over
    cmd = [cli, "-p", *([] if piped else [prompt]), "--model", model, "--max-turns", str(max_turns),
           "--output-format", "stream-json", "--verbose",
           "--strict-mcp-config", "--mcp-config", empty_mcp,
           "--system-prompt", system, "--max-budget-usd", f"{max_budget_usd:.2f}",
           "--tools", tools, "--disable-slash-commands", "--no-session-persistence"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, encoding="utf-8",
                              input=prompt if piped else None, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        log_timeout(conn, t0, role, model, max_budget_usd, timeout)
        raise ClaudeCallError(f"timed out after {timeout}s") from None
    result, usage, rate, err = None, {}, {}, None
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "result":
            result = ev
            usage = ev.get("usage") or {}
        elif ev.get("type") == "rate_limit_event":
            rate = (ev.get("rate_limit_info") or {}).get("unifiedWindows") or rate
    text = (result or {}).get("result")
    ok = bool(result) and not result.get("is_error") and text is not None
    if not ok:
        err = ((result or {}).get("result") or proc.stderr or proc.stdout)[-500:]
    week = rate.get("seven_day") or {}
    five = rate.get("five_hour") or {}
    log_row(conn, t0, role, model, ok, usage, (result or {}).get("total_cost_usd"), week, five, err)
    if not ok:
        raise ClaudeCallError(err)
    return text, {"usage": usage, "cost_usd": (result or {}).get("total_cost_usd"),
                  "week_util": week.get("utilization"), "five_hour_util": five.get("utilization")}


# ---------------------------------------------------------------------- the API backend
def api_model(settings, model):
    """The API model id for a short name ("sonnet"), or the name itself when it already is an id."""
    names = dict(API_MODELS)
    names.update(((getattr(settings, "config", None) or {}).get("claude") or {}).get("api_models") or {})
    return names.get(model, model)


def api_prices(settings, model_id):
    """[input, output, cache read, cache write] dollars per million tokens, or None for a model with no price."""
    prices = dict(API_PRICES)
    prices.update(((getattr(settings, "config", None) or {}).get("claude") or {}).get("api_prices") or {})
    got = prices.get(model_id)
    return [float(x) for x in got] if got else None


def api_cost(prices, usage):
    """Dollars for one call's usage at `prices` (per million tokens)."""
    inp, out, read, write = prices
    return ((usage.get("input_tokens") or 0) * inp + (usage.get("output_tokens") or 0) * out
            + (usage.get("cache_read_input_tokens") or 0) * read
            + (usage.get("cache_creation_input_tokens") or 0) * write) / 1e6


def api_client(settings, timeout):
    """anthropic.Anthropic with ANTHROPIC_API_KEY from .env or the environment. Imported here, so the package is
    needed only by the API backend."""
    try:
        import anthropic
    except ImportError:
        raise ClaudeCallError("CLAUDE_BACKEND is api, but the anthropic package isn't installed:"
                              " pip install anthropic") from None
    key = settings.get("ANTHROPIC_API_KEY")
    if not key:
        raise ClaudeCallError("CLAUDE_BACKEND is api, but ANTHROPIC_API_KEY isn't set in .env or the environment")
    return anthropic.Anthropic(api_key=key, timeout=float(timeout), max_retries=0)


def _usage_dict(usage):
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    return {k: getattr(usage, k, None) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                 "cache_creation_input_tokens")}


def _is_timeout(e):
    return type(e).__name__ in ("APITimeoutError", "TimeoutError", "ReadTimeout", "TimeoutException")


def run_api(settings, conn, prompt, role, model="sonnet", system=DEFAULT_SYSTEM, max_turns=1,
            max_budget_usd=0.50, timeout=600, tools="", client=None):
    """One Messages API call. Refused before any spend: tools or more than one turn (the CLI backend does those),
    a model with no price in [claude] api_prices, and a prompt the ceiling can't pay for. A refusal or an answer
    cut off at the output cap is a failed call."""
    ensure_schema(conn)
    if tools or int(max_turns) > 1:
        raise ClaudeCallError("the API backend runs text-only, one-turn calls; tools and several turns need"
                              " CLAUDE_BACKEND=cli")
    model_id = api_model(settings, model)
    prices = api_prices(settings, model_id)
    if prices is None:
        raise ClaudeCallError(f"no price for {model_id} in [claude] api_prices, so its spend can't be counted;"
                              " add one before calling it")
    inp, out = prices[0], prices[1]
    prompt_usd = (len(prompt) + len(system or "")) / CHARS_PER_TOKEN * inp / 1e6
    room = math.floor((float(max_budget_usd) - prompt_usd) / out * 1e6) if out else API_MAX_TOKENS
    cfg = ((getattr(settings, "config", None) or {}).get("claude") or {})
    max_tokens = min(int(cfg.get("api_max_tokens", API_MAX_TOKENS)), room)
    if max_tokens < 256:
        raise ClaudeCallError(f"the ceiling ${max_budget_usd:.2f} leaves no room for an answer from {model_id}"
                              f" after a prompt of about ${prompt_usd:.2f}")
    body = {"model": model_id, "max_tokens": max_tokens, "system": system,
            "messages": [{"role": "user", "content": prompt}]}
    if cfg.get("api_effort"):
        body["output_config"] = {"effort": str(cfg["api_effort"])}
    t0 = time.time()
    try:
        c = client if client is not None else api_client(settings, timeout)
        resp = c.messages.create(**body)
    except ClaudeCallError as e:
        log_row(conn, t0, role, model_id, False, cost=0.0, err=str(e)[-500:])
        raise
    except Exception as e:  # noqa: BLE001  (the class and the message only: the request carries the key in a header)
        if _is_timeout(e):
            log_timeout(conn, t0, role, model_id, max_budget_usd, timeout)
            raise ClaudeCallError(f"timed out after {timeout}s") from None
        err = f"{type(e).__name__}: {str(e)[-400:]}"
        log_row(conn, t0, role, model_id, False, cost=0.0, err=err)
        raise ClaudeCallError(err) from None
    usage = _usage_dict(getattr(resp, "usage", None))
    cost = api_cost(prices, usage)
    stop = getattr(resp, "stop_reason", None)
    text = "".join(getattr(b, "text", "") or "" for b in (getattr(resp, "content", None) or [])
                   if getattr(b, "type", None) == "text")
    err = None
    if stop == "refusal":
        err = "the model refused the request (stop_reason refusal)"
    elif stop == "max_tokens":
        err = f"the answer was cut off at {max_tokens} output tokens (the ceiling ${max_budget_usd:.2f})"
    elif not text:
        err = f"no text in the answer (stop_reason {stop})"
    log_row(conn, t0, role, model_id, err is None, usage, cost, err=err)
    if err:
        raise ClaudeCallError(err)
    return text, {"usage": usage, "cost_usd": cost, "week_util": None, "five_hour_util": None}


if __name__ == "__main__":
    from . import db
    s = Settings()
    c = db.connect(s.db_path)
    out, meta = run(s, c, "Reply with exactly: claude_call ok", role="smoke test", model="haiku")
    print(out, meta)
