# MCP server

The engine comes with an MCP server, so a Claude session (or any MCP client) can drive the whole thing in plain language. Ask it to score a paragraph, start a run with a ceiling, check how it's going and read the report when it's done, and it'll do just that, without you ever touching a terminal.

## Setup

It's just one entry in your `.mcp.json`, pointing at the repo's Python and the server's entry point.

```json
{
  "mcpServers": {
    "depangram": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/depangram/mcp_server/__main__.py"]
    }
  }
}
```

It talks over stdio and reads the same `.env`, `config.toml` and `jobs/` as the command line, so whatever caps you've set there hold here too, which I think is the right default for a tool a model gets to press.

## Tools

| Tool | What it does |
| --- | --- |
| `score_text` | Scores a piece of text with Pangram (your key, billed, through the same ledger and caps) or Open Pangram (local and free). Gives back the AI share, the flagged stretches and the humanizer flag, plus the cost. |
| `list_jobs` | Every job, its document, the sections in scope and the score each section serves right now. |
| `start_run` | Starts a paragraph pass or a night in the background and hands back a run id straight away. It needs a Pangram ceiling, every time, and it refuses to start while the STOP file is there or something else holds the GPU. |
| `run_status` | Where a run's got to: running, paused, finished or stopped, the last lines of its log, the scores so far and what it's spent. |
| `run_report` | The run's report and digest, as markdown. |
| `stop_run` | Ends the run and everything it started, the model server included. |
| `review_queue` | What's waiting on a human: the paragraphs in bold with their prompts, the repairs and vetoes, and anything in the review file that no longer matches. |

## Design notes

A real run takes twenty minutes or more on the GPU, which is really far too long for a tool call to just sit there and wait. So `start_run` launches the work as its own process and returns at once, and the run's state lives in a small record on disk, which means `run_status` still works after the server restarts. A process id only counts as the run if it also started at the recorded time, so a recycled id is never mistaken for the run (and never killed by accident).

The ceiling's required on purpose, too. It'd be easy enough to default it, but a model driving a tool shouldn't be able to start a paid run without saying how much it's allowed to spend, so there's simply nothing to fall back on.

And everything a tool returns gets the same treatment as the logs, so you get plain errors with no tracebacks, paths shortened, and keys scrubbed out before they ever reach the client.
