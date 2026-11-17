"""The MCP server over stdio, on the official Python SDK (mcp 2.x, where FastMCP is now MCPServer).

    python -m mcp_server

Each tool hands its work to core.App and turns any failure into a ToolError with one plain line: no traceback, no
key, and the repo's own folder written as <repo>. A run takes 20 minutes or more on a GPU, so start_run returns a run
id at once and the run goes on in the background; run_status, run_report and stop_run take that id, and keep working
after the server restarts.
"""
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .core import App, plain_error

INSTRUCTIONS = (
    "depangram rewrites the machine-written stretches of a markdown document against Pangram, an AI-text detector."
    " score_text scores a text; list_jobs shows the jobs and the score each section serves now; start_run starts a"
    " paragraph pass or a night in the background and returns a run id for run_status, run_report and stop_run;"
    " review_queue lists what waits for the author. Pangram costs money: every Pangram call goes through the"
    " engine's ledger and caps, and start_run needs a Pangram ceiling.")


def build(app=None):
    """The server with its seven tools, all working through `app` (a core.App; the tests pass one with fakes)."""
    app = app or App()
    server = MCPServer("depangram", instructions=INSTRUCTIONS)

    def guard(fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except Exception as e:  # noqa: BLE001
            raise ToolError(plain_error(app, e)) from None

    @server.tool()
    def score_text(text: str, detector: Literal["open", "pangram"] = "open") -> dict[str, Any]:
        """Score a text for AI writing. detector "open" is Open Pangram, local and free (it loads torch on first use);
        "pangram" is Pangram itself, paid, through the engine's ledger and caps, with the cost reported. Returns the
        AI share, the flagged stretches (and what flagged means for that detector), and Pangram's humanized flag
        (null for Open Pangram)."""
        return guard(app.score_text, text, detector)

    @server.tool()
    def list_jobs() -> dict[str, Any]:
        """Every job in the jobs folder: its document, the sections in scope, the night's sections, and the score
        each section serves now, read from the paragraph pass's bank without writing anything."""
        return guard(app.list_jobs)

    @server.tool()
    def start_run(job: str, max_pangram_usd: float, kind: Literal["pass", "night"] = "pass",
                  sections: list[int] | None = None, rounds: int = 1,
                  max_claude_usd: float | None = None, pass_minutes: int | None = None) -> dict[str, Any]:
        """Start a paragraph pass (python -m engine.regrow, purpose loop) or a night (python -m engine.night) on a job
        in the background, and return its run id at once. job is a folder name in jobs/ or a job folder inside the
        repo. max_pangram_usd is required: the most the run may spend on Pangram; for a night it covers the pass, the
        finals and the repair step together. max_claude_usd caps the fact check (left out: the config's own). rounds
        counts new scoring rounds. pass_minutes (a night only) ends the night's pass that many minutes after the start
        and lets it run at any time of day; without it the night keeps its [night] clock, and started at or after
        latest_start or pass_until it skips the pass and runs the morning submission only. The submission commits,
        publishes when [git] publish is on, and posts a digest through the notifier. Refused while another run holds
        the GPU, or while the STOP file is there."""
        return guard(app.start_run, job, max_pangram_usd, kind, sections, rounds, max_claude_usd, pass_minutes)

    @server.tool()
    def run_status(run_id: str) -> dict[str, Any]:
        """Where a run is: running, finished (with its exit code), stopped or lost (its process went and left no
        exit code); the Pangram and Claude spend in the ledger since it started, the section scorings it has logged,
        and its last log lines."""
        return guard(app.run_status, run_id)

    @server.tool()
    def run_report(run_id: str) -> dict[str, Any]:
        """The report a run wrote (the paragraph pass's report in analytics/, and for a night the morning digest),
        as markdown."""
        return guard(app.run_report, run_id)

    @server.tool()
    def stop_run(run_id: str) -> dict[str, Any]:
        """End a run and its whole process tree, its llama server too."""
        return guard(app.stop_run, run_id)

    @server.tool()
    def review_queue(job: str) -> dict[str, Any]:
        """What waits for the author on a job: the paragraphs in bold for the author (the stuck entries with their
        prompts, and the hand fixes), the paragraphs the pass serves repaired, the repairs and vetoes in
        review.json, and the entries that match no paragraph now."""
        return guard(app.review_queue, job)

    return server


def main():
    build().run("stdio")


if __name__ == "__main__":
    main()
