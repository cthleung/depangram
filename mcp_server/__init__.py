"""depangram's MCP server: score a text, list the jobs, start and watch a paragraph pass or a night, stop it, and
read its report and the author's review queue, over stdio.

    python -m mcp_server

The package is named mcp_server so it never shadows the `mcp` SDK it is built on. server.py registers the tools,
core.py holds what each tool does, runs.py keeps the run records, and child.py is the small wrapper every run
starts under so its exit code outlives the server.
"""
