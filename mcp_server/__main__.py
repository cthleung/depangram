"""python -m mcp_server: the depangram MCP server over stdio. Run as a file (python path/to/mcp_server/__main__.py),
it puts the repo on the path first, so a client config needs no working folder."""
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server.server import main  # noqa: E402

if __name__ == "__main__":
    main()
