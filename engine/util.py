"""Small helpers with no engine imports, so any module can use them without pulling in the run loop.

  - redact: keys, tokens and webhook URLs out of anything the engine logs or posts (moved out of engine/run.py).
  - describe: fills the one-line description of the document into a prompt. Every prompt that names the kind of
    document it works on carries the token <<DESCRIPTION>>, and this is the one place it gets filled. Today it's a
    fixed default; a later phase reads it from the job.
  - report_file: a report's path in analytics/ (or another folder), with the folder made first, so a fresh clone
    with no analytics/ never fails at the end of a run that already spent its money.
"""
import re
from pathlib import Path

# Pangram's key travels in the x-api-key header (engine/pangram.py) and lives in .env as PANGRAM_API_KEY. pangram.py
# fixes no form for the key itself, so the key is found by where it sits: after the header's name (as text, in a
# dict's repr, or as JSON) or after a key or token setting's name, whatever characters it's made of.
SECRET = re.compile(
    r"sk-ant-[\w-]+"
    r"|https?://(?:\w+\.)?discord(?:app)?\.com/api/webhooks/\S+"
    r"|(?<=x-api-key)[\"']?\s*[:=]\s*[\"']?[^\s,\"'}\]]+"
    r"|(?<=_API_KEY)[\"']?\s*[:=]\s*[\"']?[^\s,\"'}\]]+"
    r"|(?<=_TOKEN)[\"']?\s*[:=]\s*[\"']?[^\s,\"'}\]]+"
    r"|(?<=_SECRET)[\"']?\s*[:=]\s*[\"']?[^\s,\"'}\]]+",
    re.I)
SEPARATOR = re.compile(r"[\"']?\s*[:=]\s*[\"']?")

DESCRIPTION_TOKEN = "<<DESCRIPTION>>"
DEFAULT_DESCRIPTION = "a casual, conversational essay"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "analytics"   # the repo's analytics/ (git ignores it)


def _hide(m):
    """A match with its value hidden. After a header's or a setting's name the separator stays (": ", "=", "': '"),
    so the line still reads as a header or a setting."""
    sep = SEPARATOR.match(m.group(0))
    return (sep.group(0) if sep else "") + "[redacted]"


def redact(text):
    """Keys, tokens and the webhook URL out of anything the loop logs or posts (M1 review, finding 12), Pangram's key
    included wherever it follows x-api-key or PANGRAM_API_KEY."""
    return SECRET.sub(_hide, str(text))


def describe(prompt, description=None):
    """The prompt with the document's description in place of <<DESCRIPTION>>."""
    return prompt.replace(DESCRIPTION_TOKEN, description or DEFAULT_DESCRIPTION)


def report_file(name, folder=None):
    """The path of a report called `name` in `folder` (default: REPORTS_DIR, the repo's analytics/), with the folder
    made first when it isn't there."""
    d = Path(folder) if folder is not None else REPORTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / name
