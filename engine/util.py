"""Small helpers with no engine imports, so any module can use them without pulling in the run loop.

  - redact: keys, tokens and webhook URLs out of anything the engine logs or posts (moved out of engine/run.py).
  - describe: fills the one-line description of the document into a prompt. Every prompt that names the kind of
    document it works on carries the token <<DESCRIPTION>>, and this is the one place it gets filled. Today it's a
    fixed default; a later phase reads it from the job.
"""
import re

SECRET = re.compile(r"sk-ant-[\w-]+|https?://(?:\w+\.)?discord(?:app)?\.com/api/webhooks/\S+|x-api-key[^,\s]*", re.I)

DESCRIPTION_TOKEN = "<<DESCRIPTION>>"
DEFAULT_DESCRIPTION = "a casual, conversational essay"


def redact(text):
    """Keys, tokens and the webhook URL out of anything the loop logs or posts (M1 review, finding 12)."""
    return SECRET.sub("[redacted]", str(text))


def describe(prompt, description=None):
    """The prompt with the document's description in place of <<DESCRIPTION>>."""
    return prompt.replace(DESCRIPTION_TOKEN, description or DEFAULT_DESCRIPTION)
