"""By-hand check of the two-way Discord (plan 9.1, GitHub issue #8). No Pangram, no Claude, no money.

    python -m engine.discord_check               # with a bot token: post one "[discord check]" message, read it back
    python -m engine.discord_check --read ID     # read the replies and reactions on that message, and the answer

With no DISCORD_BOT_TOKEN or DISCORD_CHANNEL_ID in .env it says the bot path is off and whether the webhook path
is on, posts nothing and exits 0. It never prints a key, a token or the webhook URL, only which ones are set.
"""
import argparse
import sys

from engine import discord
from engine.settings import Settings

TEST = ("[discord check] depangram posting through the bot. Reply to this message with a number (1 or 2), or react"
        " with a thumbs-up or a cross; the terminal that posted it has the command that reads your answer.")


def report(settings, mid, out):
    """What the loop would see on one message: the replies after it, who wrote them, the reactions, the answer."""
    replies = discord.replies_since(settings, mid)
    mine = [m for m in replies if discord.is_author(settings, m)]
    out(f"replies after it: {len(replies)} ({len(mine)} count as yours"
        f" by {'DISCORD_AUTHOR_ID' if settings.get('DISCORD_AUTHOR_ID') else 'not being a bot'})")
    if any(not m["content"] for m in mine):
        out("  a reply of yours came back with no text: turn on Message Content Intent on the bot's page")
    for m in mine[:10]:
        to = "a reply to it" if m.get("reply_to") == str(mid) else "a plain message"
        out(f"  {m['id']} ({to}): {m['content'][:80]!r}")
    reacts = discord.reactions(settings, mid)
    out("reactions: " + (", ".join(f"{e} x{len(u)}" for e, u in reacts.items()) or "none"))
    n = discord.answer_to(settings, mid, 2, veto=2, replies=replies)
    out(f"read as a two-option question: {'no answer (the default stands)' if n is None else f'option {n}'}")


def main(argv=None, settings=None, out=print):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="check the Discord bot path (plan 9.1)")
    ap.add_argument("--read", help="a message id to read the replies and reactions of, instead of posting")
    a = ap.parse_args(argv)
    settings = settings or Settings()
    kind = discord.notifier(settings)
    if kind != "discord":
        out(f"the notifier is {kind} (NOTIFIER), so Discord is off and there's nothing to check; nothing posted.")
        return 0
    if not discord.bot_on(settings):
        have = [k for k in ("DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID") if settings.get(k)]
        out("the bot path is off: DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID aren't both in .env"
            + (f" (only {', '.join(have)} is)" if have else "") + ".")
        out("the webhook path is " + ("on (DISCORD_WEBHOOK_URL is set): posts go out one way, as before."
                                      if settings.get("DISCORD_WEBHOOK_URL") else "off too: DISCORD_WEBHOOK_URL isn't set."))
        out("nothing posted. To turn the bot path on: make a Discord bot, add it to the channel with Send Messages"
            " and Read Message History, turn on its Message Content Intent, and put DISCORD_BOT_TOKEN and"
            " DISCORD_CHANNEL_ID in .env.")
        return 0
    if a.read:
        report(settings, a.read, out)
        return 0
    mid = discord.bot_post(settings, TEST)          # one message, no more
    if not mid:
        out("the bot couldn't post: check the token, the channel id and the bot's Send Messages permission there.")
        return 1
    back = discord._get(settings, f"/messages/{mid}")
    out(f"posted message {mid}; read back: {'yes' if back and str(back.get('id')) == mid else 'no'}")
    report(settings, mid, out)
    out(f"now answer it on Discord, then: python -m engine.discord_check --read {mid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
