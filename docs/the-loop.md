# The loop

The research question is whether you can bend machine text into one person's voice. But the engineering question ate more of my evenings, which is how you let a system chew on that for days at a time without it overspending, going rogue, wrecking the document or getting in my way. What I ended up with is a steerable, evaluable, white box, semi-autonomous system that takes an input and slowly shapes it into an output while learning from the process, so this doc is about the "semi-autonomous" part, and the next three words after it.

<!-- figure: pipeline -->

## Schedule

So the window loop runs on a clock. Task Scheduler (or cron, or a systemd timer) starts it at 2am, and it works the document window by window, worst first, six rounds a pass. At 9am it moves into a day phase where it leaves most of my Claude usage alone, because I'm probably awake and using Claude myself by then. At 3:30pm it stops starting new work, and at 4pm it stops the model server, posts a digest and exits, so the GPU's all mine again for the evening.

The night run is basically the short, sharp version of that. It fires at 2am and walks a fixed list.

1. It checks it's on the right branch, that there's no STOP file, that nothing else is using the GPU, and that my Claude week isn't near its ceiling.
2. It pulls in any edit I made to the document and syncs it in, so it never works on stale text.
3. It runs a round of the paragraph pass, with a deadline, then rescores what it'll serve.
4. It runs the repair step on anything new.
5. It writes the document back, every change under its original, commits, publishes and posts the digest.

<!-- figure: a night -->

But if a step runs past its deadline, the night kills its process tree (the model server included) and just carries on with the next step. If the process dies in the middle of writing a log line, the next load skips the broken line and cuts it out at the first write, so a crash at 4am never breaks the morning.

## Budget caps

Every paid call goes through a capper first, and every call lands in a ledger, so the system always knows what it's spent and what it's allowed to spend next (which, given it's spending my money while I sleep, was really non-negotiable).

<!-- figure: budget controls -->

**Pangram** has a hard monthly cap of $180 and a daily cap of $6, plus a $3 reserve held back for the round that scores the full document, which goes back to the window rounds once that round's done. On top of that there's a $15 cap on calibration work and a ceiling on every pass. That's a lot of caps, I know. But **Claude** was trickier still, because it's my own subscription and I use it all day. The loop gets at most 50% of my week, and everything stops at 90% of it no matter whose usage it is. Inside the 5-hour window, the loop can use up to 90% of it before 9am and only 50% after, which keeps my mornings free. Every call type also has its own dollar ceiling, from $0.20 for a fact check up to $0.60 for a set of notes.

The catch is that Claude's usage readings come back in coarse steps, so the capper converts dollars into "points of my week" with a rate it refits from the ledger at the end of every run-day, clamped to between 0.5 and 2 times its seed so one weird night can't throw it off. And when a cap says no, the loop doesn't just crash. It holds that door, posts why, and quietly banks the work that doesn't need the capped thing (drafting and free Open Pangram scoring keep going while Pangram's held), and picks back up when the cap resets.

Errors get the same treatment: three tries with a 30 and then a 90 second wait, then the door's held for 10 minutes while everything else carries on, and five failures in a row with nothing finished in between ends the day with a crash alert.

<!-- figure: spend by day -->

Over the run that came to 480 Pangram calls for $129.65 and 4,812 Claude calls, and every one of them is in the ledger.

## Logs and records

White box just means nothing happens off the record, so the system keeps all of this.

- **the ledger**, a SQLite file with every paid call, its cost, its purpose and, for Claude, my week and 5-hour readings at that moment (480 Pangram rows and 4,812 Claude rows by the end).
- **the banks**, append-only logs of every draft, every score and every decision, so any run can resume from its best text and any result can be traced to the draft that made it (726 entries in the window bank, 4,601 in the paragraph pass's).
- **the reports**, one per run, with the scores, the gate tallies and what changed.
- **the review file**, which holds every repair, veto and stuck span, with the reason.
- **the write-back ledger**, which knows every paragraph the machine ever wrote, so it can tell my edits from its own.
- **the digests and the logs**, which are the human-readable version of all of the above.

<!-- figure: observability -->

And because the write-back keeps every original in a comment, the document itself is part of the record: strip the comments and you're back to the source, byte for byte.

## Alerts

The loop reports to a Discord channel. It posts when a run-day starts, once an hour while it's running, whenever a cap holds or a door gets held after retries, if a window that passed suddenly flags, if Pangram's model version changes under it, if my Claude token is about to expire, and a full digest at the end of every run, crashes included. Long jobs like a fine-tune post their start, an ETA once they've measured their pace, progress at a quarter, two quarters and three quarters, and when they finish. Over the run that came to 64 hourly posts, 14 run-day starts, 13 digests and 9 cap holds.

<!-- figure: alerts -->

Oh, and it's two-way. When the loop wants to cut a run it can't crack, it asks first, and I answer from my phone with a number, a thumbs-up, a cross to veto the cut, "rest" and a window to rest it, or "stop" to end the day. It reads replies every minute, and a question I ignore closes on its default after two days. But the one thing a reply can never do is raise a cap, which is very much on purpose.

## Steering

That's the "steerable" bit, but Discord's really only one of the levers.

- **The STOP file.** Drop it in the runs folder and nothing new starts.
- **The review file.** A veto means a draft never gets picked again, a repair pins a fix, and a stuck span goes into the document in bold with a prompt for me.
- **My own edits.** Anything I change in the document gets synced in at the next run and treated as mine from then on, so it's never regrown.
- **The config and the job file**, which set the caps, the schedule, the sections in scope, the noise threshold, the house rules and the voice samples.
- **The MCP server**, which lets a Claude session score text, start a run with a ceiling, check on it and read its report ([mcp.md](mcp.md)).

## Learning

The loop already learns in the plain sense before you get anywhere fancy: every draft and score is banked, every pass restarts from the best text it's ever seen, a section never ships below its best, and the capper refits its own rates from what it actually spent.

The design goes one level up, with a learning layer on top of the loop:

1. A director (a Sonnet call) reads the bank and the night's results and proposes a strategy change, like a different setup, model or knob.
2. The change runs the next night against the current playbook, on windows neither one was tuned on, scored first on Open Pangram and then on a small paid Pangram check, with a random slice audited on the side.
3. It's adopted only if it wins without losing on voice or facts.
4. The playbook is versioned, so every adopted change can be traced back to the night that earned it.

<!-- figure: learning layer -->

It's the same rule the experiments ran on: nothing changes on vibes.
