# Jobs

A job is one markdown document the engine works on. Each job is a folder here, `jobs/NAME/`, with a `job.toml` that
says where the document is, how it marks its sections and which of them are in scope. `jobs/example/job.toml` is a
template with every field and a comment on each.

Job folders hold your text, so git ignores everything in `jobs/` except this file and the example.

## Starting a job

1. Make the folder: `jobs/NAME/`.
2. Copy `jobs/example/job.toml` into it and fill it in. At the least, set `document` and `description`, and say how
   the document marks its sections (`section_level` or `section_pattern`).
3. Copy the document into the folder as `original.md`. That's the frozen original: every window, every note and
   every marker is read against it, and only the sync step replaces it.
4. Say who wrote which line (see "Your lines" below). With no file at all and `verdicts = "file"`, every line is
   the machine's.
5. Run `python -m engine.sync --job NAME` for a dry run. It prints the sections it finds in scope and the windows it
   would cut, and writes nothing.

Every command takes `--job NAME`. Leave it out and the command reads `default` under `[job]` in `config.toml`; with
neither, it stops and says so.

## The fields

All of them live under `[job]`. Paths are relative to the job folder unless they're absolute.

| Field | Default | What it does |
| --- | --- | --- |
| `name` | the folder's name | The job's name in logs, digests and the runs folder. |
| `description` | a casual, conversational essay | One line about the document. It goes into every prompt that names the kind of document it works on, so write it as it reads after "a paragraph from": `a short practical how-to on growing herbs on a balcony`. |
| `document` | none | The document the write-back writes into. The older field `real_file` still works and is read from the repo root instead of the job folder. With neither, the write-back stops and asks for a file. |
| `original` | `original.md` | The frozen copy of the document. |
| `working` | `working.md` | The loop's working copy. The loop writes it; leave it alone. |
| `section_level` | none | Sections are the headings at this level: `3` for `### Part one`. A section runs to the next heading at its level or above. |
| `section_pattern` | `^## (\d+)\.\s` | A regex for a section's heading line, matched at the start of a line. Give this or `section_level`, never both. With neither, sections are the numbered `## 3. Title` headings older jobs used. |
| `sections` | `"all"` | The sections in scope: a list of ids, or `"all"` for every section with a line of prose in it. |
| `skip_list_sections` | `true` | A section with a list line outside a comment counts as still being drafted and stays out of scope. Set `false` for a document whose finished sections have lists. |
| `night_sections` | the scope | The sections the night's paragraph pass works on, when that's fewer than the scope. |
| `branch` | none | The git branch the night and morning steps expect this job on. |
| `samples` | `SAMPLES_FILE` from `.env` | A file of your own writing, for the writers' voice. Either `## Sample N` headings (`engine/tics/voice_audit.py`'s format, where a sample tagged `- **Source:** AI-assisted` is left out) or plain markdown, where each heading opens a sample. |
| `rules` | the repo's `rules.toml` | Your own house rules file, in the format of `rules.toml` at the repo root (banned words, dashes, spelling, the swear and filler budgets). Only the keys you change are needed. It's loaded when the job loads. |
| `voice_samples` | every sample | Which samples to use, by number: `[2, 5]` is "sample 2" and "sample 5". |
| `verdicts` | `"scan"` | Where "your lines" come from. `"file"`: only from the `your_lines` file, and a sentence it doesn't list is the machine's. `"scan"`: the sync step also runs the chat-log scan (`engine/find_lines.py`) on sentences new to the job. |
| `your_lines` | `your lines.md` | The verdicts file (below). |
| `example_lines` | `example lines.txt` | Lines the document quotes as examples, one a line. They come out of the prose before anything is scored and go back in exactly at the write-back. The sync step adds quoted-only lines it finds. |
| `protected_lines` | `protected lines.txt` | Lines that never change, one a line (a line starting with `#` is a comment). They're locked the way headings and templates are. |

`[seed]` is optional and older jobs use it: `banks` lists experiment banks in the job folder, `windows` the windows
that start from the best text in them, and `untagged_window` the window an experiment entry with no window id
belongs to. An older job whose first experiment banked entries with no window id sets `untagged_window = "s1w0"`.

## Section ids, windows and markers

A section's id is a number. With a pattern whose first group captures digits (the numbered default does), it's that
number, so `## 3. Title` is section 3. Otherwise it's the section's place in the document, counting from 1.

Windows are `s{section}w{index}` (`s2w0` is the first window of section 2), and paragraphs inside them are
`s2w0 p3`. The write-back's markers in the document use those labels. In a document whose headings aren't
numbered, a section's heading line stays in its first window as a locked paragraph, so the first prose paragraph
there is `p2`.

Ids by place move when you add or remove a section above another one. The sync step keeps each window's bank
history by its text, so the loop's own history follows the move. What's keyed by section id doesn't follow it: the
paragraph pass's scorings and best picks, the section notes in the document, and the labels in a review entry.
Those go stale for the moved sections and are earned again. A numbered document doesn't have this problem, since
its ids are the numbers in its headings.

## Your lines

`your lines.md` is a markdown table, one sentence a row, with the verdict in the first column and the sentence in
the last:

```
| Verdict | In your messages | In Claude's output | Pangram | Sentence |
| --- | --- | --- | --- | --- |
| yours | - | - | - | I drilled extra holes in my first pot. |
```

The verdicts are `yours`, `yours?`, `edited`, `both`, `claude` and `loop`. The loop never rewrites a line marked
`yours` unless Pangram flags it, and then it keeps that line's own words. The sync step appends rows for sentences
it learns about; it never changes a row that's there.

## What the engine writes here

The bank (`bank.jsonl`), the notes cache, `locks.md`, `couldnt crack.md`, the write-back's ledger
(`written back.jsonl`), the paragraph pass's files (`regrow bank.jsonl`, `regrow picks.json`, `regrow origs.json`),
the reviewer's `review.json` and the repair log. The sync step backs up what it replaces as
`NAME (before sync DATE).ext`.
