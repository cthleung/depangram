# Evals

Everything here rests on one question I kept asking myself every night, which is whether that change actually helped or I just got lucky with the dice. And a detector score on its own can't answer it, because Pangram wobbles, my own ear gets tired at 2am, and an LLM judge will happily tell you whatever sounds nice. So every change had to get past a stack of evals that each catch a different kind of lie, and none of them is allowed to see what the others said.

This doc walks through that stack, more or less in the order a draft meets it.

<!-- figure: the eval stack -->

Nothing that writes ever gets to see a score. The base-model writers get notes and voice samples, and that's it (the voice adapter doesn't even see Claude's sentence, only the notes boiled down from it). The LLM judge and the blind head-to-head see two texts and never a number. Pangram's verdicts go to code, which applies the keep rules, and to a nightly reflection that writes to me alone. Even the score for the full document is just a number for me to read, never a target, so no round and no pick ever looks at it.

That's the Goodhart guard, and it's really the reason a falling score means the writing changed, instead of some model quietly learning to flatter the detector.

## Noise floor

Before trusting any number, I measured how much Pangram moves when nothing that matters moves. Three passages, each scored as is and again with three edits that change no meaning (a comma moved, a contraction swapped, a twin word), so 12 scorings for a dollar. Most held perfectly still, but one comma edit took a passage from 1.00 to 0.79 - which, for a comma, is a lot.

So the loop only keeps a change that beats a noise threshold of 0.05, and a section never ships below its best known text. That being said, the rule earned its keep, because it caught real regressions too. A restore pass that took a section from 0.00 to 0.65 got reverted on the spot, and so did one that went from 0.06 to 0.13.

<!-- figure: noise floor -->

## Open Pangram

Pangram bills by the word, and the loop draws thousands of candidates a night, so scoring all of them would've cost a small fortune (and probably my sanity). Open Pangram, a local RoBERTa model that mirrors Pangram's approach, does the sorting for free. On 189 segments Pangram had labeled, it scored an AUC of 0.72 (0.74 on segments of 20 words or more), which is nowhere near good enough to decide anything, but plenty to sort with. And across the 9 models in the bake-offs, its ranking matched Pangram's with a Spearman of 0.94.

So it ranks, but Pangram decides. In the window loop only the top 8 candidates by Open Pangram ever reach the judge, and in the paragraph pass it orders the candidates that got through the gates. It never scores a section and never keeps a change.

<!-- figure: pangram vs open pangram -->

## Gates and the fact check

Every judgment that can be a rule is a rule, and the rules run first. A draft that loses a quote, invents a name, changes a number, drops a protected word or comes out the wrong length dies right there, before an LLM or the detector ever reads it. In the paragraph pass that was 3,167 of 3,953 drafts, so 80.1% of everything the base models wrote didn't cost a cent to evaluate.

<!-- figure: gate funnel -->

The ones that survive go to a fact check, a Sonnet read against the original and its notes, which doesn't need to say anything nice about the draft, it just has to find nothing wrong. And this is where the voice adapter really earned its place: on the same check it passed 112 of 223 drafts, against 32 of 216 for the few-shot writer, so it's roughly three times as likely to get the facts right.

## Blind judging

The candidates that pass get read by two Sonnet calls. The judge picks the best of up to 8 against the notes and the text on either side. Then the head-to-head gets the pick and the original as A and B, with no labels and the positions switched back and forth, and says A, B or the same.

Over the window loop that came to 1,067 head-to-heads, and every one of them reconciles to the bank. The pick won 161 and tied 36. When a pick lost, it usually stayed out (693 times), except in a pass's last round, where a pick that lost still replaces the Claude line (84 times), because the point is that no Claude line survives. Another 93 picks went on to fail the fact check, which is what a second, separate check is for.

<!-- figure: blind judge -->

Oh, and the harshest evaluator gets a vote too. I read two windows against their originals with the columns shuffled and nothing labeled, then checked the key. I couldn't tell which was which, and I didn't have a preference either. A few days later, reading the loop's text on its own, my note was that it read more natural than the original.

## Voice

For voice there's Burrows' Delta, the standard stylometry measure: how far a text's common-word frequencies sit from my own writing, in 300-word chunks, where lower is closer. My own chunks have a median of 0.88, so that's the bar.

When I trained a LoRA on my pre-ChatGPT writing, it moved Delta toward me in all 3 test windows (1.00 to 0.89, 0.92 to 0.82, 1.13 to 0.94), while also taking one window from 0.48 to 0.00 with every citation intact. A paraphrase adapter I'd trained first barely moved Delta at all (1.00 to 1.01, 0.92 to 0.91), so that one's parked. Later I retrained the voice adapter with more registers of my writing, and the live one stayed closer on 2 of 3 windows and beat it on Pangram, so the retrain lost and the original stayed.

<!-- figure: voice metric -->

## What stayed and what went

Put all of that together and you get the record I'm probably proudest of, which is every model and method I tried, what it scored and why it stayed or went. Nothing stayed on vibes.

<!-- figure: keep or drop -->

- **Instruct models are out.** An instruct-tuned "humanizer" model (StyleTune V2) scored 1.00 in the first bake-off, while the plain base models in the next round started at 0.08. An instruct writer brought in later got 11 of 146 drafts past the gates and pushed a 0.33 section to 0.52, so it went too. Chat tuning's the fingerprint, and no prompt gets a model around it.
- **Mistral Small 24B base writes.** It scored 0.08, and it fit on my GPU (more on that in [the machine](the-machine.md)).
- **The voice adapter stays, the retrain doesn't.** See above.
- **Editors don't move the score.** Five different editors on three windows never moved Pangram past the noise, so editing is for facts only, never for the detector.
- **A second base model didn't help.** OLMo 3 32B base on the 14 spans nobody could crack got 12 drafts past the gates and 0 past the fact check.
- **The two writer arms tie.** The adapter and the few-shot writer on the same 7 spans scored 0.53 and 0.58, which is inside the noise, so both stayed.

## Humanizer flag

Pangram has a separate humanizer head, and a scoring it flags is never a best, however low it scores. The window loop almost never tripped it (3 of 168 scorings), but the paragraph pass tripped it on 17 of 32, and when I lined the scorings up, the flag climbed with how much regrown text a section held: section 4's humanizer score went 0.66 with 4 regrown spans, 0.88 with 10 and 0.93 with 17. So that's the next experiment. The fix probably lives in how a section's mix gets put together, and I'd bet on that over any one paragraph.

<!-- figure: humanizer by pass -->

## Testing

Last thing. A gate that never fires looks just like a gate that works, so the test suite got the same treatment as the models, which I'll admit is a little obsessive. I broke features on purpose, 48 mutations in all, and checked that the tests went red, then fixed the two gaps that didn't. The loop also has a rehearsal mode that fakes Pangram, Claude and the GPU and runs a full day from 2am to 4pm with zero spend, across 12 scenarios, so the budget caps and the alerts get exercised before a real dollar's on the line.
