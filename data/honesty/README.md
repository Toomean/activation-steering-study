# Exploratory source-labelled honesty prompts

`prompts.json` fixes 23 extraction and 16 development rows from the pinned [TruthfulQA source](upstream/truthfulqa/README.md). It retains exact source-indexed question and best-answer contrast strings and the upstream category. The frozen selection combines 18 proposed fact-check keeps with 21 provisional supplements from a seeded queue. Source 474 was excluded during review because its labelled contrast was judged ambiguous; no replacement was added.

The positive label means TruthfulQA's `Best Answer` rather than independently fact-checked truth. Supplemental external sources and many candidate facts still require independent fact review. Scores are provisional source-label preference, not a validated measure of general honesty. No final holdout is selected here.
