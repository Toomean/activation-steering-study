# Frozen harmless panel

`panel.json` contains metadata for 63 development rows (63 groups, 32 quality
members) and 246 final rows (245 groups, 63 quality members). It mechanically
preserves the accepted panel's ordered rows, IDs, original UTF-8 instruction
hashes, quality flags and group assignments. The accepted panel SHA256 is
recorded in the manifest; private planning and review metadata are omitted.

This study's initial seed-42 samples contained 64 development candidates after
excluding 16 historical validation IDs, and 256 final candidates. The October 4
suitability and semantic selection removed seven suitability rows and four final
rows with semantic overlaps across splits, without replacement. Initial quality
subsets selected 32 development and 64 final members using a fresh seed 42 for
each panel; surviving membership was retained, leaving 32 and 63 respectively.
Two accepted final Wi-Fi variants share one group; every other row is a singleton.
`semantic_status: fixed_for_analysis` means assignments were fixed before model
outputs; it does not establish statistical independence.

Refusal analysis averages binary rows within groups, then weights groups equally
in first-appearance manifest order. Quality NLL/KL summaries average available
quality-row diagnostics and report missing counts.

Changes are the second condition minus the first (`condition` minus `baseline`).
Paired contrasts report positive, negative and unchanged group counts from the
signs of group-mean changes; a fractional change counts once. `no_observed_change`
means unchanged harmless group means, where opposing row flips may cancel, or
unchanged MMLU correctness. Predictions, option mass and quality diagnostics
may still differ. `interval_reason` explains an unavailable nominal BCa interval,
while `interval_status: inconclusive` describes interval availability. The
observed delta and counts are retained, including constant nonzero changes.
Intervals containing zero, unavailable intervals and `no_observed_change` do not
establish equivalence or absence of an effect.

The source files remain unchanged in `../refusal/upstream/refusal_direction/`.
They were copied from `andyrdt/refusal_direction`, revision
`9d852fae1a9121c78b29142de733cb1340770cc3`; see the adjacent
[source README](../refusal/upstream/refusal_direction/README.md) and pinned
[processing notebook](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/generate_datasets.ipynb).
Upstream selects Alpaca instructions with empty input. Alpaca's
[CC BY-NC 4.0 data license](https://github.com/tatsu-lab/stanford_alpaca#data-release)
applies; this metadata manifest supplies no additional rights.

`load_harmless(role)` verifies the frozen manifest and requested source bytes,
then loads that role only. It does not open or deserialize final instructions
for development. The common computation and analysis kernels are in
`steering/harmless_run.py` and `analysis/harmless_results.py`. Their output schema
retains row IDs, group and quality membership, all frozen phrase matches, token
counts, quality diagnostics, MMLU scores and stage timings. Phrase offsets refer
to the case-lowered decoded completion used by the frozen heuristic.

The completed positive campaign bound development selection to final execution,
recorded code/tensor/runtime/model provenance and wrote each condition once.
`steering/harmless_execution.py` is the historical strict worker;
`scripts/run_condition.py` provides a separately labelled portable rerun and
no-model validation command. Eleven development and 13 final conditions completed
on first attempts. Public run projections and accepted aggregate analyses are in
`results/`; raw generations and individual audit labels are not part of this release.
