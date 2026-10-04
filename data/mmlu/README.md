# MMLU sources and fixed panels

The local files in `upstream/cais/` are unchanged copies from
[`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) at revision
`c30699e8356da336a370243923dbaf21066bb9fe`:

| Split | Pinned source | Rows | SHA256 |
| --- | --- | --- | --- |
| validation | [all/validation-00000-of-00001.parquet](https://huggingface.co/datasets/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/all/validation-00000-of-00001.parquet) | 1531 | `66cdf0b090ccb657d18d13cd81e31fdc55c3467da9642ffb178653268a97c8ef` |
| test | [all/test-00000-of-00001.parquet](https://huggingface.co/datasets/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/all/test-00000-of-00001.parquet) | 14042 | `74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7` |

Records contain `question`, `choices`, `answer`, and `subject`; integer answers 0–3
map to A–D. Questions, choice order, and gold answers are preserved. Upstream MMLU
is distributed under the [MIT License](https://github.com/hendrycks/test/blob/master/LICENSE),
Copyright (c) 2020 Dan Hendrycks.

1. `sample_mmlu()` retains the 285-row calibration sample: five validation rows per
   sorted subject, one continuing `random.Random(42)`, sorted selected indices.
2. `load_mmlu_final()` loads the explicit IDs and content hashes in
   [test-1710.json](test-1710.json), checks source bytes and the frozen ordered-ID
   digest, and returns 30 test rows per subject. It performs no sampling. Each row
   hash covers UTF-8 compact JSON `[question, choices, answer]`, with
   `ensure_ascii=False` and `separators=(",", ":")`.

IDs retain `{subject}_val.csv:{index}` or `{subject}_test.csv:{index}`. The zero-based
index is subject-relative order in the pinned Parquet; separate CSV ordering is not asserted.

The final panel began with 30 randomly sampled test indices per sorted subject using
one continuing `random.Random(42)`. Exact deduplication uses question + ordered choices
(the model input), with gold comparison secondary. Against all 1531 validation rows,
remove exact matches; for selected duplicates keep the first in subject/index order.
Reserve every original ID and every retained key. In removed-row order, replace each
with the smallest unused same-subject index excluding validation, retained, and prior
replacement keys; sort the final IDs by subject and integer index.

| Removed ID | Replacement ID |
| --- | --- |
| college_medicine_test.csv:54 | college_medicine_test.csv:2 |
| college_medicine_test.csv:60 | college_medicine_test.csv:4 |
| college_physics_test.csv:20 | college_physics_test.csv:0 |
| college_physics_test.csv:61 | college_physics_test.csv:1 |
| college_physics_test.csv:76 | college_physics_test.csv:2 |

The accepted panel contains 1705 original sampled rows plus five deterministic
replacements. It has no exact model-input duplicates or validation overlaps;
two within-panel pairs share question text with different choices and are retained
because the exact key includes ordered choices. Semantic overlap and near-duplicates
were not assessed. The panel is unscored and
has no final runner/scorer hookup. Run the focused data checks from the repository
root with `uv run --locked pytest src/activation_steering_study/data/test_mmlu.py
src/activation_steering_study/data/test_mmlu_final.py`.
