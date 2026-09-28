# MMLU validation source

The local copy at `upstream/cais/validation-00000-of-00001.parquet` is copied unchanged from
[`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu), specifically the pinned
[`all/validation-00000-of-00001.parquet`](https://huggingface.co/datasets/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/all/validation-00000-of-00001.parquet).
It contains 1,531 validation rows from 57 subjects. Records contain `question`, `choices`,
`answer`, and `subject`; integer answers 0–3 map to A–D. This study uses the validation split only.

The study samples five rows per subject in sorted subject order using one continuing
`random.Random(42)`, then sorts the selected within-subject row indices. It preserves each question,
original choice order, and integer answer. IDs retain the calibration form
`{subject}_val.csv:{index}`, where the index is within that subject.
