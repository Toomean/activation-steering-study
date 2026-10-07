# Third-party code, models and data

This study implements extraction, interventions, fixed-condition execution and
paired analysis in `src/activation_steering_study/`. The following credited work
informs specific adaptations; code comments and adjacent data READMEs pin their
sources. No blanket project licence is assigned by this notice.

| Work | Use / distinction | Licence evidence |
| --- | --- | --- |
| Arditi et al., `andyrdt/refusal_direction` | Dataset copies/preprocessing; capture adapted to block-output hooks rather than the upstream pre-hook input; 12-substring refusal proxy | Apache 2.0 code; component data licences still apply |
| Rimsky et al., `nrimsky/CAA` | Answer-letter activation extraction and contrastive A/B method | MIT code; copied upstream sample provenance is adjacent to data |
| Qwen2.5-1.5B-Instruct | Pinned checkpoint and official chat rendering API | [Pinned model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306); weights downloaded outside Git |
| TruthfulQA | Source factual pairs; locally selected and processed panels | Apache 2.0 |
| Anthropic evals | Source sycophancy data, processed upstream and sampled locally | CC BY 4.0 |
| Stanford Alpaca | Harmless instructions selected upstream for empty input; local frozen panels | CC BY-NC 4.0 data, distinct from its code licence |
| MMLU / Hendrycks test | Unchanged pinned Parquet inputs and deterministic panel manifests | MIT |
| AdvBench, HarmBench, TDC 2023 starter kit, StrongREJECT, JailbreakBench | Components of the upstream refusal dataset; original source credits are in the pinned preprocessing notebook linked by the data README | Respective MIT notices in `third_party/`; StrongREJECT notice comes from the actual `alexandrasouly/strongreject` source |
| MaliciousInstruct / Princeton SysML Jailbreak_LLM | Component of the existing upstream refusal dataset | Redistribution permission unresolved: no licence found in the inspected upstream tree or README |

`third_party/source-manifest.json` links each included notice to its retrieved source
and SHA-256. Dataset transformations and copied revisions remain documented beside
`data/refusal/`, `data/harmless/`, `data/mmlu/`, `data/honesty/` and `data/sycophancy/`.
The notices do not grant rights beyond their upstream terms. Existing historical
files were retained; the unresolved MaliciousInstruct permission prevents treating
the redistribution review as complete.

The figure script was adapted from the study's accepted figure source, with its
original and adapted hashes recorded in `results/manifest.json`. Matplotlib performs
rendering; its separately locked environment and bundled DejaVu Sans fonts are used
for portable figure builds. The scientific source kernels retain their existing
source identities.
