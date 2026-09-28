# Activation Steering Study

## Study question

At matched on-target effect, do difference-in-means (DiM) activation directions from different
domains produce different collateral effects on Qwen2.5-1.5B-Instruct?

## Stages

### Stage 0: Smoke check

`src/activation_steering_study/smoke.py` loads the pinned Qwen2.5-1.5B-Instruct checkpoint using
its checkpoint dtype and decoding defaults, renders one user chat, and prints an answer of up to
40 new tokens. The answer may vary between runs. It is a load/chat/generation check, not an
experiment result.

From the repository root:

```sh
uv sync --locked
make smoke
```

The root Makefile includes the module-local smoke target. The model remains in the Hugging Face
cache outside Git. This command prints an answer and saves no manifest.

Run the routine type check from the repository root:

```sh
make typecheck
```

Run the first scoped mutation check for the bootstrap analysis with one worker, then inspect its
results:

```sh
make mutation
uv run --locked mutmut results
```

### Stage 1: Block-output capture

`tests/test_capture.py` uses real Qwen to check one forward-hook call and capture a nonempty
activation for the final prompt token.

```sh
make capture
```

### Stage 2: Hook intervention

`tests/test_intervention.py` uses real Qwen to verify that a fixed final-token intervention
leaves logits unchanged at `alpha=0`, changes them at `alpha=1`, and is removed cleanly.

```sh
make intervention
```

### Stage 3: Cached generation

`tests/test_generation.py` observes one prompt pass and two one-token decode passes with real Qwen.

```sh
make generation
```

### Stage 4: Steered generation

`src/activation_steering_study/steering/generation.py` adds a fixed all-ones vector at block 0's
final position and prints greedy baseline and steered generations.

```sh
make steer
make test-steering
```

### Stage 5: A/B prompt scoring

`src/activation_steering_study/evaluation/test_scoring.py` checks full-vocabulary A/B probabilities during a prompt-pass intervention.
It verifies alpha=0, a nonzero change, and restoration after hook removal for A=Paris and B=London.

```sh
make scoring
```

### Stage 6: Fixed refusal sample

`data/refusal/prompts.json` contains 96 fixed prompts for the exploratory pilot. The local sampler
reproduces it from copied inputs; [sample details](data/refusal/README.md) are recorded with
the data.

From the repository root:

```sh
make sample-refusal
```

### Stage 7: Exploratory refusal steering

`steering/refusal.py` uses the approved manual-review subset of 28 harmful and 32 harmless
prompts for a raw harmful-minus-harmless direction at block 14, then writes 16 baseline/steered
pairs for harmless validation prompts to `artifacts/refusal-pilot.json`. Each record retains its
source IDs and instruction; metadata records the pinned model/source IDs, shared greedy settings,
and the 64-new-token response limit. See the [manual review notes](data/review/README.md).
Rerunning overwrites the output file.

```sh
make refusal
```

### Stage 8: Reviewed refusal summary

`analysis/refusal.py` joins reviewed baseline/DiM and random-control responses by source path and
index. It reports three-condition refusal, quality, and unfinished counts; DiM minus baseline and
DiM minus random paired refusal changes; and exploratory BCa intervals. `full` and `mixed` count
as refusals; `none` counts as zero. The random control is descriptive and does not support a
quality or equivalence conclusion from marginal counts.

```sh
make analyze-refusal
```

### Stage 9: Paired refusal-change bootstrap

The same analysis command bootstraps the mean of each signed per-prompt change: DiM minus
baseline and DiM minus random. `-1` means a lost refusal, `0` no change, and `+1` a new refusal.
With the fixed direction, layer, and alpha settings, it prints an approximate, seeded 95% BCa
interval in percentage points for the small validation sample. It is exploratory: the calculation
reads the saved annotations and generations without modifying them; it does not establish
superiority or coverage.

### Stage 10: Random-direction control

`steering/random_control.py` reuses the cached 16-prompt pilot reference, its model revision,
block 14, alpha 1, and greedy 64-token settings. One fixed seed-42 random direction with the
reference norm is applied to all 16 prompts. Generated answers and provenance metadata are saved
in `artifacts/refusal-random.json`; the direction tensor is not saved, and rerunning overwrites
only that file. Review the answers using the existing refusal, quality, and unfinished labels
before comparison.

```sh
make refusal-random
```

See the [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
for the checkpoint source.

### Stage 11: Jev refusal-label diagnostic

`evaluation/jev.py` sends only each instruction and response to one Choice question, using Jev
`jev-1.13.0` and the existing `none`, `full`, `mixed`, and `unclear` refusal rubric. It judges
the 16 baseline answers once plus the 16 steered and 16 random-control answers, then prints
exact-label agreement and disagreements against the manual references. The output records each
manual reference, source ID, condition, question, answer, source-file SHA-256, and raw TypeSafe
response in `artifacts/refusal-jev.json`. This is an exploratory diagnostic, not inferential
validation. `TYPESAFE_API_KEY` must be available in the environment; if it is stored in a file,
use `uv run --env-file /path/to/.env ...`. Rerunning overwrites `artifacts/refusal-jev.json`.

```sh
uv run --locked python -m activation_steering_study.evaluation.jev
```

### Stage 12: Sycophancy A/B baseline

`evaluation/sycophancy.py` scores the 14 development questions in the [fixed 35-row
sample](data/sycophancy/README.md). Each question is rendered with original and semantically
reversed options. The output records full-vocabulary A/B probabilities, their combined mass,
and the probability of the source-matching answer conditional on A/B. It averages the two orders
per question, then the 14 question means, saving prompt and token provenance in
`artifacts/sycophancy-baseline.json`. This small source-label alignment baseline is exploratory.

```sh
make sycophancy-baseline
```

### Stage 13: Sycophancy direction extraction

`extraction/sycophancy.py` captures block 14 outputs for bare A and B appended to each of the
21 extraction questions in both option orders. For each order it subtracts the opposite answer
activation from the source-matching answer activation, averages the two orders per question, then
averages the question differences in float32. It saves the raw direction and per-order/source
differences in `artifacts/sycophancy-direction.pt`, with source and answer-token provenance in
`artifacts/sycophancy-direction.json`. Rerunning overwrites both files.

```sh
make sycophancy-direction
```

### Stage 14: Sycophancy A/B steering

`evaluation/sycophancy.py` can score an unanswered A/B prompt with a supplied direction applied
to the selected block's final prompt position. Omitting the direction keeps baseline scoring;
the hook is removed after each scoring call, including failures. The test checks zero, nonzero,
and cleanup behavior with pinned Qwen on a simple A/B question outside the study split.

```sh
uv run --locked pytest tests/test_sycophancy_steering.py
```

### Stage 15: Sycophancy steering pilot

After saving the Stage 13 direction, `steering/sycophancy.py` scores each of the 14 development
questions in original and swapped option order under baseline, positive direction at alpha +1,
negative direction at alpha -1, and a seed-42 random direction with the same raw norm. It applies
the raw direction at block 14's final prompt token and saves both order-level probabilities and
per-question means to `artifacts/sycophancy-pilot.json`. Rerunning overwrites only this output file.

```sh
make sycophancy-pilot
```

### Stage 16: Refusal-norm sycophancy comparison

After the Stage 15 raw pilot, this fixed follow-up scales the same FP32 direction to the norm
`12.335375785827637` recorded in `artifacts/refusal-pilot.json`. It scores the same 14 development
questions and both option orders under a fresh baseline, positive direction at alpha +1, and a
seed-42 random direction with the same applied norm. It uses block 14's final prompt token and
writes `artifacts/sycophancy-norm-matched.json`, leaving the raw pilot output intact.

```sh
make sycophancy-norm-matched
```
