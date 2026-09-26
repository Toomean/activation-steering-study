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

`tests/test_scoring.py` checks full-vocabulary A/B probabilities during a prompt-pass intervention.
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
