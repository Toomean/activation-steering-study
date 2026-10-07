# Activation Steering Study

## Study question

Do difference-in-means (DiM) directions induce the measured behaviours on
Qwen2.5-1.5B-Instruct, and what refusal and MMLU differences remain between pooled
and domain-derived refusal directions at doses frozen from development data?

The refusal directions retain unequal raw norms. Development dose selection
approximates the pooled refusal effect; final effects are not matched. The results
support inference about these panels and two fixed random directions only.

## Reproduce the released results

From the repository root, these commands use accepted aggregates and load no model:

```sh
uv sync --locked
uv run --locked python scripts/report_tables.py --check-only
uv run --locked python scripts/report_tables.py
uv run --locked python scripts/project_run_record.py
uv run --script --locked scripts/make_figures.py
```

Outputs are `build/tables.md`, `build/run-projection.json`, and SVG/PNG figures in
`build/figures/`. The figure script has its own lock for Python 3.11 and matplotlib
3.11.2; the scientific Python 3.14 environment retains its original `uv.lock`.
Run projection output is written once; choose a new `--output` path on subsequent runs.
Its `input_sha256` hashes the supplied file. For an admitted release input, the
manifest is verified and `historical_original_sha256` identifies the unpublished
original separately. New rerun records are refused by this historical projector.
The table and projection scripts also run with plain `python3` using only the
standard library. Checksum validation was tested on a Linux checkout preserving file bytes.
The plot and table commands preserve published nominal intervals as inputs.
They do not recompute intervals from unpublished paired records.

[Reproduction instructions](docs/reproduction.md) connect configurations, code,
36 completed condition records, aggregate tables and figures. They also explain
new condition reruns, historical-source archives and remaining reproduction limits.
[Third-party notices](docs/third-party-notices.md) describe data and code provenance,
including the unresolved MaliciousInstruct redistribution permission.

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

Run all tests from the repository root:

```sh
make test
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
uv run --locked pytest src/activation_steering_study/evaluation/test_choices.py
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

### Stage 17: MMLU refusal direction check

`evaluation/mmlu.py` compares next-token A-D scores for the MMLU validation sample under a fresh
baseline, the frozen block-14 refusal difference-in-means direction at alpha +1, and one seed-42
random direction with the same norm. Each prompt preserves the original question and option order;
scoring uses full-vocabulary probabilities, reports their A-D mass, and predicts the highest-probability
choice among A-D. The 285-row sample uses the validation split only; source and sampling details are in
the [MMLU data notes](data/mmlu/README.md). Results overwrite `artifacts/refusal-mmlu.json`.

```sh
make refusal-mmlu
```

### Stage 18: Shared exploratory A/B sweep

`steering/choice_sweep.py` uses the same two-order A/B scorer and completed-answer
extraction for [sycophancy](data/sycophancy/README.md) and provisional
[TruthfulQA source-labelled honesty](data/honesty/README.md). It captures blocks
6, 10, 14, 18, and 21, scales each raw direction to the fixed refusal reference
norm, and scores each block at alpha 1 against a fresh baseline and seed-42 random
control. The layer with the largest development change is selected even if every
change is negative; exact ties favour the lower block. The selected layer then gets
alpha 0.5, 1, and 2 with matched random controls. Honesty also records a block-14
negative-direction diagnostic. Outputs are `artifacts/{behaviour}-sweep.json/.pt`,
with raw pair differences, all per-source scores, settings, and hashes. The +5
percentage point benchmark is exploratory; random, order, and A/B mass diagnostics
are not extra passing criteria. Layer selection and evaluation use the same
small development set, and no final heldout is used.

```sh
make sycophancy-sweep
make honesty-sweep
```

### Stage 19: Historical exploratory refusal-domain screen

This historical screen is outside the current release reproduction commands.

The frozen provisional split adds 24 harmful extraction rows in each of cyber
intrusion, dangerous substances, and disinformation. `refusal-domain-extract`
captures their block-14 FP32 outputs against the same 32 reviewed harmless rows,
retains the existing 28-row pooled reference, and writes row tensors, raw directions,
norms, source IDs, and descriptive geometry to `artifacts/refusal-domains.pt/json`.
The geometry includes 100 seeded split-halves and 199 domain-label permutations;
neither is a distinctness test.

```sh
make refusal-domain-extract
make refusal-domain-baseline
```

The historical nonzero grid used targets
`pooled`, `cyber_intrusion`, `dangerous_substances`, and `disinformation`; alphas are
`0.5`, `1`, `2`, `-0.5`, `-1`, and `-2`; controls are `direction` and `random`.

Each condition writes `artifacts/refusal-conditions/{target}-{control}-{sign}-{magnitude}.json`
only after completing. Identical completed conditions are skipped after their input,
model, code, and baseline hashes are checked. Every run scores the fixed 285 MMLU
validation questions and generates on the same 16 harmless prompts for refusal,
clean-model response PPL, and last-prompt full-vocabulary KL. Negative conditions
also generate on the same 16 harmful prompts for suppression. The baseline is
shared, with paired MMLU changes counted against its saved item scores. Generated
token IDs and responses are retained for manual audit; absence of a substring is
only a provisional automatic label. The reported historical grid generated with
`max_new_tokens=64`, greedy decoding, and `repetition_penalty=1.1`. The current
command defaults to 256 tokens after the later cap change; it does not reproduce
the historical 64-token outputs with its default settings. A subsequent exploratory
harmless-only regeneration also used 256 tokens.

After the grid, `python -m activation_steering_study.steering.refusal_domain summarize`
writes `artifacts/refusal-conditions/summary.json` with all present curves, missing
conditions, and a descriptive nearest tested dose to pooled alpha +1 or -1. For
equal distances from the reference effect, the smaller absolute alpha is selected.

Optional `--control candidate-sycophancy` or `candidate-honesty` reads the saved
block-14 A/B tensor, scales it to that target's raw norm, and records its own A/B
source-label effect at the actual signed dose. These are exploratory candidate
controls, not certified unrelated directions. The positive and negative curves are
separate endpoints. No dose-matched equivalence or final heldout conclusion follows
from this small development screen.

### Stage 20: Harmless and MMLU computation infrastructure

The [frozen harmless metadata panel](data/harmless/README.md) supplies separate
development and final loaders. `steering/harmless_run.py` evaluates one baseline
or positive induction condition with an already loaded model and direction,
reusing generation, MMLU scoring and quality diagnostics. `analysis/harmless_results.py`
validates paired IDs and membership, weights harmless groups and MMLU subjects
equally, and returns paired point changes with nominal BCa intervals or an explicit
inconclusive result. The completed execution and provenance boundary is described in Stages 23–25.

The focused checks load frozen rows and verify manifest, source and text hashes
without model calls, and use invented fixtures for computation and analysis:

```sh
uv run --locked pytest src/activation_steering_study/data/test_harmless.py src/activation_steering_study/steering/test_harmless_run.py src/activation_steering_study/analysis/test_harmless_results.py
```


### Stage 21: Completed A/B development and final evaluation

`steering/choice_run.py` selects a dose for each sign on development groups;
`steering/choice_final.py` scores the held-out panels once at the frozen doses.
Honesty uses block 18, +2 and −2, with 128 final fact groups; sycophancy uses block 18,
+0.5 and −2, with eight final question groups. Both answer orders are averaged and
R42/R43 controls are scaled to the applied norm. The four final A/B effects were
nominally supported beyond both fixed controls. Positive sycophancy changed by
+3.48 percentage points, below its +5-point development benchmark. Conditional A/B
preferences have answer-order and option-mass limitations.

The accepted aggregate contrasts and diagnostics are in `results/ab/`; the model
entrypoint's required inputs are listed by
`uv run --locked python -m activation_steering_study.steering.choice_final --help`.
Full paired score archives and development-selection inputs are not distributed here.

### Stage 22: Block-14 unrelated-behaviour qualification

`steering/choice_control.py` re-extracts A/B directions at block 14, scales them to
each refusal target's norm and evaluates the fixed development dose grid against
both random directions. Four negative-dose sycophancy cells passed the point gate;
suitability was not established. The unrelated-behaviour control was omitted from
the new refusal comparison. `results/control/` retains all aggregate cells,
intervals and order/mass diagnostics; passing the point gate is not interval support.

### Stage 23: Completed positive refusal execution

The frozen campaign completed 11 development and 13 final conditions on their first
attempts. Pooled α=+1 was fixed; the three domain doses were selected from +0.5,
+1 and +2 by nearest development refusal effect to pooled, with smaller doses
breaking exact ties. All selected doses were +1. Development used 63 harmless
requests and 285 MMLU questions; final used 246 harmless requests in 245 groups,
63 quality members and 1,710 MMLU questions. Directions use block 14 and their raw
norms. Final controls are unsteered baseline plus R42 and R43 for each target.

`scripts/run_condition.py` validates the portable inputs and complete panel without
loading a model:

```sh
uv run --locked python scripts/run_condition.py --phase final --control real --target pooled --validate-only
```

Remove `--validate-only` and supply a fresh `--attempt-dir artifacts/rerun-pooled`
to run a new condition. This writes a separate rerun schema, records current source,
lock, tensor and reference hashes, and never overwrites an accepted historical run.
Model generation is greedy with a 256-token cap. Original campaign freezes and
ledgers are preserved as labelled projections, not rewritten portable official freezes.

### Stage 24: Positive final analysis and proxy audit

`analysis/harmless_results.py` pairs rows, weights harmless groups and MMLU
subjects equally, and calculates paired changes and nominal BCa intervals. The
15 accepted contrasts include real minus baseline, real minus each fixed random
control, and descriptive domain minus pooled gaps. Every real direction exceeded
baseline and both fixed random controls on the phrase proxy; final domain refusal
point estimates were 2.86–15.51 points below pooled. MMLU remains an estimation endpoint;
intervals including zero do not establish equivalence or absence of damage.

The primary refusal measure is the frozen 12-substring scorer. The 56-unit audit
reports both original procedurally blinded labels and the same owner's later
informed adjudication. Neither layer is independent gold. `results/audit/` contains
aggregates only; private response cards and individual ratings are excluded.
Stage 11's 48-output Jev diagnostic actually ran and is historically separate from
both the primary phrase scorer and the later advisory audit.

### Stage 25: Completed negative-MMLU supplement

`steering/negative_mmlu.py` scores 12 target/control conditions at the reversed
frozen doses on the final MMLU panel, reusing the saved positive-final baseline.
`results/negative/` contains 15 paired contrasts and subject summaries. Real minus
baseline estimates at α=−1 ranged from −0.12 to +0.53 points; all those intervals
included zero. All 15 contrasts and their nominal intervals are released.
New held-out harmful-request scoring was not done; negative
MMLU results alone do not establish refusal suppression or domain selectivity.

The portable rerun command also supports `--phase negative-mmlu`; it scores only
MMLU and uses a separate output schema. The historical strict request worker keeps
its original commit/import checks and is not claimed to accept relocated freezes.
