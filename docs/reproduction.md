# Reproduction and result trace

Run commands from the repository root. `uv sync --locked` installs the scientific
Python 3.14 environment. The figure script has independent inline metadata and a
checked-in `scripts/make_figures.py.lock` for Python 3.11, matplotlib 3.11.2 and its
transitive dependencies. Changing that environment does not change `uv.lock`.

## Rebuild accepted tables and figures without a model

```sh
uv run --locked python scripts/report_tables.py --check-only
uv run --locked python scripts/report_tables.py --output build/tables.md
uv run --locked python scripts/project_run_record.py --output build/run-projection.json
uv run --script --locked scripts/make_figures.py --output-dir build/figures
```

The first command verifies every admitted input against `results/manifest.json`.
The table command formats all positive and negative contrasts, development doses,
A/B contrasts, block-14 qualification cells, descriptive geometry and both audit
layers. The figure command verifies numeric artist positions and interval ends,
label bounds and minimum font sizes; it writes SVG plus 300-dpi PNG. The schematic
uses the accepted methods layout with bundled DejaVu Sans font metrics.

These commands use accepted aggregate estimates and nominal interval bounds as
versioned inputs. They do not recompute BCa intervals, audit individual responses,
or score model outputs. A clean run requires no model weights or external service.
The build directory is reproducible output and is ignored by Git.
The table and projection scripts require only the standard library and can also
be invoked with `python3`. These checks were tested on Linux with a checkout
preserving the committed bytes; newline conversion changes checksum identities.

| Study component | Configuration / computation | Run provenance | Released numeric input / output |
| --- | --- | --- | --- |
| Development dose choice; Figure 4.1 | `harmless_execution.development_specs`, `select_domain_doses`; positive freeze and dose-freeze projections | 11 `development-*` records in `results/runs/` | `results/development/development.csv`, `dose-selection.csv`; development figure |
| Positive final; Figure 4.2 | `harmless_run.evaluate_condition`, `harmless_results`; frozen +1 doses | 13 `final-*` records | `results/positive/final-contrasts.csv`, `final-plot-data.csv`; final figure |
| Negative MMLU | `negative_mmlu.evaluate_mmlu`, `harmless_results`; reversed frozen doses | 12 `negative-mmlu-*` records, request projections and shared positive baseline hash | `results/negative/negative-contrasts.csv`, subject summaries |
| Final A/B | `choice_final`, `choice_results`; block 18 and sign-specific development doses | `results/ab/*-run.json`, condition/summary hashes | `results/ab/*-final.json` |
| Block-14 qualification | `choice_control`, `choice_control_results`; norms scaled by target and fixed grid | `results/control/*-run.json` | `results/control/*-qualification.json` |
| Direction geometry | `analysis/geometry.py`; descriptive historical extraction | `results/provenance/geometry-provenance.json` | cosine and split-half CSVs in `results/development/` |
| Proxy audit | frozen phrase scorer and original / informed owner labels; aggregate audit source archive | original aggregate hash bindings in release manifest | `results/audit/audit-summary.csv/json` |
| Methods pipeline; Figure 3.1 | preserved schematic from accepted plotting source | original and adapted script hashes in release manifest | pipeline figure |

`results/run-index.json` indexes all 36 completed positive-core and negative-MMLU
conditions, with condition metadata, configuration hashes, completed-artifact
hashes and original run-manifest hashes. Each run projection retains input and
output hash bindings. Negative request projections preserve dose/baseline/vector
bindings. They omit machine environment maps and replace paths with logical
locations. `unpublished/` and `historical/` identify provenance locations, not files
that a marker should expect to find in this checkout.

`results/manifest.json` records the original SHA-256 and projected SHA-256
separately. An unchanged CSV has the same hashes; a projection never claims the
original record's byte identity. Original execution records, official freezes and
ledgers were not changed. Numeric development columns and completed hashes survive
the removal of private absolute paths.
The projector's `input_sha256` always hashes the actual supplied bytes. When the
input is an admitted release file, it verifies the release manifest and emits the
separate `historical_original_sha256` from that entry. It does not infer a historical
identity for an external input. `projection_sha256` hashes UTF-8 indented JSON of
the `projection` member plus one newline, as recorded in `projection_hash_encoding`.
The historical projector refuses rerun metadata/results rather than removing their marker.

## Validate or run a new condition

```sh
uv run --locked python scripts/run_condition.py --phase development --control real --target cyber_intrusion --alpha 0.5 --validate-only
uv run --locked python scripts/run_condition.py --phase final --control real --target pooled --validate-only
uv run --locked python scripts/run_condition.py --phase negative-mmlu --control random42 --target disinformation --validate-only
```

Validation checks the release checksums, pinned model/revision, block, original
tensor bytes, float32 vector shapes and norms, exact vector hashes against the
admitted historical direction record, frozen queue membership and complete
panel counts. No weights are loaded. Final doses come from the admitted dose-freeze;
negative mode uses their reversed signs. An arbitrary subset or altered dose is
not accepted as the frozen condition.

To execute a new final condition, omit `--validate-only` and add a fresh output path:

```sh
uv run --locked python scripts/run_condition.py --phase final --control real --target pooled --attempt-dir artifacts/rerun-pooled
```

This loads the pinned model and calls the unchanged scientific computation kernel.
It writes `metadata.json` and `result.json` with `record_kind: release-rerun`, enclosing
an observed `runtime.json`. These record current
source and dependency identities, exact vector bytes, signed dose, norm magnitude,
panel counts, original/projected reference hashes and accepted condition, configuration,
run projection and historical original identities. The unchanged shared runtime
validator checks the pinned model/tokenizer contract and CPU bfloat16/device and
records the current attention implementation, dependency versions and thread counts
with environment self-consistency checks. It does not compare thread counts to the
historical 12 intra-op / 24 inter-op setting. Before positive evaluation, the wrapper
checks the loaded model's inherited repetition penalty equals 1.1 and records that
observed value in `runtime.json`; metadata labels the expected value explicitly.
Greedy decoding and the 256-token cap are explicit kernel arguments. These checks
do not promise bit-for-bit historical outputs. Result records bind the metadata and
runtime hashes. Only fresh repository-relative directories under ignored `artifacts/`
are allowed; existing directories and symlink escapes are refused. The model remains
outside Git. The negative mode calls the existing
MMLU evaluator only and generates no harmful requests. Rerun outputs are not the
accepted results and do not recreate the historical supervisor's official records.

The original tensor includes numeric row activations, means and four directions,
not model weights or text. The original numeric reference contained a private path.
Its public projection changes that path and therefore its byte hash. The strict
historical `harmless_execution.prepare_direction` still requires the original
reference digest; the portable rerun validates the projected scientific fields and
exact original tensor separately. It does not weaken or change the historic worker.

## Historical source and reproduction limits

`docs/historical-source/*.py.txt` preserves inspectable campaign orchestration,
positive/negative full analysis exports and audit aggregation, with original and
projected source hashes in the manifest. Transformations are recorded per file:
equal hashes indicate unchanged bytes needing no path replacement; other archives
replace private paths with logical locations. These archives are not executable relocated campaigns: the original
supervisors bind original clean commits, import origins, environment identities,
non-overwrite ledgers and a historical hard stop. Raw paired records and individual
audit cards/labels remain unpublished. The positive exporter also had private card
creation responsibilities that are not part of public table regeneration.

The public package therefore supports accepted aggregate tables/figures, input
validation and new fixed-condition reruns. Full independent recomputation of accepted
paired intervals requires the original per-item score records, frozen analysis
versions and ordering. Reproducing individual audit aggregation requires the
unpublished label layers and mapping. `results/ab/*-panel-metadata.json` exposes ordered source IDs, split assignments and
groups; `results/ab/selection.json` preserves sign-specific dose choices and original
input hashes. Neither supplies the curated text. A/B full reruns also require the unprovided
accepted development panels, tensors and selection archives. These limitations
must not be presented as interval or individual-label reproduction.

The positive scientific source closure and `uv.lock` are unchanged between the
accepted positive core `249d0cd` and release base `411b57f`; that difference adds
only the negative-MMLU caller and its tests. The accepted negative caller is also
preserved byte-for-byte and applies block 14 internally. The portable wrapper uses
these kernels without claiming to recreate the original supervisors or environments.

Geometry source records show that the mapped current sources do not byte-match
the recorded historical source identities. The geometry values are accepted
historical aggregates, not a claim of exact-source regeneration. Historical Stage 19
used 64-token generations; the current exploratory default and later harmless-only
regeneration use 256. Final induction also uses 256; those generated responses are
not directly interchangeable with the 64-token historical outputs.

The primary final refusal endpoint uses 12 fixed substrings. The 48-output Jev
exploratory diagnostic was a separate historical measurement; later advisory audit
annotations do not create an independent human rater or independent gold standard.
The audit release contains aggregates for both the original procedurally blinded
and same-owner informed layers. Negative MMLU alone does not establish suppression.
No equivalence, no-effect or domain-selectivity conclusion follows from this package.
