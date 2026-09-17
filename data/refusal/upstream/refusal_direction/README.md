# Copied upstream inputs

The JSON files in this directory were copied unchanged from
[`https://github.com/andyrdt/refusal_direction` at revision `9d852fae1a9121c78b29142de733cb1340770cc3`](https://github.com/andyrdt/refusal_direction/tree/9d852fae1a9121c78b29142de733cb1340770cc3).
The author notebook was not rerun.

| File | Original datasets | Copied from |
| --- | --- | --- |
| `advbench.json` | [AdvBench](https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv) | [`dataset/processed/advbench.json`](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/processed/advbench.json) |
| `harmful_train.json` | AdvBench, MaliciousInstruct, and TDC 2023 | [`dataset/splits/harmful_train.json`](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/splits/harmful_train.json) |
| `harmful_val.json` | HarmBench | [`dataset/splits/harmful_val.json`](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/splits/harmful_val.json) |
| `harmful_test.json` | JailbreakBench, HarmBench, and StrongREJECT | [`dataset/splits/harmful_test.json`](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/splits/harmful_test.json) |
| `harmless_train.json` | Alpaca instructions with empty input | [`dataset/splits/harmless_train.json`](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/splits/harmless_train.json) |
| `harmless_val.json` | Alpaca instructions with empty input | [`dataset/splits/harmless_val.json`](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/splits/harmless_val.json) |
| `harmless_test.json` | Alpaca instructions with empty input | [`dataset/splits/harmless_test.json`](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/splits/harmless_test.json) |

The pinned [dataset notebook](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/generate_datasets.ipynb) documents these dataset origins and preparation.
