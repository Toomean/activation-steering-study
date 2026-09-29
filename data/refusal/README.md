# Refusal pilot prompts

`prompts.json` uses `train` for direction extraction: 32 harmful prompts from the author's mixed
`harmful_train` pool and 32 Alpaca harmless prompts. `validation` holds development prompts: 16
HarmBench harmful and 16 Alpaca harmless prompts. The harmful pool includes AdvBench,
MaliciousInstruct, and TDC2023; extraction groups are unpaired.

Each pool uses independent `random.Random(42)` sampling, then sorts source indices; records keep
`sample_path`, zero-based `sample_index`, and original `instruction`.

From the repository root:

```sh
make sample-refusal
```

See the [copied upstream inputs](upstream/refusal_direction/README.md).
# Three-domain exploratory split

`domain-splits.json` and `domain-labels.json` are byte-for-byte copies of the
pre-output `method-freeze-v2` files in the private planning scratch directory.
Their SHA-256 hashes are respectively
`a0e37795b757b52d2e49669b9b3c26798676645da89cfd096bc7cbb7ff7da69d`
and `9053cd6ee2b485fb0ada2fc8fb54344209663becd3615ed75fa817804dfaf29b`.
The original instructions are from the pinned upstream refusal-direction
`harmful_train.json` at revision `9d852fae1a9121c78b29142de733cb1340770cc3`;
the [upstream processing notebook](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/generate_datasets.ipynb)
documents its construction. This study's domain labels and 24-per-domain selection
are separate prospective choices. They are provisional and include semantic
near-paraphrases, so row-level statistics do not establish task independence.
