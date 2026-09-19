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
