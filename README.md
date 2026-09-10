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

See the [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
for the checkpoint source.
