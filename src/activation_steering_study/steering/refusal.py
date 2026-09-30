"""Save baseline and steered answers for harmless validation prompts."""

from pathlib import Path

from activation_steering_study.data.refusal import load_reviewed_refusal
from activation_steering_study.extraction.mean import mean_activation
from activation_steering_study.steering.pilot import (
    GenerationKwargs,
    generate_answers,
    save_results,
)
from activation_steering_study.utils.qwen import MODEL_ID, MODEL_REVISION, load_qwen


def main() -> None:
    # Zero-based block-output index for this exploratory mid-depth check.
    layer_index = 14
    # Strength in h + alpha * d.
    alpha = 1.0
    prompts = load_reviewed_refusal()
    tokenizer, model = load_qwen()
    harmful = [record["instruction"] for record in prompts["train"]["harmful"]]
    harmless = [record["instruction"] for record in prompts["train"]["harmless"]]
    _, harmful_mean = mean_activation(model, tokenizer, harmful, layer_index)
    _, harmless_mean = mean_activation(model, tokenizer, harmless, layer_index)

    # Arditi method context: https://doi.org/10.52202/079017-4322
    direction = harmful_mean - harmless_mean
    print(
        f"layer={layer_index}, direction shape={tuple(direction.shape)}, "
        f"norm={direction.norm().item():.4f}"
    )

    generation_kwargs: GenerationKwargs = {"max_new_tokens": 256, "do_sample": False}
    results = generate_answers(
        model,
        tokenizer,
        prompts["validation"]["harmless"],
        {"baseline": None, "steered": direction},
        layer_index,
        alpha,
        generation_kwargs,
    )
    save_results(
        Path("artifacts/refusal-pilot.json"),
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_dtype": str(model.dtype),
            "layer_index": layer_index,
            "alpha": alpha,
            "direction_norm": direction.norm().item(),
            "generation_kwargs": generation_kwargs,
            "source_revision": prompts["source_revision"],
            "sampling_seed": prompts["seed"],
            "train_counts": {
                "harmful": len(prompts["train"]["harmful"]),
                "harmless": len(prompts["train"]["harmless"]),
            },
        },
        results,
    )


if __name__ == "__main__":
    main()
