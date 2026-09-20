"""Save baseline and steered answers for harmless validation prompts."""

import json
from pathlib import Path

from activation_steering_study.data.refusal import load_reviewed_refusal
from activation_steering_study.extraction.mean import mean_activation
from activation_steering_study.steering.generation import generate_with_intervention
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

    generation_kwargs = {"max_new_tokens": 64, "do_sample": False}
    results = []
    for record in prompts["validation"]["harmless"]:
        prompt = record["instruction"]
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        baseline = model.generate(**inputs, **generation_kwargs)
        steered = generate_with_intervention(
            model,
            inputs,
            direction,
            alpha,
            layer_index=layer_index,
            **generation_kwargs,
        )
        prompt_length = inputs["input_ids"].shape[-1]
        baseline_text = tokenizer.decode(
            baseline[0, prompt_length:], skip_special_tokens=True
        ).strip()
        steered_text = tokenizer.decode(
            steered[0, prompt_length:], skip_special_tokens=True
        ).strip()
        print(f"sample: {record['sample_path']}:{record['sample_index']}")
        print(f"prompt: {prompt}")
        print(f"baseline: {baseline_text}")
        print(f"steered: {steered_text}")
        results.append(
            {
                "sample_path": record["sample_path"],
                "sample_index": record["sample_index"],
                "instruction": prompt,
                "baseline": baseline_text,
                "steered": steered_text,
            }
        )

    output = Path("artifacts/refusal-pilot.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
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
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
