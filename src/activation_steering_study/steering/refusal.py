"""Compare one harmless development generation with a raw DiM direction."""

import json
from pathlib import Path

from activation_steering_study.extraction.mean import mean_activation
from activation_steering_study.steering.generation import generate_with_intervention
from activation_steering_study.utils.qwen import load_qwen


def main() -> None:
    # Zero-based block-output index for this exploratory mid-depth check.
    layer_index = 14
    # Strength in h + alpha * d.
    alpha = 1.0
    prompts = json.loads(Path("data/refusal/prompts.json").read_text(encoding="utf-8"))
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

    prompt = prompts["validation"]["harmless"][0]["instruction"]
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    baseline = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    steered = generate_with_intervention(
        model,
        inputs,
        direction,
        alpha,
        layer_index=layer_index,
        max_new_tokens=64,
        do_sample=False,
    )
    prompt_length = inputs["input_ids"].shape[-1]
    baseline_text = tokenizer.decode(baseline[0, prompt_length:], skip_special_tokens=True).strip()
    steered_text = tokenizer.decode(steered[0, prompt_length:], skip_special_tokens=True).strip()
    print(f"prompt: {prompt}")
    print(f"baseline: {baseline_text}")
    print(f"steered: {steered_text}")


if __name__ == "__main__":
    main()
