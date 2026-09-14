"""Generate with a fixed activation intervention.

Chat rendering is adapted from:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
"""

import torch

from activation_steering_study.utils.qwen import load_qwen


def generate_with_intervention(model, inputs, direction, alpha, **generation_kwargs):
    """Generate after adding a direction at block 0's final token of the first sequence."""
    block = model.model.layers[0]

    # A forward hook can return a replacement output:
    # https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
    def add_direction(_module, _inputs, output):
        modified_output = output.clone()
        modified_output[0, -1] += alpha * direction
        return modified_output

    handle = block.register_forward_hook(add_direction)
    try:
        # Preserve the standard generation API for demos and raw-logit tests.
        # https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate
        return model.generate(**inputs, **generation_kwargs)
    finally:
        handle.remove()


def main() -> None:
    # Strength in activation + alpha * direction.
    alpha = 1.0
    tokenizer, model = load_qwen()
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    direction = torch.ones(model.config.hidden_size, dtype=model.dtype, device=model.device)

    # Greedy decoding makes the baseline and steered texts comparable.
    baseline = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    steered = generate_with_intervention(
        model,
        inputs,
        direction,
        alpha,
        max_new_tokens=40,
        do_sample=False,
    )
    prompt_length = inputs["input_ids"].shape[-1]
    baseline_text = tokenizer.decode(baseline[0, prompt_length:], skip_special_tokens=True).strip()
    steered_text = tokenizer.decode(steered[0, prompt_length:], skip_special_tokens=True).strip()
    print(f"baseline: {baseline_text}")
    print(f"steered: {steered_text}")


if __name__ == "__main__":
    main()
