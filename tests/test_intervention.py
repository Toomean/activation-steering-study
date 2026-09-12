"""Check a fixed activation intervention on one Qwen decoder block.

Chat rendering is adapted from:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
"""

import torch

from activation_steering_study.utils.qwen import load_qwen


def test_forward_hook_intervention_changes_and_restores_logits() -> None:
    # Intervention strength in activation + alpha * direction.
    # The hook reads the current value on each forward pass.
    alpha = 0.0
    tokenizer, model = load_qwen()
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    block = model.model.layers[0]

    # Use a fixed vector of ones for this hook test.
    # A forward hook can return a replacement output:
    # https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
    def add_direction(_module, _inputs, output):
        modified_output = output.clone()
        # Modify only the final-token vector of the first sequence.
        modified_output[0, -1] += alpha * torch.ones_like(modified_output[0, -1])
        return modified_output

    # Skip gradient tracking to save memory during these forward passes.
    # https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html
    with torch.no_grad():
        baseline_logits = model(**inputs).logits[0, -1]

        handle = block.register_forward_hook(add_direction)
        try:
            zero_alpha_logits = model(**inputs).logits[0, -1]

            alpha = 1.0
            intervention_logits = model(**inputs).logits[0, -1]
        finally:
            handle.remove()

        restored_logits = model(**inputs).logits[0, -1]

    assert torch.equal(baseline_logits, zero_alpha_logits), "alpha=0 changed logits"
    assert not torch.equal(baseline_logits, intervention_logits), "alpha=1 had no effect"
    assert torch.equal(baseline_logits, restored_logits), "Hook removal did not restore baseline logits"
    print("verified: alpha=0 unchanged; alpha=1 changed; hook removal restored baseline")
