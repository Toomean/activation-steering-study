"""Check one Qwen decoder-block output with a forward hook.

Chat rendering is adapted from:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
"""

import torch

from activation_steering_study.utils.qwen import load_qwen


def test_forward_hook_captures_once() -> None:
    tokenizer, model = load_qwen()
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    activations = []

    # Adapted registration/removal pattern; this hook observes output only:
    # https://docs.pytorch.org/docs/2.14/notes/modules.html#module-hooks
    def capture(_module, _inputs, output):
        # output contains hidden states for all input tokens: [batch, tokens, hidden_size].
        # Save the final-token vector of the first sequence.
        activations.append(output[0, -1].clone())

    block = model.model.layers[0]
    handle = block.register_forward_hook(capture)
    try:
        # No backward pass is needed for capture; skip gradient tracking to save memory.
        # https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    print(f"hook calls: {len(activations)}")
    assert len(activations) == 1, f"Expected one hook call, got {len(activations)}"
    activation = activations[0]
    assert activation.numel() > 0, "Captured activation is empty"
    print(f"activation: shape={tuple(activation.shape)} dtype={activation.dtype}")
    print(f"activation first 8 values: {activation[:8].tolist()}")
