"""Capture completed A/B answer activations."""

from collections.abc import Sequence

import torch


def answer_activations_at_layers(
    model, tokenizer, prompt_text: str, layer_indices: Sequence[int]
) -> dict[int, dict[str, torch.Tensor]]:
    """Capture float32 block outputs at the appended answer letter for A and B.

    Each completed answer must preserve the unanswered prompt token prefix and add
    exactly one final token. The same forward passes capture every requested block.
    """
    activations: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in layer_indices}
    handles = []
    # CAA captures -2 under Llama chat. Bare A/B is Qwen's final token here.
    # https://github.com/nrimsky/CAA/blob/5dabbbd9a0bca5f25e174501e959de378806aa48/generate_vectors.py
    for layer in layer_indices:
        def capture_output(_module, _inputs, output, *, layer_index=layer):
            activations[layer_index][label] = output[0, -1].float().clone()
        handles.append(model.model.layers[layer].register_forward_hook(capture_output))
    try:
        with torch.no_grad():
            for label in ("A", "B"):
                inputs = tokenizer(prompt_text + label, add_special_tokens=False, return_tensors="pt")
                model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return activations
