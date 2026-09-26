"""Capture block outputs at completed A/B answer letters."""

import torch


def answer_activations(model, tokenizer, prompt_text: str, layer_index: int) -> dict[str, torch.Tensor]:
    """Return float32 block outputs for bare A and B appended to a rendered prompt.

    ``prompt_text`` is an unanswered chat prompt ending at the assistant prefix.
    Each answer must preserve its tokenized prefix and add one final token.
    """
    activations: dict[str, torch.Tensor] = {}
    block = model.model.layers[layer_index]

    # CAA captures a completed answer at -2 under its Llama chat wrapper. Here the
    # bare answer letter is the final Qwen token, so capture its block output at -1.
    # https://github.com/nrimsky/CAA/blob/5dabbbd9a0bca5f25e174501e959de378806aa48/generate_vectors.py
    def capture_output(_module, _inputs, output):
        activations[label] = output[0, -1].float().clone()

    handle = block.register_forward_hook(capture_output)
    try:
        with torch.no_grad():
            for label in ("A", "B"):
                inputs = tokenizer(prompt_text + label, add_special_tokens=False, return_tensors="pt")
                model(**inputs)
    finally:
        handle.remove()

    return activations
