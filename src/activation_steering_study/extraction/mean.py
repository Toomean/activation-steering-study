"""Compute mean block-output activations for rendered user prompts."""

import torch


def mean_activation(model, tokenizer, prompts, layer_index):
    """Return float32 final-token block outputs and their mean for user prompts."""
    activations = []
    block = model.model.layers[layer_index]

    # Arditi uses a pre-hook on block input; this adaptation captures block output instead.
    # https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/pipeline/submodules/generate_directions.py
    def capture_output(_module, _inputs, output):
        # Capture the final rendered chat token, after the assistant header.
        activations.append(output[0, -1].float().clone())

    handle = block.register_forward_hook(capture_output)
    try:
        # Skip gradient tracking while collecting activations.
        # https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html
        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                model(**inputs)
    finally:
        handle.remove()

    activations = torch.stack(activations)
    return activations, activations.mean(0)
