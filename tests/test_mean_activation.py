"""Check block-output mean capture against Qwen hidden states."""

import torch

from activation_steering_study.extraction.mean import mean_activation
from activation_steering_study.utils.qwen import load_qwen


def test_mean_activation_matches_hidden_states_and_removes_hook() -> None:
    layer_index = 14
    prompts = ["Name a dog breed.", "Name a cat breed."]
    tokenizer, model = load_qwen()
    expected_activations = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            output = model(**inputs, output_hidden_states=True)
            # hidden_states[0] contains embeddings; layer_index + 1 gives this block's output.
            # https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast.hidden_states
            expected_activations.append(output.hidden_states[layer_index + 1][0, -1].float())

    block = model.model.layers[layer_index]
    hook_count = len(block._forward_hooks)
    actual_activations, actual_mean = mean_activation(model, tokenizer, prompts, layer_index)

    expected_activations = torch.stack(expected_activations)
    assert actual_activations.dtype == torch.float32, "Captured activations were not float32"
    assert actual_mean.dtype == torch.float32, "Mean activation was not float32"
    assert torch.equal(actual_activations, expected_activations), (
        "Captured block outputs did not match the corresponding hidden states"
    )
    assert torch.equal(actual_mean, expected_activations.mean(0)), (
        "Mean block output did not match the corresponding hidden states"
    )
    assert len(block._forward_hooks) == hook_count, "Mean capture did not remove its hook"
