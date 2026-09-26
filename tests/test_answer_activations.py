"""Check answer-letter block outputs against Qwen hidden states."""

import pytest
import torch

from activation_steering_study.extraction.paired import answer_activations
from activation_steering_study.utils.qwen import load_qwen


@pytest.fixture(scope="module")
def qwen_answer_prompt():
    tokenizer, model = load_qwen()
    prompt_text = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Which animal barks?\nA. Dog\nB. Cat\n\nAnswer with A or B only.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer, model, prompt_text


def test_answer_activations_match_answer_positions(qwen_answer_prompt) -> None:
    tokenizer, model, prompt_text = qwen_answer_prompt
    layer_index = 14
    answer_position = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    actual = answer_activations(model, tokenizer, prompt_text, layer_index)

    assert set(actual) == {"A", "B"}, "Capture did not return both answer letters"
    for label in ("A", "B"):
        inputs = tokenizer(prompt_text + label, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
        # hidden_states[0] contains embeddings; layer_index + 1 gives this block's output.
        # Source: https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast.hidden_states
        expected = output.hidden_states[layer_index + 1][0, answer_position].float()
        assert actual[label].dtype == torch.float32, f"Answer {label} was not float32"
        assert torch.equal(actual[label], expected), (
            f"Answer {label} did not match its hidden state"
        )


def test_answer_activations_removes_hook(qwen_answer_prompt) -> None:
    tokenizer, model, prompt_text = qwen_answer_prompt
    block = model.model.layers[14]
    hook_count = len(block._forward_hooks)

    answer_activations(model, tokenizer, prompt_text, 14)

    assert len(block._forward_hooks) == hook_count, "Answer capture did not remove its hook"
