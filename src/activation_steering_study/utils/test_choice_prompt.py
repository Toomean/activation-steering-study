"""Test unanswered choice prompt rendering and next-token boundaries."""

from typing import cast

import pytest
import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase

from activation_steering_study.utils.choice_prompt import prepare_choice_prompt
from activation_steering_study.utils.qwen import load_qwen


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerBase:
    tokenizer, _ = load_qwen()
    return tokenizer


@pytest.mark.parametrize("labels", ["AB", "ABCD"])
def test_prepares_unanswered_choice_prompt(
    tokenizer: PreTrainedTokenizerBase, labels: str
) -> None:
    content = f"Answer with one of these letters: {', '.join(labels)}."
    messages = [{"role": "user", "content": content}]
    expected_prompt_text = cast(
        str,
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ),
    )
    expected_inputs = cast(
        BatchEncoding,
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ),
    )
    prepared = prepare_choice_prompt(tokenizer, content, labels)

    assert prepared["content"] == content, "Helper changed the user content"
    assert prepared["prompt_text"] == expected_prompt_text, "Chat template output changed"
    assert prepared["prompt_text"].endswith("<|im_start|>assistant\n"), (
        "Prompt must end at the assistant generation prefix without a completed answer"
    )
    assert torch.equal(prepared["inputs"]["input_ids"], expected_inputs["input_ids"]), (
        "Model input IDs differ from independently tokenized chat template"
    )
    assert prepared["prompt_token_ids"] == expected_inputs["input_ids"][0].tolist(), (
        "Returned prompt IDs differ from independently tokenized chat template"
    )
    assert torch.equal(
        prepared["inputs"]["attention_mask"], expected_inputs["attention_mask"]
    ), "Model attention mask differs from independently tokenized chat template"

    expected_answer_ids: dict[str, int] = {}
    for label in labels:
        completed_ids = tokenizer.encode(
            expected_prompt_text + label, add_special_tokens=False
        )
        assert completed_ids[:-1] == prepared["prompt_token_ids"] and len(
            completed_ids
        ) == len(prepared["prompt_token_ids"]) + 1, (
            f"Appending {label} must preserve the prompt prefix and add one token"
        )
        expected_answer_ids[label] = completed_ids[-1]
    assert prepared["answer_token_ids"] == expected_answer_ids, (
        "Returned answer IDs differ from independently encoded completions"
    )


@pytest.mark.parametrize(
    "label", ["\n", "🫠"], ids=["changed-prefix", "multiple-tokens"]
)
def test_rejects_labels_that_cannot_be_scored_as_one_next_token(
    tokenizer: PreTrainedTokenizerBase, label: str
) -> None:
    # A newline merges with the assistant prefix; the emoji adds three tokens.
    with pytest.raises(ValueError, match="changed prompt tokenization"):
        prepare_choice_prompt(tokenizer, "Choose a label.", label)
