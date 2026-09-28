"""Test extraction arithmetic and Qwen hidden-state integration."""

import pytest
import torch

from activation_steering_study.evaluation.choices import prepare_variants
from activation_steering_study.extraction import paired
from activation_steering_study.utils.choice_prompt import (
    ANSWER_SUFFIX,
    prepare_choice_prompt,
)
from activation_steering_study.utils.qwen import load_qwen


@pytest.fixture(scope="module")
def qwen():
    return load_qwen()


def test_extract_choice_pairs_known_arithmetic(qwen, monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer, model = qwen
    sources = [
        (7, prepare_variants("Which animal barks?", "Dog", "Cat", "A")),
        (3, prepare_variants("Which animal meows?", "Dog", "Cat", "B")),
    ]
    raw_answers = [
        {"A": (3, 5), "B": (1, 1)},
        {"A": (1, 1), "B": (5, 7)},
        {"A": (1, 1), "B": (7, 9)},
        {"A": (9, 11), "B": (1, 1)},
    ]
    variants = [variant for _, orders in sources for variant in orders]
    canned_by_prompt = {}
    for variant, answers in zip(variants, raw_answers, strict=True):
        prompt_text = prepare_choice_prompt(
            tokenizer, variant["question"] + ANSWER_SUFFIX, "AB"
        )["prompt_text"]
        canned_by_prompt[prompt_text] = {
            label: torch.tensor(values, dtype=torch.float32)
            for label, values in answers.items()
        }

    def known_answer_activations(_model, _tokenizer, prompt_text, layer_indices):
        assert prompt_text in canned_by_prompt, "No canned answers for rendered prompt"
        return {layer: canned_by_prompt[prompt_text] for layer in layer_indices}

    monkeypatch.setattr(paired, "answer_activations_at_layers", known_answer_activations)
    extracted, rows = paired.extract_choice_pairs(
        model, tokenizer, sources, [14]
    )

    assert [(row["source_index"], row["order"]) for row in rows] == [
        (7, "original"), (7, "swapped"), (3, "original"), (3, "swapped")
    ], "Rows did not preserve the supplied source and variant order"
    assert torch.equal(
        extracted[14]["pair_differences"],
        torch.tensor([[[2, 4], [4, 6]], [[6, 8], [8, 10]]], dtype=torch.float32),
    ), "Pair differences did not subtract opposite answers in each variant"
    assert torch.equal(
        extracted[14]["source_differences"],
        torch.tensor([[3, 5], [7, 9]], dtype=torch.float32),
    ), "Source differences did not average the two variants per question"
    assert torch.equal(
        extracted[14]["direction"], torch.tensor([5, 7], dtype=torch.float32)
    ), "Direction did not average both source questions"


def test_extract_choice_pairs_matches_qwen_hidden_states(qwen) -> None:
    tokenizer, model = qwen
    layers = (6, 14)
    variants = prepare_variants("Which animal barks?", "Dog", "Cat", "A")
    extracted, _ = paired.extract_choice_pairs(model, tokenizer, [(7, variants)], layers)
    prompt_text = prepare_choice_prompt(
        tokenizer, variants[0]["question"] + ANSWER_SUFFIX, "AB"
    )["prompt_text"]
    hidden_by_answer = {}
    for label in ("A", "B"):
        inputs = tokenizer(prompt_text + label, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            hidden_by_answer[label] = model(**inputs, output_hidden_states=True).hidden_states

    for layer in layers:
        # hidden_states[0] is embeddings; block outputs start at layer + 1.
        # Source: https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast.hidden_states
        expected = (
            hidden_by_answer["A"][layer + 1][0, -1].float()
            - hidden_by_answer["B"][layer + 1][0, -1].float()
        )
        assert torch.equal(extracted[layer]["pair_differences"][0, 0], expected), (
            f"Block {layer}: original A-minus-B difference differs from hidden states"
        )
        assert extracted[layer]["direction"].dtype == torch.float32, (
            f"Block {layer}: direction must be float32"
        )
