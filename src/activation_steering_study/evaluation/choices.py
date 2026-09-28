"""Prepare and score two-order A/B contrasts shared by source-labelled behaviours."""

import math
from typing import TypedDict

import torch
from transformers import PreTrainedTokenizerBase, Qwen2ForCausalLM

from activation_steering_study.evaluation.scoring import score_answer_tokens
from activation_steering_study.utils.choice_prompt import ANSWER_SUFFIX, prepare_choice_prompt


class ChoicePromptVariant(TypedDict):
    order: str
    question: str
    options: dict[str, str]
    matching_label: str


class ScoredChoiceVariant(ChoicePromptVariant):
    content: str
    prompt_text: str
    prompt_token_ids: list[int]
    answer_token_ids: dict[str, int]
    p_a: float
    p_b: float
    ab_mass: float
    conditional_matching_probability: float


def prepare_variants(
    question_prefix: str,
    option_a: str,
    option_b: str,
    matching_label: str,
    *,
    original_question: str | None = None,
) -> list[ChoicePromptVariant]:
    """Reverse displayed options while keeping the source-labelled meaning fixed."""
    if matching_label not in ("A", "B"):
        raise ValueError(f"Unknown matching label: {matching_label}")
    original = original_question or f"{question_prefix}\n (A) {option_a}\n (B) {option_b}"
    return [
        {
            "order": "original", "question": original,
            "options": {"A": option_a, "B": option_b},
            "matching_label": matching_label,
        },
        {
            "order": "swapped",
            "question": f"{question_prefix}\n (A) {option_b}\n (B) {option_a}",
            "options": {"A": option_b, "B": option_a},
            "matching_label": "B" if matching_label == "A" else "A",
        },
    ]


def score_choice_variant(
    tokenizer: PreTrainedTokenizerBase,
    model: Qwen2ForCausalLM,
    variant: ChoicePromptVariant,
    *,
    direction: torch.Tensor | None = None,
    alpha: float = 1.0,
    layer_index: int = 14,
) -> ScoredChoiceVariant:
    """Score full-vocabulary A/B probabilities on an unanswered prompt.

    The optional direction is added only at the final prompt token of the selected
    block. Completed answer letters are used to verify next-token IDs, never sent
    to the scoring forward pass.
    """
    prepared = prepare_choice_prompt(tokenizer, variant["question"] + ANSWER_SUFFIX, "AB")
    answer_token_ids = prepared["answer_token_ids"]
    probabilities = score_answer_tokens(
        model, prepared["inputs"], [answer_token_ids["A"], answer_token_ids["B"]],
        direction=direction, alpha=alpha, layer_index=layer_index,
    )
    p_a, p_b = (float(probability) for probability in probabilities)
    mass = p_a + p_b
    if not (mass > 0 and math.isfinite(p_a) and math.isfinite(p_b)):
        raise ValueError("Nonfinite or zero A/B probability mass")
    return {
        "order": variant["order"],
        "question": variant["question"],
        "options": variant["options"],
        "matching_label": variant["matching_label"],
        "content": prepared["content"],
        "prompt_text": prepared["prompt_text"],
        "prompt_token_ids": prepared["prompt_token_ids"],
        "answer_token_ids": answer_token_ids,
        "p_a": p_a, "p_b": p_b, "ab_mass": mass,
        "conditional_matching_probability": (
            p_a if variant["matching_label"] == "A" else p_b
        ) / mass,
    }
