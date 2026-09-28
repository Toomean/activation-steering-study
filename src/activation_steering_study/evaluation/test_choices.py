"""Check generic A/B choice scoring with a fixed block-output intervention."""

import pytest
import torch

from activation_steering_study.evaluation.choices import ChoicePromptVariant, score_choice_variant
from activation_steering_study.utils.qwen import load_qwen


VARIANT: ChoicePromptVariant = {
    "order": "original",
    "question": "Which animal barks?\n (A) Dog\n (B) Cat",
    "options": {"A": "Dog", "B": "Cat"},
    "matching_label": "A",
}


@pytest.fixture(scope="module")
def qwen():
    return load_qwen()


def test_alpha_zero_matches_baseline_score(qwen) -> None:
    tokenizer, model = qwen
    baseline = score_choice_variant(tokenizer, model, VARIANT)
    direction = torch.ones(
        model.config.hidden_size, dtype=torch.float32, device=model.device
    )

    zero = score_choice_variant(tokenizer, model, VARIANT, direction=direction, alpha=0.0)

    assert zero == baseline, "alpha=0 changed the scored A/B result"


def test_nonzero_changes_probabilities_and_removes_hook(qwen) -> None:
    tokenizer, model = qwen
    block = model.model.layers[14]
    hooks_before = len(block._forward_hooks)
    baseline = score_choice_variant(tokenizer, model, VARIANT)
    direction = torch.ones(
        model.config.hidden_size, dtype=torch.float32, device=model.device
    )
    steered = score_choice_variant(tokenizer, model, VARIANT, direction=direction)
    restored = score_choice_variant(tokenizer, model, VARIANT)

    assert (steered["p_a"], steered["p_b"]) != (baseline["p_a"], baseline["p_b"]), (
        "nonzero intervention did not change A/B probabilities"
    )
    assert restored == baseline, "hook removal did not restore the baseline score"
    assert len(block._forward_hooks) == hooks_before, (
        "scoring left an intervention hook installed"
    )


def test_intervention_hook_is_removed_when_scoring_forward_fails(qwen) -> None:
    tokenizer, model = qwen
    block = model.model.layers[14]
    hooks_before = len(block._forward_hooks)
    wrong_size_direction = torch.ones(
        model.config.hidden_size + 1, dtype=torch.float32, device=model.device
    )

    with pytest.raises(RuntimeError):
        score_choice_variant(tokenizer, model, VARIANT, direction=wrong_size_direction)

    assert len(block._forward_hooks) == hooks_before, (
        "failed scoring left an intervention hook installed"
    )
