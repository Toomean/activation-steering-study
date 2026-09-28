"""Check final-prompt logits and next-token answer probabilities."""

import pytest
import torch

from activation_steering_study.evaluation import scoring
from activation_steering_study.utils.choice_prompt import ANSWER_SUFFIX, prepare_choice_prompt
from activation_steering_study.utils.qwen import load_qwen


@pytest.fixture(scope="module")
def qwen():
    return load_qwen()


@pytest.fixture(scope="module")
def paris_london_prompt(qwen):
    tokenizer, _ = qwen
    return prepare_choice_prompt(
        tokenizer,
        "What is the capital of France?\nA. Paris\nB. London" + ANSWER_SUFFIX,
        "AB",
    )


def test_last_token_logits_matches_real_final_prompt_logits(qwen, paris_london_prompt) -> None:
    _, model = qwen
    with torch.no_grad():
        expected = model(**paris_london_prompt["inputs"]).logits[0, -1].float()
    actual = scoring.last_token_logits(model, paris_london_prompt["inputs"])

    assert actual.shape == (model.config.vocab_size,), "Returned logits have the wrong shape"
    assert actual.dtype == torch.float32, "Returned logits must be float32"
    assert not actual.requires_grad, "Scoring logits must not retain a gradient graph"
    assert torch.equal(actual, expected), "Scoring logits differ from the model's final prompt logits"


def test_score_answer_tokens_uses_full_vocabulary_and_caller_order(monkeypatch) -> None:
    # These logits produce full-vocabulary probabilities [0.1, 0.2, 0.3, 0.4].
    logits = torch.log(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def fixed_logits(_model, _inputs, **_kwargs):
        return logits

    monkeypatch.setattr(scoring, "last_token_logits", fixed_logits)
    actual = scoring.score_answer_tokens(None, None, [3, 0, 2])

    assert torch.allclose(actual, torch.tensor([0.4, 0.1, 0.3]), rtol=0, atol=1e-7), (
        "Selected probabilities did not preserve caller order or full-vocabulary scale"
    )
    assert actual.sum() < 1.0, "Selected probabilities were renormalized over the requested tokens"


def test_zero_alpha_preserves_real_choice_scores(qwen, paris_london_prompt) -> None:
    _, model = qwen
    inputs = paris_london_prompt["inputs"]
    answer_ids = [paris_london_prompt["answer_token_ids"][label] for label in "AB"]
    direction = torch.ones(model.config.hidden_size, dtype=torch.float32, device=model.device)
    baseline = scoring.score_answer_tokens(model, inputs, answer_ids)
    zero = scoring.score_answer_tokens(
        model, inputs, answer_ids, direction=direction, alpha=0.0, layer_index=14
    )

    assert torch.equal(zero, baseline), "alpha=0 changed real A/B probabilities"
    print(f"baseline: A=Paris {baseline[0].item():.4%}, B=London {baseline[1].item():.4%}")
    print(f"alpha=0: A=Paris {zero[0].item():.4%}, B=London {zero[1].item():.4%}")


def test_nonzero_direction_changes_scores_and_restores_baseline(
    qwen, paris_london_prompt
) -> None:
    _, model = qwen
    inputs = paris_london_prompt["inputs"]
    answer_ids = [paris_london_prompt["answer_token_ids"][label] for label in "AB"]
    block = model.model.layers[14]
    hooks_before = len(block._forward_hooks)
    direction = torch.ones(model.config.hidden_size, dtype=torch.float32, device=model.device)
    baseline = scoring.score_answer_tokens(model, inputs, answer_ids)
    steered = scoring.score_answer_tokens(
        model, inputs, answer_ids, direction=direction, alpha=1.0, layer_index=14
    )
    restored = scoring.score_answer_tokens(model, inputs, answer_ids)

    assert torch.isfinite(steered).all(), "Intervention produced nonfinite A/B probabilities"
    assert not torch.allclose(baseline, steered), "Nonzero direction did not change A/B probabilities"
    assert torch.equal(restored, baseline), "Removing the hook did not restore baseline scores"
    assert len(block._forward_hooks) == hooks_before, "Scoring left an intervention hook installed"
    for label, probabilities in (("baseline", baseline), ("alpha=1", steered), ("restored", restored)):
        print(f"{label}: A=Paris {probabilities[0].item():.4%}, B=London {probabilities[1].item():.4%}")


def test_forward_failure_removes_intervention_hook(qwen, paris_london_prompt) -> None:
    _, model = qwen
    block = model.model.layers[14]
    hooks_before = len(block._forward_hooks)
    answer_ids = [paris_london_prompt["answer_token_ids"][label] for label in "AB"]
    wrong_size_direction = torch.ones(
        model.config.hidden_size + 1, dtype=torch.float32, device=model.device
    )

    with pytest.raises(RuntimeError):
        scoring.score_answer_tokens(
            model, paris_london_prompt["inputs"], answer_ids,
            direction=wrong_size_direction, alpha=1.0, layer_index=14,
        )

    assert len(block._forward_hooks) == hooks_before, "Failed scoring left an intervention hook installed"
