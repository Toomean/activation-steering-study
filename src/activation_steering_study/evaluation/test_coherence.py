"""Check the direction and response-only scope of likelihood diagnostics."""

import math

import pytest
import torch

from activation_steering_study.evaluation.coherence import (
    full_vocabulary_kl,
    prompt_distribution_kl,
    response_nll_from_logits,
    response_likelihood,
)
from activation_steering_study.evaluation.scoring import last_token_logits
from activation_steering_study.steering.pilot import generate_completion
from activation_steering_study.utils.qwen import load_qwen


PROMPT = "Answer with one short word: what color is a clear daytime sky?"


@pytest.fixture(scope="module")
def qwen():
    return load_qwen()


@pytest.fixture(scope="module")
def prompt_inputs(qwen):
    tokenizer, _ = qwen
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        return_tensors="pt",
    )


def test_kl_uses_baseline_as_the_first_distribution() -> None:
    baseline = torch.tensor([math.log(0.8), math.log(0.2)])
    steered = torch.tensor([math.log(0.3), math.log(0.7)])
    expected = 0.8 * math.log(0.8 / 0.3) + 0.2 * math.log(0.2 / 0.7)

    assert math.isclose(full_vocabulary_kl(baseline, steered), expected, rel_tol=1e-6)


def test_response_likelihood_scores_only_the_response_targets() -> None:
    # Position zero predicts prompt token one; position one predicts response one;
    # position two predicts response two. The prompt length is two.
    logits = torch.tensor(
        [[[7.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 7.0]]]
    )
    response_ids = torch.tensor([1, 0])
    first_nll = math.log1p(math.exp(-2.0))
    second_nll = math.log1p(math.exp(-3.0))
    expected_mean = (first_nll + second_nll) / 2

    result = response_nll_from_logits(logits, 2, response_ids)
    assert result["token_count"] == 2
    assert result["mean_nll"] is not None
    assert result["perplexity"] is not None
    assert math.isclose(result["mean_nll"], expected_mean, rel_tol=1e-6)
    assert math.isclose(result["perplexity"], math.exp(expected_mean), rel_tol=1e-6)


def test_bfloat16_response_logits_are_scored_in_float32() -> None:
    # Same logits and targets as the float32 case; 7, 0, 2 and 3 are exact in bfloat16.
    # The module upcasts to float32 before cross-entropy; bfloat16 cross-entropy misses
    # this expected mean by about 3e-3.
    logits = torch.tensor(
        [[[7.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 7.0]]], dtype=torch.bfloat16
    )
    response_ids = torch.tensor([1, 0])
    expected_mean = (math.log1p(math.exp(-2.0)) + math.log1p(math.exp(-3.0))) / 2

    result = response_nll_from_logits(logits, 2, response_ids)

    assert result["mean_nll"] is not None, "Two response targets produced no mean NLL"
    assert math.isclose(result["mean_nll"], expected_mean, rel_tol=1e-6), (
        f"bfloat16 logits gave mean NLL {result['mean_nll']}; expected {expected_mean}"
    )


def test_empty_response_returns_undefined_likelihood() -> None:
    result = response_nll_from_logits(torch.zeros((1, 2, 3)), 2, torch.tensor([]))
    assert result == {"token_count": 0, "mean_nll": None, "perplexity": None}


def test_empty_generation_returns_before_using_the_model() -> None:
    # object() has no device attribute and cannot be called, so any model use raises.
    result = response_likelihood(object(), [1, 2], [])

    assert result == {"token_count": 0, "mean_nll": None, "perplexity": None}, (
        f"Empty generated IDs returned {result}"
    )


def test_zero_alpha_prompt_kl_is_exactly_zero(qwen, prompt_inputs) -> None:
    _, model = qwen
    block = model.model.layers[14]
    hooks_before = len(block._forward_hooks)
    direction = torch.ones(model.config.hidden_size, dtype=torch.float32, device=model.device)

    kl = prompt_distribution_kl(model, prompt_inputs, direction, 0.0, layer_index=14)

    # alpha=0 leaves the block output unchanged, so both log-softmax tensors are identical.
    assert kl == 0.0, f"alpha=0 gave KL {kl}; expected exactly 0.0"
    assert len(block._forward_hooks) == hooks_before, "KL scoring left an intervention hook installed"


def test_prompt_kl_matches_baseline_first_prompt_pass_reference(qwen, prompt_inputs) -> None:
    _, model = qwen
    # A non-default layer makes the result depend on passing layer_index through.
    layer_index = 10
    block = model.model.layers[layer_index]
    hooks_before = len(block._forward_hooks)
    direction = torch.ones(model.config.hidden_size, dtype=torch.float32, device=model.device)

    kl = prompt_distribution_kl(model, prompt_inputs, direction, 1.0, layer_index=layer_index)
    # The reference is torch's KL with log-probability targets: kl_div(input, target) is
    # KL(target || input).
    baseline_log_probs = last_token_logits(model, prompt_inputs).log_softmax(dim=-1)
    steered_log_probs = last_token_logits(
        model, prompt_inputs, direction=direction, alpha=1.0, layer_index=layer_index
    ).log_softmax(dim=-1)
    expected = torch.nn.functional.kl_div(
        steered_log_probs, baseline_log_probs, log_target=True, reduction="sum"
    ).item()
    reverse = torch.nn.functional.kl_div(
        baseline_log_probs, steered_log_probs, log_target=True, reduction="sum"
    ).item()

    assert math.isclose(kl, expected, rel_tol=1e-6), (
        f"KL {kl} differs from KL(baseline || steered) {expected}"
    )
    assert kl > 0, f"alpha=1 gave KL {kl}; expected a positive distribution shift"
    # KL is asymmetric; a distinct reverse value makes swapped arguments fail the match above.
    assert not math.isclose(reverse, kl, rel_tol=1e-6), (
        f"KL(steered || baseline) {reverse} matches {kl}, so an argument swap would pass"
    )
    assert len(block._forward_hooks) == hooks_before, "KL scoring left an intervention hook installed"


def test_teacher_forced_likelihood_matches_prompt_pass_reference(qwen, prompt_inputs) -> None:
    tokenizer, model = qwen
    completion = generate_completion(
        model,
        tokenizer,
        PROMPT,
        None,
        14,
        1.0,
        {"max_new_tokens": 2, "do_sample": False},
    )
    ids = completion["generated_token_ids"]
    actual = response_likelihood(model, completion["prompt_token_ids"], ids)

    assert prompt_inputs["input_ids"][0].tolist() == completion["prompt_token_ids"], (
        "Chat-rendered prompt IDs differ from the generation prompt boundary"
    )
    assert len(ids) == 2, f"Expected two generated tokens from the pinned checkpoint, got {ids}"
    # Causal attention makes last-position prompt-pass logits depend only on that prefix,
    # so this reference does not share the module's full-sequence slice.
    extended_ids = torch.tensor([completion["prompt_token_ids"] + ids[:1]], device=model.device)
    extended_inputs = {
        "input_ids": extended_ids,
        "attention_mask": torch.ones_like(extended_ids),
    }
    first_nll = -last_token_logits(model, prompt_inputs).log_softmax(dim=-1)[ids[0]].item()
    second_nll = -last_token_logits(model, extended_inputs).log_softmax(dim=-1)[ids[1]].item()
    expected = (first_nll + second_nll) / 2

    assert actual["token_count"] == len(ids), (
        f"Scored {actual['token_count']} tokens for {len(ids)} generated IDs"
    )
    assert actual["mean_nll"] is not None, "Generated IDs produced no mean NLL"
    assert math.isclose(actual["mean_nll"], expected, rel_tol=1e-5), (
        f"Teacher-forced mean NLL {actual['mean_nll']} differs from prompt-pass {expected}"
    )
