"""Check A/B next-token scoring with a fixed Qwen hook intervention.

Chat rendering is adapted from:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
"""

import torch

from activation_steering_study.evaluation.scoring import score_answer_tokens
from activation_steering_study.utils.qwen import load_qwen


def test_answer_token_scoring_changes_and_restores() -> None:
    # Intervention strength in activation + alpha * direction.
    # The hook reads the current value on each forward pass.
    alpha = 0.0
    tokenizer, model = load_qwen()
    messages = [
        {
            "role": "user",
            "content": "What is the capital of France?\nA. Paris\nB. London\nAnswer with A or B only.",
        },
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0].tolist()
    completed_a_ids = tokenizer.encode(f"{prompt_text}A", add_special_tokens=False)
    completed_b_ids = tokenizer.encode(f"{prompt_text}B", add_special_tokens=False)

    # We score only the next token, so each answer letter must add exactly one token
    # without changing the prompt tokens.
    assert completed_a_ids[:-1] == prompt_ids, "Appending A changed the prompt tokenization"
    assert completed_b_ids[:-1] == prompt_ids, "Appending B changed the prompt tokenization"
    answer_token_ids = [completed_a_ids[-1], completed_b_ids[-1]]

    direction = torch.ones(model.config.hidden_size, dtype=model.dtype, device=model.device)
    hook_calls = 0
    block = model.model.layers[0]

    # A forward hook can return a replacement output:
    # https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
    def add_direction(_module, _inputs, output):
        nonlocal hook_calls
        hook_calls += 1
        modified_output = output.clone()
        # Modify the final-token vector of the first sequence.
        modified_output[0, -1] += alpha * direction
        return modified_output

    baseline_probabilities = score_answer_tokens(model, inputs, answer_token_ids)

    handle = block.register_forward_hook(add_direction)
    try:
        zero_probabilities = score_answer_tokens(model, inputs, answer_token_ids)
        alpha = 1.0
        steered_probabilities = score_answer_tokens(model, inputs, answer_token_ids)
    finally:
        handle.remove()

    restored_probabilities = score_answer_tokens(model, inputs, answer_token_ids)

    probability_pairs = (
        baseline_probabilities,
        zero_probabilities,
        steered_probabilities,
        restored_probabilities,
    )
    assert hook_calls == 2, f"Expected two hook calls, got {hook_calls}"
    assert all(pair.shape == (2,) and torch.isfinite(pair).all() for pair in probability_pairs), (
        "Expected two finite A/B probabilities from each scoring call"
    )
    assert baseline_probabilities.sum() < 1, "A/B probabilities were renormalized instead of full-vocab"
    assert baseline_probabilities[0] > baseline_probabilities[1], (
        "Expected A=Paris to be preferred over B=London for this fixed checkpoint/example"
    )
    assert torch.equal(baseline_probabilities, zero_probabilities), "alpha=0 changed A/B probabilities"
    assert not torch.allclose(baseline_probabilities, steered_probabilities), (
        "alpha=1 had no effect on A/B probabilities"
    )
    assert torch.equal(baseline_probabilities, restored_probabilities), (
        "Hook removal did not restore A/B probabilities"
    )
    for label, probabilities in zip(("baseline", "alpha=0", "alpha=1", "restored"), probability_pairs):
        print(
            f"{label}: A=Paris {probabilities[0].item():.4%}, "
            f"B=London {probabilities[1].item():.4%}"
        )
