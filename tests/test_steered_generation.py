"""Check Paris-token probability during a fixed activation intervention."""

import torch

from activation_steering_study.steering.generation import generate_with_intervention
from activation_steering_study.utils.qwen import load_qwen


def test_steered_generation_changes_paris_probability_and_removes_hook() -> None:
    tokenizer, model = load_qwen()
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    direction = torch.ones(
        model.config.hidden_size,
        dtype=model.dtype,
        device=model.device,
    )
    expected_prefix_ids = tokenizer.encode("The capital of France is", add_special_tokens=False)
    paris_token_ids = tokenizer.encode(" Paris", add_special_tokens=False)
    assert len(paris_token_ids) == 1, f"Expected ' Paris' to be one token, got {paris_token_ids}"
    paris_token_id = paris_token_ids[0]
    # Use the prefix length plus one step for the Paris-token probability.
    num_new_tokens = len(expected_prefix_ids) + 1

    # Generation settings:
    # https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
    generation_kwargs = {
        # Prevent EOS from ending before the target prediction.
        "min_new_tokens": num_new_tokens,
        # Stop generation at the target prediction.
        "max_new_tokens": num_new_tokens,

        # Choose the most likely next token without random sampling (num_beams=1 here):
        # https://huggingface.co/docs/transformers/en/generation_strategies#greedy-search
        "do_sample": False,

        # Return structured output with token IDs and requested extra outputs; required for logits.
        "return_dict_in_generate": True,
        # Keep unprocessed vocabulary logits at each step for final-step softmax.
        "output_logits": True,
    }

    # Each independent generation starts with a fresh default cache.
    baseline = model.generate(**inputs, **generation_kwargs)
    zero = generate_with_intervention(model, inputs, direction, 0.0, **generation_kwargs)
    steered = generate_with_intervention(model, inputs, direction, 1.0, **generation_kwargs)
    restored = model.generate(**inputs, **generation_kwargs)

    generations = (baseline, zero, steered, restored)
    assert all(len(generation.logits) == num_new_tokens for generation in generations), (
        f"Expected {num_new_tokens} logits tensors from each generation"
    )
    prompt_length = inputs["input_ids"].shape[1]
    for generation in generations:
        assert generation.sequences[0, prompt_length:-1].tolist() == expected_prefix_ids, (
            "Expected the generated prefix before the final token to match the Paris context"
        )

    # Softmax converts final-step raw logits into probabilities:
    # https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html
    baseline_probability = baseline.logits[-1].softmax(dim=-1)[0, paris_token_id]
    zero_probability = zero.logits[-1].softmax(dim=-1)[0, paris_token_id]
    steered_probability = steered.logits[-1].softmax(dim=-1)[0, paris_token_id]
    restored_probability = restored.logits[-1].softmax(dim=-1)[0, paris_token_id]

    assert torch.isfinite(steered_probability), "Intervention produced a non-finite Paris probability"
    assert torch.equal(baseline_probability, zero_probability), "alpha=0 changed Paris probability"
    assert not torch.isclose(baseline_probability, steered_probability), (
        "alpha=1 did not change Paris probability"
    )
    assert torch.equal(baseline_probability, restored_probability), (
        "Hook removal did not restore Paris probability"
    )
    print(
        "Paris probability: "
        f"baseline={baseline_probability.item():.4%}, "
        f"alpha=0={zero_probability.item():.4%}, "
        f"alpha=1={steered_probability.item():.4%}, "
        f"restored={restored_probability.item():.4%}"
    )
