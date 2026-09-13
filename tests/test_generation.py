"""Observe decoder-block hook outputs during Qwen generation.

Chat rendering is adapted from:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
"""

from activation_steering_study.utils.qwen import load_qwen


def test_forward_hook_sees_prompt_then_cached_tokens() -> None:
    tokenizer, model = load_qwen()
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    output_shapes = []

    # Hooks observe the decoder-block output on each forward call:
    # https://docs.pytorch.org/docs/2.14/notes/modules.html#module-hooks
    def capture_shape(_module, _inputs, output):
        output_shapes.append(tuple(output.shape))

    block = model.model.layers[0]
    handle = block.register_forward_hook(capture_shape)
    try:
        # Three new tokens ensure one prompt pass and two decode passes.
        model.generate(**inputs, min_new_tokens=3, max_new_tokens=3)
    finally:
        handle.remove()

    print(f"hook output shapes: {output_shapes}")
    assert len(output_shapes) == 3, f"Expected three hook calls, got {len(output_shapes)}"
    token_counts = [shape[1] for shape in output_shapes]
    # One full prompt pass is followed by one new token per decode forward:
    # https://huggingface.co/docs/transformers/en/cache_explanation
    assert token_counts == [inputs["input_ids"].shape[1], 1, 1], (
        f"Expected prompt then two one-token decode passes, got {token_counts}"
    )
