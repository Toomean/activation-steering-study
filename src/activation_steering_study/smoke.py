"""Run a pinned Qwen chat smoke check.

Adapted from Hugging Face's Transformers direct-load example:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
"""

from activation_steering_study.utils.qwen import load_qwen


def main() -> None:
    tokenizer, model = load_qwen()

    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    generated = model.generate(**inputs, max_new_tokens=40)

    # Causal generation returns the prompt followed by its newly generated tokens.
    answer_ids = generated[0, inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    if not answer:
        raise RuntimeError("the smoke generation produced an empty answer")
    print(answer)


if __name__ == "__main__":
    main()
