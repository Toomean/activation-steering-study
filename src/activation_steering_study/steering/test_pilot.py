"""Check the pilot's shared generation result against its legacy text row."""

from activation_steering_study.data.refusal import SampleRefusalItem
from activation_steering_study.steering.pilot import (
    GenerationKwargs,
    generate_answers,
    generate_completion,
)
from activation_steering_study.utils.qwen import load_qwen


def test_completion_ids_and_legacy_text_field() -> None:
    tokenizer, model = load_qwen()
    prompt = "Say hello in one word."
    generation_kwargs: GenerationKwargs = {"max_new_tokens": 2, "do_sample": False}
    completion = generate_completion(
        model, tokenizer, prompt, None, 14, 1.0, generation_kwargs
    )
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )["input_ids"][0].tolist()

    item: SampleRefusalItem = {
        "sample_path": "synthetic.json",
        "sample_index": 0,
        "instruction": prompt,
    }
    row = generate_answers(
        model, tokenizer, [item], {"baseline": None}, 14, 1.0, generation_kwargs
    )[0]

    assert completion["prompt_token_ids"] == prompt_ids
    assert len(completion["generated_token_ids"]) in (1, 2)
    decoded = tokenizer.decode(
        completion["generated_token_ids"], skip_special_tokens=True
    ).strip()
    assert completion["text"] == decoded
    assert row["baseline"] == completion["text"]
