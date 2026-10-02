"""Shared prompt generation and result saving for steering pilots.

Chat rendering follows the Qwen2.5-1.5B-Instruct direct-load example:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
"""

from pathlib import Path
from typing import TypedDict, cast

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase, Qwen2ForCausalLM

from activation_steering_study.data.refusal import SampleRefusalItem
from activation_steering_study.steering.generation import generate_with_intervention
from activation_steering_study.utils.json_io import save_json


class GenerationKwargs(TypedDict):
    max_new_tokens: int
    do_sample: bool


class GeneratedCompletion(TypedDict):
    text: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]


def generate_completion(
    model: Qwen2ForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    direction: torch.Tensor | None,
    layer_index: int,
    alpha: float,
    generation_kwargs: GenerationKwargs,
) -> GeneratedCompletion:
    """Render and generate one prompt, retaining the exact token boundary."""
    inputs = cast(
        BatchEncoding,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ),
    )
    prompt_length = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        if direction is None:
            # Mypy rejects Qwen's generate() because of a typing mismatch in Transformers 5.17.
            generated = model.generate(  # type: ignore[misc]
                **inputs, **generation_kwargs
            )
        else:
            generated = generate_with_intervention(
                model,
                inputs,
                direction,
                alpha,
                layer_index=layer_index,
                **generation_kwargs,
            )

    return {
        "text": cast(
            str,
            tokenizer.decode(
                generated[0, prompt_length:], skip_special_tokens=True
            ),
        ).strip(),
        "prompt_token_ids": inputs["input_ids"][0].tolist(),
        # Keep EOS when generate returned it so response scoring uses model output as-is.
        "generated_token_ids": generated[0, prompt_length:].tolist(),
    }


def generate_answers(
    model: Qwen2ForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[SampleRefusalItem],
    # None runs baseline generation without a hook; a tensor supplies an intervention direction.
    directions: dict[str, torch.Tensor | None],
    layer_index: int,
    alpha: float,
    generation_kwargs: GenerationKwargs,
) -> list[dict[str, str | int]]:
    """Generate answer fields named by the direction mapping, in mapping order."""
    results = [
        {key: record[key] for key in ("sample_path", "sample_index", "instruction")}
        for record in prompts
    ]
    for record in results:
        prompt = cast(str, record["instruction"])
        print(f"sample: {record['sample_path']}:{record['sample_index']}")
        print(f"prompt: {prompt}")
        for name, direction in directions.items():
            completion = generate_completion(
                model,
                tokenizer,
                prompt,
                direction,
                layer_index,
                alpha,
                generation_kwargs,
            )
            text = completion["text"]
            print(f"{name}: {text}")
            record[name] = text
    return results


def save_results(output: Path, metadata, results) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, {**metadata, "results": results})
    print(f"saved {output}")
