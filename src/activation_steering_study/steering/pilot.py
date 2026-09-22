"""Shared prompt generation and result saving for steering pilots.

Chat rendering follows the Qwen2.5-1.5B-Instruct direct-load example:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
"""

import json
from pathlib import Path
from typing import TypedDict, cast

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase, Qwen2ForCausalLM

from activation_steering_study.data.refusal import SampleRefusalItem
from activation_steering_study.steering.generation import generate_with_intervention


class GenerationKwargs(TypedDict):
    max_new_tokens: int
    do_sample: bool


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
        inputs = cast(
            BatchEncoding,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ),
        )
        prompt_length = inputs["input_ids"].shape[-1]
        print(f"sample: {record['sample_path']}:{record['sample_index']}")
        print(f"prompt: {prompt}")
        for name, direction in directions.items():
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
            text = cast(
                str,
                tokenizer.decode(
                    generated[0, prompt_length:], skip_special_tokens=True
                ),
            ).strip()
            print(f"{name}: {text}")
            record[name] = text
    return results


def save_results(output: Path, metadata, results) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {**metadata, "results": results},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {output}")
