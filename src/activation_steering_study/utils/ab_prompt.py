"""Prepare tokenized A/B prompts shared by scoring and extraction."""

from typing import TypedDict, cast

from transformers import BatchEncoding, PreTrainedTokenizerBase


ANSWER_SUFFIX = "\nAnswer with A or B only."


class PreparedABPrompt(TypedDict):
    content: str
    prompt_text: str
    inputs: BatchEncoding
    prompt_token_ids: list[int]
    answer_token_ids: dict[str, int]


def prepare_ab_prompt(
    tokenizer: PreTrainedTokenizerBase, question: str
) -> PreparedABPrompt:
    """Render and tokenize an unanswered A/B prompt and its completed answers."""
    content = question + ANSWER_SUFFIX
    # Follow the Qwen2.5-1.5B-Instruct model card's chat format:
    # https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
    prompt_text = cast(
        str,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        ),
    )
    # The chat template already rendered special tokens, so avoid adding another set;
    # PyTorch tensors are required by the model. See
    # https://huggingface.co/docs/transformers/en/chat_templating
    inputs = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    # input_ids has shape [1, prompt_length]; select the single prompt for the
    # prefix comparison and recorded metadata below.
    prompt_token_ids = inputs["input_ids"][0].tolist()
    # These IDs represent prompt+'A' and prompt+'B', not generated model responses.
    completed = {
        label: tokenizer.encode(prompt_text + label, add_special_tokens=False)
        for label in ("A", "B")
    }
    # Next-token logits represent each answer only if its letter adds one token
    # without changing the tokenized prompt prefix.
    for label, ids in completed.items():
        if len(ids) != len(prompt_token_ids) + 1 or ids[:-1] != prompt_token_ids:
            raise ValueError(f"Appending {label} changed prompt tokenization")
    answer_token_ids = {label: ids[-1] for label, ids in completed.items()}
    return {
        "content": content,
        "prompt_text": prompt_text,
        "inputs": inputs,
        "prompt_token_ids": prompt_token_ids,
        "answer_token_ids": answer_token_ids,
    }
