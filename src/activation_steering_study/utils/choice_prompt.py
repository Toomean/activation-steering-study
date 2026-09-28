"""Render and tokenize next-letter choice prompts."""

from typing import TypedDict, cast

from transformers import BatchEncoding, PreTrainedTokenizerBase


ANSWER_SUFFIX = "\nAnswer with A or B only."


class PreparedChoicePrompt(TypedDict):
    content: str
    prompt_text: str
    inputs: BatchEncoding
    prompt_token_ids: list[int]
    answer_token_ids: dict[str, int]


def prepare_choice_prompt(
    tokenizer: PreTrainedTokenizerBase, content: str, labels: str
) -> PreparedChoicePrompt:
    """Render an unanswered user chat and identify single-token answer letters."""
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
    # The chat template already rendered special tokens.
    # https://huggingface.co/docs/transformers/en/chat_templating
    inputs = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    prompt_token_ids = inputs["input_ids"][0].tolist()
    # Next-token logits score a label only when appending it preserves the prompt
    # token prefix and adds exactly one token.
    completed = {
        label: tokenizer.encode(prompt_text + label, add_special_tokens=False)
        for label in labels
    }
    for label, ids in completed.items():
        if len(ids) != len(prompt_token_ids) + 1 or ids[:-1] != prompt_token_ids:
            raise ValueError(f"Appending {label} changed prompt tokenization")
    return {
        "content": content,
        "prompt_text": prompt_text,
        "inputs": inputs,
        "prompt_token_ids": prompt_token_ids,
        "answer_token_ids": {label: ids[-1] for label, ids in completed.items()},
    }
