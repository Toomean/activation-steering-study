"""Load the pinned Qwen checkpoint.

Adapted from Hugging Face's Transformers direct-load example:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
"""

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# Hugging Face commit SHA; pins model and tokenizer files to one revision.
# Source: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/commit/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def load_qwen():
    # PyCharm falsely reports these Hugging Face factories as returning None.
    # noinspection PyNoneFunctionAssignment
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    # noinspection PyNoneFunctionAssignment
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    return tokenizer, model
