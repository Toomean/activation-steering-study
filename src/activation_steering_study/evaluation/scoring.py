"""Score next-token answer probabilities."""

import torch


def score_answer_tokens(model, inputs, answer_token_ids):
    """Return first-sequence probabilities for single-token answer IDs in caller order.

    Probabilities are normalized over the full vocabulary before selection.
    """
    # The last prompt output predicts the next token:
    # https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2ForCausalLM.forward
    # Disable gradient tracking for this scoring forward:
    # https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]

    # Float32 avoids coarse BF16 probability rounding; normalise before selecting A/B.
    # https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html
    probabilities = logits.float().softmax(dim=-1)
    return probabilities[answer_token_ids]
