"""Read final-prompt logits and score next-token answer probabilities."""

import torch

from activation_steering_study.steering.intervention import register_intervention


def last_token_logits(
    model, inputs, *, direction: torch.Tensor | None = None,
    alpha: float = 1.0, layer_index: int = 14,
) -> torch.Tensor:
    """Return float32 logits from the last prompt position, removing the owned hook.

    The last prompt output predicts the next token. Float32 supports stable
    full-vocabulary probabilities and KL from a BF16 checkpoint.
    https://huggingface.co/docs/transformers/en/model_doc/qwen2#transformers.Qwen2ForCausalLM.forward
    """
    remove_hook = (
        register_intervention(model.model.layers[layer_index], direction, alpha)
        if direction is not None else None
    )
    try:
        with torch.no_grad():
            return model(**inputs).logits[0, -1].float()
    finally:
        if remove_hook is not None:
            remove_hook()


def score_answer_tokens(
    model, inputs, answer_token_ids, *, direction: torch.Tensor | None = None,
    alpha: float = 1.0, layer_index: int = 14,
):
    """Return full-vocabulary probabilities for single-token answers in caller order."""
    logits = last_token_logits(
        model, inputs, direction=direction, alpha=alpha, layer_index=layer_index
    )
    return logits.softmax(dim=-1)[answer_token_ids]
