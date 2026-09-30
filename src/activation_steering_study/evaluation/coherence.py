"""Response-token likelihood and last-prompt distribution-shift diagnostics."""

import math
from typing import TypedDict

import torch
import torch.nn.functional as F
from transformers import BatchEncoding

from activation_steering_study.evaluation.scoring import last_token_logits


class ResponseLikelihood(TypedDict):
    token_count: int
    mean_nll: float | None
    perplexity: float | None


def response_nll_from_logits(
    logits: torch.Tensor,
    prompt_length: int,
    response_token_ids: torch.Tensor,
) -> ResponseLikelihood:
    """Score only response targets, using the prompt's final token to predict token one."""
    token_count = response_token_ids.numel()
    if token_count == 0:
        return {"token_count": 0, "mean_nll": None, "perplexity": None}
    prediction_logits = logits[0, prompt_length - 1 : prompt_length + token_count - 1].float()
    targets = response_token_ids.to(device=prediction_logits.device, dtype=torch.long)
    # Cross-entropy averages target-token NLL across the response.
    # https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html
    mean_nll = float(F.cross_entropy(prediction_logits, targets.reshape(-1)).item())
    return {
        "token_count": token_count,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
    }


def response_likelihood(
    model,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
) -> ResponseLikelihood:
    """Teacher-force returned prompt and generation IDs on the clean model.

    Returned EOS tokens are scored when present. Prompt padding is absent because
    only the unpadded ID lists from generation are accepted here.
    """
    if not generated_token_ids:
        return {"token_count": 0, "mean_nll": None, "perplexity": None}
    all_ids = torch.tensor(
        [prompt_token_ids + generated_token_ids],
        dtype=torch.long,
        device=model.device,
    )
    with torch.no_grad():
        logits = model(input_ids=all_ids).logits
    return response_nll_from_logits(
        logits,
        len(prompt_token_ids),
        all_ids[0, len(prompt_token_ids) :],
    )


def full_vocabulary_kl(
    baseline_logits: torch.Tensor, steered_logits: torch.Tensor
) -> float:
    """Return KL(baseline || steered) across the complete next-token vocabulary."""
    baseline_log_probs = baseline_logits.float().log_softmax(dim=-1)
    steered_log_probs = steered_logits.float().log_softmax(dim=-1)
    baseline_probs = baseline_log_probs.exp()
    return float((baseline_probs * (baseline_log_probs - steered_log_probs)).sum().item())


def prompt_distribution_kl(
    model,
    inputs: BatchEncoding,
    direction: torch.Tensor,
    alpha: float,
    *,
    layer_index: int = 14,
) -> float:
    """Compare baseline and steered next-token distributions at the last prompt token."""
    baseline_logits = last_token_logits(model, inputs)
    steered_logits = last_token_logits(
        model,
        inputs,
        direction=direction,
        alpha=alpha,
        layer_index=layer_index,
    )
    return full_vocabulary_kl(baseline_logits, steered_logits)
