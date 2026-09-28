"""Build source-level A/B answer differences and means."""

from collections.abc import Sequence

import torch

from activation_steering_study.evaluation.choices import ChoicePromptVariant
from activation_steering_study.extraction.answer_activations import answer_activations_at_layers
from activation_steering_study.utils.choice_prompt import ANSWER_SUFFIX, prepare_choice_prompt


def extract_choice_pairs(
    model, tokenizer, sources: Sequence[tuple[int, list[ChoicePromptVariant]]],
    layer_indices: Sequence[int],
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, object]]]:
    """Build matching-minus-opposite answer differences for source questions.

    Each source is one original dataset question. For each supplied variant, subtract
    the opposite-answer activation from the activation for the answer selected by
    ``matching_label``. Average the two variant differences for each question, then
    average question means.

    For each requested layer, the returned tensors have these shapes:
    - ``pair_differences``: [N, 2, H], by question, supplied variant, activation component.
    - ``source_differences``: [N, H], the two-variant mean for each question.
    - ``direction``: [H], the mean of ``source_differences`` across questions.

    N is the number of source questions; H is the hidden size. Source and variant
    input order is preserved. Rows retain source ID, variant order, matching label,
    answer token IDs, and the zero-based position of the appended answer token.
    """
    pairs: dict[int, list[torch.Tensor]] = {layer: [] for layer in layer_indices}
    rows: list[dict[str, object]] = []
    for source_index, variants in sources:
        source_pairs: dict[int, list[torch.Tensor]] = {layer: [] for layer in layer_indices}
        for variant in variants:
            prepared = prepare_choice_prompt(
                tokenizer, variant["question"] + ANSWER_SUFFIX, "AB"
            )
            captured = answer_activations_at_layers(
                model, tokenizer, prepared["prompt_text"], layer_indices
            )
            matching = variant["matching_label"]
            opposite = "B" if matching == "A" else "A"
            for layer in layer_indices:
                source_pairs[layer].append(captured[layer][matching] - captured[layer][opposite])
            rows.append({
                "source_index": source_index, "order": variant["order"],
                "matching_label": matching,
                "answer_token_ids": prepared["answer_token_ids"],
                "answer_position": len(prepared["prompt_token_ids"]),
            })
        if len(variants) != 2:
            raise ValueError(f"Source {source_index} must have exactly two orders")
        for layer in layer_indices:
            pairs[layer].append(torch.stack(source_pairs[layer]))
    tensors = {}
    for layer in layer_indices:
        pair_differences = torch.stack(pairs[layer])
        source_differences = pair_differences.mean(dim=1)
        tensors[layer] = {
            "pair_differences": pair_differences,
            "source_differences": source_differences,
            "direction": source_differences.mean(dim=0),
        }
    return tensors, rows
