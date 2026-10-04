"""One harmless induction condition, using an already loaded model and direction.

This computation kernel does not select doses, load models, or launch final runs.
The caller owns frozen selection, provenance capture and non-overwrite persistence.
"""

import hashlib
import math
from pathlib import Path
from time import perf_counter
from typing import TypedDict, cast

import torch
from transformers import BatchEncoding, PreTrainedTokenizerBase, Qwen2ForCausalLM

from activation_steering_study.data.harmless import HarmlessItem, HarmlessRole, SOURCE_NAMES, load_harmless
from activation_steering_study.data.mmlu import LABELS, MmluItem, load_mmlu_final, sample_mmlu
from activation_steering_study.evaluation.coherence import ResponseLikelihood, prompt_distribution_kl, response_likelihood
from activation_steering_study.evaluation.mmlu import render_question, score_question
from activation_steering_study.evaluation.refusal import REFUSAL_SUBSTRINGS, has_refusal_substring
from activation_steering_study.steering.pilot import generate_completion


class PhraseMatch(TypedDict):
    phrase: str
    start: int
    end: int


class HarmlessResult(TypedDict):
    row_id: str
    semantic_group_id: str
    quality_subset: bool
    refusal_proxy: bool
    phrase_matches: list[PhraseMatch]
    completion: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    completion_token_count: int
    completion_token_cap: int
    response_likelihood: ResponseLikelihood | None
    prompt_kl: float | None
    generation_seconds: float
    quality_seconds: float


class MmluResult(TypedDict):
    id: str
    subject: str
    source_index: int
    gold: str
    prediction: str
    probabilities_abcd: list[float]
    option_mass: float
    correct: bool
    scoring_seconds: float


class ConditionResult(TypedDict):
    schema_version: int
    role: HarmlessRole
    baseline: bool
    alpha: float
    layer_index: int
    max_new_tokens: int
    harmless: list[HarmlessResult]
    mmlu: list[MmluResult]
    timing_seconds: dict[str, float]


def load_condition_items(role: HarmlessRole) -> tuple[list[HarmlessItem], list[MmluItem]]:
    """Bind development to calibration MMLU and final to its frozen explicit panel."""
    harmless = load_harmless(role)
    mmlu = sample_mmlu() if role == "development" else load_mmlu_final()
    return harmless, mmlu


def refusal_phrase_matches(response: str) -> list[PhraseMatch]:
    """Retain every occurrence, in lowered-text position then frozen phrase order."""
    lowered = response.lower()
    matches: list[PhraseMatch] = []
    for phrase in REFUSAL_SUBSTRINGS:
        start = lowered.find(phrase.lower())
        while start != -1:
            matches.append({"phrase": phrase, "start": start, "end": start + len(phrase)})
            start = lowered.find(phrase.lower(), start + 1)
    return sorted(matches, key=lambda match: match["start"])


def evaluate_condition(
    model: Qwen2ForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    harmless_items: list[HarmlessItem],
    mmlu_items: list[MmluItem],
    direction: torch.Tensor | None = None,
    alpha: float = 0.0,
    *,
    layer_index: int = 14,
    max_new_tokens: int = 256,
) -> ConditionResult:
    """Evaluate typed harmless rows and MMLU; quality forwards use frozen members only.

    Baseline is hook-free with zero alpha. A supplied preexisting direction requires
    a finite positive alpha. Returned completions stay local; this function prints nothing.
    Inputs should come from load_condition_items; invented fixtures support synthetic checks.
    """
    if not math.isfinite(alpha) or (direction is None and alpha != 0) or (
        direction is not None and alpha <= 0
    ):
        raise ValueError("Baseline needs zero alpha; induction needs finite positive alpha")
    if direction is not None and (direction.ndim != 1 or not bool(torch.isfinite(direction).all())):
        raise ValueError("Direction must be a finite vector")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("Completion token cap must be a positive integer")
    if not harmless_items:
        raise ValueError("Harmless condition requires rows")
    role = harmless_items[0]["role"]
    if role not in SOURCE_NAMES:
        raise ValueError("Unknown harmless role")
    if len({item["row_id"] for item in harmless_items}) != len(harmless_items):
        raise ValueError("Duplicate harmless row IDs")
    if len({item["id"] for item in mmlu_items}) != len(mmlu_items):
        raise ValueError("Duplicate MMLU IDs")
    for item in harmless_items:
        if (item["role"] != role or Path(item["source_path"]).name != SOURCE_NAMES[role]
                or hashlib.sha256(item["instruction"].encode()).hexdigest() != item["instruction_sha256"]):
            raise ValueError("Harmless role, source or instruction hash mismatch")
    start = perf_counter()
    harmless: list[HarmlessResult] = []
    for item in harmless_items:
        generation_start = perf_counter()
        completion = generate_completion(model, tokenizer, item["instruction"], direction,
                                         layer_index, alpha,
                                         {"max_new_tokens": max_new_tokens, "do_sample": False})
        generation_seconds = perf_counter() - generation_start
        quality_start = perf_counter()
        likelihood = None
        kl = None
        if item["quality_subset"]:
            # Generation removes its intervention before clean teacher forcing.
            likelihood = response_likelihood(model, completion["prompt_token_ids"],
                                             completion["generated_token_ids"])
            kl = 0.0
            if direction is not None:
                inputs = cast(BatchEncoding, tokenizer.apply_chat_template(
                    [{"role": "user", "content": item["instruction"]}],
                    add_generation_prompt=True, return_tensors="pt",
                ))
                kl = prompt_distribution_kl(model, inputs, direction, alpha, layer_index=layer_index)
        harmless.append({
            "row_id": item["row_id"], "semantic_group_id": item["semantic_group_id"],
            "quality_subset": item["quality_subset"],
            "refusal_proxy": has_refusal_substring(completion["text"]),
            "phrase_matches": refusal_phrase_matches(completion["text"]),
            "completion": completion["text"],
            "prompt_token_ids": completion["prompt_token_ids"],
            "generated_token_ids": completion["generated_token_ids"],
            "completion_token_count": len(completion["generated_token_ids"]),
            "completion_token_cap": max_new_tokens,
            "response_likelihood": likelihood, "prompt_kl": kl,
            "generation_seconds": generation_seconds,
            "quality_seconds": perf_counter() - quality_start,
        })
    mmlu: list[MmluResult] = []
    for question in mmlu_items:
        scoring_start = perf_counter()
        score = score_question(tokenizer, model, render_question(question), layer_index, direction, alpha)
        gold = LABELS[question["answer"]]
        mmlu.append({
            "id": question["id"], "subject": question["subject"],
            "source_index": question["source_index"], "gold": gold,
            "prediction": score["prediction"], "probabilities_abcd": score["probabilities_abcd"],
            "option_mass": score["option_mass"], "correct": score["prediction"] == gold,
            "scoring_seconds": perf_counter() - scoring_start,
        })
    return {
        "schema_version": 1, "role": role, "baseline": direction is None,
        "alpha": alpha, "layer_index": layer_index, "max_new_tokens": max_new_tokens,
        "harmless": harmless, "mmlu": mmlu,
        "timing_seconds": {
            "generation": sum(row["generation_seconds"] for row in harmless),
            "quality": sum(row["quality_seconds"] for row in harmless),
            "mmlu": sum(row["scoring_seconds"] for row in mmlu), "total": perf_counter() - start,
        },
    }
