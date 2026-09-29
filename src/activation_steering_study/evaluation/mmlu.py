"""Score baseline, refusal DiM, and norm-matched random MMLU choices."""

import hashlib
import json
from pathlib import Path
from typing import NotRequired, TypedDict

import torch
from transformers import PreTrainedTokenizerBase, Qwen2ForCausalLM

from activation_steering_study.data.mmlu import LABELS, SOURCE_PATH, MmluItem, sample_mmlu
from activation_steering_study.data.refusal import load_reviewed_refusal
from activation_steering_study.evaluation.scoring import score_answer_tokens
from activation_steering_study.extraction.mean import mean_activation
from activation_steering_study.steering.pilot import save_results
from activation_steering_study.steering.random_control import sample_random_direction
from activation_steering_study.utils.choice_prompt import prepare_choice_prompt
from activation_steering_study.utils.qwen import MODEL_ID, MODEL_REVISION, load_qwen


INSTRUCTION = "Answer with one letter: A, B, C, or D."
REFUSAL_PATH = Path("artifacts/refusal-pilot.json")
RANDOM_PATH = Path("artifacts/refusal-random.json")
OUTPUT_PATH = Path("artifacts/refusal-mmlu.json")


class MmluScore(TypedDict):
    probabilities_abcd: list[float]
    prediction: str
    option_mass: float
    correct: NotRequired[bool]


class MmluResult(TypedDict):
    id: str
    subject: str
    source_index: int
    gold: str
    conditions: dict[str, MmluScore]


def render_question(item: MmluItem) -> str:
    """Keep the calibration screen's zero-shot wording and original option order."""
    return "\n".join(
        [
            item["question"],
            *(f"{label}. {choice}" for label, choice in zip(LABELS, item["choices"])),
            INSTRUCTION,
        ]
    )


def score_question(
    tokenizer: PreTrainedTokenizerBase,
    model: Qwen2ForCausalLM,
    content: str,
    layer_index: int,
    direction: torch.Tensor | None,
    alpha: float,
) -> MmluScore:
    """Score A-D from full-vocabulary logits, removing any hook on failure."""
    prepared = prepare_choice_prompt(tokenizer, content, LABELS)
    answer_ids = prepared["answer_token_ids"]
    probabilities = score_answer_tokens(
        model, prepared["inputs"], [answer_ids[label] for label in LABELS],
        direction=direction, alpha=alpha, layer_index=layer_index,
    )
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("Nonfinite MMLU choice probability")
    values = [float(value) for value in probabilities]
    prediction = LABELS[max(range(len(LABELS)), key=values.__getitem__)]
    return {
        "probabilities_abcd": values,
        "prediction": prediction,
        "option_mass": sum(values),
    }


def main() -> None:
    # Frozen refusal pilot settings: block-output 14, h + 1 * (harmful - harmless).
    layer_index = 14
    alpha = 1.0
    random_seed = 42
    refusal_bytes = REFUSAL_PATH.read_bytes()
    refusal = json.loads(refusal_bytes)
    random_reference = json.loads(RANDOM_PATH.read_text(encoding="utf-8"))
    assert refusal["model_id"] == MODEL_ID, "Refusal pilot model ID changed"
    assert refusal["model_revision"] == MODEL_REVISION, "Refusal pilot model revision changed"
    assert refusal["layer_index"] == layer_index, "Refusal pilot block differs from MMLU settings"
    assert refusal["alpha"] == alpha, "Refusal pilot alpha differs from MMLU settings"
    assert random_reference["source_artifact_sha256"] == hashlib.sha256(refusal_bytes).hexdigest(), (
        "Random control was not generated from this refusal pilot artifact"
    )
    assert random_reference["random_seed"] == random_seed, "Random control seed changed"
    assert random_reference["reference_direction_norm"] == refusal["direction_norm"], (
        "Random control reference norm differs from the refusal direction"
    )

    items = sample_mmlu()
    prompts = load_reviewed_refusal()
    tokenizer, model = load_qwen()
    assert str(model.dtype) == refusal["model_dtype"], "Loaded model dtype differs from refusal pilot"
    harmful = [record["instruction"] for record in prompts["train"]["harmful"]]
    harmless = [record["instruction"] for record in prompts["train"]["harmless"]]
    assert len(harmful) == refusal["train_counts"]["harmful"], (
        "Reviewed harmful training prompts differ from refusal pilot inputs"
    )
    assert len(harmless) == refusal["train_counts"]["harmless"], (
        "Reviewed harmless training prompts differ from refusal pilot inputs"
    )
    _, harmful_mean = mean_activation(model, tokenizer, harmful, layer_index)
    _, harmless_mean = mean_activation(model, tokenizer, harmless, layer_index)
    # Same difference-in-means construction as the reference refusal pilot.
    # https://doi.org/10.52202/079017-4322
    direction = harmful_mean - harmless_mean
    direction_norm = direction.norm().item()
    assert abs(direction_norm - refusal["direction_norm"]) <= 1e-4, (
        "Recreated refusal direction norm differs from the saved pilot direction"
    )
    random_direction = sample_random_direction(model.config.hidden_size, direction_norm, random_seed)

    conditions: dict[str, torch.Tensor | None] = {
        "baseline": None,
        "refusal_dim": direction,
        "random": random_direction,
    }
    results: list[MmluResult] = []
    for index, item in enumerate(items, start=1):
        content = render_question(item)
        gold = LABELS[item["answer"]]
        condition_scores: dict[str, MmluScore] = {}
        for name, applied_direction in conditions.items():
            score = score_question(
                tokenizer, model, content, layer_index, applied_direction, alpha
            )
            score["correct"] = score["prediction"] == gold
            condition_scores[name] = score
        results.append(
            {
                "id": item["id"],
                "subject": item["subject"],
                "source_index": item["source_index"],
                "gold": gold,
                "conditions": condition_scores,
            }
        )
        if index % 25 == 0:
            print(f"scored {index}/{len(items)}", flush=True)

    # Bind the run to copied source bytes as well as the deterministic sampled rows.
    save_results(
        OUTPUT_PATH,
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_dtype": str(model.dtype),
            "layer_index": layer_index,
            "alpha": alpha,
            "direction_norm": direction_norm,
            "random_seed": random_seed,
            "random_direction_norm": random_direction.norm().item(),
            "refusal_reference_path": str(REFUSAL_PATH),
            "refusal_reference_sha256": hashlib.sha256(refusal_bytes).hexdigest(),
            "mmlu_split": "validation",
            "sampling": "five rows per sorted subject; one random.Random(42); sorted indices",
            "source_parquet_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
            "sample_sha256": hashlib.sha256(
                json.dumps(items, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "prompt_instruction": INSTRUCTION,
            "scoring": "argmax among A-D; probabilities normalized over full vocabulary",
        },
        results,
    )
    for name in conditions:
        scores = [result["conditions"][name] for result in results]
        correct = sum(score["correct"] for score in scores)
        mass = sum(score["option_mass"] for score in scores) / len(scores)
        print(f"{name}: {correct}/{len(scores)} ({correct / len(scores):.3f}), mean A-D mass {mass:.3f}")
    baseline = [result["conditions"]["baseline"]["correct"] for result in results]
    for name in conditions:
        if name == "baseline":
            continue
        changed = [result["conditions"][name]["correct"] for result in results]
        losses = sum(before and not after for before, after in zip(baseline, changed))
        gains = sum(not before and after for before, after in zip(baseline, changed))
        print(
            f"{name} vs baseline: correct->wrong {losses}, wrong->correct {gains}, "
            f"delta {gains - losses:+d}/{len(results)}"
        )


if __name__ == "__main__":
    main()
