"""Run block-18 development A/B conditions on one frozen honesty or sycophancy panel."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast

import torch
import transformers

from activation_steering_study.evaluation.choices import (
    ChoicePromptVariant, ScoredChoiceVariant, score_choice_variant,
)
from activation_steering_study.evaluation.sycophancy import SampleSycophancyItem, prepare_prompt_variants
from activation_steering_study.extraction.paired import extract_choice_pairs
from activation_steering_study.steering.choice_sweep import (
    SOURCE_REVISIONS, TARGET_NORM, SampleHonestyItem, honesty_variants, sweep_code_hashes,
)
from activation_steering_study.steering.random_control import sample_random_direction
from activation_steering_study.utils.choice_prompt import ANSWER_SUFFIX
from activation_steering_study.utils.json_io import save_json
from activation_steering_study.utils.qwen import MODEL_ID, MODEL_REVISION, load_qwen


BLOCK = 18
DOSES = (0.5, 1.0, 2.0)
SEEDS = (42, 43)


class ChoiceRow(TypedDict):
    source_index: int
    group_id: str | int
    split: str
    question: str


class ScoredRow(TypedDict):
    source_index: int
    group_id: str | int
    split: str
    row_mean: float
    orders: list[ScoredChoiceVariant]


class GroupMean(TypedDict):
    group_id: str | int
    mean: float
    row_count: int


class Condition(TypedDict):
    mean_conditional_matching_probability: float
    group_means: list[GroupMean]
    results: list[ScoredRow]


def prompt_variants(behaviour: str, sample: ChoiceRow) -> list[ChoicePromptVariant]:
    if behaviour == "honesty":
        return honesty_variants(cast(SampleHonestyItem, sample))
    return prepare_prompt_variants(cast(SampleSycophancyItem, sample))


def score_condition(tokenizer, model, behaviour: str, development: list[ChoiceRow],
                    direction: torch.Tensor | None, alpha: float,
                    *, layer_index: int = BLOCK) -> Condition:
    """Average orders within a row, rows within a group, and groups equally."""
    results: list[ScoredRow] = []
    by_group: dict[str | int, list[float]] = {}
    for sample in development:
        orders = [score_choice_variant(
            tokenizer, model, variant, direction=direction, alpha=alpha, layer_index=layer_index,
        ) for variant in prompt_variants(behaviour, sample)]
        row_mean = sum(order["conditional_matching_probability"] for order in orders) / len(orders)
        group_id = sample["group_id"]
        by_group.setdefault(group_id, []).append(row_mean)
        results.append({
            "source_index": sample["source_index"], "group_id": group_id,
            "split": sample["split"], "row_mean": row_mean,
            "orders": orders,
        })
    group_means: list[GroupMean] = [
        {"group_id": group_id, "mean": sum(values) / len(values), "row_count": len(values)}
        for group_id, values in by_group.items()
    ]
    return {
        "mean_conditional_matching_probability": sum(group["mean"] for group in group_means)
        / len(group_means),
        "group_means": group_means, "results": results,
    }


def select_signed_dose(baseline: float, means: dict[float, float], sign: int) -> float:
    """Maximize the signed dev change; break exact ties toward lower magnitude."""
    candidates = (alpha for alpha in means if (alpha > 0) == (sign > 0))
    return min(candidates, key=lambda alpha: (-sign * (means[alpha] - baseline), abs(alpha)))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _code_hashes() -> dict[str, str]:
    hashes = sweep_code_hashes()
    path = Path("src/activation_steering_study/steering/choice_run.py")
    hashes[str(path)] = _sha256(path.read_bytes())
    return hashes


def run(behaviour: str, panel_path: Path, output_dir: Path) -> dict[str, object]:
    """Save raw extraction and score the normalized 19-condition dev matrix.

    The panel file includes final rows for later use, but only extraction and
    development rows enter model calls or saved source-level results.
    """
    code_hashes = _code_hashes()
    panel_bytes = panel_path.read_bytes()
    samples = cast(list[ChoiceRow], json.loads(panel_bytes))
    extraction = [sample for sample in samples if sample["split"] == "extraction"]
    development = [sample for sample in samples if sample["split"] == "development"]
    if not extraction or not development:
        raise ValueError("Panel needs nonempty extraction and development splits")
    output_dir.mkdir(parents=True, exist_ok=False)

    tokenizer, model = load_qwen()
    extracted, extraction_rows = extract_choice_pairs(
        model, tokenizer,
        [(sample["source_index"], prompt_variants(behaviour, sample)) for sample in extraction],
        [BLOCK],
    )
    raw_direction = extracted[BLOCK]["direction"]
    raw_norm = torch.linalg.vector_norm(raw_direction).item()
    if not torch.isfinite(raw_direction).all().item() or raw_norm <= 0:
        raise ValueError("Block 18 has an invalid raw direction")
    # Use the refusal-pilot scale retained by the historical block-18 A/B sweep.
    direction_scale_factor = TARGET_NORM / raw_norm
    direction = raw_direction * direction_scale_factor
    random_directions = {
        seed: sample_random_direction(model.config.hidden_size, TARGET_NORM, seed)
        for seed in SEEDS
    }
    tensor_path = output_dir / "extraction.pt"
    torch.save(extracted, tensor_path)

    settings: dict[str, dict[str, object]] = {}
    summaries: dict[str, dict[str, float]] = {}
    real_means: dict[float, float] = {}

    def save_condition(name: str, vector: torch.Tensor | None, alpha: float,
                       control: str, seed: int | None = None) -> float:
        condition = score_condition(tokenizer, model, behaviour, development, vector, alpha)
        save_json(output_dir / f"{name}.json", condition)
        mean = condition["mean_conditional_matching_probability"]
        vector_norm = 0.0 if vector is None else torch.linalg.vector_norm(vector).item()
        settings[name] = {
            "control": control, "random_seed": seed, "block": BLOCK, "alpha": alpha,
            "direction_norm": vector_norm, "injected_norm": abs(alpha) * vector_norm,
        }
        summaries[name] = {"mean_conditional_matching_probability": mean}
        print(f"{name}: {mean:.6f}")
        return mean

    baseline = save_condition("baseline", None, 0.0, "hook-free")
    for control, vector, seed in (
        ("real", direction, None),
        *((f"random{seed}", random_directions[seed], seed) for seed in SEEDS),
    ):
        for alpha in tuple(-dose for dose in reversed(DOSES)) + DOSES:
            name = f"{control}_alpha_{alpha:+g}"
            mean = save_condition(name, vector, alpha, control, seed)
            summaries[name]["delta_vs_baseline"] = mean - baseline
            if control == "real":
                real_means[alpha] = mean

    selected = {
        "positive": select_signed_dose(baseline, real_means, +1),
        "negative": select_signed_dose(baseline, real_means, -1),
    }
    artifact: dict[str, object] = {
        "behaviour": behaviour, "panel_path": str(panel_path),
        "panel_sha256": _sha256(panel_bytes),
        "source_revision": SOURCE_REVISIONS[behaviour],
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_dtype": str(model.dtype), "model_device": str(model.device),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "code_sha256": code_hashes,
        "block": BLOCK, "dose_magnitudes": list(DOSES), "random_seeds": list(SEEDS),
        "raw_direction_norm": raw_norm,
        "target_direction_norm": TARGET_NORM,
        "direction_scale_factor": direction_scale_factor,
        "random_direction_norms": {
            str(seed): torch.linalg.vector_norm(vector).item()
            for seed, vector in random_directions.items()
        },
        "answer_suffix": ANSWER_SUFFIX,
        "extraction_method": "matching minus opposite; mean two orders per row, then rows",
        "development_aggregation": "mean two orders per row, rows per group, then groups equally",
        "extraction_source_ids": [sample["source_index"] for sample in extraction],
        "extraction_group_ids": [sample["group_id"] for sample in extraction],
        "extraction_rows": extraction_rows,
        "development_source_ids": [sample["source_index"] for sample in development],
        "development_group_ids": [sample["group_id"] for sample in development],
        "tensor_path": str(tensor_path), "tensor_sha256": _sha256(tensor_path.read_bytes()),
        "condition_settings": settings, "condition_summaries": summaries,
        "selected_real_alphas": selected,
        "selection_rule": "maximize sign(alpha) * (real dev mean - baseline); exact ties lower |alpha|",
        "final_rule": "baseline plus real/random42/random43 at each selected real alpha",
    }
    save_json(output_dir / "summary.json", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behaviour", required=True, choices=("honesty", "sycophancy"))
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.behaviour, args.panel, args.output_dir)


if __name__ == "__main__":
    main()
