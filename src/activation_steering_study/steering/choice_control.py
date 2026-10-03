"""Measure own-task evidence for block-14 controls at refusal reference scales."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import TypedDict, cast

import torch
import transformers

from activation_steering_study.extraction.paired import extract_choice_pairs
from activation_steering_study.steering.choice_run import (
    DOSES, SEEDS, ChoiceRow, prompt_variants, score_condition,
)
from activation_steering_study.steering.choice_sweep import SOURCE_REVISIONS, sweep_code_hashes
from activation_steering_study.steering.random_control import sample_random_direction
from activation_steering_study.utils.choice_prompt import ANSWER_SUFFIX
from activation_steering_study.utils.json_io import save_json
from activation_steering_study.utils.qwen import MODEL_ID, MODEL_REVISION, load_qwen


BLOCK = 14
# The frozen refusal matrix retains each target's own raw magnitude.
TARGETS = ("pooled", "cyber_intrusion", "dangerous_substances", "disinformation")
ALPHAS = tuple(-dose for dose in reversed(DOSES)) + DOSES
BASELINE_ATOL = 1e-12


class CellEvidence(TypedDict):
    signed_real_change: float
    signed_random_changes: dict[str, float]
    passes: bool


def cell_evidence(baseline: float, real: float, random_means: dict[int, float],
                  alpha: float) -> CellEvidence:
    """Test one signed dose in probability units; 0.05 means five percentage points."""
    if set(random_means) != set(SEEDS) or alpha not in ALPHAS:
        raise ValueError("Cell needs both frozen random seeds and a frozen signed dose")
    if not all(math.isfinite(value) for value in (baseline, real, *random_means.values())):
        raise ValueError("Nonfinite own-task probability")
    sign = 1 if alpha > 0 else -1
    change = sign * (real - baseline)
    random_changes = {str(seed): sign * (mean - baseline)
                      for seed, mean in random_means.items()}
    return {"signed_real_change": change, "signed_random_changes": random_changes,
            "passes": change >= 0.05 and all(change > shift for shift in random_changes.values())}


def run(behaviour: str, panel_path: Path, refusal_metadata_path: Path,
        output_dir: Path, *, baseline_path: Path) -> dict[str, object]:
    """Extract fresh block-14 pairs and save 73 dev conditions plus per-cell evidence.

    Final rows are inspected only for split isolation. Refusal metadata supplies
    numerical scales, not proof of current refusal extraction provenance. Existing
    output directories are refused; a failed run requires a new directory. The
    archived baseline must match dev IDs/groups and reproduce within absolute
    tolerance before extraction or any steered scoring.
    """
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if behaviour not in SOURCE_REVISIONS:
        raise ValueError("Unknown behaviour")
    panel_bytes = panel_path.read_bytes()
    reference_bytes = refusal_metadata_path.read_bytes()
    baseline_bytes = baseline_path.read_bytes()
    samples = cast(list[ChoiceRow], json.loads(panel_bytes))
    reference = json.loads(reference_bytes)
    if (reference["model_id"] != MODEL_ID or reference["model_revision"] != MODEL_REVISION
            or reference["layer_index"] != BLOCK):
        raise ValueError("Refusal scale reference differs from pinned model or block 14")
    if reference["model_dtype"] not in ("torch.float32", "torch.float16", "torch.bfloat16"):
        raise ValueError("Invalid reference model_dtype")
    norms = reference["direction_norms"]
    if not isinstance(norms, dict) or set(norms) != set(TARGETS) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value <= 0 for value in norms.values()
    ):
        raise ValueError("Reference needs four finite positive direction_norms")

    source_ids: set[int] = set()
    group_splits: dict[str | int, str] = {}
    for sample in samples:
        split, source_id, group_id = sample["split"], sample["source_index"], sample["group_id"]
        if split not in ("extraction", "development", "final"):
            raise ValueError("Unknown panel split")
        if source_id in source_ids:
            raise ValueError("Duplicate or overlapping source IDs")
        if group_id in group_splits and group_splits[group_id] != split:
            raise ValueError("Group IDs overlap panel splits")
        source_ids.add(source_id)
        group_splits[group_id] = split
    extraction = [sample for sample in samples if sample["split"] == "extraction"]
    development = [sample for sample in samples if sample["split"] == "development"]
    if not extraction or not development:
        raise ValueError("Panel needs nonempty extraction and development splits")

    archived_baseline = json.loads(baseline_bytes)
    if [(row["source_index"], row["group_id"]) for row in archived_baseline["results"]] != [
        (row["source_index"], row["group_id"]) for row in development
    ]:
        raise ValueError("Archived baseline source/group sequence differs from development")
    expected_baseline = cast(float, archived_baseline["mean_conditional_matching_probability"])
    if not math.isfinite(expected_baseline):
        raise ValueError("Nonfinite archived baseline mean")
    baseline_validation: dict[str, object] = {
        "path": str(baseline_path), "sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "expected_mean": expected_baseline, "absolute_tolerance": BASELINE_ATOL,
    }

    extraction_sources = [(sample["source_index"], prompt_variants(behaviour, sample))
                          for sample in extraction]
    for sample in development:
        prompt_variants(behaviour, sample)

    code_hashes = sweep_code_hashes()
    for module in ("choice_run", "choice_control"):
        path = Path(f"src/activation_steering_study/steering/{module}.py")
        code_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    artifact: dict[str, object] = {
        "behaviour": behaviour, "source_revision": SOURCE_REVISIONS[behaviour],
        "panel_path": str(panel_path), "panel_sha256": hashlib.sha256(panel_bytes).hexdigest(),
        "refusal_scale_reference": {
            "path": str(refusal_metadata_path), "sha256": hashlib.sha256(reference_bytes).hexdigest(),
            "direction_norms": norms, "declared_tensor_sha256": reference.get("tensor_sha256"),
            "declared_source_metadata_path": reference.get("source_metadata_path"),
            "declared_source_metadata_sha256": reference.get("source_metadata_sha256"),
            "scope": "numerical scales only; current refusal provenance validated separately",
        },
        "baseline_validation": baseline_validation,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "code_sha256": code_hashes,
        "block": BLOCK, "dose_magnitudes": list(DOSES), "random_seeds": list(SEEDS),
        "answer_suffix": ANSWER_SUFFIX, "hook_policy": "final prompt token",
        "extraction_method": "matching minus opposite; mean two orders per row, then rows",
        "development_aggregation": "mean two orders per row, rows per group, then groups equally",
        "extraction_source_ids": [sample["source_index"] for sample in extraction],
        "extraction_group_ids": [sample["group_id"] for sample in extraction],
        "development_source_ids": [sample["source_index"] for sample in development],
        "development_group_ids": [sample["group_id"] for sample in development],
        "qualification_rule": "sign(alpha)*(real-baseline)>=0.05 and strictly above both signed random changes",
        "qualification_scope": "separate decision per target and signed dose; no global candidate selection",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer, model = load_qwen()
    if str(model.dtype) != reference["model_dtype"]:
        raise ValueError("Loaded model dtype differs from refusal scale reference")
    artifact.update(
        model_dtype=str(model.dtype), model_device=str(model.device),
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        attention_implementation=model.config._attn_implementation,
    )
    # Persist execution provenance before the first scientific forward pass.
    save_json(output_dir / "execution.json", artifact)
    settings: dict[str, dict[str, object]] = {}
    summaries: dict[str, dict[str, float]] = {}

    def save_condition(name: str, vector: torch.Tensor | None, alpha: float,
                       target: str | None, control: str, seed: int | None = None) -> float:
        vector_norm = 0.0 if vector is None else torch.linalg.vector_norm(vector).item()
        if (not math.isfinite(vector_norm) or not math.isfinite(abs(alpha) * vector_norm)
                or (vector is not None and not torch.isfinite(vector).all().item())):
            raise ValueError("Nonfinite control direction")
        condition = score_condition(tokenizer, model, behaviour, development, vector, alpha,
                                    layer_index=BLOCK)
        mean = condition["mean_conditional_matching_probability"]
        if not math.isfinite(mean):
            raise ValueError("Nonfinite own-task mean")
        save_json(output_dir / f"{name}.json", condition)
        settings[name] = {
            "target": target, "control": control, "random_seed": seed,
            "block": BLOCK, "alpha": alpha, "direction_norm": vector_norm,
            "injected_norm": abs(alpha) * vector_norm,
        }
        summaries[name] = {"mean_conditional_matching_probability": mean}
        print(f"{name}: {mean:.6f}", flush=True)
        return mean

    baseline = save_condition("baseline", None, 0.0, None, "hook-free")
    delta = baseline - expected_baseline
    matches = abs(delta) <= BASELINE_ATOL
    baseline_validation.update(actual_mean=baseline, delta=delta, matches=matches)
    # Keep baseline scores and the comparison even when drift stops qualification.
    save_json(output_dir / "execution.json", artifact)
    if not matches:
        raise ValueError("Recomputed baseline differs from archive beyond absolute tolerance")
    extracted, extraction_rows = extract_choice_pairs(
        model, tokenizer,
        extraction_sources,
        [BLOCK],
    )
    raw = extracted[BLOCK]["direction"]
    raw_norm = torch.linalg.vector_norm(raw).item()
    if not torch.isfinite(raw).all().item() or not math.isfinite(raw_norm) or raw_norm <= 0:
        raise ValueError("Block 14 has an invalid raw direction")
    tensor_path = output_dir / "extraction.pt"
    # Scaling creates new vectors; the saved raw pair tensors are left unchanged.
    torch.save(extracted, tensor_path)
    artifact.update(raw_direction_norm=raw_norm, extraction_rows=extraction_rows,
                    tensor_path=str(tensor_path),
                    tensor_sha256=hashlib.sha256(tensor_path.read_bytes()).hexdigest())
    cells: dict[str, CellEvidence] = {}
    scales: dict[str, float] = {}
    for target in TARGETS:
        norm = norms[target]
        scales[target] = norm / raw_norm
        vectors = (
            ("real", raw * scales[target], None),
            *((f"random{seed}", sample_random_direction(model.config.hidden_size, norm, seed), seed)
              for seed in SEEDS),
        )
        for alpha in ALPHAS:
            means: dict[str, float] = {}
            for control, vector, seed in vectors:
                name = f"{target}_{control}_alpha_{alpha:+g}"
                mean = save_condition(name, vector, alpha, target, control, seed)
                summaries[name]["delta_vs_baseline"] = mean - baseline
                means[control] = mean
            cells[f"{target}_alpha_{alpha:+g}"] = cell_evidence(
                baseline, means["real"], {seed: means[f"random{seed}"] for seed in SEEDS}, alpha,
            )
    artifact.update(direction_scale_factors=scales, condition_settings=settings,
                    condition_summaries=summaries, cell_evidence=cells)
    save_json(output_dir / "summary.json", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behaviour", required=True, choices=("honesty", "sycophancy"))
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--refusal-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    args = parser.parse_args()
    run(args.behaviour, args.panel, args.refusal_metadata, args.output_dir, baseline_path=args.baseline)


if __name__ == "__main__":
    main()
