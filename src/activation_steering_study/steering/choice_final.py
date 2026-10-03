"""Evaluate final rows using the frozen development direction and selected doses."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import torch
import transformers

from activation_steering_study.steering.choice_run import BLOCK, SEEDS, ChoiceRow, score_condition
from activation_steering_study.steering.random_control import sample_random_direction
from activation_steering_study.utils.json_io import save_json
from activation_steering_study.utils.qwen import MODEL_ID, MODEL_REVISION, load_qwen


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} differs from frozen development provenance")


def run(dev_dir: Path, selection_path: Path, output_dir: Path) -> dict[str, object]:
    """Validate frozen inputs before model loading, then save seven final conditions.

    Run from the repository root. The dev directory may be an archived copy;
    its extraction.pt is authoritative, rather than the original tensor_path.
    Existing output directories are refused; failed runs require a new directory.
    """
    if output_dir.exists():
        raise FileExistsError(output_dir)
    summary_path = dev_dir / "summary.json"
    summary_bytes = summary_path.read_bytes()
    selection_bytes = selection_path.read_bytes()
    dev = json.loads(summary_bytes)
    selection = json.loads(selection_bytes)
    selection_hash = hashlib.sha256(selection_bytes).hexdigest()
    behaviour = dev["behaviour"]
    frozen = selection["behaviours"][behaviour]
    summary_hash = hashlib.sha256(summary_bytes).hexdigest()
    _require_equal(summary_hash, frozen["summary_sha256"], "Dev summary hash")

    panel_path = Path(dev["panel_path"])
    panel_bytes = panel_path.read_bytes()
    tensor_path = dev_dir / "extraction.pt"
    input_hashes = {
        "panel_sha256": hashlib.sha256(panel_bytes).hexdigest(), "tensor_sha256": _sha256(tensor_path),
        "uv_lock_sha256": _sha256(Path("uv.lock")),
    }
    for field, digest in input_hashes.items():
        _require_equal(digest, frozen[field], field)
        if field in dev:
            _require_equal(digest, dev[field], f"Dev {field}")
    _require_equal(dev["block"], BLOCK, "Block")
    _require_equal(dev["block"], selection["block"], "Frozen block")
    _require_equal(dev["target_direction_norm"], selection["target_direction_norm"], "Target norm")
    _require_equal(dev["random_seeds"], list(SEEDS), "Random seeds")
    selected = frozen["selected_real_alphas"]
    _require_equal(selected, dev["selected_real_alphas"], "Selected real alphas")
    positive, negative = selected["positive"], selected["negative"]
    if not positive > 0 or not negative < 0:
        raise ValueError("Frozen doses must have positive and negative signs")
    schedule: list[tuple[str, str, int | None, float]] = [("baseline", "hook-free", None, 0.0)] + [
        (f"{control}_alpha_{alpha:+g}", control, seed, alpha)
        for alpha in (positive, negative)
        for control, seed in (("real", None), *((f"random{seed}", seed) for seed in SEEDS))
    ]
    _require_equal([name for name, *_ in schedule], frozen["planned_final_conditions"],
                   "Planned final conditions")
    for field, value in {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
    }.items():
        _require_equal(value, dev[field], field)
    code_hashes = {path: _sha256(Path(path)) for path in dev["code_sha256"]}
    _require_equal(code_hashes, dev["code_sha256"], "Dependency code hashes")
    final_module = Path("src/activation_steering_study/steering/choice_final.py")
    code_hashes[str(final_module)] = _sha256(final_module)

    final = [sample for sample in cast(list[ChoiceRow], json.loads(panel_bytes))
             if sample["split"] == "final"]
    if not final:
        raise ValueError("Panel needs a nonempty final split")
    used_sources = set(dev["extraction_source_ids"]) | set(dev["development_source_ids"])
    used_groups = set(dev["extraction_group_ids"]) | set(dev["development_group_ids"])
    if used_sources.intersection(sample["source_index"] for sample in final):
        raise ValueError("Final source_ids overlap extraction or development")
    if used_groups.intersection(sample["group_id"] for sample in final):
        raise ValueError("Final group_ids overlap extraction or development")

    # Collect static metadata before model work so incomplete inputs cannot produce partial scores.
    artifact: dict[str, object] = {
        "behaviour": behaviour, "source_revision": dev["source_revision"],
        "dev_summary_path": str(summary_path), "dev_summary_sha256": summary_hash,
        "selection_path": str(selection_path), "selection_sha256": selection_hash,
        "selection_frozen_utc": selection["frozen_utc"],
        "producer_implementation_commit": frozen["implementation_commit"],
        "producer_planning_commit": frozen["planning_commit"],
        "panel_path": str(panel_path), "tensor_path": str(tensor_path), **input_hashes,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_dtype": dev["model_dtype"], "model_device": dev["model_device"],
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "producer_code_sha256": dev["code_sha256"], "code_sha256": code_hashes,
        "block": BLOCK, "random_seeds": list(SEEDS),
        "raw_direction_norm": dev["raw_direction_norm"],
        "target_direction_norm": dev["target_direction_norm"],
        "direction_scale_factor": dev["direction_scale_factor"],
        "selected_real_alphas": selected, "planned_final_conditions": frozen["planned_final_conditions"],
        "answer_suffix": dev["answer_suffix"], "final_aggregation": dev["development_aggregation"],
        "final_source_ids": [sample["source_index"] for sample in final],
        "final_group_ids": [sample["group_id"] for sample in final],
    }
    # Preserve the dev multiplication order and scale; final scores never select doses.
    raw_direction = torch.load(tensor_path, map_location="cpu", weights_only=True)[BLOCK]["direction"]
    direction = raw_direction * dev["direction_scale_factor"]
    tokenizer, model = load_qwen()
    _require_equal(str(model.dtype), artifact["model_dtype"], "Loaded model dtype")
    _require_equal(str(model.device), artifact["model_device"], "Loaded model device")
    artifact.update(
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        attention_implementation=model.config._attn_implementation,
    )
    vectors = {
        "hook-free": None, "real": direction,
        **{f"random{seed}": sample_random_direction(
            model.config.hidden_size, dev["target_direction_norm"], seed,
        ) for seed in SEEDS},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    settings: dict[str, dict[str, object]] = {}
    summaries: dict[str, dict[str, float]] = {}
    baseline = 0.0
    for name, control, seed, alpha in schedule:
        vector = vectors[control]
        condition = score_condition(tokenizer, model, behaviour, final, vector, alpha)
        save_json(output_dir / f"{name}.json", condition)
        mean = condition["mean_conditional_matching_probability"]
        if name == "baseline":
            baseline = mean
        vector_norm = 0.0 if vector is None else torch.linalg.vector_norm(vector).item()
        settings[name] = {
            "control": control, "random_seed": seed, "block": BLOCK, "alpha": alpha,
            "direction_norm": vector_norm, "injected_norm": abs(alpha) * vector_norm,
        }
        summaries[name] = {"mean_conditional_matching_probability": mean,
                           "delta_vs_baseline": mean - baseline}
        print(f"{name}: {mean:.6f}")
    artifact["condition_settings"] = settings
    artifact["condition_summaries"] = summaries
    save_json(output_dir / "summary.json", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-dir", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.dev_dir, args.selection, args.output_dir)


if __name__ == "__main__":
    main()
