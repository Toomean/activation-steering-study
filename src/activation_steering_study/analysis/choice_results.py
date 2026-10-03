"""Summarize completed A/B runs without exporting prompts or loading a model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy

from activation_steering_study.analysis.bootstrap import bootstrap_mean_interval
from activation_steering_study.utils.json_io import save_json


def _load_verified_run(run_dir: Path, manifest_path: Path) -> tuple[dict, str, dict, dict]:
    """Read the seven selected conditions from a completed dev or final execution."""
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "completed" or manifest.get("returncode") != 0:
        raise ValueError("Manifest must describe a completed successful child")
    summary_bytes = (run_dir / "summary.json").read_bytes()
    summary_hash = hashlib.sha256(summary_bytes).hexdigest()
    if summary_hash != manifest["output_sha256"]["summary.json"]:
        raise ValueError("Summary hash differs from execution manifest")
    summary = json.loads(summary_bytes)
    split = "final" if "final_source_ids" in summary else "development"
    if split == "final" and (manifest.get("exit_code") != 0 or manifest.get("inputs_unchanged") is not True):
        raise ValueError("Final manifest must have exit_code 0 and unchanged inputs")
    selected = summary["selected_real_alphas"]
    positive, negative = selected["positive"], selected["negative"]
    if not positive > 0 or not negative < 0:
        raise ValueError("Selected doses must have positive and negative signs")
    names = ["baseline"] + [f"{control}_alpha_{alpha:+g}"
                            for alpha in (positive, negative)
                            for control in ("real", "random42", "random43")]
    conditions, condition_hashes = {}, {}
    for name in names:
        filename = f"{name}.json"
        data = (run_dir / filename).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != manifest["output_sha256"][filename]:
            raise ValueError(f"Condition hash differs from execution manifest: {name}")
        conditions[name] = json.loads(data)
        condition_hashes[name] = digest
    return summary, split, conditions, {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "summary_sha256": summary_hash, "condition_sha256": condition_hashes,
    }


def _paired_contrast(real: dict, reference: dict, alpha: float) -> dict:
    """Pair by group identity in a fixed order before seeded group resampling."""
    if not real or set(real) != set(reference):
        raise ValueError("Contrast needs matching nonempty group sets")
    differences = [real[group] - reference[group] for group in sorted(real, key=str)]
    interval = bootstrap_mean_interval(differences)
    status = "inconclusive" if interval is None else "not_established"
    if interval is not None and ((alpha > 0 and interval[0] > 0) or (alpha < 0 and interval[1] < 0)):
        status = "expected_direction"
    return {
        "group_count": len(differences), "delta_pp": float(np.mean(differences)) * 100,
        "interval_pp": None if interval is None else [bound * 100 for bound in interval],
        "status": status,
    }


def _condition_diagnostics(condition: dict, source_groups: dict,
                           baseline_variants: dict | None = None) -> tuple[dict, dict, dict]:
    """Check identities, then aggregate both orders and mass without excluding variants."""
    row_means: dict[str | int, list[float]] = {}
    variants, seen_sources = {}, set()
    for row in condition["results"]:
        source = row["source_index"]
        if source in seen_sources:
            raise ValueError("Duplicate condition source_index")
        seen_sources.add(source)
        if source not in source_groups or row["group_id"] != source_groups[source]:
            raise ValueError("Condition source/group identity differs from summary")
        seen_orders = set()
        for variant in row["orders"]:
            order = variant["order"]
            if order in seen_orders:
                raise ValueError("Duplicate condition answer order")
            seen_orders.add(order)
            variants[source, order] = (row["group_id"], variant["conditional_matching_probability"], variant["ab_mass"])
        if seen_orders != {"original", "swapped"}:
            raise ValueError("Condition needs both answer orders")
        row_mean = sum(variant["conditional_matching_probability"] for variant in row["orders"]) / len(row["orders"])
        row_means.setdefault(row["group_id"], []).append(row_mean)
    if seen_sources != set(source_groups):
        raise ValueError("Condition source sets differ from summary")
    # Preserve producer order-to-row-to-group arithmetic; float ties affect BCa bias correction.
    groups = {group: sum(values) / len(values) for group, values in row_means.items()}
    if baseline_variants is None:
        baseline_variants = variants
    if set(variants) != set(baseline_variants):
        raise ValueError("Diagnostic variant sets differ from baseline")
    order_means, order_deltas = {}, {}
    masses_by_group: dict[str | int, list[float]] = {}
    drops_by_group: dict[str | int, list[float]] = {}
    for order in ("original", "swapped"):
        probabilities: dict[str | int, list[float]] = {}
        changes: dict[str | int, list[float]] = {}
        for key, (group, probability, mass) in variants.items():
            if key[1] != order:
                continue
            baseline = baseline_variants[key]
            probabilities.setdefault(group, []).append(probability)
            changes.setdefault(group, []).append(probability - baseline[1])
            masses_by_group.setdefault(group, []).append(mass)
            drops_by_group.setdefault(group, []).append(baseline[2] - mass)
        order_means[order] = float(np.mean([np.mean(values) for values in probabilities.values()]))
        order_deltas[order] = float(np.mean([np.mean(values) for values in changes.values()])) * 100
    masses = [value[2] for value in variants.values()]
    drops = [baseline_variants[key][2] - value[2] for key, value in variants.items()]
    diagnostics = {
        "order_means": order_means, "order_delta_vs_baseline_pp": order_deltas,
        "swapped_minus_original_pp": (order_means["swapped"] - order_means["original"]) * 100,
        "ab_mass_mean": float(np.mean([np.mean(values) for values in masses_by_group.values()])),
        "ab_mass_minimum": min(masses), "low_mass_count": sum(mass < 0.80 for mass in masses),
        "mass_drop_mean": float(np.mean([np.mean(values) for values in drops_by_group.values()])),
        "mass_drop_over_0_10_count": sum(drop > 0.10 for drop in drops),
        "variant_count": len(variants),
    }
    return groups, variants, diagnostics


def analyze(run_dir: Path, manifest_path: Path, output: Path) -> dict:
    """Save unrounded aggregates; completed dev runs retain their dev-selected doses."""
    if output.exists():
        raise FileExistsError(output)
    summary, split, conditions, hashes = _load_verified_run(run_dir, manifest_path)
    source_ids, group_ids = summary[f"{split}_source_ids"], summary[f"{split}_group_ids"]
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("Summary needs unique nonempty source IDs")
    source_groups = dict(zip(source_ids, group_ids, strict=True))
    groups_by_condition, table = {}, {}
    baseline_variants = None
    baseline_mean = 0.0
    for name, condition in conditions.items():
        groups, variants, diagnostics = _condition_diagnostics(condition, source_groups, baseline_variants)
        groups_by_condition[name] = groups
        mean = float(np.mean(list(groups.values())))
        if name == "baseline":
            baseline_variants, baseline_mean = variants, mean
        table[name] = {
            "alpha": summary["condition_settings"][name]["alpha"],
            "random_seed": summary["condition_settings"][name]["random_seed"],
            "mean_conditional_matching_probability": mean, "delta_vs_baseline_pp": (mean - baseline_mean) * 100,
            "source_count": len(source_groups), "group_count": len(groups), "diagnostics": diagnostics,
        }
    contrasts, control_support = [], {}
    selected = summary["selected_real_alphas"]
    for sign in ("positive", "negative"):
        alpha = selected[sign]
        real_name = f"real_alpha_{alpha:+g}"
        secondary = []
        for control in ("baseline", "random42", "random43"):
            reference_name = control if control == "baseline" else f"{control}_alpha_{alpha:+g}"
            result = _paired_contrast(groups_by_condition[real_name], groups_by_condition[reference_name], alpha)
            result.update(real_condition=real_name, reference_condition=reference_name, alpha=alpha,
                          role="primary" if control == "baseline" else "secondary")
            if summary["behaviour"] == "sycophancy" and sign == "positive" and control == "baseline":
                result["positive_5pp_point_benchmark_met"] = result["delta_pp"] >= 5.0
            contrasts.append(result)
            if control != "baseline":
                secondary.append(result["status"])
        control_support[sign] = (
            "inconclusive" if "inconclusive" in secondary else
            "supported_against_both_fixed_controls" if all(status == "expected_direction" for status in secondary)
            else "not_established"
        )
    code_paths = [Path("src/activation_steering_study/analysis/choice_results.py"),
                  Path("src/activation_steering_study/analysis/bootstrap.py")]
    artifact = {
        "run_dir": str(run_dir), "manifest_path": str(manifest_path),
        "behaviour": summary["behaviour"], "split": split, "selected_real_alphas": selected, **hashes,
        "analysis_code_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths},
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "interval_method": "nominal 95% BCa; 9999 resamples; rng 42; groups paired by group_id",
        "conditions": table, "contrasts": contrasts, "fixed_control_support": control_support,
        "control_support_scope": "These two fixed directions only; no population, familywise, or equivalence inference",
    }
    save_json(output, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.manifest, args.output)
    for name, condition in result["conditions"].items():
        print(f"{name}: {condition['mean_conditional_matching_probability'] * 100:.2f}% "
              f"delta {condition['delta_vs_baseline_pp']:+.2f} pp")
    for contrast in result["contrasts"]:
        interval = contrast["interval_pp"]
        bounds = "inconclusive" if interval is None else f"nominal 95% BCa [{interval[0]:+.2f}, {interval[1]:+.2f}] pp"
        print(f"{contrast['real_condition']} - {contrast['reference_condition']}: "
              f"{contrast['delta_pp']:+.2f} pp; {bounds}; {contrast['status']}")


if __name__ == "__main__":
    main()
