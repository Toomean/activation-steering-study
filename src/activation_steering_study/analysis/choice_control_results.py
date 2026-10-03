"""Verify and diagnose all block-14 control cells without selecting or loading a model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy

from activation_steering_study.analysis.choice_results import condition_diagnostics, paired_contrast
from activation_steering_study.steering.choice_control import ALPHAS, BLOCK, SEEDS, TARGETS, cell_evidence
from activation_steering_study.utils.json_io import save_json


def signed_contrast(real: dict, reference: dict, alpha: float) -> dict:
    """Represent the existing paired BCa contrast in the dose's expected direction."""
    contrast = paired_contrast(real, reference, alpha)
    sign = 1 if alpha > 0 else -1
    interval = contrast["interval_pp"]
    return {
        "group_count": contrast["group_count"], "signed_delta_pp": sign * contrast["delta_pp"],
        "signed_interval_pp": interval if sign > 0 or interval is None else [-interval[1], -interval[0]],
        "status": contrast["status"],
    }


def analyze(run_dir: Path, manifest_path: Path, output: Path) -> dict:
    """Bind completed output bytes, recompute eligibility, and save only aggregates."""
    if output.exists():
        raise FileExistsError(output)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("status") != "completed" or manifest.get("returncode") != 0
            or manifest.get("exit_code") != 0 or manifest.get("inputs_unchanged") is not True):
        raise ValueError("Manifest must describe a completed successful run with unchanged inputs")
    output_hashes = manifest["output_sha256"]
    for filename, digest in output_hashes.items():
        if hashlib.sha256((run_dir / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Output hash differs from manifest: {filename}")
    summary_bytes = (run_dir / "summary.json").read_bytes()
    summary_hash = hashlib.sha256(summary_bytes).hexdigest()
    if output_hashes["summary.json"] != summary_hash:
        raise ValueError("Summary hash differs from manifest")
    summary = json.loads(summary_bytes)
    if summary["block"] != BLOCK or summary["random_seeds"] != list(SEEDS):
        raise ValueError("Control analysis needs the frozen block and both random seeds")
    settings = summary["condition_settings"]
    names = ["baseline"] + [f"{target}_{control}_alpha_{alpha:+g}"
                            for target in TARGETS for alpha in ALPHAS
                            for control in ("real", "random42", "random43")]
    expected_cells = {f"{target}_alpha_{alpha:+g}" for target in TARGETS for alpha in ALPHAS}
    if set(settings) != set(names) or set(summary["cell_evidence"]) != expected_cells:
        raise ValueError("Control run needs exactly 73 conditions and 24 cell keys")
    source_groups = dict(zip(summary["development_source_ids"], summary["development_group_ids"], strict=True))
    table, groups_by_condition, baseline_variants = {}, {}, None
    for name in names:
        filename = f"{name}.json"
        data = (run_dir / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != output_hashes[filename]:
            raise ValueError(f"Condition hash differs from manifest: {name}")
        condition = json.loads(data)
        groups, variants, diagnostics = condition_diagnostics(condition, source_groups, baseline_variants)
        mean = sum(groups.values()) / len(groups)
        if abs(mean - condition["mean_conditional_matching_probability"]) > 1e-12:
            raise ValueError(f"Saved mean differs from group recomputation: {name}")
        if name == "baseline":
            baseline_variants = variants
        groups_by_condition[name] = groups
        table[name] = {**settings[name], "mean_conditional_matching_probability": mean,
                       "diagnostics": diagnostics}
    baseline = table["baseline"]["mean_conditional_matching_probability"]
    cells = {}
    for target in TARGETS:
        for alpha in ALPHAS:
            key = f"{target}_alpha_{alpha:+g}"
            real_name = f"{target}_real_alpha_{alpha:+g}"
            real = table[real_name]
            expected = cell_evidence(baseline, real["mean_conditional_matching_probability"], {
                seed: table[f"{target}_random{seed}_alpha_{alpha:+g}"]["mean_conditional_matching_probability"]
                for seed in SEEDS
            }, alpha)
            if summary["cell_evidence"][key] != expected:
                raise ValueError(f"Saved eligibility differs from recomputation: {key}")
            contrasts = {}
            for control in ("baseline", "random42", "random43"):
                reference_name = "baseline" if control == "baseline" else f"{target}_{control}_alpha_{alpha:+g}"
                contrasts[control] = signed_contrast(groups_by_condition[real_name],
                                                     groups_by_condition[reference_name], alpha)
            cells[key] = {"target": target, "alpha": alpha, "sign": 1 if alpha > 0 else -1,
                          "direction_norm": real["direction_norm"], "injected_norm": real["injected_norm"],
                          "evidence": expected, "contrasts": contrasts}
    code_paths = [Path("src/activation_steering_study") / path for path in (
        "analysis/choice_control_results.py", "analysis/choice_results.py", "analysis/bootstrap.py",
        "steering/choice_control.py", "utils/json_io.py",
    )]
    artifact = {
        "run_dir": str(run_dir), "manifest_path": str(manifest_path), "behaviour": summary["behaviour"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "summary_sha256": summary_hash,
        "condition_sha256": {name: output_hashes[f"{name}.json"] for name in names},
        "analysis_code_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths},
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "interval_method": "nominal 95% BCa; 9999 resamples; rng 42; paired group_id sorted by str",
        "diagnostic_scope": "both fixed controls; intervals and mass/order diagnostics do not change eligibility",
        "conditions": table, "cells": cells,
    }
    save_json(output, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.run_dir, args.manifest, args.output)


if __name__ == "__main__":
    main()
