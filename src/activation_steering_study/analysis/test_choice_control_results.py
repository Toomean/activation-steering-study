"""Check numeric-only historical diagnostics and completed synthetic control analysis."""

import hashlib
import json
from pathlib import Path

import pytest

from activation_steering_study.analysis import choice_control_results, choice_results
from activation_steering_study.steering.choice_control import ALPHAS, SEEDS, TARGETS, cell_evidence
from activation_steering_study.utils.json_io import save_json


@pytest.mark.parametrize("alpha,delta,bounds", [(1.0, 10.0, [5.0, 15.0]),
                                                (-1.0, -10.0, [-15.0, -5.0])])
def test_signed_contrast_keeps_seeded_pairing_and_transforms_bounds(monkeypatch, alpha, delta, bounds):
    seen = []

    def interval(values):
        seen.append(values)
        return 0.05, 0.15

    monkeypatch.setattr(choice_results, "bootstrap_mean_interval", interval)
    result = choice_control_results.signed_contrast({"b": 0.3, "a": 0.7}, {"a": 0.5, "b": 0.3}, alpha)
    assert seen[0] == pytest.approx([0.2, 0.0]), "Pair by group_id in canonical string order before resampling"
    assert result["signed_delta_pp"] == pytest.approx(delta), "The contrast must carry the dose sign"
    assert result["signed_interval_pp"] == bounds, "Negative alpha must reverse and negate interval endpoints"


def _run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    condition = lambda mean: {
        "mean_conditional_matching_probability": mean,
        "results": [{"source_index": 1, "group_id": "g", "orders": [
            {"order": "original", "conditional_matching_probability": mean, "ab_mass": 0.9},
            {"order": "swapped", "conditional_matching_probability": mean, "ab_mass": 0.8},
        ]}],
    }
    save_json(run / "baseline.json", condition(0.5))
    settings = {"baseline": {"target": None, "alpha": 0.0, "block": 14,
                             "control": "hook-free", "random_seed": None,
                             "direction_norm": 0.0, "injected_norm": 0.0}}
    cells = {}
    for target in TARGETS:
        for alpha in ALPHAS:
            means = {"real": 0.5 + alpha * 0.1, "random42": 0.5 + alpha * 0.02,
                     "random43": 0.5 - alpha * 0.03}
            for control, seed in (("real", None), ("random42", 42), ("random43", 43)):
                name = f"{target}_{control}_alpha_{alpha:+g}"
                save_json(run / f"{name}.json", condition(means[control]))
                settings[name] = {"target": target, "alpha": alpha, "block": 14, "control": control,
                                  "random_seed": seed, "direction_norm": 5.0, "injected_norm": abs(alpha) * 5}
            cells[f"{target}_alpha_{alpha:+g}"] = cell_evidence(
                0.5, means["real"], {seed: means[f"random{seed}"] for seed in SEEDS}, alpha,
            )
    save_json(run / "summary.json", {"behaviour": "honesty", "block": 14, "random_seeds": [42, 43],
                                     "development_source_ids": [1], "development_group_ids": ["g"],
                                     "condition_settings": settings, "cell_evidence": cells})
    manifest = tmp_path / "manifest.json"
    _bind(run, manifest)
    return run, manifest, tmp_path / "analysis.json"


def _bind(run, manifest):
    save_json(manifest, {"status": "completed", "returncode": 0, "exit_code": 0, "inputs_unchanged": True,
                         "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob("*.json")}})


def test_all_cells_keep_both_controls_and_undefined_intervals_diagnostic(monkeypatch, tmp_path):
    paths = _run(tmp_path)
    monkeypatch.setattr(choice_results, "bootstrap_mean_interval", lambda _values: None)
    result = choice_control_results.analyze(*paths)
    assert len(result["conditions"]) == 73 and len(result["cells"]) == 24, "Diagnose the complete fixed matrix"
    cell = result["cells"]["pooled_alpha_-1"]
    assert cell["evidence"]["passes"] is True and cell["sign"] == -1 and cell["injected_norm"] == 5.0, (
        "Diagnostic uncertainty must not silently change a passing cell"
    )
    assert set(cell["contrasts"]) == {"baseline", "random42", "random43"}, "Keep both fixed comparators"
    assert all(c["signed_interval_pp"] is None and c["status"] == "inconclusive" for c in cell["contrasts"].values()), (
        "Undefined BCa intervals must remain explicit inconclusive diagnostics"
    )
    assert cell["contrasts"]["random43"]["signed_delta_pp"] == pytest.approx(13.0), "Preserve the opposite seed43 effect"
    assert result["conditions"]["pooled_real_alpha_+1"]["diagnostics"]["ab_mass_mean"] == pytest.approx(0.85), (
        "Reuse the A+B mass diagnostic without making it an eligibility gate"
    )
    assert all("results" not in row for row in result["conditions"].values()), (
        "Aggregate analysis must not export raw prompt/source rows"
    )
    with pytest.raises(FileExistsError):
        choice_control_results.analyze(*paths)


@pytest.mark.parametrize("case", ["failed", "hash", "evidence", "incomplete"])
def test_altered_or_incomplete_outputs_refused(tmp_path, case):
    run, manifest, output = _run(tmp_path)
    if case == "failed":
        record = json.loads(manifest.read_bytes())
        record["status"] = "failed"
        save_json(manifest, record)
    elif case == "hash":
        (run / "baseline.json").write_text("{}")
    else:
        summary = json.loads((run / "summary.json").read_bytes())
        if case == "evidence":
            summary["cell_evidence"]["pooled_alpha_+1"]["passes"] = False
        else:
            summary["cell_evidence"].pop("pooled_alpha_+1")
        save_json(run / "summary.json", summary)
        _bind(run, manifest)
    with pytest.raises(ValueError):
        choice_control_results.analyze(run, manifest, output)
    assert not output.exists(), "Invalid evidence must not produce diagnostic output"
