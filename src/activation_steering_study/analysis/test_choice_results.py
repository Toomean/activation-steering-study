"""Check paired group effects, variant diagnostics, and completed-run provenance."""

import hashlib
import json
from pathlib import Path

import pytest

from activation_steering_study.analysis import choice_results
from activation_steering_study.utils.json_io import save_json


def _condition(offset: float = 0.0) -> dict:
    return {
        "group_means": [{"group_id": "b", "mean": 0.8 + offset},
                        {"group_id": "a", "mean": 0.3 + offset}],
        "results": [
            {"source_index": source, "group_id": group, "orders": [
                {"order": "swapped", "conditional_matching_probability": probability + 0.1 + offset,
                 "ab_mass": mass - 0.1},
                {"order": "original", "conditional_matching_probability": probability - 0.1 + offset,
                 "ab_mass": mass},
            ]}
            for source, group, probability, mass in [(1, "a", 0.2, 0.9), (2, "a", 0.4, 0.7), (3, "b", 0.8, 1.0)]
        ],
    }


@pytest.fixture
def run_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    names = ["baseline", "real_alpha_+1", "random42_alpha_+1", "random43_alpha_+1",
             "real_alpha_-0.5", "random42_alpha_-0.5", "random43_alpha_-0.5"]
    settings = {name: {"alpha": 0.0 if name == "baseline" else 1.0 if "+1" in name else -0.5,
                       "random_seed": 42 if "random42" in name else 43 if "random43" in name else None}
                for name in names}
    save_json(run_dir / "summary.json", {
        "behaviour": "sycophancy", "selected_real_alphas": {"positive": 1.0, "negative": -0.5},
        "development_source_ids": [1, 2, 3], "development_group_ids": ["a", "a", "b"],
        "condition_settings": settings,
    })
    for name in names:
        offset = (0.06 if name == "real_alpha_+1" else -0.04 if name == "real_alpha_-0.5"
                  else 0.02 if name in ("random42_alpha_+1", "random43_alpha_+1") else 0.0)
        save_json(run_dir / f"{name}.json", _condition(offset))
    # Unselected dev data must not be needed for the seven-condition analysis.
    (run_dir / "real_alpha_+2.json").write_text("UNSELECTED_SENTINEL")
    manifest = tmp_path / "manifest.json"
    _bind_manifest(run_dir, manifest)
    return run_dir, manifest, tmp_path / "analysis.json"


def _bind_manifest(run_dir: Path, manifest: Path, **updates) -> None:
    save_json(manifest, {
        "status": "completed", "returncode": 0,
        "output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in run_dir.glob("*.json")}, **updates,
    })


def test_group_pairing_is_invariant_to_list_order(monkeypatch: pytest.MonkeyPatch) -> None:
    left, right = {"c": 0.9, "a": 0.6, "b": 0.7}, {"b": 0.5, "c": 0.6, "a": 0.5}
    result = choice_results.paired_contrast(left, right, 1.0)
    reordered = choice_results.paired_contrast(dict(reversed(list(left.items()))), right, 1.0)
    assert result == reordered, "Canonical group IDs must make seeded intervals independent of JSON ordering"
    assert result["delta_pp"] == pytest.approx(20.0), "Pair changes by identity: [0.1, 0.2, 0.3]"
    assert result["interval_pp"] == pytest.approx([10.0, 30.0]), "Positional pairing can preserve the mean but changes this CI"
    inputs: list[list[float]] = []

    def interval(values: list[float]) -> tuple[float, float]:
        inputs.append(values)
        return 0.0, 0.25

    monkeypatch.setattr(choice_results, "bootstrap_mean_interval", interval)
    choice_results.paired_contrast({2: 0.75, 10: 0.375, 3: 0.625}, {3: 0.5, 2: 0.5, 10: 0.5}, 1.0)
    assert inputs == [[-0.125, 0.25, 0.125]], "Integer IDs 10, 2, 3 must reach bootstrap in canonical string order"


@pytest.mark.parametrize("alpha,interval,status", [
    (1.0, (0.000000001, 0.03), "expected_direction"), (-0.5, (-0.03, -0.000000001), "expected_direction"),
    (1.0, (0.0, 0.03), "not_established"), (-0.5, (-0.03, 0.0), "not_established"),
])
def test_signed_status_uses_strict_unrounded_bounds(
    monkeypatch: pytest.MonkeyPatch, alpha: float, interval: tuple[float, float], status: str,
) -> None:
    monkeypatch.setattr(choice_results, "bootstrap_mean_interval", lambda _values: interval)
    real = {"a": 0.51, "b": 0.52} if alpha > 0 else {"a": 0.49, "b": 0.48}
    result = choice_results.paired_contrast(real, {"a": 0.5, "b": 0.5}, alpha)
    assert result["delta_pp"] == pytest.approx(1.5 if alpha > 0 else -1.5), "Retain raw signed paired effects"
    assert result["status"] == status, "A zero endpoint does not exclude zero in either direction"
    assert result["interval_pp"] == [value * 100 for value in interval], "JSON keeps unrounded percentage-point bounds"


def test_constant_group_changes_remain_inconclusive() -> None:
    with pytest.warns(RuntimeWarning):
        result = choice_results.paired_contrast({"a": 0.6, "b": 0.6}, {"a": 0.5, "b": 0.5}, 1.0)
    assert result["interval_pp"] is None and result["status"] == "inconclusive", "Do not replace undefined BCa bounds"
    assert result["delta_pp"] == pytest.approx(10.0), "Undefined intervals retain their point estimates"


def test_unequal_group_sizes_preserve_equal_weight_in_paired_effect() -> None:
    source_groups = {1: "a", 2: "a", 3: "b"}
    baseline, after = _condition(), _condition()
    for condition, probabilities in ((baseline, [0.0, 1.0, 0.0]), (after, [1.0, 1.0, 0.0])):
        for row, probability in zip(condition["results"], probabilities, strict=True):
            for variant in row["orders"]:
                variant["conditional_matching_probability"] = probability
    baseline_groups, baseline_variants, _ = choice_results.condition_diagnostics(baseline, source_groups)
    after_groups, _, diagnostics = choice_results.condition_diagnostics(after, source_groups, baseline_variants)
    assert baseline_groups == {"a": 0.5, "b": 0.0}, "Average the two group-a rows before weighting groups equally"
    assert after_groups == {"a": 1.0, "b": 0.0}, "The singleton group-b row must retain its own group mean"
    assert diagnostics["order_delta_vs_baseline_pp"] == {"original": 25.0, "swapped": 25.0}, (
        "Each order must average group changes [0.5, 0.0], giving 25pp rather than the row-weighted 100/3pp"
    )
    contrast = choice_results.paired_contrast(after_groups, baseline_groups, 1.0)
    assert contrast["group_count"] == 2, "The paired contrast must count groups rather than the three source rows"
    assert contrast["delta_pp"] == pytest.approx(25.0), (
        "Equal group weighting must give a probability change of 0.25 from group changes [0.5, 0.0]"
    )


def test_diagnostics_weight_groups_and_align_variant_pairs() -> None:
    source_groups = {1: "a", 2: "a", 3: "b"}
    _, baseline, baseline_diagnostics = choice_results.condition_diagnostics(_condition(), source_groups)
    assert baseline[1, "swapped"][2] == 0.80 and baseline_diagnostics["low_mass_count"] == 2, (
        "Mass exactly 0.80 must not join the two variants below the diagnostic threshold"
    )
    condition = _condition(0.05)
    condition["group_means"][0]["mean"] = 0.0
    condition["results"].reverse()
    for row in condition["results"]:
        row["orders"].reverse()
        for variant in row["orders"]:
            variant["ab_mass"] -= 0.15
    groups, _, result = choice_results.condition_diagnostics(condition, source_groups, baseline)
    assert sum(groups.values()) / 2 == pytest.approx(0.6), "Validated probabilities define equally weighted group means"
    assert result["order_means"] == {"original": pytest.approx(0.5), "swapped": pytest.approx(0.7)}, (
        "Per-order means must weight the two groups equally"
    )
    assert result["order_delta_vs_baseline_pp"] == {"original": pytest.approx(5.0), "swapped": pytest.approx(5.0)}, (
        "Pair each order with its own baseline before group aggregation"
    )
    assert result["swapped_minus_original_pp"] == pytest.approx(20.0), "Both-order diagnostics retain their difference"
    assert result["ab_mass_mean"] == pytest.approx(0.7), "Mass also gives each group equal weight"
    assert result["ab_mass_minimum"] == pytest.approx(0.45), "Retain the smallest variant mass"
    assert result["low_mass_count"] == 5 and result["variant_count"] == 6, "Count low mass without excluding variants"
    assert result["mass_drop_over_0_10_count"] == 6, "Align drops by source and order, not list position"
    asymmetric = _condition()
    asymmetric["results"][0]["orders"][0]["ab_mass"] -= 0.3
    asymmetric["results"].reverse()
    _, _, drop = choice_results.condition_diagnostics(asymmetric, source_groups, baseline)
    assert drop["mass_drop_mean"] == pytest.approx(0.3 / 4 / 2), (
        "One group-a variant drops 0.3 across four variants; equal weighting then averages two groups"
    )


@pytest.mark.parametrize("changed,prefix", [
    ("group", "Condition source/group identity"), ("missing_source", "Condition source sets"),
    ("duplicate_source", "Duplicate condition source_index"), ("duplicate_order", "Duplicate condition answer order"),
    ("missing_order", "Condition needs both answer orders"),
])
def test_identity_errors_are_explicit(changed: str, prefix: str) -> None:
    condition = _condition()
    if changed == "group":
        condition["results"][0]["group_id"] = "foreign"
    elif changed == "missing_source":
        condition["results"].pop()
    elif changed == "duplicate_source":
        condition["results"].append(condition["results"][0])
    elif changed == "duplicate_order":
        condition["results"][0]["orders"].append(condition["results"][0]["orders"][0])
    else:
        condition["results"][0]["orders"].pop()
    with pytest.raises(ValueError, match=f"^{prefix}"):
        choice_results.condition_diagnostics(condition, {1: "a", 2: "a", 3: "b"})


@pytest.mark.parametrize("split,behaviour,positive_support", [
    ("development", "sycophancy", "supported_against_both_fixed_controls"),
    ("final", "honesty", "supported_against_both_fixed_controls"),
    ("development", "sycophancy", "not_established"), ("final", "sycophancy", "inconclusive"),
])
def test_completed_run_aggregates_selected_conditions_and_control_support(
    run_files: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, split: str, behaviour: str,
    positive_support: str,
) -> None:
    run_dir, manifest, output = run_files
    summary = json.loads((run_dir / "summary.json").read_text())
    summary["behaviour"] = behaviour
    if split == "final":
        for field in ("source_ids", "group_ids"):
            summary[f"final_{field}"] = summary.pop(f"development_{field}")
    save_json(run_dir / "summary.json", summary)
    _bind_manifest(run_dir, manifest, exit_code=0, inputs_unchanged=True)
    calls: list[list[float]] = []

    def interval(values):
        calls.append(values)
        if len(calls) == 2 and positive_support == "inconclusive":
            return None
        if len(calls) == 3 and positive_support in ("not_established", "inconclusive"):
            return -0.01, 0.08
        return min(values) - 0.001, max(values) + 0.001

    monkeypatch.setattr(choice_results, "bootstrap_mean_interval", interval)
    result = choice_results.analyze(run_dir, manifest, output)
    assert result["run_dir"] == str(run_dir) and result["manifest_path"] == str(manifest), "Identify the analyzed run and manifest"
    assert len(result["conditions"]) == 7 and len(result["contrasts"]) == 6, "Analyze only selected dev doses and controls"
    assert result["conditions"]["baseline"]["mean_conditional_matching_probability"] == pytest.approx(0.55), "Equal group weighting"
    assert result["conditions"]["random42_alpha_-0.5"]["random_seed"] == 42, "Table carries saved control metadata"
    real = result["conditions"]["real_alpha_+1"]
    assert real["alpha"] == 1.0 and real["delta_vs_baseline_pp"] == pytest.approx(6.0), "Carry dose and baseline effect into the table"
    assert real["diagnostics"]["order_delta_vs_baseline_pp"] == {
        "original": pytest.approx(6.0), "swapped": pytest.approx(6.0),
    }, "Both order-specific real changes must propagate through analyze"
    assert result["conditions"]["random42_alpha_+1"]["diagnostics"]["order_delta_vs_baseline_pp"] == {
        "original": pytest.approx(2.0), "swapped": pytest.approx(2.0),
    }, "Random diagnostics must retain their own smaller effect"
    assert result["conditions"]["baseline"]["diagnostics"]["order_means"] == {
        "original": pytest.approx(0.45), "swapped": pytest.approx(0.65),
    }, "Baseline diagnostics must preserve unequal group sizes and answer-order differences"
    assert result["conditions"]["baseline"]["diagnostics"]["order_delta_vs_baseline_pp"] == {
        "original": 0.0, "swapped": 0.0,
    }, "Baseline order changes must remain zero"
    assert [(contrast["real_condition"], contrast["reference_condition"], contrast["role"])
            for contrast in result["contrasts"]] == [
        ("real_alpha_+1", "baseline", "primary"),
        ("real_alpha_+1", "random42_alpha_+1", "secondary"),
        ("real_alpha_+1", "random43_alpha_+1", "secondary"),
        ("real_alpha_-0.5", "baseline", "primary"),
        ("real_alpha_-0.5", "random42_alpha_-0.5", "secondary"),
        ("real_alpha_-0.5", "random43_alpha_-0.5", "secondary"),
    ], "Keep baseline contrasts primary and each named random contrast secondary at the same dose"
    assert [contrast["delta_pp"] for contrast in result["contrasts"]] == pytest.approx([6.0, 4.0, 4.0, -4.0, -4.0, -4.0]), (
        "Baseline and fixed controls define distinct paired effects"
    )
    assert result["fixed_control_support"] == {"positive": positive_support,
                                               "negative": "supported_against_both_fixed_controls"}, "Both fixed controls must support each sign"
    if positive_support == "inconclusive":
        assert [contrast["status"] for contrast in result["contrasts"][1:3]] == ["inconclusive", "not_established"], (
            "An undefined control interval dominates a mixed conjunction without erasing either control status"
        )
    flags = {(contrast["real_condition"], contrast["reference_condition"]): contrast["positive_5pp_point_benchmark_met"]
             for contrast in result["contrasts"] if "positive_5pp_point_benchmark_met" in contrast}
    assert flags == ({("real_alpha_+1", "baseline"): True} if behaviour == "sycophancy" else {}), (
        "The +5pp point flag belongs only to positive sycophancy real-baseline, not its +4pp random contrasts"
    )
    assert "results" not in result and all("results" not in row for row in result["conditions"].values()), (
        "Aggregate output does not copy source records"
    )
    with pytest.raises(FileExistsError):
        choice_results.analyze(run_dir, manifest, output)


@pytest.mark.parametrize("changed,prefix", [
    ("failed", "Manifest must describe"), ("returncode", "Manifest must describe"),
    ("hash", "Condition hash differs"), ("summary", "Summary hash differs"),
    ("final_flags", "Final manifest must have"),
])
def test_invalid_run_rejected_before_output(
    run_files: tuple[Path, Path, Path], changed: str, prefix: str,
) -> None:
    run_dir, manifest, output = run_files
    record = json.loads(manifest.read_text())
    if changed == "failed":
        record["status"] = "failed"
    elif changed == "returncode":
        record["returncode"] = 7
    elif changed in ("hash", "summary"):
        path = run_dir / ("baseline.json" if changed == "hash" else "summary.json")
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        summary = json.loads((run_dir / "summary.json").read_text())
        summary["final_source_ids"] = summary["development_source_ids"]
        summary["final_group_ids"] = summary["development_group_ids"]
        save_json(run_dir / "summary.json", summary)
        record["output_sha256"]["summary.json"] = hashlib.sha256((run_dir / "summary.json").read_bytes()).hexdigest()
    save_json(manifest, record)
    with pytest.raises(ValueError, match=f"^{prefix}"):
        choice_results.analyze(run_dir, manifest, output)
    assert not output.exists(), "Failed, partial, or altered runs must not create analysis output"
