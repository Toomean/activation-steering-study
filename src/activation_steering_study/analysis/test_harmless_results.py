"""Invented grouped rows and MMLU outcomes pin the new paired analysis path."""

from copy import deepcopy
from typing import cast

import pytest

from activation_steering_study.analysis import harmless_results
from activation_steering_study.data.harmless import HarmlessRow
from activation_steering_study.steering.harmless_run import ConditionResult, HarmlessResult, MmluResult


def _manifest() -> list[HarmlessRow]:
    return [{"row_id": f"invented:{index}", "source_path": "harmless_val.json", "source_index": index,
             "instruction_sha256": "0" * 64, "role": "development", "quality_subset": index < 2,
             "semantic_group_id": "wifi" if index < 2 else "singleton", "semantic_status": "fixed_for_analysis"}
            for index in range(3)]


def _result(values: list[int]) -> ConditionResult:
    rows: list[HarmlessResult] = []
    for entry, value in zip(_manifest(), values, strict=True):
        rows.append({"row_id": entry["row_id"], "semantic_group_id": entry["semantic_group_id"],
                     "quality_subset": entry["quality_subset"], "refusal_proxy": bool(value),
                     "phrase_matches": [], "completion": "Invented response.",
                     "prompt_token_ids": [1], "generated_token_ids": [],
                     "completion_token_count": 0, "completion_token_cap": 256,
                     "response_likelihood": {"token_count": 0, "mean_nll": None, "perplexity": None}
                     if entry["quality_subset"] else None,
                     "prompt_kl": 0.0 if entry["quality_subset"] else None,
                     "generation_seconds": 0.0, "quality_seconds": 0.0})
    return {"schema_version": 1, "role": "development", "baseline": True, "alpha": 0.0,
            "layer_index": 14, "max_new_tokens": 256, "harmless": rows, "mmlu": [],
            "timing_seconds": {"total": 0.0}}


def _mmlu(subject: str, index: int, correct: bool) -> MmluResult:
    probabilities = [0.4, 0.1, 0.1, 0.1] if correct else [0.1, 0.4, 0.1, 0.1]
    return {"id": f"{subject}:{index}", "subject": subject, "source_index": index,
            "gold": "A", "prediction": "A" if correct else "B", "correct": correct,
            "probabilities_abcd": probabilities, "option_mass": 0.7, "scoring_seconds": 0.0}


def test_wifi_pair_equal_group_weight_and_permuted_alignment() -> None:
    baseline = _result([0, 1, 0])
    condition = _result([1, 1, 0])
    condition["harmless"].reverse()
    summary = harmless_results.summarize_condition(_manifest(), condition)
    assert [row["group_id"] for row in summary["harmless"]["group_means"]] == ["wifi", "singleton"]
    assert summary["harmless"]["group_means"][0]["row_count"] == 2
    comparison = harmless_results.compare_conditions(_manifest(), baseline, condition, expected_direction="increase")
    assert comparison["harmless"]["before"] == 0.25
    assert comparison["harmless"]["after"] == 0.5
    assert comparison["harmless"]["delta"] == 0.25, "Paired Wi-Fi rows count as one group, so delta is .25, not 1/3"
    # Resampling the two group deltas (.5, 0) gives means 0, .25, .5.
    assert comparison["harmless"]["nominal_interval"] == pytest.approx((0.0, 0.5)), (
        "The Wi-Fi interval must resample group deltas; row resampling has upper bound 1"
    )


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "group", "quality"])
def test_changed_row_contract_rejected(mutation: str) -> None:
    result = _result([0, 1, 0])
    if mutation == "duplicate":
        result["harmless"].append(deepcopy(result["harmless"][0]))
    elif mutation == "missing":
        result["harmless"].pop()
    elif mutation == "group":
        result["harmless"][0]["semantic_group_id"] = "different"
    else:
        result["harmless"][0]["quality_subset"] = False
    with pytest.raises(ValueError):
        harmless_results.summarize_condition(_manifest(), result)


def test_constant_changes_and_missing_quality_are_inconclusive() -> None:
    baseline = _result([0, 0, 0])
    condition = _result([1, 1, 1])
    comparison = harmless_results.compare_conditions(_manifest(), baseline, condition, expected_direction="increase")
    assert comparison["harmless"]["delta"] == 1.0
    assert comparison["harmless"]["nominal_interval"] is None
    assert comparison["harmless"]["interval_status"] == "inconclusive"
    assert comparison["harmless"]["interval_reason"] == "constant_nonzero_change"
    singleton_before, singleton_after = _result([0, 0, 0]), _result([1, 1, 1])
    singleton_before["harmless"] = singleton_before["harmless"][:1]
    singleton_after["harmless"] = singleton_after["harmless"][:1]
    singleton = harmless_results.compare_conditions(_manifest()[:1], singleton_before, singleton_after, expected_direction="increase")["harmless"]
    assert singleton["nominal_interval"] is None and singleton["interval_reason"] == "insufficient_units"
    assert singleton["delta"] == 1.0 and singleton["positive_group_count"] == 1
    condition["harmless"][1]["response_likelihood"] = {"token_count": 1, "mean_nll": 2.0, "perplexity": 7.389}
    condition["harmless"][1]["prompt_kl"] = 0.4
    quality = harmless_results.summarize_condition(_manifest(), condition)["harmless"]["quality"]
    assert quality == {"row_count": 2, "missing_nll_count": 1, "missing_kl_count": 0,
                       "mean_nll": 2.0, "mean_kl": 0.2}, "Missing NLLs must not become zeros"
    condition["harmless"][2]["prompt_kl"] = 0.1
    with pytest.raises(ValueError, match="outside"):
        harmless_results.summarize_condition(_manifest(), condition)


@pytest.mark.parametrize("field", ["role", "layer_index", "max_new_tokens", "schema_version"])
def test_comparisons_require_matching_evaluation_contract(field: str) -> None:
    baseline = _result([0, 1, 0])
    condition = deepcopy(baseline)
    cast(dict, condition)[field] = "final" if field == "role" else 999
    with pytest.raises(ValueError, match="contract"):
        harmless_results.compare_conditions(_manifest(), baseline, condition, expected_direction="increase")


def test_unequal_mmlu_subjects_keep_equal_weights_and_id_pairing() -> None:
    baseline, condition = _result([0, 1, 0]), _result([0, 1, 0])
    baseline["mmlu"] = [_mmlu("small", index, False) for index in range(3)] + [
        _mmlu("large", index, index == 3) for index in range(8)]
    condition["mmlu"] = [_mmlu("small", index, index != 1) for index in range(3)] + [
        _mmlu("large", index, index == 1) for index in range(8)]
    condition["mmlu"].reverse()
    comparison = harmless_results.compare_conditions(_manifest(), baseline, condition, expected_direction="increase")["mmlu"]
    assert comparison["delta"] == pytest.approx(1 / 3), "Equal-subject delta differs from pooled 2/11"
    assert comparison["gains"] == 3 and comparison["losses"] == 1
    assert comparison["before"] == pytest.approx(1 / 16)
    assert comparison["after"] == pytest.approx((2 / 3 + 1 / 8) / 2)
    assert comparison["option_mass_delta"] == pytest.approx(0.0)
    assert comparison["nominal_interval"] == pytest.approx((-1 / 16, 7 / 12)), (
        "The fixed unequal-subject interval must distinguish within-subject resampling from pooled rows"
    )
    assert comparison["interval_status"] == "nominal" and comparison["interval_reason"] is None
    baseline["mmlu"].reverse()
    condition["mmlu"].reverse()
    assert harmless_results.compare_conditions(_manifest(), baseline, condition, expected_direction="increase")["mmlu"] == comparison
    condition["mmlu"][0]["gold"] = "B"
    condition["mmlu"][0]["prediction"] = "B"
    condition["mmlu"][0]["correct"] = True
    with pytest.raises(ValueError, match="subject, gold"):
        harmless_results.compare_conditions(_manifest(), baseline, condition, expected_direction="increase")


def test_mmlu_singleton_subject_and_constant_strata_have_no_interval() -> None:
    baseline, condition = _result([0, 1, 0]), _result([0, 1, 0])
    baseline["mmlu"] = [_mmlu("small", 0, False)]
    condition["mmlu"] = [_mmlu("small", 0, True)]
    comparison = harmless_results.compare_conditions(_manifest(), baseline, condition, expected_direction="increase")["mmlu"]
    assert comparison["delta"] == 1.0 and comparison["nominal_interval"] is None
    assert comparison["interval_reason"] == "insufficient_within_subject_observations"
    assert comparison["gains"] == 1 and comparison["losses"] == 0
    baseline["mmlu"].append(_mmlu("small", 1, False))
    condition["mmlu"].append(_mmlu("small", 1, True))
    constant = harmless_results.compare_conditions(_manifest(), baseline, condition, expected_direction="increase")["mmlu"]
    assert constant["nominal_interval"] is None and constant["interval_reason"] == "constant_within_subjects"
    assert constant["delta"] == 1.0 and constant["gains"] == 2
    negative = harmless_results.compare_conditions(_manifest(), condition, baseline, expected_direction="increase")["mmlu"]
    assert negative["interval_reason"] == "constant_within_subjects" and negative["delta"] == -1.0
    assert negative["gains"] == 0 and negative["losses"] == 2
    unchanged = harmless_results.compare_conditions(_manifest(), baseline, baseline, expected_direction="increase")["mmlu"]
    assert unchanged["interval_reason"] == "no_observed_change" and unchanged["delta"] == 0.0


@pytest.mark.parametrize("before,after,delta,reason,counts", [
    ([0, 0, 0], [0, 0, 0], 0.0, "no_observed_change", (0, 0, 2)),
    ([0, 0, 0], [1, 1, 1], 1.0, "constant_nonzero_change", (2, 0, 0)),
    ([1, 1, 1], [0, 0, 0], -1.0, "constant_nonzero_change", (0, 2, 0)),
    ([0, 1, 0], [1, 1, 0], 0.25, None, (1, 0, 1)),
    ([1, 1, 0], [0, 1, 0], -0.25, None, (0, 1, 1)),
    ([0, 1, 1], [1, 1, 0], -0.25, None, (1, 1, 0)),
    ([0, 1, 0], [1, 0, 0], 0.0, "no_observed_change", (0, 0, 2)),
])
def test_group_change_diagnostics_preserve_observed_effects(before, after, delta, reason, counts) -> None:
    comparison = harmless_results.compare_conditions(_manifest(), _result(before), _result(after), expected_direction="increase")
    change = comparison["harmless"]
    assert change["delta"] == delta and change["interval_reason"] == reason
    assert (change["positive_group_count"], change["negative_group_count"], change["unchanged_group_count"]) == counts
    assert change["interval_status"] == ("nominal" if reason is None else "inconclusive")
    assert comparison["mmlu"]["delta"] is None and comparison["mmlu"]["interval_reason"] == "missing_endpoint"


def test_nonfinite_bootstrap_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    baseline, condition = _result([0, 1, 0]), _result([1, 1, 0])
    baseline["mmlu"] = [_mmlu("small", index, False) for index in range(2)]
    condition["mmlu"] = [_mmlu("small", index, index == 0) for index in range(2)]
    monkeypatch.setattr(harmless_results, "bootstrap_mean_interval", lambda _: None)
    monkeypatch.setattr(harmless_results, "bootstrap", lambda *_args, **_kwargs: SimpleNamespace(
        confidence_interval=SimpleNamespace(low=float("nan"), high=1.0)))
    comparison = harmless_results.compare_conditions(_manifest(), baseline, condition, expected_direction="increase")
    for change in (comparison["harmless"], comparison["mmlu"]):
        assert change["nominal_interval"] is None
        assert change["interval_status"] == "inconclusive" and change["interval_reason"] == "nonfinite_bootstrap"
    assert comparison["harmless"]["delta"] == 0.25 and comparison["mmlu"]["delta"] == 0.5


def _reporting_panel(differences: list[float]) -> tuple[list[HarmlessRow], ConditionResult, ConditionResult]:
    """Invent binary rows whose full group-mean deltas are the supplied small numbers."""
    pairs = {1.0: [(0, 1)], -1.0: [(1, 0)], 0.0: [(0, 0)],
             0.5: [(0, 1), (0, 0)], -0.5: [(1, 0), (0, 0)]}
    baseline, condition = _result([0, 0, 0]), _result([0, 0, 0])
    template = baseline["harmless"][0]
    baseline["harmless"], condition["harmless"] = [], []
    manifest: list[HarmlessRow] = []
    for group_index, difference in enumerate(differences):
        for left, right in pairs[difference]:
            index = len(manifest)
            row_id, group_id = f"invented:{index}", f"group:{group_index}"
            manifest.append({**_manifest()[0], "row_id": row_id, "source_index": index,
                             "quality_subset": False, "semantic_group_id": group_id})
            for result, value in ((baseline, left), (condition, right)):
                result["harmless"].append({**template, "row_id": row_id, "semantic_group_id": group_id,
                                          "quality_subset": False, "refusal_proxy": bool(value),
                                          "response_likelihood": None, "prompt_kl": None})
    return manifest, baseline, condition


def test_direction_is_required_and_runtime_validated() -> None:
    baseline, condition = _result([0, 1, 0]), _result([1, 1, 0])
    with pytest.raises(TypeError, match="expected_direction"):
        harmless_results.compare_conditions(_manifest(), baseline, condition)  # type: ignore[call-arg]
    for value in ("positive", "Increase", None):
        with pytest.raises(ValueError, match="Expected direction"):
            harmless_results.compare_conditions(_manifest(), baseline, condition,
                expected_direction=cast(harmless_results.HarmlessDirection, value))


@pytest.mark.parametrize("k,supported", [(19, False), (20, True)])
def test_sparse_threshold_and_partial_panel_use_real_bootstrap(k: int, supported: bool) -> None:
    manifest, baseline, condition = _reporting_panel([1.0] * k + [0.0])
    change = harmless_results.compare_conditions(manifest, baseline, condition, expected_direction="increase")["harmless"]
    interval = change["nominal_interval"]
    assert interval is not None and interval[0] > 0, "The sparse guard must block even a strictly positive nominal interval"
    support = change["support"]
    assert support["group_count"] == k + 1 and support["nonzero_group_count"] == k
    assert support["supported"] is supported
    assert support["method"] == ("nominal_bca" if supported else None), "An unchanged group prevents full-panel saturation"
    assert support["reason"] == ("strict_nominal_bca" if supported else "sparse_evidence")
    assert support["one_sided_probability_upper_bound"] is None and support["two_sided_probability_upper_bound"] is None


def test_rare_three_changes_of_245_are_sparse_even_when_interval_excludes_zero() -> None:
    manifest, baseline, condition = _reporting_panel([1.0] * 3 + [0.0] * 242)
    change = harmless_results.compare_conditions(manifest, baseline, condition, expected_direction="increase")["harmless"]
    assert change["delta"] == pytest.approx(3 / 245)
    assert change["nominal_interval"] == pytest.approx((1 / 245, 8 / 245))
    assert change["support"]["nonzero_group_count"] == 3
    assert not change["support"]["supported"] and change["support"]["reason"] == "sparse_evidence"


@pytest.mark.parametrize("expected_direction,bounds,supported", [
    ("increase", (0.0, 0.5), False), ("decrease", (-0.5, 0.0), False),
    ("increase", (-0.5, -0.1), False), ("decrease", (0.1, 0.5), False),
    ("increase", (0.1, 0.5), True), ("decrease", (-0.5, -0.1), True),
])
def test_nominal_interval_direction_and_strict_zero_boundary(
    monkeypatch: pytest.MonkeyPatch, expected_direction, bounds, supported: bool,
) -> None:
    # Nonconstant, nonsparse input reaches the existing interval routine.
    manifest, baseline, condition = _reporting_panel([1.0] * 19 + [-1.0])
    monkeypatch.setattr(harmless_results, "bootstrap_mean_interval", lambda _: bounds)
    change = harmless_results.compare_conditions(manifest, baseline, condition, expected_direction=expected_direction)["harmless"]
    assert change["nominal_interval"] == bounds and change["interval_status"] == "nominal", "Support decisions must retain the nominal interval"
    assert change["support"]["supported"] is supported, "The requested direction requires strict exclusion of zero"
    assert change["support"]["method"] == ("nominal_bca" if supported else None), "Only a qualifying interval supplies nominal BCa support"
    assert change["support"]["reason"] == ("strict_nominal_bca" if supported else "interval_not_strict_in_expected_direction"), "Zero and opposite-sign bounds must report the blocking reason"
    assert change["support"]["one_sided_probability_upper_bound"] is None, "Ordinary intervals must not acquire endpoint bounds"


@pytest.mark.parametrize("difference,expected_direction", [(1.0, "increase"), (-1.0, "decrease")])
@pytest.mark.parametrize("n,bounds", [
    (20, (0.00000095367431640625, 0.0000019073486328125)),
    (21, (0.000000476837158203125, 0.00000095367431640625)),
])
def test_full_panel_endpoint_bounds_leave_bca_unavailable(difference: float, expected_direction, n: int, bounds) -> None:
    # One extra independent group halves both endpoint-event probability bounds.
    manifest, baseline, condition = _reporting_panel([difference] * n)
    change = harmless_results.compare_conditions(manifest, baseline, condition, expected_direction=expected_direction)["harmless"]
    assert change["delta"] == difference and change["nominal_interval"] is None, "Saturation must retain its point delta and unavailable BCa interval"
    assert change["interval_status"] == "inconclusive" and change["interval_reason"] == "constant_nonzero_change", "Endpoint support must not relabel BCa availability"
    support = change["support"]
    assert support["supported"] and support["method"] == "full_panel_endpoint", "The complete expected endpoint panel uses a separate support route"
    assert support["group_count"] == support["nonzero_group_count"] == n, "Endpoint support uses the full panel size"
    assert support["one_sided_probability_upper_bound"] == bounds[0], "The one-sided endpoint bound must scale with n"
    assert support["two_sided_probability_upper_bound"] == bounds[1], "The two-sided endpoint bound must scale with n"
    opposite: harmless_results.HarmlessDirection = "decrease" if expected_direction == "increase" else "increase"
    blocked = harmless_results.compare_conditions(manifest, baseline, condition, expected_direction=opposite)["harmless"]["support"]
    assert not blocked["supported"] and blocked["method"] is None, "Opposite pre-specified directions cannot receive endpoint support"
    assert blocked["one_sided_probability_upper_bound"] is None and blocked["two_sided_probability_upper_bound"] is None, "Opposite directions must not acquire endpoint bounds"


def test_19_saturated_groups_are_still_sparse_and_fractional_constants_do_not_qualify() -> None:
    for difference, n in ((1.0, 19), (0.5, 20), (-0.5, 20)):
        expected: harmless_results.HarmlessDirection = "increase" if difference > 0 else "decrease"
        manifest, baseline, condition = _reporting_panel([difference] * n)
        change = harmless_results.compare_conditions(manifest, baseline, condition, expected_direction=expected)["harmless"]
        assert change["nominal_interval"] is None and change["interval_reason"] == "constant_nonzero_change", "Constant changes retain unavailable BCa intervals"
        support = change["support"]
        assert not support["supported"] and support["method"] is None, "Sparse endpoints and either fractional sign must block support"
        assert support["reason"] == ("sparse_evidence" if n == 19 else "unavailable_interval"), "The blocking reason must distinguish sparse from fractional constants"
        assert support["one_sided_probability_upper_bound"] is None, "Fractional constants must not acquire endpoint bounds"


def test_one_fractional_wifi_group_among_full_changes_uses_bca() -> None:
    manifest, baseline, condition = _reporting_panel([1.0] * 19 + [0.5])
    change = harmless_results.compare_conditions(manifest, baseline, condition, expected_direction="increase")["harmless"]
    assert change["delta"] == 0.975 and change["positive_group_count"] == 20
    assert change["nominal_interval"] is not None and change["nominal_interval"][0] > 0
    assert change["support"]["supported"] and change["support"]["method"] == "nominal_bca"
    assert change["support"]["one_sided_probability_upper_bound"] is None


def test_unchanged_and_cancelling_row_flips_cannot_supply_nonzero_groups() -> None:
    for before, after in (([0, 0, 0], [0, 0, 0]), ([0, 1, 0], [1, 0, 0])):
        change = harmless_results.compare_conditions(_manifest(), _result(before), _result(after), expected_direction="increase")["harmless"]
        assert change["delta"] == 0.0 and change["unchanged_group_count"] == 2
        assert change["interval_reason"] == "no_observed_change"
        assert change["support"]["nonzero_group_count"] == 0 and not change["support"]["supported"]


@pytest.mark.parametrize("bounds", [None, (float("nan"), 1.0), (0.1, float("inf"))])
def test_unavailable_or_nonfinite_bca_blocks_non_endpoint_support(monkeypatch: pytest.MonkeyPatch, bounds) -> None:
    manifest, baseline, condition = _reporting_panel([1.0] * 19 + [-1.0])
    monkeypatch.setattr(harmless_results, "bootstrap_mean_interval", lambda _: bounds)
    change = harmless_results.compare_conditions(manifest, baseline, condition, expected_direction="increase")["harmless"]
    assert change["nominal_interval"] is None and change["interval_reason"] == "nonfinite_bootstrap"
    assert not change["support"]["supported"] and change["support"]["method"] is None
    assert change["support"]["reason"] == "unavailable_interval"


def test_selected_changed_subset_cannot_pass_full_manifest_validation() -> None:
    manifest, baseline, condition = _reporting_panel([1.0] * 20 + [0.0])
    baseline["harmless"].pop()
    condition["harmless"].pop()
    with pytest.raises(ValueError, match="IDs must uniquely match"):
        harmless_results.compare_conditions(manifest, baseline, condition, expected_direction="increase")


@pytest.mark.parametrize("first,second", [(True, True), (True, False), (False, True), (False, False)])
def test_two_controls_require_both_actual_real_minus_random_contrasts(first: bool, second: bool) -> None:
    manifest, zero, real = _reporting_panel([1.0] * 20)
    random42, random43 = (deepcopy(zero) if first else deepcopy(real)), (deepcopy(zero) if second else deepcopy(real))
    comparison = harmless_results.compare_random_controls(manifest, real, random42, random43, expected_direction="increase")
    left, right = comparison["contrasts"]["random42"], comparison["contrasts"]["random43"]
    assert left["harmless"]["support"]["supported"] is first and right["harmless"]["support"]["supported"] is second
    assert left["harmless"]["support"]["nonzero_group_count"] == (20 if first else 0)
    assert right["harmless"]["support"]["nonzero_group_count"] == (20 if second else 0)
    assert comparison["supported"] is (first and second), "Either blocked actual contrast must block the conjunction"
    if not second:
        unrelated = harmless_results.compare_conditions(manifest, zero, random43, expected_direction="increase")
        assert unrelated["harmless"]["support"]["supported"], "Random-versus-baseline support cannot replace real-versus-random support"
    assert "support" not in left["mmlu"] and "support" not in right["mmlu"], "MMLU stays estimation-only"
