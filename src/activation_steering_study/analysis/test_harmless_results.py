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
    comparison = harmless_results.compare_conditions(_manifest(), baseline, condition)
    assert comparison["harmless"]["before"] == 0.25
    assert comparison["harmless"]["after"] == 0.5
    assert comparison["harmless"]["delta"] == 0.25, "Paired Wi-Fi rows count as one group, so delta is .25, not 1/3"


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
    comparison = harmless_results.compare_conditions(_manifest(), baseline, condition)
    assert comparison["harmless"]["delta"] == 1.0
    assert comparison["harmless"]["nominal_interval"] is None
    assert comparison["harmless"]["interval_status"] == "inconclusive"
    singleton_before, singleton_after = _result([0, 0, 0]), _result([1, 1, 1])
    singleton_before["harmless"] = singleton_before["harmless"][:1]
    singleton_after["harmless"] = singleton_after["harmless"][:1]
    assert harmless_results.compare_conditions(_manifest()[:1], singleton_before, singleton_after)["harmless"]["nominal_interval"] is None
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
        harmless_results.compare_conditions(_manifest(), baseline, condition)


def test_unequal_mmlu_subjects_keep_equal_weights_and_id_pairing() -> None:
    baseline, condition = _result([0, 1, 0]), _result([0, 1, 0])
    baseline["mmlu"] = [_mmlu("small", index, False) for index in range(3)] + [
        _mmlu("large", index, index == 3) for index in range(8)]
    condition["mmlu"] = [_mmlu("small", index, index != 1) for index in range(3)] + [
        _mmlu("large", index, index == 1) for index in range(8)]
    condition["mmlu"].reverse()
    comparison = harmless_results.compare_conditions(_manifest(), baseline, condition)["mmlu"]
    assert comparison["delta"] == pytest.approx(1 / 3), "Equal-subject delta differs from pooled 2/11"
    assert comparison["gains"] == 3 and comparison["losses"] == 1
    assert comparison["before"] == pytest.approx(1 / 16)
    assert comparison["after"] == pytest.approx((2 / 3 + 1 / 8) / 2)
    assert comparison["option_mass_delta"] == pytest.approx(0.0)
    assert comparison["nominal_interval"] == pytest.approx((-1 / 16, 7 / 12)), (
        "The fixed unequal-subject interval must distinguish within-subject resampling from pooled rows"
    )
    assert comparison["interval_status"] == "nominal"
    baseline["mmlu"].reverse()
    condition["mmlu"].reverse()
    assert harmless_results.compare_conditions(_manifest(), baseline, condition)["mmlu"] == comparison
    condition["mmlu"][0]["gold"] = "B"
    condition["mmlu"][0]["prediction"] = "B"
    condition["mmlu"][0]["correct"] = True
    with pytest.raises(ValueError, match="subject, gold"):
        harmless_results.compare_conditions(_manifest(), baseline, condition)


def test_mmlu_singleton_subject_and_constant_strata_have_no_interval() -> None:
    baseline, condition = _result([0, 1, 0]), _result([0, 1, 0])
    baseline["mmlu"] = [_mmlu("small", 0, False)]
    condition["mmlu"] = [_mmlu("small", 0, True)]
    comparison = harmless_results.compare_conditions(_manifest(), baseline, condition)["mmlu"]
    assert comparison["delta"] == 1.0 and comparison["nominal_interval"] is None
    baseline["mmlu"].append(_mmlu("small", 1, False))
    condition["mmlu"].append(_mmlu("small", 1, True))
    assert harmless_results.compare_conditions(_manifest(), baseline, condition)["mmlu"]["nominal_interval"] is None
