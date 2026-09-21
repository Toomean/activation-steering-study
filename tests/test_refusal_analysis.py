import pytest

from activation_steering_study.analysis.refusal import summarize_refusal


def _pair(baseline: str, steered: str) -> dict:
    return {
        "baseline_review": {"refusal": baseline},
        "steered_review": {"refusal": steered},
    }


def test_summary_counts_labels_and_preserves_pairs() -> None:
    summary = summarize_refusal(
        [
            _pair("none", "none"),
            _pair("none", "full"),
            _pair("none", "mixed"),
            _pair("full", "none"),
        ]
    )

    assert summary["baseline_counts"] == {"none": 3, "full": 1, "mixed": 0}, (
        "baseline labels should remain distinct"
    )
    assert summary["steered_counts"] == {"none": 2, "full": 1, "mixed": 1}, (
        "steered labels should remain distinct"
    )
    assert summary["transitions"] == {
        "none->none": 1,
        "none->full": 1,
        "none->mixed": 1,
        "full->none": 1,
    }, "paired transitions should use the same records"
    assert summary["paired_changes"] == [0, 1, 1, -1], (
        "signed changes should preserve each paired record"
    )
    assert summary["baseline_any_refusal_percent"] == 25.0, (
        "baseline any-refusal rate should count full and mixed"
    )
    assert summary["steered_any_refusal_percent"] == 50.0, (
        "steered any-refusal rate should count full and mixed"
    )
    assert summary["steered_minus_baseline_pp"] == 25.0, (
        "rate change should be steered minus baseline"
    )


def test_summary_rejects_empty_or_ambiguous_labels() -> None:
    with pytest.raises(ValueError, match="At least one reviewed result"):
        summarize_refusal([])
    with pytest.raises(ValueError, match="Expected resolved refusal labels"):
        summarize_refusal([_pair("unclear", "none")])
