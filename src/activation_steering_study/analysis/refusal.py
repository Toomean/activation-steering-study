"""Summarize reviewed baseline and steered refusal labels."""

import json
from pathlib import Path

from activation_steering_study.analysis.bootstrap import bootstrap_mean_interval
from activation_steering_study.evaluation.labels import REFUSAL_LABELS


def summarize_refusal(results: list[dict]) -> dict:
    """Return refusal counts, paired transitions, and percentage-point change."""
    if not results:
        raise ValueError("At least one reviewed result is required")

    resolved_labels = tuple(label for label in REFUSAL_LABELS if label != "unclear")
    baseline_counts = {label: 0 for label in resolved_labels}
    steered_counts = {label: 0 for label in resolved_labels}
    transitions: dict[str, int] = {}
    paired_changes: list[int] = []

    for result in results:
        baseline = result["baseline_review"]["refusal"]
        steered = result["steered_review"]["refusal"]
        if baseline not in resolved_labels or steered not in resolved_labels:
            raise ValueError(
                f"Expected resolved refusal labels: {resolved_labels}; "
                f"got {baseline!r}, {steered!r}"
            )

        baseline_counts[baseline] += 1
        steered_counts[steered] += 1
        # Full/mixed map to 1 and none to 0; steered minus baseline is per prompt.
        # +1 means a refusal appeared, -1 means it disappeared, and 0 means its presence is unchanged.
        paired_changes.append(
            int(steered in ("full", "mixed"))
            - int(baseline in ("full", "mixed"))
        )
        transition = f"{baseline}->{steered}"
        transitions[transition] = transitions.get(transition, 0) + 1

    pair_count = len(results)
    # Full and mixed count as any refusal; none is zero refusal.
    baseline_any = baseline_counts["full"] + baseline_counts["mixed"]
    steered_any = steered_counts["full"] + steered_counts["mixed"]
    baseline_percent = 100 * baseline_any / pair_count
    steered_percent = 100 * steered_any / pair_count

    return {
        "pair_count": pair_count,
        "baseline_counts": baseline_counts,
        "steered_counts": steered_counts,
        "transitions": transitions,
        "paired_changes": paired_changes,
        "baseline_any_refusal_percent": baseline_percent,
        "steered_any_refusal_percent": steered_percent,
        "steered_minus_baseline_pp": steered_percent - baseline_percent,
    }


def main() -> None:
    artifact = json.loads(
        Path("artifacts/refusal-pilot-review.json").read_text(encoding="utf-8")
    )
    summary = summarize_refusal(artifact["results"])

    print(f"Reviewed pairs: {summary['pair_count']}")
    print(f"Baseline refusal counts (none/full/mixed): {summary['baseline_counts']}")
    print(f"Steered refusal counts (none/full/mixed): {summary['steered_counts']}")
    print("Paired transitions:")
    for transition, count in summary["transitions"].items():
        print(f"  {transition}: {count}")
    interval = bootstrap_mean_interval(summary["paired_changes"])
    if interval is None:
        print("95% BCa interval: unavailable because observed changes are constant")
    else:
        print(
            "95% BCa interval for steered minus baseline: "
            f"{100 * interval[0]:+.1f} to {100 * interval[1]:+.1f} percentage points"
        )
    print(
        "Any refusal rate: "
        f"{summary['baseline_any_refusal_percent']:.1f}% -> "
        f"{summary['steered_any_refusal_percent']:.1f}% "
        f"(steered minus baseline: {summary['steered_minus_baseline_pp']:+.1f} percentage points)"
    )


if __name__ == "__main__":
    main()
