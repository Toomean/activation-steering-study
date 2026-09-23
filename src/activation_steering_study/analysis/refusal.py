"""Summarize reviewed baseline, DiM, and random-control responses."""

import json
import math
from collections import Counter
from pathlib import Path

from activation_steering_study.analysis.bootstrap import bootstrap_mean_interval
from activation_steering_study.evaluation.labels import (
    QUALITY_LABELS,
    REFUSAL_LABELS,
    UNFINISHED_LABELS,
)


def _count_labels(pairs: list[dict], field: str, labels: tuple[str, ...]) -> dict:
    counts = {}
    allowed = set(labels)
    for condition in ("baseline", "dim", "random"):
        observed = Counter(pair[condition][field] for pair in pairs)
        if unknown := set(observed) - allowed:
            raise ValueError(f"Unsupported {field} labels: {unknown!r}")
        counts[condition] = {label: observed[label] for label in labels}
    return counts


def summarize_refusal(results: list[dict], random_results: list[dict]) -> dict:
    """Return label counts and paired DiM comparisons for matching prompt records."""
    if not results:
        raise ValueError("At least one reviewed result is required")

    # The review vocabulary includes "unclear"; paired refusal rates require resolved labels.
    resolved_refusal_labels = tuple(label for label in REFUSAL_LABELS if label != "unclear")
    pilot_by_id = {
        (result["sample_path"], result["sample_index"]): result for result in results
    }
    random_by_id = {
        (result["sample_path"], result["sample_index"]): result for result in random_results
    }
    if len(pilot_by_id) != len(results) or len(random_by_id) != len(random_results):
        raise ValueError("Duplicate review record")
    if pilot_by_id.keys() != random_by_id.keys():
        raise ValueError("Pilot and random reviews have unmatched records")

    pairs = [
        {
            "baseline": result["baseline_review"],
            "dim": result["steered_review"],
            "random": random_by_id[record_id],
        }
        for record_id, result in pilot_by_id.items()
    ]
    refusal_counts = _count_labels(pairs, "refusal", resolved_refusal_labels)
    quality_counts = _count_labels(pairs, "quality", QUALITY_LABELS)
    unfinished_counts = _count_labels(pairs, "unfinished", UNFINISHED_LABELS)

    transitions: dict[str, int] = {}
    dim_minus_baseline: list[int] = []
    dim_minus_random: list[int] = []
    for pair in pairs:
        baseline = pair["baseline"]["refusal"]
        dim = pair["dim"]["refusal"]
        random = pair["random"]["refusal"]
        transition = f"{baseline}->{dim}"
        transitions[transition] = transitions.get(transition, 0) + 1
        # Full/mixed are 1 and none is 0: +1 means refusal appeared, -1 disappeared,
        # and 0 means refusal presence is unchanged.
        dim_refused = int(dim in ("full", "mixed"))
        dim_minus_baseline.append(dim_refused - int(baseline in ("full", "mixed")))
        dim_minus_random.append(dim_refused - int(random in ("full", "mixed")))

    pair_count = len(pairs)
    any_refusal = {
        condition: 100 * (counts["full"] + counts["mixed"]) / pair_count
        for condition, counts in refusal_counts.items()
    }
    return {
        "pair_count": pair_count,
        "refusal_counts": refusal_counts,
        "quality_counts": quality_counts,
        "unfinished_counts": unfinished_counts,
        "transitions": transitions,
        "dim_minus_baseline": dim_minus_baseline,
        "dim_minus_random": dim_minus_random,
        "any_refusal_percent": any_refusal,
    }


def main() -> None:
    pilot = json.loads(Path("artifacts/refusal-pilot-review.json").read_text(encoding="utf-8"))
    random = json.loads(Path("artifacts/refusal-random-review.json").read_text(encoding="utf-8"))
    for field in (
        "model_id",
        "model_revision",
        "model_dtype",
        "layer_index",
        "alpha",
        "generation_kwargs",
    ):
        if pilot[field] != random[field]:
            raise ValueError(f"Pilot and random-control metadata differ: {field}")
    if pilot["direction_norm"] != random["reference_direction_norm"]:
        raise ValueError("Pilot and random-control reference direction norms differ")
    # Float32 normalization and stored norm rounding can differ slightly.
    if not math.isclose(
        random["random_direction_norm"],
        random["reference_direction_norm"],
        rel_tol=1e-6,
    ):
        raise ValueError("Random direction norm differs from its reference norm")
    summary = summarize_refusal(pilot["results"], random["results"])

    print(f"Reviewed prompts: {summary['pair_count']}")
    print("Refusal counts")
    print("condition  none  full  mixed  any refusal")
    for condition in ("baseline", "dim", "random"):
        counts = summary["refusal_counts"][condition]
        print(
            f"{condition:9} {counts['none']:4} {counts['full']:4} {counts['mixed']:5} "
            f"{summary['any_refusal_percent'][condition]:10.1f}%"
        )
    for label, counts in (
        ("Quality", summary["quality_counts"]),
        ("Unfinished", summary["unfinished_counts"]),
    ):
        print(f"{label} counts")
        for condition in ("baseline", "dim", "random"):
            print(f"  {condition}: {counts[condition]}")
    print(f"Paired baseline-to-DiM transitions: {summary['transitions']}")
    for label, changes in (
        ("DiM minus baseline", summary["dim_minus_baseline"]),
        ("DiM minus random", summary["dim_minus_random"]),
    ):
        observed_change = 100 * sum(changes) / summary["pair_count"]
        print(f"{label} any-refusal change: {observed_change:+.1f} percentage points")
        interval = bootstrap_mean_interval(changes)
        if interval is None:
            print(f"{label} 95% BCa interval: unavailable because changes are constant")
        else:
            print(
                f"{label} 95% BCa interval: "
                f"{100 * interval[0]:+.2f} to {100 * interval[1]:+.2f} percentage points"
            )


if __name__ == "__main__":
    main()
