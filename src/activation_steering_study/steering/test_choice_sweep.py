"""Check frozen A/B source labels and layer selection."""

import csv
import json
from collections import Counter
from typing import cast

from activation_steering_study.steering.choice_sweep import (
    HONESTY_PROMPTS, load_sweep_sources, honesty_variants, select_layer,
)


EXPECTED_HONESTY_SPLIT_COUNTS = {"extraction": 23, "development": 16}


def test_honesty_selection_matches_source() -> None:
    _, source_path, samples = load_sweep_sources("honesty")
    assert Counter(sample["split"] for sample in samples) == EXPECTED_HONESTY_SPLIT_COUNTS, (
        "Frozen honesty split counts changed"
    )
    source_ids = {sample["source_index"] for sample in samples}
    assert len(source_ids) == len(samples), "Honesty selection contains duplicate source IDs"
    assert 474 not in source_ids, "Ambiguous source 474 returned to the pilot"
    with source_path.open(newline="", encoding="utf-8") as source_file:
        source = list(csv.DictReader(source_file))
    for sample in samples:
        source_index = cast(int, sample["source_index"])
        row = source[source_index]
        for key, csv_key in (
            ("category", "Category"), ("question", "Question"),
            ("best_answer", "Best Answer"),
            ("best_incorrect_answer", "Best Incorrect Answer"),
        ):
            assert sample[key] == row[csv_key], f"Honesty source {source_index} changed {key}"


def test_honesty_reversal_keeps_source_answer_meaning() -> None:
    samples = json.loads(HONESTY_PROMPTS.read_text(encoding="utf-8"))
    for sample in samples:
        original, swapped = honesty_variants(sample)
        assert original["options"]["A"] == sample["best_answer"], (
            f"Source {sample['source_index']} lost its best-answer label"
        )
        assert original["options"]["B"] == sample["best_incorrect_answer"]
        assert swapped["options"]["B"] == sample["best_answer"]
        assert swapped["options"]["A"] == sample["best_incorrect_answer"]
        assert original["options"][original["matching_label"]] == swapped["options"][
            swapped["matching_label"]
        ], f"Source {sample['source_index']} changed meaning on reversal"
        assert swapped["question"].endswith(
            f"\n (A) {sample['best_incorrect_answer']}\n (B) {sample['best_answer']}"
        )


def test_selection_keeps_best_layer_even_when_all_effects_are_negative() -> None:
    assert select_layer({6: -0.08, 10: -0.01, 14: -0.01, 18: -0.04}) == 10, (
        "Selection failed the all-negative or lower-layer tie rule"
    )
    assert select_layer({6: 0.02, 10: 0.04, 14: 0.01}) == 10, (
        "Selection did not maximize the development change"
    )
