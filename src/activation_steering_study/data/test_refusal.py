import json
from pathlib import Path
import unicodedata

from activation_steering_study.data.refusal import load_reviewed_refusal, sample_refusal


def _read_rows(relative_path):
    upstream = Path("data/refusal/upstream/refusal_direction")
    return json.loads((upstream / relative_path).read_text(encoding="utf-8"))


def _normalize_for_overlap(instruction):
    # Normalize Unicode, ignore letter case, and collapse whitespace for duplicate checks.
    return " ".join(unicodedata.normalize("NFKC", instruction).casefold().split())


def test_default_sample_matches_saved_prompts():
    expected = json.loads(Path("data/refusal/prompts.json").read_text(encoding="utf-8"))

    assert sample_refusal() == expected, "Default seed must reproduce the saved 96 prompts"


def test_selected_sample_is_unique_and_has_no_test_overlap():
    sample = sample_refusal()
    test_rows = _read_rows("harmful_test.json") + _read_rows("harmless_test.json")
    selected_instructions = [
        _normalize_for_overlap(record["instruction"])
        for split in (sample["train"], sample["validation"])
        for record in split["harmful"] + split["harmless"]
    ]
    test_instructions = {
        _normalize_for_overlap(row["instruction"])
        for row in test_rows
    }

    assert len(selected_instructions) == len(set(selected_instructions)), (
        "Selected sample contains duplicate normalized prompts"
    )
    assert set(selected_instructions).isdisjoint(test_instructions), (
        "Selected sample overlaps test prompts"
    )


def test_reviewed_sample_matches_manual_review():
    expected = json.loads(Path("data/refusal/prompts.json").read_text(encoding="utf-8"))
    # Harmful training rows excluded during manual label review.
    excluded_indices = {50, 114, 125, 179}
    expected["train"]["harmful"] = [
        record
        for record in expected["train"]["harmful"]
        if record["sample_index"] not in excluded_indices
    ]

    assert load_reviewed_refusal() == expected, (
        "Reviewed sample must match the original sample after manual exclusions"
    )
