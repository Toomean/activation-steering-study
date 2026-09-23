import json
from pathlib import Path

import pytest

from activation_steering_study.analysis.refusal import main, summarize_refusal


def _pair(
    source: str,
    index: int,
    baseline: str,
    dim: str,
    random: str,
) -> tuple[dict, dict]:
    record = {"sample_path": source, "sample_index": index}
    return (
        {
            **record,
            "baseline_review": {
                "refusal": baseline,
                "quality": "ok",
                "unfinished": "no",
            },
            "steered_review": {
                "refusal": dim,
                "quality": "not_applicable",
                "unfinished": "yes",
            },
        },
        {
            **record,
            "refusal": random,
            "quality": "unclear",
            "unfinished": "unclear",
        },
    )


def _copy_review_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    source_artifacts = Path(__file__).parents[1] / "artifacts"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    pilot_path = artifact_dir / "refusal-pilot-review.json"
    random_path = artifact_dir / "refusal-random-review.json"
    pilot_path.write_text(
        (source_artifacts / pilot_path.name).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    random_path.write_text(
        (source_artifacts / random_path.name).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return pilot_path, random_path


def test_summary_joins_shuffled_random_records_and_preserves_pair_signs() -> None:
    first, random_first = _pair("harmless-a.json", 1, "none", "full", "none")
    second, random_second = _pair("harmless-b.json", 1, "full", "none", "full")
    third, random_third = _pair("harmless-a.json", 2, "none", "mixed", "full")

    summary = summarize_refusal(
        [first, second, third],
        [random_third, random_first, random_second],
    )

    assert summary["refusal_counts"] == {
        "baseline": {"none": 2, "full": 1, "mixed": 0},
        "dim": {"none": 1, "full": 1, "mixed": 1},
        "random": {"none": 1, "full": 2, "mixed": 0},
    }, "all three conditions should keep full and mixed separate"
    assert summary["quality_counts"]["random"] == {
        "ok": 0,
        "problem": 0,
        "unclear": 3,
        "not_applicable": 0,
    }, "quality counts should retain explicit unclear and not_applicable categories"
    assert summary["unfinished_counts"]["dim"] == {"no": 0, "yes": 3, "unclear": 0}, (
        "unfinished counts should retain the explicit unclear category"
    )
    assert summary["transitions"] == {"none->full": 1, "full->none": 1, "none->mixed": 1}, (
        "transitions should use baseline and DiM records with matching IDs"
    )
    assert summary["dim_minus_baseline"] == [1, -1, 1], (
        "DiM-minus-baseline signs should follow each paired record"
    )
    assert summary["dim_minus_random"] == [1, -1, 0], (
        "DiM-minus-random signs should join shuffled random records by both identifiers"
    )
    assert (
        summary["any_refusal_percent"]["dim"]
        - summary["any_refusal_percent"]["baseline"]
        == pytest.approx(100 / 3)
    ), "the reported DiM-minus-baseline rate should preserve the paired direction"
    assert summary["any_refusal_percent"]["dim"] - summary["any_refusal_percent"]["random"] == 0, (
        "the reported DiM-minus-random rate should preserve the matched controls"
    )


def test_summary_rejects_empty_unresolved_or_unmatched_reviews() -> None:
    first, random_first = _pair("harmless-a.json", 1, "none", "full", "none")

    with pytest.raises(ValueError, match="At least one reviewed result"):
        summarize_refusal([], [])
    with pytest.raises(ValueError, match="Unsupported refusal labels"):
        summarize_refusal(
            [
                {
                    **first,
                    "baseline_review": {
                        "refusal": "unclear",
                        "quality": "ok",
                        "unfinished": "no",
                    },
                }
            ],
            [random_first],
        )
    for field in ("quality", "unfinished"):
        with pytest.raises(ValueError, match=f"Unsupported {field} labels"):
            summarize_refusal([first], [{**random_first, field: "invalid"}])
    with pytest.raises(ValueError, match="unmatched records"):
        summarize_refusal([first], [])
    with pytest.raises(ValueError, match="Duplicate review record"):
        summarize_refusal([first], [random_first, random_first])
    with pytest.raises(ValueError, match="Duplicate review record"):
        summarize_refusal([first, first], [random_first])
    with pytest.raises(ValueError, match="unmatched records"):
        summarize_refusal(
            [first],
            [random_first, {**random_first, "sample_index": 2}],
        )


def test_main_rejects_mismatched_reference_direction_norm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot_path, random_path = _copy_review_artifacts(tmp_path)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    random = json.loads(random_path.read_text(encoding="utf-8"))
    random["reference_direction_norm"] = pilot["direction_norm"] + 1
    random_path.write_text(json.dumps(random), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="reference direction norms differ"):
        main()


def test_main_rejects_mismatched_random_direction_norm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, random_path = _copy_review_artifacts(tmp_path)
    random = json.loads(random_path.read_text(encoding="utf-8"))
    random["random_direction_norm"] = random["reference_direction_norm"] + 1
    random_path.write_text(json.dumps(random), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Random direction norm differs"):
        main()
