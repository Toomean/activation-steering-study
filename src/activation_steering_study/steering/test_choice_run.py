"""Check group weighting, signed dose choice, and the bounded dev schedule."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from activation_steering_study.evaluation.choices import ScoredChoiceVariant
from activation_steering_study.steering import choice_run


def _honesty(source_index: int, group_id: str, split: str, question: str) -> dict[str, object]:
    return {
        "source_index": source_index, "group_id": group_id, "split": split,
        "category": "animals", "question": question,
        "best_answer": "Dog", "best_incorrect_answer": "Cat",
    }


def test_group_means_weight_groups_equally(monkeypatch: pytest.MonkeyPatch) -> None:
    probabilities = {"First?": 0.2, "Second?": 0.4, "Third?": 0.9}

    def score(_tokenizer, _model, variant, **_kwargs):
        question = variant["question"].split("\n", 1)[0]
        order_offset = -0.1 if variant["order"] == "original" else 0.1
        matching = probabilities[question] + order_offset
        p_a = 0.8 * (matching if variant["matching_label"] == "A" else 1 - matching)
        return cast(ScoredChoiceVariant, {
            **variant, "p_a": p_a, "p_b": 0.8 - p_a, "ab_mass": 0.8,
            "conditional_matching_probability": matching,
        })

    monkeypatch.setattr(choice_run, "score_choice_variant", score)
    rows = cast(list[choice_run.ChoiceRow], [
        _honesty(1, "one", "development", "First?"),
        _honesty(2, "one", "development", "Second?"),
        _honesty(3, "two", "development", "Third?"),
    ])
    result = choice_run.score_condition(None, None, "honesty", rows, None, 0.0)
    assert result["mean_conditional_matching_probability"] == pytest.approx(0.6), (
        "Unequal row counts must not give the first group extra weight"
    )
    assert result["group_means"] == [
        {"group_id": "one", "mean": pytest.approx(0.3), "row_count": 2},
        {"group_id": "two", "mean": pytest.approx(0.9), "row_count": 1},
    ], "Saved group means must show the row counts used in weighting"
    assert all(len(row["orders"]) == 2 for row in result["results"]), (
        "Each source row must retain both answer orders"
    )


def test_signed_dose_selection_handles_ties_and_adverse_effects() -> None:
    means = {0.5: 0.55, 1.0: 0.55, 2.0: 0.51,
             -0.5: 0.57, -1.0: 0.57, -2.0: 0.54}
    assert choice_run.select_signed_dose(0.5, means, +1) == 0.5, (
        "Positive exact ties must select the smaller dose"
    )
    assert choice_run.select_signed_dose(0.5, means, -1) == -2.0, (
        "Negative selection must maximize the decrease in matching probability"
    )
    adverse = {-0.5: 0.53, -1.0: 0.53, -2.0: 0.57}
    assert choice_run.select_signed_dose(0.5, adverse, -1) == -0.5, (
        "Even when all negative doses have adverse effects, choose the least adverse and break ties"
    )


def test_runner_schedules_19_conditions_and_excludes_final(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    panel = [
        _honesty(1, "extract", "extraction", "Which animal barks?"),
        _honesty(2, "dev-a", "development", "Which animal meows?"),
        _honesty(3, "dev-b", "development", "Which animal purrs?"),
        # Rendering this final row would fail because it has no answer fields.
        {"source_index": 999, "group_id": "final", "split": "final",
         "question": "NEVER_RENDER_FINAL"},
    ]
    panel_path = tmp_path / "panel.json"
    panel_bytes = json.dumps(panel).encode()
    panel_path.write_bytes(panel_bytes)
    model = SimpleNamespace(dtype=torch.float32, device=torch.device("cpu"),
                            config=SimpleNamespace(hidden_size=2))
    output_dir = tmp_path / "new-run"
    capture_order: list[str] = []

    def code_hashes():
        assert not output_dir.exists(), "Code hashes must be captured before output creation"
        capture_order.append("hash")
        return {"runner": "snapshot-before-model-work"}

    def load():
        assert capture_order == ["hash"], "Code hashes must precede model work"
        capture_order.append("load")
        return object(), model

    monkeypatch.setattr(choice_run, "_code_hashes", code_hashes)
    monkeypatch.setattr(choice_run, "load_qwen", load)
    monkeypatch.setattr(choice_run, "TARGET_NORM", 10.0)
    raw_direction = torch.tensor([3.0, 4.0])
    extraction_ids: list[int] = []
    scored_controls: list[str] = []
    random_draws: list[tuple[int, float]] = []

    def random_direction(_hidden_size, norm, seed):
        random_draws.append((seed, norm))
        return torch.tensor([10.0, 0.0] if seed == 42 else [0.0, 10.0])

    monkeypatch.setattr(choice_run, "sample_random_direction", random_direction)

    def extract(_model, _tokenizer, sources, layers):
        extraction_ids.extend(source_id for source_id, _ in sources)
        assert layers == [18], "Extraction must use only block 18"
        assert all("NEVER_RENDER_FINAL" not in variant["question"]
                   for _, variants in sources for variant in variants), (
            "Final prompts must not enter extraction"
        )
        return {18: {"direction": raw_direction}}, [
            {"source_index": 1, "order": "original", "matching_label": "A"},
            {"source_index": 1, "order": "swapped", "matching_label": "B"},
        ]

    def score(_tokenizer, _model, variant, *, direction, alpha, layer_index):
        assert layer_index == 18, "Scoring must use block 18"
        assert "NEVER_RENDER_FINAL" not in variant["question"], (
            "Final prompts must not enter scoring"
        )
        if direction is None:
            control, effect = "baseline", 0.0
        elif torch.equal(direction, torch.tensor([6.0, 8.0])):
            control = "real"
            effect = {-2.0: -0.01, -1.0: -0.02, -0.5: -0.03,
                      0.5: 0.01, 1.0: 0.04, 2.0: 0.02}[alpha]
        elif torch.equal(direction, torch.tensor([10.0, 0.0])):
            control = "random42"
            effect = {-2.0: -0.20, -1.0: -0.10, -0.5: -0.05,
                      0.5: 0.05, 1.0: 0.10, 2.0: 0.20}[alpha]
        elif torch.equal(direction, torch.tensor([0.0, 10.0])):
            control = "random43"
            effect = {-2.0: -0.02, -1.0: -0.30, -0.5: -0.01,
                      0.5: 0.30, 1.0: 0.01, 2.0: 0.02}[alpha]
        else:
            raise AssertionError("Scoring received a vector other than real, random42, or random43")
        scored_controls.append(control)
        matching = 0.5 + effect
        p_a = 0.8 * (matching if variant["matching_label"] == "A" else 1 - matching)
        return cast(ScoredChoiceVariant, {
            **variant, "p_a": p_a, "p_b": 0.8 - p_a, "ab_mass": 0.8,
            "conditional_matching_probability": matching,
        })

    monkeypatch.setattr(choice_run, "extract_choice_pairs", extract)
    monkeypatch.setattr(choice_run, "score_choice_variant", score)
    artifact = choice_run.run("honesty", panel_path, output_dir)

    assert extraction_ids == [1], "Only extraction rows should define the direction"
    assert capture_order == ["hash", "load"], "Code hash snapshot must be taken only before model work"
    assert artifact["code_sha256"] == {"runner": "snapshot-before-model-work"}, (
        "Saved code hashes must use the pre-execution snapshot"
    )
    assert random_draws == [(42, 10.0), (43, 10.0)], (
        "Each local random seed must receive the historical target norm"
    )
    settings = cast(dict[str, dict[str, object]], artifact["condition_settings"])
    random_norms = cast(dict[str, float], artifact["random_direction_norms"])
    assert len(settings) == 19, "Dev matrix needs one baseline and 18 steered conditions"
    assert Counter(scored_controls) == {
        "baseline": 4, "real": 24, "random42": 24, "random43": 24,
    }, "Every control must score its own vector at six doses on both dev rows and orders"
    assert artifact["raw_direction_norm"] == 5.0, "Metadata must retain the raw extraction norm"
    assert artifact["target_direction_norm"] == 10.0, "Metadata must record the target norm"
    assert artifact["direction_scale_factor"] == 2.0, "Scaling must multiply raw [3, 4] by two"
    assert torch.equal(raw_direction, torch.tensor([3.0, 4.0])), (
        "Normalization must leave the extracted tensor unchanged"
    )
    saved_extraction = torch.load(output_dir / "extraction.pt", weights_only=True)
    assert torch.equal(saved_extraction[18]["direction"], torch.tensor([3.0, 4.0])), (
        "Saved extraction must retain raw [3, 4], not the scoring vector [6, 8]"
    )
    assert artifact["tensor_sha256"] == hashlib.sha256(
        (output_dir / "extraction.pt").read_bytes()
    ).hexdigest(), "Tensor provenance must hash the saved raw extraction bytes"
    assert artifact["model_device"] == "cpu", "Execution metadata must record model device"
    assert settings["real_alpha_+2"]["direction_norm"] == 10.0, (
        "Real scoring must use the normalized vector [6, 8]"
    )
    assert settings["real_alpha_-2"]["injected_norm"] == 20.0, (
        "Injected magnitude must use the absolute signed dose and target norm"
    )
    assert random_norms["42"] == pytest.approx(10.0), "Random42 must match target norm"
    assert random_norms["43"] == pytest.approx(10.0), "Random43 must match target norm"
    assert artifact["selected_real_alphas"] == {"positive": 1.0, "negative": -0.5}, (
        "Dev decision must use real effects, not the different random optima"
    )
    assert artifact["panel_sha256"] == hashlib.sha256(panel_bytes).hexdigest(), (
        "Panel provenance must hash the exact input bytes"
    )
    assert artifact["development_source_ids"] == [2, 3], "Dev IDs must exclude final"
    assert artifact["extraction_group_ids"] == ["extract"], "Extraction groups must be saved"
    assert len(list(output_dir.glob("*.json"))) == 20, (
        "Each condition and the final dev decision must be saved"
    )
    assert "NEVER_RENDER_FINAL" not in (output_dir / "summary.json").read_text(), (
        "Final row content must not appear in summary metadata"
    )
    assert settings["baseline"]["direction_norm"] == 0.0, "Baseline must be hook-free"


@pytest.mark.parametrize("layer", [None, 14])
def test_score_condition_forwards_default_or_explicit_layer(
    monkeypatch: pytest.MonkeyPatch, layer: int | None,
) -> None:
    direction = torch.tensor([3.0, 4.0])
    calls = []

    def score(_tokenizer, _model, variant, *, direction, alpha, layer_index):
        calls.append((direction, alpha, layer_index))
        return cast(ScoredChoiceVariant, {**variant, "conditional_matching_probability": 0.5})

    monkeypatch.setattr(choice_run, "score_choice_variant", score)
    rows = cast(list[choice_run.ChoiceRow], [_honesty(1, "dev", "development", "Which?")])
    if layer is None:
        choice_run.score_condition(None, None, "honesty", rows, direction, -0.5)
    else:
        choice_run.score_condition(None, None, "honesty", rows, direction, -0.5, layer_index=layer)
    assert len(calls) == 2, "Both orders must use the caller's layer"
    assert all(vector is direction and alpha == -0.5 and block == (18 if layer is None else 14)
               for vector, alpha, block in calls), "Only the explicit layer override changes block 18"
