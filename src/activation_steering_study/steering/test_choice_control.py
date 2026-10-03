"""Check control qualification with synthetic panels and fixed probability examples."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from activation_steering_study.evaluation.choices import ScoredChoiceVariant
from activation_steering_study.steering import choice_control, choice_run
from activation_steering_study.steering.random_control import sample_random_direction


def _panel() -> list[dict]:
    return [{
        "source_index": index, "group_id": group, "split": split, "question": question,
        "category": "synthetic", "best_answer": "Dog", "best_incorrect_answer": "Cat",
    } for index, group, split, question in (
        (1, "extract", "extraction", "First?"), (2, "dev-a", "development", "Second?"),
        (3, "dev-a", "development", "Third?"), (4, "dev-b", "development", "Fourth?"),
    )] + [{"source_index": 999, "group_id": "final", "split": "final",
           "question": "NEVER_RENDER_FINAL"}]


def _reference() -> dict:
    return {
        "model_id": choice_control.MODEL_ID, "model_revision": choice_control.MODEL_REVISION,
        "layer_index": 14, "model_dtype": "torch.float32",
        "direction_norms": dict(zip(choice_control.TARGETS, (5.0, 10.0, 15.0, 20.0), strict=True)),
        "tensor_sha256": "declared-not-verified-here", "source_metadata_sha256": "original-hash",
    }


def _baseline() -> dict:
    return {
        "mean_conditional_matching_probability": 0.45,
        "group_means": [{"group_id": "dev-a", "mean": 0.2, "row_count": 2},
                        {"group_id": "dev-b", "mean": 0.7, "row_count": 1}],
        "results": [{"source_index": index, "group_id": group, "split": "development",
                     "row_mean": mean, "orders": []}
                    for index, group, mean in ((2, "dev-a", 0.1), (3, "dev-a", 0.3), (4, "dev-b", 0.7))],
    }


def _inputs(tmp_path: Path, panel: list[dict] | None = None,
            reference: dict | None = None) -> tuple[Path, Path, Path]:
    panel_path, reference_path = tmp_path / "panel.json", tmp_path / "reference.json"
    panel_path.write_text(json.dumps(_panel() if panel is None else panel))
    reference_path.write_text(json.dumps(_reference() if reference is None else reference))
    (tmp_path / "baseline.json").write_text(json.dumps(_baseline()))
    return panel_path, reference_path, tmp_path / "run"


@pytest.mark.parametrize("behaviour", ["honesty", "sycophancy"])
def test_full_schedule_keeps_fresh_block14_raw_pairs_and_excludes_final(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, behaviour: str,
) -> None:
    panel = _panel()
    if behaviour == "sycophancy":
        for row in panel[:-1]:
            row.update(question=row["question"] + "\n (A) Dog\n (B) Cat",
                       answer_matching_behavior="(A)", answer_not_matching_behavior="(B)")
    panel_path, reference_path, output = _inputs(tmp_path, panel)
    panel_bytes, reference_bytes = panel_path.read_bytes(), reference_path.read_bytes()
    baseline_path = tmp_path / "baseline.json"
    baseline_bytes = baseline_path.read_bytes()
    model = SimpleNamespace(dtype=torch.float32, device=torch.device("cpu"),
                            config=SimpleNamespace(hidden_size=2, _attn_implementation="eager"))
    raw = torch.tensor([3.0, 4.0])
    captured: list[tuple[torch.Tensor | None, float]] = []
    draws: list[tuple[float, int, torch.Tensor]] = []

    def load():
        # The stored hashes must describe input bytes from before model loading.
        panel_path.write_text("[]")
        reference_path.write_text("{}")
        baseline_path.write_text("{}")
        return object(), model

    def extract(_model, _tokenizer, sources, layers):
        execution = json.loads((output / "execution.json").read_text())
        assert execution["attention_implementation"] == "eager", "Record execution before extraction"
        assert execution["torch_num_threads"] == torch.get_num_threads(), "Keep actual thread count"
        assert execution["model_dtype"] == "torch.float32" and execution["model_device"] == "cpu", (
            "Execution metadata must retain the actual dtype and device"
        )
        assert execution["baseline_validation"]["matches"] is True and captured == [(None, 0.0)], (
            "The fresh baseline must reproduce the archive before extraction"
        )
        assert layers == [14] and [source_id for source_id, _ in sources] == [1], (
            "Fresh extraction must use only block 14 and extraction sources"
        )
        assert [v["matching_label"] for v in sources[0][1]] == ["A", "B"], (
            "Both source-labelled orders must enter extraction"
        )
        return {14: {"direction": raw, "pair_differences": raw.reshape(1, 1, 2)}}, [
            {"source_index": 1, "order": "original", "matching_label": "A"},
        ]

    def random_vector(hidden_size, norm, seed):
        vector = sample_random_direction(hidden_size, norm, seed)
        draws.append((norm, seed, vector))
        return vector

    def score(_tokenizer, _model, variant, *, direction, alpha, layer_index):
        assert layer_index == 14 and "NEVER_RENDER_FINAL" not in variant["question"], (
            "Every own-task scoring call must use block 14 and development prompts"
        )
        assert (output / "execution.json").exists(), "Persist execution provenance before the first baseline forward"
        question = variant["question"].split("\n", 1)[0]
        if question == "Second?" and variant["order"] == "original":
            captured.append((None if direction is None else direction.clone(), alpha))
        real = direction is not None and torch.allclose(direction / direction.norm(), raw / raw.norm())
        random42 = direction is not None and torch.allclose(
            direction / direction.norm(), sample_random_direction(2, 1.0, 42),
        )
        effect = 0.0 if direction is None else alpha * (0.06 if real else 0.02 if random42 else 0.04)
        probability = {"Second?": 0.1, "Third?": 0.3, "Fourth?": 0.7}[question] + effect
        return cast(ScoredChoiceVariant, {**variant, "p_a": probability, "p_b": 1 - probability,
                                         "ab_mass": 1.0, "conditional_matching_probability": probability})

    monkeypatch.setattr(choice_control, "load_qwen", load)
    monkeypatch.setattr(choice_control, "extract_choice_pairs", extract)
    monkeypatch.setattr(choice_control, "sample_random_direction", random_vector)
    monkeypatch.setattr(choice_run, "score_choice_variant", score)
    artifact = choice_control.run(behaviour, panel_path, reference_path, output, baseline_path=baseline_path)
    settings = cast(dict[str, dict], artifact["condition_settings"])
    summaries = cast(dict[str, dict], artifact["condition_summaries"])
    cells = cast(dict[str, dict], artifact["cell_evidence"])
    assert len(settings) == len(captured) == 73, "Score one fresh baseline and 72 steered conditions"
    assert summaries["baseline"]["mean_conditional_matching_probability"] == pytest.approx(0.45), (
        "The two dev groups must receive equal weight despite unequal row counts"
    )
    assert captured[0] == (None, 0.0), "The baseline must be hook-free and computed exactly once"
    assert [(norm, seed) for norm, seed, _ in draws] == [
        (5.0, 42), (5.0, 43), (10.0, 42), (10.0, 43),
        (15.0, 42), (15.0, 43), (20.0, 42), (20.0, 43),
    ], "Sample both frozen seeds at each distinct target norm"
    for setting, (vector, alpha) in zip(list(settings.values())[1:], captured[1:], strict=True):
        assert vector is not None, "Every steered condition must receive a vector"
        norm = _reference()["direction_norms"][setting["target"]]
        assert vector.norm().item() == pytest.approx(norm), "Every passed vector must have its target norm"
        assert setting["alpha"] == alpha and setting["injected_norm"] == pytest.approx(abs(alpha) * norm), (
            "Saved signed alpha and injected norm must describe the actual scoring call"
        )
        expected = (raw * (norm / 5) if setting["control"] == "real" else next(
            draw for draw_norm, seed, draw in draws if draw_norm == norm and seed == setting["random_seed"]
        ))
        assert torch.allclose(vector, expected), "Keep real polarity and the actual seed-specific random orientation"
    assert set(setting["alpha"] for setting in settings.values()) == {-2, -1, -.5, 0, .5, 1, 2}, (
        "The schedule must cover the frozen signed doses plus baseline"
    )
    assert cells["pooled_alpha_+1"]["signed_random_changes"] == {
        "42": pytest.approx(0.02), "43": pytest.approx(0.04),
    }, "Preserve unequal seed-specific effects under the correct saved labels"
    assert len(cells) == 24 and cells["pooled_alpha_+1"]["passes"] is True, (
        "Each target/dose needs its own evidence; the synthetic +1 real shift clears both random shifts"
    )
    assert cells["pooled_alpha_-1"]["signed_real_change"] == pytest.approx(0.06), (
        "A negative dose must sign-reverse the decrease into a positive own-task effect"
    )
    assert cells["pooled_alpha_+0.5"]["passes"] is False, "Passing at one dose cannot qualify smaller doses"
    assert torch.equal(raw, torch.tensor([3.0, 4.0])), "Raw extraction must not be scaled in place"
    assert torch.equal(torch.load(output / "extraction.pt", weights_only=True)[14]["direction"], raw), (
        "Saved extraction must preserve raw block-14 direction bytes rather than scaled vectors"
    )
    assert artifact["panel_sha256"] == hashlib.sha256(panel_bytes).hexdigest(), (
        "Panel provenance must use bytes captured before model loading"
    )
    baseline_record = cast(dict, artifact["baseline_validation"])
    assert baseline_record["sha256"] == hashlib.sha256(baseline_bytes).hexdigest(), (
        "The archived baseline hash must use the initial bytes even if the file changes later"
    )
    assert baseline_record["expected_mean"] == 0.45 and baseline_record["actual_mean"] == pytest.approx(0.45), (
        "Baseline comparison must record both known archived and fresh scalar means"
    )
    reference_record = cast(dict, artifact["refusal_scale_reference"])
    assert reference_record["sha256"] == hashlib.sha256(reference_bytes).hexdigest(), (
        "Scale provenance must use bytes captured before model loading"
    )
    assert reference_record["declared_tensor_sha256"] == "declared-not-verified-here", (
        "Retain the reference tensor hash as a declaration without claiming fresh validation"
    )
    assert artifact["uv_lock_sha256"] == hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(), (
        "Execution provenance must identify the dependency lock"
    )
    assert artifact["tensor_sha256"] == hashlib.sha256((output / "extraction.pt").read_bytes()).hexdigest(), (
        "Raw extraction provenance must identify the exact saved tensor artifact"
    )
    hashes = cast(dict, artifact["code_sha256"])
    assert all(hashes[path] == hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in hashes), (
        "Every recorded code digest must identify the actual dependency bytes"
    )
    assert set(hashes) == {
        "src/activation_steering_study/" + path for path in (
            "evaluation/choices.py", "evaluation/scoring.py", "evaluation/sycophancy.py",
            "extraction/answer_activations.py", "extraction/paired.py", "steering/choice_sweep.py",
            "steering/intervention.py", "steering/random_control.py", "utils/choice_prompt.py",
            "utils/json_io.py", "utils/qwen.py", "steering/choice_run.py", "steering/choice_control.py",
        )
    }, "The snapshot must cover all local functions used by extraction, formatting and scoring"
    assert set(hashes) >= {
        "src/activation_steering_study/steering/choice_control.py",
        "src/activation_steering_study/steering/choice_run.py",
    }, "Code provenance must include both the shared scorer and the new runner"
    assert artifact["extraction_source_ids"] == [1] and artifact["development_source_ids"] == [2, 3, 4], (
        "Recorded extraction and development IDs must exclude final"
    )
    assert "NEVER_RENDER_FINAL" not in (output / "summary.json").read_text(), (
        "Final prompt content must not leak into qualification metadata"
    )
    assert len(list(output.glob("*.json"))) == 75, "Save each Condition plus execution and summary provenance"


@pytest.mark.parametrize("baseline,real,randoms,alpha,passes,change", [
    (0.0, 0.05, {42: 0.01, 43: 0.02}, 1.0, True, 0.05),
    (0.05, 0.0, {42: 0.04, 43: 0.03}, -1.0, True, 0.05),
    (0.0, 0.049, {42: 0.01, 43: 0.02}, 1.0, False, 0.049),
    (0.0, 0.05, {42: 0.05, 43: 0.02}, 1.0, False, 0.05),
    (0.0, 0.05, {42: 0.01, 43: 0.06}, 1.0, False, 0.05),
    (0.5, 0.6, {42: 0.4, 43: 0.3}, -1.0, False, -0.1),
    (0.1, 0.0, {42: 0.05, 43: 0.0}, -1.0, False, 0.1),
    (0.2, 0.1, {42: 0.15, 43: 0.0}, -1.0, False, 0.1),
])
def test_cell_threshold_sign_and_both_random_comparisons(baseline, real, randoms, alpha, passes, change):
    result = choice_control.cell_evidence(baseline, real, randoms, alpha)
    assert result["passes"] is passes and result["signed_real_change"] == pytest.approx(change), (
        "Eligibility must use the signed threshold and strictly beat both fixed controls"
    )


@pytest.mark.parametrize("baseline,real,randoms", [
    (float("nan"), 0.1, {42: 0.0, 43: 0.0}),
    (0.0, float("inf"), {42: 0.0, 43: 0.0}),
    (0.0, 0.1, {42: 0.0, 43: float("nan")}),
])
def test_nonfinite_cell_arithmetic_fails(baseline, real, randoms):
    with pytest.raises(ValueError, match="Nonfinite"):
        choice_control.cell_evidence(baseline, real, randoms, 1.0)


@pytest.mark.parametrize("case", ["existing", "split", "empty-extract", "empty-dev", "duplicate",
    "source-dev-final", "source-ext-final", "group-ext-dev", "group-dev-final", "group-ext-final",
    "model", "revision", "block", "dtype", "missing-norm", "extra-norm", "zero-norm",
    "negative-norm", "nan-norm", "inf-norm", "bool-norm", "string-norm"])
def test_invalid_inputs_fail_before_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str):
    panel, reference = _panel(), _reference()
    if case == "split":
        panel[0]["split"] = "training"
    elif case in ("empty-extract", "empty-dev"):
        absent = "extraction" if case == "empty-extract" else "development"
        panel = [row for row in panel if row["split"] != absent]
    elif case == "duplicate":
        panel[2]["source_index"] = panel[1]["source_index"]
    elif case.startswith("source-") or case.startswith("group-"):
        field = "source_index" if case.startswith("source-") else "group_id"
        before, after = {"ext-dev": (0, 1), "dev-final": (1, -1), "ext-final": (0, -1)}[
            case.split("-", 1)[1]]
        panel[after][field] = panel[before][field]
    elif case in ("model", "revision", "block", "dtype"):
        key = {"model": "model_id", "revision": "model_revision", "block": "layer_index",
               "dtype": "model_dtype"}[case]
        reference[key] = "invalid"
    elif case.endswith("norm"):
        if case == "missing-norm":
            reference["direction_norms"].pop("pooled")
        elif case == "extra-norm":
            reference["direction_norms"]["extra"] = 1
        else:
            reference["direction_norms"]["pooled"] = {
                "zero-norm": 0, "negative-norm": -1, "nan-norm": float("nan"),
                "inf-norm": float("inf"), "bool-norm": True, "string-norm": "5",
            }[case]
    panel_path, reference_path, output = _inputs(tmp_path, panel, reference)
    if case == "existing":
        output.mkdir()
        (output / "sentinel").write_text("preserve")
    monkeypatch.setattr(choice_control, "load_qwen", lambda: pytest.fail("Invalid inputs must not load a model"))
    with pytest.raises((ValueError, FileExistsError)):
        choice_control.run("honesty", panel_path, reference_path, output, baseline_path=tmp_path / "baseline.json")
    if case == "existing":
        assert (output / "sentinel").read_text() == "preserve", "Never overwrite existing artifacts"
    else:
        assert not output.exists(), "Static validation must finish before creating outputs"


def test_loaded_dtype_mismatch_fails_before_extraction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(choice_control, "load_qwen", lambda: (None, SimpleNamespace(dtype=torch.bfloat16)))
    monkeypatch.setattr(choice_control, "extract_choice_pairs", lambda *args: pytest.fail("No extraction on dtype mismatch"))
    with pytest.raises(ValueError, match="dtype differs"):
        choice_control.run("honesty", *inputs, baseline_path=tmp_path / "baseline.json")


@pytest.mark.parametrize("case", ["source", "group", "order", "nan", "inf"])
def test_invalid_archived_baseline_fails_before_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str,
) -> None:
    inputs = _inputs(tmp_path)
    archived = _baseline()
    if case in ("source", "group"):
        archived["results"][0]["source_index" if case == "source" else "group_id"] = "different"
    elif case == "order":
        archived["results"].reverse()
    else:
        archived["mean_conditional_matching_probability"] = float(case)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(archived))
    monkeypatch.setattr(choice_control, "load_qwen", lambda: pytest.fail("Invalid archive must not load a model"))
    with pytest.raises(ValueError, match="baseline"):
        choice_control.run("honesty", *inputs, baseline_path=baseline_path)
    assert not inputs[2].exists(), "Baseline identity/finite validation must finish before output creation"


@pytest.mark.parametrize("expected,actual,passes", [
    (0.45, 0.45, True), (0.0, 1e-12, True),
    (0.0, 1.0000000000000002e-12, False), (0.45, 0.45 + 1e-11, False),
])
def test_baseline_absolute_tolerance_stops_drift_before_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, expected: float, actual: float, passes: bool,
) -> None:
    inputs = _inputs(tmp_path)
    archived = _baseline()
    archived["mean_conditional_matching_probability"] = expected
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(archived))
    model = SimpleNamespace(dtype=torch.float32, device=torch.device("cpu"),
                            config=SimpleNamespace(_attn_implementation="eager"))
    calls = []

    def score(_tokenizer, _model, _behaviour, _rows, vector, alpha, *, layer_index):
        calls.append((vector, alpha, layer_index))
        return {"mean_conditional_matching_probability": actual, "group_means": [], "results": []}

    def extract(*_args):
        if not passes:
            pytest.fail("Baseline drift must stop before extraction")
        raise RuntimeError("baseline validated")

    monkeypatch.setattr(choice_control, "load_qwen", lambda: (None, model))
    monkeypatch.setattr(choice_control, "score_condition", score)
    monkeypatch.setattr(choice_control, "extract_choice_pairs", extract)
    error = RuntimeError if passes else ValueError
    message = "baseline validated" if passes else "beyond absolute tolerance"
    with pytest.raises(error, match=message):
        choice_control.run("honesty", *inputs, baseline_path=baseline_path)
    assert calls == [(None, 0.0, 14)], "Baseline must be the only scoring call before extraction"
    output = inputs[2]
    check = json.loads((output / "execution.json").read_text())["baseline_validation"]
    assert check["expected_mean"] == expected and check["actual_mean"] == actual, (
        "Preserve the archived and recomputed scalar means even when drift stops the run"
    )
    assert check["delta"] == actual - expected and check["absolute_tolerance"] == 1e-12, (
        "Record the computed delta and absolute-only tolerance without a relative allowance"
    )
    assert check["matches"] is passes, "Equality at the computed absolute tolerance must pass"
    assert json.loads((output / "baseline.json").read_text())["mean_conditional_matching_probability"] == actual, (
        "Preserve fresh baseline scores for diagnosis on failure"
    )
    assert not (output / "summary.json").exists() and not (output / "extraction.pt").exists(), (
        "An interrupted or drifting baseline must not produce qualification or extraction artifacts"
    )


@pytest.mark.parametrize("alpha,real,random42,random43,shifts", [
    (1.0, 0.56, 0.4, 0.49, {"42": -0.1, "43": -0.01}),
    (-1.0, 0.44, 0.6, 0.51, {"42": -0.1, "43": -0.01}),
])
def test_opposite_random_shifts_are_signed_not_absolute(alpha, real, random42, random43, shifts):
    evidence = choice_control.cell_evidence(0.5, real, {42: random42, 43: random43}, alpha)
    assert evidence["passes"] is True, "Opposite-direction random effects must not become positive competitors"
    assert evidence["signed_random_changes"] == {seed: pytest.approx(value) for seed, value in shifts.items()}, (
        "Retain each seed's signed change rather than its absolute magnitude"
    )


def test_bad_development_prompt_fails_before_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    panel = _panel()
    panel[1].pop("best_answer")
    inputs = _inputs(tmp_path, panel)
    monkeypatch.setattr(choice_control, "load_qwen", lambda: pytest.fail("Malformed active prompts must fail before model load"))
    with pytest.raises(KeyError, match="best_answer"):
        choice_control.run("honesty", *inputs, baseline_path=tmp_path / "baseline.json")
    assert not inputs[2].exists(), "Prompt validation must precede output creation; final markers remain unformatted"
