"""Synthetic execution checks; frozen source text is never scored or printed."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from activation_steering_study.steering import harmless_execution as execution
from activation_steering_study.steering.harmless_run import ConditionResult, load_condition_items


@pytest.fixture
def directions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    tensors = {name: torch.arange(1536, dtype=torch.float32) / (1000 + index)
               for index, name in enumerate(execution.TARGETS)}
    tensor_path = tmp_path / "directions.pt"
    torch.save({"directions": tensors}, tensor_path)
    digest = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    reference = {"tensor_sha256": digest, "model_id": execution.MODEL_ID,
                 "model_revision": execution.MODEL_REVISION, "model_dtype": "torch.bfloat16",
                 "layer_index": 14, "direction_norms": {name: value.norm().item() for name, value in tensors.items()}}
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    monkeypatch.setattr(execution, "TENSOR_SHA256", digest)
    monkeypatch.setattr(execution, "REFERENCE_SHA256", hashlib.sha256(reference_path.read_bytes()).hexdigest())
    return tensor_path, reference_path


def result_for(spec: execution.ConditionSpec, refusal_count: int = 0) -> ConditionResult:
    harmless, mmlu = load_condition_items(spec.role)
    return {"schema_version": 1, "role": spec.role, "baseline": spec.control == "baseline",
            "alpha": spec.alpha, "layer_index": 14, "max_new_tokens": 256,
            "harmless": [{"row_id": item["row_id"], "semantic_group_id": item["semantic_group_id"],
                          "quality_subset": item["quality_subset"], "refusal_proxy": index < refusal_count,
                          "phrase_matches": execution.refusal_phrase_matches("I cannot" if index < refusal_count else ""),
                          "completion": "I cannot" if index < refusal_count else "",
                          "prompt_token_ids": [1, 2], "generated_token_ids": [3],
                          "completion_token_count": 1, "completion_token_cap": 256,
                          "response_likelihood": {"token_count": 1, "mean_nll": 0.0, "perplexity": 1.0} if item["quality_subset"] else None,
                          "prompt_kl": 0.0 if item["quality_subset"] else None,
                          "generation_seconds": 0.0, "quality_seconds": 0.0}
                         for index, item in enumerate(harmless)],
            "mmlu": [{"id": item["id"], "subject": item["subject"], "source_index": item["source_index"],
                      "gold": "ABCD"[item["answer"]], "prediction": "A", "probabilities_abcd": [0.4, 0.2, 0.1, 0.1],
                      "option_mass": 0.8, "correct": item["answer"] == 0, "scoring_seconds": 0.0} for item in mmlu],
            "timing_seconds": {"generation": 0.0, "quality": 0.0, "mmlu": 0.0, "total": 0.0}}


def model_pair() -> tuple[Any, Any]:
    tokenizer = SimpleNamespace(name_or_path=execution.MODEL_ID,
                                init_kwargs={"vocab_file": f"/invented/snapshots/{execution.MODEL_REVISION}/vocab.json"})
    model = SimpleNamespace(config=SimpleNamespace(_name_or_path=execution.MODEL_ID,
                            _commit_hash=execution.MODEL_REVISION, _attn_implementation="sdpa", hidden_size=1536),
                            dtype=torch.bfloat16, device=torch.device("cpu"), training=False)
    return tokenizer, model


def test_fixed_queue_ids_and_exact_ties() -> None:
    dev = execution.development_specs()
    assert len(dev) == 11 and len({spec.condition_id for spec in dev}) == 11
    assert [spec.condition_id for spec in dev[:5]] == ["development-baseline", "development-pooled-real-a1",
        "development-cyber_intrusion-real-a0p5", "development-cyber_intrusion-real-a1", "development-cyber_intrusion-real-a2"]
    counts = [12, 20, 19, 21, 25, 5, 18, 22, 20, 20, 20]
    artifacts = [cast(execution.CompletedArtifact, {"condition": spec.to_dict(), "result": result_for(spec, count)})
                 for spec, count in zip(dev, counts, strict=True)]
    doses = execution.select_domain_doses(artifacts)
    assert doses == {"pooled": 1.0, "cyber_intrusion": 0.5, "dangerous_substances": 1.0, "disinformation": 0.5}
    final = execution.final_specs(doses)
    assert len(final) == 13 and len({spec.condition_id for spec in final}) == 13
    assert [spec.control for spec in final[1:4]] == ["real", "random42", "random43"]
    assert final[4].condition_id == "final-cyber_intrusion-real"
    assert final[4].alpha == 0.5
    for invalid in (artifacts[:-1], [*artifacts[:-1], artifacts[0]]):
        with pytest.raises(ValueError):
            execution.select_domain_doses(invalid)
    wrong_control = deepcopy(artifacts)
    wrong_control[1]["condition"] = execution.ConditionSpec("development", "random42", "pooled", 1.0).to_dict()
    with pytest.raises(ValueError):
        execution.select_domain_doses(wrong_control)


@pytest.mark.parametrize("role", ["development", "final"])
def test_full_frozen_panels_and_empty_decoded_responses_valid(role) -> None:
    spec = execution.ConditionSpec(role, "baseline")
    result = result_for(spec)
    assert execution.validate_condition_result(result, spec) is result
    first = result["harmless"][0]
    first["generated_token_ids"] = []
    first["completion_token_count"] = 0
    if first["quality_subset"]:
        first["response_likelihood"] = {"token_count": 0, "mean_nll": None, "perplexity": None}
    assert execution.validate_condition_result(result, spec) is result


@pytest.mark.parametrize("field,value", [("role", "final"), ("baseline", 1), ("alpha", False),
                                        ("layer_index", True), ("max_new_tokens", 255), ("schema_version", True)])
def test_condition_metadata_invalid(field, value) -> None:
    spec = execution.ConditionSpec("development", "baseline")
    result: Any = result_for(spec)
    result[field] = value
    with pytest.raises(ValueError):
        execution.validate_condition_result(result, spec)


@pytest.mark.parametrize("mutation", ["subset", "duplicate", "order", "group", "quality", "missing_quality", "phrase", "tokens", "nan", "negative_time"])
def test_invalid_harmless_output(mutation) -> None:
    spec = execution.ConditionSpec("development", "baseline")
    result: Any = result_for(spec)
    rows = result["harmless"]
    if mutation == "subset": rows.pop()
    elif mutation == "duplicate": rows[1] = deepcopy(rows[0])
    elif mutation == "order": rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "group": rows[0]["semantic_group_id"] = "invented-group"
    elif mutation == "quality": rows[0]["quality_subset"] = not rows[0]["quality_subset"]
    elif mutation == "missing_quality": next(row for row in rows if row["quality_subset"])["response_likelihood"] = None
    elif mutation == "phrase": rows[0]["refusal_proxy"] = True
    elif mutation == "tokens": rows[0]["generated_token_ids"] = [True]
    elif mutation == "nan": next(row for row in rows if row["quality_subset"])["prompt_kl"] = float("nan")
    elif mutation == "negative_time": rows[0]["generation_seconds"] = -1
    with pytest.raises(ValueError):
        execution.validate_condition_result(result, spec)


@pytest.mark.parametrize("mutation", ["subset", "wrong_id", "source_bool", "prediction", "mass", "probability_bool", "correct_bool", "missing"])
def test_invalid_mmlu_output(mutation) -> None:
    spec = execution.ConditionSpec("development", "baseline")
    result: Any = result_for(spec)
    row = result["mmlu"][0]
    if mutation == "subset": result["mmlu"].pop()
    elif mutation == "wrong_id": row["id"] = "invented-id"
    elif mutation == "source_bool": row["source_index"] = False
    elif mutation == "prediction": row["prediction"] = "B"
    elif mutation == "mass": row["option_mass"] = 0.9
    elif mutation == "probability_bool": row["probabilities_abcd"][0] = True
    elif mutation == "correct_bool": row["correct"] = 1
    elif mutation == "missing": del row["option_mass"]
    with pytest.raises(ValueError):
        execution.validate_condition_result(result, spec)


def test_direction_identity_and_raw_scale(directions) -> None:
    tensor, reference = directions
    hashes = []
    for control in ("real", "random42", "random43"):
        spec = execution.ConditionSpec("final", cast(execution.Control, control), "cyber_intrusion", 2.0)
        vector, provenance = execution.prepare_direction(spec, tensor, reference)
        assert vector is not None
        assert provenance["random_seed"] == spec.random_seed
        assert provenance["injected_norm"] == 2 * provenance["vector_norm"]
        assert provenance["vector_norm"] == pytest.approx(provenance["raw_norm"], rel=1e-6)
        hashes.append(provenance["vector_sha256"])
    assert len(set(hashes)) == 3, "Both actual seeded controls must retain distinct vector identities"
    direction, provenance = execution.prepare_direction(execution.ConditionSpec("final", "baseline"), tensor, reference)
    assert direction is None and provenance["target"] is None and provenance["injected_norm"] == 0


def test_fixed_cpu_seed_draws_across_target_scales(directions) -> None:
    tensor, reference = directions
    for seed in (42, 43):
        raw = torch.randn(1536, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)
        expected_unit = raw / raw.norm()
        for target in execution.TARGETS:
            spec = execution.ConditionSpec("final", cast(execution.Control, f"random{seed}"), target, 1.0)
            vector, provenance = execution.prepare_direction(spec, tensor, reference)
            assert vector is not None
            expected = raw * (provenance["raw_norm"] / raw.norm())
            assert torch.equal(vector, expected), "Recorded seed must identify the actual fixed CPU draw"
            assert torch.allclose(vector / vector.norm(), expected_unit, atol=1e-7, rtol=1e-6), (
                "All target norms must scale the same seed orientation")


@pytest.mark.parametrize("mutation", ["norm", "layer", "dtype", "nonfinite", "missing"])
def test_bad_reference_or_vector_rejected(directions, monkeypatch, mutation) -> None:
    tensor, reference = directions
    numerical = json.loads(reference.read_text())
    if mutation == "norm": numerical["direction_norms"]["disinformation"] *= 2
    elif mutation == "layer": numerical["layer_index"] = True
    else:
        saved = torch.load(tensor, weights_only=True)
        if mutation == "dtype": saved["directions"]["disinformation"] = saved["directions"]["disinformation"].double()
        elif mutation == "nonfinite": saved["directions"]["disinformation"][0] = float("nan")
        elif mutation == "missing": del saved["directions"]["disinformation"]
        torch.save(saved, tensor)
        numerical["tensor_sha256"] = hashlib.sha256(tensor.read_bytes()).hexdigest()
        monkeypatch.setattr(execution, "TENSOR_SHA256", numerical["tensor_sha256"])
    reference.write_text(json.dumps(numerical))
    monkeypatch.setattr(execution, "REFERENCE_SHA256", hashlib.sha256(reference.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        execution.prepare_direction(execution.ConditionSpec("development", "baseline"), tensor, reference)


def test_publish_never_replaces_including_competing_writers(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    def write(index):
        try:
            return execution.publish_json(path, {"writer": index})
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, [1, 2]))
    assert sum(result is not None for result in results) == 1
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        execution.publish_json(path, {"writer": 3})
    assert path.read_bytes() == original
    assert hashlib.sha256(original).hexdigest() in results
    with pytest.raises(ValueError):
        execution.publish_json(tmp_path / "nan.json", {"invalid": float("nan")})
    assert not (tmp_path / "nan.json").exists()


def test_run_records_before_loading_and_revalidates(directions, tmp_path, monkeypatch, capsys) -> None:
    tensor, reference = directions
    attempt = tmp_path / "attempt"
    spec = execution.ConditionSpec("development", "random43", "pooled", 1.0)
    def load():
        assert (attempt / "execution.json").is_file(), "Execution metadata must precede the first model load"
        return model_pair()
    def evaluate(model, tokenizer, harmless, mmlu, direction, alpha, **kwargs):
        assert len(harmless) == 63 and len(mmlu) == 285
        assert direction is not None and alpha == 1.0
        assert kwargs == {"layer_index": 14, "max_new_tokens": 256}
        return result_for(spec)
    monkeypatch.setattr(execution, "load_qwen", load)
    monkeypatch.setattr(execution, "evaluate_condition", evaluate)
    path = execution.run_condition(spec, tensor_path=tensor, reference_path=reference, attempt_dir=attempt)
    artifact = execution.validate_completed_artifact(path, spec, tensor_path=tensor, reference_path=reference)
    assert artifact["condition"]["random_seed"] == 43
    assert artifact["provenance"]["direction"]["random_seed"] == 43
    assert str(Path(execution.__file__).resolve()) in artifact["provenance"]["inputs"]
    with pytest.raises(FileExistsError):
        execution.run_condition(spec, tensor_path=tensor, reference_path=reference, attempt_dir=attempt)
    altered = deepcopy(artifact)
    altered["provenance"]["direction"]["random_seed"] = 42
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(altered))
    with pytest.raises(ValueError):
        execution.validate_completed_artifact(bad, spec, tensor_path=tensor, reference_path=reference)
    with pytest.raises(ValueError):
        execution.validate_completed_artifact(path, execution.ConditionSpec("final", "baseline"), tensor_path=tensor, reference_path=reference)
    assert capsys.readouterr().out == "", "Runner must keep raw prompts and responses out of stdout"


@pytest.mark.parametrize("kind", ["returned_invalid", "kernel_exception", "model_exception"])
def test_failed_attempts_keep_available_evidence(directions, tmp_path, monkeypatch, kind) -> None:
    tensor, reference = directions
    spec = execution.ConditionSpec("development", "baseline")
    attempt = tmp_path / "failure"
    def load():
        assert (attempt / "execution.json").exists()
        if kind == "model_exception": raise RuntimeError("invented failure")
        return model_pair()
    def evaluate(*args, **kwargs):
        if kind == "kernel_exception": raise RuntimeError("invented failure")
        result = result_for(spec)
        result["mmlu"] = []
        return result
    monkeypatch.setattr(execution, "load_qwen", load)
    monkeypatch.setattr(execution, "evaluate_condition", evaluate)
    with pytest.raises((ValueError, RuntimeError)):
        execution.run_condition(spec, tensor_path=tensor, reference_path=reference, attempt_dir=attempt)
    assert (attempt / "execution.json").exists() and (attempt / "failure.json").exists()
    assert not (attempt / "completed.json").exists()
    assert (attempt / "result.partial.json").exists() is (kind == "returned_invalid")


@pytest.mark.parametrize("value", [True, False, float("nan"), -1.0])
def test_boolean_or_invalid_dose_rejected(value) -> None:
    with pytest.raises(ValueError):
        execution.ConditionSpec("development", "real", "pooled", value)
    with pytest.raises(ValueError):
        execution.final_specs({target: value for target in execution.TARGETS})


@pytest.mark.parametrize("content", ["", " ", "{", '{"schema_version": NaN}', '{"schema_version": 1, "schema_version": 1}', '{}'])
def test_invalid_json_is_not_completed(directions, tmp_path, content) -> None:
    tensor, reference = directions
    path = tmp_path / "incomplete.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        execution.validate_completed_artifact(path, execution.ConditionSpec("development", "baseline"),
                                             tensor_path=tensor, reference_path=reference)


def test_changed_input_blocks_publication(directions, tmp_path, monkeypatch) -> None:
    tensor, reference = directions
    spec = execution.ConditionSpec("development", "baseline")
    attempt = tmp_path / "changed"
    monkeypatch.setattr(execution, "load_qwen", model_pair)
    def evaluate(*args, **kwargs):
        reference.write_bytes(reference.read_bytes() + b"\n")
        return result_for(spec)
    monkeypatch.setattr(execution, "evaluate_condition", evaluate)
    with pytest.raises(ValueError, match="Inputs changed"):
        execution.run_condition(spec, tensor_path=tensor, reference_path=reference, attempt_dir=attempt)
    assert (attempt / "result.partial.json").exists()
    assert not (attempt / "completed.json").exists()


def test_missing_panel_rejected_before_model_load(directions, tmp_path, monkeypatch) -> None:
    tensor, reference = directions
    items, questions = load_condition_items("development")
    monkeypatch.setattr(execution, "load_condition_items", lambda role: (items[:-1], questions))
    def forbidden_model_load():
        pytest.fail("Panel validation must precede model loading")
    monkeypatch.setattr(execution, "load_qwen", forbidden_model_load)
    with pytest.raises(ValueError, match="Frozen loader"):
        execution.run_condition(execution.ConditionSpec("development", "baseline"), tensor_path=tensor,
                                reference_path=reference, attempt_dir=tmp_path / "missing")


def test_tiny_negative_kl_keeps_existing_floating_point_contract() -> None:
    spec = execution.ConditionSpec("development", "real", "pooled", 1.0)
    result = result_for(spec)
    next(row for row in result["harmless"] if row["quality_subset"])["prompt_kl"] = -1e-8
    execution.validate_condition_result(result, spec)


def test_final_pooled_dose_is_fixed() -> None:
    with pytest.raises(ValueError, match="pooled dose"):
        execution.ConditionSpec("final", "real", "pooled", 0.5)
