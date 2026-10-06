"""Synthetic endpoint checks; official target text is neither generated nor printed."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest
import torch

from activation_steering_study.steering import negative_mmlu as negative
from activation_steering_study.steering.test_harmless_execution import directions, model_pair, result_for


def test_signed_alpha_and_unchanged_vector_reach_the_owned_scorer(monkeypatch):
    vector = torch.tensor([2.0, -3.0])
    question: negative.mmlu_data.MmluItem = {"id": "invented:0", "subject": "arithmetic", "source_index": 0,
                "question": "How many corners does a square have?", "choices": ["Four", "Two", "One", "Three"], "answer": 0}
    calls = []
    def score(tokenizer, model, content, layer_index, direction, alpha):
        calls.append((layer_index, direction, alpha))
        return {"probabilities_abcd": [0.4, 0.2, 0.1, 0.1], "prediction": "A", "option_mass": 0.8}
    monkeypatch.setattr(negative.mmlu, "score_question", score)
    result = negative.evaluate_mmlu(None, None, [question], vector, -2.0)
    assert calls == [(14, vector, -2.0)], "The evaluator must pass a negative scalar and retain the vector orientation"
    assert result["mmlu"][0]["correct"] is True
    for alpha in (0, 1, float("nan"), False):
        with pytest.raises(ValueError):
            negative.evaluate_mmlu(None, None, [question], vector, alpha)


def test_reversed_doses_preserve_each_actual_seed_and_norm(directions):
    tensor, reference = directions
    doses: dict[str, float] = {target: 1.0 for target in negative.positive.TARGETS}
    doses["cyber_intrusion"] = 2.0
    templates = negative.positive.final_specs(doses)[1:]
    assert len(templates) == 12
    for target in negative.positive.TARGETS:
        hashes = []
        for template in (spec for spec in templates if spec.target == target):
            positive_vector, original = negative.positive.prepare_direction(template, tensor, reference)
            vector, observed = negative.negative_direction(template, tensor, reference)
            assert vector is not None and positive_vector is not None
            assert torch.equal(vector, positive_vector), "A reversed condition must not negate or resample the vector"
            assert observed == {**original, "alpha": -template.alpha}
            assert observed["injected_norm"] == abs(observed["alpha"]) * observed["vector_norm"]
            assert negative.condition_metadata(template)["random_seed"] == template.random_seed
            hashes.append(observed["vector_sha256"])
        assert len(set(hashes)) == 3
    with pytest.raises(ValueError):
        negative.condition_metadata(negative.positive.ConditionSpec("final", "baseline"))


def valid_result() -> dict[str, Any]:
    result = result_for(negative.positive.ConditionSpec("final", "baseline"))
    return {"mmlu": result["mmlu"], "timing_seconds": {"mmlu": 0.0, "total": 0.0}}


@pytest.mark.parametrize("mutation", ["subset", "duplicate", "order", "id", "gold", "subject", "index",
                                        "nan", "bool", "mass", "prediction", "correct", "time", "extra"])
def test_complete_mmlu_validation_rejects_mutation(mutation):
    result = valid_result()
    negative.validate_result(result)
    row = result["mmlu"][0]
    if mutation == "subset": result["mmlu"].pop()
    elif mutation == "duplicate": result["mmlu"][1] = deepcopy(row)
    elif mutation == "order": result["mmlu"][0], result["mmlu"][1] = result["mmlu"][1], row
    elif mutation == "id": row["id"] = "invented"
    elif mutation == "gold": row["gold"] = "invalid"
    elif mutation == "subject": row["subject"] = "invented"
    elif mutation == "index": row["source_index"] = True
    elif mutation == "nan": row["probabilities_abcd"][0] = float("nan")
    elif mutation == "bool": row["probabilities_abcd"][0] = True
    elif mutation == "mass": row["option_mass"] = 0.2
    elif mutation == "prediction": row["prediction"] = "B"
    elif mutation == "correct": row["correct"] = 1
    elif mutation == "time": result["timing_seconds"]["mmlu"] = 1.0
    elif mutation == "extra": result["harmless"] = []
    with pytest.raises(ValueError):
        negative.validate_result(result)


@pytest.fixture
def request_fixture(directions, monkeypatch, tmp_path):
    tensor, reference = directions
    baseline = tmp_path / "baseline.json"
    dose = tmp_path / "dose.json"
    baseline.write_text("{}")
    dose.write_text(json.dumps({"doses": dict.fromkeys(negative.positive.TARGETS, 1.0)}))
    runtime = {"environment": {"fixture": True}}
    monkeypatch.setattr(negative, "checked_repository", lambda *args: None)
    monkeypatch.setattr(negative, "shared_runtime", lambda *args: runtime if args else runtime["environment"])
    monkeypatch.setattr(negative.positive, "validate_completed_artifact", lambda *args, **kwargs: {"provenance": {"runtime": runtime}})
    template = negative.positive.final_specs(dict.fromkeys(negative.positive.TARGETS, 1.0))[2]
    _, direction = negative.negative_direction(template, tensor, reference)
    caller = str(Path(negative.__file__).resolve())
    origins = {name: str(Path(getattr(module, "__file__")).resolve()) for name, module in tuple(sys.modules.items())
               if name.startswith("activation_steering_study") and getattr(module, "__file__", None)}
    request = {"schema_version": 1, "configuration_sha256": "invented", "condition_index": 1,
               "condition": negative.condition_metadata(template), "direction": direction,
               "tensor": str(tensor), "reference": str(reference), "dose_freeze": str(dose), "baseline": str(baseline),
               "integrity": {str(path): negative.file_sha(path) for path in (tensor, reference, dose, baseline, Path(caller))},
               "caller": {"root": str(Path.cwd()), "commit": "invented", "path": caller},
               "core": {"root": str(Path.cwd()), "commit": negative.CORE_COMMIT},
               "environment": dict(os.environ), "runtime": runtime, "import_origins": origins}
    return request


@pytest.mark.parametrize("mutation", ["baseline", "dose", "code", "environment", "runtime", "origin", "core", "seed", "alpha", "index"])
def test_request_rejects_frozen_provenance_or_identity_drift(request_fixture, mutation):
    request = request_fixture
    request["environment"] = dict(os.environ)
    assert negative.validate_request(request).random_seed == 42
    if mutation in ("baseline", "dose"):
        Path(request["baseline" if mutation == "baseline" else "dose_freeze"]).write_text("changed")
    elif mutation == "code": request["integrity"][request["caller"]["path"]] = "changed"
    elif mutation == "environment": request["environment"]["OMP_NUM_THREADS"] = "7"
    elif mutation == "runtime": request["runtime"] = {"environment": {"fixture": False}}
    elif mutation == "origin": request["import_origins"]["activation_steering_study.evaluation.mmlu"] = "/invented/mmlu.py"
    elif mutation == "core": request["core"]["commit"] = "changed"
    elif mutation == "seed": request["condition"]["random_seed"] = 43
    elif mutation == "alpha": request["condition"]["alpha"] = 1.0
    elif mutation == "index": request["condition_index"] = True
    with pytest.raises(ValueError):
        negative.validate_request(request)


def test_execution_precedes_model_and_publication_cannot_overwrite(request_fixture, monkeypatch, tmp_path):
    request = request_fixture
    request["environment"] = dict(os.environ)
    request_path = tmp_path / "request.json"
    negative.positive.publish_json(request_path, request)
    output = tmp_path / "attempt"
    def load():
        assert (output / "execution.json").exists(), "Execution metadata must be durable before model loading"
        return model_pair()
    observed = []
    def evaluate(model, tokenizer, questions, direction, alpha):
        observed.append((len(questions), alpha, negative.file_sha(Path(request["tensor"]))))
        return valid_result()
    monkeypatch.setattr(negative, "load_qwen", load)
    monkeypatch.setattr(negative, "evaluate_mmlu", evaluate)
    completed = negative.run_condition(request_path, negative.file_sha(request_path), output)
    assert observed == [(1710, -1.0, request["integrity"][request["tensor"]])]
    artifact = negative.validate_completed(completed, request)
    assert artifact["request"]["condition"]["alpha"] == -1.0
    original = completed.read_bytes()
    with pytest.raises(FileExistsError):
        negative.run_condition(request_path, negative.file_sha(request_path), output)
    assert completed.read_bytes() == original
    negative.positive.publish_json(output / "failure.json", {"stage": "invented_failure"})
    with pytest.raises(ValueError, match="Failed attempt"):
        negative.validate_completed(completed, request)


def test_failed_model_load_retains_execution_and_failure(request_fixture, monkeypatch, tmp_path):
    request_fixture["environment"] = dict(os.environ)
    request_path = tmp_path / "request.json"
    negative.positive.publish_json(request_path, request_fixture)
    def fail():
        raise RuntimeError("invented model failure")
    monkeypatch.setattr(negative, "load_qwen", fail)
    with pytest.raises(RuntimeError):
        negative.run_condition(request_path, negative.file_sha(request_path), tmp_path / "failed")
    assert negative.read_json(tmp_path / "failed/failure.json")["stage"] == "model_loading"
    assert (tmp_path / "failed/execution.json").exists()
    assert not (tmp_path / "failed/completed.json").exists()
