"""Run one reversed, frozen final dose on the complete MMLU panel only.

Production invokes this file by absolute filename while PYTHONPATH and cwd retain
the positive frozen checkout. The caller has its own commit; all package imports
resolve to the unchanged positive endpoint. The launcher owns attempt admission.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from activation_steering_study.data import mmlu as mmlu_data
from activation_steering_study.evaluation import mmlu
from activation_steering_study.steering import harmless_execution as positive
from activation_steering_study.utils.qwen import load_qwen

CORE_COMMIT = "249d0cd148d8e49fe1fd290012258ed618454f13"
POSITIVE_EXECUTION_SHA256 = "e3fe527567d4611454c6c95db6e1039f636a59940eb6ce8fa7f871a7638b657a"


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    def invalid(value):
        raise ValueError("Nonfinite JSON number")

    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def checked_repository(root: Path, commit: str) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root)
    if head != commit or dirty:
        raise ValueError("Frozen repository commit or clean state differs")


def condition_metadata(template: positive.ConditionSpec) -> dict[str, Any]:
    if template.role != "final" or template.control == "baseline":
        raise ValueError("Negative MMLU requires a nonbaseline positive-final template")
    return {**template.to_dict(), "condition_id": f"negative-mmlu-{template.target}-{template.control}",
            "alpha": -template.alpha, "endpoint": "mmlu", "layer_index": 14}


def negative_direction(template: positive.ConditionSpec, tensor: Path, reference: Path):
    condition_metadata(template)
    vector, provenance = positive.prepare_direction(template, tensor, reference)
    # Keep the raw vector bytes and positive norm magnitude; only the multiplier changes sign.
    return vector, {**provenance, "alpha": -template.alpha}


def shared_runtime(tokenizer=None, model=None) -> dict[str, Any]:
    """Version-pinned bridge to the positive worker's exact runtime comparison.

    These private helpers are deliberately reused to preserve the original runtime
    contract without editing or duplicating the frozen implementation.
    """
    if file_sha(Path(positive.__file__)) != POSITIVE_EXECUTION_SHA256:
        raise ValueError("Shared positive runtime bridge source differs")
    return positive._environment() if model is None else positive._runtime(tokenizer, model)


def validate_request(request: dict[str, Any]) -> positive.ConditionSpec:
    """Reject drift and derive dose/control identity from the sealed positive records."""
    fields = {"schema_version", "configuration_sha256", "condition_index", "condition", "direction",
              "tensor", "reference", "dose_freeze", "baseline", "integrity", "caller", "core",
              "environment", "runtime", "import_origins"}
    if set(request) != fields or type(request["schema_version"]) is not int or request["schema_version"] != 1:
        raise ValueError("Invalid negative request schema")
    for filename, digest in request["integrity"].items():
        if file_sha(Path(filename)) != digest:
            raise ValueError("Frozen input, positive record or caller bytes differ")
    for name, expected_commit in (("caller", request["caller"]["commit"]), ("core", CORE_COMMIT)):
        source = request[name]
        if source["commit"] != expected_commit:
            raise ValueError("Shared positive commit differs")
        checked_repository(Path(source["root"]), expected_commit)
    if (Path(request["caller"]["path"]).resolve() != Path(__file__).resolve()
            or request["integrity"].get(str(Path(__file__).resolve())) != file_sha(Path(__file__))
            or Path.cwd().resolve() != Path(request["core"]["root"])
            or dict(os.environ) != request["environment"]
            or shared_runtime() != request["runtime"]["environment"]):
        raise ValueError("Caller, cwd, environment or runtime differs")
    for name, filename in request["import_origins"].items():
        module = importlib.import_module(name)
        module_path = getattr(module, "__file__", None)
        if not module_path or str(Path(module_path).resolve()) != filename:
            raise ValueError("Shared positive import origin differs")
    origins = {name: str(Path(getattr(module, "__file__")).resolve()) for name, module in tuple(sys.modules.items())
               if name.startswith("activation_steering_study") and getattr(module, "__file__", None)}
    if origins != request["import_origins"]:
        raise ValueError("Unexpected package source outside the shared positive closure")
    for name in ("tensor", "reference", "dose_freeze", "baseline"):
        if request[name] not in request["integrity"]:
            raise ValueError("Required input lacks a frozen hash")
    dose_freeze = read_json(Path(request["dose_freeze"]))
    queue = positive.final_specs(dose_freeze["doses"])[1:]
    index = request["condition_index"]
    if type(index) is not int or not 0 <= index < len(queue):
        raise ValueError("Condition index outside the fixed negative queue")
    template = queue[index]
    _, direction = negative_direction(template, Path(request["tensor"]), Path(request["reference"]))
    if request["condition"] != condition_metadata(template) or request["direction"] != direction:
        raise ValueError("Condition does not match the frozen dose and actual vector")
    baseline = positive.validate_completed_artifact(Path(request["baseline"]), positive.ConditionSpec("final", "baseline"),
                tensor_path=Path(request["tensor"]), reference_path=Path(request["reference"]))
    if baseline["provenance"]["runtime"] != request["runtime"]:
        raise ValueError("Baseline runtime is incompatible with the negative endpoint")
    return template


def evaluate_mmlu(model, tokenizer, questions: list[mmlu_data.MmluItem], direction: torch.Tensor,
                  alpha: float) -> dict[str, Any]:
    """Score supplied questions; production uses only load_mmlu_final's complete panel."""
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha >= 0:
        raise ValueError("The actual MMLU multiplier must be finite and negative")
    started = time.monotonic()
    rows = []
    for question in questions:
        tick = time.monotonic()
        score = mmlu.score_question(tokenizer, model, mmlu.render_question(question), 14, direction, alpha)
        gold = mmlu_data.LABELS[question["answer"]]
        rows.append({"id": question["id"], "subject": question["subject"], "source_index": question["source_index"],
                     "gold": gold, **score, "correct": score["prediction"] == gold,
                     "scoring_seconds": time.monotonic() - tick})
    return {"mmlu": rows, "timing_seconds": {"mmlu": sum(row["scoring_seconds"] for row in rows),
                                            "total": time.monotonic() - started}}


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite nonnegative number")
    return value


def validate_result(result: Any) -> None:
    questions = mmlu_data.load_mmlu_final()
    if (type(result) is not dict or set(result) != {"mmlu", "timing_seconds"}
            or type(result["mmlu"]) is not list or len(result["mmlu"]) != 1710
            or len(questions) != 1710 or len({item["id"] for item in questions}) != 1710):
        raise ValueError("Complete ordered 1710-item MMLU panel required")
    for row, item in zip(result["mmlu"], questions, strict=True):
        if (type(row) is not dict or set(row) != {"id", "subject", "source_index", "gold", "prediction",
                "probabilities_abcd", "option_mass", "correct", "scoring_seconds"}
                or row["id"] != item["id"] or row["subject"] != item["subject"]
                or type(row["source_index"]) is not int or row["source_index"] != item["source_index"]
                or row["gold"] != mmlu_data.LABELS[item["answer"]] or type(row["correct"]) is not bool):
            raise ValueError("MMLU row identity, label or schema differs")
        values = row["probabilities_abcd"]
        if type(values) is not list or len(values) != 4 or any(_number(value) > 1 for value in values):
            raise ValueError("Invalid full-vocabulary A-D probabilities")
        mass = _number(row["option_mass"])
        if (mass > 1 + 1e-6 or not math.isclose(sum(values), mass, rel_tol=1e-7, abs_tol=1e-7)
                or row["prediction"] != mmlu_data.LABELS[max(range(4), key=values.__getitem__)]
                or row["correct"] != (row["prediction"] == row["gold"])):
            raise ValueError("MMLU probability, prediction or correctness differs")
        _number(row["scoring_seconds"])
    timing = result["timing_seconds"]
    if type(timing) is not dict or set(timing) != {"mmlu", "total"}:
        raise ValueError("Invalid endpoint timing")
    if (not math.isclose(_number(timing["mmlu"]), sum(row["scoring_seconds"] for row in result["mmlu"]),
                         rel_tol=1e-9, abs_tol=1e-9) or _number(timing["total"]) + 1e-6 < timing["mmlu"]):
        raise ValueError("Aggregate MMLU timing differs")


def validate_completed(path: Path, request: dict[str, Any]) -> dict[str, Any]:
    if (path.parent / "failure.json").exists():
        raise ValueError("Failed attempt cannot enter scientific evidence")
    request_path = path.parent.parent / "request.json"
    execution = read_json(path.parent / "execution.json")
    if (execution.get("request") != request or read_json(request_path) != request
            or execution.get("request_sha256") != file_sha(request_path)):
        raise ValueError("Execution record differs from the sealed request")
    artifact = read_json(path)
    if (type(artifact) is not dict or set(artifact) != {"schema_version", "request", "runtime", "result"}
            or type(artifact["schema_version"]) is not int or artifact["schema_version"] != 1
            or artifact["request"] != request or artifact["runtime"] != request["runtime"]):
        raise ValueError("Completed artifact provenance differs")
    validate_result(artifact["result"])
    return artifact


def run_condition(request_path: Path, request_sha256: str, attempt_dir: Path) -> Path:
    attempt_dir.mkdir(parents=True, exist_ok=False)
    stage = "preparation"
    try:
        if (request_path.resolve() != (attempt_dir.parent / "request.json").resolve()
                or file_sha(request_path) != request_sha256):
            raise ValueError("Execution request hash differs")
        request = read_json(request_path)
        template = validate_request(request)
        direction, _ = negative_direction(template, Path(request["tensor"]), Path(request["reference"]))
        if direction is None:
            raise ValueError("Negative endpoint requires an actual vector")
        questions = mmlu_data.load_mmlu_final()
        positive.publish_json(attempt_dir / "execution.json", {
            "schema_version": 1, "request_sha256": request_sha256, "request": request,
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        stage = "model_loading"
        tokenizer, model = load_qwen()
        runtime = shared_runtime(tokenizer, model)
        if runtime != request["runtime"]:
            raise ValueError("Loaded endpoint differs from the positive-final baseline")
        stage = "evaluation"
        result = evaluate_mmlu(model, tokenizer, questions, direction, -template.alpha)
        with (attempt_dir / "result.partial.json").open("x") as stream:
            json.dump(result, stream, allow_nan=True)
            stream.flush()
            os.fsync(stream.fileno())
        stage = "validation"
        validate_result(result)
        validate_request(request)
        if file_sha(request_path) != request_sha256:
            raise ValueError("Execution request changed during scoring")
        artifact = {"schema_version": 1, "request": request, "runtime": runtime, "result": result}
        output = attempt_dir / "completed.json"
        stage = "publication"
        positive.publish_json(output, artifact)
        validate_completed(output, request)
        return output
    except BaseException as error:
        positive.publish_json(attempt_dir / "failure.json", {
            "schema_version": 1, "stage": stage, "error_type": type(error).__name__,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_condition(args.request, args.request_sha256, args.attempt_dir)
    except Exception as error:
        raise SystemExit(f"Negative MMLU failed ({type(error).__name__}); inspect retained records") from None
    print(f"Completed {result}")


if __name__ == "__main__":
    main()
