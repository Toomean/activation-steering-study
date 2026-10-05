"""Execute and publish one positive harmless/MMLU condition without replacing evidence.

The launcher owns the accepted run freeze, dose freeze, attempt admission and retry policy.
The unchanged whole-condition kernel has no checkpoints: an exception during evaluation
retains execution/failure records, but its internal in-memory rows cannot be recovered.
"""

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Literal, TypedDict, cast

import torch

from activation_steering_study.data import harmless as harmless_data, mmlu as mmlu_data
from activation_steering_study.data.harmless import HarmlessRole
from activation_steering_study.steering.harmless_run import (
    ConditionResult, evaluate_condition, load_condition_items, refusal_phrase_matches,
)
from activation_steering_study.steering.random_control import sample_random_direction
from activation_steering_study.utils.qwen import MODEL_ID, MODEL_REVISION, load_qwen

Target = Literal["pooled", "cyber_intrusion", "dangerous_substances", "disinformation"]
Control = Literal["baseline", "real", "random42", "random43"]
TARGETS: tuple[Target, ...] = ("pooled", "cyber_intrusion", "dangerous_substances", "disinformation")
DOSES = (0.5, 1.0, 2.0)
TENSOR_SHA256 = "28150fee085e3e0b0f2aa0422d8a414a654251be593a61e26b359010855893da"
REFERENCE_SHA256 = "81547047c5082bb19dba0ad5ca1c2f41d0d9dfcde607ba7ad4c5dfe903eff18f"
_SOURCE_MODULES = (
    "data.harmless", "data.mmlu", "evaluation.coherence", "evaluation.mmlu",
    "evaluation.refusal", "evaluation.scoring", "steering.harmless_run",
    "steering.random_control", "steering.pilot", "steering.generation",
    "steering.intervention", "utils.choice_prompt", "utils.qwen",
)


@dataclass(frozen=True)
class ConditionSpec:
    role: HarmlessRole
    control: Control
    target: Target | None = None
    alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.role not in ("development", "final") or self.control not in (
            "baseline", "real", "random42", "random43"
        ):
            raise ValueError("Unknown condition role or control")
        _number(self.alpha, "alpha", minimum=0)
        if self.control == "baseline":
            if self.target is not None or self.alpha != 0:
                raise ValueError("Baseline requires no target and alpha zero")
        elif self.target not in TARGETS or self.alpha not in DOSES:
            raise ValueError("Positive conditions require a frozen target and dose")
        elif self.role == "final" and self.target == "pooled" and self.alpha != 1.0:
            raise ValueError("Positive final pooled dose is fixed at +1")

    @property
    def random_seed(self) -> int | None:
        return {"random42": 42, "random43": 43}.get(self.control)

    @property
    def condition_id(self) -> str:
        if self.control == "baseline":
            return f"{self.role}-baseline"
        prefix = f"{self.role}-{self.target}-{self.control}"
        return prefix if self.role == "final" else f"{prefix}-a{str(float(self.alpha)).removesuffix('.0').replace('.', 'p')}"

    def to_dict(self) -> "ConditionMetadata":
        return {"condition_id": self.condition_id, "role": self.role, "control": self.control,
                "target": self.target, "alpha": self.alpha, "random_seed": self.random_seed}


class ConditionMetadata(TypedDict):
    condition_id: str
    role: HarmlessRole
    control: Control
    target: Target | None
    alpha: float
    random_seed: int | None


class DirectionProvenance(TypedDict):
    tensor_sha256: str
    reference_sha256: str
    target: Target | None
    random_seed: int | None
    vector_sha256: str | None
    raw_norm: float
    vector_norm: float
    injected_norm: float
    alpha: float


class Provenance(TypedDict):
    inputs: dict[str, str]
    direction: DirectionProvenance
    runtime: dict[str, Any]


class CompletedArtifact(TypedDict):
    """Public mapping: condition identifies the intervention; result is the unchanged kernel output."""
    schema_version: int
    condition: ConditionMetadata
    provenance: Provenance
    result: ConditionResult


def development_specs() -> tuple[ConditionSpec, ...]:
    return (ConditionSpec("development", "baseline"), ConditionSpec("development", "real", "pooled", 1.0),
            *(ConditionSpec("development", "real", target, alpha)
              for target in TARGETS[1:] for alpha in DOSES))


def final_specs(doses: Mapping[str, float]) -> tuple[ConditionSpec, ...]:
    if set(doses) != set(TARGETS) or doses["pooled"] != 1.0:
        raise ValueError("Final doses require all four targets and pooled +1")
    for value in doses.values():
        _number(value, "dose", minimum=0)
        if value not in DOSES:
            raise ValueError("Final dose was not tested")
    return (ConditionSpec("final", "baseline"),
            *(ConditionSpec("final", control, target, doses[target])
              for target in TARGETS for control in cast(tuple[Control, ...], ("real", "random42", "random43"))))


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid finite numeric field: {name}")
    return float(value)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid integer field: {name}")
    return value


def _object(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"Invalid fields: {name}")
    return value


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: Any, *, allow_nan: bool = False) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=allow_nan) + "\n").encode()


def _read_json(path: Path) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Any:
        raise ValueError("Nonfinite JSON number")

    content = path.read_bytes()
    if not content.strip():
        raise ValueError("Empty JSON artifact")
    return json.loads(content, object_pairs_hook=unique, parse_constant=invalid_constant)


def _spec_from_metadata(value: Any) -> ConditionSpec:
    metadata = _object(value, {"condition_id", "role", "control", "target", "alpha", "random_seed"}, "condition")
    spec = ConditionSpec(metadata["role"], metadata["control"], metadata["target"], metadata["alpha"])
    if metadata["random_seed"] is not None:
        _integer(metadata["random_seed"], "random_seed")
    if metadata != spec.to_dict():
        raise ValueError("Condition identity or actual random seed differs")
    return spec


def prepare_direction(spec: ConditionSpec, tensor_path: Path, reference_path: Path) -> tuple[torch.Tensor | None, DirectionProvenance]:
    """Read only the existing direction mapping, checking every frozen raw norm first."""
    tensor_bytes, reference_bytes = tensor_path.read_bytes(), reference_path.read_bytes()
    if _sha(tensor_bytes) != TENSOR_SHA256 or _sha(reference_bytes) != REFERENCE_SHA256:
        raise ValueError("Historical tensor or numerical reference digest differs")
    reference = json.loads(reference_bytes)
    if (reference.get("tensor_sha256") != TENSOR_SHA256 or reference.get("model_id") != MODEL_ID
            or reference.get("model_revision") != MODEL_REVISION or reference.get("model_dtype") != "torch.bfloat16"
            or type(reference.get("layer_index")) is not int or reference["layer_index"] != 14):
        raise ValueError("Numerical reference model, tensor or layer differs")
    norms = reference["direction_norms"]
    if set(norms) != set(TARGETS):
        raise ValueError("Numerical reference directions differ")
    saved = torch.load(io.BytesIO(tensor_bytes), map_location="cpu", weights_only=True)
    directions = saved["directions"]
    if type(directions) is not dict or set(directions) != set(TARGETS):
        raise ValueError("Historical direction mapping differs")
    for target in TARGETS:
        vector = directions[target]
        norm = _number(norms[target], "raw reference norm", minimum=0)
        if (not isinstance(vector, torch.Tensor) or vector.dtype != torch.float32 or vector.shape != (1536,)
                or not bool(torch.isfinite(vector).all()) or norm <= 0
                or not math.isclose(float(vector.norm().item()), norm, rel_tol=1e-6, abs_tol=1e-6)):
            raise ValueError("Historical vector or raw norm differs")
    direction = None
    raw_norm = vector_norm = 0.0
    vector_sha = None
    if spec.target is not None:
        direction = directions[spec.target].contiguous()
        raw_norm = float(direction.norm().item())
        if spec.random_seed is not None:
            direction = sample_random_direction(direction.numel(), raw_norm, spec.random_seed)
        vector_norm = float(direction.norm().item())
        if not bool(torch.isfinite(direction).all()) or not math.isclose(vector_norm, raw_norm, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Control norm differs from its raw target norm")
        vector_sha = _sha(direction.numpy().tobytes())
    return direction, {"tensor_sha256": TENSOR_SHA256, "reference_sha256": REFERENCE_SHA256,
                       "target": spec.target, "random_seed": spec.random_seed, "vector_sha256": vector_sha,
                       "raw_norm": raw_norm, "vector_norm": vector_norm,
                       "injected_norm": spec.alpha * vector_norm, "alpha": spec.alpha}


def capture_inputs(role: HarmlessRole, tensor_path: Path, reference_path: Path) -> dict[str, str]:
    """Hash panels and direct execution sources; the launcher freezes the complete import closure."""
    paths = [tensor_path, reference_path, Path("uv.lock"), harmless_data.MANIFEST_PATH,
             harmless_data.SOURCE_ROOT / harmless_data.SOURCE_NAMES[role], Path(__file__)]
    paths.extend([mmlu_data.SOURCE_PATH] if role == "development" else [mmlu_data.FINAL_SOURCE_PATH, mmlu_data.FINAL_MANIFEST_PATH])
    for name in _SOURCE_MODULES:
        module = importlib.import_module(f"activation_steering_study.{name}")
        if module.__file__ is None:
            raise ValueError("Execution source has no file identity")
        paths.append(Path(module.__file__))
    return {str(path.resolve()): _sha(path.read_bytes()) for path in paths}


def _environment() -> dict[str, Any]:
    return {"python": platform.python_version(), "executable": str(Path(sys.executable).resolve()),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "pyarrow")},
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}


def _runtime(tokenizer: Any, model: Any) -> dict[str, Any]:
    # Transformers consumes revision kwargs; retained resolved file paths identify the snapshot.
    revisions = set()
    for key, value in tokenizer.init_kwargs.items():
        if key.endswith("_file") and isinstance(value, str):
            parts = Path(value).parts
            if "snapshots" in parts and len(parts) > parts.index("snapshots") + 1:
                revisions.add(parts[parts.index("snapshots") + 1])
    if len(revisions) != 1:
        raise ValueError("Tokenizer resolved snapshot revision unavailable or inconsistent")
    runtime = {"model_id": model.config._name_or_path, "model_revision": model.config._commit_hash,
               "tokenizer_id": tokenizer.name_or_path,
               "tokenizer_revision": revisions.pop(),
               "dtype": str(model.dtype), "device": str(model.device),
               "attention_implementation": model.config._attn_implementation,
               "hidden_size": model.config.hidden_size, "training": model.training,
               "environment": _environment()}
    _validate_runtime(runtime)
    return runtime


def _validate_runtime(value: Any) -> None:
    runtime = _object(value, {"model_id", "model_revision", "tokenizer_id", "tokenizer_revision", "dtype", "device",
                             "attention_implementation", "hidden_size", "training", "environment"}, "runtime")
    if (runtime["model_id"] != MODEL_ID or runtime["tokenizer_id"] != MODEL_ID
            or runtime["model_revision"] != MODEL_REVISION or runtime["tokenizer_revision"] != MODEL_REVISION
            or runtime["dtype"] != "torch.bfloat16" or runtime["device"] != "cpu" or runtime["training"] is not False
            or _integer(runtime["hidden_size"], "hidden_size") != 1536
            or type(runtime["attention_implementation"]) is not str or not runtime["attention_implementation"]):
        raise ValueError("Runtime model/tokenizer contract differs")
    environment = _object(runtime["environment"], set(_environment()), "environment")
    for field in ("torch_threads", "torch_interop_threads"):
        _integer(environment[field], field, minimum=1)
    if environment != _environment():
        raise ValueError("Runtime versions or thread settings differ")


def _load_frozen_items(role: HarmlessRole) -> tuple[list[harmless_data.HarmlessItem], list[mmlu_data.MmluItem]]:
    items, questions = load_condition_items(role)
    expected_counts = harmless_data.COUNTS[role]
    if ((len(items), len({item["semantic_group_id"] for item in items}), sum(item["quality_subset"] for item in items)) != expected_counts
            or len(questions) != {"development": 285, "final": 1710}[role]
            or len({item["row_id"] for item in items}) != len(items)
            or len({question["id"] for question in questions}) != len(questions)
            or any(item["role"] != role for item in items)):
        raise ValueError("Frozen loader panel contract differs")
    return items, questions


def validate_condition_result(value: Any, spec: ConditionSpec) -> ConditionResult:
    """Validate every mandatory endpoint against the requested role's frozen loader order."""
    result = _object(value, {"schema_version", "role", "baseline", "alpha", "layer_index", "max_new_tokens",
                             "harmless", "mmlu", "timing_seconds"}, "result")
    if (_integer(result["schema_version"], "schema_version") != 1 or result["role"] != spec.role
            or result["baseline"] is not (spec.control == "baseline")
            or _number(result["alpha"], "alpha") != spec.alpha
            or _integer(result["layer_index"], "layer_index") != 14
            or _integer(result["max_new_tokens"], "max_new_tokens") != 256):
        raise ValueError("Condition result settings differ")
    items, questions = _load_frozen_items(spec.role)
    harmless, mmlu = result["harmless"], result["mmlu"]
    if type(harmless) is not list or type(mmlu) is not list or len(harmless) != len(items) or len(mmlu) != len(questions):
        raise ValueError("Missing required panel rows")
    for row, item in zip(harmless, items, strict=True):
        row = _object(row, {"row_id", "semantic_group_id", "quality_subset", "refusal_proxy", "phrase_matches", "completion",
                            "prompt_token_ids", "generated_token_ids", "completion_token_count", "completion_token_cap",
                            "response_likelihood", "prompt_kl", "generation_seconds", "quality_seconds"}, "harmless row")
        if (row["row_id"] != item["row_id"] or row["semantic_group_id"] != item["semantic_group_id"]
                or row["quality_subset"] is not item["quality_subset"] or type(row["refusal_proxy"]) is not bool
                or type(row["completion"]) is not str):
            raise ValueError("Harmless identity, grouping or membership differs")
        count = _integer(row["completion_token_count"], "completion_token_count")
        if _integer(row["completion_token_cap"], "completion_token_cap") != 256 or count > 256:
            raise ValueError("Completion cap differs")
        for field in ("prompt_token_ids", "generated_token_ids"):
            if type(row[field]) is not list or any(type(token) is not int or token < 0 for token in row[field]):
                raise ValueError("Invalid token IDs")
        if not row["prompt_token_ids"] or count != len(row["generated_token_ids"]):
            raise ValueError("Token count differs")
        if count == 0 and row["completion"]:
            raise ValueError("Nonempty completion without generated tokens")
        matches = refusal_phrase_matches(row["completion"])
        if type(row["phrase_matches"]) is not list:
            raise ValueError("Invalid phrase matches")
        for match in row["phrase_matches"]:
            match = _object(match, {"phrase", "start", "end"}, "phrase match")
            _integer(match["start"], "phrase start")
            _integer(match["end"], "phrase end")
        if row["phrase_matches"] != matches or row["refusal_proxy"] != bool(matches):
            raise ValueError("Phrase/refusal metadata differs")
        for field in ("generation_seconds", "quality_seconds"):
            _number(row[field], field, minimum=0)
        if not item["quality_subset"]:
            if row["response_likelihood"] is not None or row["prompt_kl"] is not None:
                raise ValueError("Quality diagnostics outside membership")
        else:
            kl = _number(row["prompt_kl"], "prompt_kl")
            if spec.control == "baseline" and kl != 0:
                raise ValueError("Baseline KL differs from zero")
            likelihood = _object(row["response_likelihood"], {"token_count", "mean_nll", "perplexity"}, "likelihood")
            if _integer(likelihood["token_count"], "likelihood token_count") != count:
                raise ValueError("Likelihood token count differs")
            if count == 0:
                if likelihood["mean_nll"] is not None or likelihood["perplexity"] is not None:
                    raise ValueError("Zero-token likelihood must be unavailable")
            else:
                nll = _number(likelihood["mean_nll"], "mean_nll", minimum=0)
                perplexity = _number(likelihood["perplexity"], "perplexity", minimum=1)
                if not math.isclose(math.log(perplexity), nll, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError("NLL and perplexity differ")
    for row, question in zip(mmlu, questions, strict=True):
        row = _object(row, {"id", "subject", "source_index", "gold", "prediction", "probabilities_abcd", "option_mass", "correct", "scoring_seconds"}, "MMLU row")
        if (row["id"] != question["id"] or row["subject"] != question["subject"]
                or _integer(row["source_index"], "source_index") != question["source_index"]
                or row["gold"] != mmlu_data.LABELS[question["answer"]] or type(row["correct"]) is not bool):
            raise ValueError("MMLU identity or gold differs")
        probabilities = row["probabilities_abcd"]
        if type(probabilities) is not list or len(probabilities) != 4:
            raise ValueError("Missing MMLU probabilities")
        values = [_number(probability, "probability", minimum=0) for probability in probabilities]
        mass = _number(row["option_mass"], "option_mass", minimum=0)
        if (any(value > 1 for value in values) or mass > 1 + 1e-6
                or not math.isclose(sum(values), mass, rel_tol=1e-7, abs_tol=1e-7)
                or row["prediction"] != mmlu_data.LABELS[max(range(4), key=values.__getitem__)]
                or row["correct"] != (row["prediction"] == row["gold"])):
            raise ValueError("MMLU probability, prediction or correctness differs")
        _number(row["scoring_seconds"], "scoring_seconds", minimum=0)
    timing = _object(result["timing_seconds"], {"generation", "quality", "mmlu", "total"}, "timing")
    for key, value in timing.items():
        _number(value, key, minimum=0)
    for key, rows, field in (("generation", harmless, "generation_seconds"), ("quality", harmless, "quality_seconds"), ("mmlu", mmlu, "scoring_seconds")):
        if not math.isclose(timing[key], sum(row[field] for row in rows), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Aggregate timing differs")
    if timing["total"] + 1e-6 < sum(timing[key] for key in ("generation", "quality", "mmlu")):
        raise ValueError("Total timing omits endpoint work")
    return cast(ConditionResult, result)


def select_domain_doses(artifacts: Sequence[CompletedArtifact]) -> dict[str, float]:
    """Select from validated artifacts using exact singleton-group counts; baseline cancels."""
    expected = {spec.condition_id: spec for spec in development_specs()}
    counts: dict[str, int] = {}
    for artifact in artifacts:
        spec = _spec_from_metadata(artifact["condition"])
        if spec.condition_id not in expected or spec != expected[spec.condition_id] or spec.condition_id in counts:
            raise ValueError("Dose selection requires exactly the eleven development conditions")
        result = validate_condition_result(artifact["result"], spec)
        counts[spec.condition_id] = sum(row["refusal_proxy"] for row in result["harmless"])
    if set(counts) != set(expected):
        raise ValueError("Missing required development condition")
    pooled = counts["development-pooled-real-a1"]
    doses = {"pooled": 1.0}
    for target in TARGETS[1:]:
        doses[target] = min(DOSES, key=lambda alpha: (
            abs(counts[ConditionSpec("development", "real", target, alpha).condition_id] - pooled), alpha))
    return doses


def _validate_artifact(value: Any, spec: ConditionSpec, inputs: dict[str, str], direction: DirectionProvenance) -> CompletedArtifact:
    artifact = _object(value, {"schema_version", "condition", "provenance", "result"}, "completed artifact")
    if _integer(artifact["schema_version"], "schema_version") != 1 or _spec_from_metadata(artifact["condition"]) != spec:
        raise ValueError("Completed condition identity differs")
    provenance = _object(artifact["provenance"], {"inputs", "direction", "runtime"}, "provenance")
    if provenance["inputs"] != inputs:
        raise ValueError("Frozen input hashes differ")
    observed = _object(provenance["direction"], set(direction), "direction provenance")
    for key in ("raw_norm", "vector_norm", "injected_norm", "alpha"):
        _number(observed[key], key, minimum=0)
    if observed["random_seed"] is not None:
        _integer(observed["random_seed"], "random_seed")
    if observed != direction:
        raise ValueError("Vector bytes, seed, target or norm provenance differs")
    _validate_runtime(provenance["runtime"])
    validate_condition_result(artifact["result"], spec)
    return cast(CompletedArtifact, artifact)


def validate_completed_artifact(path: Path, expected_spec: ConditionSpec, *, tensor_path: Path, reference_path: Path) -> CompletedArtifact:
    """Re-read completion against frozen inputs and the current runtime environment.

    The validating launcher must start with the same versions and thread settings as
    its child; setting only the child's environment after importing torch is insufficient.
    """
    if (path.parent / "failure.json").exists():
        raise ValueError("Attempt has a failure record")
    inputs = capture_inputs(expected_spec.role, tensor_path, reference_path)
    _, direction = prepare_direction(expected_spec, tensor_path, reference_path)
    result = _validate_artifact(_read_json(path), expected_spec, inputs, direction)
    if capture_inputs(expected_spec.role, tensor_path, reference_path) != inputs:
        raise ValueError("Inputs changed during artifact validation")
    return result


def publish_json(path: Path, value: Any) -> str:
    """Publish strictly encoded JSON using a same-directory hard link; retain failed temp files."""
    content = _json_bytes(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path = Path(temporary)
    if temporary_path.stat().st_size == 0:
        raise ValueError("Empty temporary artifact")
    _read_json(temporary_path)
    os.link(temporary_path, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if _sha(path.read_bytes()) != _sha(content):
        raise ValueError("Published artifact readback differs")
    temporary_path.unlink()
    return _sha(content)


def run_condition(spec: ConditionSpec, *, tensor_path: Path, reference_path: Path, attempt_dir: Path) -> Path:
    """Run one fresh attempt; caller must never reuse a failed attempt directory."""
    attempt_dir.mkdir(parents=True, exist_ok=False)
    stage = "preparation"
    try:
        inputs = capture_inputs(spec.role, tensor_path, reference_path)
        direction, direction_provenance = prepare_direction(spec, tensor_path, reference_path)
        harmless, mmlu = _load_frozen_items(spec.role)
        publish_json(attempt_dir / "execution.json", {
            "schema_version": 1, "condition": spec.to_dict(), "inputs": inputs,
            "direction": direction_provenance, "environment": _environment(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        stage = "model_loading"
        tokenizer, model = load_qwen()
        runtime = _runtime(tokenizer, model)
        stage = "evaluation"
        result = evaluate_condition(model, tokenizer, harmless, mmlu, direction, spec.alpha,
                                    layer_index=14, max_new_tokens=256)
        # Invalid returned output remains archive-only, including any nonfinite diagnostics.
        partial = attempt_dir / "result.partial.json"
        with partial.open("xb") as stream:
            stream.write(_json_bytes(result, allow_nan=True))
            stream.flush()
            os.fsync(stream.fileno())
        stage = "validation"
        artifact: CompletedArtifact = {"schema_version": 1, "condition": spec.to_dict(),
                                       "provenance": {"inputs": inputs, "direction": direction_provenance, "runtime": runtime},
                                       "result": result}
        _validate_artifact(artifact, spec, inputs, direction_provenance)
        if capture_inputs(spec.role, tensor_path, reference_path) != inputs:
            raise ValueError("Inputs changed during condition")
        stage = "publication"
        output = attempt_dir / "completed.json"
        publish_json(output, artifact)
        return output
    except BaseException as error:
        publish_json(attempt_dir / "failure.json", {
            "schema_version": 1, "condition": spec.to_dict(), "stage": stage,
            "error_type": type(error).__name__, "failed_at": datetime.now(timezone.utc).isoformat(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("development", "final"), required=True)
    parser.add_argument("--control", choices=("baseline", "real", "random42", "random43"), required=True)
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        path = run_condition(ConditionSpec(args.role, args.control, args.target, args.alpha),
                             tensor_path=args.tensor, reference_path=args.reference, attempt_dir=args.attempt_dir)
    except Exception as error:
        raise SystemExit(f"Condition failed ({type(error).__name__}); inspect attempt records") from None
    print(f"Completed {path}")


if __name__ == "__main__":
    main()
