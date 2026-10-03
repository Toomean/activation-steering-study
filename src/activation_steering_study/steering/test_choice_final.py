"""Check frozen-input rejection and the final-only seven-condition schedule."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import transformers

from activation_steering_study.evaluation.choices import ScoredChoiceVariant
from activation_steering_study.steering import choice_final, choice_run
from activation_steering_study.utils.json_io import save_json


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.chdir(tmp_path)
    final_module = tmp_path / "src/activation_steering_study/steering/choice_final.py"
    final_module.parent.mkdir(parents=True)
    final_module.write_bytes(Path(choice_final.__file__).read_bytes())
    (tmp_path / "dependency.py").write_text("# Synthetic producer dependency\n")
    (tmp_path / "uv.lock").write_text("# Synthetic frozen lock\n")
    panel_path = tmp_path / "panel.json"
    save_json(panel_path, [
        {"source_index": 1, "group_id": "extract", "split": "extraction"},
        {"source_index": 2, "group_id": "dev", "split": "development"},
        {"source_index": 3, "group_id": "final", "split": "final", "question": "Animal?",
         "best_answer": "Dog", "best_incorrect_answer": "Cat"},
        {"source_index": 4, "group_id": "final", "split": "final", "question": "Pet?",
         "best_answer": "Dog", "best_incorrect_answer": "Cat"},
    ])
    dev_dir = tmp_path / "archived-dev"
    dev_dir.mkdir()
    tensor = dev_dir / "extraction.pt"
    torch.save({18: {"direction": torch.tensor([3.0, 4.0])}}, tensor)
    selected = {"positive": 1.0, "negative": -0.5}
    dev = {
        "behaviour": "honesty", "source_revision": "synthetic-upstream-revision",
        "panel_path": str(panel_path), "panel_sha256": _hash(panel_path),
        "tensor_path": "DO_NOT_FOLLOW_ORIGINAL_PATH", "tensor_sha256": _hash(tensor),
        "block": 18, "random_seeds": [42, 43], "raw_direction_norm": 5.0,
        "target_direction_norm": 10.0, "direction_scale_factor": 2.0,
        "selected_real_alphas": selected,
        "model_id": choice_final.MODEL_ID, "model_revision": choice_final.MODEL_REVISION,
        "model_dtype": "torch.float32", "model_device": "cpu",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "code_sha256": {"dependency.py": _hash(tmp_path / "dependency.py")},
        "extraction_source_ids": [1], "extraction_group_ids": ["extract"],
        "development_source_ids": [2], "development_group_ids": ["dev"],
        "answer_suffix": "\nAnswer with A or B only.",
        "development_aggregation": "orders, rows per group, groups equally",
    }
    save_json(dev_dir / "summary.json", dev)
    selection_path = tmp_path / "selection.json"
    save_json(selection_path, {
        "frozen_utc": "synthetic-freeze", "block": 18, "target_direction_norm": 10.0,
        "behaviours": {"honesty": {
            "summary_sha256": _hash(dev_dir / "summary.json"),
            "panel_sha256": _hash(panel_path), "tensor_sha256": _hash(tensor),
            "uv_lock_sha256": _hash(tmp_path / "uv.lock"),
            "implementation_commit": "synthetic-producer-commit",
            "planning_commit": "synthetic-plan-commit", "selected_real_alphas": selected,
            "planned_final_conditions": ["baseline", "real_alpha_+1", "random42_alpha_+1",
                                         "random43_alpha_+1", "real_alpha_-0.5",
                                         "random42_alpha_-0.5", "random43_alpha_-0.5"],
        }},
    })
    return dev_dir, selection_path, tmp_path / "final-run"


@pytest.mark.parametrize("raw,scale,expected", [
    ([3.0, 4.0], 2.0, [6.0, 8.0]),
    ([1.0, 3.0], 3.1622776218199133, [3.1622776985168457, 9.486833572387695]),
])
def test_final_reuses_direction_and_frozen_doses(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
    raw: list[float], scale: float, expected: list[float],
) -> None:
    dev_dir, selection, output = inputs
    tensor = dev_dir / "extraction.pt"
    torch.save({18: {"direction": torch.tensor(raw)}}, tensor)
    dev = json.loads((dev_dir / "summary.json").read_bytes())
    dev.update(direction_scale_factor=scale, raw_direction_norm=torch.linalg.vector_norm(torch.tensor(raw)).item(),
               tensor_sha256=_hash(tensor))
    save_json(dev_dir / "summary.json", dev)
    frozen = json.loads(selection.read_bytes())
    frozen["behaviours"]["honesty"].update(
        summary_sha256=_hash(dev_dir / "summary.json"), tensor_sha256=_hash(tensor),
    )
    save_json(selection, frozen)
    original_tensor = (dev_dir / "extraction.pt").read_bytes()
    model = SimpleNamespace(dtype=torch.float32, device=torch.device("cpu"),
                            config=SimpleNamespace(hidden_size=2, _attn_implementation="eager"))
    monkeypatch.setattr(choice_final, "load_qwen", lambda: (object(), model))
    calls: list[tuple[torch.Tensor | None, float]] = []
    shared_score = choice_run.score_condition

    def score(tokenizer, loaded, behaviour, rows, vector, alpha):
        assert [row["source_index"] for row in rows] == [3, 4], "Only final rows may be scored"
        calls.append((vector, alpha))
        return shared_score(tokenizer, loaded, behaviour, rows, vector, alpha)

    def score_variant(_tokenizer, _model, variant, **kwargs):
        matching = 0.5 + kwargs["alpha"] * 0.1
        return cast(ScoredChoiceVariant, {
            **variant, "p_a": matching, "p_b": 1 - matching, "ab_mass": 1.0,
            "conditional_matching_probability": matching,
        })

    monkeypatch.setattr(choice_final, "score_condition", score)
    monkeypatch.setattr(choice_run, "score_choice_variant", score_variant)
    read_counts = {dev_dir / "summary.json": 0, selection: 0, Path(dev["panel_path"]): 0}
    read_bytes = Path.read_bytes

    def read_once(path: Path) -> bytes:
        if path in read_counts:
            read_counts[path] += 1
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_once)
    artifact = choice_final.run(dev_dir, selection, output)
    assert list(read_counts.values()) == [1, 1, 1], "Parse and hash each JSON from one byte snapshot"
    assert [alpha for _, alpha in calls] == [0.0, 1.0, 1.0, 1.0, -0.5, -0.5, -0.5], (
        "Controls must reuse both frozen real alphas in the planned seven-condition order"
    )
    assert calls[0][0] is None, "Baseline must be hook-free"
    real, random42, random43 = (vector for vector, _ in calls[1:4])
    assert real is not None and random42 is not None and random43 is not None, (
        "All steered controls must provide a direction"
    )
    assert torch.equal(real, torch.tensor(expected)), "Use stored multiplication, including its float32 rounding"
    for index, seed, vector in ((2, 42, random42), (3, 43, random43)):
        assert torch.equal(vector, choice_final.sample_random_direction(2, 10.0, seed)), (
            "Each random control must use its named seed and the retained norm"
        )
        assert not torch.equal(vector, real), "Random and real vectors must differ"
        negative_vector = calls[index + 3][0]
        assert negative_vector is not None and torch.equal(vector, negative_vector), (
            "Reuse each control for both signs"
        )
    assert not torch.equal(random42, random43), "Seeds 42 and 43 must produce different controls"
    settings = cast(dict[str, dict[str, object]], artifact["condition_settings"])
    summaries = cast(dict[str, dict[str, float]], artifact["condition_summaries"])
    for name, summary in summaries.items():
        assert summary["delta_vs_baseline"] == pytest.approx(
            summary["mean_conditional_matching_probability"] - 0.5,
        ), f"{name} must retain its change from the known 0.5 baseline"
    assert settings["real_alpha_-0.5"]["injected_norm"] == 5.0, "Injection norm must include dose magnitude"
    assert artifact["torch_num_threads"] == torch.get_num_threads(), "Record threads from the scoring process"
    assert artifact["torch_num_interop_threads"] == torch.get_num_interop_threads(), "Record inter-op threads"
    assert artifact["attention_implementation"] == "eager", "Record the loaded model's attention implementation"
    assert artifact["final_source_ids"] == [3, 4] and artifact["final_group_ids"] == ["final", "final"], (
        "Summary must preserve final source and group membership"
    )
    assert artifact["source_revision"] == "synthetic-upstream-revision", "Retain the panel source revision"
    assert (artifact["producer_implementation_commit"], artifact["producer_planning_commit"]) == (
        "synthetic-producer-commit", "synthetic-plan-commit",
    ) and "implementation_commit" not in artifact and "planning_commit" not in artifact, (
        "Producer commits must be named separately from final execution provenance"
    )
    assert artifact["dev_summary_sha256"] == _hash(dev_dir / "summary.json"), "Bind the producer summary"
    assert artifact["selection_sha256"] == _hash(selection), "Bind the frozen selection"
    assert len(list(output.glob("*.json"))) == 8, "Seven conditions plus one concise summary are saved"
    result = json.loads((output / "real_alpha_+1.json").read_text())
    assert result["group_means"] == [{"group_id": "final", "mean": 0.6, "row_count": 2}], (
        "Saved group metadata must retain both final rows"
    )
    assert all(len(row["orders"]) == 2 for row in result["results"]), "Retain both answer orders"
    assert "extraction_rows" not in artifact, "Final summary must not spread dev raw prompts"
    assert (dev_dir / "extraction.pt").read_bytes() == original_tensor, "Raw tensor bytes must survive"


@pytest.mark.parametrize("changed,prefix", [
    ("summary", "Dev summary hash"), ("panel", "panel_sha256"), ("tensor", "tensor_sha256"),
    ("code", "Dependency code hashes"), ("lock", "uv_lock_sha256"), ("dose", "Selected real alphas"),
])
def test_changed_inputs_rejected_before_model_or_output(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, changed: str, prefix: str,
) -> None:
    dev_dir, selection, output = inputs
    paths = {"summary": dev_dir / "summary.json", "panel": Path("panel.json"),
             "tensor": dev_dir / "extraction.pt", "code": Path("dependency.py"),
             "lock": Path("uv.lock")}
    if changed == "dose":
        frozen = json.loads(selection.read_text())
        frozen["behaviours"]["honesty"]["selected_real_alphas"]["positive"] = 2.0
        frozen["behaviours"]["honesty"]["planned_final_conditions"] = [
            name.replace("alpha_+1", "alpha_+2")
            for name in frozen["behaviours"]["honesty"]["planned_final_conditions"]
        ]
        save_json(selection, frozen)
    else:
        path = paths[changed]
        path.write_bytes(path.read_bytes() + b"\n")
    loads: list[bool] = []
    monkeypatch.setattr(choice_final, "load_qwen", lambda: loads.append(True))
    with pytest.raises(ValueError, match=f"^{prefix} differs from frozen development provenance$"):
        choice_final.run(dev_dir, selection, output)
    assert not loads and not output.exists(), "Input rejection must precede model and output creation"


@pytest.mark.parametrize("field,value,prefix", [
    ("split", "development", "Panel needs a nonempty final split"),
    ("source_index", 1, "Final source_ids overlap"), ("group_id", "dev", "Final group_ids overlap"),
])
def test_invalid_final_split_rejected_before_model(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
    field: str, value: object, prefix: str,
) -> None:
    dev_dir, selection, output = inputs
    panel_path = Path("panel.json")
    panel = json.loads(panel_path.read_text())
    for row in panel[2:]:
        row[field] = value
    save_json(panel_path, panel)
    dev = json.loads((dev_dir / "summary.json").read_text())
    dev["panel_sha256"] = _hash(panel_path)
    save_json(dev_dir / "summary.json", dev)
    frozen = json.loads(selection.read_text())
    frozen["behaviours"]["honesty"].update(
        panel_sha256=_hash(panel_path), summary_sha256=_hash(dev_dir / "summary.json"),
    )
    save_json(selection, frozen)
    loads: list[bool] = []
    monkeypatch.setattr(choice_final, "load_qwen", lambda: loads.append(True))
    with pytest.raises(ValueError, match=f"^{prefix}"):
        choice_final.run(dev_dir, selection, output)
    assert not loads and not output.exists(), "Final split checks must precede model work"


@pytest.mark.parametrize("scope,field", [
    ("dev", "source_revision"), ("dev", "raw_direction_norm"), ("dev", "answer_suffix"),
    ("dev", "development_aggregation"), ("dev", "model_dtype"), ("dev", "model_device"),
    ("selection", "frozen_utc"), ("frozen", "implementation_commit"), ("frozen", "planning_commit"),
])
def test_missing_static_metadata_rejected_before_loader(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, scope: str, field: str,
) -> None:
    dev_dir, selection, output = inputs
    summary = dev_dir / "summary.json"
    path = summary if scope == "dev" else selection
    metadata = json.loads(path.read_bytes())
    target = metadata["behaviours"]["honesty"] if scope == "frozen" else metadata
    del target[field]
    save_json(path, metadata)
    if scope == "dev":
        frozen = json.loads(selection.read_bytes())
        frozen["behaviours"]["honesty"]["summary_sha256"] = _hash(summary)
        save_json(selection, frozen)
    monkeypatch.setattr(choice_final, "load_qwen", lambda: pytest.fail("Incomplete metadata must precede loader"))
    with pytest.raises(KeyError, match=f"^'{field}'$"):
        choice_final.run(dev_dir, selection, output)
    assert not output.exists(), "Incomplete static metadata must not create a result directory"


@pytest.mark.parametrize("field,value", [("dtype", torch.bfloat16), ("device", torch.device("meta"))])
def test_loaded_model_mismatch_precedes_output_and_scoring(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, field: str, value: object,
) -> None:
    dev_dir, selection, output = inputs
    model = SimpleNamespace(dtype=torch.float32, device=torch.device("cpu"),
                            config=SimpleNamespace(hidden_size=2, _attn_implementation="eager"))
    setattr(model, field, value)
    monkeypatch.setattr(choice_final, "load_qwen", lambda: (object(), model))
    monkeypatch.setattr(choice_final, "score_condition", lambda *_args: pytest.fail("Mismatch must precede scoring"))
    with pytest.raises(ValueError, match=f"^Loaded model {field} differs from frozen development provenance$"):
        choice_final.run(dev_dir, selection, output)
    assert not output.exists(), "Model mismatch must not create a result directory"


def test_existing_output_is_preserved(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    dev_dir, selection, output = inputs
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text("existing result")
    monkeypatch.setattr(choice_final, "load_qwen", lambda: pytest.fail("Existing output must precede loader"))
    with pytest.raises(FileExistsError):
        choice_final.run(dev_dir, selection, output)
    assert sentinel.read_text() == "existing result", "Existing runs must never be overwritten"
