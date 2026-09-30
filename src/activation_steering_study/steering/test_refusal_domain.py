"""Check refusal-domain direction, conditions, and candidate summaries."""

import json
import math
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from activation_steering_study.data.mmlu import MmluItem
from activation_steering_study.data.refusal import SampleRefusalItem
from activation_steering_study.steering import refusal_domain as domain
from activation_steering_study.steering.refusal_domain import (
    condition_key, direction_from_rows, paired_accuracy,
)


def test_direction_uses_shared_harmless_mean_in_fp32() -> None:
    harmful = torch.tensor([[4, 6], [8, 10]], dtype=torch.bfloat16)
    harmless = torch.tensor([[1, 3], [3, 5]], dtype=torch.bfloat16)
    result = direction_from_rows(harmful, harmless)
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.tensor([4.0, 4.0])), "DiM sign or means changed"
    with pytest.raises(ValueError, match="Zero or nonfinite"):
        direction_from_rows(harmless, harmless)


def test_condition_names_preserve_target_control_and_sign() -> None:
    assert condition_key("cyber_intrusion", -0.5, "random") == "cyber_intrusion-random-negative-0p5"
    assert condition_key("pooled", 2.0, "direction") == "pooled-direction-positive-2p0"
    assert condition_key("pooled", 0.0, "baseline") == "baseline"


def test_paired_accuracy_requires_same_source_order() -> None:
    before = [
        {"id": "a", "correct": True, "option_mass": 0.8},
        {"id": "b", "correct": False, "option_mass": 0.6},
    ]
    after = [
        {"id": "a", "correct": False, "option_mass": 0.7},
        {"id": "b", "correct": True, "option_mass": 0.5},
    ]
    result = paired_accuracy(before, after)
    assert result["correct_to_wrong"] == 1 and result["wrong_to_correct"] == 1
    assert result["paired_accuracy_change"] == 0
    with pytest.raises(ValueError, match="source IDs"):
        paired_accuracy(before, list(reversed(after)))


def test_paired_accuracy_reports_one_sided_loss() -> None:
    """One loss without a gain exposes a flipped change sign or swapped transition counts."""
    before = [
        {"id": "a", "correct": True, "option_mass": 0.6},
        {"id": "b", "correct": True, "option_mass": 0.5},
        {"id": "c", "correct": False, "option_mass": 0.4},
    ]
    after = [
        {"id": "a", "correct": False, "option_mass": 0.9},
        {"id": "b", "correct": True, "option_mass": 0.8},
        {"id": "c", "correct": False, "option_mass": 0.7},
    ]
    result = paired_accuracy(before, after)
    assert result["correct"] == 1, "Current accuracy must count only row b"
    assert result["baseline_correct"] == 2, "Baseline accuracy must count rows a and b"
    assert result["wrong_to_correct"] == 0, "No row changed from wrong to correct"
    assert result["correct_to_wrong"] == 1, "Row a changed from correct to wrong"
    assert result["paired_accuracy_change"] == pytest.approx(-1 / 3), (
        "One loss among three rows must lower paired accuracy by 1/3"
    )
    assert result["mean_option_mass"] == pytest.approx(0.8), (
        "Option mass must average the current rows 0.9, 0.8 and 0.7, not the baseline rows"
    )


def test_candidate_sweep_rejects_mixed_tensor_and_stale_policy(tmp_path) -> None:
    """A saved candidate must belong to this source and current A/B producer."""
    from activation_steering_study.steering.choice_sweep import (
        ALPHAS as CHOICE_ALPHAS, LAYERS as CHOICE_LAYERS, SOURCE_REVISIONS,
        TARGET_NORM, load_sweep_sources, sweep_code_hashes,
    )
    from activation_steering_study.steering.refusal_domain import _validate_candidate_sweep, sha256
    from activation_steering_study.utils.choice_prompt import ANSWER_SUFFIX

    tensor_path = tmp_path / "candidate.pt"
    torch.save({14: {"direction": torch.ones(2)}}, tensor_path)
    prompts_path, source_path, samples = load_sweep_sources("honesty")
    development_ids = [row["source_index"] for row in samples if row["split"] == "development"]
    sweep = {
        "behaviour": "honesty", "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "model_dtype": "torch.bfloat16", "source_revision": SOURCE_REVISIONS["honesty"],
        "source_path": str(source_path), "source_sha256": sha256(source_path),
        "selected_path": str(prompts_path), "selected_sha256": sha256(prompts_path),
        "code_sha256": sweep_code_hashes(), "answer_suffix": ANSWER_SUFFIX,
        "hook_policy": "final prompt token",
        "capture_policy": "completed bare answer letter final token; float32 block outputs",
        "pair_formula": "source-labelled matching answer activation - opposite answer activation",
        "aggregation": "mean of original/swapped within source, then mean across sources",
        "layers": list(CHOICE_LAYERS), "screen_alpha": 1.0,
        "selected_alphas": list(CHOICE_ALPHAS), "target_norm": TARGET_NORM,
        "random_seed": 42, "tensor_path": str(tensor_path), "tensor_sha256": sha256(tensor_path),
        "extraction_count": 23, "development_count": 16, "selected_layer": 18,
        "conditions": {"baseline": {"results": [{"source_index": index} for index in development_ids]}},
    }
    original_tensor_bytes = tensor_path.read_bytes()
    _validate_candidate_sweep(sweep, "honesty", tensor_path, "torch.bfloat16")

    torch.save({14: {"direction": torch.zeros(2)}}, tensor_path)
    with pytest.raises(ValueError, match="tensor_sha256"):
        _validate_candidate_sweep(sweep, "honesty", tensor_path, "torch.bfloat16")
    tensor_path.write_bytes(original_tensor_bytes)

    for field, stale in (("model_dtype", "torch.float32"),
                         ("code_sha256", {}), ("source_sha256", "stale"),
                         ("hook_policy", "different token")):
        changed = {**sweep, field: stale}
        with pytest.raises(ValueError, match=field):
            _validate_candidate_sweep(changed, "honesty", tensor_path, "torch.bfloat16")


def test_summary_indexes_present_candidate_without_counting_absent_ones(tmp_path, monkeypatch) -> None:
    """Optional candidate conditions remain visible but outside the required-grid count."""
    monkeypatch.setattr(domain, "CONDITIONS_DIR", tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}", encoding="utf-8")
    baseline_sha = domain.sha256(baseline)
    for control in ("direction", "random", "candidate-honesty"):
        key = domain.condition_key("pooled", 1.0, control)
        (tmp_path / f"{key}.json").write_text(json.dumps({
            "metadata": {"baseline_sha256": baseline_sha},
            "endpoint": "induction on 16 harmless", "endpoint_effect_vs_baseline": 0.25,
            "mmlu_summary": {}, "benign_summary": {}, "harmful_summary": None,
            "own_behaviour": {"status": "candidate control only"} if control.startswith("candidate-") else None,
            "status": "manual response audit pending",
        }), encoding="utf-8")
    domain.summarize()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    candidate_key = domain.condition_key("pooled", 1.0, "candidate-honesty")
    assert candidate_key in summary["curves"], "Saved candidate condition disappeared from summary"
    assert summary["curves"][candidate_key]["own_behaviour"]["status"] == "candidate control only", (
        "Summary lost candidate own-function status"
    )
    assert len(summary["missing_frozen_conditions"]) == 46, (
        "Absent optional candidates changed the 48-condition required-grid count"
    )


def _fail_if_called(message: str):
    """Return a stand-in collaborator that fails the test when the code under test calls it."""
    def fail(*_args, **_kwargs):
        raise AssertionError(message)
    return fail


def test_extraction_load_rejects_each_changed_provenance_input(tmp_path, monkeypatch) -> None:
    """Model ID and revision, layer, source hashes, code hashes, and tensor bytes all gate loading."""
    extraction_path = tmp_path / "refusal-domains.json"
    direction_path = tmp_path / "refusal-domains.pt"
    monkeypatch.setattr(domain, "EXTRACTION_PATH", extraction_path)
    monkeypatch.setattr(domain, "DIRECTION_PATH", direction_path)
    monkeypatch.setattr(domain, "source_hashes", lambda: {"data/a.json": "aa"})
    monkeypatch.setattr(domain, "code_hashes", lambda: {"src/b.py": "bb"})
    torch.save({"directions": {"pooled": torch.tensor([1.0, 2.0])}}, direction_path)
    recorded = {
        "model_id": domain.MODEL_ID, "model_revision": domain.MODEL_REVISION,
        "layer_index": domain.LAYER, "source_sha256": {"data/a.json": "aa"},
        "code_sha256": {"src/b.py": "bb"}, "tensor_sha256": domain.sha256(direction_path),
    }
    extraction_path.write_text(json.dumps(recorded), encoding="utf-8")
    metadata, tensors = domain._load_extraction()
    assert metadata == recorded, "Matching provenance did not return the saved metadata"
    assert torch.equal(
        cast(dict[str, torch.Tensor], tensors["directions"])["pooled"], torch.tensor([1.0, 2.0])
    ), "Matching provenance did not return the saved pooled direction"

    for field, stale in (("model_id", "other/model"), ("model_revision", "0" * 40),
                         ("layer_index", 13), ("source_sha256", {"data/a.json": "ab"}),
                         ("code_sha256", {"src/b.py": "bc"})):
        extraction_path.write_text(json.dumps({**recorded, field: stale}), encoding="utf-8")
        with pytest.raises(ValueError, match="Extraction provenance changed"):
            domain._load_extraction()
    extraction_path.write_text(json.dumps(recorded), encoding="utf-8")
    torch.save({"directions": {"pooled": torch.tensor([1.0, 3.0])}}, direction_path)
    with pytest.raises(ValueError, match="Extraction provenance changed"):
        domain._load_extraction()


def test_generations_average_nll_per_nonempty_prompt_and_kl_per_prompt(monkeypatch) -> None:
    """Response perplexity skips empty responses; prompt KL covers every prompt."""
    items: list[SampleRefusalItem] = [
        {"sample_path": "harmless_val.json", "sample_index": index, "instruction": f"prompt {index}"}
        for index in range(3)
    ]
    completions = {
        "prompt 0": {"text": "I'm sorry, no", "prompt_token_ids": [9], "generated_token_ids": [1]},
        "prompt 1": {"text": "Sure", "prompt_token_ids": [9], "generated_token_ids": [1, 2, 3]},
        "prompt 2": {"text": "", "prompt_token_ids": [9], "generated_token_ids": []},
    }

    def canned_likelihood(_model, _prompt_ids, generated_ids):
        mean_nll = {0: None, 1: math.log(2), 3: math.log(8)}[len(generated_ids)]
        return {"token_count": len(generated_ids), "mean_nll": mean_nll,
                "perplexity": None if mean_nll is None else math.exp(mean_nll)}

    kl_values = iter([0.5, 1.0, 1.5])
    monkeypatch.setattr(domain, "generate_completion",
                        lambda _model, _tokenizer, prompt, *_args: completions[prompt])
    monkeypatch.setattr(domain, "response_likelihood", canned_likelihood)
    monkeypatch.setattr(domain, "prompt_distribution_kl", lambda *_args, **_kwargs: next(kl_values))
    tokenizer = SimpleNamespace(apply_chat_template=lambda *_args, **_kwargs: "rendered prompt")

    rows, summary = domain._generations(tokenizer, object(), items, torch.ones(2), 1.0, coherence=True)
    assert summary["refusal_count"] == 1, "Only the apology contains a refusal substring"
    assert summary["refusal_rate"] == pytest.approx(1 / 3), "Refusal rate must divide by all three prompts"
    assert summary["nonempty_count"] == 2, "Two responses have scored tokens"
    assert summary["empty_count"] == 1, "The empty response must be counted apart from scored responses"
    # Per-prompt NLLs ln 2 and ln 8 average to ln 4; token weighting over lengths 1 and 3
    # would give 2 ** 2.5, and dividing by all three prompts would give 2 ** (4 / 3).
    assert summary["clean_ppl_exp_mean_per_prompt_nll"] == pytest.approx(4.0), (
        "Perplexity must exponentiate the mean NLL of the nonempty responses"
    )
    assert summary["mean_prompt_kl_baseline_to_condition"] == pytest.approx(1.0), (
        "Prompt KL must average all three prompts, including the empty response"
    )
    assert [row["prompt_kl_baseline_to_condition"] for row in rows] == [0.5, 1.0, 1.5], (
        "Per-row KL must match the measured prompt KL values in prompt order"
    )

    monkeypatch.setattr(domain, "prompt_distribution_kl", _fail_if_called("baseline KL must not be measured"))
    rows, summary = domain._generations(tokenizer, object(), items, None, 0.0, coherence=True)
    assert [row["prompt_kl_baseline_to_condition"] for row in rows] == [0.0, 0.0, 0.0], (
        "Unsteered rows must record zero KL without measuring it"
    )
    assert summary["mean_prompt_kl_baseline_to_condition"] == 0.0, "Unsteered mean KL must be zero"

    monkeypatch.setattr(domain, "response_likelihood", _fail_if_called("responses must not be scored"))
    rows, summary = domain._generations(tokenizer, object(), items, torch.ones(2), -1.0, coherence=False)
    assert set(summary) == {"count", "refusal_count", "refusal_rate"}, (
        "Summaries without coherence scoring must report only refusal counts"
    )
    assert all("clean_response_likelihood" not in row and "prompt_kl_baseline_to_condition" not in row
               for row in rows), "Rows without coherence scoring must not carry likelihood or KL fields"


def test_mmlu_rows_keep_source_fields_and_reject_changed_sample_size(monkeypatch) -> None:
    """Rows keep item identity and gold letters, and only matching predictions count as correct."""
    items: list[MmluItem] = [
        {"id": f"q{index}", "subject": "algebra", "source_index": index,
         "question": f"Question {index}?", "choices": ["w", "x", "y", "z"], "answer": index % 4}
        for index in range(285)
    ]
    # Even source indices predict the gold letter; odd ones predict the following letter.
    predictions = {domain.render_question(item): "ABCD"[(item["answer"] + item["source_index"] % 2) % 4]
                   for item in items}
    monkeypatch.setattr(domain, "sample_mmlu", lambda: items)
    monkeypatch.setattr(domain, "score_question", lambda _tokenizer, _model, prompt, *_args: {
        "prediction": predictions[prompt], "option_mass": 0.9,
    })
    rows = domain._mmlu(object(), object(), None, 0.0)
    assert len(rows) == 285, "Each of the 285 fixed MMLU items must produce one row"
    assert [(row["id"], row["subject"], row["source_index"]) for row in rows] == [
        (item["id"], item["subject"], item["source_index"]) for item in items
    ], "Rows must keep each item's ID, subject, and source index in sample order"
    assert [row["gold"] for row in rows] == list("ABCD" * 71 + "A"), (
        "Gold letters must map answer indices 0 to 3 to A to D"
    )
    assert [row["source_index"] for row in rows if row["correct"]] == list(range(0, 285, 2)), (
        "Exactly the 143 even-index rows, whose prediction equals gold, must be correct"
    )

    monkeypatch.setattr(domain, "sample_mmlu", lambda: items[:284])
    with pytest.raises(ValueError, match="Fixed MMLU sample changed"):
        domain._mmlu(object(), object(), None, 0.0)


def test_run_rejects_condition_outside_frozen_grid_before_loading(tmp_path, monkeypatch) -> None:
    """An off-grid target or alpha fails before extraction files or the model are read."""
    monkeypatch.setattr(domain, "CONDITIONS_DIR", tmp_path)
    monkeypatch.setattr(domain, "_load_extraction", _fail_if_called("extraction must not load"))
    monkeypatch.setattr(domain, "load_qwen", _fail_if_called("model must not load"))
    with pytest.raises(ValueError, match="outside frozen target/alpha grid"):
        domain.run("pooled", 0.3, "direction")
    with pytest.raises(ValueError, match="outside frozen target/alpha grid"):
        domain.run("other", 1.0, "direction")


def test_run_saves_signed_effects_and_reuses_only_identical_conditions(
    tmp_path, monkeypatch, capsys
) -> None:
    """Saved effects follow the endpoint sign; reruns skip identical metadata and reject stale inputs."""
    conditions = tmp_path / "conditions"
    monkeypatch.setattr(domain, "CONDITIONS_DIR", conditions)
    for name, filename in (("EXTRACTION_PATH", "extraction.json"), ("DIRECTION_PATH", "directions.pt"),
                           ("SOURCE_PATH", "mmlu-source.parquet")):
        (tmp_path / filename).write_text(f"{name} stand-in", encoding="utf-8")
        monkeypatch.setattr(domain, name, tmp_path / filename)
    # Stubs replace the model loader, reviewed-prompt loader, extraction loader, code-hash reader,
    # and the two model-bound scoring loops, _generations and _mmlu. Metadata assembly, baseline
    # and saved-condition comparisons, the skip-or-raise branch, and endpoint-effect signs run
    # unchanged.
    monkeypatch.setattr(domain, "code_hashes", lambda: {"src/b.py": "bb"})
    monkeypatch.setattr(domain, "_load_extraction", lambda: (
        {"model_dtype": "torch.bfloat16"}, {"directions": {"pooled": torch.ones(4)}}
    ))
    monkeypatch.setattr(domain, "load_qwen", lambda: (object(), SimpleNamespace(dtype=torch.bfloat16)))
    monkeypatch.setattr(domain, "load_reviewed_refusal", lambda: {"validation": {
        "harmless": [{"sample_path": "harmless_val.json", "sample_index": index,
                      "instruction": f"harmless {index}"} for index in range(16)],
        "harmful": [{"sample_path": "harmful_val.json", "sample_index": index,
                     "instruction": f"harmful {index}"} for index in range(16)],
    }})
    baseline_mmlu = [{"id": "q1", "correct": True, "option_mass": 0.9},
                     {"id": "q2", "correct": True, "option_mass": 0.8},
                     {"id": "q3", "correct": False, "option_mass": 0.7}]
    current_mmlu = [{"id": "q1", "correct": False, "option_mass": 0.9},
                    {"id": "q2", "correct": True, "option_mass": 0.8},
                    {"id": "q3", "correct": False, "option_mass": 0.7}]

    expected_alpha = 0.0

    def canned_mmlu(_tokenizer, _model, direction, alpha):
        assert alpha == expected_alpha, "Run must pass its original signed alpha to _mmlu"
        if alpha == 0.0:
            assert direction is None, "Baseline MMLU must receive no direction"
            return [dict(row) for row in baseline_mmlu]
        assert direction is not None and torch.equal(direction, torch.ones(4)), (
            "Steered MMLU must receive the raw positive direction"
        )
        return [dict(row) for row in current_mmlu]

    monkeypatch.setattr(domain, "_mmlu", canned_mmlu)
    # Refusal counts among 16 prompts per (prompt list, direction, signed alpha, coherence). Run scores
    # coherence only for the harmless prompts, so any other combination is a lookup failure.
    # Baseline and steered endpoint counts independently produce effects 0.375 and 0.75.
    refusals = {("harmless_val.json", None, 0.0, True): 4,
                ("harmful_val.json", None, 0.0, False): 14,
                ("harmless_val.json", "direction", 1.0, True): 10,
                ("harmless_val.json", "direction", -1.0, True): 8,
                ("harmful_val.json", "direction", -1.0, False): 2}

    def canned_generations(_tokenizer, _model, items, direction, alpha, *, coherence):
        assert alpha == expected_alpha, "Run must pass its original signed alpha to _generations"
        direction_key = None if direction is None else "direction"
        if direction is None:
            assert alpha == 0.0, "Baseline generations must receive alpha zero"
        else:
            assert torch.equal(direction, torch.ones(4)), (
                "Steered generations must receive the raw positive direction"
            )
        count = refusals[(items[0]["sample_path"], direction_key, alpha, coherence)]
        rows = [{"sample_path": item["sample_path"], "sample_index": item["sample_index"]}
                for item in items]
        return rows, {"count": len(items), "refusal_count": count, "refusal_rate": count / len(items)}

    monkeypatch.setattr(domain, "_generations", canned_generations)

    domain.run("pooled", 0.0, "baseline")
    baseline_path = conditions / "baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["endpoint_effect_vs_baseline"] is None, "Baseline must not report an effect on itself"
    assert baseline["harmful_summary"] is not None, "Baseline must score the harmful prompts"
    assert baseline["mmlu_summary"]["correct"] == 2, "Baseline MMLU accuracy must count q1 and q2"
    assert baseline["benign_summary"]["count"] == baseline["harmful_summary"]["count"] == 16, (
        "Run fixture must use 16 synthetic prompts in each group"
    )

    positive_path = conditions / "pooled-direction-positive-1p0.json"
    expected_alpha = 1.0
    domain.run("pooled", 1.0, "direction")
    positive = json.loads(positive_path.read_text(encoding="utf-8"))
    assert positive["endpoint_effect_vs_baseline"] == 0.375, (
        "Induction effect must be steered benign 0.625 minus baseline benign 0.25"
    )
    assert positive["harmful"] == [] and positive["harmful_summary"] is None, (
        "Positive alpha must not generate the harmful prompts"
    )
    assert positive["mmlu_summary"]["paired_accuracy_change"] == pytest.approx(-1 / 3), (
        "One baseline-correct item lost with no gains must reduce paired MMLU accuracy by 1/3"
    )

    expected_alpha = -1.0
    domain.run("pooled", -1.0, "direction")
    negative = json.loads((conditions / "pooled-direction-negative-1p0.json").read_text(encoding="utf-8"))
    assert negative["endpoint_effect_vs_baseline"] == 0.75, (
        "Suppression effect must be baseline harmful 0.875 minus steered harmful 0.125"
    )
    assert negative["harmful_summary"] is not None, "Negative alpha must score the harmful prompts"

    monkeypatch.setattr(domain, "load_qwen", _fail_if_called("model must not load"))
    saved_bytes = positive_path.read_bytes()
    saved_sha = domain.sha256(positive_path)
    capsys.readouterr()
    domain.run("pooled", 1.0, "direction")
    assert "already complete" in capsys.readouterr().out, "Identical rerun did not report completion"
    assert domain.sha256(positive_path) == saved_sha, "Identical rerun rewrote the saved condition"

    stale = json.loads(saved_bytes)
    stale["metadata"]["layer_index"] = 13
    positive_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="stale metadata"):
        domain.run("pooled", 1.0, "direction")
    positive_path.write_bytes(saved_bytes)

    baseline["metadata"]["generation"]["max_new_tokens"] = 64
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises(ValueError, match="stale generation"):
        domain.run("pooled", 2.0, "direction")


def test_summary_prefers_smaller_matching_dose_and_binds_baseline(tmp_path, monkeypatch) -> None:
    """Nearest-dose matching ranks effect mismatch before dose size and pins the baseline hash."""
    monkeypatch.setattr(domain, "CONDITIONS_DIR", tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}", encoding="utf-8")
    baseline_sha = domain.sha256(baseline)
    effects = {
        ("pooled", 1.0): 0.5, ("pooled", -1.0): 0.375,
        ("cyber_intrusion", 0.5): 0.25, ("cyber_intrusion", 1.0): 0.5, ("cyber_intrusion", 2.0): 0.5,
        ("cyber_intrusion", -0.5): 0.25, ("cyber_intrusion", -1.0): 0.375,
        ("cyber_intrusion", -2.0): 0.375,
    }
    for (target, alpha), effect in effects.items():
        (tmp_path / f"{condition_key(target, alpha, 'direction')}.json").write_text(json.dumps({
            "metadata": {"baseline_sha256": baseline_sha},
            "endpoint": "induction on 16 harmless" if alpha > 0 else "suppression on 16 harmful",
            "endpoint_effect_vs_baseline": effect,
            "mmlu_summary": {}, "benign_summary": {}, "harmful_summary": None,
            "status": "manual response audit pending",
        }), encoding="utf-8")
    domain.summarize()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    positive = summary["closest_tested_dose"]["cyber_intrusion-positive"]
    negative = summary["closest_tested_dose"]["cyber_intrusion-negative"]
    # Doses +1 and +2 both match the pooled +1 effect 0.5 exactly, so the smaller dose must win.
    assert positive["candidate"] == condition_key("cyber_intrusion", 1.0, "direction"), (
        "An exact-effect tie did not resolve to the smaller alpha"
    )
    assert positive["absolute_refusal_rate_mismatch"] == 0.0, "Positive match must reproduce effect 0.5"
    # Both -1 and -2 match exactly; mismatch ranks first, then smaller absolute alpha wins.
    assert negative["candidate"] == condition_key("cyber_intrusion", -1.0, "direction"), (
        "An exact-effect tie did not resolve to the smaller absolute alpha"
    )
    assert negative["absolute_refusal_rate_mismatch"] == 0.0, "Negative match must reproduce effect 0.375"
    assert len(summary["missing_frozen_conditions"]) == 40, (
        "Eight present direction conditions must leave 40 of the 48 required conditions missing"
    )

    corrupted = tmp_path / f"{condition_key('cyber_intrusion', 1.0, 'direction')}.json"
    record = json.loads(corrupted.read_text(encoding="utf-8"))
    record["metadata"]["baseline_sha256"] = "0" * 64
    corrupted.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Condition baseline changed"):
        domain.summarize()
