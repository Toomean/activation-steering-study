"""Synthetic condition checks with no calls on frozen data panels."""

import hashlib
from typing import Any, cast

import pytest
import torch

from activation_steering_study.data.harmless import HarmlessItem
from activation_steering_study.data.mmlu import MmluItem
from activation_steering_study.steering import harmless_run


def _item(index: int, quality: bool) -> HarmlessItem:
    text = f"Name a garden flower number {index}."
    return {"row_id": f"harmless_val.json:{index}", "source_path": "harmless_val.json",
            "source_index": index, "instruction_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "role": "development", "quality_subset": quality,
            "semantic_group_id": f"garden-{index}", "semantic_status": "fixed_for_analysis",
            "instruction": text}


def test_condition_reuses_generation_quality_and_mmlu(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls = []
    quality_calls = []
    kl_calls = []
    mmlu_calls = []
    direction = torch.tensor([1.0, 2.0])

    def generate(_model, _tokenizer, prompt, vector, layer, alpha, settings):
        calls.append((prompt, vector, layer, alpha, settings))
        return {"text": "I cannot. I'm sorry. I cannot.", "prompt_token_ids": [1, 2],
                "generated_token_ids": [3, 4, 5]}

    def likelihood(_model, prompt_ids, response_ids):
        quality_calls.append((prompt_ids, response_ids))
        return {"token_count": 3, "mean_nll": 2.0, "perplexity": 7.389}

    def kl(_model, _inputs, vector, alpha, *, layer_index):
        kl_calls.append((vector, alpha, layer_index))
        return 0.25

    def score(_tokenizer, _model, text, layer, vector, alpha):
        mmlu_calls.append((text, layer, vector, alpha))
        return {"probabilities_abcd": [0.4, 0.1, 0.1, 0.1], "prediction": "A", "option_mass": 0.7}

    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            assert messages[0]["content"].startswith("Name a garden flower")
            return {"input_ids": torch.tensor([[1, 2]])}

    monkeypatch.setattr(harmless_run, "generate_completion", generate)
    monkeypatch.setattr(harmless_run, "response_likelihood", likelihood)
    monkeypatch.setattr(harmless_run, "prompt_distribution_kl", kl)
    monkeypatch.setattr(harmless_run, "score_question", score)
    question: MmluItem = {"id": "garden:0", "subject": "garden", "source_index": 0,
                          "question": "Which is a flower?", "choices": ["Rose", "Rock", "Cloud", "Table"], "answer": 0}
    items = [_item(0, True), _item(1, False)]
    result = harmless_run.evaluate_condition(cast(Any, None), cast(Any, Tokenizer()), items, [question],
                                            direction, 0.5, layer_index=7, max_new_tokens=8)
    assert len(calls) == 2 and all(call[1] is direction and call[2:4] == (7, 0.5) for call in calls)
    assert calls[0][-1] == {"max_new_tokens": 8, "do_sample": False}
    assert quality_calls == [([1, 2], [3, 4, 5])] and len(kl_calls) == 1
    assert kl_calls[0][0] is direction and kl_calls[0][1:] == (0.5, 7)
    assert result["harmless"][0]["completion_token_count"] == 3
    assert result["harmless"][1]["response_likelihood"] is None and result["harmless"][1]["prompt_kl"] is None
    assert result["harmless"][0]["phrase_matches"] == [
        {"phrase": "I cannot", "start": 0, "end": 8},
        {"phrase": "I'm sorry", "start": 10, "end": 19},
        {"phrase": "I cannot", "start": 21, "end": 29},
    ], "Every frozen phrase occurrence must retain its position"
    assert all(row["refusal_proxy"] for row in result["harmless"])
    assert len(mmlu_calls) == 1 and "A. Rose" in mmlu_calls[0][0] and mmlu_calls[0][1] == 7
    assert result["mmlu"][0]["gold"] == "A" and result["mmlu"][0]["correct"]
    assert all(value >= 0 for value in result["timing_seconds"].values())
    baseline = harmless_run.evaluate_condition(cast(Any, None), cast(Any, Tokenizer()), items, [question])
    assert all(call[1] is None and call[3] == 0.0 for call in calls[2:])
    assert baseline["harmless"][0]["prompt_kl"] == 0.0 and len(kl_calls) == 1
    assert baseline["max_new_tokens"] == 256
    assert capsys.readouterr().out == "", "The computation kernel must not print prompts or completions"


@pytest.mark.parametrize("direction,alpha", [(None, 1.0), (torch.ones(2), 0.0),
                                          (torch.ones(2), -1.0), (torch.ones(2), float("nan")),
                                          (torch.ones(2), float("inf"))])
def test_invalid_induction_settings_fail_before_model(direction, alpha) -> None:
    with pytest.raises(ValueError, match="alpha"):
        harmless_run.evaluate_condition(cast(Any, None), cast(Any, None), [_item(0, False)], [], direction, alpha)


def test_role_dispatch_uses_only_requested_mmlu_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    def harmless(role):
        calls.append(role)
        return []

    def calibration():
        calls.append("calibration")
        return []

    def final():
        calls.append("frozen-final")
        return []

    monkeypatch.setattr(harmless_run, "load_harmless", harmless)
    monkeypatch.setattr(harmless_run, "sample_mmlu", calibration)
    monkeypatch.setattr(harmless_run, "load_mmlu_final", final)
    assert harmless_run.load_condition_items("development") == ([], [])
    assert calls == ["development", "calibration"], "Development must not deserialize final MMLU"
    calls.clear()
    assert harmless_run.load_condition_items("final") == ([], [])
    assert calls == ["final", "frozen-final"], "Final must use explicit frozen MMLU IDs"
