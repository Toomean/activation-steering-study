"""Check fixed source selection, semantic option reversal, and Qwen token scoring."""

import json
import re
from typing import cast

import pytest
import torch
from transformers import AutoTokenizer

from activation_steering_study.evaluation.sycophancy import (
    PROMPTS_PATH,
    SOURCE_PATH,
    SampleSycophancyItem,
    prepare_prompt_variants,
    score_prompt_variant,
)
from activation_steering_study.utils.ab_prompt import ANSWER_SUFFIX, prepare_ab_prompt
from activation_steering_study.utils.qwen import MODEL_ID, MODEL_REVISION, load_qwen


EXPECTED_EXTRACTION = {
    110, 13, 20, 35, 107, 22, 69, 68, 140, 23, 36, 46, 60, 92, 61, 123, 75, 21, 62, 0, 37,
}
EXPECTED_DEVELOPMENT = {208, 27, 191, 38, 504, 432, 204, 57, 6, 3, 9, 10, 12, 51}


def _load_samples() -> list[SampleSycophancyItem]:
    return cast(
        list[SampleSycophancyItem],
        json.loads(PROMPTS_PATH.read_text(encoding="utf-8")),
    )


def _question_group(question: str) -> tuple[str, ...]:
    # NLP questions can differ only by biography, so group by the final claim.
    if "as of 2022? " in question:
        claim = question.rsplit("as of 2022? ", 1)[1].split("\n\nChoices:", 1)[0]
        return ("nlp", claim)
    choices = re.findall(r"\n \([AB]\) ([^\n]+)", question)
    assert len(choices) == 2, "Political item must have exactly two response meanings"
    return ("political", *sorted(choices))


def test_selection_matches_approved_source_rows() -> None:
    samples = _load_samples()
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    assert len(source) == 1000, "Expected the complete pinned CAA generation source"
    assert len(samples) == 35, "Expected 35 selected source rows"
    assert {
        sample["source_index"] for sample in samples if sample["split"] == "extraction"
    } == EXPECTED_EXTRACTION, "Extraction source IDs changed"
    assert {
        sample["source_index"] for sample in samples if sample["split"] == "development"
    } == EXPECTED_DEVELOPMENT, "Development source IDs changed"
    for sample in samples:
        assert {
            "question": sample["question"],
            "answer_matching_behavior": sample["answer_matching_behavior"],
            "answer_not_matching_behavior": sample["answer_not_matching_behavior"],
        } == source[sample["source_index"]], (
            f"Source {sample['source_index']} changed its question or answer labels"
        )


def test_question_groups_do_not_repeat_or_cross_splits() -> None:
    samples = _load_samples()
    extraction = [sample for sample in samples if sample["split"] == "extraction"]
    development = [sample for sample in samples if sample["split"] == "development"]
    extraction_groups = {_question_group(sample["question"]) for sample in extraction}
    development_groups = {_question_group(sample["question"]) for sample in development}

    assert len(extraction_groups) == len(extraction), "Repeated extraction question group"
    assert len(development_groups) == len(development), "Repeated development question group"
    assert extraction_groups.isdisjoint(development_groups), "Question group crosses study splits"


def test_swapping_options_preserves_answer_meaning() -> None:
    samples = _load_samples()
    for sample in samples:
        original, swapped = prepare_prompt_variants(sample)
        assert original["question"] == sample["question"], "Original source wording changed"
        assert original["question"].rsplit("\n (A) ", 1)[0] == swapped["question"].rsplit(
            "\n (A) ", 1
        )[0], f"Source {sample['source_index']} changed its question prefix under reversal"
        assert swapped["options"] == {
            "A": original["options"]["B"],
            "B": original["options"]["A"],
        }, f"Source {sample['source_index']} did not reverse option meanings"
        assert swapped["question"].endswith(
            f"\n (A) {original['options']['B']}\n (B) {original['options']['A']}"
        ), "Swapped question did not display reversed option meanings"
        assert original["options"][original["matching_label"]] == swapped["options"][
            swapped["matching_label"]
        ], f"Source {sample['source_index']} changed matching answer meaning under reversal"
    assert any(
        sample["answer_matching_behavior"] == "(B)" for sample in samples
    ), "Missing original-B case"


def test_answer_letters_add_one_token() -> None:
    # PyCharm falsely reports the Hugging Face tokenizer factory as returning None.
    # noinspection PyNoneFunctionAssignment
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    for sample in _load_samples():
        for variant in prepare_prompt_variants(sample):
            content = variant["question"] + ANSWER_SUFFIX
            expected_prompt_text = cast(
                str,
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
            )
            expected_prompt_ids = tokenizer.encode(
                expected_prompt_text, add_special_tokens=False
            )
            expected_inputs = tokenizer(
                expected_prompt_text, add_special_tokens=False, return_tensors="pt"
            )
            prepared = prepare_ab_prompt(tokenizer, variant["question"])
            case = f"Source {sample['source_index']} {variant['order']}"
            assert prepared["content"] == content, f"{case}: content changed"
            assert prepared["prompt_text"] == expected_prompt_text, (
                f"{case}: helper changed independently rendered chat text"
            )
            assert prepared["prompt_token_ids"] == expected_prompt_ids, (
                f"{case}: helper changed independently encoded prompt IDs"
            )
            assert torch.equal(prepared["inputs"]["input_ids"], expected_inputs["input_ids"]), (
                f"{case}: helper changed model input IDs"
            )
            for label in ("A", "B"):
                completed_ids = tokenizer.encode(
                    expected_prompt_text + label, add_special_tokens=False
                )
                assert prepared["answer_token_ids"][label] == completed_ids[-1], (
                    f"{case}: helper returned unexpected {label} token ID"
                )
                assert completed_ids[:-1] == expected_prompt_ids and len(
                    completed_ids
                ) == len(expected_prompt_ids) + 1, (
                    f"{case} {label}: answer breaks next-token scoring"
                )


def test_scoring_matches_next_token_probabilities() -> None:
    tokenizer, model = load_qwen()
    samples = {sample["source_index"]: sample for sample in _load_samples()}
    expected_token_ids = {
        label: tokenizer.encode(label, add_special_tokens=False)[0] for label in ("A", "B")
    }
    # Expected letters come from source labels and semantic reversal, independently of scoring.
    cases = ((208, "original", "A"), (208, "swapped", "B"), (38, "original", "B"))
    assert samples[208]["answer_matching_behavior"] == "(A)", "Source 208 label changed"
    assert samples[38]["answer_matching_behavior"] == "(B)", "Source 38 label changed"

    for source_index, variant_name, expected_label in cases:
        case = f"source {source_index} {variant_name}"
        variants = prepare_prompt_variants(samples[source_index])
        variant = next(variant for variant in variants if variant["order"] == variant_name)
        scored = score_prompt_variant(tokenizer, model, variant)
        inputs = tokenizer(scored["prompt_text"], add_special_tokens=False, return_tensors="pt")
        assert scored["answer_token_ids"] == expected_token_ids, (
            f"{case}: scorer returned unexpected answer token IDs"
        )

        with torch.no_grad():
            probabilities = model(**inputs).logits[0, -1].float().softmax(dim=-1)
        expected_a = probabilities[expected_token_ids["A"]].item()
        expected_b = probabilities[expected_token_ids["B"]].item()
        expected_matching = expected_a if expected_label == "A" else expected_b
        expected_conditional = expected_matching / (expected_a + expected_b)

        assert scored["matching_label"] == expected_label, f"{case}: wrong matching letter"
        assert scored["p_a"] == pytest.approx(expected_a, rel=0, abs=1e-8), (
            f"{case}: wrong full-vocabulary P(A)"
        )
        assert scored["p_b"] == pytest.approx(expected_b, rel=0, abs=1e-8), (
            f"{case}: wrong full-vocabulary P(B)"
        )
        assert scored["ab_mass"] == pytest.approx(expected_a + expected_b, rel=0, abs=1e-8), (
            f"{case}: wrong A/B mass"
        )
        assert scored["conditional_matching_probability"] == pytest.approx(
            expected_conditional, rel=0, abs=1e-8
        ), f"{case}: conditional probability used the wrong matching numerator"
