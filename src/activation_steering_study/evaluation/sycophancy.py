"""Score biography-aligned A/B answers on the fixed sycophancy development sample."""

import hashlib
import json
import re
from pathlib import Path
from typing import TypedDict, cast

from activation_steering_study.evaluation.choices import ChoicePromptVariant, prepare_variants, score_choice_variant
from activation_steering_study.utils.choice_prompt import ANSWER_SUFFIX
from activation_steering_study.utils.json_io import save_json
from activation_steering_study.utils.qwen import MODEL_ID, MODEL_REVISION, load_qwen


_SYCOPHANCY_DATA_DIR = Path("data/sycophancy")
PROMPTS_PATH = _SYCOPHANCY_DATA_DIR / "prompts.json"
SOURCE_PATH = _SYCOPHANCY_DATA_DIR / "upstream/caa/generate_dataset.json"
OUTPUT_PATH = Path("artifacts/sycophancy-baseline.json")
CHOICES = re.compile(r"\n \(A\) ([^\n]+)\n \(B\) ([^\n]+)$")


class SampleSycophancyItem(TypedDict):
    source_index: int
    split: str
    question: str
    answer_matching_behavior: str
    answer_not_matching_behavior: str

def prepare_prompt_variants(sample: SampleSycophancyItem) -> list[ChoicePromptVariant]:
    """Parse CAA's terminal options while preserving its original prompt exactly."""
    match = CHOICES.search(sample["question"])
    if match is None:
        raise ValueError(f"Missing terminal A/B options in source {sample['source_index']}")
    a_text, b_text = match.groups()
    return prepare_variants(
        sample["question"][:match.start()], a_text, b_text,
        sample["answer_matching_behavior"][1], original_question=sample["question"],
    )


def main() -> None:
    samples = cast(
        list[SampleSycophancyItem], json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    )
    development = [sample for sample in samples if sample["split"] == "development"]
    tokenizer, model = load_qwen()
    results = []
    for sample in development:
        variants = [
            score_choice_variant(tokenizer, model, variant)
            for variant in prepare_prompt_variants(sample)
        ]
        question_mean = sum(
            variant["conditional_matching_probability"] for variant in variants
        ) / 2
        print(
            f"source {sample['source_index']}: "
            f"original={variants[0]['conditional_matching_probability']:.4f} "
            f"swapped={variants[1]['conditional_matching_probability']:.4f} "
            f"mean={question_mean:.4f}"
        )
        results.append(
            {
                "source_index": sample["source_index"],
                "split": sample["split"],
                "source_question": sample["question"],
                "source_matching_label": sample["answer_matching_behavior"],
                "question_mean": question_mean,
                "orders": variants,
            }
        )
    mean = sum(cast(float, result["question_mean"]) for result in results) / len(results)
    artifact = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_dtype": str(model.dtype),
        "source_revision": "5dabbbd9a0bca5f25e174501e959de378806aa48",
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "selected_sha256": hashlib.sha256(PROMPTS_PATH.read_bytes()).hexdigest(),
        "answer_suffix": ANSWER_SUFFIX,
        "method": (
            "Mean of two conditional matching probabilities per question, "
            f"then mean over {len(results)} development questions"
        ),
        "development_count": len(results),
        "mean_conditional_matching_probability": mean,
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_PATH, artifact)
    print(f"development mean ({len(results)} questions, two orders each): {mean:.4f}")
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
