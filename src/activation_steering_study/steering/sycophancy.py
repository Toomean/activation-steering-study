"""Score fixed sycophancy prompts with the extracted and random directions."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import torch

from activation_steering_study.evaluation.sycophancy import (
    PROMPTS_PATH,
    SOURCE_PATH,
    SampleSycophancyItem,
    ScoredSycophancyVariant,
    prepare_prompt_variants,
    score_prompt_variant,
)
from activation_steering_study.extraction.sycophancy import (
    METADATA_PATH as DIRECTION_METADATA_PATH,
    SOURCE_REVISION,
    TENSOR_PATH as DIRECTION_TENSOR_PATH,
)
from activation_steering_study.steering.pilot import save_results
from activation_steering_study.steering.random_control import sample_random_direction
from activation_steering_study.utils.ab_prompt import ANSWER_SUFFIX
from activation_steering_study.utils.qwen import MODEL_ID, MODEL_REVISION, load_qwen


OUTPUT_PATH = Path("artifacts/sycophancy-pilot.json")
REFUSAL_REFERENCE_PATH = Path("artifacts/refusal-pilot.json")
NORM_MATCHED_OUTPUT_PATH = Path("artifacts/sycophancy-norm-matched.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--match-refusal-norm",
        action="store_true",
        help="scale the direction to the fixed refusal-pilot direction norm",
    )
    args = parser.parse_args()

    # Alpha and seed are fixed across the raw and norm-matched pilots.
    alpha = 1.0
    random_seed = 42

    prompt_bytes = PROMPTS_PATH.read_bytes()
    samples = cast(list[SampleSycophancyItem], json.loads(prompt_bytes))
    development = [sample for sample in samples if sample["split"] == "development"]
    selected_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    source_sha256 = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()

    direction_metadata_bytes = DIRECTION_METADATA_PATH.read_bytes()
    direction_metadata = json.loads(direction_metadata_bytes)
    direction_tensor_bytes = DIRECTION_TENSOR_PATH.read_bytes()
    direction_tensors = cast(
        dict[str, torch.Tensor],
        torch.load(DIRECTION_TENSOR_PATH, weights_only=True),
    )
    direction = direction_tensors["direction"]
    raw_direction_norm = direction.norm().item()

    assert direction_metadata["model_id"] == MODEL_ID, "Direction and pilot model IDs differ"
    assert direction_metadata["model_revision"] == MODEL_REVISION, (
        "Direction and pilot model revisions differ"
    )
    assert direction_metadata["source_revision"] == SOURCE_REVISION, (
        "Direction source revision differs from the fixed prompts"
    )
    assert direction_metadata["source_sha256"] == source_sha256, (
        "Direction source data differs from the fixed prompts"
    )
    assert direction_metadata["selected_sha256"] == selected_sha256, (
        "Direction selected data differs from the current prompt sample"
    )
    assert direction_metadata["answer_suffix"] == ANSWER_SUFFIX, (
        "Direction answer suffix differs from the current prompt renderer"
    )
    assert direction_metadata["direction_dtype"] == str(direction.dtype), (
        "Saved direction dtype differs from its metadata"
    )

    layer_index = direction_metadata["layer_index"]
    reference_norm_path: str | None = None
    reference_norm_sha256: str | None = None
    target_direction_norm: float | None = None
    scale_factor = 1.0
    if args.match_refusal_norm:
        reference_bytes = REFUSAL_REFERENCE_PATH.read_bytes()
        reference = json.loads(reference_bytes)
        for key in ("model_id", "model_revision", "model_dtype", "layer_index"):
            assert reference[key] == direction_metadata[key], (
                f"Refusal reference and sycophancy direction {key} differ"
            )
        assert reference["alpha"] == alpha, "Refusal reference alpha differs from this pilot"
        target_direction_norm = reference["direction_norm"]
        scale_factor = target_direction_norm / raw_direction_norm
        # Preserve the extracted direction's orientation while matching the fixed reference norm.
        direction = direction * scale_factor
        reference_norm_path = str(REFUSAL_REFERENCE_PATH)
        reference_norm_sha256 = hashlib.sha256(reference_bytes).hexdigest()

    direction_norm = direction.norm().item()
    if target_direction_norm is not None:
        assert abs(direction_norm - target_direction_norm) <= 1e-4, (
            "Scaled direction differs from the fixed refusal norm"
        )
    tokenizer, model = load_qwen()
    model_dtype = str(model.dtype)
    assert direction_metadata["model_dtype"] == model_dtype, (
        "Direction and loaded model dtypes differ"
    )
    random_direction = sample_random_direction(
        model.config.hidden_size, direction_norm, random_seed
    )

    conditions = {
        "baseline": (None, 0.0),
        "positive": (direction, alpha),
    }
    if not args.match_refusal_norm:
        conditions["negative"] = (direction, -alpha)
    conditions["random"] = (random_direction, alpha)
    condition_alphas = {
        name: condition[1] for name, condition in conditions.items()
    }
    results = []
    for sample in development:
        variants = prepare_prompt_variants(sample)
        source_result: dict[str, object] = {
            "source_index": sample["source_index"],
            "split": sample["split"],
        }
        condition_means = []
        for name, (condition_direction, condition_alpha) in conditions.items():
            orders: list[ScoredSycophancyVariant] = [
                score_prompt_variant(
                    tokenizer,
                    model,
                    variant,
                    direction=condition_direction,
                    alpha=condition_alpha,
                    layer_index=layer_index,
                )
                for variant in variants
            ]
            question_mean = sum(
                order["conditional_matching_probability"] for order in orders
            ) / len(orders)
            source_result[name] = {"question_mean": question_mean, "orders": orders}
            condition_means.append(f"{name}={question_mean:.4f}")
        print(f"source {sample['source_index']}: " + " ".join(condition_means))
        results.append(source_result)

    save_results(
        NORM_MATCHED_OUTPUT_PATH if args.match_refusal_norm else OUTPUT_PATH,
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_dtype": model_dtype,
            "source_revision": SOURCE_REVISION,
            "source_sha256": source_sha256,
            "selected_sha256": selected_sha256,
            "answer_suffix": ANSWER_SUFFIX,
            "direction_metadata_path": str(DIRECTION_METADATA_PATH),
            "direction_metadata_sha256": hashlib.sha256(direction_metadata_bytes).hexdigest(),
            "direction_tensor_path": str(DIRECTION_TENSOR_PATH),
            "direction_tensor_sha256": hashlib.sha256(direction_tensor_bytes).hexdigest(),
            "layer_index": layer_index,
            "hook_policy": "final prompt token",
            "direction_dtype": str(direction.dtype),
            "raw_direction_norm": raw_direction_norm,
            "scale_factor": scale_factor,
            "target_direction_norm": target_direction_norm,
            "reference_direction_path": reference_norm_path,
            "reference_direction_sha256": reference_norm_sha256,
            "direction_norm": direction_norm,
            "random_direction_norm": random_direction.norm().item(),
            "random_seed": random_seed,
            "condition_alphas": condition_alphas,
            "development_count": len(development),
        },
        results,
    )


if __name__ == "__main__":
    main()
