"""Extract a source-matching A/B direction at Qwen block 14."""

import hashlib
import json
from pathlib import Path
from typing import cast

import torch

from activation_steering_study.evaluation.sycophancy import (
    PROMPTS_PATH,
    SOURCE_PATH,
    SampleSycophancyItem,
    prepare_prompt_variants,
)
from activation_steering_study.extraction.paired import extract_choice_pairs
from activation_steering_study.utils.choice_prompt import ANSWER_SUFFIX
from activation_steering_study.utils.json_io import save_json
from activation_steering_study.utils.qwen import MODEL_ID, MODEL_REVISION, load_qwen


SOURCE_REVISION = "5dabbbd9a0bca5f25e174501e959de378806aa48"
METADATA_PATH = Path("artifacts/sycophancy-direction.json")
TENSOR_PATH = Path("artifacts/sycophancy-direction.pt")


def main() -> None:
    # Zero-based block-output index for this exploratory direction check.
    layer_index = 14
    samples = cast(
        list[SampleSycophancyItem], json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    )
    extraction = [sample for sample in samples if sample["split"] == "extraction"]
    tokenizer, model = load_qwen()
    extracted, rows = extract_choice_pairs(
        model, tokenizer,
        [(sample["source_index"], prepare_prompt_variants(sample)) for sample in extraction],
        [layer_index],
    )
    tensors = extracted[layer_index]
    pair_differences = tensors["pair_differences"]
    direction = tensors["direction"]
    metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_dtype": str(model.dtype),
        "layer_index": layer_index,
        "capture_dtype": str(pair_differences.dtype),
        "direction_dtype": str(direction.dtype),
        "pair_formula": "matching_label activation - opposite_label activation",
        "aggregation": "mean of two orders per source, then mean of extraction sources",
        "source_revision": SOURCE_REVISION,
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "selected_sha256": hashlib.sha256(PROMPTS_PATH.read_bytes()).hexdigest(),
        "answer_suffix": ANSWER_SUFFIX,
        "extraction_count": len(extraction),
        "direction_norm": direction.norm().item(),
        "tensor_file": TENSOR_PATH.name,
        "rows": rows,
    }
    TENSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, TENSOR_PATH)
    save_json(METADATA_PATH, metadata, ensure_ascii=True)
    print(f"block {layer_index}: {len(extraction)} sources, direction norm {metadata['direction_norm']:.6f}")
    print(f"saved {METADATA_PATH} and {TENSOR_PATH}")


if __name__ == "__main__":
    main()
