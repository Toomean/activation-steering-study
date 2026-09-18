"""Sample refusal pilot prompts from copied upstream inputs.

The pinned notebook documents upstream split construction:
https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/generate_datasets.ipynb
This study defines the 96-row sampling below; overlap invariants are tested separately.
"""

import json
import random
from pathlib import Path
from typing import Literal, TypedDict

type SampleSplit = Literal["train", "validation"]
type SampleType = Literal["harmful", "harmless"]
type SampleGroup = tuple[SampleSplit, SampleType, str, int]

class SampleRefusalItem(TypedDict):
    sample_path: str
    sample_index: int
    instruction: str

class SampleRefusalSplit(TypedDict):
    harmful: list[SampleRefusalItem]
    harmless: list[SampleRefusalItem]

class SampleRefusalResult(TypedDict):
    source_revision: str
    seed: int
    train: SampleRefusalSplit
    validation: SampleRefusalSplit

_REFUSAL_DATA_DIR = Path("data/refusal")
_REFUSAL_UPSTREAM_DIR = _REFUSAL_DATA_DIR / "upstream" / "refusal_direction"


def _read_upstream_json(relative_path: str):
    datasets_path = _REFUSAL_UPSTREAM_DIR / relative_path

    return json.loads(datasets_path.read_text())


def sample_refusal(seed: int = 42) -> SampleRefusalResult:
    """Return the 64 extraction and 32 development prompts for one sampling seed."""
    result: SampleRefusalResult = {
        "source_revision": "9d852fae1a9121c78b29142de733cb1340770cc3",
        "seed": seed,
        "train": {"harmful": [], "harmless": []},
        "validation": {"harmful": [], "harmless": []},
    }

    groups: tuple[SampleGroup, ...] = (
        ("train", "harmful", "harmful_train.json", 32),
        ("train", "harmless", "harmless_train.json", 32),
        ("validation", "harmful", "harmful_val.json", 16),
        ("validation", "harmless", "harmless_val.json", 16),
    )

    for split, sample_type, sample_path, sample_count in groups:
        sample_json = _read_upstream_json(sample_path)

        selected_indices = random.Random(seed).sample(range(len(sample_json)), sample_count)
        # A fresh RNG keeps pools independent; sorting restores source order.
        selected_indices = sorted(selected_indices)

        result[split][sample_type] = [
            {
                "sample_path": sample_path,
                "sample_index": sample_index,
                "instruction": sample_json[sample_index]["instruction"],
            }
            for sample_index in selected_indices
        ]
    return result


if __name__ == "__main__":
    output = _REFUSAL_DATA_DIR / "prompts.json"
    output.write_text(
        json.dumps(sample_refusal(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved 96 prompts to {output}")
