"""Recreate the fixed MMLU validation sample from its copied CAIS Parquet source."""

import random
from pathlib import Path
from typing import TypedDict

import pyarrow.parquet as pq  # type: ignore[import-untyped]


SOURCE_PATH = Path("data/mmlu/upstream/cais/validation-00000-of-00001.parquet")
LABELS = "ABCD"


class MmluItem(TypedDict):
    id: str
    subject: str
    source_index: int
    question: str
    choices: list[str]
    answer: int


def sample_mmlu() -> list[MmluItem]:
    """Select five source rows per subject with the calibration sample's one RNG."""
    # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html
    source_rows = pq.read_table(SOURCE_PATH).to_pylist()
    # The calibration screen used sorted subjects and one continuing RNG.
    rng = random.Random(42)
    items: list[MmluItem] = []
    for subject in sorted({row["subject"] for row in source_rows}):
        rows = [row for row in source_rows if row["subject"] == subject]
        for index in sorted(rng.sample(range(len(rows)), 5)):
            row = rows[index]
            # Existing calibration artifacts refer to these CSV-style IDs.
            items.append(
                {
                    "id": f"{subject}_val.csv:{index}",
                    "subject": subject,
                    "source_index": index,
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": row["answer"],
                }
            )
    return items
