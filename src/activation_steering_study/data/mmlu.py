"""Load the MMLU calibration sample and frozen final panel from local sources."""

import hashlib
import json
import random
from pathlib import Path
from typing import TypedDict

import pyarrow.parquet as pq  # type: ignore[import-untyped]


SOURCE_PATH = Path("data/mmlu/upstream/cais/validation-00000-of-00001.parquet")
FINAL_SOURCE_PATH = Path("data/mmlu/upstream/cais/test-00000-of-00001.parquet")
FINAL_MANIFEST_PATH = Path("data/mmlu/test-1710.json")
# These digests bind the final loader to the accepted source bytes and ordered IDs.
FINAL_SOURCE_SHA256 = "74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7"
FINAL_IDS_SHA256 = "afe28e6a48069f170728ab00452d6841b2d0330a461bc8b0e7bc6d36cc2a6bc7"
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


def load_mmlu_final() -> list[MmluItem]:
    """Load 1710 explicit IDs in manifest order; reject changed source or content."""
    manifest = json.loads(FINAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest["items"]
    ids = [entry["id"] for entry in entries]
    ids_digest = hashlib.sha256(
        json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if len(entries) != 1710 or ids_digest != FINAL_IDS_SHA256:
        raise ValueError("Frozen MMLU final IDs changed")
    if manifest["ordered_ids_sha256"] != ids_digest:
        raise ValueError("MMLU final manifest ID digest mismatch")
    if (
        manifest["source_sha256"] != FINAL_SOURCE_SHA256
        or hashlib.sha256(FINAL_SOURCE_PATH.read_bytes()).hexdigest() != FINAL_SOURCE_SHA256
    ):
        raise ValueError("Frozen MMLU test source bytes changed")

    # Legacy CSV-style indices mean subject-relative order in this pinned Parquet.
    source_items: dict[str, MmluItem] = {}
    subject_counts: dict[str, int] = {}
    for row in pq.read_table(FINAL_SOURCE_PATH).to_pylist():
        subject = row["subject"]
        index = subject_counts.get(subject, 0)
        subject_counts[subject] = index + 1
        item_id = f"{subject}_test.csv:{index}"
        source_items[item_id] = {
            "id": item_id,
            "subject": subject,
            "source_index": index,
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
        }

    items: list[MmluItem] = []
    for entry in entries:
        item = source_items[entry["id"]]
        content_digest = hashlib.sha256(
            json.dumps(
                [item["question"], item["choices"], item["answer"]],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if content_digest != entry["content_sha256"]:
            raise ValueError(f"Frozen MMLU content hash mismatch for {item['id']}")
        items.append(item)
    return items
