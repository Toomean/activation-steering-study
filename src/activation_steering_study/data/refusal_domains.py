"""Load the frozen, provisional three-domain refusal extraction split."""

import hashlib
import json
from pathlib import Path

from activation_steering_study.data.refusal import SampleRefusalItem, load_reviewed_refusal


SPLIT_PATH = Path("data/refusal/domain-splits.json")
LABELS_PATH = Path("data/refusal/domain-labels.json")
SOURCE_PATH = Path("data/refusal/upstream/refusal_direction/harmful_train.json")


def load_domain_extraction() -> tuple[dict[str, list[SampleRefusalItem]], list[SampleRefusalItem]]:
    """Bind selected source IDs to the copied upstream instructions."""
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    if hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest() != labels["policy"]["source_sha256"]:
        raise ValueError("Frozen domain labels refer to different upstream source bytes")
    labelled = {row["source_index"]: row for row in labels["rows"]}
    domains: dict[str, list[SampleRefusalItem]] = {}
    seen: set[int] = set()
    for name, selection in split["domains"].items():
        indices = selection["train_extraction_indices"]
        if len(indices) != 24 or len(set(indices)) != 24 or seen.intersection(indices):
            raise ValueError(f"Invalid frozen extraction IDs for {name}")
        if any(labelled[index]["domain"] != name or not labelled[index]["domain_pilot_eligible"]
               or labelled[index]["instruction"] != source[index]["instruction"] for index in indices):
            raise ValueError(f"Frozen extraction rows disagree with domain labels: {name}")
        seen.update(indices)
        domains[name] = [
            {"sample_path": SOURCE_PATH.name, "sample_index": index,
             "instruction": source[index]["instruction"]}
            for index in indices
        ]
    if set(domains) != {"cyber_intrusion", "dangerous_substances", "disinformation"}:
        raise ValueError("Frozen refusal domains changed")
    harmless = load_reviewed_refusal()["train"]["harmless"]
    if len(harmless) != 32:
        raise ValueError("Expected 32 reviewed harmless extraction rows")
    return domains, harmless


def source_hashes() -> dict[str, str]:
    paths = (SPLIT_PATH, LABELS_PATH, SOURCE_PATH, Path("data/review/prompts_checked.json"))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
