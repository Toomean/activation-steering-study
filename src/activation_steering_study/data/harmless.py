"""Load only the requested role of the frozen harmless metadata panel."""

import hashlib
import json
from pathlib import Path
from typing import Literal, TypedDict, cast

HarmlessRole = Literal["development", "final"]
MANIFEST_PATH = Path("data/harmless/panel.json")
MANIFEST_SHA256 = "b9d8f7cd7a9ff76af0a26be23e4c9234959d3116ed2ca49276ef9e80f94c28db"
SOURCE_ROOT = Path("data/refusal/upstream/refusal_direction")
SOURCE_NAMES: dict[HarmlessRole, str] = {"development": "harmless_val.json", "final": "harmless_test.json"}
COUNTS: dict[HarmlessRole, tuple[int, int, int]] = {"development": (63, 63, 32), "final": (246, 245, 63)}


class HarmlessRow(TypedDict):
    row_id: str
    source_path: str
    source_index: int
    instruction_sha256: str
    role: HarmlessRole
    quality_subset: bool
    semantic_group_id: str
    semantic_status: str


class HarmlessItem(HarmlessRow):
    instruction: str


class HarmlessManifest(TypedDict):
    source_revision: str
    source_sha256: dict[str, str]
    accepted_panel_sha256: str
    counts: dict[str, int]
    ordered_ids_sha256: str
    ordered_group_assignments_sha256: str
    items: list[HarmlessRow]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_harmless_manifest() -> HarmlessManifest:
    """Validate frozen bytes, order, groups, roles, and quality membership."""
    content = MANIFEST_PATH.read_bytes()
    if hashlib.sha256(content).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Frozen harmless manifest bytes changed")
    manifest = cast(HarmlessManifest, json.loads(content))
    rows = manifest["items"]
    ids = [row["row_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate harmless row IDs")
    if _digest(ids) != manifest["ordered_ids_sha256"]:
        raise ValueError("Harmless ordered ID digest mismatch")
    if _digest([[row["row_id"], row["semantic_group_id"]] for row in rows]) != manifest[
        "ordered_group_assignments_sha256"
    ]:
        raise ValueError("Harmless ordered group digest mismatch")
    groups: dict[str, set[str]] = {}
    hashes: dict[str, set[str]] = {}
    for role, expected in COUNTS.items():
        selected = [row for row in rows if row["role"] == role]
        groups[role] = {row["semantic_group_id"] for row in selected}
        hashes[role] = {row["instruction_sha256"] for row in selected}
        observed = (len(selected), len(groups[role]), sum(row["quality_subset"] for row in selected))
        declared = tuple(manifest["counts"][f"{role}_{field}"] for field in (
            "rows", "groups", "quality_rows"
        ))
        if observed != expected or declared != expected:
            raise ValueError("Frozen harmless role counts changed")
        source_path = str(SOURCE_ROOT / SOURCE_NAMES[role])
        for row in selected:
            if (row["source_path"] != source_path or type(row["source_index"]) is not int
                    or row["source_index"] < 0
                    or row["row_id"] != f"{SOURCE_NAMES[role]}:{row['source_index']}"
                    or type(row["quality_subset"]) is not bool):
                raise ValueError("Invalid harmless source or row metadata")
    if any(row["role"] not in COUNTS for row in rows):
        raise ValueError("Unknown harmless role")
    if groups["development"] & groups["final"] or hashes["development"] & hashes["final"]:
        raise ValueError("Harmless roles overlap in groups or instruction hashes")
    return manifest


def load_harmless(role: HarmlessRole) -> list[HarmlessItem]:
    """Check source bytes and each selected text hash; never open the other role's source."""
    if role not in SOURCE_NAMES:
        raise ValueError("Harmless role must be development or final")
    manifest = load_harmless_manifest()
    source_path = SOURCE_ROOT / SOURCE_NAMES[role]
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != manifest["source_sha256"][str(source_path)]:
        raise ValueError("Frozen harmless source bytes changed")
    source = json.loads(source_bytes)
    items: list[HarmlessItem] = []
    for row in manifest["items"]:
        if row["role"] != role:
            continue
        instruction = source[row["source_index"]]["instruction"]
        if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != row["instruction_sha256"]:
            raise ValueError(f"Frozen harmless instruction hash mismatch for {row['row_id']}")
        items.append({**row, "instruction": instruction})
    return items
