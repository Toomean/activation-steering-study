"""Frozen harmless metadata and role-specific no-model loading checks."""

import hashlib
import json
from pathlib import Path

import pytest

from activation_steering_study.data import harmless


def test_frozen_roles_counts_and_order() -> None:
    manifest = harmless.load_harmless_manifest()
    for role, expected in harmless.COUNTS.items():
        items = harmless.load_harmless(role)
        assert (len(items), len({row["semantic_group_id"] for row in items}),
                sum(row["quality_subset"] for row in items)) == expected
        assert [row["row_id"] for row in items] == [
            row["row_id"] for row in manifest["items"] if row["role"] == role
        ], "Loader must preserve accepted manifest order"
    assert manifest["ordered_ids_sha256"] == "164299bccb3ca2bf565b4609b094885d8e03608d48f86315a4934437d532ed3f"
    assert manifest["ordered_group_assignments_sha256"] == "de824087ccdec8716213ef4c3596e41b164d47aa3a2e1fe5a14fcfd74f0ebf9c"
    assert manifest["accepted_panel_sha256"] == "8bda452f7e5e7553ec1e679f4ca20788dacbee1e62476dbc121038d2961dc161"


def test_development_never_opens_final_source(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_bytes

    def read(path):
        assert path.name != "harmless_test.json", "Development must not read final source bytes"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    assert len(harmless.load_harmless("development")) == 63
    with pytest.raises(ValueError, match="role"):
        harmless.load_harmless("validation")  # type: ignore[arg-type]


def test_changed_manifest_bytes_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "panel.json"
    path.write_bytes(harmless.MANIFEST_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(harmless, "MANIFEST_PATH", path)
    with pytest.raises(ValueError, match="manifest bytes"):
        harmless.load_harmless("development")


@pytest.mark.parametrize("change,error", [("group", "group digest"), ("text_hash", "instruction hash"),
                                         ("quality", "counts"), ("source", "source or row")])
def test_changed_metadata_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: str, error: str,
) -> None:
    # Rebind only the outer byte digest to exercise independent metadata checks.
    manifest = json.loads(harmless.MANIFEST_PATH.read_bytes())
    row = manifest["items"][0]
    if change == "group":
        row["semantic_group_id"] = "invented-group"
    elif change == "text_hash":
        row["instruction_sha256"] = "0" * 64
    elif change == "quality":
        row["quality_subset"] = False
    else:
        row["source_path"] = "data/other.json"
    content = json.dumps(manifest).encode()
    path = tmp_path / "panel.json"
    path.write_bytes(content)
    monkeypatch.setattr(harmless, "MANIFEST_PATH", path)
    monkeypatch.setattr(harmless, "MANIFEST_SHA256", hashlib.sha256(content).hexdigest())
    with pytest.raises(ValueError, match=error):
        harmless.load_harmless("development")


def test_source_bytes_changed_before_deserialization(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_bytes

    def read(path):
        content = original(path)
        return content + b"\n" if path.name == "harmless_val.json" else content

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ValueError, match="source bytes"):
        harmless.load_harmless("development")
