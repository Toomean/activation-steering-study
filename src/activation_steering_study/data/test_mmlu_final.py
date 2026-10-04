"""Check the accepted final panel and its source/content integrity guards."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from activation_steering_study.data import mmlu


def test_final_panel_loads_accepted_ids_without_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_rng(*args: object, **kwargs: object) -> None:
        pytest.fail("Frozen final panel must load explicit IDs without sampling")

    monkeypatch.setattr(mmlu.random, "Random", unexpected_rng)
    items = mmlu.load_mmlu_final()
    assert len(items) == 1710, "Expected the accepted 1710-row final panel"
    counts = Counter(item["subject"] for item in items)
    assert len(counts) == 57 and set(counts.values()) == {30}, (
        "Expected exactly 30 final rows per subject"
    )
    assert hashlib.sha256(
        json.dumps([item["id"] for item in items], ensure_ascii=False, separators=(",", ":"))
        .encode("utf-8")
    ).hexdigest() == "afe28e6a48069f170728ab00452d6841b2d0330a461bc8b0e7bc6d36cc2a6bc7", (
        "Final ordered IDs differ from the accepted ordered-ID digest"
    )


@pytest.mark.parametrize("change", ["id", "id_digest", "content", "source_digest"])
def test_final_panel_rejects_changed_manifest(
    change: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = json.loads(mmlu.FINAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if change == "id":
        manifest["items"][0]["id"] = "abstract_algebra_test.csv:999"
    elif change == "id_digest":
        manifest["ordered_ids_sha256"] = "0" * 64
    elif change == "content":
        manifest["items"][0]["content_sha256"] = "0" * 64
    else:
        manifest["source_sha256"] = "0" * 64
    changed_manifest = tmp_path / "manifest.json"
    changed_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(mmlu, "FINAL_MANIFEST_PATH", changed_manifest)
    messages = {
        "id": "final IDs changed",
        "id_digest": "manifest ID digest mismatch",
        "content": "content hash mismatch",
        "source_digest": "source bytes changed",
    }
    with pytest.raises(ValueError, match=messages[change]):
        mmlu.load_mmlu_final()


def test_final_panel_rejects_changed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed_source = tmp_path / "test.parquet"
    changed_source.write_bytes(b"changed source bytes")
    monkeypatch.setattr(mmlu, "FINAL_SOURCE_PATH", changed_source)
    with pytest.raises(ValueError, match="source bytes changed"):
        mmlu.load_mmlu_final()
