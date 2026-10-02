"""Test the shared JSON artifact text format."""

from pathlib import Path

from activation_steering_study.utils.json_io import save_json


def test_saves_indented_utf8_with_final_newline(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    save_json(path, {"text": "café", "values": [1, True]})

    assert path.read_bytes() == (
        b'{\n  "text": "caf\xc3\xa9",\n  "values": [\n    1,\n    true\n  ]\n}\n'
    ), "JSON artifacts must preserve UTF-8, two-space indentation, and a final newline"


def test_saves_list_with_ascii_escapes_when_requested(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    save_json(path, ["café"], ensure_ascii=True)

    assert path.read_bytes() == b'[\n  "caf\\u00e9"\n]\n', (
        "Callers requesting ASCII must keep escaped Unicode and the same text format"
    )
