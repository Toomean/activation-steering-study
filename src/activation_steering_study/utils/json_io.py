"""Write project JSON artifacts in the shared text format."""

import json
from pathlib import Path


def save_json(path: Path, value: object, *, ensure_ascii: bool = False) -> None:
    """Write indented UTF-8 JSON with a final newline; the parent must exist."""
    path.write_text(json.dumps(value, ensure_ascii=ensure_ascii, indent=2) + "\n", encoding="utf-8")
