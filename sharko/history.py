from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class HistoryLog:
    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, record: Mapping[str, Any]) -> dict:
        stored = dict(record)
        stored["timestamp"] = datetime.now(timezone.utc).isoformat()
        line = json.dumps(stored, default=_json_default) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)
        return json.loads(line.strip())

    def read(self) -> list[dict]:
        if not self._path.is_file():
            return []
        records: list[dict] = []
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue
        return records
