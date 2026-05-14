from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: dict[str, Any]) -> None:
        raw = json.dumps(event, ensure_ascii=False)
        print(raw, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(raw + "\n")
