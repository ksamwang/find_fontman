from __future__ import annotations

import os
import sys
from pathlib import Path


def runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


ROOT = runtime_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("FONTMAN_RUNTIME_ROOT", str(ROOT))

from python_service.vision.server import main


if __name__ == "__main__":
    main()
