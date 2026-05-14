from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMAGE_WIDTH = 384
IMAGE_HEIGHT = 128
EMBED_DIM = 512
AI_DIR_NAME = "font_ai"
CHECKPOINT_NAME = "font_embedding.pt"
INDEX_NAME = "font_index.npz"
META_NAME = "font_index_meta.json"
TRAIN_CONFIG_NAME = "train_config.json"
TRAIN_LOG_NAME = "train_log.jsonl"


@dataclass(frozen=True)
class AIPaths:
    root: Path
    fonts: Path
    output: Path

    @property
    def ai_dir(self) -> Path:
        return self.output / AI_DIR_NAME

    @property
    def checkpoint(self) -> Path:
        return self.ai_dir / CHECKPOINT_NAME

    @property
    def index(self) -> Path:
        return self.ai_dir / INDEX_NAME

    @property
    def metadata(self) -> Path:
        return self.ai_dir / META_NAME

    @property
    def train_config(self) -> Path:
        return self.ai_dir / TRAIN_CONFIG_NAME

    @property
    def train_log(self) -> Path:
        return self.ai_dir / TRAIN_LOG_NAME


DEFAULT_TEXTS = [
    "\u5b57\u4f53\u8bc6\u522b",
    "\u54c1\u724c\u6d77\u62a5",
    "\u8bbe\u8ba1\u6807\u9898",
    "\u6c38\u548c\u4e5d\u5e74",
    "\u85cf\u5668\u8bc6\u6e56",
    "\u4e2d\u6587\u7b80\u4f53",
    "\u89c6\u89c9\u6a21\u578b",
    "\u62db\u724c\u5b57\u4f53",
]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()
