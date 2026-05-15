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
TRAIN_VERSION = "2"


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


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()
