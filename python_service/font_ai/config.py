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


@dataclass(frozen=True)
class AIPaths:
    root: Path
    fonts: Path
    data: Path

    @property
    def ai_dir(self) -> Path:
        return self.data / AI_DIR_NAME

    @property
    def checkpoint(self) -> Path:
        return self.ai_dir / CHECKPOINT_NAME

    @property
    def index(self) -> Path:
        return self.ai_dir / INDEX_NAME

    @property
    def metadata(self) -> Path:
        return self.ai_dir / META_NAME


DEFAULT_TEXTS = [
    "字体识别",
    "品牌海报",
    "设计标题",
    "永和九年",
    "藏器识湖",
    "中文简体",
    "视觉模型",
    "招牌字体",
]
