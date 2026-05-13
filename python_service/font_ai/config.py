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
    "\u5b57\u4f53\u8bc6\u522b",
    "\u54c1\u724c\u6d77\u62a5",
    "\u8bbe\u8ba1\u6807\u9898",
    "\u6c38\u548c\u4e5d\u5e74",
    "\u85cf\u5668\u8bc6\u6e56",
    "\u4e2d\u6587\u7b80\u4f53",
    "\u89c6\u89c9\u6a21\u578b",
    "\u62db\u724c\u5b57\u4f53",
]


def resolve_fonts_dir(root: Path, fonts: str | Path) -> Path:
    if isinstance(fonts, str) and not fonts.strip():
        fonts_root = root / "fonts"
    else:
        fonts_path = Path(fonts)
        if str(fonts_path).strip():
            return fonts_path.resolve()
        fonts_root = root / "fonts"

    if not fonts_root.exists():
        raise RuntimeError(f"fonts directory does not exist: {fonts_root}")
    simplified_dirs = sorted(path for path in fonts_root.iterdir() if path.is_dir() and path.name.startswith("1"))
    if not simplified_dirs:
        raise RuntimeError("could not find the simplified Chinese fonts directory under fonts")
    return simplified_dirs[0].resolve()
