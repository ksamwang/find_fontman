from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import AIPaths
from .dataset import read_metadata
from .deps import torch
from .transforms import image_to_tensor


class FontEmbeddingMatcher:
    def __init__(self, root: Path, fonts: Path, output: Path, device: str = "") -> None:
        self.paths = AIPaths(root=root, fonts=fonts, output=output)
        self.available = False
        self.model = None
        self.records: list[dict[str, Any]] = []
        self.embeddings = None
        self.device_name = device
        self.load()

    def load(self) -> None:
        if torch is None:
            return
        if not self.paths.checkpoint.exists() or not self.paths.index.exists() or not self.paths.metadata.exists():
            return
        from .model import create_model

        device = torch.device(self.device_name if self.device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(self.paths.checkpoint, map_location=device)
        metadata = read_metadata(self.paths.metadata)
        index = np.load(self.paths.index)
        model = create_model(num_classes=int(ckpt["num_classes"])).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self.device = device
        self.model = model
        self.records = metadata["fonts"]
        self.embeddings = index["embeddings"].astype(np.float32)
        self.available = True

    def match(self, image, top_k: int = 10) -> list[dict[str, Any]]:
        if not self.available:
            raise RuntimeError("font embedding model or index is not available")
        query = self.embed_image(image)
        scores = self.embeddings @ query
        order = np.argsort(-scores)[:top_k]
        return [
            {
                "font_name": self.records[int(idx)]["name"],
                "font_path": self.records[int(idx)]["path"],
                "embedding_score": float(scores[int(idx)]),
            }
            for idx in order
        ]

    def embed_image(self, image):
        tensor = image_to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vector = self.model(tensor).cpu().numpy()[0].astype(np.float32)
        return vector / max(1e-12, float(np.linalg.norm(vector)))
