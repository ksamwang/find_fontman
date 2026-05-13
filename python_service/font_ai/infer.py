from __future__ import annotations

import json
import base64
import hashlib
import io
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .config import AIPaths
from .dataset import read_metadata
from .deps import torch
from .transforms import image_to_tensor
from python_service.vision.deps import np


class FontEmbeddingMatcher:
    def __init__(self, root: Path, fonts: Path, data: Path, device: str = "") -> None:
        self.paths = AIPaths(root=root, fonts=fonts, data=data)
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

    def match(self, image, text: str = "", top_k: int = 10, top_n: int = 100) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("font embedding model or index is not available")
        query = self.embed_image(image)
        scores = self.embeddings @ query
        order = np.argsort(-scores)[: max(top_k, top_n)]
        results = []
        for idx in order[:top_k]:
            record = self.records[int(idx)]
            results.append(
                {
                    "font_name": record["name"],
                    "font_path": record["path"],
                    "score_total": float(scores[int(idx)]),
                    "embedding_score": float(scores[int(idx)]),
                    "match_mode": "embedding",
                    "preview_path": "",
                    "preview_base64": self.preview_base64(record["path"], text),
                    "preview_mime": "image/png",
                }
            )
        return {
            "results": results,
            "candidate_size": int(len(self.records)),
            "elapsed_ms": 0,
            "warning": "",
            "match_mode": "embedding",
            "top_indices": [int(i) for i in order[:top_n]],
        }

    def embed_image(self, image):
        tensor = image_to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vector = self.model(tensor).cpu().numpy()[0].astype(np.float32)
        return vector / max(1e-12, float(np.linalg.norm(vector)))

    def preview_base64(self, font_path: str, text: str) -> str:
        text = text.strip() or "Font"
        try:
            font = ImageFont.truetype(font_path, 56)
            bbox = font.getbbox(text)
            width = max(240, bbox[2] - bbox[0] + 48)
            height = max(96, bbox[3] - bbox[1] + 36)
            image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(image)
            x = (width - (bbox[2] - bbox[0])) // 2 - bbox[0]
            y = (height - (bbox[3] - bbox[1])) // 2 - bbox[1]
            draw.text((x, y), text, font=font, fill="black")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return ""
