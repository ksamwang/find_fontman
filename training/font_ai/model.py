from __future__ import annotations

import math
from contextlib import nullcontext

from .config import EMBED_DIM
from .deps import F, nn, torch


class SmallFontCNN(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.features = nn.Sequential(
            block(3, 32, stride=2),
            block(32, 64, stride=2),
            block(64, 128, stride=2),
            block(128, 256, stride=2),
            block(256, 384, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(384, embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return F.normalize(x, dim=1, eps=1e-6)


def block(in_channels: int, out_channels: int, stride: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.35) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        context = torch.cuda.amp.autocast(enabled=False) if embeddings.is_cuda else nullcontext()
        with context:
            embeddings = embeddings.float()
            weight = self.weight.float()
            cosine = F.linear(F.normalize(embeddings, eps=1e-6), F.normalize(weight, eps=1e-6))
            cosine = cosine.clamp(-1.0 + 1e-4, 1.0 - 1e-4)
            sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-6))
            target = cosine * self.cos_m - sine * self.sin_m
            target = torch.where(cosine > self.threshold, target, cosine - self.mm)
            one_hot = F.one_hot(labels, num_classes=cosine.size(1)).to(dtype=cosine.dtype, device=cosine.device)
            logits = cosine * (1.0 - one_hot) + target * one_hot
            return logits * self.scale


class FontEmbeddingModel(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.backbone = SmallFontCNN(embed_dim)
        self.head = ArcFaceHead(embed_dim, num_classes)

    def forward(self, x, labels=None):
        embeddings = self.backbone(x)
        if labels is None:
            return embeddings
        return self.head(embeddings, labels)


def create_model(num_classes: int, embed_dim: int = EMBED_DIM) -> FontEmbeddingModel:
    return FontEmbeddingModel(num_classes=num_classes, embed_dim=embed_dim)
