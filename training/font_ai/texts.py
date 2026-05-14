from __future__ import annotations

import random
from dataclasses import dataclass


DEFAULT_PROBE_TEXTS = [
    "\u5b57\u4f53\u8bc6\u522b",
    "\u54c1\u724c\u6d77\u62a5",
    "\u8bbe\u8ba1\u6807\u9898",
]

COMMON_TEXTS = [
    "\u5f20\u4f1f",
    "\u738b\u82b3",
    "\u674e\u5a1c",
    "\u9648\u660e",
    "\u8d75\u4e00\u9e23",
    "\u5218\u5b50\u6db5",
    "\u6768\u601d\u8fdc",
    "\u9ec4\u96e8\u6850",
    "\u4e91\u542f\u79d1\u6280",
    "\u661f\u6cb3\u4f20\u5a92",
    "\u9752\u6728\u8bbe\u8ba1",
    "\u77e5\u884c\u6559\u80b2",
    "\u6f84\u5149\u533b\u7597",
    "\u4e07\u8c61\u96c6\u56e2",
    "\u4e1c\u65b9\u9910\u996e",
    "\u7f8e\u597d\u751f\u6d3b",
    "\u54c1\u724c\u5347\u7ea7",
    "\u65b0\u54c1\u4e0a\u5e02",
    "\u5f00\u4e1a\u5927\u5409",
    "\u9650\u65f6\u4f18\u60e0",
    "\u4f01\u4e1a\u6587\u5316",
    "\u667a\u80fd\u5236\u9020",
    "\u533b\u7597\u5065\u5eb7",
    "\u91d1\u878d\u670d\u52a1",
    "\u57ce\u5e02\u6587\u65c5",
    "Coffee Studio",
    "AI\u8bbe\u8ba1",
    "2026\u65b0\u54c1",
    "No.1 Brand",
]

SURNAMES = list("\u8d75\u94b1\u5b59\u674e\u5468\u5434\u90d1\u738b\u51af\u9648\u891a\u536b\u848b\u6c88\u97e9\u6768\u6731\u79e6\u5c24\u8bb8\u4f55\u5415\u65bd\u5f20\u5b54\u66f9\u4e25\u534e\u91d1\u9b4f\u9676\u59dc\u8c22\u90b9\u55bb\u67cf\u6c34\u7aa6\u7ae0\u4e91\u82cf\u6f58\u845b\u595a\u8303\u5f6d\u90ce")
GIVEN_CHARS = list("\u4f1f\u82b3\u5a1c\u654f\u9759\u4e3d\u5f3a\u78ca\u519b\u6d0b\u52c7\u8273\u6770\u5a1f\u6d9b\u660e\u8d85\u79c0\u971e\u5e73\u521a\u6842\u82f1\u534e\u6167\u5de7\u7f8e\u4eae\u601d\u8fdc\u96e8\u6850\u5b50\u6db5\u4e00\u9e23\u8bd7\u8bed\u6c90\u8fb0\u5b87\u8f69\u6893\u777f")
BRAND_PREFIXES = [
    "\u4e91\u542f",
    "\u661f\u6cb3",
    "\u9752\u6728",
    "\u6f84\u5149",
    "\u77e5\u884c",
    "\u4e07\u8c61",
    "\u65b0\u7a0b",
    "\u7b51\u68a6",
    "\u7d20\u9526",
    "\u5c71\u6d77",
    "\u7075\u9e7f",
    "\u68ee\u670b",
    "\u767d\u5854",
    "\u84dd\u8c37",
    "\u672a\u6765",
]
INDUSTRIES = [
    "\u79d1\u6280",
    "\u8bbe\u8ba1",
    "\u4f20\u5a92",
    "\u6559\u80b2",
    "\u9910\u996e",
    "\u533b\u7597",
    "\u91d1\u878d",
    "\u5730\u4ea7",
    "\u6587\u65c5",
    "\u7f8e\u5bb9",
    "\u670d\u9970",
    "\u9152\u5e97",
    "\u751f\u7269",
    "\u54a8\u8be2",
    "\u96f6\u552e",
    "\u7269\u6d41",
]
COMPANY_SUFFIXES = ["\u6709\u9650\u516c\u53f8", "\u96c6\u56e2", "\u5de5\u4f5c\u5ba4", "\u4e2d\u5fc3", "\u54c1\u724c", "\u5b66\u9662"]
PROMO_WORDS = [
    "\u65b0\u54c1\u4e0a\u5e02",
    "\u5f00\u4e1a\u5927\u5409",
    "\u9650\u65f6\u4f18\u60e0",
    "\u5468\u5e74\u5e86\u5178",
    "\u54c1\u724c\u5347\u7ea7",
    "\u4f1a\u5458\u4e13\u4eab",
    "\u6625\u5b63\u62db\u751f",
    "\u57ce\u5e02\u66f4\u65b0",
]
ENGLISH_WORDS = ["Coffee", "Studio", "Design", "AI", "Brand", "Center", "Space", "Plus"]


@dataclass(frozen=True)
class TextSample:
    text: str
    kind: str


class TextSampler:
    def __init__(self, fixed_texts: list[str] | None = None) -> None:
        self.fixed_texts = [text for text in fixed_texts or [] if text.strip()]

    def sample(self, rng: random.Random) -> TextSample:
        choices = ["fixed", "person", "company", "industry", "promo", "mixed", "number"]
        if not self.fixed_texts:
            choices.remove("fixed")
        kind = rng.choice(choices)
        if kind == "fixed":
            return TextSample(rng.choice(self.fixed_texts), kind)
        if kind == "person":
            return TextSample(self.person_name(rng), kind)
        if kind == "company":
            return TextSample(self.company_name(rng), kind)
        if kind == "industry":
            return TextSample(self.industry_name(rng), kind)
        if kind == "promo":
            return TextSample(rng.choice(PROMO_WORDS), kind)
        if kind == "mixed":
            return TextSample(self.mixed_text(rng), kind)
        return TextSample(self.number_text(rng), kind)

    def preview(self, limit: int = 24) -> list[str]:
        rng = random.Random(20260514)
        values = list(COMMON_TEXTS[: min(len(COMMON_TEXTS), limit // 2)])
        while len(values) < limit:
            values.append(self.sample(rng).text)
        return values[:limit]

    def probe_texts(self) -> list[str]:
        return DEFAULT_PROBE_TEXTS

    @staticmethod
    def person_name(rng: random.Random) -> str:
        length = rng.choice([1, 2])
        return rng.choice(SURNAMES) + "".join(rng.choice(GIVEN_CHARS) for _ in range(length))

    @staticmethod
    def company_name(rng: random.Random) -> str:
        return rng.choice(BRAND_PREFIXES) + rng.choice(INDUSTRIES) + rng.choice(COMPANY_SUFFIXES)

    @staticmethod
    def industry_name(rng: random.Random) -> str:
        pattern = rng.choice(["{industry}\u670d\u52a1", "{industry}\u4e2d\u5fc3", "\u4e13\u4e1a{industry}", "{brand}{industry}"])
        return pattern.format(industry=rng.choice(INDUSTRIES), brand=rng.choice(BRAND_PREFIXES))

    @staticmethod
    def mixed_text(rng: random.Random) -> str:
        pattern = rng.choice(["{en} {brand}", "{brand}{en}", "{en}\u8bbe\u8ba1", "{en} {year}"])
        return pattern.format(en=rng.choice(ENGLISH_WORDS), brand=rng.choice(BRAND_PREFIXES), year=rng.randint(2024, 2029))

    @staticmethod
    def number_text(rng: random.Random) -> str:
        pattern = rng.choice(["{year}\u65b0\u54c1", "{n}\u5468\u5e74\u5e86", "No.{n}", "{n}% OFF"])
        return pattern.format(year=rng.randint(2024, 2029), n=rng.randint(1, 99))


def read_fixed_texts(path, defaults: bool = True) -> list[str]:
    texts = list(COMMON_TEXTS) if defaults else []
    if path is not None and path.exists():
        texts.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return list(dict.fromkeys(texts))
