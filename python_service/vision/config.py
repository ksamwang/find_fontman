from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    addr: str
    root: Path
    fonts: Path
    data: Path
    previews: Path


def parse_args() -> Settings:
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="127.0.0.1:9091")
    parser.add_argument("--root", default=".")
    parser.add_argument("--fonts", default="fonts")
    parser.add_argument("--data", default="data")
    parser.add_argument("--previews", default="data/previews")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    return Settings(
        addr=args.addr,
        root=root,
        fonts=Path(args.fonts).resolve(),
        data=Path(args.data).resolve(),
        previews=Path(args.previews).resolve(),
    )
