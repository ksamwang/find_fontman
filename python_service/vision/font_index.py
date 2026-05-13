from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .deps import ImageFont

FONT_EXTS = {".ttf", ".otf", ".ttc"}


@dataclass(frozen=True)
class FontRecord:
    path: str
    name: str
    category: str
    mtime: float
    size: int


class FontIndex:
    def __init__(self, fonts_dir: Path, index_path: Path) -> None:
        self.fonts_dir = fonts_dir
        self.index_path = index_path
        self._records: list[FontRecord] | None = None

    def ensure(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.index_path) as db:
            db.execute(
                """
                create table if not exists fonts (
                    path text primary key,
                    name text not null,
                    category text not null,
                    mtime real not null,
                    size integer not null,
                    probe_text text not null default '',
                    can_render_probe integer not null default 0
                )
                """
            )
            self._ensure_columns(db)
            existing = {
                row[0]: (row[1], row[2])
                for row in db.execute("select path, mtime, size from fonts")
            }
            seen: set[str] = set()
            for path in self.iter_font_paths():
                stat = path.stat()
                key = str(path)
                seen.add(key)
                if existing.get(key) == (stat.st_mtime, stat.st_size):
                    continue
                db.execute(
                    """
                    insert into fonts(path, name, category, mtime, size, probe_text, can_render_probe)
                    values(?, ?, ?, ?, ?, '', 0)
                    on conflict(path) do update set
                        name=excluded.name,
                        category=excluded.category,
                        mtime=excluded.mtime,
                        size=excluded.size,
                        probe_text='',
                        can_render_probe=0
                    """,
                    (key, self.read_font_name(path), self.category_for(path), stat.st_mtime, stat.st_size),
                )
            for (path,) in db.execute("select path from fonts").fetchall():
                if path not in seen:
                    db.execute("delete from fonts where path = ?", (path,))
            db.commit()
        self._records = None

    def count(self) -> int:
        with sqlite3.connect(self.index_path) as db:
            return int(db.execute("select count(*) from fonts").fetchone()[0])

    def records(self) -> list[FontRecord]:
        if self._records is None:
            with sqlite3.connect(self.index_path) as db:
                rows = db.execute(
                    "select path, name, category, mtime, size from fonts order by category, name"
                ).fetchall()
            self._records = [FontRecord(*row) for row in rows]
        return self._records

    def cached_can_render(self, record: FontRecord, text: str) -> bool | None:
        with sqlite3.connect(self.index_path) as db:
            row = db.execute(
                "select probe_text, can_render_probe from fonts where path = ?",
                (record.path,),
            ).fetchone()
        if not row or row[0] != text:
            return None
        return bool(row[1])

    def save_can_render(self, record: FontRecord, text: str, can_render: bool) -> None:
        with sqlite3.connect(self.index_path) as db:
            db.execute(
                "update fonts set probe_text = ?, can_render_probe = ? where path = ?",
                (text, 1 if can_render else 0, record.path),
            )
            db.commit()

    def _ensure_columns(self, db: sqlite3.Connection) -> None:
        columns = {row[1] for row in db.execute("pragma table_info(fonts)").fetchall()}
        if "probe_text" not in columns:
            db.execute("alter table fonts add column probe_text text not null default ''")
        if "can_render_probe" not in columns:
            db.execute("alter table fonts add column can_render_probe integer not null default 0")

    def iter_font_paths(self):
        for path in self.fonts_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in FONT_EXTS:
                yield path

    def read_font_name(self, path: Path) -> str:
        if ImageFont is not None:
            try:
                font = ImageFont.truetype(str(path), size=16)
                names = font.getname()
                joined = " ".join(part for part in names if part).strip()
                if joined:
                    return joined
            except Exception:
                pass
        return path.stem

    def category_for(self, path: Path) -> str:
        try:
            rel = path.relative_to(self.fonts_dir)
            return rel.parts[0] if rel.parts else ""
        except ValueError:
            return ""
