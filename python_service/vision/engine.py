from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .alignment import AlignmentSearcher
from .engine_types import ScoreResult
from .font_index import FontRecord
from .preprocess import extract_target_features


class FontMatchEngine:
    def __init__(self, max_workers: int) -> None:
        self.max_workers = max(1, max_workers)
        self.aligner = AlignmentSearcher()

    def prepare_target(self, crop: Any):
        return extract_target_features(crop)

    def coarse_rank(
        self,
        records: list[FontRecord],
        target,
        text: str,
        progress: Callable[[dict[str, Any]], None] | None,
    ) -> list[dict[str, Any]]:
        return self._rank(records, target, text, phase="coarse", progress=progress, fine=False)

    def fine_rank(
        self,
        records: list[FontRecord],
        target,
        text: str,
        progress: Callable[[dict[str, Any]], None] | None,
    ) -> list[dict[str, Any]]:
        return self._rank(records, target, text, phase="fine", progress=progress, fine=True)

    def _rank(
        self,
        records: list[FontRecord],
        target,
        text: str,
        phase: str,
        progress: Callable[[dict[str, Any]], None] | None,
        fine: bool,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        total = len(records)
        if total == 0:
            return results
        done = 0
        next_emit = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(self._score_record, rec, target, text, fine) for rec in records]
            for future in as_completed(futures):
                done += 1
                item = future.result()
                if item is not None:
                    results.append(item)
                percent = int(done * 100 / total)
                if progress and (percent >= next_emit or done == total):
                    progress({"phase": phase, "done": done, "total": total, "message": f"{phase} {done}/{total}"})
                    next_emit = percent + 5
        results.sort(key=lambda item: item["score"].score_total, reverse=True)
        return results

    def _score_record(self, rec: FontRecord, target, text: str, fine: bool) -> dict[str, Any] | None:
        try:
            pair = self.aligner.fine_align(rec.path, text, target) if fine else self.aligner.coarse_align(rec.path, text, target)
            if pair is None:
                return None
            rendered, score = pair
            return {"record": rec, "rendered": rendered, "score": score}
        except Exception:
            return None
