from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .service import VisionService


@dataclass
class MatchTask:
    id: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: dict[str, Any] | None = None
    error: str = ""
    events: queue.Queue[dict[str, Any]] = field(default_factory=queue.Queue)


class TaskManager:
    def __init__(self, service: VisionService) -> None:
        self.service = service
        self.tasks: dict[str, MatchTask] = {}
        self.lock = threading.Lock()

    def start_match(self, payload: dict[str, Any]) -> str:
        task = MatchTask(id=uuid.uuid4().hex)
        with self.lock:
            self.tasks[task.id] = task
        threading.Thread(target=self._run_match, args=(task, payload), daemon=True).start()
        return task.id

    def get(self, task_id: str) -> MatchTask | None:
        with self.lock:
            return self.tasks.get(task_id)

    def _run_match(self, task: MatchTask, payload: dict[str, Any]) -> None:
        try:
            task.status = "running"
            self._emit(task, {"type": "progress", "phase": "start", "done": 0, "total": 0, "message": "match started"})
            result = self.service.match_fonts(payload, progress=lambda event: self._emit(task, {"type": "progress", **event}))
            task.result = result
            task.status = "done"
            self._emit(task, {"type": "done", "result": result})
        except Exception as exc:
            task.error = str(exc)
            task.status = "error"
            self._emit(task, {"type": "error", "error": task.error})
        finally:
            task.updated_at = time.time()

    def _emit(self, task: MatchTask, event: dict[str, Any]) -> None:
        task.updated_at = time.time()
        task.events.put(event)
