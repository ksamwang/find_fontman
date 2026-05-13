from __future__ import annotations

import json
import queue
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .config import parse_args
from .service import VisionService
from .tasks import TaskManager


class Handler(BaseHTTPRequestHandler):
    service: VisionService
    tasks: TaskManager

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.write_json(self.service.health())
            return
        if parsed.path.startswith("/match/events/"):
            self.stream_match_events(parsed.path.rsplit("/", 1)[-1])
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            payload = self.read_json()
            parsed = urlparse(self.path)
            if parsed.path == "/analyze":
                self.write_json(self.service.analyze_crop(payload))
            elif parsed.path == "/match":
                self.write_json(self.service.match_fonts(payload))
            elif parsed.path == "/match/start":
                self.write_json({"task_id": self.tasks.start_match(payload)})
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(str(exc).encode("utf-8"))

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def write_json(self, value: dict[str, Any]) -> None:
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def stream_match_events(self, task_id: str) -> None:
        task = self.tasks.get(task_id)
        if task is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        while True:
            try:
                event = task.events.get(timeout=15)
            except queue.Empty:
                event = {"type": "heartbeat"}
            raw = f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
            try:
                self.wfile.write(raw)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            if event.get("type") in {"done", "error"}:
                return

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[vision] " + fmt % args + "\n")


def main() -> None:
    settings = parse_args()
    Handler.service = VisionService(settings)
    Handler.tasks = TaskManager(Handler.service)
    host, port_text = settings.addr.rsplit(":", 1)
    server = ThreadingHTTPServer((host, int(port_text)), Handler)
    print(f"[vision] listening on http://{settings.addr}")
    server.serve_forever()
