from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .config import parse_args
from .service import VisionService


class Handler(BaseHTTPRequestHandler):
    service: VisionService

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.write_json(self.service.health())
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

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[vision] " + fmt % args + "\n")


def main() -> None:
    settings = parse_args()
    Handler.service = VisionService(settings)
    host, port_text = settings.addr.rsplit(":", 1)
    server = ThreadingHTTPServer((host, int(port_text)), Handler)
    print(f"[vision] listening on http://{settings.addr}")
    server.serve_forever()
