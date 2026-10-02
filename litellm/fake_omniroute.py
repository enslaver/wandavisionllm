"""Stand-in OmniRoute for tests: logs model + headers, answers OpenAI chat (stream or not).

    python3 fake_omniroute.py [PORT] [LOG]
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 4199
LOG = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fake-omniroute.jsonl"


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        with open(LOG, "a") as f:
            f.write(json.dumps({"path": self.path, "model": body.get("model"), "stream": body.get("stream"),
                                "x_session_id": self.headers.get("X-Session-Id"),
                                "auth": bool(self.headers.get("Authorization"))}) + "\n")
        text = f"fake-omniroute answered as {body.get('model')}"
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in ({"role": "assistant", "content": text}, {}):
                d = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": body.get("model"),
                     "choices": [{"index": 0, "delta": chunk, "finish_reason": None if chunk else "stop"}]}
                self.wfile.write(f"data: {json.dumps(d)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        out = {"id": "x", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
               "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
