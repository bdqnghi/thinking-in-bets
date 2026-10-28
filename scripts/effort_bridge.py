"""HTTP proxy shim that maps ``model@<effort>`` suffixes and ``reasoning_effort=xhigh``
onto Gemini ``thinking_config.thinking_budget`` parameters for OpenAI-compatible and
native Gemini v1beta endpoints.

Run: ``python scripts/effort_bridge.py --port 8767 --upstream https://generativelanguage.googleapis.com``
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EFFORTS = ("low", "medium", "high", "xhigh")
XHIGH_BUDGET = 65535
# Native-path budgets for low/medium/high: Gemini's documented OpenAI-compat mapping (low 1024, medium 8192,
# high 24576); xhigh = the maximum budget.
NATIVE_BUDGET = {"low": 1024, "medium": 8192, "high": 24576, "xhigh": XHIGH_BUDGET}
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding"}
_NATIVE_RE = re.compile(r"^(?P<pre>.*/models/)(?P<model>[^/:]+)(?P<post>:[A-Za-z]+.*)$")


def split_effort(model: str) -> tuple[str, str | None]:
    if "@" in model:
        base, eff = model.rsplit("@", 1)
        if eff in EFFORTS:
            return base, eff
        raise ValueError(f"unknown effort suffix {eff!r} (expected one of {EFFORTS})")
    return model, None


def rewrite_openai(body: dict) -> tuple[dict, dict]:
    """Returns (new body, log info). Pure function (unit-tested)."""
    model = str(body.get("model", ""))
    base, suffix = split_effort(model)
    eff = suffix or body.get("reasoning_effort")
    out = dict(body)
    out["model"] = base
    info = {"model_in": model, "model": base, "effort": eff, "path": "openai"}
    if eff == "xhigh":
        out.pop("reasoning_effort", None)
        eb = dict(out.get("extra_body") or {})
        g = dict(eb.get("google") or {})
        tc = dict(g.get("thinking_config") or {})
        tc["thinking_budget"] = XHIGH_BUDGET
        g["thinking_config"] = tc
        eb["google"] = g
        out["extra_body"] = eb
        info["thinking_budget"] = XHIGH_BUDGET
    elif eff is not None:
        if eff not in EFFORTS:
            raise ValueError(f"unknown reasoning_effort {eff!r}")
        out["reasoning_effort"] = eff
    return out, info


def rewrite_native(path: str, body: dict | None) -> tuple[str, dict | None, dict]:
    m = _NATIVE_RE.match(path)
    if not m:
        return path, body, {"path": "native", "rewritten": False}
    raw_model = urllib.parse.unquote(m.group("model"))
    base, eff = split_effort(raw_model)
    new_path = m.group("pre") + urllib.parse.quote(base, safe="-._~") + m.group("post")
    info = {"path": "native", "model_in": raw_model, "model": base, "effort": eff}
    if eff is None or body is None:
        return new_path, body, info
    out = dict(body)
    gc = dict(out.get("generationConfig") or {})
    tc = dict(gc.get("thinkingConfig") or {})
    if "thinkingBudget" not in tc and "thinkingLevel" not in tc:
        tc["thinkingBudget"] = NATIVE_BUDGET[eff]
    gc["thinkingConfig"] = tc
    out["generationConfig"] = gc
    info["thinking_budget"] = tc.get("thinkingBudget")
    return new_path, out, info


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream: urllib.parse.ParseResult
    log_path: Path | None = None
    log_lock = threading.Lock()

    def log_message(self, fmt, *args):  # silence default stderr access log
        pass

    def _log(self, rec: dict) -> None:
        if self.log_path is None:
            return
        rec = dict(rec, t=round(time.time(), 3))
        with self.log_lock, open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _send_error(self, code: int, msg: str) -> None:
        data = json.dumps({"error": {"code": code, "message": msg, "status": "INVALID_ARGUMENT"}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _forward(self, method: str) -> None:
        t0 = time.time()
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        path = self.path
        info: dict = {"method": method, "path_in": path.split("?")[0]}
        try:
            if method == "POST" and raw:
                ctype = self.headers.get("Content-Type", "")
                if "json" in ctype or raw[:1] == b"{":
                    body = json.loads(raw)
                    if path.split("?")[0].endswith("/chat/completions"):
                        body, i2 = rewrite_openai(body)
                        info.update(i2)
                        raw = json.dumps(body).encode()
                    elif "/models/" in path:
                        path, body, i2 = rewrite_native(path, body)
                        info.update(i2)
                        raw = json.dumps(body).encode()
        except ValueError as e:
            self._log(dict(info, status=400, error=str(e)))
            return self._send_error(400, str(e))
        up = self.upstream
        conn_cls = http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(up.hostname, up.port, timeout=1800)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        if raw:
            headers["Content-Length"] = str(len(raw))
        try:
            conn.request(method, path, body=raw if raw else None, headers=headers)
            resp = conn.getresponse()
        except Exception as e:  # upstream down (bridge restart): 502 so clients retry
            self._log(dict(info, status=502, error=f"{type(e).__name__}: {e}"[:300]))
            return self._send_error(502, f"upstream error: {e}")
        try:
            self.send_response(resp.status)
            chunked = False
            for k, v in resp.getheaders():
                kl = k.lower()
                if kl in HOP:
                    continue
                self.send_header(k, v)
            clen = resp.getheader("Content-Length")
            if clen is not None:
                self.send_header("Content-Length", clen)
            else:
                chunked = True
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                if not chunk:
                    break
                if chunked:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                else:
                    self.wfile.write(chunk)
                self.wfile.flush()
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()
            if info.get("effort") is not None or info.get("model_in", "").count("@"):
                self._log(dict(info, status=resp.status, seconds=round(time.time() - t0, 2)))

    def do_POST(self):
        self._forward("POST")

    def do_GET(self):
        self._forward("GET")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hosts", default="127.0.0.1,172.17.0.1")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--upstream", default="https://generativelanguage.googleapis.com")
    ap.add_argument("--log", default=str(Path.home() / ".cache/tib/effort_shim.jsonl"))
    a = ap.parse_args()
    Handler.upstream = urllib.parse.urlparse(a.upstream)
    Handler.log_path = Path(a.log)
    Handler.log_path.parent.mkdir(parents=True, exist_ok=True)
    servers = []
    for h in a.hosts.split(","):
        try:
            s = ThreadingHTTPServer((h.strip(), a.port), Handler)
        except OSError as e:
            print(f"cannot bind {h}:{a.port}: {e}", file=sys.stderr, flush=True)
            continue
        s.daemon_threads = True
        threading.Thread(target=s.serve_forever, daemon=True).start()
        servers.append(s)
        print(f"effort shim listening on {h}:{a.port} -> {a.upstream}", flush=True)
    if not servers:
        raise SystemExit("no listener bound")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
