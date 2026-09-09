"""Small, local-only HTTP lab for running the maintained component harnesses."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .harnesses import HARNESSES, HERE, all_specs

ROOT = HERE.parents[1]
RUN_ROOT = ROOT / ".live" / "lab-runs"
PAGE = Path(__file__).with_name("index.html")


@dataclass
class Run:
    id: str
    harness: str
    version: str
    command: list[str]
    output: str
    started: float = field(default_factory=time.time)
    finished: float | None = None
    returncode: int | None = None
    status: str = "queued"
    log: str = ""
    validation: dict | None = None

    def public(self) -> dict:
        value = asdict(self)
        value["elapsed_seconds"] = round((self.finished or time.time()) - self.started, 2)
        return value


class RunManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.runs: dict[str, Run] = {}
        RUN_ROOT.mkdir(parents=True, exist_ok=True)

    def start(self, payload: dict) -> Run:
        name = str(payload.get("harness", ""))
        if name not in HARNESSES:
            raise ValueError(f"unknown harness {name!r}")
        spec = HARNESSES[name]
        version = str(payload.get("version", spec.supported_versions[0] if spec.supported_versions else "component")).lower()
        if version not in spec.supported_versions:
            raise ValueError(f"{name} supports versions {', '.join(spec.supported_versions)}")
        run_id = f"{name}-{uuid.uuid4().hex[:10]}"
        output = RUN_ROOT / run_id
        command = spec.command(
            output=output,
            steps=payload.get("steps"),
            substeps=payload.get("substeps"),
            seeds=[int(s) for s in payload.get("seeds", spec.default_seeds)],
            world=payload.get("world"),
            version=version,
        )
        run = Run(run_id, name, version, command, str(output))
        with self.lock:
            self.runs[run_id] = run
        threading.Thread(target=self._execute, args=(run,), daemon=True).start()
        return run

    def _execute(self, run: Run) -> None:
        try:
            process = subprocess.Popen(run.command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, env={**__import__('os').environ, "PYTHONUNBUFFERED": "1"})
            with self.lock:
                run.status = "running"
                run.log += f"$ {' '.join(run.command)}\n"
            lines = []
            assert process.stdout is not None
            for line in process.stdout:
                lines.append(line)
                with self.lock:
                    run.log = (run.log + line)[-20000:]
            process.wait()
            # A harness process is not the scientific verdict by itself.  A
            # child can exit zero after writing a partial directory, or an
            # acceptance JSON can be stale.  Re-read the evidence through the
            # independent raw-trace validator before exposing "passed" in the
            # lab UI.  Its behavioural red result remains red; this only adds
            # the second scientific check.
            spec = HARNESSES[run.harness]
            if spec.version_policy == "strict_agent":
                validator_path = Path(run.output) / "validator.json"
                validator_cmd = [
                    sys.executable,
                    "-m",
                    "simulations.active_inference.experiments.version_evidence_validator",
                    run.output,
                    "--output",
                    str(validator_path),
                ]
                validator_process = subprocess.run(
                    validator_cmd,
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    env={**__import__('os').environ, "PYTHONUNBUFFERED": "1"},
                )
                try:
                    validation = json.loads(validator_path.read_text())
                except (OSError, json.JSONDecodeError):
                    validation = {
                        "valid": False,
                        "behavior_passed": False,
                        "failures": ["validator did not produce a JSON report"],
                    }
                validation["returncode"] = validator_process.returncode
            else:
                # Shared lower-level probes and the quarantined compass are
                # deliberately not V1--V4 claims.  Do not pretend the strict
                # version validator can certify their legacy/full-brain
                # evidence; expose the boundary explicitly in the run record.
                validation = {
                    "skipped": True,
                    "valid": False,
                    "behavior_passed": bool(process.returncode == 0),
                    "reason": f"version policy is {spec.version_policy}, not strict_agent",
                }
            with self.lock:
                run.validation = validation
            with self.lock:
                run.returncode = process.returncode
                run.finished = time.time()
                run.status = (
                    "passed"
                    if process.returncode == 0 and (
                        validation.get("valid") and validation.get("behavior_passed")
                        or validation.get("skipped")
                    ) else "failed"
                )
                run.log = (run.log + "\nIndependent evidence validator:\n"
                           + json.dumps(validation, indent=2))[-24000:]
                acceptance = Path(run.output) / "acceptance.json"
                if acceptance.exists():
                    try:
                        run.log = (run.log + "\n" + json.dumps(json.loads(acceptance.read_text()), indent=2))[-24000:]
                    except json.JSONDecodeError:
                        pass
        except Exception as exc:
            with self.lock:
                run.finished = time.time(); run.status = "error"; run.returncode = -1
                run.log += f"\n{type(exc).__name__}: {exc}\n"

    def get(self, run_id: str) -> dict | None:
        with self.lock:
            run = self.runs.get(run_id)
            return run.public() if run else None

    def list(self) -> list[dict]:
        """Return recent run records for the lab's evidence dock."""
        with self.lock:
            values = [run.public() for run in self.runs.values()]
        return sorted(values, key=lambda value: value.get("started", 0), reverse=True)[:32]


MANAGER = RunManager()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def send(self, body, content_type="application/json; charset=utf-8", code=200):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def json(self, value, code=200):
        self.send(json.dumps(value, separators=(",", ":")), code=code)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/lab", "/index.html"):
            self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/healthz":
            self.json({"ok": True, "service": "active-inference-harness-lab"})
        elif path == "/api/harnesses":
            self.json({"protocol": "aif-lab/1", "harnesses": all_specs()})
        elif path == "/api/runs":
            self.json({"protocol": "aif-lab/1", "runs": MANAGER.list()})
        elif path.startswith("/api/runs/"):
            run = MANAGER.get(path.rsplit("/", 1)[-1])
            self.json(run or {"error": "unknown run"}, 200 if run else 404)
        else:
            self.send("not found", "text/plain; charset=utf-8", 404)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode() or "{}")
        except json.JSONDecodeError:
            self.json({"error": "request body must be JSON"}, 400)
            return
        if path == "/api/harness/run":
            try:
                run = MANAGER.start(payload)
            except (ValueError, TypeError) as exc:
                self.json({"error": str(exc)}, 400)
                return
            self.json(run.public(), 202)
            return
        self.json({"error": "not found"}, 404)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8850)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"harness lab on http://127.0.0.1:{args.port}/lab", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
