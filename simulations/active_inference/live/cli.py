"""Typer control surface for live PAULA brains and the harness laboratory.

Examples (from ``active-inference/``)::

    uv run aif-live versions
    uv run aif-live start --version v1 --port 8770
    uv run aif-live start --version v2 --port 8780
    uv run aif-live start --version v3 --port 8790
    uv run aif-live start --version v4 --world obstacle_detour --port 8800
    uv run aif-live status
    uv run aif-live send v3 run --ticks 500
    uv run aif-live lab --port 8850

The browser is a client of the same HTTP/WebSocket service.  ``start`` only
manages the process; it does not create an alternate simulation implementation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import typer

from .versions import get_version, version_ids, VERSIONS

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Run and inspect PAULA live brains.")
_ROOT = Path(__file__).resolve().parents[3]
_RUN_DIR = _ROOT / ".live"
_PID_DIR = _RUN_DIR / "pids"
_LOG_DIR = _RUN_DIR / "logs"


def _ensure_run_dirs() -> None:
    _PID_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _record(version: str) -> Path:
    return _PID_DIR / f"{get_version(version).id}.json"


def _url(port: int, path: str) -> str:
    return f"http://127.0.0.1:{int(port)}{path}"


def _get(port: int, path: str) -> dict:
    with urlopen(_url(port, path), timeout=2.5) as response:
        return json.loads(response.read())


def _post(port: int, path: str, payload: dict) -> dict:
    request = Request(_url(port, path), data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=3) as response:
        return json.loads(response.read())


@app.command()
def versions() -> None:
    """List selectable agent compositions."""
    for key in version_ids():
        spec = VERSIONS[key]
        typer.echo(f"{spec.id}: {spec.label} — {spec.description}")
        typer.echo(f"  components: {', '.join(spec.components)}")


@app.command()
def serve(
    version: str = typer.Option("v1", "--version", "-v", help="v1, v2, v3, or v4"),
    port: int = typer.Option(8770, help="HTTP port; WebSocket uses port + 1"),
    world: str = typer.Option("meadow", help="meadow, minefield, sparse, obstacle_detour, obstacle_corner, obstacle_chicane, or obstacle_maze"),
    ticks: int = typer.Option(0, help="start immediately for N ticks; -1 runs continuously"),
) -> None:
    """Run one selected live brain in the foreground."""
    spec = get_version(version)
    command = [sys.executable, "-m", "simulations.active_inference.live_brain",
               "--version", spec.id, "--port", str(port), "--world", world, "--ticks", str(ticks)]
    os.execv(sys.executable, command)


@app.command()
def start(
    version: str = typer.Option("v1", "--version", "-v", help="v1, v2, v3, or v4"),
    port: int = typer.Option(8770, help="HTTP port; WebSocket uses port + 1"),
    world: str = typer.Option("meadow", help="meadow, minefield, sparse, obstacle_detour, obstacle_corner, obstacle_chicane, or obstacle_maze"),
    run: bool = typer.Option(False, "--run", help="start the simulation immediately"),
) -> None:
    """Start a managed live brain in the background."""
    spec = get_version(version)
    _ensure_run_dirs()
    path = _record(spec.id)
    if path.exists():
        try:
            old = json.loads(path.read_text())
            os.kill(int(old["pid"]), 0)
            raise typer.BadParameter(f"{spec.id} is already running (pid {old['pid']})")
        except ProcessLookupError:
            path.unlink(missing_ok=True)
        except KeyError:
            path.unlink(missing_ok=True)
    log_path = _LOG_DIR / f"{spec.id}.log"
    ticks = -1 if run else 0
    command = [sys.executable, "-m", "simulations.active_inference.live_brain",
               "--version", spec.id, "--port", str(port), "--world", world, "--ticks", str(ticks)]
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    path.write_text(json.dumps({"version": spec.id, "pid": process.pid, "port": port,
                                "log": str(log_path), "command": command}, indent=2))
    typer.echo(f"starting {spec.id} (pid {process.pid}); log={log_path}")
    deadline = time.time() + 900
    while time.time() < deadline:
        if process.poll() is not None:
            typer.echo(f"{spec.id} exited with code {process.returncode}; see {log_path}", err=True)
            raise typer.Exit(code=1)
        try:
            health = _get(port, "/api/health")
            if health.get("ok"):
                typer.echo(f"ready: http://127.0.0.1:{port}/ (WebSocket :{port + 1})")
                return
        except (OSError, URLError, ValueError):
            pass
        time.sleep(0.5)
    typer.echo(f"timed out waiting for {spec.id}; see {log_path}", err=True)
    raise typer.Exit(code=1)


@app.command()
def stop(version: str = typer.Argument(..., help="v1, v2, v3, or v4")) -> None:
    """Stop one managed live brain."""
    path = _record(version)
    if not path.exists():
        typer.echo(f"{get_version(version).id} is not managed")
        return
    record = json.loads(path.read_text())
    pid = int(record["pid"])
    try:
        os.kill(pid, signal.SIGTERM)
        typer.echo(f"stopped {record['version']} (pid {pid})")
    except ProcessLookupError:
        typer.echo(f"{record['version']} was already stopped")
    finally:
        path.unlink(missing_ok=True)


@app.command()
def status() -> None:
    """Show managed processes and their live protocol sessions."""
    _ensure_run_dirs()
    for key in version_ids():
        path = _record(key)
        if not path.exists():
            typer.echo(f"{key}: stopped")
            continue
        record = json.loads(path.read_text())
        try:
            os.kill(int(record["pid"]), 0)
            session = _get(int(record["port"]), "/api/session")
            typer.echo(f"{key}: pid={record['pid']} port={record['port']} neurons={session['neuron_count']} "
                       f"status=ready log={record['log']}")
        except (OSError, URLError, KeyError, ValueError):
            typer.echo(f"{key}: stale pid={record.get('pid')} port={record.get('port')} log={record.get('log')}")


@app.command("send")
def send_command(
    version: str = typer.Argument(..., help="v1, v2, v3, or v4"),
    command: str = typer.Argument(..., help="run, pause, step, world, reset, or rebuild"),
    ticks: int = typer.Option(-1, "--ticks", help="for run: number of agent ticks; -1 means continuous"),
    world: str = typer.Option("meadow", help="for world: meadow, minefield, sparse, obstacle_detour, obstacle_corner, obstacle_chicane, or obstacle_maze"),
) -> None:
    """Send a control command to a managed live brain."""
    path = _record(version)
    if not path.exists():
        raise typer.BadParameter(f"{get_version(version).id} is not managed; run start first")
    record = json.loads(path.read_text())
    payload = {"c": command}
    if command == "run": payload["n"] = ticks
    if command == "world": payload["w"] = world
    try:
        typer.echo(json.dumps(_post(int(record["port"]), "/api/command", payload), indent=2))
    except (OSError, URLError) as exc:
        typer.echo(f"cannot reach {record['version']}: {exc}", err=True)
        raise typer.Exit(code=1)


@app.command()
def lab(
    port: int = typer.Option(8850, help="Harness lab HTTP port"),
) -> None:
    """Run the component/harness web laboratory in the foreground."""
    os.execv(sys.executable, [sys.executable, "-m", "simulations.active_inference.lab.server", "--port", str(port)])


if __name__ == "__main__":
    app()
