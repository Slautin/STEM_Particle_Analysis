"""Start, stop and check the asyncroscopy digital twin + MCP server in the background.

Runs inside ASYNCROSCOPY_DIR (set in .env) with that repo's own uv environment.
Logs and PIDs go to runs/twin_logs/.

Usage:
    uv run python scripts/twin.py start
    uv run python scripts/twin.py status
    uv run python scripts/twin.py stop
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

TWIN_YAML = ROOT / "twin" / "digital_twin_particles.yaml"
MCP_YAML = "configs/mcp_dt.yaml"                   # inside ASYNCROSCOPY_DIR
LOG_DIR = ROOT / "runs" / "twin_logs"
PID_FILE = LOG_DIR / "pids.json"
PORTS = {"tango": 9094, "tiled": 9091, "mcp": 8000}

# name -> (command, line in the log that means "ready", timeout in s)
SERVICES = {
    "servers": (["uv", "run", "startup_scripts/run_servers.py", "--yaml", str(TWIN_YAML)],
                "All asyncroscopy servers are ready.", 600),
    "mcp":     (["uv", "run", "startup_scripts/run_mcp.py", "--yaml", MCP_YAML],
                "MCP ready:", 180),
}


def asyncroscopy_dir() -> Path:
    raw = os.getenv("ASYNCROSCOPY_DIR")
    if not raw:
        sys.exit("ASYNCROSCOPY_DIR is not set; add it to .env")
    path = Path(raw).expanduser()
    if not (path / "startup_scripts" / "run_servers.py").exists():
        sys.exit(f"ASYNCROSCOPY_DIR={path} is not an asyncroscopy checkout")
    return path


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pids() -> dict[str, int]:
    return json.loads(PID_FILE.read_text()) if PID_FILE.exists() else {}


def wait_ready(name: str, proc: subprocess.Popen, log: Path, marker: str, timeout: float) -> str:
    """Block until `marker` appears in the log; fail if the process exits first."""
    start = time.time()
    while time.time() - start < timeout:
        text = log.read_text(errors="replace")
        for line in text.splitlines():
            if marker in line:
                return line.strip()
        if proc.poll() is not None:
            tail = "\n".join(text.splitlines()[-20:])
            sys.exit(f"[twin] {name} exited with code {proc.returncode} before it was ready:\n{tail}")
        time.sleep(1)
    sys.exit(f"[twin] {name} not ready after {timeout:.0f} s; see {log}")


def start() -> None:
    busy = [f"{k}:{p}" for k, p in PORTS.items() if port_open(p)]
    if busy:
        sys.exit(f"[twin] ports already in use ({', '.join(busy)}); run `status` or `stop` first")

    cwd = asyncroscopy_dir()
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}   # use asyncroscopy's own .venv
    env["PYTHONUNBUFFERED"] = "1"                    # ready lines must reach the log immediately
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pids: dict[str, int] = {}
    for name, (command, marker, timeout) in SERVICES.items():
        log = LOG_DIR / f"{name}.log"
        with log.open("w") as out:
            # own process group, so stop() can signal the whole tree
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        pids[name] = proc.pid
        PID_FILE.write_text(json.dumps(pids))
        print(f"[twin] starting {name} (pid {proc.pid}), log {log.relative_to(ROOT)}")
        line = wait_ready(name, proc, log, marker, timeout)
        if proc.poll() is not None:                  # marker flushed on exit, not a real start
            sys.exit(f"[twin] {name} exited with code {proc.returncode}; see {log}")
        print(f"[twin] {name} ready: {line}")
    print(f"[twin] MCP at http://127.0.0.1:{PORTS['mcp']}/mcp")


def stop() -> None:
    pids = read_pids()
    if not pids:
        print("[twin] nothing to stop (no pid file)")
    for name in reversed(list(pids)):                # MCP first, then the device stack
        pid = pids[name]
        # SIGINT first: like Ctrl+C, the asyncroscopy launchers then stop the servers
        # they started in their own process groups (run_mcp.py ignores SIGTERM for that)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            if not alive(pid):
                break
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                break
            for _ in range(60):
                if not alive(pid):
                    break
                time.sleep(0.5)
        print(f"[twin] stopped {name} (pid {pid})")
    PID_FILE.unlink(missing_ok=True)

    for _ in range(10):
        left = [f"{k}:{p}" for k, p in PORTS.items() if port_open(p)]
        if not left:
            return
        time.sleep(0.5)
    print(f"[twin] WARNING: ports still open ({', '.join(left)}); "
          f"find the process with `lsof -nP -iTCP -sTCP:LISTEN`")


def status() -> None:
    for name, pid in read_pids().items():
        print(f"[twin] {name:<8} pid {pid:<7} {'running' if alive(pid) else 'not running'}")
    for name, port in PORTS.items():
        print(f"[twin] {name:<8} port {port:<6} {'open' if port_open(port) else 'closed'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    actions = {"start": start, "stop": stop, "status": status}
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        sys.exit(__doc__)
    actions[sys.argv[1]]()
