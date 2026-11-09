"""Snapshot of the instrument. Read-only. One node for any place in a graph.

    status_node    reads the instrument, appends the snapshot to `snapshots`, refreshes `current`

Gate:    the FIRST snapshot of a run (snapshots empty, run_status "running") stops the run if
         the instrument is not ready, before anything moves. Later snapshots only record.
         preflight starts a run with snapshots = [], so a new run is always gated.
Reads:   run (folder, instrument), snapshots, current, run_status
Writes:  snapshots ([0] = start of the run, [-1] = latest); current (merged, so the corrector
         offsets that only the tune nodes know are kept); gate only: run_status="failed" + errors
MCP:     get_parameters, get_stage, get_fov, get_defocus, every <TangoClass>_State
Checks:  config.READY_CHECKS on normalized fields, config.BAD_DEVICE_STATES;
         unknown values are problems only where INSTRUMENTS[instrument]["strict_ready"]
Saves:   status_NN.json in the run folder (everything read)
Graph:   one function, any node name:  graph.add_node("status_start", status_node)
                                       graph.add_node("status_end", status_node)
Never fixes anything itself (no unblanking, no valve opening).
Source:  new
"""
import json
from datetime import datetime
from pathlib import Path

from stem_agent.config import BAD_DEVICE_STATES, INSTRUMENTS, READY_CHECKS
from stem_agent.mcp import session
from stem_agent.nodes import common
from stem_agent.utils.safety import safe_node

# Tango DevState order, for servers that send the state as a number
DEV_STATES = ["ON", "OFF", "CLOSE", "OPEN", "INSERT", "EXTRACT", "MOVING", "STANDBY",
              "FAULT", "INIT", "RUNNING", "ALARM", "DISABLE", "UNKNOWN"]


def device_state(value) -> str:
    """'ON', 'DevState.ON', 0, '0' -> 'ON' (asyncroscopy MCP sends the number as text)."""
    text = str(value).strip()
    if text.isdigit() and int(text) < len(DEV_STATES):
        return DEV_STATES[int(text)]
    return text.upper().rsplit(".", 1)[-1]


def normalize_parameters(params: dict) -> dict:
    """Real (AutoScript) and twin keys -> the READY_CHECKS fields. Missing -> None."""
    def text(key):
        value = params.get(key)
        return None if value is None else str(value).lower()

    stem = params.get("stem_mode")
    if stem is None and text("optical_mode") is not None:
        stem = text("optical_mode").endswith("stem")
    blanked = params.get("beam_blanker")
    ht_on = params.get("is_accelerator_on")
    print(text("column_valves"))
    return {
        "stem_mode":          None if stem is None else bool(stem),
        "beam_blanked":       None if blanked is None else bool(blanked),
        "column_valves_open": None if text("column_valves") is None else text("column_valves").endswith("opened"),
        "high_tension_on":    None if ht_on is None else bool(ht_on),
        "beam_stopper_out":   None if text("beam_stopper") is None else "retract" in text("beam_stopper"),
    }


def assess(fields: dict, devices: dict[str, str], strict: bool) -> tuple[list[str], list[str]]:
    """-> (problems, warnings). At the gate, problems stop the run."""
    problems, warnings = [], []
    for field, wanted in READY_CHECKS.items():
        value = fields[field]
        if value is None:
            (problems if strict else warnings).append(f"{field} unknown")
        elif value != wanted:
            problems.append(f"{field} is {value}, needs {wanted}")
    problems += [f"device {name} is {s}" for name, s in devices.items() if s in BAD_DEVICE_STATES]
    return problems, warnings


async def _read(warnings: list[str], label: str, coro):
    """A single unreadable field becomes a warning, not a crash."""
    try:
        return await coro
    except Exception as exc:
        warnings.append(f"{label} unreadable: {exc}")
        return None


async def read_snapshot(run_dir: Path, index: int, strict: bool) -> dict:
    """Read everything, save status_<index>.json, return the snapshot."""
    time = datetime.now().isoformat(timespec="seconds")
    warnings: list[str] = []
    params = await _read(warnings, "parameters", session.call("get_parameters"))
    if isinstance(params, str):
        params = json.loads(params)
    params = params or {}
    stage = await common.get_stage()
    fov = await _read(warnings, "fov", session.call("get_fov"))
    defocus = await _read(warnings, "defocus", session.call("get_defocus"))

    devices = {}
    for tool in sorted(await session.client().list_tools()):
        if tool.endswith("_State"):
            value = await _read(warnings, tool, session.call_tool(tool))
            devices[tool.removesuffix("_State")] = device_state(value)

    fields = normalize_parameters(params)
    problems, unknown = assess(fields, devices, strict)
    raw_path = run_dir / f"status_{index:02d}.json"
    raw_path.write_text(json.dumps({"index": index, "time": time, "parameters": params,
                                    "devices": devices, "stage": stage,
                                    "fov_m": fov, "defocus_m": defocus, "fields": fields,
                                    "problems": problems, "warnings": unknown + warnings},
                                   indent=2, default=str), encoding="utf-8")
    return {"index": index, "time": time,
            "stage": stage, "fov_m": fov, "defocus_m": defocus, **fields, "devices": devices,
            "ready": not problems, "problems": problems, "warnings": unknown + warnings,
            "raw_path": str(raw_path)}


@safe_node()
async def status_node(state: dict) -> dict:
    run = state["run"]
    snapshots = state.get("snapshots") or []
    gate = not snapshots and state.get("run_status") == "running"
    snap = await read_snapshot(Path(run["run_dir"]), len(snapshots),
                               INSTRUMENTS[run["instrument"]]["strict_ready"])

    label = f"status {snap['index']:02d}{' gate' if gate else ''}"
    stage = ", ".join(f"{v:+.3g}" for v in snap["stage"])
    bad = {k: v for k, v in snap["devices"].items() if v != "ON"}
    print(f"[{label}] stage [{stage}] | FOV {snap['fov_m']} m | "
          f"devices {len(snap['devices'])} ({bad or 'all ON'}) | ready {snap['ready']}")
    for line in snap["problems"]:
        print(f"[{label}] PROBLEM: {line}")
    for line in snap["warnings"]:
        print(f"[{label}] warning: {line}")

    update = {"snapshots": snapshots + [snap],
              "current": {**(state.get("current") or {}), **snap}}
    if gate and not snap["ready"]:
        update |= {"run_status": "failed",
                   "errors": [f"status: not ready: {'; '.join(snap['problems'])}"]}
    return update
