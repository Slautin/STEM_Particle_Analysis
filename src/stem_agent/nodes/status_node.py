"""Is the microscope ready? Read-only snapshot of the instrument.

    make_status_node(check_ready) -> node
        status_start_node  writes start, current; if not ready: run_status="failed" (nothing moves)
        status_end_node    writes end; no checks; runs also after a failure

MCP:     get_parameters, get_stage, get_fov, get_defocus, every <TangoClass>_State
Checks:  config.READY_CHECKS on normalized fields, config.BAD_DEVICE_STATES;
         unknown values stop the run only where TARGETS[target]["strict_ready"]
Saves:   status_start.json / status_end.json in the run folder (everything read)
Never fixes anything itself (no unblanking, no valve opening).
Source:  new
"""
import json
from pathlib import Path

from stem_agent.config import BAD_DEVICE_STATES, READY_CHECKS, TARGETS
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
    return {
        "stem_mode":          None if stem is None else bool(stem),
        "beam_blanked":       None if blanked is None else bool(blanked),
        "column_valves_open": None if text("column_valves") is None else text("column_valves").endswith("open"),
        "high_tension_on":    None if ht_on is None else bool(ht_on),
        "beam_stopper_out":   None if text("beam_stopper") is None else "retract" in text("beam_stopper"),
    }


def assess(fields: dict, devices: dict[str, str], strict: bool) -> tuple[list[str], list[str]]:
    """-> (problems, warnings). Problems stop the run at status_start."""
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


async def read_snapshot(run_dir: Path, name: str, strict: bool) -> dict:
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
    raw_path = run_dir / f"{name}.json"
    raw_path.write_text(json.dumps({"parameters": params, "devices": devices, "stage": stage,
                                    "fov_m": fov, "defocus_m": defocus, "fields": fields,
                                    "problems": problems, "warnings": unknown + warnings},
                                   indent=2, default=str), encoding="utf-8")
    return {"stage": stage, "fov_m": fov, "defocus_m": defocus, **fields, "devices": devices,
            "ready": not problems, "problems": problems, "warnings": unknown + warnings,
            "raw_path": str(raw_path)}


def make_status_node(check_ready: bool):
    name = "status_start" if check_ready else "status_end"

    async def status_node(state: dict) -> dict:
        run = state["run"]
        strict = TARGETS[run["target"]]["strict_ready"]
        snap = await read_snapshot(Path(run["run_dir"]), name, strict)
        stage = ", ".join(f"{v:+.3g}" for v in snap["stage"])
        bad = {k: v for k, v in snap["devices"].items() if v != "ON"}
        print(f"[{name}] stage [{stage}] | FOV {snap['fov_m']} m | "
              f"devices {len(snap['devices'])} ({bad or 'all ON'}) | ready {snap['ready']}")
        for line in snap["problems"]:
            print(f"[{name}] PROBLEM: {line}")
        for line in snap["warnings"]:
            print(f"[{name}] warning: {line}")

        if not check_ready:
            return {"end": snap}
        update = {"start": snap, "current": snap}
        if not snap["ready"]:
            update |= {"run_status": "failed",
                       "error": f"{name}: not ready: {'; '.join(snap['problems'])}"}
        return update

    status_node.__name__ = f"{name}_node"
    return safe_node()(status_node)


status_start_node = make_status_node(check_ready=True)
status_end_node = make_status_node(check_ready=False)
