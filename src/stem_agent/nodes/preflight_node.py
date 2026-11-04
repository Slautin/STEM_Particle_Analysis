"""Is the software ready? Read-only.

Reads:   settings
Writes:  run, run_status ("running", or "failed" + errors; `run` is written in both cases)
MCP:     list tools, DATA_get_config
Checks:  every tool in config.TOOLS exists on the target; settings are in range;
         tuning metrics are in SAFE_METRICS; search limits fit the per-move limits
Saves:   settings.json in a new run folder, created before any check
Undo:    none
Source:  notebook cell 1 (partly)
"""
import json
from datetime import datetime

from stem_agent.config import (LIMITS, SAFE_METRICS, TOOLS, TUNING, new_run, search_settings,
                               segmentation_settings)
from stem_agent.data import tiled_io
from stem_agent.mcp import session
from stem_agent.mcp.session import InstrumentError
from stem_agent.utils.safety import safe_node


def check_settings(settings: dict) -> None:
    lo, hi = LIMITS["fov_range_m"]
    if not lo <= settings["tuning_fov_m"] <= hi:
        raise ValueError(f"tuning_fov_m={settings['tuning_fov_m']:.3g} outside {lo:.0e}..{hi:.0e} m")
    if len(settings["target_xy_m"]) != 2:
        raise ValueError("target_xy_m must be (x, y)")
    if not 1 <= settings.get("max_particles", LIMITS["max_particles"]) <= LIMITS["max_particles"]:
        raise ValueError(f"max_particles must be 1..{LIMITS['max_particles']}")
    if settings.get("beam_order", "col_row") not in ("col_row", "row_col"):
        raise ValueError("beam_order must be 'col_row' or 'row_col'")
    for name in TUNING:
        check_search(name, search_settings(settings, name))
    seg = segmentation_settings(settings)
    if seg["method"] != "otsu":
        raise ValueError(f"segmentation method {seg['method']!r} not available (otsu only)")
    if seg["smooth_nm"] <= 0 or seg["min_diameter_nm"] < 0 or seg["edge_px"] < 0:
        raise ValueError(f"bad segmentation settings: {seg}")


def check_search(name: str, cfg: dict) -> None:
    if cfg["metric"] not in SAFE_METRICS:
        raise ValueError(f"tuning {name}: metric {cfg['metric']!r} is not allowed; it drives the "
                         f"search toward blur (block 2). Use one of {SAFE_METRICS}")
    if not 0 < cfg["step_m"] <= cfg["max_offset_m"]:
        raise ValueError(f"tuning {name}: need 0 < step_m <= max_offset_m, got {cfg}")
    move_limit = LIMITS["max_z_step_m"] if name == "z" else LIMITS["max_corrector_step_m"]
    if cfg["max_offset_m"] > move_limit / 2:          # any jump inside +-max_offset is one move
        raise ValueError(f"tuning {name}: max_offset_m {cfg['max_offset_m']:.3g} m is over half "
                         f"the per-move limit ({move_limit:.3g} m)")
    if cfg["max_iterations"] < 0:
        raise ValueError(f"tuning {name}: max_iterations must be >= 0")


async def check_software(settings: dict, run: dict) -> None:
    """Settings, MCP connection, tools, Tiled. Fills `run` as it goes."""
    check_settings(settings)
    client = session.connect(settings["target"])
    run["mcp_url"] = client.url
    tools = await client.list_tools()
    needed = {short: session.tool_name(short) for short in TOOLS}
    missing = sorted(short for short, name in needed.items() if name not in tools)
    if missing:
        raise InstrumentError(f"missing MCP tools on {client.url}: {missing}")
    run["tools"] = needed
    run["tiled_uri"] = json.loads(await session.call("get_tiled_config"))["uri"]
    tiled_io.connect(run["tiled_uri"])


@safe_node()
async def preflight_node(state: dict) -> dict:
    settings = state["settings"]
    # the run folder comes first, so that a failed preflight still leaves a record
    run_dir = new_run(settings.get("run_tag", ""))
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    run = {"run_dir": str(run_dir), "target": settings.get("target"), "mcp_url": None,
           "tiled_uri": None, "tools": {}, "started": datetime.now().isoformat(timespec="seconds")}
    try:
        await check_software(settings, run)
    except Exception as exc:
        print(f"[preflight] FAILED: {exc}")
        return {"run": run, "run_status": "failed", "errors": [f"preflight: {type(exc).__name__}: {exc}"]}

    print(f"[preflight] {run['target']} at {run['mcp_url']}: {len(run['tools'])}/{len(TOOLS)} tools, "
          f"Tiled {run['tiled_uri']}, run folder {run_dir}")
    return {"run": run, "run_status": "running"}
