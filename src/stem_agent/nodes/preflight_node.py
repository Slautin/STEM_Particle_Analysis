"""Is the software ready? Read-only. General: any graph can start with it.

Reads:   settings.instrument, settings.run_tag (optional)
Writes:  run, run_status ("running", or "failed" + errors; `run` is written in both cases),
         snapshots = [] (a new run: its first status_node call is the readiness gate)
MCP:     list tools, DATA_get_config
Checks:  no renamed settings keys (RENAMED); MCP reachable; every tool in config.TOOLS
         exists on the instrument; Tiled reachable
Saves:   settings.json in a new run folder, created before any check
Undo:    none
Note:    each node checks its own settings before it moves anything
         (config.protocol_settings, config.segmentation_settings, set_fov, eds_settings);
         particle_graph.check_settings runs them all before the graph starts
"""
import json
from datetime import datetime

from stem_agent.config import TOOLS, new_run
from stem_agent.data import tiled_io
from stem_agent.mcp import session
from stem_agent.mcp.session import InstrumentError
from stem_agent.utils.safety import safe_node

# old settings key -> new one (renamed 2026-11-05); an old key stops the run with a clear message
RENAMED = {"target": "instrument", "target_xy_m": "stage_xy_m", "tuning": "protocol"}


async def connect(instrument: str, run: dict) -> None:
    """MCP, tools, Tiled. Fills `run` as it goes."""
    client = session.connect(instrument)
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
    run = {"run_dir": str(run_dir), "instrument": settings.get("instrument"), "mcp_url": None,
           "tiled_uri": None, "tools": {}, "started": datetime.now().isoformat(timespec="seconds")}
    try:
        old = [f"{key!r} is now {new!r}" for key, new in RENAMED.items() if key in settings]
        if old:
            raise ValueError(f"renamed settings: {'; '.join(old)}")
        await connect(settings["instrument"], run)
    except Exception as exc:
        print(f"[preflight] FAILED: {exc}")
        return {"run": run, "run_status": "failed", "snapshots": [],
                "errors": [f"preflight: {type(exc).__name__}: {exc}"]}

    print(f"[preflight] {run['instrument']} at {run['mcp_url']}: {len(run['tools'])}/{len(TOOLS)} tools, "
          f"Tiled {run['tiled_uri']}, run folder {run_dir}")
    return {"run": run, "run_status": "running", "snapshots": []}
