"""Set the field of view (particle graph: for the overview). SIDE EFFECT: changes FOV.

    set_fov(fov_m)       also used by tune_node for a step's fov_m

Reads:   settings.tuning_fov_m, current.fov_m
Writes:  current
MCP:     set_fov, get_fov
Checks:  FOV within LIMITS["fov_range_m"]; readback within LIMITS["fov_rel_tolerance"]
Undo:    undo_fov: back to the FOV the node started from
Note:    the particle twin rebuilds its sample when the FOV changes
Source:  notebook cell 12
"""
from stem_agent.config import LIMITS, check_fov
from stem_agent.mcp import session
from stem_agent.mcp.session import InstrumentError
from stem_agent.utils.safety import safe_node


async def set_fov(fov_m: float) -> float:
    check_fov(fov_m)
    await session.call("set_fov", float(fov_m))
    got = float(await session.call("get_fov"))
    if abs(got - fov_m) > LIMITS["fov_rel_tolerance"] * fov_m:
        raise InstrumentError(f"FOV readback {got:.4g} m, requested {fov_m:.4g} m")
    return got


async def undo_fov(state: dict) -> None:
    old = state["current"].get("fov_m")
    if old:
        await set_fov(old)


@safe_node(undo=undo_fov)
async def set_fov_node(state: dict) -> dict:
    before = state["current"].get("fov_m")
    fov = await set_fov(state["settings"]["tuning_fov_m"])
    print(f"[set_fov] {before * 1e9 if before else float('nan'):.1f} -> {fov * 1e9:.1f} nm")
    return {"current": {**state["current"], "fov_m": fov}}
