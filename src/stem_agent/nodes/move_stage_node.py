"""Move the stage to the target (x, y), keep z and tilt. SIDE EFFECT: moves the stage.

Reads:   settings.target_xy_m, current.stage
Writes:  current
MCP:     get_stage, move_stage (via common.move_stage)
Checks:  step within LIMITS["max_xy_move_m"]; readback within LIMITS["stage_tolerance_m"]
Undo:    undo_move: back to the x, y the node started from
Source:  new (named in notebook cell 0 only)
"""
from stem_agent.nodes import common
from stem_agent.utils.safety import safe_node


async def undo_move(state: dict) -> None:
    x, y = state["current"]["stage"][:2]
    await common.move_stage(x=x, y=y)


@safe_node(undo=undo_move)
async def move_stage_node(state: dict) -> dict:
    x, y = state["settings"]["target_xy_m"]
    before = state["current"]["stage"]
    reached = await common.move_stage(x=x, y=y)
    print(f"[move] ({before[0] * 1e6:+.4f}, {before[1] * 1e6:+.4f}) -> "
          f"({reached[0] * 1e6:+.4f}, {reached[1] * 1e6:+.4f}) um")
    return {"current": {**state["current"], "stage": reached}}
