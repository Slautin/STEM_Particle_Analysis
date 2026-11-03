"""Move the stage to the target (x, y), keep z and tilt. SIDE EFFECT: moves the stage.

Reads:   settings.target_xy_m, current.stage
Writes:  current
MCP:     move_stage, get_stage
Checks:  readback within config.LIMITS move tolerance
Undo:    back to start x, y
Source:  new (named in notebook cell 0 only)
"""
