"""Set the field of view used for tuning and the overview. SIDE EFFECT: changes FOV.

Reads:   settings.tuning_fov_m
Writes:  current
MCP:     set_fov, get_fov
Checks:  readback matches
Undo:    previous FOV
Source:  notebook cell 12
"""
