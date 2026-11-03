"""One tuning node for stage z, C1 and A1. SIDE EFFECT: changes focus / corrector.

Planned:
    make_tune_node(name, dims, apply) -> node
        focus_z_node = make_tune_node("z",  dims=1, apply=set_stage_z)     absolute: z0 + dz
        tune_c1_node = make_tune_node("C1", dims=1, apply=correct_delta)   relative steps
        tune_a1_node = make_tune_node("A1", dims=2, apply=correct_delta)

Loop:    QuadraticSearch.ask -> apply -> acquire_haadf -> metric -> tell; end at best
Reads:   settings.focus / c1 / a1, current
Writes:  tuning[name], current
MCP:     move_stage or correct_aberration, acquire_scanned_image
Checks:  every offset within max_offset before sending
Undo:    z back to start; C1/A1 undo total applied offset
Saves:   history JSON + figure PNG
Source:  notebook cells 5, 7, 10, 13, 15
"""
