"""Particle-analysis graph.

    preflight -> status_start -> move -> focus_z -> set_fov -> tune_c1 -> tune_a1
              -> overview -> segment -> eds -> status_end -> save -> END

Routing:
    after each node: next, or status_end if run_status == "failed"
    after segment:   eds if particles, else status_end

Planned:
    build_graph() -> CompiledStateGraph
"""
