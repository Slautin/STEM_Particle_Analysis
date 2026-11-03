"""Error handling for nodes that change the microscope.

Planned:
    safe_node(undo) -> decorator
        on any exception: run undo(state), return {"run_status": "failed", "error": "<node>: <msg>"}
"""
