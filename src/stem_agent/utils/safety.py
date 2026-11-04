"""Error handling for nodes.

    safe_node(undo=None) -> decorator
        on any exception: print it, run undo(state) with the state the node received,
        return {"run_status": "failed", "errors": ["<node>: <msg>"]}  (errors add up in the state)
        the graph router then jumps to status_end -> save
"""
import functools
from collections.abc import Awaitable, Callable

Node = Callable[[dict], Awaitable[dict]]


def safe_node(undo: Node | None = None) -> Callable[[Node], Node]:
    def wrap(node: Node) -> Node:
        @functools.wraps(node)
        async def run(state: dict) -> dict:
            try:
                return await node(state)
            except Exception as exc:
                name = node.__name__.removesuffix("_node")
                message = f"{name}: {type(exc).__name__}: {exc}"
                print(f"[{name}] FAILED: {exc}")
                if undo is not None:
                    try:
                        await undo(state)
                        print(f"[{name}] undo done")
                    except Exception as undo_exc:
                        message += f" | undo failed: {undo_exc}"
                        print(f"[{name}] UNDO FAILED: {undo_exc}")
                return {"run_status": "failed", "errors": [message]}
        return run
    return wrap
