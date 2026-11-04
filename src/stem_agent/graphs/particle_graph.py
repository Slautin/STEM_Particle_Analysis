"""Particle-analysis graph.

    preflight -> status_start -> move -> focus_z -> set_fov -> tune_c1 -> tune_a1
              -> overview -> segment -> eds -> status_end -> save -> END

    after every node: the next one, or status_end when run_status is "failed" / "no_particles"
    status_end and save always run, so every run leaves a record

    build_graph() -> compiled LangGraph graph
    run(settings, verbose=True) -> final state       streams node by node
    mermaid() -> str                                 the graph as mermaid text (made locally)
"""
from langgraph.graph import END, START, StateGraph

from stem_agent.nodes.eds_node import eds_node
from stem_agent.nodes.move_stage_node import move_stage_node
from stem_agent.nodes.overview_node import overview_node
from stem_agent.nodes.preflight_node import preflight_node
from stem_agent.nodes.save_node import save_node
from stem_agent.nodes.segment_node import segment_node
from stem_agent.nodes.set_fov_node import set_fov_node
from stem_agent.nodes.status_node import status_end_node, status_start_node
from stem_agent.nodes.tune_node import focus_z_node, tune_a1_node, tune_c1_node
from stem_agent.states.experiment_state import ParticleRunState

STEPS = [
    ("preflight", preflight_node),
    ("status_start", status_start_node),
    ("move", move_stage_node),
    ("focus_z", focus_z_node),
    ("set_fov", set_fov_node),
    ("tune_c1", tune_c1_node),
    ("tune_a1", tune_a1_node),
    ("overview", overview_node),
    ("segment", segment_node),
    ("eds", eds_node),
]
STOP = ("failed", "no_particles")


def _route(next_node: str):
    def route(state: dict) -> str:
        return "status_end" if state.get("run_status") in STOP else next_node
    route.__name__ = f"route_to_{next_node}"
    return route


def build_graph():
    graph = StateGraph(ParticleRunState)
    for name, node in STEPS:
        graph.add_node(name, node)
    graph.add_node("status_end", status_end_node)
    graph.add_node("save", save_node)

    graph.add_edge(START, "preflight")
    names = [name for name, _ in STEPS] + ["status_end"]
    for name, next_name in zip(names[:-1], names[1:]):
        if next_name == "status_end":
            graph.add_edge(name, next_name)
        else:
            graph.add_conditional_edges(name, _route(next_name), [next_name, "status_end"])
    graph.add_edge("status_end", "save")
    graph.add_edge("save", END)
    return graph.compile()


def mermaid() -> str:
    return build_graph().get_graph().draw_mermaid()


async def run(settings: dict, verbose: bool = True) -> dict:
    """Run the whole graph; print one line per finished node. Returns the final state."""
    final: dict = {}
    async for mode, chunk in build_graph().astream({"settings": settings},
                                                   stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
        elif verbose:
            for node, update in chunk.items():
                update = update or {}
                status = f" | run_status {update['run_status']}" if "run_status" in update else ""
                print(f"== {node} done: {sorted(update)}{status}\n")
    return final
