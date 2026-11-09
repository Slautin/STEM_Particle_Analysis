"""Particle-analysis graph.

    preflight -> status_start -> move -> tune (loops over settings.protocol) -> set_fov
              -> overview -> segment -> eds -> status_end -> save -> END

    after every node: the next one, or status_end when run_status is "failed" / "no_particles"
    status_end and save always run, so every run leaves a record
    status_start and status_end are the same status_node; only the first snapshot gates
    segment = sam_segment_node (SAM2 device on the overview image; settings["sam"])

    build_graph() -> compiled LangGraph graph
    check_settings(settings)                         every node's settings; ValueError if bad
    run(settings, verbose=True) -> final state       checks settings, then streams node by node
    mermaid() -> str                                 the graph as mermaid text (made locally)
"""
from langgraph.graph import END, START, StateGraph

from stem_agent.config import check_fov, protocol_settings, sam_settings
from stem_agent.nodes.eds_node import eds_node, eds_settings
from stem_agent.nodes.move_stage_node import move_stage_node
from stem_agent.nodes.overview_node import overview_node
from stem_agent.nodes.preflight_node import preflight_node
from stem_agent.nodes.save_node import save_node
from stem_agent.nodes.sam_segment_node import sam_segment_node
from stem_agent.nodes.set_fov_node import set_fov_node
from stem_agent.nodes.status_node import status_node
from stem_agent.nodes.tune_node import next_step, tune_node
from stem_agent.states.experiment_state import ParticleRunState

STEPS = [
    ("preflight", preflight_node),
    ("status_start", status_node),
    ("move", move_stage_node),
    ("tune", tune_node),                       # repeats until every protocol step is done
    ("set_fov", set_fov_node),                 # the overview FOV, after tuning
    ("overview", overview_node),
    ("segment", sam_segment_node),             # SAM2 on the overview image
    ("eds", eds_node),
]
STOP = ("failed", "no_particles")


def _route(next_node: str):
    def route(state: dict) -> str:
        return "status_end" if state.get("run_status") in STOP else next_node
    route.__name__ = f"route_to_{next_node}"
    return route


def _after_tune(next_node: str):
    def route(state: dict) -> str:
        if state.get("run_status") in STOP:
            return "status_end"
        return "tune" if next_step(state) else next_node
    route.__name__ = f"tune_again_or_{next_node}"
    return route


def build_graph():
    graph = StateGraph(ParticleRunState)
    for name, node in STEPS:
        graph.add_node(name, node)
    graph.add_node("status_end", status_node)
    graph.add_node("save", save_node)

    graph.add_edge(START, "preflight")
    names = [name for name, _ in STEPS] + ["status_end"]
    for name, next_name in zip(names[:-1], names[1:]):
        if next_name == "status_end":
            graph.add_edge(name, next_name)
        elif name == "tune":
            graph.add_conditional_edges(name, _after_tune(next_name), ["tune", next_name, "status_end"])
        else:
            graph.add_conditional_edges(name, _route(next_name), [next_name, "status_end"])
    graph.add_edge("status_end", "save")
    graph.add_edge("save", END)
    return graph.compile()


def check_settings(settings: dict) -> None:
    """Fail before anything runs, not halfway (each node checks its own settings again)."""
    if settings.get("stage_xy_m") is not None and len(settings["stage_xy_m"]) != 2:
        raise ValueError("stage_xy_m must be (x, y) or None")
    check_fov(settings["tuning_fov_m"])
    protocol_settings(settings)
    if "segmentation" in settings:
        raise ValueError("the graph segments with SAM: use settings['sam'] "
                         "(min_visible, min_diameter_nm, edge_px), not 'segmentation'")
    sam_settings(settings)
    eds_settings(settings)


def mermaid() -> str:
    return build_graph().get_graph().draw_mermaid()


async def run(settings: dict, verbose: bool = True) -> dict:
    """Run the whole graph; print one line per finished node. Returns the final state."""
    check_settings(settings)
    # LangGraph stops after 25 node runs by default; the tune loop adds one per protocol step
    limit = len(STEPS) + len(settings["protocol"]) + 10
    final: dict = {}
    async for mode, chunk in build_graph().astream({"settings": settings},
                                                   stream_mode=["updates", "values"],
                                                   config={"recursion_limit": limit}):
        if mode == "values":
            final = chunk
        elif verbose:
            for node, update in chunk.items():
                update = update or {}
                status = f" | run_status {update['run_status']}" if "run_status" in update else ""
                print(f"== {node} done: {sorted(update)}{status}\n")
    return final
