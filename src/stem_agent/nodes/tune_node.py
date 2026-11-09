"""One tuning node: runs the next step of settings.protocol. SIDE EFFECT: changes focus / corrector / FOV.

    tune_node(state)     runs the first protocol step not done yet in this run; {} when all are done
    next_step(state)     that step (merged with config.TUNING and checked), or None

Protocol: settings["protocol"], required, a list of steps, e.g.
              {"name": "z_10k", "tune": "z",  "fov_m": 10e-6,  "step_m": 10e-6}
              {"name": "C1_5M", "tune": "C1", "fov_m": 20e-9,  "step_m": 2e-9}
          name: record key and file names; tune: "z" / "C1" / "A1"; fov_m: set before tuning
          (optional); any key of config.TUNING overrides its default (incl. min_step_m:
          the smallest change the stage / corrector can make). The whole protocol is
          checked on every call (config.protocol_settings): a typo in a later step stops the first.
Progress: a step is done when its name is in state["tuning"] with this run's run_dir.
          A graph loops on this node until next_step(state) is None.
Search:   step "method": "quadratic" (QuadraticSearch, default) or "grid" (GridSearch: grid,
          normal-curve fit, one image at the fitted peak, optional zoom rounds; z / C1 only)
Loop:     search.ask -> apply -> acquire_haadf -> all metrics -> tell(reached, metric);
          at the end, apply the best measured point
Writes:   tuning[step name], current (stage for z; c1_offset_m / a1_offset_m; fov_m)
MCP:      set_fov / get_fov (when the step has fov_m)
          z:     get_stage / move_stage (common.move_stage), absolute z0 + offset;
                 "reached" = stage readback (the fit uses it, as in the notebook)
          C1/A1: correct_aberration "<name> v [v]", relative steps (notebook apply_increment);
                 "reached" = the node's own running total (the corrector has no readback)
Checks:   every point within +-max_offset and >= min_step_m from every measured point, so no
          move is smaller than min_step_m (QuadraticSearch); per-move limits (common.move_stage,
          LIMITS["max_corrector_step_m"])
Restore:  on any error, z / the corrector go back to where THIS step started (earlier steps keep
          their result; the FOV is not restored), then run_status="failed"
Saves:    tune_<step name>.json (every point: asked, reached, all metrics, Tiled key), tune_<step name>.png
Source:   notebook cells 5, 7, 10, 13, 15
"""
import json
from pathlib import Path

import numpy as np

from stem_agent.analysis.grid_search import GridSearch
from stem_agent.analysis.metrics import all_metrics
from stem_agent.analysis.optimizer import QuadraticSearch
from stem_agent.config import LIMITS, protocol_settings
from stem_agent.mcp import session
from stem_agent.nodes import common
from stem_agent.nodes.set_fov_node import set_fov
from stem_agent.utils import plots
from stem_agent.utils.safety import safe_node


class StageZ:
    """Absolute z moves around the z where the node started."""
    dims, unit, unit_name = 1, 1e-6, "um"

    async def start(self) -> None:
        self.stage = await common.get_stage()
        self.z0 = self.stage[2]

    async def apply(self, offset: np.ndarray) -> np.ndarray:
        self.stage = await common.move_stage(z=self.z0 + float(offset[0]))
        return np.array([self.stage[2] - self.z0])

    async def restore(self) -> None:
        self.stage = await common.move_stage(z=self.z0)

    def current(self, current: dict) -> dict:
        return {**current, "stage": self.stage}


class Corrector:
    """Relative correct_aberration steps; the node keeps the running total itself."""
    unit, unit_name = 1e-9, "nm"

    def __init__(self, name: str) -> None:
        self.name = name
        self.dims = 2 if name == "A1" else 1
        self.total = np.zeros(self.dims)

    async def start(self) -> None:
        self.total = np.zeros(self.dims)

    async def step(self, delta: np.ndarray) -> None:
        if np.max(np.abs(delta)) > LIMITS["max_corrector_step_m"]:
            raise ValueError(f"{self.name} step {delta} m exceeds "
                             f"{LIMITS['max_corrector_step_m']:.3g} m")
        await session.call("correct_aberration", " ".join([self.name, *[f"{v:.12g}" for v in delta]]))
        self.total = self.total + delta

    async def apply(self, offset: np.ndarray) -> np.ndarray:
        delta = np.asarray(offset, dtype=float) - self.total
        if np.any(delta != 0):
            await self.step(delta)
        return self.total.copy()

    async def restore(self) -> None:
        if np.any(self.total != 0):
            await self.step(-self.total)

    def current(self, current: dict) -> dict:
        if self.name == "C1":
            return {**current, "c1_offset_m": current.get("c1_offset_m", 0.0) + float(self.total[0])}
        before = current.get("a1_offset_m", [0.0, 0.0])
        return {**current, "a1_offset_m": [before[0] + float(self.total[0]),
                                           before[1] + float(self.total[1])]}


def next_step(state: dict) -> dict | None:
    """First protocol step not done in this run (merged with its defaults), or None."""
    run_dir = state["run"]["run_dir"]
    done = {name for name, result in (state.get("tuning") or {}).items()
            if result.get("run_dir") == run_dir}
    return next((step for step in protocol_settings(state["settings"]) if step["name"] not in done), None)


@safe_node()
async def tune_node(state: dict) -> dict:
    step = next_step(state)
    if step is None:
        print("[tune] protocol done, nothing to tune")
        return {}
    name = step["name"]
    detector = state["settings"].get("detector", "haadf")
    run_dir = Path(state["run"]["run_dir"])
    current = state.get("current") or {}
    actuator = StageZ() if step["tune"] == "z" else Corrector(step["tune"])
    if step["method"] == "grid":
        search = GridSearch(step["step_m"], step["max_offset_m"], step["min_step_m"], step["rounds"])
    else:
        search = QuadraticSearch(actuator.dims, step["step_m"], step["max_offset_m"],
                                 step["max_iterations"], min_step=step["min_step_m"])
    await actuator.start()
    rows = []
    try:
        if step.get("fov_m") is not None:
            current = {**current, "fov_m": await set_fov(step["fov_m"])}
        while (offset := search.ask()) is not None:
            reached = await actuator.apply(offset)
            key, image = await common.acquire_haadf(detector)
            metrics = all_metrics(image)
            search.tell(reached, metrics[step["metric"]])
            rows.append({"kind": search.history[-1].kind, "asked_m": offset.tolist(),
                         "reached_m": reached.tolist(), "metrics": metrics, "tiled_key": key})
            print(f"[tune {name}] {len(rows):2d} {rows[-1]['kind']:<5} offset "
                  f"{np.round(reached / actuator.unit, 3).tolist()} {actuator.unit_name}  "
                  f"{step['metric']} {metrics[step['metric']]:.5g}")
        best = search.best()
        await actuator.apply(best.offset)
    except Exception as exc:
        cause = f"step {name!r} {type(exc).__name__}: {exc}"
        try:
            await actuator.restore()
            print(f"[tune {name}] restored the start value after: {exc}")
        except Exception as restore_exc:
            raise RuntimeError(f"{cause} | RESTORE FAILED: {restore_exc}") from exc
        raise RuntimeError(cause) from exc

    best_row = rows[int(np.argmax([point.score for point in search.history]))]
    history_path = run_dir / f"tune_{name}.json"
    history_path.write_text(json.dumps({"name": name, "settings": step, "points": rows,
                                        "stop_reason": search.stop_reason}, indent=2),
                            encoding="utf-8")
    figure_path = run_dir / f"tune_{name}.png"
    plots.plot_search(search, f"tune {name} ({step['metric']})", unit=actuator.unit,
                      unit_name=actuator.unit_name, path=figure_path)
    print(f"[tune {name}] best {np.round(best.offset / actuator.unit, 3).tolist()} "
          f"{actuator.unit_name}, {step['metric']} {best.score:.5g}, {len(rows)} images, "
          f"stop: {search.stop_reason}")
    result = {"name": name, "tune": step["tune"], "fov_m": current.get("fov_m"),
              "run_dir": str(run_dir), "settings": step, "best_offset_m": best.offset.tolist(),
              "best_score": best.score, "n_images": len(rows), "stop_reason": search.stop_reason,
              "best_image_key": best_row["tiled_key"], "history_path": str(history_path),
              "figure_path": str(figure_path)}
    return {"tuning": {name: result}, "current": actuator.current(current)}
