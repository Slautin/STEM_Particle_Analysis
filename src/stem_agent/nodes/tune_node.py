"""One tuning node for stage z, C1 and A1. SIDE EFFECT: changes focus / corrector.

    focus_z_node, tune_c1_node, tune_a1_node      = make_tune_node("z" / "C1" / "A1")

Loop:    QuadraticSearch.ask -> apply -> acquire_haadf -> all metrics -> tell(reached, metric);
         at the end, apply the best point
Reads:   settings (tuning overrides via config.search_settings, detector), current
Writes:  tuning[name], current (stage for z; c1_offset_m / a1_offset_m for the corrector)
MCP:     z:     get_stage / move_stage (common.move_stage), absolute z0 + offset;
                "reached" = stage readback (the fit uses it, as in the notebook)
         C1/A1: correct_aberration "<name> v [v]", relative steps (notebook apply_increment);
                "reached" = the node's own running total (the corrector has no readback)
Checks:  every point within +-max_offset (QuadraticSearch); per-move limits (common.move_stage,
         LIMITS["max_corrector_step_m"])
Restore: on any error the node puts z / the corrector back where it started (like the
         notebook's z search `finally`), then reports run_status="failed"
Saves:   tune_<name>.json (every point: asked, reached, all metrics, Tiled key), tune_<name>.png
Source:  notebook cells 5, 7, 10, 13, 15
"""
import json
from pathlib import Path

import numpy as np

from stem_agent.analysis.metrics import all_metrics
from stem_agent.analysis.optimizer import QuadraticSearch
from stem_agent.config import LIMITS, search_settings
from stem_agent.mcp import session
from stem_agent.nodes import common
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


def make_tune_node(name: str):
    async def tune_node(state: dict) -> dict:
        cfg = search_settings(state["settings"], name)
        detector = state["settings"].get("detector", "haadf")
        run_dir = Path(state["run"]["run_dir"])
        actuator = StageZ() if name == "z" else Corrector(name)
        search = QuadraticSearch(actuator.dims, cfg["step_m"], cfg["max_offset_m"],
                                 cfg["max_iterations"])
        await actuator.start()
        rows = []
        try:
            while (offset := search.ask()) is not None:
                reached = await actuator.apply(offset)
                key, image = await common.acquire_haadf(detector)
                metrics = all_metrics(image)
                search.tell(reached, metrics[cfg["metric"]])
                rows.append({"kind": search.history[-1].kind, "asked_m": offset.tolist(),
                             "reached_m": reached.tolist(), "metrics": metrics, "tiled_key": key})
                print(f"[tune_{name}] {len(rows):2d} {rows[-1]['kind']:<5} offset "
                      f"{np.round(reached / actuator.unit, 3).tolist()} {actuator.unit_name}  "
                      f"{cfg['metric']} {metrics[cfg['metric']]:.5g}")
            best = search.best()
            await actuator.apply(best.offset)
        except Exception as exc:
            try:
                await actuator.restore()
                print(f"[tune_{name}] restored the start value after: {exc}")
            except Exception as restore_exc:
                raise RuntimeError(f"{exc} | RESTORE FAILED: {restore_exc}") from exc
            raise

        best_row = rows[int(np.argmax([point.score for point in search.history]))]
        history_path = run_dir / f"tune_{name}.json"
        history_path.write_text(json.dumps({"name": name, "settings": cfg, "points": rows,
                                            "stop_reason": search.stop_reason}, indent=2),
                                encoding="utf-8")
        figure_path = run_dir / f"tune_{name}.png"
        plots.plot_search(search, f"tune {name} ({cfg['metric']})", unit=actuator.unit,
                          unit_name=actuator.unit_name, path=figure_path)
        print(f"[tune_{name}] best {np.round(best.offset / actuator.unit, 3).tolist()} "
              f"{actuator.unit_name}, {cfg['metric']} {best.score:.5g}, {len(rows)} images, "
              f"stop: {search.stop_reason}")
        result = {"name": name, "settings": cfg, "best_offset_m": best.offset.tolist(),
                  "best_score": best.score, "n_images": len(rows),
                  "stop_reason": search.stop_reason, "best_image_key": best_row["tiled_key"],
                  "history_path": str(history_path), "figure_path": str(figure_path)}
        return {"tuning": {name: result}, "current": actuator.current(state["current"])}

    tune_node.__name__ = f"tune_{name}_node"
    return safe_node()(tune_node)


focus_z_node = make_tune_node("z")
tune_c1_node = make_tune_node("C1")
tune_a1_node = make_tune_node("A1")
