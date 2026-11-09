"""Quadratic trust-region search, shared by stage z, C1 and A1.

The notebook calls it "hill climbing"; it fits a quadratic to every measured point and
measures where the fit predicts the best score, inside a trust radius around the best point.

    search = QuadraticSearch(dims=1, step=1e-6, max_offset=20e-6, max_iterations=8)
    while (offset := search.ask()) is not None:      # offset from the start, metres
        search.tell(offset_reached, score)           # pass the offset actually reached
    search.best()                                    # SearchPoint(offset, score, ...)
    search.history, search.stop_reason, search.predict(offsets)

Algorithm (same numbers as the notebook):
    start points: 0, -step, +step (1D) or 0 + 8 points on a circle of radius step (2D)
    fit a ridge-regularized quadratic (normalized offsets) to all points
    propose the best predicted unmeasured point within the trust radius (401 points in 1D,
    81x81 grid inside a disk in 2D), never further than max_offset from the start
    radius x1.2 (cap 2*step and max_offset) if the score beats the previous best by > 0.1 %,
    else x0.65; stop when radius < min_step (default step/16), after max_iterations model
    points, or when no unmeasured candidate is left
    a candidate must be > radius/40 from every measured point; when min_step is GIVEN, also
    > min_step, so no move is smaller than what the stage / corrector can do (default: as the
    notebook)

Source: notebook cell 5 (fit_response, propose_next_z, hill_climb_stage_z) and
        cell 7 (fit_response_surface, propose_next_point, hill_climb_aberration), merged;
        max_offset is new for C1/A1.
Difference: the notebook's grids place candidates exactly on the "too close" (r/40) and disk
        edges, so its choice there depends on float rounding (units, start z). Here those edges
        follow exact arithmetic. tests/test_optimizer.py compares both with the notebook code.
"""
import math
from dataclasses import dataclass

import numpy as np

EDGE = 1e-9             # relative tolerance at the grid's exact boundaries (see _propose)


@dataclass
class SearchPoint:
    offset: np.ndarray        # from the start, metres; shape (dims,)
    score: float
    kind: str                 # "start" or "model"
    radius: float             # trust radius when the point was measured


class QuadraticSearch:
    def __init__(self, dims: int, step: float, max_offset: float = math.inf,
                 max_iterations: int = 8, min_step: float | None = None) -> None:
        if dims not in (1, 2):
            raise ValueError("dims must be 1 or 2")
        if step <= 0 or max_offset <= 0:
            raise ValueError("step and max_offset must be positive")
        if step > max_offset:
            raise ValueError("step must not exceed max_offset")
        self.dims = dims
        self.step = float(step)
        self.max_offset = float(max_offset)
        self.max_iterations = int(max_iterations)
        self.min_step = self.step / 16 if min_step is None else float(min_step)
        self._min_gap = 0.0 if min_step is None else float(min_step)   # given: no smaller moves
        self.radius = self.step
        self.iterations = 0
        self.history: list[SearchPoint] = []
        self.stop_reason: str | None = None
        self._queue = [np.zeros(dims), *self._start_points()]
        self._pending: tuple[np.ndarray, str] | None = None
        self._previous_best = -math.inf

    def _start_points(self) -> list[np.ndarray]:
        if self.dims == 1:
            return [np.array([-self.step]), np.array([self.step])]
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        return [self.step * np.array([np.cos(a), np.sin(a)]) for a in angles]

    def ask(self) -> np.ndarray | None:
        """Next offset to measure, or None when the search is done."""
        if self._pending is not None:
            raise RuntimeError("tell() the score of the last offset before asking again")
        if self.stop_reason is not None:
            return None
        if self._queue:
            offset, kind = self._queue.pop(0), "start"
        else:
            if self.iterations >= self.max_iterations:
                self.stop_reason = "max_iterations"
                return None
            offset = self._propose()
            if offset is None:
                self.stop_reason = "no_candidates"
                return None
            kind = "model"
            self._previous_best = max(point.score for point in self.history)
        self._pending = (offset, kind)
        return offset.copy()

    def tell(self, offset, score: float) -> None:
        """Record a measurement. The fit uses `offset` as given (e.g. the stage readback)."""
        if self._pending is None:
            raise RuntimeError("ask() before tell()")
        offset = np.atleast_1d(np.asarray(offset, dtype=float))
        if offset.shape != (self.dims,):
            raise ValueError(f"offset must have {self.dims} value(s)")
        _, kind = self._pending
        self._pending = None
        self.history.append(SearchPoint(offset.copy(), float(score), kind, self.radius))
        if kind == "model":
            self.iterations += 1
            if score > self._previous_best * 1.001:
                self.radius = min(self.radius * 1.20, self.step * 2, self.max_offset)
            else:
                self.radius *= 0.65
            if self.radius < self.min_step:
                self.stop_reason = "converged"

    def best(self) -> SearchPoint:
        if not self.history:
            raise RuntimeError("no measurements yet")
        return self.history[int(np.argmax([point.score for point in self.history]))]

    def predict(self, offsets) -> np.ndarray:
        """Fitted score at offsets, shape (n, dims) or (n,) in 1D."""
        offsets = np.asarray(offsets, dtype=float)
        if offsets.ndim == 1:
            offsets = offsets[:, None] if self.dims == 1 else offsets[None, :]
        points = np.array([point.offset for point in self.history])
        scores = np.array([point.score for point in self.history])
        center = points.mean(axis=0)
        scale = np.maximum(points.std(axis=0), 1e-15)
        design = self._design((points - center) / scale)
        ridge = 1e-6 * np.eye(design.shape[1])
        coefficients = np.linalg.solve(design.T @ design + ridge, design.T @ scores)
        return self._design((offsets - center) / scale) @ coefficients

    @staticmethod
    def _design(z: np.ndarray) -> np.ndarray:
        if z.shape[1] == 2:
            x, y = z.T
            return np.column_stack([np.ones(len(z)), x, y, x * x, x * y, y * y])
        x = z[:, 0]
        return np.column_stack([np.ones(len(z)), x, x * x])

    def _propose(self) -> np.ndarray | None:
        offsets = np.array([point.offset for point in self.history])
        best = offsets[int(np.argmax([point.score for point in self.history]))]
        r = self.radius
        # The grids put candidates exactly on the disk edge and exactly r/40 from measured
        # points; EDGE makes those cases follow exact arithmetic instead of float rounding.
        if self.dims == 1:
            lower = max(best[0] - r, -self.max_offset)
            upper = min(best[0] + r, self.max_offset)
            candidates = np.linspace(lower, upper, 401)[:, None]
        else:
            axis = np.linspace(-r, r, 81)
            dx, dy = np.meshgrid(axis, axis)
            inside = dx * dx + dy * dy <= r * r * (1 + EDGE)
            candidates = np.column_stack([best[0] + dx[inside], best[1] + dy[inside]])
            candidates = candidates[np.hypot(candidates[:, 0], candidates[:, 1])
                                    <= self.max_offset * (1 + EDGE)]
        separation = np.linalg.norm(candidates[:, None, :] - offsets[None, :, :], axis=2).min(axis=1)
        candidates = candidates[separation > max(r / 40, self._min_gap, 1e-15) * (1 + EDGE)]
        if not len(candidates):
            return None
        return candidates[int(np.argmax(self.predict(candidates)))]
