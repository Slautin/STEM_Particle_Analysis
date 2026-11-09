"""Grid search with a normal-curve fit, for one value (stage z or C1). Same interface as
QuadraticSearch, so tune_node, its files and plot_search work unchanged.

    search = GridSearch(step=200e-9, max_offset=2e-6, min_step=50e-9, rounds=1)
    while (offset := search.ask()) is not None:      # offset from the start, metres
        search.tell(offset_reached, score)           # pass the offset actually reached
    search.best()                                    # best MEASURED point
    search.history, search.stop_reason, search.predict(offsets), search.fit

One round:
    grid      center +- k*spacing (k = floor(half / spacing)), clipped to +-max_offset, measured
              in ONE direction (no backlash; time runs with position, so the fit's slope absorbs
              drift). Round 1: center 0 (the start is one of the points), half = max_offset,
              spacing = step.
    fit       score = b + c*x + a*exp(-(x - x0)^2 / (2 sigma^2))  to every point in the range
    check     the peak is real when the fit converged, a > 3 x residual noise, x0 is not within
              half a spacing of the range's ends, and spacing/2 < sigma < 2 x range
    confirm   one more image at x0 ("fit" point), unless x0 is within min_step of a measured point
    next      rounds > 1: a new grid around the best point, +-2 old spacings, half the spacing;
              stops when that spacing would be < min_step
The node ends at the best measured point, so a bad fit never leaves it worse than the grid.
No point is ever closer than min_step to a measured one (the stage / corrector dead zone).

stop_reason: "converged" (peak found and confirmed), "peak_at_edge" (optimum probably outside the
range: widen max_offset), "no_peak" (flat or noisy)
"""
import math

import numpy as np
from scipy.optimize import curve_fit

from stem_agent.analysis.optimizer import SearchPoint

MAX_POINTS = 51          # per grid; checked by config.step_settings too


def gaussian_line(u, b, c, a, u0, sigma):
    """Normal curve on a sloped baseline."""
    return b + c * u + a * np.exp(-0.5 * ((u - u0) / sigma) ** 2)


class GridSearch:
    dims = 1
    fit_name = "normal fit"

    def __init__(self, step: float, max_offset: float, min_step: float | None = None,
                 rounds: int = 1) -> None:
        if step <= 0 or max_offset <= 0:
            raise ValueError("step and max_offset must be positive")
        if step > max_offset:
            raise ValueError("step must not exceed max_offset")
        points = 2 * math.floor(max_offset / step + 1e-9) + 1
        if not 7 <= points <= MAX_POINTS:
            raise ValueError(f"grid of {points} points; need 7..{MAX_POINTS} "
                             f"(max_offset between 3 and {(MAX_POINTS - 1) // 2} steps)")
        if not 1 <= int(rounds) <= 5:
            raise ValueError("rounds must be 1..5")
        self.step = float(step)
        self.max_offset = float(max_offset)
        self.min_step = self.step / 16 if min_step is None else float(min_step)
        if not 0 < self.min_step <= self.step:
            raise ValueError("need 0 < min_step <= step")
        self.rounds = int(rounds)
        self.history: list[SearchPoint] = []
        self.stop_reason: str | None = None
        self.round = 0
        self.fit: dict | None = None          # last fit, in metres: b, c, a, x0, sigma (+ "ok")
        self._norm: tuple[float, float, float, float] | None = None
        self._pending: tuple[float, str] | None = None
        self._start_round(center=0.0, half=self.max_offset, spacing=self.step)

    # --- ask / tell, as QuadraticSearch ---
    def ask(self) -> np.ndarray | None:
        """Next offset to measure, or None when the search is done."""
        if self._pending is not None:
            raise RuntimeError("tell() the score of the last offset before asking again")
        while not self._queue and self.stop_reason is None:
            self._next_phase()
        if self.stop_reason is not None:
            return None
        self._pending = self._queue.pop(0)
        return np.array([self._pending[0]])

    def tell(self, offset, score: float) -> None:
        """Record a measurement. The fit uses `offset` as given (e.g. the stage readback)."""
        if self._pending is None:
            raise RuntimeError("ask() before tell()")
        offset = np.atleast_1d(np.asarray(offset, dtype=float))
        if offset.shape != (1,):
            raise ValueError("offset must have 1 value")
        _, kind = self._pending
        self._pending = None
        self.history.append(SearchPoint(offset.copy(), float(score), kind, self.spacing))

    def best(self) -> SearchPoint:
        if not self.history:
            raise RuntimeError("no measurements yet")
        return self.history[int(np.argmax([point.score for point in self.history]))]

    def predict(self, offsets) -> np.ndarray:
        """The last fitted curve at offsets (NaN when no fit was possible)."""
        x = np.asarray(offsets, dtype=float).reshape(-1)
        if self.fit is None:
            return np.full(x.shape, np.nan)
        mid, half, y_mean, y_std = self._norm
        p = self._fit_params
        return y_mean + y_std * gaussian_line((x - mid) / half, *p)

    # --- rounds ---
    def _start_round(self, center: float, half: float, spacing: float) -> None:
        self.round += 1
        self.spacing = spacing
        k = math.floor(half / spacing + 1e-9)
        points = center + spacing * np.arange(-k, k + 1)
        points = points[np.abs(points) <= self.max_offset * (1 + 1e-9)]
        self._lo, self._hi = float(points.min()), float(points.max())
        measured = [point.offset[0] for point in self.history]
        self._queue = [(float(x), "grid") for x in points
                       if all(abs(x - m) > self.min_step * (1 - 1e-9) for m in measured)]
        self._phase = "grid"

    def _next_phase(self) -> None:
        if self._phase == "grid":
            verdict = self._fit_round()
            if verdict is not None:
                self.stop_reason = verdict
                return
            self._phase = "confirm"
            x0 = self.fit["x0"]
            if all(abs(x0 - point.offset[0]) > self.min_step for point in self.history):
                self._queue = [(x0, "fit")]
                return
        # confirm measured (or not needed): zoom in, or stop
        next_spacing = self.spacing / 2
        if self.round >= self.rounds or next_spacing < self.min_step:
            self.stop_reason = "converged"
            return
        self._start_round(center=self.best().offset[0], half=2 * self.spacing, spacing=next_spacing)
        if not self._queue:
            self.stop_reason = "converged"

    def _fit_round(self) -> str | None:
        """Fit the points in this round's range. None if the peak is real, else a stop reason."""
        inside = [point for point in self.history
                  if self._lo - 1e-15 <= point.offset[0] <= self._hi + 1e-15]
        x = np.array([point.offset[0] for point in inside])
        y = np.array([point.score for point in inside])
        if len(x) < 6 or np.ptp(y) == 0:
            return "no_peak"
        mid, half = (self._lo + self._hi) / 2, (self._hi - self._lo) / 2
        y_mean, y_std = float(y.mean()), float(y.std())
        u, v = (x - mid) / half, (y - y_mean) / y_std
        width = self.spacing / half                                  # one spacing, normalized
        lower = [-np.inf, -np.inf, 0.0, -1.0, width / 2]
        upper = [np.inf, np.inf, np.inf, 1.0, 4.0]                   # sigma up to 2 x range
        guess = [v.min(), 0.0, np.ptp(v), float(u[np.argmax(v)]), 0.3]
        guess[4] = min(max(guess[4], lower[4] * 1.01), upper[4] * 0.99)
        try:
            p, _ = curve_fit(gaussian_line, u, v, p0=guess, bounds=(lower, upper), maxfev=5000)
        except (RuntimeError, ValueError):
            return "no_peak"
        self._norm, self._fit_params = (mid, half, y_mean, y_std), p
        noise = float(np.sqrt(np.sum((v - gaussian_line(u, *p)) ** 2) / max(len(u) - 5, 1)))
        b, c, a, u0, sigma = p
        self.fit = {"b": y_mean + y_std * (b - c * mid / half), "c": y_std * c / half,
                    "a": y_std * a, "x0": mid + half * u0, "sigma": half * sigma,
                    "noise": y_std * noise, "ok": False}
        if a <= 3 * noise or not lower[4] * 1.01 < sigma < upper[4] * 0.99:
            return "no_peak"
        if not self._lo + self.spacing / 2 <= self.fit["x0"] <= self._hi - self.spacing / 2:
            return "peak_at_edge"
        self.fit["ok"] = True
        return None
