"""QuadraticSearch: same points as the notebook, finds a known optimum, stays inside its limits."""
import math

import numpy as np
import pytest
from notebook_reference import hill_climb_aberration, hill_climb_stage_z

from stem_agent.analysis.optimizer import QuadraticSearch


def peak_1d(center: float, width: float):
    """Smooth positive score, highest at `center` (not a quadratic, so the fit has work to do)."""
    return lambda x: 1.0 / (1.0 + ((np.atleast_1d(x)[0] - center) / width) ** 2)


def peak_2d(center, width: float):
    a, b = center
    def score(p):
        dx, dy = (p[0] - a) / width, (p[1] - b) / width
        return 1.0 / (1.0 + dx * dx + 2 * dy * dy + dx * dy)
    return score


def run(search: QuadraticSearch, score) -> QuadraticSearch:
    while (offset := search.ask()) is not None:
        search.tell(offset, score(offset))
    return search


# --- same as the notebook ---
# The notebook's grids put candidates exactly on its r/40 and disk edges, so at a tie its choice
# follows float rounding; QuadraticSearch follows exact arithmetic there, and after a tie the two
# paths differ. So, over many random peaks (fixed seed): identical start points, the first model
# point identical in most cases and otherwise only a few grid steps away, and the same accuracy
# and number of measurements.

def sweep(kind: str, n: int = 60) -> dict:
    rng = np.random.default_rng(7)
    first_exact, first_gap, err_ours, err_notebook, count_ours, count_notebook = 0, [], [], [], [], []
    for _ in range(n):
        if kind == "z":
            start, step, center = -112.27e-6, 1e-6, rng.uniform(-12e-6, 12e-6)
            score = peak_1d(center, rng.uniform(2e-6, 6e-6))
            reference = hill_climb_stage_z(lambda z: score(z - start), start, step,
                                           max_iterations=20, max_offset_m=20e-6)
            notebook = [np.array([item["stage_z_m"] - start]) for item in reference]
            search = run(QuadraticSearch(1, step, 20e-6, max_iterations=20), score)
            truth, n_start, grid = np.array([center]), 3, 2 * step / 400
        elif kind == "c1":
            step, center = 10e-9, rng.uniform(-40e-9, 40e-9)
            score = peak_1d(center, rng.uniform(8e-9, 25e-9))
            notebook = [item["offsets_m"] for item in hill_climb_aberration(score, step, 1)]
            search = run(QuadraticSearch(1, step), score)
            truth, n_start, grid = np.array([center]), 3, 2 * step / 400
        else:
            step, center = 2e-9, rng.uniform(-4e-9, 4e-9, size=2)
            score = peak_2d(center, rng.uniform(2e-9, 5e-9))
            notebook = [item["offsets_m"] for item in hill_climb_aberration(score, step, 2)]
            search = run(QuadraticSearch(2, step), score)
            truth, n_start, grid = center, 9, 2 * step / 80
        ours = [point.offset for point in search.history]
        np.testing.assert_allclose(ours[:n_start], notebook[:n_start], rtol=0, atol=1e-9 * step)
        gap = np.linalg.norm(ours[n_start] - notebook[n_start]) / grid
        first_exact += gap < 1e-6
        first_gap.append(gap)
        notebook_best = notebook[int(np.argmax([score(p) for p in notebook]))]
        err_ours.append(np.linalg.norm(search.best().offset - truth) / step)
        err_notebook.append(np.linalg.norm(notebook_best - truth) / step)
        count_ours.append(len(ours))
        count_notebook.append(len(notebook))
    return {"first_exact": first_exact / n, "first_gap": max(first_gap),
            "err_ours": np.median(err_ours), "err_notebook": np.median(err_notebook),
            "count_ours": np.mean(count_ours), "count_notebook": np.mean(count_notebook)}


@pytest.mark.parametrize("kind", ["z", "c1", "a1"])
def test_matches_notebook(kind):
    result = sweep(kind)
    assert result["first_exact"] >= 0.8, result
    assert result["first_gap"] <= 5, result                    # grid steps, ties only
    assert result["err_ours"] <= result["err_notebook"] * 1.1 + 0.01, result
    assert abs(result["count_ours"] - result["count_notebook"]) <= 1, result


# --- behaviour ---

def test_finds_optimum_1d():
    search = run(QuadraticSearch(1, 1e-6, 20e-6, max_iterations=30), peak_1d(3.3e-6, 4e-6))
    assert search.best().offset[0] == pytest.approx(3.3e-6, abs=1e-6 / 8)
    assert search.stop_reason == "converged"


def test_finds_optimum_2d():
    search = run(QuadraticSearch(2, 2e-9, max_iterations=30), peak_2d((1.0e-9, -0.6e-9), 3e-9))
    assert np.hypot(*(search.best().offset - [1.0e-9, -0.6e-9])) < 2e-9 / 8


@pytest.mark.parametrize("dims, center", [(1, 30e-6), (2, (30e-6, 5e-6))])
def test_never_leaves_max_offset(dims, center):
    score = peak_1d(center, 5e-6) if dims == 1 else peak_2d(center, 5e-6)
    search = run(QuadraticSearch(dims, 2e-6, max_offset=10e-6, max_iterations=40), score)
    assert max(np.hypot(*point.offset) if dims == 2 else abs(point.offset[0])
               for point in search.history) <= 10e-6 * (1 + 1e-12)


def test_stops_after_max_iterations():
    search = run(QuadraticSearch(1, 1e-6, 20e-6, max_iterations=2, min_step=1e-15),
                 peak_1d(3e-6, 4e-6))
    assert search.stop_reason == "max_iterations"
    assert [p.kind for p in search.history] == ["start"] * 3 + ["model"] * 2


def test_uses_offset_reached():
    search = QuadraticSearch(1, 1e-6, 20e-6)
    asked = search.ask()
    search.tell(asked + 3e-9, 1.0)                  # e.g. stage readback
    assert search.history[0].offset[0] == pytest.approx(3e-9)


def test_ask_tell_order():
    search = QuadraticSearch(1, 1e-6, 20e-6)
    with pytest.raises(RuntimeError):
        search.tell([0.0], 1.0)
    search.ask()
    with pytest.raises(RuntimeError):
        search.ask()


@pytest.mark.parametrize("kwargs", [{"dims": 3, "step": 1.0}, {"dims": 1, "step": 0.0},
                                    {"dims": 1, "step": 2.0, "max_offset": 1.0}])
def test_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        QuadraticSearch(**kwargs)


def test_predict_matches_quadratic():
    search = run(QuadraticSearch(1, 1.0, math.inf, max_iterations=3),
                 lambda x: 5.0 - (np.atleast_1d(x)[0] - 0.3) ** 2)
    assert search.predict([0.3])[0] == pytest.approx(5.0, abs=1e-4)
