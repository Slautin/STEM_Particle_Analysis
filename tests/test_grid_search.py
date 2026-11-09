"""GridSearch: grid, normal-curve fit, confirm point, zoom rounds, guards."""
import numpy as np
import pytest

from stem_agent.analysis.grid_search import GridSearch, gaussian_line


def run(search, score, rng=None, noise=0.0):
    while (offset := search.ask()) is not None:
        value = score(offset[0], len(search.history))
        search.tell(offset, value + (rng.normal(0, noise) if rng is not None else 0.0))
    return search


def peak(x0, width, height=1.0, base=0.2):
    return lambda x, i: base + height * np.exp(-0.5 * ((x - x0) / width) ** 2)


def test_grid_then_one_image_at_the_fitted_peak():
    search = run(GridSearch(step=200e-9, max_offset=2e-6), peak(0.37e-6, 0.8e-6))
    kinds = [p.kind for p in search.history]
    assert kinds == ["grid"] * 21 + ["fit"]
    grid = [p.offset[0] for p in search.history[:21]]
    assert grid == sorted(grid) and 0.0 in grid                          # one direction; start included
    assert search.fit["x0"] == pytest.approx(0.37e-6, abs=10e-9)
    assert search.best().kind == "fit" and search.stop_reason == "converged"


def test_slope_absorbs_drift():
    drift = lambda x, i: peak(0.3e-6, 0.8e-6)(x, i) - 0.02 * i            # image gets worse over time
    search = run(GridSearch(step=200e-9, max_offset=2e-6), drift)
    raw_best = search.history[int(np.argmax([p.score for p in search.history[:21]]))].offset[0]
    assert search.fit["x0"] == pytest.approx(0.3e-6, abs=40e-9)           # the fit is not fooled ...
    assert abs(raw_best - 0.3e-6) >= abs(search.fit["x0"] - 0.3e-6)       # ... the raw grid best is no better


def test_noise_still_finds_the_peak():
    rng = np.random.default_rng(1)
    search = run(GridSearch(step=200e-9, max_offset=2e-6), peak(-0.5e-6, 0.7e-6), rng, noise=0.03)
    assert search.stop_reason == "converged"
    assert search.fit["x0"] == pytest.approx(-0.5e-6, abs=100e-9)


def test_flat_is_no_peak_and_no_extra_image():
    rng = np.random.default_rng(2)
    search = run(GridSearch(step=1e-6, max_offset=5e-6), lambda x, i: 0.5, rng, noise=0.01)
    assert search.stop_reason == "no_peak" and len(search.history) == 11


def test_optimum_outside_is_peak_at_edge():
    search = run(GridSearch(step=1e-6, max_offset=5e-6), peak(9e-6, 3e-6))
    assert search.stop_reason == "peak_at_edge" and len(search.history) == 11
    assert search.best().offset[0] == pytest.approx(5e-6)                 # best grid point


def test_rounds_zoom_in_and_respect_min_step():
    search = run(GridSearch(step=200e-9, max_offset=2e-6, min_step=50e-9, rounds=5),
                 peak(0.33e-6, 0.6e-6))
    points = np.array([p.offset[0] for p in search.history])
    gaps = np.abs(points[:, None] - points[None, :])[np.triu_indices(len(points), 1)]
    assert gaps.min() > 50e-9 * (1 - 1e-9)                               # never inside the dead zone
    assert search.round == 3 and search.spacing == pytest.approx(50e-9)  # 200 -> 100 -> 50 nm, then stop
    assert search.stop_reason == "converged"
    assert search.best().offset[0] == pytest.approx(0.33e-6, abs=25e-9)


def test_predict_is_the_fitted_curve():
    search = run(GridSearch(step=200e-9, max_offset=2e-6), peak(0.0, 0.8e-6))
    x = np.array([p.offset[0] for p in search.history])
    y = np.array([p.score for p in search.history])
    np.testing.assert_allclose(search.predict(x), y, atol=1e-3)


def test_gaussian_line():
    assert gaussian_line(0.0, 1.0, 0.0, 2.0, 0.0, 1.0) == pytest.approx(3.0)


@pytest.mark.parametrize("kwargs", [
    {"step": 1e-6, "max_offset": 2e-6},                  # 5 points: too few to fit
    {"step": 1e-7, "max_offset": 3e-6},                  # 61 points: too many
    {"step": 2e-6, "max_offset": 1e-6},
    {"step": 1e-6, "max_offset": 5e-6, "rounds": 0},
    {"step": 1e-6, "max_offset": 5e-6, "min_step": 2e-6},
])
def test_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        GridSearch(**kwargs)


def test_ask_tell_order():
    search = GridSearch(step=1e-6, max_offset=5e-6)
    search.ask()
    with pytest.raises(RuntimeError):
        search.ask()
