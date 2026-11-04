"""Metrics: sharper scores higher; flat images do not crash; synthetic blur behaves."""
import numpy as np
import pytest

from stem_agent.analysis.metrics import METRICS, all_metrics
from stem_agent.analysis.synthetic import SyntheticScope, blur, particle_image


@pytest.fixture(scope="module")
def sharp():
    return particle_image(size=128, seed=3)[0]


@pytest.mark.parametrize("name", ["contrast", "edge_sharpness"])
@pytest.mark.parametrize("noise", [0.0, 0.03])
def test_sharper_scores_higher(name, noise, sharp):
    fn = METRICS[name]
    rng = np.random.default_rng(0)
    scores = [fn(blur(sharp, defocus_px=d, noise=noise, rng=rng)) for d in (0.0, 1.5, 3.0, 5.0)]
    assert scores == sorted(scores, reverse=True)


def test_high_frequency_power_near_focus_only(sharp):
    near = [METRICS["high_frequency_power"](blur(sharp, defocus_px=d)) for d in (0.0, 0.75, 1.5)]
    far = [METRICS["high_frequency_power"](blur(sharp, defocus_px=d)) for d in (2.0, 5.0)]
    assert near == sorted(near, reverse=True)
    assert far[1] > far[0]                     # rises again far from focus (finding, see metrics.py)


def test_entropy_rises_with_blur(sharp):
    scores = [METRICS["entropy"](blur(sharp, defocus_px=d)) for d in (0.0, 1.5, 3.0)]
    assert scores == sorted(scores)            # maximizing entropy would defocus (finding)


@pytest.mark.parametrize("name", list(METRICS))
def test_flat_image(name):
    assert np.isfinite(METRICS[name](np.full((32, 32), 0.5)))


def test_all_metrics_names(sharp):
    assert set(all_metrics(sharp)) == set(METRICS)


def test_astigmatism_direction_matters(sharp):
    stretched = blur(sharp, astigmatism_px=(3.0, 0.0))
    other = blur(sharp, astigmatism_px=(-3.0, 0.0))
    assert not np.allclose(stretched, other)


def test_synthetic_scope_peaks_at_optimum():
    scope = SyntheticScope(1, optimum_m=2e-6, px_per_m=1e6, size=96)
    assert scope.score(2e-6) > scope.score(0.0) > scope.score(-3e-6)
