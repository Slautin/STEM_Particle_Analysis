"""Shared fixtures and helpers for node tests on the fake microscope."""
import numpy as np
import pytest
from fake_mcp import FakeMicroscope

from stem_agent.nodes.preflight_node import preflight_node
from stem_agent.nodes.status_node import status_node


@pytest.fixture
def fake(monkeypatch, tmp_path):
    return FakeMicroscope("twin").install(monkeypatch, tmp_path)


@pytest.fixture
def fake_real(monkeypatch, tmp_path):
    return FakeMicroscope("real").install(monkeypatch, tmp_path)


PROTOCOL = [{"name": "z", "tune": "z"}, {"name": "C1", "tune": "C1"}, {"name": "A1", "tune": "A1"}]


def one(tune: str, **values) -> list[dict]:
    """A one-step protocol, named after what it tunes."""
    return [{"name": tune, "tune": tune, **values}]


def settings(instrument="twin", **extra) -> dict:
    return {"instrument": instrument, "stage_xy_m": (100e-9, 0.0), "tuning_fov_m": 40e-9,
            "protocol": PROTOCOL, **extra}


async def started(instrument="twin", **extra) -> dict:
    """preflight + the first status snapshot (the gate), as the graph would run them."""
    state = {"settings": settings(instrument, **extra)}
    state = {**state, **await preflight_node(state)}
    return {**state, **await status_node(state)}


def sam_cluster() -> tuple[np.ndarray, list[dict]]:
    """SAM-style labels, 40 x 60: mask 1 tightly around particles 2 and 3 (so mostly hidden by
    them), mask 4 = a part of particle 3. Areas as SAM reports them (before painting over)."""
    labels = np.zeros((40, 60), dtype=np.uint32)
    labels[8:32, 8:52] = 1
    labels[10:30, 10:29] = 2
    labels[10:30, 31:50] = 3
    labels[15:20, 35:40] = 4
    areas = [{"id": 1, "area_px": 24 * 44}, {"id": 2, "area_px": 380},
             {"id": 3, "area_px": 380}, {"id": 4, "area_px": 25}]
    return labels, areas
