"""Shared fixtures and helpers for node tests on the fake microscope."""
import pytest
from fake_mcp import FakeMicroscope

from stem_agent.nodes.preflight_node import preflight_node
from stem_agent.nodes.status_node import status_start_node


@pytest.fixture
def fake(monkeypatch, tmp_path):
    return FakeMicroscope("twin").install(monkeypatch, tmp_path)


@pytest.fixture
def fake_real(monkeypatch, tmp_path):
    return FakeMicroscope("real").install(monkeypatch, tmp_path)


def settings(target="twin", **extra) -> dict:
    return {"target": target, "target_xy_m": (100e-9, 0.0), "tuning_fov_m": 40e-9, **extra}


async def started(target="twin", **extra) -> dict:
    """preflight + status_start, as the graph would run them."""
    state = {"settings": settings(target, **extra)}
    state = {**state, **await preflight_node(state)}
    return {**state, **await status_start_node(state)}
