"""The whole graph on the fake microscope: normal run, no particles, failures, not ready."""
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import one, settings

from stem_agent.config import LIMITS
from stem_agent.graphs.particle_graph import build_graph, check_settings, run
from stem_agent.utils.plots import plot_graph

NODES = ["preflight", "status_start", "move", "tune", "set_fov", "overview", "segment", "eds",
         "status_end", "save"]


def quick(instrument="twin", **extra) -> dict:
    """Settings with short searches, so a full run stays fast."""
    protocol = [{"name": "z", "tune": "z", "max_iterations": 8},
                {"name": "C1", "tune": "C1", "fov_m": 40e-9}, {"name": "A1", "tune": "A1"}]
    return settings(instrument, **{"protocol": protocol, **extra})


def summary(state: dict) -> dict:
    return json.loads(Path(state["summary_path"]).read_text())


def test_graph_structure():
    drawable = build_graph().get_graph()
    assert [n for n in drawable.nodes if not n.startswith("__")] == NODES
    jumps = {e.source for e in drawable.edges if e.target == "status_end" and e.conditional}
    assert jumps == set(NODES[:7])                     # every step up to segment can jump
    assert ("tune", "tune") in {(e.source, e.target) for e in drawable.edges}   # the protocol loop


def test_plot_graph_draws_the_loop():
    fig = plot_graph(build_graph())
    assert "repeat" in [text.get_text() for text in fig.axes[0].texts]


async def test_long_protocol_fits_the_step_limit(fake):
    fake.enable_blur()                                                # an image with particles
    protocol = [{"name": f"z{i:02d}", "tune": "z", "max_iterations": 0} for i in range(20)]
    state = await run(quick(protocol=protocol), verbose=False)       # > LangGraph's default 25 runs
    assert state["run_status"] == "ok" and len(state["tuning"]) == 20


async def test_normal_run(fake):
    fake.enable_blur(z_m=3e-6, c1_m=-15e-9, a1_m=(2e-9, -1e-9))
    fake.spectrum = (np.array([0.7, 0.3]), ["Au", "Pt"])
    state = await run(quick(), verbose=False)
    assert state["run_status"] == "ok" and state["errors"] == []
    assert set(state["tuning"]) == {"z", "C1", "A1"}
    assert state["tuning"]["z"]["best_offset_m"][0] == pytest.approx(3e-6, abs=0.5e-6)
    assert state["tuning"]["C1"]["best_offset_m"][0] == pytest.approx(-15e-9, abs=3e-9)
    assert len(state["particles"]) > 0
    assert len(state["spectra"]) == min(len(state["particles"]), 20)
    assert len(state["snapshots"]) == 2 and Path(state["summary_path"]).exists()   # start, end
    assert (Path(state["run"]["run_dir"]) / "summary.png").exists()
    saved = summary(state)
    assert saved["run_status"] == "ok" and saved["tuning"]["A1"]["n_images"] > 0


async def test_no_particles_skips_eds(fake):
    fake.enable_blur()
    state = await run(quick(sam={"min_diameter_nm": 50.0}), verbose=False)
    assert state["run_status"] == "no_particles"
    assert not fake.called("place_beam") and "spectra" not in state
    assert len(state["snapshots"]) == 2 and summary(state)["run_status"] == "no_particles"


async def test_failure_during_focus(fake):
    fake.enable_blur(z_m=3e-6)
    fake.stage = [0.0, 0.0, -7e-6, 0.0, 0.0]
    fake.fail_image_after = 4                          # the 5th image (inside focus_z) fails
    state = await run(quick(), verbose=False)
    assert state["run_status"] == "failed"
    assert len(state["errors"]) == 1 and "simulated acquisition failure" in state["errors"][0]
    assert fake.stage[2] == pytest.approx(-7e-6)       # z restored
    assert not fake.called("correct_aberration")       # C1 / A1 never ran
    assert state["snapshots"][-1]["stage"][2] == pytest.approx(-7e-6)  # end snapshot sees it
    assert summary(state)["errors"] == state["errors"]


async def test_not_ready_moves_nothing(fake_real):
    fake_real.params["beam_blanker"] = True
    state = await run(quick("real"), verbose=False)
    assert state["run_status"] == "failed" and "not ready" in state["errors"][0]
    assert not fake_real.called("move_stage") and not fake_real.called("correct_aberration")
    assert Path(state["summary_path"]).exists()


@pytest.mark.parametrize("extra, message", [
    ({"protocol": one("z", metric="entropy")}, "toward blur"),
    ({"protocol": one("A1", metric="high_frequency_power")}, "toward blur"),
    ({"protocol": one("C1", max_offset_m=LIMITS["max_corrector_step_m"])}, "half the per-move limit"),
    ({"protocol": one("z", step_m=30e-6)}, "step_m <= max_offset_m"),
    ({"protocol": one("z", stepsize=1e-6)}, "unknown keys"),
    ({"protocol": []}, "no protocol"),
    ({"tuning_fov_m": 1.0}, "FOV"),
    ({"segmentation": {"min_diameter_nm": 2.0}}, "segments with SAM"),
    ({"sam": {"min_visible": 0}}, "bad sam settings"),
    ({"sam": {"min_area": 5}}, "unknown sam settings"),
    ({"max_particles": 0}, "max_particles"),
    ({"stage_xy_m": (1.0,)}, "stage_xy_m"),
])
async def test_bad_settings_stop_before_the_run(fake, extra, message):
    with pytest.raises(ValueError, match=message):
        check_settings(settings(**extra))
    with pytest.raises(ValueError, match=message):
        await run(settings(**extra), verbose=False)
    assert fake.calls == []                            # not even preflight ran


async def test_preflight_failure_still_saves(fake):
    fake.remove_tool("place_beam")
    state = await run(quick(), verbose=False)
    assert state["run_status"] == "failed" and "missing MCP tools" in state["errors"][0]
    assert Path(state["run"]["run_dir"], "settings.json").exists()
    assert Path(state["summary_path"]).exists()


async def test_errors_add_up(fake):
    fake.remove_tool("place_beam")                     # preflight fails ...
    fake.fail[next(n for n in fake._tools if n.endswith("_get_stage"))] = "stage offline"   # ... and status_end too
    state = await run(quick(), verbose=False)
    assert len(state["errors"]) == 2
    assert state["errors"][0].startswith("preflight") and state["errors"][1].startswith("status:")
