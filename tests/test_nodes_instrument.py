"""Block 1 nodes on the fake microscope: preflight, status, move, set_fov, overview, eds."""
from pathlib import Path

import numpy as np
import pytest
from conftest import settings, started

from stem_agent.config import LIMITS
from stem_agent.nodes.eds_node import beam_position, eds_node
from stem_agent.nodes.move_stage_node import move_stage_node
from stem_agent.nodes.overview_node import overview_node
from stem_agent.nodes.preflight_node import preflight_node
from stem_agent.nodes.set_fov_node import set_fov_node
from stem_agent.nodes.status_node import device_state, normalize_parameters, status_node
from stem_agent.utils.safety import safe_node


# --- preflight ---

async def test_preflight_creates_run(fake):
    update = await preflight_node({"settings": settings()})
    assert update["run_status"] == "running"
    assert update["run"]["tools"]["acquire_image"] == "DigitalTwinParticles_acquire_scanned_image"
    assert (Path(update["run"]["run_dir"]) / "settings.json").exists()


async def test_preflight_missing_tool(fake):
    fake.remove_tool("place_beam")
    update = await preflight_node({"settings": settings()})
    assert update["run_status"] == "failed" and "place_beam" in update["errors"][0]


async def test_preflight_renamed_keys(fake):
    update = await preflight_node({"settings": {"target": "twin", "target_xy_m": (0.0, 0.0),
                                                "tuning": {"z": {}}}})
    assert update["run_status"] == "failed"
    assert "'target' is now 'instrument'" in update["errors"][0] and "stage_xy_m" in update["errors"][0]
    assert "'tuning' is now 'protocol'" in update["errors"][0]


async def test_preflight_needs_only_instrument(fake):
    update = await preflight_node({"settings": {"instrument": "twin"}})
    assert update["run_status"] == "running"
    assert not fake.called("move_stage") and not fake.called("set_fov")


# --- status ---

@pytest.mark.parametrize("value, expected", [("0", "ON"), (0, "ON"), ("8", "FAULT"),
                                             ("DevState.ALARM", "ALARM"), ("on", "ON")])
def test_device_state(value, expected):
    assert device_state(value) == expected


def test_normalize_real_and_twin():
    real = normalize_parameters({"optical_mode": "Stem", "beam_blanker": False,
                                 "column_valves": "Open", "is_accelerator_on": True,
                                 "beam_stopper": "Retracted"})
    assert real == {"stem_mode": True, "beam_blanked": False, "column_valves_open": True,
                    "high_tension_on": True, "beam_stopper_out": True}
    twin = normalize_parameters({"stem_mode": True})
    assert twin["stem_mode"] is True and twin["beam_blanked"] is None


async def test_status_twin_lenient(fake):
    state = await started()
    first = state["snapshots"][0]
    assert first["ready"] and first["index"] == 0 and state["run_status"] == "running"
    assert "beam_blanked unknown" in first["warnings"]
    assert first["devices"] == {"DATA": "ON", "SCAN": "ON", "Stage": "ON"}
    assert Path(first["raw_path"]).name == "status_00.json" and Path(first["raw_path"]).exists()


async def test_status_real_ready(fake_real):
    state = await started("real")
    assert state["snapshots"][0]["ready"] and not state["snapshots"][0]["warnings"]


async def test_status_real_unknown_stops(fake_real):
    del fake_real.params["beam_stopper"]
    state = await started("real")
    assert state["run_status"] == "failed" and "beam_stopper_out unknown" in state["errors"][0]


async def test_status_bad_value_and_device(fake_real):
    fake_real.params["beam_blanker"] = True
    fake_real.states["SCAN"] = "8"
    state = await started("real")
    first = state["snapshots"][0]
    assert not first["ready"] and state["run_status"] == "failed"
    assert any("beam_blanked" in p for p in first["problems"])
    assert "device SCAN is FAULT" in first["problems"]


async def test_later_snapshot_only_records(fake):
    state = await started()
    fake.states["SCAN"] = "8"
    update = await status_node(state)
    assert "run_status" not in update and "errors" not in update        # no gate
    assert [s["index"] for s in update["snapshots"]] == [0, 1]
    assert not update["snapshots"][-1]["ready"]                         # but recorded
    assert Path(update["snapshots"][-1]["raw_path"]).name == "status_01.json"


async def test_snapshot_keeps_tracked_offsets(fake):
    state = await started()
    state["current"] = {**state["current"], "c1_offset_m": 5e-9, "a1_offset_m": [1e-9, -2e-9]}
    fake.stage = [0.0, 0.0, 3e-6, 0.0, 0.0]
    update = await status_node(state)
    assert update["current"]["c1_offset_m"] == 5e-9                     # only the tune nodes know it
    assert update["current"]["a1_offset_m"] == [1e-9, -2e-9]
    assert update["current"]["stage"][2] == pytest.approx(3e-6)         # readback refreshed


async def test_new_run_is_gated_again(fake_real):
    state = await started("real")                                       # run 1: ready
    fake_real.params["beam_blanker"] = True
    state = {**state, **await preflight_node(state)}                    # run 2 on the same state
    assert state["snapshots"] == []
    update = await status_node(state)
    assert update["run_status"] == "failed" and "beam_blanked is True" in update["errors"][0]


async def test_no_gate_after_a_failure(fake):
    state = await started()
    state = {**state, "snapshots": [], "run_status": "failed"}         # e.g. preflight failed
    fake.states["SCAN"] = "8"
    update = await status_node(state)
    assert "run_status" not in update and len(update["snapshots"]) == 1


# --- move ---

async def test_move_keeps_tilt(fake):
    fake.stage = [0.0, 0.0, 5e-6, 1.5, 0.0]
    state = await started()
    update = await move_stage_node(state)
    assert fake.called("move_stage")[-1]["position"] == [100e-9, 0.0, 5e-6, 1.5, 0.0]
    assert update["current"]["stage"][0] == pytest.approx(100e-9)


async def test_move_without_stage_xy_stays(fake):
    state = await started(stage_xy_m=None)
    assert await move_stage_node(state) == {}
    assert fake.called("move_stage") == []


async def test_move_over_limit_does_not_move(fake):
    state = await started(stage_xy_m=(1.0, 0.0))
    update = await move_stage_node(state)
    assert update["run_status"] == "failed" and "exceeds" in update["errors"][0]
    assert [args["position"][0] for args in fake.called("move_stage")] == [0.0]   # undo only


async def test_move_bad_readback_runs_undo(fake):
    fake.stage_error_m = 2 * LIMITS["stage_tolerance_m"]
    state = await started()
    update = await move_stage_node(state)
    assert update["run_status"] == "failed" and "landed" in update["errors"][0]
    assert len(fake.called("move_stage")) == 2               # the move, then the undo


# --- set_fov ---

async def test_set_fov(fake):
    state = await started()
    update = await set_fov_node(state)
    assert update["current"]["fov_m"] == pytest.approx(40e-9)


async def test_set_fov_out_of_range(fake):
    state = await started()
    state["settings"] = {**state["settings"], "tuning_fov_m": 1e-12}
    update = await set_fov_node(state)
    assert update["run_status"] == "failed"
    assert [args["fov"] for args in fake.called("set_fov")] == [20e-9]   # undo only


# --- overview ---

async def test_overview(fake):
    state = await started()
    update = await overview_node(state)
    assert update["overview"]["tiled_key"] == "image-1"
    assert update["overview"]["shape"] == [64, 64]
    assert Path(update["overview"]["preview_path"]).exists()


# --- eds ---

@pytest.mark.parametrize("order, expected", [("col_row", [0.75, 0.25]), ("row_col", [0.25, 0.75])])
def test_beam_position(order, expected):
    assert beam_position(row=16, col=48, shape=[64, 64], order=order) == expected


def test_beam_position_outside():
    with pytest.raises(ValueError):
        beam_position(row=70, col=10, shape=[64, 64], order="col_row")


async def test_eds_twin_elements(fake):
    fake.spectrum = (np.array([0.6, 0.4]), ["Au", "Pt"])
    state = await started(beam_order="row_col")
    state = {**state, **await overview_node(state),
             "particles": [{"id": 1, "row": 16, "col": 48}, {"id": 2, "row": 32, "col": 32}]}
    update = await eds_node(state)
    assert [r["beam_xy"] for r in update["spectra"]] == [[0.25, 0.75], [0.5, 0.5]]
    assert fake.called("place_beam")[0]["position"] == [0.25, 0.75]
    assert update["spectra"][0]["elements"] == ["Au", "Pt"]
    assert np.load(update["spectra"][0]["npy_path"]).tolist() == [0.6, 0.4]


async def test_eds_cap(fake):
    state = await started(max_particles=2)
    state = {**state, **await overview_node(state),
             "particles": [{"id": i, "row": 10, "col": 10} for i in range(5)]}
    update = await eds_node(state)
    assert len(update["spectra"]) == 2 and "elements" not in update["spectra"][0]


# --- safe_node ---

async def test_safe_node_runs_undo():
    seen = []

    async def undo(state):
        seen.append(state["x"])

    @safe_node(undo=undo)
    async def broken_node(state):
        raise RuntimeError("boom")

    update = await broken_node({"x": 1})
    assert update == {"run_status": "failed", "errors": ["broken: RuntimeError: boom"]}
    assert seen == [1]


async def test_overview_uses_live_fov(fake):
    state = await started()                    # status recorded 20 nm
    fake.fov = 186e-9                          # changed outside the state
    update = await overview_node(state)
    assert update["overview"]["fov_m"] == pytest.approx(186e-9)
