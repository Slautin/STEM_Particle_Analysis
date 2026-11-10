"""Block 3 nodes on the fake microscope: tune (z / C1 / A1), segment (Otsu) and SAM segment.

The fake's image blurs with the distance of stage z / C1 / A1 from a hidden optimum, so these
tests check that each node finds it through the full path: MCP calls -> image -> metric -> search.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import one, sam_cluster, started

from stem_agent.analysis.synthetic import blur, particle_image
from stem_agent.mcp import session
from stem_agent.nodes.overview_node import overview_node
from stem_agent.nodes.sam_segment_node import sam_segment_node
from stem_agent.nodes.segment_node import segment_node
from stem_agent.nodes.tune_node import next_step, tune_node


# --- tune: finds the optimum, ends there ---

async def test_focus_z_finds_optimum(fake):
    fake.enable_blur(z_m=-112e-6 + 4.3e-6)                               # absolute z
    fake.stage = [0.0, 0.0, -112e-6, 0.0, 0.0]
    state = await started(protocol=one("z"))
    update = await tune_node(state)
    result = update["tuning"]["z"]
    assert result["best_offset_m"][0] == pytest.approx(4.3e-6, abs=0.25e-6)
    assert fake.stage[2] == pytest.approx(-112e-6 + result["best_offset_m"][0])   # ended at best
    assert update["current"]["stage"][2] == pytest.approx(fake.stage[2])
    assert Path(result["history_path"]).exists() and Path(result["figure_path"]).exists()
    points = json.loads(Path(result["history_path"]).read_text())["points"]
    assert len(points) == result["n_images"] and set(points[0]["metrics"]) >= {"contrast", "entropy"}


async def test_tune_c1_finds_optimum(fake):
    fake.enable_blur(c1_m=23e-9)
    state = await started(protocol=one("C1"))
    update = await tune_node(state)
    best = update["tuning"]["C1"]["best_offset_m"][0]
    assert best == pytest.approx(23e-9, abs=2.5e-9)
    assert fake.corrector["C1"][0] == pytest.approx(best)                 # ended at best
    assert update["current"]["c1_offset_m"] == pytest.approx(best)
    assert fake.called("correct_aberration")[0]["args"] == "C1 -1e-08"    # first step: -step


async def test_tune_a1_finds_optimum(fake):
    fake.enable_blur(a1_m=(3e-9, -2e-9))
    state = await started(protocol=one("A1"))
    update = await tune_node(state)
    best = np.array(update["tuning"]["A1"]["best_offset_m"])
    assert np.hypot(*(best - [3e-9, -2e-9])) < 1e-9
    assert fake.corrector["A1"] == pytest.approx(best.tolist())
    assert update["current"]["a1_offset_m"] == pytest.approx(best.tolist())
    assert fake.called("correct_aberration")[0]["args"] == "A1 2e-09 0"   # first circle point


# --- tune: limits ---

async def test_z_stays_within_max_offset(fake):
    fake.enable_blur(z_m=60e-6)                                           # far beyond the limit
    state = await started(protocol=one("z", max_offset_m=10e-6, step_m=2e-6))
    update = await tune_node(state)
    zs = [args["position"][2] for args in fake.called("move_stage")]
    assert max(abs(z) for z in zs) <= 10e-6 * (1 + 1e-9)
    assert update["tuning"]["z"]["best_offset_m"][0] == pytest.approx(10e-6, abs=1e-6)


async def test_corrector_steps_within_limit(fake):
    fake.enable_blur(c1_m=-45e-9)
    state = await started(protocol=one("C1"))
    await tune_node(state)
    steps = [abs(float(v)) for args in fake.called("correct_aberration") for v in args["args"].split()[1:]]
    assert max(steps) <= 100e-9


# --- tune: min_step_m, on a stage / corrector that ignores small moves ---

async def test_z_min_step_no_stall_on_a_dead_zone_stage(fake):
    fake.enable_blur(z_m=-0.6e-6)
    fake.stage_dead_zone_m = 40e-9                                         # ignores smaller moves
    state = await started(protocol=one("z", step_m=200e-9, max_offset_m=2e-6, max_iterations=20,
                                       min_step_m=50e-9))
    update = await tune_node(state)
    result = update["tuning"]["z"]
    points = json.loads(Path(result["history_path"]).read_text())["points"]
    reached = [p["reached_m"][0] for p in points]
    assert len(set(reached)) == len(reached)                               # every image: a new z
    zs = [0.0] + [args["position"][2] for args in fake.called("move_stage")]
    moves = [abs(b - a) for a, b in zip(zs, zs[1:]) if b != a]
    assert min(moves) > 50e-9                                              # never below min_step
    assert result["settings"]["min_step_m"] == 50e-9
    assert result["best_offset_m"][0] == pytest.approx(-0.6e-6, abs=60e-9)


async def test_c1_min_step_every_step_reaches_the_corrector(fake):
    fake.enable_blur(c1_m=-23e-9)
    fake.corrector_dead_zone_m = 2e-9                                      # ignores smaller steps
    state = await started(protocol=one("C1", max_iterations=20, min_step_m=2e-9))
    update = await tune_node(state)
    sent = [float(args["args"].split()[1]) for args in fake.called("correct_aberration")]
    assert min(abs(v) for v in sent) > 2e-9                                # none ignored
    assert fake.corrector["C1"][0] == pytest.approx(update["current"]["c1_offset_m"], abs=1e-18)
    assert update["tuning"]["C1"]["best_offset_m"][0] == pytest.approx(-23e-9, abs=1.5e-9)


@pytest.mark.parametrize("step, message", [
    ({"tune": "z", "step_m": 200e-9}, "below the stage tolerance"),         # default 12.5 nm
    ({"tune": "z", "step_m": 200e-9, "min_step_m": 10e-9}, "below the stage tolerance"),
    ({"tune": "C1", "step_m": 5e-9, "min_step_m": 6e-9}, "min_step_m <= step_m"),
    ({"tune": "A1", "step_m": 2e-9, "min_step_m": 1.6e-9}, "min_step_m <= 0.75 * step_m"),
    ({"tune": "C1", "step_m": 5e-9, "min_step_m": 0.0}, "min_step_m <= step_m"),
])
async def test_bad_min_step_moves_nothing(fake, step, message):
    state = await started(protocol=[{"name": "s", **step}])
    update = await tune_node(state)
    assert update["run_status"] == "failed" and message in update["errors"][0]
    assert not fake.called("move_stage") and not fake.called("correct_aberration")


async def test_min_step_default_is_step_over_16(fake):
    fake.enable_blur(c1_m=5e-9)
    update = await tune_node(await started(protocol=one("C1", step_m=8e-9)))
    assert update["tuning"]["C1"]["settings"]["min_step_m"] == pytest.approx(0.5e-9)


# --- tune: grid search ---

async def test_grid_z_finds_optimum_and_ends_there(fake):
    fake.enable_blur(z_m=2.3e-6)
    state = await started(protocol=one("z", method="grid", step_m=1e-6, max_offset_m=6e-6,
                                       min_step_m=100e-9))
    update = await tune_node(state)
    result = update["tuning"]["z"]
    points = json.loads(Path(result["history_path"]).read_text())["points"]
    assert [p["kind"] for p in points] == ["grid"] * 13 + ["fit"]
    sweep = [p["asked_m"][0] for p in points[:13]]
    assert sweep == sorted(sweep)                                          # one direction
    assert result["stop_reason"] == "converged" and result["settings"]["method"] == "grid"
    assert result["best_offset_m"][0] == pytest.approx(2.3e-6, abs=0.4e-6)
    assert fake.stage[2] == pytest.approx(result["best_offset_m"][0])     # ended at the best point
    assert Path(result["figure_path"]).exists()


async def test_grid_c1_on_a_dead_zone_corrector(fake):
    fake.enable_blur(c1_m=-14e-9)
    fake.corrector_dead_zone_m = 2e-9
    state = await started(protocol=one("C1", method="grid", step_m=4e-9, max_offset_m=40e-9,
                                       min_step_m=2e-9, rounds=2))
    update = await tune_node(state)
    sent = [float(args["args"].split()[1]) for args in fake.called("correct_aberration")]
    assert min(abs(v) for v in sent) > 2e-9                                # every step reached it
    assert fake.corrector["C1"][0] == pytest.approx(update["current"]["c1_offset_m"], abs=1e-18)
    assert update["tuning"]["C1"]["best_offset_m"][0] == pytest.approx(-14e-9, abs=2e-9)


async def test_grid_flat_response_stays_at_the_best_grid_point(fake):
    fake.image = np.full((64, 64), 0.3)                                    # nothing changes with z
    state = await started(protocol=one("z", method="grid", step_m=1e-6, max_offset_m=4e-6))
    update = await tune_node(state)
    assert update["tuning"]["z"]["stop_reason"] == "no_peak"
    assert update["tuning"]["z"]["n_images"] == 9                          # no extra image


@pytest.mark.parametrize("step, message", [
    ({"tune": "A1", "method": "grid"}, "use method 'quadratic' for A1"),
    ({"tune": "z", "method": "scan"}, "method must be"),
    ({"tune": "z", "method": "grid", "step_m": 1e-6, "max_offset_m": 2e-6}, "need 7..51"),
    ({"tune": "C1", "method": "grid", "step_m": 1e-9, "max_offset_m": 40e-9}, "need 7..51"),
    ({"tune": "C1", "method": "grid", "step_m": 5e-9, "max_offset_m": 30e-9, "rounds": 0}, "rounds"),
])
async def test_bad_grid_settings_move_nothing(fake, step, message):
    update = await tune_node(await started(protocol=[{"name": "s", **step}]))
    assert update["run_status"] == "failed" and message in update["errors"][0]
    assert not fake.called("move_stage") and not fake.called("correct_aberration")


# --- tune: failure restores the start value ---

async def test_failure_restores_z(fake):
    fake.enable_blur(z_m=4e-6)
    fake.stage = [0.0, 0.0, -5e-6, 0.0, 0.0]
    state = await started(protocol=one("z"))
    fake.fail_image_after = 5                                             # 6th image fails
    update = await tune_node(state)
    assert update["run_status"] == "failed" and "simulated acquisition failure" in update["errors"][0]
    assert fake.stage[2] == pytest.approx(-5e-6)


@pytest.mark.parametrize("name", ["C1", "A1"])
async def test_failure_restores_corrector(fake, name):
    fake.enable_blur(c1_m=20e-9, a1_m=(2e-9, 1e-9))
    state = await started(protocol=one(name))
    fake.fail_image_after = 6
    update = await tune_node(state)
    assert update["run_status"] == "failed"
    assert fake.corrector[name] == pytest.approx([0.0] * len(fake.corrector[name]), abs=1e-20)


# --- tune: bad settings stop the node before anything moves ---

async def test_typo_in_a_later_step_stops_the_first(fake):
    state = await started(protocol=one("z") + [{"name": "A1", "tune": "A1", "metric": "entropy"}])
    update = await tune_node(state)
    assert update["run_status"] == "failed" and "toward blur" in update["errors"][0]
    assert not fake.called("move_stage") and not fake.called("correct_aberration")


@pytest.mark.parametrize("protocol, message", [
    (None, "no protocol"),
    ([], "no protocol"),
    ([{"tune": "z"}], "needs a name"),
    ([{"name": "b", "tune": "B2"}], "tune must be one of"),
    ([{"name": "z", "tune": "z", "stepsize": 1e-6}], "unknown keys"),
    (one("z") + one("z"), "unique"),
    (one("C1", fov_m=1.0), "FOV"),
])
async def test_bad_protocol_moves_nothing(fake, protocol, message):
    state = await started(protocol=protocol)
    update = await tune_node(state)
    assert update["run_status"] == "failed" and message in update["errors"][0]
    assert not fake.called("move_stage") and not fake.called("set_fov")


# --- tune: the protocol, step by step ---

async def test_protocol_runs_steps_in_order(fake):
    fake.enable_blur(z_m=2e-6, c1_m=-12e-9)
    protocol = [{"name": "z_coarse", "tune": "z", "fov_m": 1e-6, "step_m": 2e-6},
                {"name": "C1_fine", "tune": "C1", "fov_m": 40e-9, "step_m": 4e-9}]
    state = await started(protocol=protocol)
    for expected in ("z_coarse", "C1_fine"):
        update = await tune_node(state)
        assert list(update["tuning"]) == [expected]
        state = {**state, **update, "tuning": {**state.get("tuning", {}), **update["tuning"]}}
    assert await tune_node(state) == {} and next_step(state) is None            # protocol done
    assert [args["fov"] for args in fake.called("set_fov")] == [1e-6, 40e-9]      # FOV per step
    z, c1 = state["tuning"]["z_coarse"], state["tuning"]["C1_fine"]
    assert (z["tune"], z["fov_m"], c1["tune"], c1["fov_m"]) == ("z", 1e-6, "C1", 40e-9)
    assert z["best_offset_m"][0] == pytest.approx(2e-6, abs=0.5e-6)
    assert c1["best_offset_m"][0] == pytest.approx(-12e-9, abs=2e-9)
    assert c1["settings"]["max_offset_m"] == 50e-9                              # default kept
    run_dir = Path(state["run"]["run_dir"])
    assert (run_dir / "tune_z_coarse.json").exists() and (run_dir / "tune_C1_fine.png").exists()


async def test_failed_step_keeps_earlier_steps(fake):
    fake.enable_blur(z_m=2e-6, c1_m=10e-9)
    state = await started(protocol=one("z") + one("C1"))
    update = await tune_node(state)
    state = {**state, **update}
    z_after_step_1 = fake.stage[2]
    fake.fail_image_after = len(fake.called("acquire_image")) + 3              # fails inside C1
    update = await tune_node(state)
    assert update["run_status"] == "failed" and "step 'C1'" in update["errors"][0]
    assert fake.corrector["C1"][0] == pytest.approx(0.0, abs=1e-20)            # C1 restored
    assert fake.stage[2] == pytest.approx(z_after_step_1)                      # z result kept
    assert next_step(state)["name"] == "C1"                                     # C1 still to do


async def test_steps_of_another_run_do_not_count(fake):
    state = await started(protocol=one("z"))
    state["tuning"] = {"z": {"run_dir": "/some/older/run"}}
    assert next_step(state)["name"] == "z"


# --- segment ---

async def test_segment_finds_particles(fake):
    image, disks = particle_image(size=128, n=8, radius_px=(8, 14), seed=2)
    fake.image = blur(image, defocus_px=1.0)
    state = await started()
    state = {**state, **await overview_node(state)}
    update = await segment_node(state)
    assert update["segmentation"]["n_particles"] == len(update["particles"]) == len(disks)
    assert [p["id"] for p in update["particles"]] == list(range(1, len(disks) + 1))
    assert Path(update["segmentation"]["labels_path"]).exists()
    assert Path(update["segmentation"]["overlay_path"]).exists()
    assert "run_status" not in update


async def test_segment_no_particles(fake):
    fake.image = np.full((64, 64), 0.05)
    state = await started()
    state = {**state, **await overview_node(state)}
    update = await segment_node(state)
    assert update["particles"] == [] and update["run_status"] == "no_particles"


async def test_segment_settings_in_nm(fake):
    image, _ = particle_image(size=128, n=8, radius_px=(8, 14), seed=2)
    fake.image = blur(image, defocus_px=1.0)
    state = await started(segmentation={"min_diameter_nm": 20.0})        # bigger than any particle
    state = {**state, **await overview_node(state)}
    update = await segment_node(state)
    assert update["run_status"] == "no_particles"


# --- SAM segment ---

async def with_overview(**extra) -> dict:
    state = await started(**extra)
    return {**state, **await overview_node(state)}


async def test_sam_segment_finds_particles(fake):
    image, disks = particle_image(size=128, n=8, radius_px=(8, 14), seed=2)
    fake.image = blur(image, defocus_px=1.0)
    state = await with_overview()
    images = len(fake.called("acquire_image"))
    update = await sam_segment_node(state)
    assert fake.called("segment") == [{"data_key": state["overview"]["tiled_key"]}]
    assert len(fake.called("acquire_image")) == images                  # no new scan
    seg = update["segmentation"]
    assert seg["method"] == "sam" and seg["labels_key"] == "segmentation-1"
    assert seg["n_particles"] == len(update["particles"]) == len(disks)
    assert [p["id"] for p in update["particles"]] == list(range(1, len(disks) + 1))
    assert Path(seg["labels_path"]).exists() and Path(seg["overlay_path"]).exists()
    assert "run_status" not in update


async def test_sam_segment_cleans_overlaps(fake):
    fake.image = np.zeros((40, 60))
    fake.sam_labels = sam_cluster()
    update = await sam_segment_node(await with_overview())
    assert update["segmentation"]["n_particles"] == 2
    assert np.load(update["segmentation"]["labels_path"]).max() == 2


async def test_sam_segment_no_particles(fake):
    fake.image = np.zeros((32, 32))
    fake.sam_labels = (np.zeros((32, 32), dtype=np.uint32), [])
    update = await sam_segment_node(await with_overview())
    assert update["particles"] == [] and update["run_status"] == "no_particles"


async def test_sam_segment_settings_in_nm(fake):
    fake.image = np.zeros((40, 60))
    fake.sam_labels = sam_cluster()
    update = await sam_segment_node(await with_overview(sam={"min_diameter_nm": 1000.0}))
    assert update["run_status"] == "no_particles"
    fake.sam_labels = sam_cluster()                                  # FOV 20 nm / 60 px: particles ~7 nm
    update = await sam_segment_node(await with_overview(sam={"max_diameter_nm": 5.0}))
    assert update["run_status"] == "no_particles"
    fake.sam_labels = sam_cluster()
    update = await sam_segment_node(await with_overview(sam={"max_diameter_nm": 10.0}))
    assert update["segmentation"]["n_particles"] == 2


async def test_sam_device_error_fails_the_node(fake):
    fake.image = np.zeros((32, 32))
    fake.fail[session.tool_name("segment", "twin")] = "SAM2 model is not initialized"
    update = await sam_segment_node(await with_overview())
    assert update["run_status"] == "failed" and "SAM2 model is not initialized" in update["errors"][0]


async def test_sam_labels_must_match_the_overview(fake):
    fake.image = np.zeros((32, 32))
    fake.sam_labels = (np.zeros((16, 16), dtype=np.uint32), [])
    update = await sam_segment_node(await with_overview())
    assert update["run_status"] == "failed" and "do not match the overview" in update["errors"][0]
