"""Block 3 nodes on the fake microscope: tune (z / C1 / A1) and segment.

The fake's image blurs with the distance of stage z / C1 / A1 from a hidden optimum, so these
tests check that each node finds it through the full path: MCP calls -> image -> metric -> search.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import started

from stem_agent.analysis.synthetic import blur, particle_image
from stem_agent.nodes.overview_node import overview_node
from stem_agent.nodes.preflight_node import preflight_node
from stem_agent.nodes.segment_node import segment_node
from stem_agent.nodes.tune_node import focus_z_node, tune_a1_node, tune_c1_node


# --- tune: finds the optimum, ends there ---

async def test_focus_z_finds_optimum(fake):
    fake.enable_blur(z_m=-112e-6 + 4.3e-6)                               # absolute z
    fake.stage = [0.0, 0.0, -112e-6, 0.0, 0.0]
    state = await started()
    update = await focus_z_node(state)
    result = update["tuning"]["z"]
    assert result["best_offset_m"][0] == pytest.approx(4.3e-6, abs=0.25e-6)
    assert fake.stage[2] == pytest.approx(-112e-6 + result["best_offset_m"][0])   # ended at best
    assert update["current"]["stage"][2] == pytest.approx(fake.stage[2])
    assert Path(result["history_path"]).exists() and Path(result["figure_path"]).exists()
    points = json.loads(Path(result["history_path"]).read_text())["points"]
    assert len(points) == result["n_images"] and set(points[0]["metrics"]) >= {"contrast", "entropy"}


async def test_tune_c1_finds_optimum(fake):
    fake.enable_blur(c1_m=23e-9)
    state = await started()
    update = await tune_c1_node(state)
    best = update["tuning"]["C1"]["best_offset_m"][0]
    assert best == pytest.approx(23e-9, abs=2.5e-9)
    assert fake.corrector["C1"][0] == pytest.approx(best)                 # ended at best
    assert update["current"]["c1_offset_m"] == pytest.approx(best)
    assert fake.called("correct_aberration")[0]["args"] == "C1 -1e-08"    # first step: -step


async def test_tune_a1_finds_optimum(fake):
    fake.enable_blur(a1_m=(3e-9, -2e-9))
    state = await started()
    update = await tune_a1_node(state)
    best = np.array(update["tuning"]["A1"]["best_offset_m"])
    assert np.hypot(*(best - [3e-9, -2e-9])) < 1e-9
    assert fake.corrector["A1"] == pytest.approx(best.tolist())
    assert update["current"]["a1_offset_m"] == pytest.approx(best.tolist())
    assert fake.called("correct_aberration")[0]["args"] == "A1 2e-09 0"   # first circle point


# --- tune: limits ---

async def test_z_stays_within_max_offset(fake):
    fake.enable_blur(z_m=60e-6)                                           # far beyond the limit
    state = await started(tuning={"z": {"max_offset_m": 10e-6, "step_m": 2e-6}})
    update = await focus_z_node(state)
    zs = [args["position"][2] for args in fake.called("move_stage")]
    assert max(abs(z) for z in zs) <= 10e-6 * (1 + 1e-9)
    assert update["tuning"]["z"]["best_offset_m"][0] == pytest.approx(10e-6, abs=1e-6)


async def test_corrector_steps_within_limit(fake):
    fake.enable_blur(c1_m=-45e-9)
    state = await started()
    await tune_c1_node(state)
    steps = [abs(float(v)) for args in fake.called("correct_aberration") for v in args["args"].split()[1:]]
    assert max(steps) <= 100e-9


# --- tune: failure restores the start value ---

async def test_failure_restores_z(fake):
    fake.enable_blur(z_m=4e-6)
    fake.stage = [0.0, 0.0, -5e-6, 0.0, 0.0]
    state = await started()
    fake.fail_image_after = 5                                             # 6th image fails
    update = await focus_z_node(state)
    assert update["run_status"] == "failed" and "simulated acquisition failure" in update["errors"][0]
    assert fake.stage[2] == pytest.approx(-5e-6)


@pytest.mark.parametrize("node, name", [(tune_c1_node, "C1"), (tune_a1_node, "A1")])
async def test_failure_restores_corrector(fake, node, name):
    fake.enable_blur(c1_m=20e-9, a1_m=(2e-9, 1e-9))
    state = await started()
    fake.fail_image_after = 6
    update = await node(state)
    assert update["run_status"] == "failed"
    assert fake.corrector[name] == pytest.approx([0.0] * len(fake.corrector[name]), abs=1e-20)


# --- preflight guards for tuning ---

@pytest.mark.parametrize("tuning, message", [
    ({"z": {"metric": "entropy"}}, "toward blur"),
    ({"A1": {"metric": "high_frequency_power"}}, "toward blur"),
    ({"C1": {"max_offset_m": 80e-9}}, "half the per-move limit"),
    ({"z": {"step_m": 30e-6}}, "step_m <= max_offset_m"),
    ({"z": {"stepsize": 1e-6}}, "unknown tuning settings"),
])
async def test_preflight_tuning_guards(fake, tuning, message):
    update = await preflight_node({"settings": {"target": "twin", "target_xy_m": (0.0, 0.0),
                                                "tuning_fov_m": 40e-9, "tuning": tuning}})
    assert update["run_status"] == "failed" and message in update["errors"][0]


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
