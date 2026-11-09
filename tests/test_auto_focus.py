"""auto_focus_node on the fake microscope: kept, worse -> defocus restored, waits past the MCP
timeout, still busy -> fails, twin without auto_focus, skipped without settings, settings checks."""
from pathlib import Path

import pytest
from conftest import started

from stem_agent.analysis.metrics import contrast
from stem_agent.analysis.synthetic import blur, particle_image
from stem_agent.config import auto_focus_settings
from stem_agent.nodes import auto_focus_node as module
from stem_agent.nodes.auto_focus_node import auto_focus_node, is_timeout

SHARP, _ = particle_image(size=96, n=8, radius_px=(6, 12), seed=0)
BLURRED = blur(SHARP, defocus_px=4.0)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(module, "POLL_S", 0.0)


def test_images_differ_enough():
    assert contrast(BLURRED) < 0.9 * contrast(SHARP)              # "clearly worse" for the tests


async def test_keeps_a_better_focus(fake):
    fake.image, fake.defocus = BLURRED, 0.0
    fake.auto_focus_effect = (-25e-9, SHARP)
    update = await auto_focus_node(await started(auto_focus={"fov_m": 40e-9}))
    result = update["tuning"]["auto_focus"]
    assert result["stop_reason"] == "done" and result["best_offset_m"] == [pytest.approx(-25e-9)]
    assert fake.defocus == -25e-9 and not fake.called("set_defocus")
    assert update["current"]["defocus_m"] == -25e-9
    assert update["current"]["fov_m"] == pytest.approx(40e-9) and fake.fov == pytest.approx(40e-9)
    assert result["n_images"] == 2 and result["best_score"] == pytest.approx(contrast(SHARP))
    run_dir = Path(result["run_dir"])
    assert Path(result["history_path"]).exists() and Path(result["figure_path"]).exists()
    assert (run_dir / "auto_focus_before.png").exists()


async def test_worse_image_restores_defocus(fake):
    fake.image, fake.defocus = SHARP, 10e-9
    fake.auto_focus_effect = (300e-9, BLURRED)
    update = await auto_focus_node(await started(auto_focus={}))
    result = update["tuning"]["auto_focus"]
    assert result["stop_reason"] == "worse_restored" and result["best_offset_m"] == [0.0]
    assert fake.defocus == 10e-9 and fake.called("set_defocus") == [{"defocus": 10e-9}]
    assert update["current"]["defocus_m"] == 10e-9 and "run_status" not in update


async def test_waits_past_the_mcp_timeout(fake):
    fake.image = BLURRED
    fake.auto_focus_effect = (-25e-9, SHARP)
    fake.auto_focus_mode, fake.busy_polls = "timeout", 3
    update = await auto_focus_node(await started(auto_focus={}))
    assert update["tuning"]["auto_focus"]["stop_reason"] == "done" and fake.defocus == -25e-9
    assert len(fake.called("get_defocus")) >= 3 + 2                # busy polls, then free


async def test_still_busy_fails_and_reports_the_restore(fake):
    fake.image, fake.defocus = BLURRED, 5e-9
    fake.auto_focus_mode, fake.busy_polls = "timeout", 10 ** 6
    update = await auto_focus_node(await started(auto_focus={"max_wait_s": 0.05}))
    assert update["run_status"] == "failed"
    assert "still busy" in update["errors"][0] and "RESTORE FAILED" in update["errors"][0]


async def test_twin_without_auto_focus_fails_cleanly(fake):
    fake.image, fake.defocus = BLURRED, 5e-9
    fake.auto_focus_mode = "missing"
    update = await auto_focus_node(await started(auto_focus={}))
    assert update["run_status"] == "failed" and "does not implement" in update["errors"][0]
    assert fake.defocus == 5e-9 and fake.called("set_defocus") == [{"defocus": 5e-9}]


async def test_skipped_without_settings(fake):
    assert await auto_focus_node(await started()) == {}
    assert not fake.called("auto_focus")


@pytest.mark.parametrize("text, expected", [
    ("API_DeviceTimedOut: Timeout (30000 mS) exceeded on device x, command auto_focus", True),
    ("API_CommandTimedOut: Not able to acquire serialization monitor", True),
    ("Timed out while waiting for response", True),
    ("DigitalTwinParticles does not implement _auto_focus", False),
])
def test_is_timeout(text, expected):
    assert is_timeout(RuntimeError(text)) is expected


@pytest.mark.parametrize("bad, message", [
    ({"metric": "entropy"}, "auto_focus metric"),
    ({"max_wait": 5}, "unknown auto_focus settings"),
    ({"worse_tolerance": 1.0}, "bad auto_focus settings"),
    ({"max_wait_s": 0}, "bad auto_focus settings"),
    ({"fov_m": 1.0}, "FOV"),
])
def test_bad_settings(bad, message):
    with pytest.raises(ValueError, match=message):
        auto_focus_settings({"auto_focus": bad})
