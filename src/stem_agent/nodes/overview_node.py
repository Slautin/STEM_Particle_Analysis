"""Acquire the HAADF overview image of the area.

Reads:   settings.detector, current.fov_m
Writes:  overview
MCP:     acquire_scanned_image (via common.acquire_haadf)
Saves:   overview.png in the run folder
Note:    on the twin, EDS positions follow the last image taken, so run this right before EDS
Source:  notebook cells 9, 20
"""
from pathlib import Path

from stem_agent.nodes import common
from stem_agent.utils import plots
from stem_agent.utils.safety import safe_node


@safe_node()
async def overview_node(state: dict) -> dict:
    detector = state["settings"].get("detector", "haadf")
    key, image = await common.acquire_haadf(detector)
    fov = state.get("current", {}).get("fov_m")
    preview = Path(state["run"]["run_dir"]) / "overview.png"
    title = f"{detector.upper()} overview" + (f", FOV {fov * 1e9:.0f} nm" if fov else "")
    plots.save_image(image, preview, title)
    print(f"[overview] {key} {image.shape} -> {preview.name}")
    return {"overview": {"tiled_key": key, "fov_m": fov, "shape": list(image.shape),
                         "preview_path": str(preview)}}
