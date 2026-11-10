"""Acquire a HAADF image of the area. Use it as often as needed (e.g. before and after focusing).

Reads:   settings.detector, images
Writes:  overview = this image (what segment / EDS use); images = every image of the run, in order
         ([0] first, [-1] latest = overview); each with index, Tiled key, FOV, shape, metrics, png
MCP:     get_fov (the live value, so the FOV always matches the image), acquire_scanned_image
Saves:   overview_NN.png in the run folder (NN = index)
Note:    on the twin, EDS positions follow the last image taken, so run this right before EDS
Source:  notebook cells 9, 20
"""
from pathlib import Path

from stem_agent.analysis.metrics import all_metrics
from stem_agent.mcp import session
from stem_agent.nodes import common
from stem_agent.utils import plots
from stem_agent.utils.safety import safe_node


@safe_node()
async def overview_node(state: dict) -> dict:
    detector = state["settings"].get("detector", "haadf")
    images = state.get("images") or []
    index = len(images)
    fov = float(await session.call("get_fov"))     # segmentation in nm depends on it
    key, image = await common.acquire_haadf(detector)
    metrics = all_metrics(image)
    preview = Path(state["run"]["run_dir"]) / f"overview_{index:02d}.png"
    title = (f"{detector.upper()} image {index}" + (f", FOV {fov * 1e9:.0f} nm" if fov else "")
             + f", contrast {metrics['contrast']:.4g}")
    plots.save_image(image, preview, title)
    print(f"[overview {index:02d}] {key} {image.shape}, contrast {metrics['contrast']:.4g} -> {preview.name}")
    record = {"index": index, "tiled_key": key, "fov_m": fov, "shape": list(image.shape),
              "metrics": metrics, "preview_path": str(preview)}
    return {"overview": record, "images": images + [record]}
