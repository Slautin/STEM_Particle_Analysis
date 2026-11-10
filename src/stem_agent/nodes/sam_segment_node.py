"""Find particles in the overview with the SAM2 segmentation device.

Reads:   overview (Tiled key, shape, FOV; the image too: min_bright and the overlay),
         settings.sam (config.sam_settings), detector
Writes:  segmentation, particles (largest first, beam aim points inside each particle);
         run_status="no_particles" if none (the graph then skips EDS)
MCP:     SEGMENTATION segment: overview key -> label-image key. No new scan, so the particles
         belong to the image EDS aims by.
Saves:   segmentation_labels.npy (cleaned labels) + segmentation.png in the run folder
Note:    keeps every particle; the EDS node applies max_particles
"""
from pathlib import Path

import numpy as np

from stem_agent.analysis.segmentation import pixel_size_m, sam_particles
from stem_agent.config import sam_settings
from stem_agent.data import tiled_io
from stem_agent.mcp import session
from stem_agent.utils import plots
from stem_agent.utils.safety import safe_node


@safe_node()
async def sam_segment_node(state: dict) -> dict:
    cfg = sam_settings(state["settings"])
    overview = state["overview"]
    fov = overview.get("fov_m") or (state.get("current") or {}).get("fov_m")
    if not fov:
        raise ValueError("overview has no FOV; sam diameters are in nm")
    labels_key = await session.call("segment", overview["tiled_key"])
    raw, areas = tiled_io.read_labels(labels_key)
    if list(raw.shape) != list(overview["shape"]):
        raise ValueError(f"SAM labels {raw.shape} do not match the overview {overview['shape']}")
    image = tiled_io.read_image(overview["tiled_key"], state["settings"].get("detector", "haadf"))
    labels, particles = sam_particles(raw, areas, pixel_size_m(fov, raw.shape), **cfg,  # cfg keys = its arguments
                                      image=image)

    run_dir = Path(state["run"]["run_dir"])
    labels_path = run_dir / "segmentation_labels.npy"
    np.save(labels_path, labels)
    overlay_path = run_dir / "segmentation.png"
    plots.plot_particles(image, labels, particles,
                         f"SAM2: {len(particles)} particles, FOV {fov * 1e9:.0f} nm", path=overlay_path)
    print(f"[sam] {len(particles)} particles from {int(raw.max())} SAM masks, FOV {fov * 1e9:.0f} nm")

    update = {"segmentation": {"method": "sam", "settings": cfg, "labels_key": labels_key,
                               "n_particles": len(particles), "labels_path": str(labels_path),
                               "overlay_path": str(overlay_path)},
              "particles": particles}
    if not particles:
        update["run_status"] = "no_particles"
    return update
