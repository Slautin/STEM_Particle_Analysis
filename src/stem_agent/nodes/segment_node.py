"""Find particles in the overview. Local, no MCP.

Reads:   overview (Tiled key, FOV), settings.segmentation (config.segmentation_settings), detector
Writes:  segmentation, particles (largest first, beam aim points inside each particle);
         run_status="no_particles" if none (the graph then skips EDS)
Saves:   segmentation_labels.npy + segmentation.png in the run folder
Note:    keeps every particle; the EDS node applies max_particles
Source:  notebook cell 21 (Otsu); SAM (cell 28) planned as the optional "sam" extra
"""
from pathlib import Path

import numpy as np

from stem_agent.analysis.segmentation import segment
from stem_agent.config import segmentation_settings
from stem_agent.data import tiled_io
from stem_agent.utils import plots
from stem_agent.utils.safety import safe_node


@safe_node()
async def segment_node(state: dict) -> dict:
    cfg = segmentation_settings(state["settings"])
    overview = state["overview"]
    fov = overview.get("fov_m") or state.get("current", {}).get("fov_m")
    if not fov:
        raise ValueError("overview has no FOV; segmentation settings are in nm")
    image = tiled_io.read_image(overview["tiled_key"], state["settings"].get("detector", "haadf"))
    labels, particles = segment(image, fov, cfg["smooth_nm"], cfg["min_diameter_nm"], cfg["edge_px"])

    run_dir = Path(state["run"]["run_dir"])
    labels_path = run_dir / "segmentation_labels.npy"
    np.save(labels_path, labels)
    overlay_path = run_dir / "segmentation.png"
    plots.plot_particles(image, labels, particles,
                         f"{len(particles)} particles ({cfg['method']}), FOV {fov * 1e9:.0f} nm",
                         path=overlay_path)
    print(f"[segment] {len(particles)} particles ({int(labels.max())} objects incl. edge ones), "
          f"FOV {fov * 1e9:.0f} nm, image {image.shape}")

    update = {"segmentation": {"method": cfg["method"], "settings": cfg,
                               "n_particles": len(particles), "labels_path": str(labels_path),
                               "overlay_path": str(overlay_path)},
              "particles": particles}
    if not particles:
        update["run_status"] = "no_particles"
    return update
