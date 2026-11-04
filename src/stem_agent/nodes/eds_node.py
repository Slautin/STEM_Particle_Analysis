"""Acquire one EDS spectrum at the centre of each particle. SIDE EFFECT: parks the beam.

Reads:   particles, overview.shape, settings.max_particles / beam_order, run.target
Writes:  spectra
MCP:     place_beam, acquire_spectrum("eds")
Checks:  beam position within 0..1; at most max_particles points
Saves:   spectra/particle_NNN.npy + eds.png in the run folder
Beam order: pixel (row, col) -> place_beam [x, y], per target (config.TARGETS beam_order)
Source:  notebook cells 22, 24, 29
"""
from pathlib import Path

import numpy as np

from stem_agent.config import LIMITS, TARGETS
from stem_agent.data import tiled_io
from stem_agent.mcp import session
from stem_agent.utils import plots
from stem_agent.utils.safety import safe_node


def beam_position(row: float, col: float, shape: list[int], order: str) -> list[float]:
    """Image pixel -> place_beam [x, y] as fractions of the frame."""
    height, width = shape[:2]
    by_col, by_row = col / width, row / height
    xy = [by_col, by_row] if order == "col_row" else [by_row, by_col]
    if not all(0.0 <= v <= 1.0 for v in xy):
        raise ValueError(f"pixel ({row:.1f}, {col:.1f}) is outside the {height}x{width} frame")
    return xy


@safe_node()
async def eds_node(state: dict) -> dict:
    settings, run = state["settings"], state["run"]
    particles = state.get("particles") or []
    limit = settings.get("max_particles", LIMITS["max_particles"])
    if len(particles) > limit:
        print(f"[eds] {len(particles)} particles, measuring the first {limit}")
        particles = particles[:limit]
    order = settings.get("beam_order") or TARGETS[run["target"]]["beam_order"]
    shape = state["overview"]["shape"]

    out_dir = Path(run["run_dir"]) / "spectra"
    out_dir.mkdir(exist_ok=True)
    records, spectra, elements = [], [], []
    for particle in particles:
        xy = beam_position(particle["row"], particle["col"], shape, order)
        await session.call("place_beam", xy)
        key = await session.call("acquire_spectrum", "eds")
        values, names = tiled_io.read_spectrum(key)
        path = out_dir / f"particle_{particle['id']:03d}.npy"
        np.save(path, values)
        record = {"particle_id": particle["id"], "beam_xy": xy, "tiled_key": key,
                  "npy_path": str(path)}
        if names is not None:
            record["elements"] = names
        records.append(record)
        spectra.append(values)
        elements.append(names)
        print(f"[eds] particle {particle['id']} at beam ({xy[0]:.3f}, {xy[1]:.3f}): "
              f"{values.size} values, max {values.max():.3g}")

    if records:
        plots.plot_spectra(spectra, [f"particle {r['particle_id']}" for r in records],
                           Path(run["run_dir"]) / "eds.png", elements)
    return {"spectra": records}
