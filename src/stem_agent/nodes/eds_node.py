"""Acquire one EDS spectrum at the centre of each particle. SIDE EFFECT: parks the beam.

Reads:   particles, overview (shape, Tiled key), settings.instrument / max_particles / beam_order / detector
Writes:  spectra
MCP:     place_beam, acquire_spectrum("eds")
Checks:  eds_settings (max_particles 1..LIMITS cap, beam_order); beam position within 0..1
Saves:   spectra/particle_NNN.npy + eds.png in the run folder
         (eds.png: the overview with each beam spot | the spectra, spot and spectrum in one colour)
Beam order: pixel (row, col) -> place_beam [x, y], per instrument (config.INSTRUMENTS beam_order)
Source:  notebook cells 22, 24, 29
"""
from pathlib import Path

import numpy as np

from stem_agent.config import INSTRUMENTS, LIMITS
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


def eds_settings(settings: dict) -> tuple[int, str]:
    """-> (max_particles, beam_order), checked."""
    limit = settings.get("max_particles", LIMITS["max_particles"])
    if not 1 <= limit <= LIMITS["max_particles"]:
        raise ValueError(f"max_particles must be 1..{LIMITS['max_particles']}")
    order = settings.get("beam_order") or INSTRUMENTS[settings["instrument"]]["beam_order"]
    if order not in ("col_row", "row_col"):
        raise ValueError("beam_order must be 'col_row' or 'row_col'")
    return limit, order


@safe_node()
async def eds_node(state: dict) -> dict:
    settings, run = state["settings"], state["run"]
    limit, order = eds_settings(settings)
    particles = state.get("particles") or []
    if len(particles) > limit:
        print(f"[eds] {len(particles)} particles, measuring the first {limit}")
        particles = particles[:limit]
    shape = state["overview"]["shape"]
    image = tiled_io.read_image(state["overview"]["tiled_key"], settings.get("detector", "haadf"))   # before the beam moves

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
        plots.plot_spectra(spectra, [f"#{r['particle_id']}" for r in records],
                           Path(run["run_dir"]) / "eds.png", elements,
                           image=image, points=[(p["row"], p["col"]) for p in particles])
    return {"spectra": records}
