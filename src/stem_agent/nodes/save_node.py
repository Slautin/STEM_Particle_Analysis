"""Finish the run: summary.json + summary.png, final status, optional beam blanking.

Reads:   whole state
Writes:  run_status ("ok" unless already "failed" / "no_particles"), finished, summary_path
MCP:     blank_beam, only if settings.blank_beam_at_end
Saves:   summary.json (the whole state; it holds only numbers, keys and paths), summary.png
Runs after every run, also failed ones. Problems with the figure or the beam blanking are
added to errors; they do not stop the summary from being written.
Source:  new
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from stem_agent.config import new_run
from stem_agent.data import tiled_io
from stem_agent.mcp import session
from stem_agent.utils import plots
from stem_agent.utils.safety import safe_node


def _load_figure_data(state: dict):
    """Overview image (from Tiled), labels and spectra (from the run folder), when present."""
    overview, segmentation = state.get("overview"), state.get("segmentation")
    image = tiled_io.read_image(overview["tiled_key"], state["settings"].get("detector", "haadf")) \
        if overview else None
    labels = np.load(segmentation["labels_path"]) if segmentation else None
    spectra = [np.load(r["npy_path"]) for r in state.get("spectra") or []]
    return image, labels, spectra


@safe_node()
async def save_node(state: dict) -> dict:
    errors: list[str] = []
    run = state.get("run") or {}
    run_dir = Path(run["run_dir"]) if run.get("run_dir") else new_run("no_preflight")
    status = state.get("run_status")
    if status not in ("failed", "no_particles"):
        status = "ok"

    if state["settings"].get("blank_beam_at_end"):
        try:
            await session.call("blank_beam")
            print("[save] beam blanked")
        except Exception as exc:
            errors.append(f"save: blank_beam failed: {exc}")

    figure_path = run_dir / "summary.png"
    try:
        image, labels, spectra = _load_figure_data(state)
        plots.plot_summary({**state, "run_status": status}, image, labels, spectra, path=figure_path)
    except Exception as exc:
        errors.append(f"save: summary figure failed: {type(exc).__name__}: {exc}")
        figure_path = None

    finished = datetime.now().isoformat(timespec="seconds")
    summary_path = run_dir / "summary.json"
    summary = {**state, "run_status": status, "errors": (state.get("errors") or []) + errors,
               "finished": finished, "summary_figure": str(figure_path) if figure_path else None}
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"[save] run_status {status} | {len(summary['errors'])} error(s) | {summary_path}")
    return {"run_status": status, "finished": finished, "summary_path": str(summary_path),
            "errors": errors}
