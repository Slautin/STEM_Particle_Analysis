"""Figures saved as PNG to the run folder (no display / clear_output).

Uses matplotlib.figure.Figure directly, so it never changes the notebook backend.

Done:
    save_image(image, path, title, points=None)    overview preview, optional marked points
    plot_spectra(spectra, labels, path, elements)  EDS: channel lines, or element bars (twin)

Planned (block 3):
    plot_search(history, name, metric, path)        notebook show_progress_z / show_progress_abb
    plot_overview(image, labels, particles, path)   notebook cell 24
"""
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def save_image(image: np.ndarray, path: str | Path, title: str = "",
               points: list[tuple[float, float]] | None = None) -> Path:
    """points: (row, col) pixels, numbered in order."""
    fig = Figure(figsize=(6, 6), layout="constrained")
    ax = fig.add_subplot()
    ax.imshow(image, cmap="gray", interpolation="none")
    for i, (row, col) in enumerate(points or [], start=1):
        ax.plot(col, row, "o", mfc="none", mec="tab:red", ms=12, mew=2)
        ax.text(col + 8, row - 8, str(i), color="tab:red", weight="bold")
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, dpi=100)
    return Path(path)


def plot_spectra(spectra: list[np.ndarray], labels: list[str], path: str | Path,
                 elements: list[list[str] | None] | None = None) -> Path:
    """Channel spectra stacked as lines; element spectra (twin) as grouped bars."""
    fig = Figure(figsize=(9, 4), layout="constrained")
    ax = fig.add_subplot()
    if elements and all(e is not None for e in elements):
        names = sorted({name for names in elements for name in names})
        width = 0.8 / max(len(spectra), 1)
        for i, (values, names_i, label) in enumerate(zip(spectra, elements, labels)):
            lookup = dict(zip(names_i, values))
            ax.bar(np.arange(len(names)) + i * width, [lookup.get(n, 0.0) for n in names],
                   width=width, label=label)
        ax.set_xticks(np.arange(len(names)) + 0.4 - width / 2, names)
        ax.set(ylabel="fraction", title="EDS by element")
    else:
        offset = max((float(np.max(s)) for s in spectra), default=1.0) * 0.6
        for i, (values, label) in enumerate(zip(spectra, labels)):
            ax.plot(values + i * offset, lw=1, label=label)
        ax.set(xlabel="channel", ylabel="counts (offset)", title="EDS spectra")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=100)
    return Path(path)
