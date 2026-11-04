"""Figures saved as PNG to the run folder (no display / clear_output).

Uses matplotlib.figure.Figure directly, so it never changes the notebook backend.

    save_image(image, path, title, points=None)    overview preview, optional marked points
    plot_spectra(spectra, labels, path, elements)  EDS: channel lines, or element bars (twin)
    plot_search(search, title, unit, path, truth)  measured points + fitted curve / surface
                                                   (notebook show_progress_z / show_progress_abb)
    plot_particles(image, labels, particles, title, path)
                                                   outlines + numbered aim points (notebook cell 24)
The plot_* functions return the Figure (shown by Jupyter) and save it when a path is given.
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


def plot_search(search, title: str = "", unit: float = 1e-9, unit_name: str = "nm",
                path: str | Path | None = None, truth=None) -> Figure:
    """Left: score per measurement. Right: score vs offset (1D) or path on the fit (2D)."""
    offsets = np.array([p.offset for p in search.history]) / unit
    scores = np.array([p.score for p in search.history])
    best = search.best()
    fig = Figure(figsize=(11, 4), layout="constrained")
    ax_order, ax_map = fig.subplots(1, 2)
    ax_order.plot(scores, "o-", color="0.5")
    ax_order.set(xlabel="measurement", ylabel="score", title="score per measurement")

    if search.dims == 1:
        if len(scores) >= 3:
            margin = max(np.ptp(offsets) * 0.15, 1e-12)
            grid = np.linspace(offsets.min() - margin, offsets.max() + margin, 400)
            ax_map.plot(grid, search.predict(grid * unit), color="tab:blue", label="quadratic fit")
        ax_map.scatter(offsets[:, 0], scores, c=np.arange(len(scores)), cmap="viridis", zorder=3)
        ax_map.plot(best.offset[0] / unit, best.score, "*", ms=16, color="tab:red", label="best")
        if truth is not None:
            ax_map.axvline(np.atleast_1d(truth)[0] / unit, color="tab:green", ls="--", label="true optimum")
        ax_map.set(xlabel=f"offset ({unit_name})", ylabel="score")
    else:
        if len(scores) >= 6:
            margin = max(np.ptp(offsets), 1e-12) * 0.2
            gx = np.linspace(offsets[:, 0].min() - margin, offsets[:, 0].max() + margin, 100)
            gy = np.linspace(offsets[:, 1].min() - margin, offsets[:, 1].max() + margin, 100)
            xx, yy = np.meshgrid(gx, gy)
            surface = search.predict(np.column_stack([xx.ravel(), yy.ravel()]) * unit).reshape(xx.shape)
            ax_map.contourf(xx, yy, surface, levels=24, cmap="viridis", alpha=0.8)
        ax_map.plot(offsets[:, 0], offsets[:, 1], color="0.6", lw=0.8)
        ax_map.scatter(offsets[:, 0], offsets[:, 1], c=scores, cmap="viridis", edgecolor="white", zorder=3)
        ax_map.plot(*(best.offset / unit), "*", ms=16, color="tab:red", label="best")
        if truth is not None:
            ax_map.plot(*(np.asarray(truth) / unit), "+", ms=16, mew=2, color="tab:green", label="true optimum")
        ax_map.set(xlabel=f"component 1 ({unit_name})", ylabel=f"component 2 ({unit_name})", aspect="equal")
    ax_map.set_title(f"stop: {search.stop_reason}, {len(scores)} measurements")
    ax_map.legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    if path is not None:
        fig.savefig(path, dpi=100)
    return fig


def plot_particles(image: np.ndarray, labels: np.ndarray, particles: list[dict], title: str = "",
                   path: str | Path | None = None) -> Figure:
    fig = Figure(figsize=(6, 6), layout="constrained")
    ax = fig.add_subplot()
    ax.imshow(image, cmap="gray", interpolation="none")
    ax.contour(labels > 0, levels=[0.5], colors="cyan", linewidths=0.8)
    for particle in particles:
        ax.plot(particle["col"], particle["row"], "o", mfc="none", mec="tab:red", ms=8, mew=1.5)
        ax.text(particle["col"] + 3, particle["row"] - 3, str(particle["id"]), color="tab:red",
                fontsize=8, weight="bold")
    ax.set_title(title or f"{len(particles)} particles")
    ax.axis("off")
    if path is not None:
        fig.savefig(path, dpi=100)
    return fig
