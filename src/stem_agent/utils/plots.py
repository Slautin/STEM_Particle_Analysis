"""Figures saved as PNG to the run folder (no display / clear_output).

Uses matplotlib.figure.Figure directly, so it never changes the notebook backend.

    save_image(image, path, title, points=None)    overview preview, optional marked points
    plot_spectra(spectra, labels, path, elements, image, points)
                                                   EDS: channel lines, or element bars (twin);
                                                   optional image with the beam spots beside them
    plot_search(search, title, unit, path, truth)  measured points + fitted curve / surface
                                                   (notebook show_progress_z / show_progress_abb)
    plot_particles(image, labels, particles, title, path)
                                                   outlines + numbered aim points (notebook cell 24)
    plot_summary(state, image, labels, spectra, path)  one page per run (save node)
    plot_graph(graph, path)                        the LangGraph graph, drawn locally
The plot_* functions return the Figure (shown by Jupyter) and save it when a path is given.
"""
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from skimage.segmentation import find_boundaries


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


def _outline(ax, labels: np.ndarray) -> None:
    """One cyan outline per particle (touching particles stay apart)."""
    edges = find_boundaries(labels, mode="inner")
    ax.imshow(np.ma.masked_where(~edges, edges), cmap=ListedColormap(["cyan"]), interpolation="none")


def _draw_spectra(ax, spectra: list[np.ndarray], labels: list[str],
                  elements: list[list[str] | None] | None) -> None:
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
    if len(labels) <= 12:
        ax.legend(frameon=False, fontsize=7)


def plot_spectra(spectra: list[np.ndarray], labels: list[str], path: str | Path,
                 elements: list[list[str] | None] | None = None, image: np.ndarray | None = None,
                 points: list[tuple[float, float]] | None = None) -> Path:
    """Channel spectra stacked as lines; element spectra (twin) as grouped bars.
    With image + points ((row, col) of each spectrum): the image with the beam spots on the left,
    each spot in the colour of its spectrum and named by its label."""
    if image is None:
        fig = Figure(figsize=(9, 4), layout="constrained")
        ax_eds = fig.add_subplot()
    else:
        fig = Figure(figsize=(14, 5.5), layout="constrained")
        ax_img, ax_eds = fig.subplots(1, 2, width_ratios=[1, 1.6])
        ax_img.imshow(image, cmap="gray", interpolation="none")
        for i, ((row, col), label) in enumerate(zip(points or [], labels)):
            color = f"C{i % 10}"                     # the default line / bar colour cycle
            ax_img.plot(col, row, "o", mfc="none", mec=color, ms=10, mew=2)
            ax_img.text(col + 4, row - 4, label, color=color, weight="bold")
        ax_img.set_title("beam positions")
        ax_img.axis("off")
    _draw_spectra(ax_eds, spectra, labels, elements)
    fig.savefig(path, dpi=100)
    return Path(path)


def plot_summary(state: dict, image: np.ndarray | None, labels: np.ndarray | None,
                 spectra: list[np.ndarray], path: str | Path | None = None) -> Figure:
    """One page per run: overview + particles | what happened | spectra."""
    fig = Figure(figsize=(16, 5.5), layout="constrained")
    ax_img, ax_text, ax_eds = fig.subplots(1, 3, width_ratios=[1, 1.1, 1.3])
    particles = state.get("particles") or []
    if image is not None:
        ax_img.imshow(image, cmap="gray", interpolation="none")
        if labels is not None and labels.shape == image.shape:
            _outline(ax_img, labels)
        for p in particles:
            ax_img.plot(p["col"], p["row"], "o", mfc="none", mec="tab:red", ms=6, mew=1.2)
            ax_img.text(p["col"] + 2, p["row"] - 2, str(p["id"]), color="tab:red", fontsize=7)
        fov = (state.get("overview") or {}).get("fov_m")
        ax_img.set_title(f"overview{f', FOV {fov * 1e9:.0f} nm' if fov else ''}, {len(particles)} particles")
    else:
        ax_img.text(0.5, 0.5, "no overview", ha="center", va="center", color="0.5")
    ax_img.axis("off")

    run = state.get("run") or {}
    lines = [f"status     {state.get('run_status')}",
             f"instrument {run.get('instrument')}",
             f"run        {Path(run['run_dir']).name if run.get('run_dir') else '-'}", ""]
    for name, result in (state.get("tuning") or {}).items():
        unit, unit_name = (1e-6, "um") if result.get("tune", name) == "z" else (1e-9, "nm")
        best = ", ".join(f"{v / unit:+.3f}" for v in result["best_offset_m"])
        lines.append(f"tune {name:<3} best [{best}] {unit_name}  ({result['n_images']} images, "
                     f"{result['stop_reason']})")
    lines += ["", f"particles {len(particles)}, spectra {len(state.get('spectra') or [])}"]
    errors = state.get("errors") or []
    if errors:
        lines += ["", "errors:"] + [f"  {e[:90]}" for e in errors[:4]]
    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)
    ax_text.axis("off")

    records = state.get("spectra") or []
    if spectra:
        _draw_spectra(ax_eds, spectra, [f"particle {r['particle_id']}" for r in records],
                      [r.get("elements") for r in records])
    else:
        ax_eds.text(0.5, 0.5, "no spectra", ha="center", va="center", color="0.5")
        ax_eds.axis("off")
    fig.suptitle(f"particle analysis: {state.get('run_status')}")
    if path is not None:
        fig.savefig(path, dpi=100)
    return fig


def plot_graph(graph, path: str | Path | None = None) -> Figure:
    """A compiled LangGraph graph, drawn locally: main path down the middle (solid), jumps
    taken by a router (dashed, on the right). No web service involved."""
    drawable = graph.get_graph()
    names = list(drawable.nodes)
    edges = [(e.source, e.target) for e in drawable.edges]
    loops = {s for s, t in edges if s == t}             # a node that repeats (tune)
    edges = [(s, t) for s, t in edges if s != t]
    incoming = {n: sum(1 for _, t in edges if t == n) for n in names}
    order, ready = [], [n for n in names if incoming[n] == 0]
    while ready:                                        # topological order, insertion order on ties
        node = ready.pop(0)
        order.append(node)
        for source, target in edges:
            if source == node:
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
    row = {n: i for i, n in enumerate(order)}

    fig = Figure(figsize=(5, 0.5 * len(order) + 1), layout="constrained")
    ax = fig.add_subplot()
    for source, target in edges:
        a, b = row[source], row[target]
        if b == a + 1:
            ax.annotate("", xy=(0, -b + 0.22), xytext=(0, -a - 0.22),
                        arrowprops={"arrowstyle": "-|>", "color": "0.3"})
        else:
            ax.annotate("", xy=(0.34, -b), xytext=(0.34, -a),
                        arrowprops={"arrowstyle": "-|>", "color": "tab:red", "ls": "--", "lw": 0.8,
                                    "shrinkA": 0, "shrinkB": 2,
                                    "connectionstyle": f"arc3,rad={-0.12 - 0.015 * (b - a)}"})
    for node in loops:
        ax.annotate("", xy=(-0.42, -row[node] + 0.12), xytext=(-0.42, -row[node] - 0.12),
                    arrowprops={"arrowstyle": "-|>", "color": "tab:blue", "lw": 0.9,
                                "connectionstyle": "arc3,rad=-1.6"})
        ax.text(-0.6, -row[node], "repeat", color="tab:blue", fontsize=7, ha="right", va="center")
    for node in order:
        label = {"__start__": "START", "__end__": "END"}.get(node, node)
        terminal = node.startswith("__")
        ax.text(0, -row[node], label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.35", "fc": "white" if terminal else "#eef2ff",
                      "ec": "0.4"})
    ax.text(0.3, 0.4, "- - -  failed / no particles", color="tab:red", fontsize=8, va="center")
    ax.set(xlim=(-0.8, 1.4), ylim=(-len(order) + 0.4, 0.8))
    ax.axis("off")
    if path is not None:
        fig.savefig(path, dpi=100)
    return fig


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
            ax_map.plot(grid, search.predict(grid * unit), color="tab:blue",
                        label=getattr(search, "fit_name", "quadratic fit"))
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
    _outline(ax, labels)
    for particle in particles:
        ax.plot(particle["col"], particle["row"], "o", mfc="none", mec="tab:red", ms=8, mew=1.5)
        ax.text(particle["col"] + 3, particle["row"] - 3, str(particle["id"]), color="tab:red",
                fontsize=8, weight="bold")
    ax.set_title(title or f"{len(particles)} particles")
    ax.axis("off")
    if path is not None:
        fig.savefig(path, dpi=100)
    return fig
