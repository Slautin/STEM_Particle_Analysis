# STEM Particle Analysis Agent

LangGraph agent that runs particle analysis on a STEM through the asyncroscopy MCP server:
move to (x, y) → focus (stage z) → tune C1 / A1 → HAADF overview → find particles → EDS on each particle.

Design and development plan: [docs/PLAN.md](docs/PLAN.md).

## Setup

```bash
uv sync
cp .env.example .env    # set ASYNCROSCOPY_DIR (and STEM_REAL_MCP_URL for the microscope)
```

Optional SAM segmentation:

```bash
uv sync --extra sam
```

The digital twin needs an asyncroscopy `main` checkout with its own environment (`uv sync` inside it).

## Digital twin

```bash
uv run python scripts/twin.py start     # device stack + Tiled + MCP in the background
uv run python scripts/twin.py status
uv run python scripts/twin.py stop
```

Logs: `runs/twin_logs/`. MCP: `http://127.0.0.1:8000/mcp`.

The twin's images are 128 px (set by `twin.py` so its EDS lands on particles). Other twin quirks are listed in [docs/PLAN.md](docs/PLAN.md).

## Run an analysis

Open `notebooks/run_particle_analysis.ipynb`, edit the settings cell, run all. Or from Python:

```python
from stem_agent.graphs.particle_graph import run
state = await run({"target": "twin", "target_xy_m": (5e-9, 0.0), "tuning_fov_m": 40e-9})
```

Results: `runs/<time>_<tag>/` (`summary.json`, `summary.png`, tuning plots, spectra).

## On the real microscope

`notebooks/05_real_microscope.ipynb` goes level by level: MCP and data → single nodes → instrument checks (metric response, corrector sign, beam x/y order) → tuning → segmentation and EDS → your own graph → full run. Safety switches (`ALLOW_STAGE`, `ALLOW_CORRECTOR`, `ALLOW_BEAM`) are off by default, so nothing moves until you switch it on. Dry-run it first with `TARGET = "twin"`.

## Test

```bash
uv run python scripts/check_mcp.py      # tools + one HAADF on the twin
uv run pytest
```

Block notebooks: `notebooks/00_connection.ipynb`, `01_nodes_instrument.ipynb`, `02_analysis.ipynb`, `03_nodes_analysis.ipynb`, `04_graph_twin.ipynb`, `05_real_microscope.ipynb` (kernel: `.venv`, first cell `TARGET = "twin"` or `"real"`).

## Status

- **Block 0 (twin + connection):** done.
- **Block 1 (instrument nodes: preflight, status, move, set FOV, overview, EDS):** done on the twin.
- **Block 2 (analysis: metrics, quadratic search, Otsu segmentation):** done, checked against the notebook's code and on synthetic and twin images.
- **Block 3 (tune z / C1 / A1, segment):** done; finding the optimum tested on a fake microscope, mechanics and restore on the twin.
- **Block 4 (graph, save, entry point):** done; normal, no-particles and failure runs checked on the twin.
- **Block 5 (real microscope):** `notebooks/05_real_microscope.ipynb` is ready (dry-run checked on the twin); to be run on the instrument.
