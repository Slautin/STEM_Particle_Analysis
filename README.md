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

## Test

```bash
uv run python scripts/check_mcp.py      # tools + one HAADF on the twin
uv run pytest
```

Block notebooks: `notebooks/00_connection.ipynb`, `01_nodes_instrument.ipynb`, `02_analysis.ipynb`, `03_nodes_analysis.ipynb`, … (kernel: `.venv`, first cell `TARGET = "twin"` or `"real"`).

## Status

- **Block 0 (twin + connection):** done.
- **Block 1 (instrument nodes: preflight, status, move, set FOV, overview, EDS):** done on the twin.
- **Block 2 (analysis: metrics, quadratic search, Otsu segmentation):** done, checked against the notebook's code and on synthetic and twin images.
- **Block 3 (tune z / C1 / A1, segment):** done; finding the optimum tested on a fake microscope, mechanics and restore on the twin.
- The graph (block 4) is still a header; see the plan.
