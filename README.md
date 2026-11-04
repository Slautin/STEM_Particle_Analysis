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

## Test

```bash
uv run python scripts/check_mcp.py      # tools + one HAADF on the twin
uv run pytest
```

Block notebooks: `notebooks/00_connection.ipynb`, … (kernel: `.venv`, first cell `TARGET = "twin"` or `"real"`).

## Status

Block 0 (twin + connection) done. Other modules are headers; see the plan.
