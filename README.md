# STEM Particle Analysis Agent

LangGraph agent that runs particle analysis on a STEM through the asyncroscopy MCP server:
move to (x, y) → focus (stage z) → tune C1 / A1 → HAADF overview → find particles → EDS on each particle.

Design and development plan: [docs/PLAN.md](docs/PLAN.md).

## Setup

```bash
uv sync
cp .env.example .env    # set STEM_MCP_URL
```

Optional SAM segmentation:

```bash
uv sync --extra sam
```

## Run

```bash
uv run python scripts/check_mcp.py   # phase 0: list MCP tools
uv run pytest
```

Entry point: `notebooks/run_particle_analysis.ipynb` (kernel: `.venv`).

## Status

Skeleton only. Files contain headers; code comes in phases 0–3 of the plan.
