# STEM Particle Analysis Agent: Overview and Plan

Source notebook: `asyncroscopy/notebooks/focus_stage_navigation_and_particle_spectroscopy.ipynb` (commit `d11fe3d`).
Template: `Slautin/SPM_agent_simple`.
Instrument access: the asyncroscopy MCP server. This project only connects to it as a client.

## 1. Verdict

**Yes, we can build it.** The science is already in the notebook: focus and aberration searches, segmentation, EDS at particles. It has run on the microscope. The work is to reorganize it into LangGraph nodes and add the parts around it: the MCP layer, state, saving and safety.

Effort: **1–2 weeks** for v1, plus **1–2 microscope sessions**.

## 2. Workflow

```
preflight → status_start → move (x,y) → focus z → set FOV → tune C1 → tune A1
          → overview → find particles → EDS per particle → status_end → save

no particles            → status_end → save
any failure (after undo) → status_end → save
```

## 3. Layers

```
node → mcp/ops.py → mcp/session.py → MCP server → Tango → microscope
```

- Full images and spectra are read from **Tiled over HTTP**, because MCP returns only a preview.
- Metrics, search math and segmentation run **locally**, with no microscope.

## 4. Project structure

```
STEM_Particle_Analysis/
├── pyproject.toml   uv.lock   .python-version   .env.example   README.md   docs/PLAN.md
├── src/stem_agent/
│   ├── config.py              # MCP URL, tool-name map, limits, step sizes, run folders
│   ├── mcp/
│   │   ├── client.py          # MultiServerMCPClient wrapper (from SPM)
│   │   ├── session.py         # call(short_name, args), InstrumentError
│   │   └── ops.py             # get_stage, move_stage, set_fov, acquire_haadf, correct,
│   │                          # place_beam, acquire_eds, get_status (real + twin → one format)
│   ├── data/tiled_io.py       # read_image, read_spectrum (real 4096-ch + twin per-element)
│   ├── analysis/
│   │   ├── metrics.py         # contrast, edge_sharpness, entropy, high_frequency_power
│   │   ├── optimizer.py       # QuadraticSearch: 1D/2D, ask/tell, hard bounds
│   │   └── segmentation.py    # otsu (default), sam (optional)
│   ├── states/experiment_state.py
│   ├── nodes/
│   │   ├── preflight_node.py
│   │   ├── status_node.py     # make_status_node(check_ready=True/False)
│   │   ├── move_stage_node.py
│   │   ├── tune_node.py       # make_tune_node("z" | "C1" | "A1", dims, apply)
│   │   ├── set_fov_node.py
│   │   ├── overview_node.py
│   │   ├── segment_node.py
│   │   ├── eds_node.py
│   │   └── save_node.py
│   ├── graphs/particle_graph.py   # nodes + router (next / save on failure / skip EDS)
│   └── utils/
│       ├── safety.py          # @safe_node(undo=...)
│       └── plots.py           # save PNGs to run folder
├── twin/digital_twin_particles.yaml   # particle twin + corrector, for scripts/twin.py
├── scripts/
│   ├── twin.py                # start / status / stop the twin + MCP in the background
│   └── check_mcp.py           # block 0: list MCP tools, one test acquisition
├── notebooks/run_particle_analysis.ipynb
└── tests/  fake_mcp.py  test_optimizer.py  test_segmentation.py  test_graph_fake.py
```

## 5. Nodes

Every node has the same shape: `async def x_node(state) -> dict`. It reads its inputs, acts through `ops`, reads data from Tiled, computes locally, saves files, and returns only what it changed.

| Node | MCP calls | Writes | Undo on error | From notebook | On twin |
|---|---|---|---|---|---|
| preflight | list tools, `DATA_get_config` | `run` | – | cell 1 (partly) | ✅ |
| status_start | `get_parameters`, `get_stage`, `get_fov`, `get_defocus`, `<Device>_State` | `start`, `current` | – | new | ✅ |
| move | `move_stage`, `get_stage` | `current` | back to start x, y | new | ✅ |
| focus z | `move_stage` + `acquire_scanned_image` (loop) | `tuning["z"]`, `current` | back to start z | cell 5 | ⚠️ mechanics only |
| set FOV | `set_fov`, `get_fov` | `current` | old FOV | cell 12 | ✅ |
| tune C1 / A1 | `correct_aberration` + `acquire_scanned_image` (loop) | `tuning["C1"/"A1"]`, `current` | undo total offset | cell 7 (add limit + undo) | ⚠️ mechanics only, needs a corrector in the twin yaml |
| overview | `acquire_scanned_image` | `overview` | – | cells 9, 20 | ✅ |
| find particles | none (local) | `segmentation`, `particles` | – | cell 21 (Otsu), 28 (SAM) | ✅ |
| EDS | `place_beam`, `acquire_spectrum` | `spectra` | – | cells 22, 29 | ✅ (twin gives per-element values) |
| status_end | same as status_start | `end` | – | new | ✅ |
| save | `blank_beam` (optional) | `run_status`, `summary.json` | – | new | ✅ |

The status_start readiness check stops the run before anything moves unless: optical mode is STEM, column valves are open, high tension is on, the beam stopper is out, the beam is not blanked, and all devices are ON.

## 6. State

```python
class ParticleRunState(TypedDict):
    settings: RunSettings                         # input, never changed
    run:      NotRequired[RunInfo]                # run_dir, tiled_uri, instrument_class, tool map
    start:    NotRequired[InstrumentSnapshot]     # status_start, never changed
    current:  NotRequired[InstrumentSnapshot]     # updated after each move/tune
    end:      NotRequired[InstrumentSnapshot]     # status_end
    tuning:   Annotated[dict[str, TuneResult], operator.or_]   # "z", "C1", "A1"
    overview: NotRequired[Overview]               # tiled_key, fov_m, shape
    segmentation: NotRequired[Segmentation]       # method, mask_path, overlay_path, n
    particles: NotRequired[list[Particle]]        # id, row, col, area_px, beam_xy
    spectra:   NotRequired[list[SpectrumRecord]]  # particle_id, tiled_key, npy_path
    run_status: NotRequired[Literal["running", "ok", "no_particles", "failed"]]
    error:     NotRequired[str]
```

`InstrumentSnapshot` holds: stage `[x, y, z, α, β]`, `fov_m`, `defocus_m`, the C1/A1 offsets we applied, device states, `ready`, `warnings`, and `raw_path` (the full parameters saved as JSON).

**Rules**

1. No images in the state, only Tiled keys and file paths.
2. Nodes return only what they change.
3. `start` is never changed.
4. On failure, undo, set `run_status="failed"`, then go to status_end → save.

## 7. Gaps, risks, safety

**Gaps (none blocks the project)**

1. **Scan and EDS settings are not reachable.** `imsize`, `dwell_time`, `scan_region` and EDS exposure are Tango attributes, and MCP publishes only commands. *Fix:* a small asyncroscopy PR that adds `set_settings` commands. Until then, the operator sets these by hand.
2. **MCP returns only previews.** *Fix:* read full data from Tiled over HTTP.
3. **MCP has a 30 s timeout per command.** SAM on the server would time out. *Fix:* run segmentation locally.

**Check on the microscope:** beam x/y direction for `place_beam`, that the focus metric responds, corrector sign, a `NaN` alpha through the JSON round trip, and real timings.

**Twin limits:** z, C1 and A1 changes do not change the image. *Fix:* test the searches with a synthetic fake microscope, or patch the twin (open decision).

**Safety rules**

- Hard limits on z, C1 and A1.
- Undo on error.
- status_start blocks the run if the microscope is not ready.
- No automatic retries on commands that change the microscope. `correct_aberration` is relative, so a retry would apply the step twice.
- Only read-only calls are retried, and only on network errors.

## 8. Plan

The work goes in blocks. Each block is tested on the digital twin with its own notebook, then the same notebooks run on the real microscope.

For each block:

1. Write the code in `src/stem_agent/`. Notebooks hold no logic, only calls and plots.
2. Run the test notebook `notebooks/NN_<block>.ipynb` on the twin. Its first cell is `TARGET = "twin"`.
3. Add pytest tests for the pure parts.
4. Commit and review.

| # | Block | What we build | Test notebook (twin) | Done when |
|---|---|---|---|---|
| 0 | Twin + connection | `twin/` config, `scripts/twin.py`, config, MCP client/session, Tiled image read | `00_connection.ipynb`: list tools, `get_stage`, one HAADF | the image comes back from the twin |
| 1 | Microscope actions | `ops.py`: stage, FOV, HAADF, corrector, beam, EDS, status | `01_ops.ipynb`: call each action, check readback | every action works |
| 2 | Analysis | metrics, `QuadraticSearch`, segmentation | `02_analysis.ipynb`: search on synthetic blur, Otsu on a twin image | the search finds the known best point; particles found |
| 3 | Nodes | all nodes + `safe_node` | `03_nodes.ipynb`: hand-made state, each node alone, one forced error | each node runs alone; undo works |
| 4 | Graph | state, router, save | `04_graph_twin.ipynb`: normal, no-particles and failure runs | `summary.json` + spectra in `runs/` |
| 5 | Real microscope | checks only | notebooks 00 → 04 with `TARGET = "real"` | one full run on a real sample |
| later | LLM nodes, multi-position loop, B2/A2 | – | – | – |

**Digital twin:** run `uv run python scripts/twin.py start | status | stop`. It runs from `ASYNCROSCOPY_DIR` (an asyncroscopy `main` checkout) using that repo's own uv environment, with `twin/digital_twin_particles.yaml` (the particle twin plus a corrector). Logs go to `runs/twin_logs/`.

In parallel: the asyncroscopy PR for scan and EDS settings commands.

**Open decisions**

1. Test the focus searches with synthetic tests only, or also patch the twin?
2. Who opens the asyncroscopy PR for scan settings?
