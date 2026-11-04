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
node → mcp/session.py → MCP server → Tango → microscope
```

- Nodes call MCP tools directly through `session.call(short_name, value)`. Only the actions used by several nodes live in `nodes/common.py`: `get_stage`, `move_stage` (keeps tilt, step limit, readback check) and `acquire_haadf`.

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
│   │   └── session.py         # call(short_name, value), InstrumentError
│   ├── data/tiled_io.py       # read_image, read_spectrum (real 4096-ch + twin per-element)
│   ├── analysis/
│   │   ├── metrics.py         # contrast, edge_sharpness, entropy, high_frequency_power
│   │   ├── optimizer.py       # QuadraticSearch: 1D/2D, ask/tell, hard bounds
│   │   └── segmentation.py    # otsu (default), sam (optional)
│   ├── states/experiment_state.py
│   ├── nodes/
│   │   ├── common.py          # get_stage, move_stage, acquire_haadf (shared by several nodes)
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
├── notebooks/  00_connection, 01_nodes_instrument, … (one per block), run_particle_analysis
└── tests/  fake_mcp.py  test_session.py  test_nodes_instrument.py  test_optimizer.py  …
```

## 5. Nodes

Every node has the same shape: `async def x_node(state) -> dict`. It reads its inputs, calls MCP (directly, or through `nodes/common.py`), reads data from Tiled, computes locally, saves files, and returns only what it changed. Nodes that change the microscope are wrapped in `@safe_node(undo=...)`.

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
| 1 | Instrument nodes | preflight, status, move, set_fov, overview, eds, `nodes/common.py`, `safe_node` | `01_nodes_instrument.ipynb`: hand-made state, each node alone, beam-order test, one forced error | readbacks match; EDS hits particles; undo works |
| 2 | Analysis | metrics, `QuadraticSearch`, segmentation | `02_analysis.ipynb`: search on synthetic blur, Otsu on a twin image | the search finds the known best point; particles found |
| 3 | Analysis nodes | tune (z / C1 / A1), segment | `03_nodes_analysis.ipynb`: each node alone on the twin | searches run within limits; particles found |
| 4 | Graph | state, router, save | `04_graph_twin.ipynb`: normal, no-particles and failure runs | `summary.json` + spectra in `runs/` |
| 5 | Real microscope | checks only | notebooks 00 → 04 with `TARGET = "real"` | one full run on a real sample |
| later | LLM nodes, multi-position loop, B2/A2 | – | – | – |

**Digital twin:** run `uv run python scripts/twin.py start | status | stop`. It runs from `ASYNCROSCOPY_DIR` (an asyncroscopy `main` checkout) using that repo's own uv environment, with `twin/digital_twin_particles.yaml` (the particle twin plus a corrector). Logs go to `runs/twin_logs/`.

Twin quirks found in block 1 (asyncroscopy bugs, worked around here):

- The twin keeps particle positions on a 128 px map but checks the EDS beam in image pixels, so EDS only works when the image is 128 px too. `twin.py start` sets `scan.imsize = 128`.
- At 512 px, images fail at some FOVs (50, 100 and 200 nm) because of a rounding mismatch in the twin's image size. At the 128 px that `twin.py` sets, every FOV tried works (20–186 nm).
- The twin's beam order is `row_col`: beam x follows image rows. The real microscope is assumed to be `col_row`, which is still to be checked.
- Device `State` comes back over MCP as a number in text (`'0'` = ON).
- The twin's corrector **subtracts** what it is sent: after `correct_aberration` totals of +x, its internal C1/A1 change by −x (measured in notebook 03). The real CEOS sign is still to be checked.
- The twin keeps its state (stage, FOV, corrector values) between runs until it is restarted.

In parallel: the asyncroscopy PR for scan and EDS settings commands.

Findings from block 2 (`notebooks/02_analysis.ipynb`, synthetic images):

- **Metrics.** `contrast` and `edge_sharpness` fall steadily with blur, with or without noise, so they are safe to maximize. `entropy` **rises** with blur, so maximizing it defocuses. `high_frequency_power` falls near focus but rises again further out and with noise, because noise and the image's non-periodic edges keep a fixed high-frequency share. In the 1D search, `contrast` and `edge_sharpness` end about 0.2–0.3 step from the optimum; `entropy` and `high_frequency_power` end 10–19 steps away. Use `contrast` (the notebook default) or `edge_sharpness`.
- **Search vs notebook.** `QuadraticSearch` matches the notebook code. Over 60 random peaks, the first model point is identical in 100 % (z, C1) and 93 % (A1) of cases; the rest are 1–2 grid steps away, at ties. Accuracy and the number of measurements are the same. The notebook's grids put candidates exactly on its r/40 and disk edges, so at a tie its choice depends on float rounding (units, start z). The new code follows exact arithmetic there.
- **Segmentation.** Otsu with settings in nm finds 20 of 20 synthetic particles. On the twin, dim particles are missed: one global threshold does not catch particles much fainter than the others. SAM (optional extra) or a local threshold would help.

Findings from block 3 (`tests/test_nodes_analysis.py`, `notebooks/03_nodes_analysis.ipynb`):

- **The tune nodes find a hidden optimum** through the full path (MCP calls → Tiled image → metric → search) on a fake microscope whose image blurs: z +4.36 µm (truth +4.3), C1 24.9 nm (truth 23), A1 (2.8, −2.3) nm (truth 3, −2). They end at the best point.
- **Restore works.** When an image fails in the middle of a search, z and the corrector go back exactly to where the node started. This is tested on the fake microscope and on the twin.
- **Guards.** Preflight refuses `entropy` and `high_frequency_power` for tuning, a step larger than `max_offset`, a `max_offset` over half the per-move limit (z 50 µm, corrector 100 nm per call), and unknown setting names.
- **Overview → segment → EDS on the twin:** 26 particles at a 40 nm FOV; EDS at the first 20 aim points gives 20 particle compositions.

**Open decisions**

1. Who opens the asyncroscopy PR for scan settings?
2. Add SAM segmentation (the optional `sam` extra) now, or after the graph works?
