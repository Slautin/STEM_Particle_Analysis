"""State of one particle-analysis run.

Rules:
    no pixel arrays in state, only Tiled keys and file paths
    nodes return only the keys they change
    start is never changed

Planned (block 3): SearchSettings, TuneResult, Segmentation; tuning offsets in InstrumentSnapshot.
"""
from typing import Literal

from typing_extensions import NotRequired, TypedDict


class RunSettings(TypedDict):
    target: Literal["twin", "real"]
    target_xy_m: tuple[float, float]           # stage x, y for the move node
    tuning_fov_m: float                        # FOV for tuning and the overview
    detector: NotRequired[str]                 # scanning detector, default "haadf"
    max_particles: NotRequired[int]            # EDS cap, default LIMITS["max_particles"]
    beam_order: NotRequired[str]               # override TARGETS[target]["beam_order"]
    run_tag: NotRequired[str]


class RunInfo(TypedDict):
    run_dir: str
    target: str
    mcp_url: str
    tiled_uri: str
    tools: dict[str, str]                      # short name -> MCP tool name


class InstrumentSnapshot(TypedDict):
    stage: list[float]                         # [x, y, z, alpha, beta], m and deg
    fov_m: float | None
    defocus_m: float | None
    stem_mode: bool | None                     # None = the target does not report it
    beam_blanked: bool | None
    column_valves_open: bool | None
    high_tension_on: bool | None
    beam_stopper_out: bool | None
    devices: dict[str, str]                    # Tango class -> state ("ON", "FAULT", ...)
    ready: bool
    problems: list[str]                        # failed readiness checks
    warnings: list[str]                        # unknown values, unreadable fields
    raw_path: str                              # JSON with everything read


class Overview(TypedDict):
    tiled_key: str
    fov_m: float | None
    shape: list[int]                           # [rows, cols]
    preview_path: str


class Particle(TypedDict):
    id: int
    row: float                                 # centre in overview pixels
    col: float
    area_px: NotRequired[int]


class SpectrumRecord(TypedDict):
    particle_id: int
    beam_xy: list[float]                       # sent to place_beam
    tiled_key: str
    npy_path: str
    elements: NotRequired[list[str]]           # twin: one value per element


class ParticleRunState(TypedDict):
    settings: RunSettings
    run: NotRequired[RunInfo]
    start: NotRequired[InstrumentSnapshot]
    current: NotRequired[InstrumentSnapshot]
    end: NotRequired[InstrumentSnapshot]
    overview: NotRequired[Overview]
    particles: NotRequired[list[Particle]]
    spectra: NotRequired[list[SpectrumRecord]]
    run_status: NotRequired[Literal["running", "ok", "no_particles", "failed"]]
    error: NotRequired[str]
