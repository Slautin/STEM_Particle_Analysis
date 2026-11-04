"""State of one particle-analysis run.

Rules:
    no pixel arrays in state, only Tiled keys and file paths
    nodes return only the keys they change
    start is never changed
"""
import operator
from typing import Annotated, Literal

from typing_extensions import NotRequired, TypedDict


class SearchSettings(TypedDict, total=False):  # overrides of config.TUNING[name]
    metric: str                                # config.SAFE_METRICS only
    step_m: float
    max_offset_m: float
    max_iterations: int


class SegmentationSettings(TypedDict, total=False):   # overrides of config.SEGMENTATION
    method: Literal["otsu"]
    smooth_nm: float
    min_diameter_nm: float
    edge_px: int


class RunSettings(TypedDict):
    target: Literal["twin", "real"]
    target_xy_m: tuple[float, float]           # stage x, y for the move node
    tuning_fov_m: float                        # FOV for tuning and the overview
    detector: NotRequired[str]                 # scanning detector, default "haadf"
    max_particles: NotRequired[int]            # EDS cap, default LIMITS["max_particles"]
    beam_order: NotRequired[str]               # override TARGETS[target]["beam_order"]
    tuning: NotRequired[dict[str, SearchSettings]]      # "z" / "C1" / "A1" -> overrides
    segmentation: NotRequired[SegmentationSettings]
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
    c1_offset_m: NotRequired[float]            # corrector change applied in this run
    a1_offset_m: NotRequired[list[float]]      #   (the corrector has no readback)


class TuneResult(TypedDict):
    name: str                                  # "z", "C1", "A1"
    settings: dict                             # search settings used
    best_offset_m: list[float]                 # from the start of the node
    best_score: float
    n_images: int
    stop_reason: str                           # "converged", "max_iterations", "no_candidates"
    best_image_key: str
    history_path: str                          # tune_<name>.json: every point
    figure_path: str                           # tune_<name>.png


class Segmentation(TypedDict):
    method: str
    settings: dict
    n_particles: int
    labels_path: str                           # .npy label image
    overlay_path: str                          # .png outlines + aim points


class Overview(TypedDict):
    tiled_key: str
    fov_m: float | None
    shape: list[int]                           # [rows, cols]
    preview_path: str


class Particle(TypedDict):
    id: int                                    # 1 = largest
    row: float                                 # beam aim point in overview pixels:
    col: float                                 #   the pixel deepest inside the particle
    area_px: NotRequired[int]
    centroid: NotRequired[list[float]]         # centre of mass [row, col]
    label: NotRequired[int]                    # value in the segmentation label image


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
    tuning: NotRequired[Annotated[dict[str, TuneResult], operator.or_]]   # merged per node
    overview: NotRequired[Overview]
    segmentation: NotRequired[Segmentation]
    particles: NotRequired[list[Particle]]
    spectra: NotRequired[list[SpectrumRecord]]
    run_status: NotRequired[Literal["running", "ok", "no_particles", "failed"]]
    error: NotRequired[str]
