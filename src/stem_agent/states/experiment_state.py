"""State of one particle-analysis run.

Rules:
    no pixel arrays in state, only Tiled keys and file paths
    nodes return only the keys they change
    settings is the input of the run: never changed by a node
"""
import operator
from typing import Annotated, Literal

from typing_extensions import NotRequired, TypedDict


class TuneStep(TypedDict):                     # one step of settings.protocol
    name: str                                  # record key and file names, unique
    tune: Literal["z", "C1", "A1"]             # what the step changes
    fov_m: NotRequired[float]                  # set before tuning; missing: keep the current FOV
    metric: NotRequired[str]                   # missing keys: config.TUNING[tune]
    step_m: NotRequired[float]
    max_offset_m: NotRequired[float]
    max_iterations: NotRequired[int]
    min_step_m: NotRequired[float]             # smallest move worth making; default step_m / 16
    method: NotRequired[Literal["quadratic", "grid"]]   # search; default "quadratic"
    rounds: NotRequired[int]                   # grid only: zoom-in rounds, default 1


class SegmentationSettings(TypedDict, total=False):   # overrides of config.SEGMENTATION
    method: Literal["otsu"]
    smooth_nm: float
    min_diameter_nm: float
    edge_px: int


class SamSettings(TypedDict, total=False):            # overrides of config.SAM
    min_visible: float
    min_diameter_nm: float
    max_diameter_nm: float | None
    edge_px: int
    min_bright: float                          # 0..1, 0 = off


class AutoFocusSettings(TypedDict, total=False):      # overrides of config.AUTO_FOCUS
    fov_m: float | None
    metric: str
    worse_tolerance: float
    max_wait_s: float


class RunSettings(TypedDict):
    instrument: Literal["twin", "real"]        # config.INSTRUMENTS key
    stage_xy_m: NotRequired[tuple[float, float] | None]    # stage x, y for the move node; None: stay
    tuning_fov_m: float                        # particle graph: FOV of the overview (set_fov node)
    detector: NotRequired[str]                 # scanning detector, default "haadf"
    max_particles: NotRequired[int]            # EDS cap, default LIMITS["max_particles"]
    beam_order: NotRequired[str]               # override INSTRUMENTS[instrument]["beam_order"]
    protocol: NotRequired[list[TuneStep]]      # required by tune_node; no default
    segmentation: NotRequired[SegmentationSettings]    # Otsu segment_node
    sam: NotRequired[SamSettings]              # sam_segment_node (the particle graph)
    auto_focus: NotRequired[AutoFocusSettings]   # auto_focus_node runs only when present
    blank_beam_at_end: NotRequired[bool]      # save node blanks the beam; default off
    run_tag: NotRequired[str]


class RunInfo(TypedDict):
    run_dir: str
    instrument: str
    mcp_url: str | None                        # None if preflight failed before connecting
    tiled_uri: str | None
    tools: dict[str, str]                      # short name -> MCP tool name
    started: str                               # ISO time


class InstrumentSnapshot(TypedDict):
    index: int                                 # 0 = first snapshot of the run (the gate)
    time: str                                  # ISO time
    stage: list[float]                         # [x, y, z, alpha, beta], m and deg
    fov_m: float | None
    defocus_m: float | None
    stem_mode: bool | None                     # None = the instrument does not report it
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
    name: str                                  # the protocol step's name
    tune: str                                  # "z", "C1", "A1"; "auto_focus" (auto_focus_node)
    fov_m: float | None                        # FOV during the step
    run_dir: str                               # the run it belongs to (progress counts this run only)
    settings: dict                             # the step as run: defaults merged
    best_offset_m: list[float]                 # from the start of the node
    best_score: float
    n_images: int
    stop_reason: str                           # quadratic: "converged", "max_iterations", "no_candidates";
                                               # grid: "converged", "peak_at_edge", "no_peak"
    best_image_key: str
    history_path: str                          # tune_<name>.json: every point
    figure_path: str                           # tune_<name>.png


class Segmentation(TypedDict):
    method: str                                # "otsu" or "sam"
    settings: dict
    n_particles: int
    labels_path: str                           # .npy label image
    labels_key: NotRequired[str]               # sam: SAM's raw label image in Tiled
    overlay_path: str                          # .png outlines + aim points


class Overview(TypedDict):                     # one overview_node image
    index: int                                 # position in state["images"]
    tiled_key: str
    fov_m: float | None
    shape: list[int]                           # [rows, cols]
    metrics: dict[str, float]                  # analysis.metrics.all_metrics
    preview_path: str                          # overview_NN.png


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
    snapshots: NotRequired[list[InstrumentSnapshot]]   # every status_node call: [0] start, [-1] latest
    current: NotRequired[InstrumentSnapshot]   # latest snapshot + offsets tracked by the tune nodes
    tuning: NotRequired[Annotated[dict[str, TuneResult], operator.or_]]   # step name -> result
    overview: NotRequired[Overview]            # the latest image: what segment / EDS use
    images: NotRequired[list[Overview]]        # every overview_node image: [0] first, [-1] latest
    segmentation: NotRequired[Segmentation]
    particles: NotRequired[list[Particle]]
    spectra: NotRequired[list[SpectrumRecord]]
    run_status: NotRequired[Literal["running", "ok", "no_particles", "failed"]]
    errors: NotRequired[Annotated[list[str], operator.add]]   # every failure, in order
    finished: NotRequired[str]                 # ISO time, set by save
    summary_path: NotRequired[str]             # summary.json
