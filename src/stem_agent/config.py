"""Project settings. One place for every number and name.

    INSTRUMENTS               "twin" / "real": MCP URL, Tango class names, beam order, strictness
    TOOLS                     short name -> (device role, Tango command)
    LIMITS                    hard safety limits (unit-mistake guards) and readback tolerances
    READY_CHECKS              status gate conditions; BAD_DEVICE_STATES
    TUNING, SAFE_METRICS      search defaults per actuator (z / C1 / A1); metrics allowed for tuning
    SEGMENTATION              Otsu segmentation defaults (nm)
    SAM                       SAM2 particle defaults (sam_segment_node)
    AUTO_FOCUS                OptiSTEM C1/A1 autofocus defaults (auto_focus_node)
    protocol_settings(settings)   settings["protocol"] (required): every step merged with
                              TUNING[step["tune"]] and checked
    segmentation_settings(settings)   defaults merged with the run's overrides, and checked
    sam_settings(settings)    SAM with settings["sam"] merged, and checked
    auto_focus_settings(settings)  AUTO_FOCUS with settings["auto_focus"] merged, and checked
    check_fov(fov_m)          FOV within LIMITS["fov_range_m"]
    RUNS_ROOT, new_run(tag)   run folder: runs/<timestamp>_<tag>/

Planned:
    READ_RETRY                RetryPolicy for read-only calls, network errors only
"""
import math
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# MCP tool names are <TangoClass>_<command>; the class differs between twin and real.
# beam_order: how an image pixel maps to place_beam [x, y] (fractions 0..1)
#   "col_row": x = col / width, y = row / height   (notebook assumption for the real microscope)
#   "row_col": x = row / height, y = col / width
# strict_ready: an unknown readiness value stops the run (real) or only warns (twin)
INSTRUMENTS = {
    "twin": {
        "mcp_url": os.getenv("STEM_MCP_URL", "http://127.0.0.1:8000/mcp"),
        "microscope": "DigitalTwinParticles",
        "corrector": "DigitalTwinCorrector",
        "beam_order": "row_col",             # measured in notebook 01 on the particle twin
        "strict_ready": False,
    },
    "real": {
        "mcp_url": os.getenv("STEM_REAL_MCP_URL"),
        "microscope": "AutoScriptMicroscope",
        "corrector": "CEOSCorrector",
        "beam_order": "col_row",             # UNVERIFIED on the instrument
        "strict_ready": True,
    },
}

# short name -> (device role, Tango command). Role None = native MCP tool;
# "microscope" / "corrector" = class from INSTRUMENTS; anything else = literal Tango class.
TOOLS = {
    "list_devices":       (None, "list_devices"),
    "get_tiled_config":   ("DATA", "get_config"),
    "get_stage":          ("microscope", "get_stage"),
    "move_stage":         ("microscope", "move_stage"),
    "get_fov":            ("microscope", "get_fov"),
    "set_fov":            ("microscope", "set_fov"),
    "get_defocus":        ("microscope", "get_defocus"),
    "get_parameters":     ("microscope", "get_parameters"),
    "acquire_image":      ("microscope", "acquire_scanned_image"),
    "acquire_spectrum":   ("microscope", "acquire_spectrum"),
    "place_beam":         ("microscope", "place_beam"),
    "blank_beam":         ("microscope", "blank_beam"),
    "correct_aberration": ("corrector", "correct_aberration"),
    "corrector_info":     ("corrector", "get_info"),
    "segment":            ("SEGMENTATION", "segment"),   # SAM2: image key -> label-image key
    "set_defocus":        ("microscope", "set_defocus"),  # only to restore after a bad autofocus
    "auto_focus":         ("microscope", "auto_focus"),   # AutoScript: OptiSTEM, method "C1A1"
}

# client-side limit for one MCP request; the server's Tango timeout is 30 s
MCP_TIMEOUT_S = 60.0

LIMITS = {
    "max_xy_move_m":       400e-6,         # one stage move in x/y
    "max_z_step_m":        700e-6,          # one stage move in z
    "max_corrector_step_m": 2000e-9,        # one correct_aberration call, per component
    "fov_range_m":         (1e-9, 50e-6),
    "stage_tolerance_m":   50e-9,          # readback vs target, per axis
    "fov_rel_tolerance":   0.01,
    "max_particles":       20,             # EDS points per run
}

# Search defaults per actuator, from the notebook (cells 10, 13, 15). A protocol step
# overrides any of them. max_offset must be <= half the per-move limit above, so that any
# jump inside +-max_offset is one allowed move (checked by protocol_settings).
TUNING = {
    "z":  {"metric": "contrast", "step_m": 1e-6,  "max_offset_m": 20e-6, "max_iterations": 20,
           "min_step_m": None, "method": "quadratic", "rounds": 1},
    "C1": {"metric": "contrast", "step_m": 10e-9, "max_offset_m": 50e-9, "max_iterations": 8,
           "min_step_m": None, "method": "quadratic", "rounds": 1},
    "A1": {"metric": "contrast", "step_m": 2e-9,  "max_offset_m": 10e-9, "max_iterations": 8,
           "min_step_m": None, "method": "quadratic", "rounds": 1},
}
# min_step_m: the smallest change worth making (the stage / corrector resolution). The search
# never asks for a point closer than this to one it measured, so every move is >= min_step_m,
# and it stops when its trust radius falls below it. None = step_m / 16 (notebook behaviour).
# method: "quadratic" (QuadraticSearch, max_iterations model points) or "grid" (GridSearch: a
# grid +-max_offset_m, step_m apart, a normal-curve fit, one image at the fitted peak; rounds > 1
# zoom in with half the spacing; z and C1 only).
# block 2: entropy rises with blur and high_frequency_power rises again with noise, so
# maximizing either drives toward blur
SAFE_METRICS = ("contrast", "edge_sharpness")

SEGMENTATION = {"method": "otsu", "smooth_nm": 0.3, "min_diameter_nm": 1.0, "edge_px": 1}

# SAM2 particles. min_visible: share of a SAM mask that must stay visible (smaller masks paint
# over larger ones) to count; min_diameter_nm: drop smaller particles (0 = keep all);
# edge_px: drop particles touching the frame edge
SAM = {"min_visible": 0.5, "min_diameter_nm": 0.0, "edge_px": 1}

# OptiSTEM C1/A1 autofocus (auto_focus_node); the node runs only when settings has "auto_focus".
# fov_m: set before (None = keep); metric: before/after check (larger = better);
# worse_tolerance: after < before x (1 - this) counts as worse -> defocus restored
#   (contrast noise is ~3 %, so 10 % means "clearly worse");
# max_wait_s: total time allowed for OptiSTEM, the 30 s MCP timeout included
AUTO_FOCUS = {"fov_m": None, "metric": "contrast", "worse_tolerance": 0.1, "max_wait_s": 300.0}


def check_fov(fov_m: float) -> None:
    lo, hi = LIMITS["fov_range_m"]
    if not lo <= fov_m <= hi:
        raise ValueError(f"FOV {fov_m:.3g} m outside {lo:.0e}..{hi:.0e} m")


def step_settings(step: dict) -> dict:
    """One protocol step: TUNING[step["tune"]] + the step's own values, checked."""
    name, tune = step.get("name"), step.get("tune")
    if not name:
        raise ValueError(f"every protocol step needs a name: {step}")
    if tune not in TUNING:
        raise ValueError(f"step {name!r}: tune must be one of {sorted(TUNING)}, got {tune!r}")
    unknown = set(step) - set(TUNING[tune]) - {"name", "tune", "fov_m"}
    if unknown:
        raise ValueError(f"step {name!r}: unknown keys {sorted(unknown)}")
    cfg = {**TUNING[tune], **step}
    if cfg["metric"] not in SAFE_METRICS:
        raise ValueError(f"step {name!r}: metric {cfg['metric']!r} is not allowed; it drives the "
                         f"search toward blur (block 2). Use one of {SAFE_METRICS}")
    if not 0 < cfg["step_m"] <= cfg["max_offset_m"]:
        raise ValueError(f"step {name!r}: need 0 < step_m <= max_offset_m, got {cfg}")
    move_limit = LIMITS["max_z_step_m"] if tune == "z" else LIMITS["max_corrector_step_m"]
    if cfg["max_offset_m"] > move_limit / 2:          # any jump inside +-max_offset is one move
        raise ValueError(f"step {name!r}: max_offset_m {cfg['max_offset_m']:.3g} m is over half "
                         f"the per-move limit ({move_limit:.3g} m)")
    if cfg["max_iterations"] < 0:
        raise ValueError(f"step {name!r}: max_iterations must be >= 0")
    if cfg["min_step_m"] is None:
        cfg["min_step_m"] = cfg["step_m"] / 16
    most = 0.75 * cfg["step_m"] if tune == "A1" else cfg["step_m"]   # A1 start circle: 0.77 step apart
    if not 0 < cfg["min_step_m"] <= most:
        raise ValueError(f"step {name!r}: need 0 < min_step_m <= {'0.75 * ' if tune == 'A1' else ''}"
                         f"step_m, got {cfg['min_step_m']:.3g} m")
    if tune == "z" and cfg["min_step_m"] < LIMITS["stage_tolerance_m"]:
        raise ValueError(f"step {name!r}: z min_step_m {cfg['min_step_m']:.3g} m is below the stage "
                         f"tolerance ({LIMITS['stage_tolerance_m']:.3g} m); the stage cannot "
                         f"reliably make a smaller move. Set min_step_m (e.g. its smallest real move)")
    if cfg["method"] not in ("quadratic", "grid"):
        raise ValueError(f"step {name!r}: method must be 'quadratic' or 'grid', got {cfg['method']!r}")
    if cfg["method"] == "grid":
        if tune == "A1":
            raise ValueError(f"step {name!r}: the grid search is for one value (z or C1); "
                             f"use method 'quadratic' for A1")
        points = 2 * math.floor(cfg["max_offset_m"] / cfg["step_m"] + 1e-9) + 1
        if not 7 <= points <= 51:
            raise ValueError(f"step {name!r}: grid of {points} points, need 7..51 "
                             f"(max_offset_m between 3 and 25 x step_m)")
        if not 1 <= cfg["rounds"] <= 5:
            raise ValueError(f"step {name!r}: rounds must be 1..5")
    if cfg.get("fov_m") is not None:
        check_fov(cfg["fov_m"])
    return cfg


def protocol_settings(settings: dict) -> list[dict]:
    """settings["protocol"]: every step merged with its defaults and checked. Required: no fallback."""
    protocol = settings.get("protocol")
    if not protocol:
        raise ValueError("settings has no protocol: a list of tune steps, e.g. "
                         "[{'name': 'z', 'tune': 'z'}, {'name': 'C1', 'tune': 'C1', 'fov_m': 40e-9}]")
    steps = [step_settings(step) for step in protocol]
    names = [step["name"] for step in steps]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"protocol step names must be unique, repeated: {repeated}")
    return steps


def segmentation_settings(settings: dict) -> dict:
    """SEGMENTATION with the run's overrides, checked. ValueError on a typo or a bad value."""
    override = settings.get("segmentation") or {}
    unknown = set(override) - set(SEGMENTATION)
    if unknown:
        raise ValueError(f"unknown segmentation settings: {sorted(unknown)}")
    cfg = {**SEGMENTATION, **override}
    if cfg["method"] != "otsu":
        raise ValueError(f"segmentation method {cfg['method']!r} not available (otsu only)")
    if cfg["smooth_nm"] <= 0 or cfg["min_diameter_nm"] < 0 or cfg["edge_px"] < 0:
        raise ValueError(f"bad segmentation settings: {cfg}")
    return cfg


def sam_settings(settings: dict) -> dict:
    """SAM with settings["sam"] merged, checked. ValueError on a typo or a bad value."""
    override = settings.get("sam") or {}
    unknown = set(override) - set(SAM)
    if unknown:
        raise ValueError(f"unknown sam settings: {sorted(unknown)}")
    cfg = {**SAM, **override}
    if not 0 < cfg["min_visible"] <= 1 or cfg["min_diameter_nm"] < 0 or cfg["edge_px"] < 0:
        raise ValueError(f"bad sam settings: {cfg}")
    return cfg


def auto_focus_settings(settings: dict) -> dict:
    """AUTO_FOCUS with settings["auto_focus"] merged, checked. ValueError on a typo or a bad value."""
    override = settings.get("auto_focus") or {}
    unknown = set(override) - set(AUTO_FOCUS)
    if unknown:
        raise ValueError(f"unknown auto_focus settings: {sorted(unknown)}")
    cfg = {**AUTO_FOCUS, **override}
    if cfg["metric"] not in SAFE_METRICS:
        raise ValueError(f"auto_focus metric must be one of {SAFE_METRICS}")
    if not 0 <= cfg["worse_tolerance"] < 1 or cfg["max_wait_s"] <= 0:
        raise ValueError(f"bad auto_focus settings: {cfg}")
    if cfg["fov_m"] is not None:
        check_fov(cfg["fov_m"])
    return cfg

# status gate (first snapshot of a run): normalized field -> required value
READY_CHECKS = {
    "stem_mode":          True,
    "beam_blanked":       False,
    "column_valves_open": True,
    "high_tension_on":    True,
    "beam_stopper_out":   True,
}
BAD_DEVICE_STATES = {"FAULT", "ALARM", "OFF", "UNKNOWN", "DISABLE"}

RUNS_ROOT = PROJECT_ROOT / "runs"


def new_run(tag: str = "") -> Path:
    """Create runs/<YYYYmmdd_HHMMSS>_<tag>/ and return it."""
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
    run_dir = RUNS_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
