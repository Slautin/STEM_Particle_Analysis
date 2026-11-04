"""Project settings. One place for every number and name.

    TARGETS                   "twin" / "real": MCP URL, Tango class names, beam order, strictness
    TOOLS                     short name -> (device role, Tango command)
    LIMITS                    hard safety limits (unit-mistake guards) and readback tolerances
    READY_CHECKS              status_start conditions; BAD_DEVICE_STATES
    TUNING, SAFE_METRICS      search defaults for z / C1 / A1; metrics allowed for tuning
    SEGMENTATION              segmentation defaults (nm)
    search_settings(settings, name), segmentation_settings(settings)
                              defaults merged with the run's overrides
    RUNS_ROOT, new_run(tag)   run folder: runs/<timestamp>_<tag>/

Planned:
    READ_RETRY                RetryPolicy for read-only calls, network errors only
"""
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
TARGETS = {
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
# "microscope" / "corrector" = class from TARGETS; anything else = literal Tango class.
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
}

# client-side limit for one MCP request; the server's Tango timeout is 30 s
MCP_TIMEOUT_S = 60.0

LIMITS = {
    "max_xy_move_m":       100e-6,         # one stage move in x/y
    "max_z_step_m":        50e-6,          # one stage move in z
    "max_corrector_step_m": 100e-9,        # one correct_aberration call, per component
    "fov_range_m":         (1e-9, 50e-6),
    "stage_tolerance_m":   10e-9,          # readback vs target, per axis
    "fov_rel_tolerance":   0.01,
    "max_particles":       20,             # EDS points per run
}

# Search defaults from the notebook (cells 10, 13, 15); a run overrides them with
# settings["tuning"][name]. max_offset must be <= half the per-move limit above, so that
# any jump inside +-max_offset is one allowed move (checked by preflight).
TUNING = {
    "z":  {"metric": "contrast", "step_m": 1e-6,  "max_offset_m": 20e-6, "max_iterations": 20},
    "C1": {"metric": "contrast", "step_m": 10e-9, "max_offset_m": 50e-9, "max_iterations": 8},
    "A1": {"metric": "contrast", "step_m": 2e-9,  "max_offset_m": 10e-9, "max_iterations": 8},
}
# block 2: entropy rises with blur and high_frequency_power rises again with noise, so
# maximizing either drives toward blur
SAFE_METRICS = ("contrast", "edge_sharpness")

SEGMENTATION = {"method": "otsu", "smooth_nm": 0.3, "min_diameter_nm": 1.0, "edge_px": 1}


def search_settings(settings: dict, name: str) -> dict:
    """TUNING[name] with the run's overrides; unknown keys are an error (typo guard)."""
    override = (settings.get("tuning") or {}).get(name, {})
    unknown = set(override) - set(TUNING[name])
    if unknown:
        raise ValueError(f"unknown tuning settings for {name}: {sorted(unknown)}")
    return {**TUNING[name], **override}


def segmentation_settings(settings: dict) -> dict:
    override = settings.get("segmentation") or {}
    unknown = set(override) - set(SEGMENTATION)
    if unknown:
        raise ValueError(f"unknown segmentation settings: {sorted(unknown)}")
    return {**SEGMENTATION, **override}

# status_start: normalized field -> required value
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
