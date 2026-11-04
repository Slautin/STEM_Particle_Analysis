"""Project settings. One place for every number and name.

Done (block 0):
    TARGETS                   "twin" / "real": MCP URL and Tango class names
    TOOLS                     short name -> (device role, Tango command)

Planned (later blocks):
    DEFAULT_SETTINGS          RunSettings defaults (step sizes, metrics, iterations from the notebook)
    LIMITS                    hard safety limits: max z offset, max C1/A1 offset, move tolerance,
                              max particles, min particle area
    READY_CHECKS              status_start conditions (STEM mode, valves open, HT on, stopper out, not blanked)
    RUNS_ROOT, new_run(tag)   run folder: runs/<timestamp>_<tag>/
    READ_RETRY                RetryPolicy for read-only calls, network errors only
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# MCP tool names are <TangoClass>_<command>; the class differs between twin and real.
TARGETS = {
    "twin": {
        "mcp_url": os.getenv("STEM_MCP_URL", "http://127.0.0.1:8000/mcp"),
        "microscope": "DigitalTwinParticles",
        "corrector": "DigitalTwinCorrector",
    },
    "real": {
        "mcp_url": os.getenv("STEM_REAL_MCP_URL"),
        "microscope": "AutoScriptMicroscope",
        "corrector": "CEOSCorrector",
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
