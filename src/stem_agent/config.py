"""Project settings. One place for every number and name.

Planned:
    STEM_MCP_URL              from .env, MCP server config for MultiServerMCPClient
    TOOL_NAMES                short name -> real MCP tool name, per instrument class
                              (AutoScriptMicroscope / DigitalTwinParticles, CEOSCorrector / DigitalTwinCorrector)
    DEFAULT_SETTINGS          RunSettings defaults (step sizes, metrics, iterations from the notebook)
    LIMITS                    hard safety limits: max z offset, max C1/A1 offset, move tolerance,
                              max particles, min particle area
    READY_CHECKS              status_start conditions (STEM mode, valves open, HT on, stopper out, not blanked)
    RUNS_ROOT, new_run(tag)   run folder: runs/<timestamp>_<tag>/
    READ_RETRY                RetryPolicy for read-only calls, network errors only
"""
