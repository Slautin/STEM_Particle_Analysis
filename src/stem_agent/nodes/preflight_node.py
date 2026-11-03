"""Is the software ready? Read-only.

Reads:   settings
Writes:  run
MCP:     list tools, DATA_get_config
Checks:  every tool in config.TOOL_NAMES exists; settings are valid (step <= max_offset, ...)
Undo:    none
Source:  notebook cell 1 (partly)
"""
