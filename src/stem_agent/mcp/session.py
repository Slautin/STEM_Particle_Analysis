"""Process-wide MCP session: short tool names, result parsing, errors.

Planned:
    session() -> STEMMCPClient         one client per process
    resolve(short_name) -> str         short name -> real tool name (config.TOOL_NAMES)
    call(short_name, args) -> Any      call the tool, parse the result:
                                       text block (Tiled key / JSON) -> Python value;
                                       image preview blocks are ignored
    class InstrumentError              the instrument refused or failed a command

Notes:
    asyncroscopy tools are named <TangoClass>_<command>.
    MCP-side Tango timeout is 30 s per command.
"""
