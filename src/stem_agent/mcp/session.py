"""Process-wide MCP session: short tool names, argument mapping, result parsing, errors.

    connect(instrument) -> STEMMCPClient "twin" or "real"; one client per process
    instrument() -> str                  the connected instrument
    tool_name(short, instrument) -> str  short name -> real tool name (config.TOOLS + INSTRUMENTS)
    call(short, value=None) -> Any       call a tool by short name, return the parsed result
    call_tool(name, value=None) -> Any   same, by real tool name (e.g. "<Device>_State")
    build_args(tool, value) -> dict      a plain value is sent as the tool's only argument
    parse_result(result, tool) -> Any    structured result, else text; raises on errors
    class InstrumentError                the instrument refused or failed a command

Notes:
    asyncroscopy tools are named <TangoClass>_<command>.
    Argument names come from the Tango docs, or "arg" when none is given,
    so callers pass a plain value and build_args finds the name.
    MCP-side Tango timeout is 30 s per command.
"""
import json
from typing import Any

from mcp.types import CallToolResult, TextContent, Tool

from stem_agent.config import INSTRUMENTS, TOOLS
from stem_agent.mcp.client import STEMMCPClient

_client: STEMMCPClient | None = None
_instrument: str | None = None


class InstrumentError(RuntimeError):
    """The instrument refused or failed a command."""


def connect(instrument: str = "twin") -> STEMMCPClient:
    """Select the instrument and create its client (reused on repeat calls)."""
    global _client, _instrument
    if instrument not in INSTRUMENTS:
        raise ValueError(f"Unknown instrument {instrument!r}; use one of {sorted(INSTRUMENTS)}")
    url = INSTRUMENTS[instrument]["mcp_url"]
    if not url:
        raise ValueError(f"No MCP URL for instrument {instrument!r}; set it in .env")
    if _client is None or _instrument != instrument or _client.url != url:
        _client, _instrument = STEMMCPClient(url), instrument
    return _client


def client() -> STEMMCPClient:
    if _client is None:
        raise RuntimeError("Not connected; call session.connect(instrument) first")
    return _client


def instrument() -> str:
    if _instrument is None:
        raise RuntimeError("Not connected; call session.connect(instrument) first")
    return _instrument


def tool_name(short: str, instrument_name: str | None = None) -> str:
    if short not in TOOLS:
        raise KeyError(f"Unknown tool {short!r}; known: {sorted(TOOLS)}")
    role, command = TOOLS[short]
    if role is None:
        return command
    tango_class = INSTRUMENTS[instrument_name or instrument()].get(role, role)
    return f"{tango_class}_{command}"


def build_args(tool: Tool, value: Any = None) -> dict:
    """None -> no arguments; dict -> as is; anything else -> the tool's single argument."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    names = list((tool.inputSchema or {}).get("properties", {}))
    if len(names) != 1:
        raise ValueError(f"{tool.name} takes {names or 'no arguments'}; pass a dict")
    return {names[0]: value}


def parse_result(result: CallToolResult, tool: str) -> Any:
    """Return the tool's value. Image preview blocks are ignored."""
    texts = [block.text for block in result.content if isinstance(block, TextContent)]
    if result.isError:
        raise InstrumentError(f"{tool}: {' '.join(texts) or 'unknown error'}")

    structured = result.structuredContent
    if isinstance(structured, dict) and set(structured) == {"result"}:
        return structured["result"]               # FastMCP wraps non-object returns
    if structured is not None:
        return structured

    text = "\n".join(texts).strip()
    if not text or text == "null":
        return None
    if text[0] in "[{":
        try:
            return json.loads(text)
        except ValueError:
            pass
    return text


async def call_tool(name: str, value: Any = None) -> Any:
    mcp_client = client()
    tool = await mcp_client.get_tool(name)
    result = await mcp_client.call_tool(name, build_args(tool, value))
    return parse_result(result, name)


async def call(short: str, value: Any = None) -> Any:
    return await call_tool(tool_name(short), value)
