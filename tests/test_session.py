"""Session tests: tool names, argument mapping, result parsing. No MCP server needed."""
import pytest
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

from stem_agent.mcp import session


def tool(props: dict) -> Tool:
    return Tool(name="tool", inputSchema={"type": "object", "properties": props})


def text(value: str) -> TextContent:
    return TextContent(type="text", text=value)


def result(*blocks, structured=None, error=False) -> CallToolResult:
    return CallToolResult(content=list(blocks), structuredContent=structured, isError=error)


@pytest.mark.parametrize("target, short, expected", [
    ("twin", "acquire_image", "DigitalTwinParticles_acquire_scanned_image"),
    ("real", "acquire_image", "AutoScriptMicroscope_acquire_scanned_image"),
    ("twin", "correct_aberration", "DigitalTwinCorrector_correct_aberration"),
    ("real", "correct_aberration", "CEOSCorrector_correct_aberration"),
    ("twin", "get_tiled_config", "DATA_get_config"),
    ("real", "list_devices", "list_devices"),
])
def test_tool_name(target, short, expected):
    assert session.tool_name(short, target) == expected


def test_tool_name_unknown():
    with pytest.raises(KeyError):
        session.tool_name("nope", "twin")


def test_build_args_single_value_uses_schema_name():
    assert session.build_args(tool({"detector_list": {}}), ["haadf"]) == {"detector_list": ["haadf"]}


def test_build_args_none_and_dict():
    t = tool({"arg": {}})
    assert session.build_args(t) == {}
    assert session.build_args(t, {"arg": 1.0}) == {"arg": 1.0}


def test_build_args_ambiguous():
    with pytest.raises(ValueError):
        session.build_args(tool({"a": {}, "b": {}}), 1.0)


def test_parse_structured_result_ignores_preview():
    preview = ImageContent(type="image", data="aGk=", mimeType="image/png")
    r = result(text("key-123"), preview, structured={"result": "key-123"})
    assert session.parse_result(r, "tool") == "key-123"


def test_parse_text_results():
    assert session.parse_result(result(text("[1.0, 2.0]")), "tool") == [1.0, 2.0]
    assert session.parse_result(result(text("123")), "tool") == "123"     # keys stay strings
    assert session.parse_result(result(), "tool") is None


def test_parse_error_raises():
    with pytest.raises(session.InstrumentError, match="tool: boom"):
        session.parse_result(result(text("boom"), error=True), "tool")
