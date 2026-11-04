"""Block 0: connect to the MCP server and check the tools we need.

    lists the needed tools (config.TOOLS) with their arguments, flags missing ones
    reads the stage, acquires one HAADF image, reads it back from Tiled

Run:  uv run python scripts/check_mcp.py [twin|real] [--all]
      --all also lists every other tool on the server
"""
import asyncio
import json
import sys

from stem_agent.config import TOOLS
from stem_agent.data import tiled_io
from stem_agent.mcp import session


def describe_args(tool) -> str:
    props = (tool.inputSchema or {}).get("properties", {})
    return ", ".join(f"{name}: {spec.get('type', '?')}" for name, spec in props.items())


async def main(target: str, show_all: bool) -> int:
    client = session.connect(target)
    tools = await client.list_tools()
    print(f"[check] {client.url} ({target}): {len(tools)} tools")

    needed = {short: session.tool_name(short) for short in TOOLS}
    for short, name in needed.items():
        tool = tools.get(name)
        mark = "ok     " if tool else "MISSING"
        print(f"  {mark} {short:<20} {name}({describe_args(tool) if tool else ''})")
    missing = [short for short, name in needed.items() if name not in tools]

    if show_all:
        others = sorted(set(tools) - set(needed.values()))
        print(f"[check] other tools ({len(others)}):")
        for name in others:
            print(f"  {name}({describe_args(tools[name])})")

    stage = await session.call("get_stage")
    print(f"[check] stage [x, y, z, alpha, beta] = {stage}")

    uri = json.loads(await session.call("get_tiled_config"))["uri"]
    tiled_io.connect(uri)
    key = await session.call("acquire_image", ["haadf"])
    image = tiled_io.read_image(key)
    print(f"[check] HAADF from {uri}: key={key} shape={image.shape} "
          f"range=[{image.min():.3g}, {image.max():.3g}]")

    if missing:
        print(f"[check] MISSING tools: {missing}")
        return 1
    print("[check] all needed tools present")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    target = next((a for a in argv if not a.startswith("-")), "twin")
    sys.exit(asyncio.run(main(target, "--all" in argv)))
