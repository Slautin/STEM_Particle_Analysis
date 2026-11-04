"""Fake microscope for tests. Stands in for the MCP client and Tiled; no network.

    FakeMicroscope(target="twin")
        tools named like the real server (session.tool_name), with the same argument names
        stage / fov / beam / params / device states held in plain attributes
        calls: list of (tool name, args) in order
        fail: tool name -> error text, returned as an MCP error result
        stage_error_m: added to x after every move (bad readback)
        install(monkeypatch, tmp_path): plug into session, tiled_io and config.RUNS_ROOT

Planned (block 3): synthetic particle image whose blur depends on stage z, C1 and A1.
"""
import json

import numpy as np
from mcp.types import CallToolResult, TextContent, Tool

from stem_agent import config
from stem_agent.data import tiled_io
from stem_agent.mcp import session

TWIN_PARAMS = {"stem_mode": True, "defocus_m": 0.0}
REAL_PARAMS = {"optical_mode": "Stem", "beam_blanker": False, "column_valves": "Open",
               "is_accelerator_on": True, "beam_stopper": "Retracted"}


class FakeMicroscope:
    def __init__(self, target: str = "twin") -> None:
        self.target = target
        self.url = config.TARGETS["twin"]["mcp_url"]
        self.stage = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.fov = 20e-9
        self.beam = [0.5, 0.5]
        self.params = dict(TWIN_PARAMS if target == "twin" else REAL_PARAMS)
        self.states = {"DATA": "0", "SCAN": "0", "Stage": "0"}
        self.stage_error_m = 0.0
        self.fail: dict[str, str] = {}
        self.calls: list[tuple[str, dict]] = []
        self.image = np.zeros((64, 64))
        self.spectrum = (np.arange(8, dtype=float), None)
        self._tools = self._build_tools()

    # --- tool table: real name -> (argument name or None, handler) ---
    def _build_tools(self):
        name = lambda short: session.tool_name(short, self.target)
        tools = {
            name("list_devices"):       (None, lambda: ["asyncroscopy/instrument/default"]),
            name("get_tiled_config"):   (None, lambda: json.dumps({"uri": "http://fake-tiled"})),
            name("get_stage"):          (None, lambda: list(self.stage)),
            name("move_stage"):         ("position", self._move),
            name("get_fov"):            (None, lambda: self.fov),
            name("set_fov"):            ("fov", self._set_fov),
            name("get_defocus"):        (None, lambda: 0.0),
            name("get_parameters"):     (None, lambda: json.dumps(self.params)),
            name("acquire_image"):      ("detector_list", lambda detectors: "image-key"),
            name("acquire_spectrum"):   ("detector_name", lambda detector: "spectrum-key"),
            name("place_beam"):         ("position", self._place_beam),
            name("blank_beam"):         (None, lambda: None),
            name("correct_aberration"): ("args", lambda args: "{}"),
            name("corrector_info"):     (None, lambda: "{}"),
        }
        for cls in self.states:
            tools[f"{cls}_State"] = (None, lambda cls=cls: self.states[cls])
        return tools

    def _move(self, position):
        self.stage = [float(v) for v in position]
        self.stage[0] += self.stage_error_m

    def _set_fov(self, fov):
        self.fov = float(fov)

    def _place_beam(self, position):
        self.beam = [float(v) for v in position]

    # --- MCP client interface used by session ---
    async def list_tools(self) -> dict[str, Tool]:
        return {
            name: Tool(name=name, inputSchema={"type": "object",
                                               "properties": {arg: {}} if arg else {}})
            for name, (arg, _) in self._tools.items()
        }

    async def get_tool(self, name: str) -> Tool:
        return (await self.list_tools())[name]

    async def call_tool(self, name: str, args: dict | None = None) -> CallToolResult:
        args = args or {}
        self.calls.append((name, args))
        if name in self.fail:
            return CallToolResult(content=[TextContent(type="text", text=self.fail[name])],
                                  isError=True)
        arg, handler = self._tools[name]
        value = handler(args[arg]) if arg else handler()
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(value))],
                              structuredContent={"result": value}, isError=False)

    def called(self, short: str) -> list[dict]:
        """Arguments of every call to a tool, by short name."""
        real = session.tool_name(short, self.target)
        return [args for name, args in self.calls if name == real]

    def remove_tool(self, short: str) -> None:
        del self._tools[session.tool_name(short, self.target)]

    def install(self, monkeypatch, tmp_path) -> "FakeMicroscope":
        monkeypatch.setattr(session, "_client", self)
        monkeypatch.setattr(session, "_target", self.target)
        monkeypatch.setattr(session, "connect", lambda target="twin": self)
        monkeypatch.setattr(tiled_io, "connect", lambda uri: None)
        monkeypatch.setattr(tiled_io, "read_image", lambda key, detector="HAADF": self.image)
        monkeypatch.setattr(tiled_io, "read_spectrum", lambda key: self.spectrum)
        monkeypatch.setattr(config, "RUNS_ROOT", tmp_path / "runs")
        return self
