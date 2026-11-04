"""Thin wrapper around langchain_mcp_adapters.MultiServerMCPClient.

Nodes use raw MCP calls (exact result blocks, no LangChain conversion);
LangChain tools stay available for later LLM nodes.

Source: SPM_agent_simple/src/spm_agent/mcp/spm_mcp_client.py
"""
from datetime import timedelta

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.types import CallToolResult, Tool

from stem_agent.config import MCP_TIMEOUT_S

SERVER = "stem"


class STEMMCPClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self.client = MultiServerMCPClient({
            SERVER: {
                "transport": "streamable_http",
                "url": url,
                "timeout": timedelta(seconds=MCP_TIMEOUT_S),
                "sse_read_timeout": timedelta(seconds=MCP_TIMEOUT_S),
            },
        })
        self._tools: dict[str, Tool] | None = None

    async def list_tools(self) -> dict[str, Tool]:
        """Tool registry with input schemas, fetched once."""
        if self._tools is None:
            async with self.client.session(SERVER) as session:
                result = await session.list_tools()
            self._tools = {tool.name: tool for tool in result.tools}
        return self._tools

    async def get_tool(self, name: str) -> Tool:
        tools = await self.list_tools()
        if name not in tools:
            raise ValueError(f"Tool {name!r} not found on MCP server {self.url}. "
                             f"Available: {sorted(tools)}")
        return tools[name]

    async def call_tool(self, name: str, args: dict | None = None) -> CallToolResult:
        async with self.client.session(SERVER) as session:
            return await session.call_tool(name, args or {},
                                           read_timeout_seconds=timedelta(seconds=MCP_TIMEOUT_S))

    async def get_langchain_tools(self):
        """LangChain tools, for LLM nodes."""
        return await self.client.get_tools(server_name=SERVER)
