"""Thin wrapper around langchain_mcp_adapters.MultiServerMCPClient.

Planned:
    class STEMMCPClient
        get_tools()                    fetch and cache the tool registry once
        get_tool(name)                 lookup with a clear error listing available tools
        call_tool(name, args)          invoke one tool

Source: SPM_agent_simple/src/spm_agent/mcp/spm_mcp_client.py
"""
