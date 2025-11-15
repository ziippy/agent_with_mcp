# -*- coding: utf-8 -*-
"""
MCP Connection Module
MCP 서버 연결 및 도구 관리
"""

from typing import Dict, List
from dataclasses import dataclass

from google.adk.tools import BaseTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams


@dataclass
class MCPServerConnection:
    """단일 MCP 서버 연결"""
    name: str
    toolset: McpToolset


class MCPManager:
    """MCP 서버 연결 관리자"""

    def __init__(self):
        self.servers: Dict[str, MCPServerConnection] = {}
        self.all_tools: List[BaseTool] = []

    def _normalize_url(self, url: str) -> str:
        """URL 정규화"""
        return url if url.endswith("/") else url + "/"

    def _prefix_tool(self, tool: BaseTool, prefix: str) -> BaseTool:
        """도구 이름에 prefix 추가하고 input_schema 수정"""
        class PrefixedTool(tool.__class__):
            @property
            def name(self):
                original = getattr(super(), "name", getattr(tool, "name", type(tool).__name__))
                return f"{prefix}__{original}"

        wrapped = PrefixedTool.__new__(PrefixedTool)
        wrapped.__dict__ = tool.__dict__.copy()

        # input_schema가 None이면 raw_mcp_tool에서 가져오기
        if getattr(wrapped, 'input_schema', None) is None and hasattr(wrapped, 'raw_mcp_tool'):
            raw_tool = wrapped.raw_mcp_tool
            if hasattr(raw_tool, 'inputSchema'):
                wrapped.input_schema = raw_tool.inputSchema
            elif hasattr(raw_tool, 'input_schema'):
                wrapped.input_schema = raw_tool.input_schema

        return wrapped

    async def connect_mcp_server(self, server_name: str, base_url: str, auth_bearer: str = "",
                                 tenant_uuid: str = "", account_id: str = "") -> MCPServerConnection:
        """MCP 서버 연결"""
        base_url = self._normalize_url(base_url)
        headers = {}

        # Authorization 헤더
        if auth_bearer:
            headers["Authorization"] = f"Bearer {auth_bearer}"

        # 커스텀 헤더 추가
        if tenant_uuid:
            headers["X-Tenant-UUID"] = tenant_uuid
        if account_id:
            headers["X-Account-ID"] = account_id

        try:
            conn_params = StreamableHTTPConnectionParams(
                url=base_url,
                headers=headers if headers else None,
                timeout=10.0,
                sse_read_timeout=300.0,
            )
            toolset = McpToolset(connection_params=conn_params)
            tools = await toolset.get_tools()

            # 디버깅: 도구 schema 확인
            print(f"\n🔍 [{server_name}] 도구 Schema 디버깅:", flush=True)
            if headers:
                print(f"   📤 전송 헤더: {', '.join([k for k in headers.keys()])}", flush=True)
            for tool in tools:
                tool_name = getattr(tool, 'name', type(tool).__name__)
                tool_input_schema = getattr(tool, 'input_schema', None)

                # raw_mcp_tool에서 schema 확인
                if tool_input_schema is None and hasattr(tool, 'raw_mcp_tool'):
                    raw_tool = tool.raw_mcp_tool
                    if hasattr(raw_tool, 'inputSchema'):
                        tool_input_schema = raw_tool.inputSchema
                        print(f"   ✅ [{tool_name}] Found inputSchema in raw_mcp_tool", flush=True)
                    elif hasattr(raw_tool, 'input_schema'):
                        tool_input_schema = raw_tool.input_schema
                        print(f"   ✅ [{tool_name}] Found input_schema in raw_mcp_tool", flush=True)

                if tool_input_schema:
                    print(f"   {tool_name}: {len(tool_input_schema.get('properties', {}))} parameters", flush=True)

            connection = MCPServerConnection(
                name=server_name,
                toolset=toolset
            )
            self.servers[server_name] = connection

            # 도구에 prefix 추가
            for tool in tools:
                prefixed_tool = self._prefix_tool(tool, server_name)
                self.all_tools.append(prefixed_tool)

            return connection

        except Exception as e:
            error_msg = str(e)
            print(f"\n[Error] Failed to connect to {server_name}: {error_msg}")
            print(f" URL: {base_url}")
            if server_name in self.servers:
                del self.servers[server_name]
            raise RuntimeError(f"Failed to connect to MCP server {server_name} at {base_url}: {error_msg}") from e

    async def close_all_servers(self):
        """모든 MCP 서버 연결 종료"""
        import asyncio
        for server_name, server in list(self.servers.items()):
            try:
                if server.toolset:
                    await asyncio.wait_for(server.toolset.close(), timeout=5.0)
                    print(f"✓ Closed {server_name}")
            except asyncio.TimeoutError:
                print(f"⚠️  Timeout closing server {server_name}")
            except asyncio.CancelledError:
                print(f"⚠️  Cancelled while closing server {server_name}")
            except Exception as e:
                print(f"⚠️  Error closing server {server_name}: {e}")
        self.servers.clear()
        self.all_tools.clear()

