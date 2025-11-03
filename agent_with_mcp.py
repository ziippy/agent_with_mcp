import os
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AzureOpenAI

from google.adk import Agent, Runner
from google.adk.tools import BaseTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

load_dotenv()


class AzureOpenAIWrapper:
    """Azure OpenAI를 위한 간단한 래퍼 (ADK Tool 시스템과 함께 사용)"""

    def __init__(self, api_key: str, api_version: str, azure_endpoint: str, deployment: str):
        try:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
            )
            self.deployment = deployment
        except Exception as e:
            print("[AOAI] chat completions failed ->", e)

    def chat_completion(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None):
        """Azure OpenAI Chat Completion 호출"""
        return self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
        )


@dataclass
class MCPServerConnection:
    """단일 MCP 서버 연결을 관리하는 클래스"""
    name: str
    toolset: McpToolset


class SuperAgent:
    """Google ADK 기반 2개 MCP 서버를 관리하는 Super Agent"""

    def __init__(self):
        self.servers: Dict[str, MCPServerConnection] = {}
        self.agent: Optional[Agent] = None
        self.runner: Optional[Runner] = None
        self.all_tools: List[BaseTool] = []

    async def connect_mcp_server(self, server_name: str, base_url: str, auth_bearer: str = "") -> MCPServerConnection:
        headers = {}
        if auth_bearer:
            headers["Authorization"] = f"Bearer {auth_bearer}"

        try:
            conn_params = StreamableHTTPConnectionParams(
                url=base_url,
                headers=headers if headers else None,
                timeout=10.0,
                sse_read_timeout=300.0,
            )
            toolset = McpToolset(connection_params=conn_params)
            tools = await toolset.get_tools()
            connection = MCPServerConnection(
                name=server_name,
                toolset=toolset
            )
            self.servers[server_name] = connection
            for tool in tools:
                self.all_tools.append(tool)
            return connection
        except Exception as e:
            error_msg = str(e)
            if "400 Bad Request" in error_msg:
                print(f"\n[Error] HTTP 400 Bad Request for {server_name}")
                print(f" URL: {base_url}")
                print(f" 가능한 원인:")
                print(f" - URL이 올바르지 않거나 엔드포인트가 다름")
                print(f" - 인증 토큰이 필요하거나 잘못됨 (현재 bearer: {'설정됨' if auth_bearer else '없음'})")
                print(f" - 서버가 해당 경로를 지원하지 않음")
            elif "Connection" in error_msg or "connect" in error_msg.lower():
                print(f"\n[Error] Connection failed for {server_name}")
                print(f" URL: {base_url}")
                print(f" 가능한 원인: 네트워크 문제 또는 서버가 다운됨")
            else:
                print(f"\n[Error] Failed to connect to {server_name}: {error_msg}")
                print(f" URL: {base_url}")
            if server_name in self.servers:
                del self.servers[server_name]
            raise RuntimeError(f"Failed to connect to MCP server {server_name} at {base_url}: {error_msg}") from e

    def initialize_agent(self):
        self.aoai_wrapper = AzureOpenAIWrapper(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        )
        self.agent = Agent(
            name="super_agent",
            description="Super agent that can use tools from 2 MCP servers",
            model="",
            tools=self.all_tools,
            instruction=(
                "You are a super agent that can use tools from multiple MCP servers. "
                "Use the appropriate tools from different servers to help the user."
            ),
        )
        from google.adk.sessions import InMemorySessionService
        session_service = InMemorySessionService()
        self.runner = Runner(app_name="agents", agent=self.agent, session_service=session_service)

    async def close_all_servers(self):
        for server_name, server in list(self.servers.items()):
            try:
                if server.toolset:
                    await server.toolset.close()
            except Exception as e:
                print(f"Error closing server {server_name}: {e}")
        self.servers.clear()
        self.all_tools.clear()


async def initialize_super_agent() -> SuperAgent:
    agent = SuperAgent()
    mcp1_connected = False
    mcp2_connected = False
    try:
        mcp1_url = os.environ.get("MCP_SERVER_1_URL", "")
        mcp1_bearer = os.environ.get("MCP_SERVER_1_AUTH_BEARER", "")
        if mcp1_url:
            print(f"Connecting to MCP Server 1: {mcp1_url}")
            try:
                await agent.connect_mcp_server("mcp1", mcp1_url, mcp1_bearer)
                print("✓ Connected to MCP Server 1")
                mcp1_connected = True
            except Exception as e:
                print(f"✗ Failed to connect to MCP Server 1: {e}")
        else:
            raise RuntimeError("MCP_SERVER_1_URL 환경 변수가 필요합니다")
        mcp2_url = os.environ.get("MCP_SERVER_2_URL", "")
        mcp2_bearer = os.environ.get("MCP_SERVER_2_AUTH_BEARER", "")
        if mcp2_url:
            print(f"Connecting to MCP Server 2: {mcp2_url}")
            try:
                await agent.connect_mcp_server("mcp2", mcp2_url, mcp2_bearer)
                print("✓ Connected to MCP Server 2")
                mcp2_connected = True
            except Exception as e:
                print(f"✗ Failed to connect to MCP Server 2: {e}")
        else:
            raise RuntimeError("MCP_SERVER_2_URL 환경 변수가 필요합니다")
        if not mcp1_connected and not mcp2_connected:
            raise RuntimeError("Both MCP servers failed to connect. At least one server must be connected.")
        if not mcp1_connected:
            print("⚠ Warning: MCP Server 1 is not connected, continuing with Server 2 only.")
        if not mcp2_connected:
            print("⚠ Warning: MCP Server 2 is not connected, continuing with Server 1 only.")
        print("Initializing Google ADK Agent...")
        agent.initialize_agent()
        print(f"✓ Agent initialized with {len(agent.all_tools)} tools")
        return agent
    except Exception as e:
        raise RuntimeError(f"Failed to initialize super agent: {e}") from e


async def run_conversation(agent: SuperAgent, user_query: str, max_iterations: int = 10) -> str:
    if not agent.agent or not agent.aoai_wrapper:
        raise RuntimeError("Agent not initialized")
    tools_for_openai = []
    for tool in agent.all_tools:
        tool_name = getattr(tool, 'name', type(tool).__name__)
        tool_description = getattr(tool, 'description', '')
        tool_input_schema = getattr(tool, 'input_schema', None) or {"type": "object", "properties": {}}
        tools_for_openai.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description or "",
                "parameters": tool_input_schema,
            },
        })
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a super agent that can use tools from multiple MCP servers. "
                "Tool names are prefixed with server names (e.g., 'mcp1__tool_name' or 'mcp2__tool_name'). "
                "Use the appropriate tools from different servers to help the user."
            )
        },
        {"role": "user", "content": user_query},
    ]
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        response = agent.aoai_wrapper.chat_completion(messages, tools=tools_for_openai)
        choice = response.choices[0].message
        if not getattr(choice, "tool_calls", None):
            return choice.content or "(no content)"
        tool_results: List[Dict[str, Any]] = []
        for tc in choice.tool_calls:
            tool_name = tc.function.name
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            tool_found = False
            for tool in agent.all_tools:
                current_tool_name = getattr(tool, 'name', type(tool).__name__)
                if current_tool_name == tool_name:
                    try:
                        from google.adk.tools.tool_context import ToolContext
                        from google.adk.models import LlmRequest
                        from google.adk.agents.invocation_context import InvocationContext
                        class DummyToolContext:
                            def __init__(self):
                                self.llm_request = LlmRequest(contents=[])

                        tool_context = DummyToolContext()
                        result = await tool.run_async(args=args, tool_context=tool_context)
                        tool_results.append({
                            "tool_call_id": tc.id,
                            "content": str(result),
                        })
                        tool_found = True
                        break
                    except Exception as e:
                        tool_results.append({
                            "tool_call_id": tc.id,
                            "content": f"Error: {str(e)}",
                        })
                        tool_found = True
                        break
            if not tool_found:
                tool_results.append({
                    "tool_call_id": tc.id,
                    "content": f"Tool {tool_name} not found",
                })
        messages.append({
            "role": "assistant",
            "content": choice.content or "",
            "tool_calls": [tc.model_dump() for tc in choice.tool_calls],
        })
        for tr in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "content": tr["content"],
            })
        if messages:
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    return msg["content"]
    return "Maximum iterations reached."


async def main():
    print("=" * 60)
    print("Super Agent: Google ADK + 2 MCP Servers + Azure OpenAI")
    print("=" * 60)
    agent = None
    try:
        agent = await initialize_super_agent()
        print("\nSuper Agent is ready!")
        print("Type 'quit' or 'exit' to stop.\n")
        while True:
            q = input("\nYou> ").strip()
            if q.lower() in {"quit", "exit", "q"}:
                break
            if not q:
                continue
            try:
                ans = await run_conversation(agent, q)
                print("\nAssistant>", ans)
            except Exception as e:
                print(f"\n[Error] {e}")
                import traceback
                traceback.print_exc()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n[Fatal Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if agent:
            print("\nClosing all MCP server connections...")
            await agent.close_all_servers()
            print("✓ All connections closed.")


if __name__ == "__main__":
    asyncio.run(main())
