import os
import json
import asyncio
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AzureOpenAI, BadRequestError

from google.adk import Agent, Runner
from google.adk.tools import BaseTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

load_dotenv()


class ContentFilterError(Exception):
    """Azure OpenAI 콘텐츠 필터링 에러"""
    def __init__(self, filtered_categories: List[str], original_error: Exception):
        self.filtered_categories = filtered_categories
        self.original_error = original_error
        super().__init__(f"Content filtered: {', '.join(filtered_categories)}")


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
            print("[AOAI] init failed ->", e)

    def chat_completion(self, messages: List[Dict[str, Any]],
                        tools: Optional[List[Dict[str, Any]]] = None,
                        stream: bool = False):
        """Azure OpenAI Chat Completion 호출"""
        try:
            return self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                stream=stream,
            )
        except BadRequestError as e:
            # 콘텐츠 필터링 에러 처리
            error_str = str(e)
            if 'content_filter' in error_str or 'content management policy' in error_str.lower():
                error_body = getattr(e, 'body', None)
                filtered_categories = []
                if error_body and isinstance(error_body, dict):
                    error_info = error_body.get('error', {})
                    inner_error = error_info.get('innererror', {})
                    filter_result = inner_error.get('content_filter_result', {})
                    for category, details in filter_result.items():
                        if isinstance(details, dict) and details.get('filtered'):
                            severity = details.get('severity', 'unknown')
                            filtered_categories.append(f"{category}={severity}")
                if not filtered_categories:
                    filtered_categories = ['content_filter']
                raise ContentFilterError(filtered_categories=filtered_categories, original_error=e)
            raise


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
        self._closing = False  # 닫는 중 플래그
        self.aoai_wrapper: Optional[AzureOpenAIWrapper] = None

    @staticmethod
    def _normalize_url(url: str) -> str:
        # 서버가 /mcp/ 같은 트레일링 슬래시를 기대하는 경우 리다이렉트 루프 방지
        return url if url.endswith("/") else url + "/"

    async def connect_mcp_server(self, server_name: str, base_url: str, auth_bearer: str = "") -> MCPServerConnection:
        if self._closing:
            raise RuntimeError("Agent is closing; refuse new connections.")

        base_url = self._normalize_url(base_url)

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
            model="",  # 모델 호출은 직접 AOAIWrapper로
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
        """모든 MCP 서버 연결을 안전하게 종료"""
        self._closing = True
        async def _safe_close(name: str, ts: McpToolset):
            try:
                await asyncio.wait_for(ts.close(), timeout=8.0)
                print(f"✓ Closed {name}")
            except asyncio.TimeoutError:
                print(f"⚠️  Timeout closing server {name} (forced close)")
            except asyncio.CancelledError:
                print(f"⚠️  Cancelled while closing server {name}")
            except Exception as e:
                print(f"⚠️  Error closing server {name}: {e}")
        # 순차 종료
        for name, srv in list(self.servers.items()):
            if srv.toolset:
                await _safe_close(name, srv.toolset)
        self.servers.clear()
        self.all_tools.clear()
        self._closing = False


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
        # ExitStack이 알아서 이미 열린 리소스들 정리
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
        print(f"\n[Iteration {iteration}]", flush=True)

        inference_start = time.time()
        try:
            response = agent.aoai_wrapper.chat_completion(messages, tools=tools_for_openai, stream=False)
            choice = response.choices[0].message
            inference_time = time.time() - inference_start
            print(f"⏱️  추론 시간: {inference_time:.2f}초", flush=True)
        except ContentFilterError as e:
            print(f"\n🚫 콘텐츠 필터링 차단: {', '.join(e.filtered_categories)}", flush=True)
            print(f"💡 프롬프트를 다시 작성해주세요.", flush=True)
            return "요청이 콘텐츠 정책에 의해 차단되었습니다."
        except BadRequestError as e:
            print(f"\n❌ API 오류: {str(e)}", flush=True)
            return f"요청 처리 실패: {str(e)}"

        # 도구 호출이 있는 경우
        if getattr(choice, "tool_calls", None):
            print(f"\n🔧 [TOOL CALL DETECTED] count = {len(choice.tool_calls)}", flush=True)

            tool_results: List[Dict[str, Any]] = []
            for tc in choice.tool_calls:
                print(f"  ├─ Tool name : {tc.function.name}", flush=True)
                print(f"  ├─ Args      : {tc.function.arguments}", flush=True)
                print(f"  └─ Call ID   : {tc.id}", flush=True)

                tool_name = tc.function.name
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                tool_found = False

                tool_start = time.time()
                for tool in agent.all_tools:
                    current_tool_name = getattr(tool, 'name', type(tool).__name__)
                    if current_tool_name == tool_name:
                        try:
                            from google.adk.models import LlmRequest
                            class DummyToolContext:
                                def __init__(self):
                                    self.llm_request = LlmRequest(contents=[])

                            tool_context = DummyToolContext()
                            result = await tool.run_async(args=args, tool_context=tool_context)
                            tool_time = time.time() - tool_start
                            print(f"  ✅ 도구 실행 완료 ({tool_time:.2f}초)", flush=True)

                            tool_results.append({
                                "tool_call_id": tc.id,
                                "content": str(result),
                            })
                            tool_found = True
                            break
                        except Exception as e:
                            tool_time = time.time() - tool_start
                            print(f"  ❌ 도구 실행 실패 ({tool_time:.2f}초): {str(e)}", flush=True)
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

        else:
            # 도구 호출이 없으면 스트리밍으로 최종 답변 출력
            print("\n💬 [Assistant] ", end="", flush=True)
            try:
                stream_start = time.time()
                stream_response = agent.aoai_wrapper.chat_completion(messages, tools=tools_for_openai, stream=True)

                collected_content = ""
                for chunk in stream_response:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            print(delta.content, end="", flush=True)
                            collected_content += delta.content

                stream_time = time.time() - stream_start
                print(f"\n⏱️  스트리밍 시간: {stream_time:.2f}초", flush=True)

                final_answer = collected_content or choice.content or "(no content)"
                return final_answer
            except ContentFilterError as e:
                print(f"\n\n🚫 콘텐츠 필터링 차단: {', '.join(e.filtered_categories)}", flush=True)
                return "응답이 콘텐츠 정책에 의해 차단되었습니다."
            except BadRequestError as e:
                print(f"\n\n❌ API 오류: {str(e)}", flush=True)
                return f"응답 생성 실패: {str(e)}"

    return "Maximum iterations reached."


async def main():
    print("=" * 60)
    print("Super Agent: Google ADK + 2 MCP Servers + Azure OpenAI")
    print("=" * 60)
    agent = None
    try:
        agent = await initialize_super_agent()
        print("\n✅ Super Agent is ready!")
        print("Type 'quit' or 'exit' to stop.\n")
        while True:
            q = input("\n🧑 You> ").strip()
            if q.lower() in {"quit", "exit", "q"}:
                break
            if not q:
                continue
            try:
                conversation_start = time.time()
                ans = await run_conversation(agent, q)
                total_time = time.time() - conversation_start
                print(f"\n⏱️  총 소요 시간: {total_time:.2f}초")
            except (ContentFilterError, BadRequestError):
                pass
            except Exception as e:
                print(f"\n❌ [Error] {e}")
                import traceback
                traceback.print_exc()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user.")
    except Exception as e:
        print(f"\n❌ [Fatal Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if agent:
            print("\n🔌 Closing all MCP server connections...")
            try:
                await agent.close_all_servers()
                print("✓ All connections closed.")
            except asyncio.CancelledError:
                print("⚠️  Connection cleanup was cancelled")
            except Exception as e:
                print(f"⚠️  Error during cleanup: {e}")


if __name__ == "__main__":
    asyncio.run(main())

