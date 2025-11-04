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


@dataclass
class AgentResponse:
    """에이전트 응답 데이터 클래스"""
    content: str
    metadata: Dict[str, Any]
    success: bool
    agent_name: str


class SpecializedAgent:
    """특화된 에이전트 기본 클래스"""

    def __init__(self, name: str, role: str, system_prompt: str, aoai_wrapper: AzureOpenAIWrapper):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.aoai_wrapper = aoai_wrapper

    async def process(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """에이전트가 입력을 처리하고 응답 반환"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

        # 컨텍스트가 있으면 추가
        if context:
            context_str = f"\n\n[Context from previous agents]\n{json.dumps(context, indent=2, ensure_ascii=False)}"
            messages[-1]["content"] += context_str

        try:
            response = self.aoai_wrapper.chat_completion(messages, stream=False)
            content = response.choices[0].message.content or ""

            return AgentResponse(
                content=content,
                metadata={"tokens": response.usage.total_tokens if hasattr(response, 'usage') else 0},
                success=True,
                agent_name=self.name
            )
        except ContentFilterError as e:
            return AgentResponse(
                content=f"콘텐츠 필터링 차단: {', '.join(e.filtered_categories)}",
                metadata={"error": str(e)},
                success=False,
                agent_name=self.name
            )
        except Exception as e:
            return AgentResponse(
                content=f"에러 발생: {str(e)}",
                metadata={"error": str(e)},
                success=False,
                agent_name=self.name
            )


class QuestionUnderstandingAgent(SpecializedAgent):
    """Agent A: 질문 이해 담당"""

    def __init__(self, aoai_wrapper: AzureOpenAIWrapper):
        system_prompt = """당신은 질문 분석 전문가입니다.
사용자의 질문을 분석하여 다음을 추출합니다:
1. 핵심 키워드
2. 질문 유형 (법률 관련, 판례 검색, 일반 질문)
3. 필요한 후속 에이전트 (legal_agent, precedent_agent, 또는 none)
4. 구조화된 쿼리

응답은 반드시 다음 JSON 형식으로 제공하세요:
{
  "keywords": ["키워드1", "키워드2"],
  "question_type": "legal|precedent|general",
  "next_agent": "legal_agent|precedent_agent|none",
  "structured_query": "재구성된 명확한 질문",
  "analysis": "간단한 분석 설명"
}"""
        super().__init__(
            name="QuestionUnderstandingAgent",
            role="질문 이해 및 분석",
            system_prompt=system_prompt,
            aoai_wrapper=aoai_wrapper
        )


class LegalExpertAgent(SpecializedAgent):
    """Agent B: 법률 전문"""

    def __init__(self, aoai_wrapper: AzureOpenAIWrapper, tools: List[BaseTool]):
        system_prompt = """당신은 법률 전문가입니다.
법률 관련 질문에 대해 정확하고 전문적인 답변을 제공합니다.
가능한 경우 관련 법조문, 법률 용어, 절차 등을 설명합니다.
필요시 제공된 도구를 사용하여 정보를 검색할 수 있습니다."""
        super().__init__(
            name="LegalExpertAgent",
            role="법률 전문 답변",
            system_prompt=system_prompt,
            aoai_wrapper=aoai_wrapper
        )
        self.tools = tools

    async def process_with_tools(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """도구를 사용하여 처리"""
        tools_for_openai = []
        for tool in self.tools:
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

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

        if context:
            context_str = f"\n\n[Context]\n{json.dumps(context, indent=2, ensure_ascii=False)}"
            messages[-1]["content"] += context_str

        # 도구 호출 루프
        max_iterations = 5
        for iteration in range(max_iterations):
            try:
                response = self.aoai_wrapper.chat_completion(messages, tools=tools_for_openai, stream=False)
                choice = response.choices[0].message

                if not getattr(choice, "tool_calls", None):
                    # 도구 호출 없음 - 최종 답변
                    return AgentResponse(
                        content=choice.content or "",
                        metadata={"iterations": iteration + 1},
                        success=True,
                        agent_name=self.name
                    )

                # 도구 호출 처리
                print(f"  🔧 [{self.name}] Tool calls: {len(choice.tool_calls)}", flush=True)
                tool_results = []

                for tc in choice.tool_calls:
                    tool_name = tc.function.name
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}

                    for tool in self.tools:
                        current_tool_name = getattr(tool, 'name', type(tool).__name__)
                        if current_tool_name == tool_name:
                            try:
                                from google.adk.models import LlmRequest
                                class DummyToolContext:
                                    def __init__(self):
                                        self.llm_request = LlmRequest(contents=[])

                                tool_context = DummyToolContext()
                                result = await tool.run_async(args=args, tool_context=tool_context)
                                tool_results.append({
                                    "tool_call_id": tc.id,
                                    "content": str(result),
                                })
                                print(f"    ✅ Tool {tool_name} executed", flush=True)
                                break
                            except Exception as e:
                                tool_results.append({
                                    "tool_call_id": tc.id,
                                    "content": f"Error: {str(e)}",
                                })
                                break

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

            except Exception as e:
                return AgentResponse(
                    content=f"에러 발생: {str(e)}",
                    metadata={"error": str(e), "iteration": iteration},
                    success=False,
                    agent_name=self.name
                )

        return AgentResponse(
            content="최대 반복 횟수 도달",
            metadata={"iterations": max_iterations},
            success=False,
            agent_name=self.name
        )


class PrecedentExpertAgent(SpecializedAgent):
    """Agent C: 판례 전문"""

    def __init__(self, aoai_wrapper: AzureOpenAIWrapper, tools: List[BaseTool]):
        system_prompt = """당신은 판례 검색 및 분석 전문가입니다.
판례 관련 질문에 대해 관련 판례를 검색하고 분석합니다.
판례의 핵심 쟁점, 판결 요지, 적용 법리 등을 명확하게 설명합니다.
필요시 제공된 도구를 사용하여 판례를 검색할 수 있습니다."""
        super().__init__(
            name="PrecedentExpertAgent",
            role="판례 검색 및 분석",
            system_prompt=system_prompt,
            aoai_wrapper=aoai_wrapper
        )
        self.tools = tools

    async def process_with_tools(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """도구를 사용하여 판례 검색 및 분석"""
        # LegalExpertAgent와 동일한 로직 사용
        agent = LegalExpertAgent(self.aoai_wrapper, self.tools)
        agent.name = self.name
        agent.system_prompt = self.system_prompt
        return await agent.process_with_tools(user_input, context)


class MultiAgentOrchestrator:
    """멀티 에이전트 오케스트레이터 - 여러 특화 에이전트를 관리하고 조율"""

    def __init__(self):
        self.servers: Dict[str, MCPServerConnection] = {}
        self.all_tools: List[BaseTool] = []
        self._closing = False
        self.aoai_wrapper: Optional[AzureOpenAIWrapper] = None

        # 특화된 에이전트들
        self.question_agent: Optional[QuestionUnderstandingAgent] = None
        self.legal_agent: Optional[LegalExpertAgent] = None
        self.precedent_agent: Optional[PrecedentExpertAgent] = None

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

    def initialize_agents(self):
        """개별 특화 에이전트들을 초기화"""
        self.aoai_wrapper = AzureOpenAIWrapper(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        )

        # Agent A: 질문 이해 담당
        self.question_agent = QuestionUnderstandingAgent(self.aoai_wrapper)

        # Agent B: 법률 전문 (MCP Server 1 도구 사용)
        legal_tools = [tool for tool in self.all_tools if 'mcp1' in getattr(tool, 'name', '')]
        self.legal_agent = LegalExpertAgent(self.aoai_wrapper, legal_tools if legal_tools else self.all_tools)

        # Agent C: 판례 전문 (MCP Server 2 도구 사용)
        precedent_tools = [tool for tool in self.all_tools if 'mcp2' in getattr(tool, 'name', '')]
        self.precedent_agent = PrecedentExpertAgent(self.aoai_wrapper, precedent_tools if precedent_tools else self.all_tools)

        print(f"✅ 에이전트 초기화 완료:")
        print(f"   - {self.question_agent.name}: {self.question_agent.role}")
        print(f"   - {self.legal_agent.name}: {self.legal_agent.role} (도구 {len(self.legal_agent.tools)}개)")
        print(f"   - {self.precedent_agent.name}: {self.precedent_agent.role} (도구 {len(self.precedent_agent.tools)}개)")

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


async def initialize_multi_agent() -> MultiAgentOrchestrator:
    """멀티 에이전트 시스템 초기화"""
    orchestrator = MultiAgentOrchestrator()

    mcp1_connected = False
    mcp2_connected = False

    try:
        mcp1_url = os.environ.get("MCP_SERVER_1_URL", "")
        mcp1_bearer = os.environ.get("MCP_SERVER_1_AUTH_BEARER", "")
        if mcp1_url:
            print(f"Connecting to MCP Server 1 (법률 도구): {mcp1_url}")
            try:
                await orchestrator.connect_mcp_server("mcp1", mcp1_url, mcp1_bearer)
                print("✓ Connected to MCP Server 1")
                mcp1_connected = True
            except Exception as e:
                print(f"✗ Failed to connect to MCP Server 1: {e}")
        else:
            raise RuntimeError("MCP_SERVER_1_URL 환경 변수가 필요합니다")

        mcp2_url = os.environ.get("MCP_SERVER_2_URL", "")
        mcp2_bearer = os.environ.get("MCP_SERVER_2_AUTH_BEARER", "")
        if mcp2_url:
            print(f"Connecting to MCP Server 2 (판례 도구): {mcp2_url}")
            try:
                await orchestrator.connect_mcp_server("mcp2", mcp2_url, mcp2_bearer)
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

        print("\nInitializing Multi-Agent System...")
        orchestrator.initialize_agents()
        print(f"✓ Multi-Agent System initialized with {len(orchestrator.all_tools)} total tools\n")

        return orchestrator

    except Exception as e:
        raise RuntimeError(f"Failed to initialize multi-agent system: {e}") from e


async def run_multi_agent_conversation(orchestrator: MultiAgentOrchestrator, user_query: str) -> str:
    """멀티 에이전트 대화 실행: Agent A → Agent B/C"""

    print(f"\n{'='*60}")
    print(f"🤖 Multi-Agent Processing Pipeline")
    print(f"{'='*60}\n")

    # Step 1: Agent A - 질문 이해
    print(f"📋 [Step 1] Agent A: 질문 분석")
    print(f"─" * 60)

    step1_start = time.time()
    question_response = await orchestrator.question_agent.process(user_query)
    step1_time = time.time() - step1_start

    if not question_response.success:
        print(f"❌ 질문 분석 실패: {question_response.content}")
        return question_response.content

    print(f"✅ 분석 완료 ({step1_time:.2f}초)")
    print(f"\n{question_response.content}\n")

    # JSON 파싱 시도
    try:
        # JSON 추출 (```json ... ``` 형식도 처리)
        content = question_response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        analysis = json.loads(content)
        next_agent = analysis.get("next_agent", "none")
        question_type = analysis.get("question_type", "general")
        structured_query = analysis.get("structured_query", user_query)

        print(f"🎯 판단 결과:")
        print(f"   질문 유형: {question_type}")
        print(f"   다음 에이전트: {next_agent}")
        print(f"   구조화된 쿼리: {structured_query}\n")

    except json.JSONDecodeError:
        print(f"⚠️  JSON 파싱 실패, 기본 처리로 진행\n")
        next_agent = "legal_agent"  # 기본값
        structured_query = user_query

    # Step 2: Agent B 또는 C로 라우팅
    if next_agent == "none" or next_agent not in ["legal_agent", "precedent_agent"]:
        print(f"💬 [Final Answer] 추가 처리 불필요")
        return question_response.content

    # Step 2: 전문 에이전트 처리
    if next_agent == "legal_agent":
        print(f"⚖️  [Step 2] Agent B: 법률 전문가 처리")
        print(f"─" * 60)
        specialist_agent = orchestrator.legal_agent
    else:  # precedent_agent
        print(f"📚 [Step 2] Agent C: 판례 전문가 처리")
        print(f"─" * 60)
        specialist_agent = orchestrator.precedent_agent

    step2_start = time.time()

    # 컨텍스트 전달
    context = {
        "original_query": user_query,
        "analysis": question_response.content,
        "structured_query": structured_query
    }

    final_response = await specialist_agent.process_with_tools(structured_query, context)
    step2_time = time.time() - step2_start

    if not final_response.success:
        print(f"❌ 처리 실패: {final_response.content}")
        return final_response.content

    print(f"\n✅ 처리 완료 ({step2_time:.2f}초)")

    # Step 3: 최종 응답 스트리밍
    print(f"\n💬 [Final Answer] ")
    print(f"─" * 60)

    # 스트리밍으로 최종 답변 출력
    try:
        messages = [
            {"role": "system", "content": "이전 에이전트들의 분석 결과를 바탕으로 사용자에게 최종 답변을 제공하세요. 명확하고 이해하기 쉽게 설명하세요."},
            {"role": "user", "content": f"원본 질문: {user_query}\n\n분석 결과:\n{question_response.content}\n\n전문가 답변:\n{final_response.content}"}
        ]

        stream_start = time.time()
        stream_response = orchestrator.aoai_wrapper.chat_completion(messages, stream=True)

        collected_content = ""
        for chunk in stream_response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    print(delta.content, end="", flush=True)
                    collected_content += delta.content

        stream_time = time.time() - stream_start
        print(f"\n\n⏱️  스트리밍 시간: {stream_time:.2f}초")

        return collected_content

    except ContentFilterError as e:
        print(f"\n🚫 콘텐츠 필터링 차단: {', '.join(e.filtered_categories)}")
        return final_response.content
    except Exception as e:
        print(f"\n⚠️  스트리밍 실패, 원본 응답 반환: {e}")
        return final_response.content
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

