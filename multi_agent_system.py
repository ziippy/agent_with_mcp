import os
import json
import asyncio
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AzureOpenAI, BadRequestError

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
    """Azure OpenAI 래퍼"""

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

    def chat_completion(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, stream: bool = False):
        """Azure OpenAI Chat Completion 호출"""
        try:
            # tools가 있을 때만 tool_choice 설정
            kwargs = {
                "model": self.deployment,
                "messages": messages,
                "temperature": 0.2,
                "stream": stream,
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            return self.client.chat.completions.create(**kwargs)
        except BadRequestError as e:
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

                raise ContentFilterError(
                    filtered_categories=filtered_categories,
                    original_error=e
                )

            raise


@dataclass
class MCPServerConnection:
    """단일 MCP 서버 연결"""
    name: str
    toolset: McpToolset


@dataclass
class AgentResponse:
    """에이전트 응답"""
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
        """에이전트 처리"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]

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
2. 질문 유형 (법률+판례 복합, 법률만, 판례만, 일반)
3. 필요한 후속 에이전트들 (배열로 복수 선택 가능)
4. 각 에이전트별 구조화된 쿼리

응답은 반드시 다음 JSON 형식으로 제공하세요:
{
  "keywords": ["키워드1", "키워드2"],
  "question_type": "legal_and_precedent|legal_only|precedent_only|general",
  "next_agents": ["legal_agent", "precedent_agent"] or ["legal_agent"] or ["precedent_agent"] or [],
  "queries": {
    "legal_agent": "법률 에이전트에게 할 질문 (해당되는 경우)",
    "precedent_agent": "판례 에이전트에게 할 질문 (해당되는 경우)"
  },
  "analysis": "간단한 분석 설명"
}

예시:
- "중대재해처벌법 + 최근 사례" → next_agents: ["legal_agent", "precedent_agent"]
- "계약법 조항" → next_agents: ["legal_agent"]
- "부당해고 판례" → next_agents: ["precedent_agent"]"""
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

        max_iterations = 5
        for iteration in range(max_iterations):
            try:
                response = self.aoai_wrapper.chat_completion(messages, tools=tools_for_openai, stream=False)
                choice = response.choices[0].message

                if not getattr(choice, "tool_calls", None):
                    return AgentResponse(
                        content=choice.content or "",
                        metadata={"iterations": iteration + 1},
                        success=True,
                        agent_name=self.name
                    )

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


class PrecedentExpertAgent(LegalExpertAgent):
    """Agent C: 판례 전문"""

    def __init__(self, aoai_wrapper: AzureOpenAIWrapper, tools: List[BaseTool]):
        system_prompt = """당신은 판례 검색 및 분석 전문가입니다.
판례 관련 질문에 대해 관련 판례를 검색하고 분석합니다.
판례의 핵심 쟁점, 판결 요지, 적용 법리 등을 명확하게 설명합니다.
필요시 제공된 도구를 사용하여 판례를 검색할 수 있습니다."""
        SpecializedAgent.__init__(
            self,
            name="PrecedentExpertAgent",
            role="판례 검색 및 분석",
            system_prompt=system_prompt,
            aoai_wrapper=aoai_wrapper
        )
        self.tools = tools


class MultiAgentOrchestrator:
    """멀티 에이전트 오케스트레이터"""

    def __init__(self):
        self.servers: Dict[str, MCPServerConnection] = {}
        self.all_tools: List[BaseTool] = []
        self.aoai_wrapper: Optional[AzureOpenAIWrapper] = None

        self.question_agent: Optional[QuestionUnderstandingAgent] = None
        self.legal_agent: Optional[LegalExpertAgent] = None
        self.precedent_agent: Optional[PrecedentExpertAgent] = None

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

        self.question_agent = QuestionUnderstandingAgent(self.aoai_wrapper)

        legal_tools = [tool for tool in self.all_tools if 'mcp1' in getattr(tool, 'name', '')]
        self.legal_agent = LegalExpertAgent(self.aoai_wrapper, legal_tools if legal_tools else self.all_tools)

        precedent_tools = [tool for tool in self.all_tools if 'mcp2' in getattr(tool, 'name', '')]
        self.precedent_agent = PrecedentExpertAgent(self.aoai_wrapper, precedent_tools if precedent_tools else self.all_tools)

        print(f"\n✅ 에이전트 초기화 완료:")
        print(f"   • {self.question_agent.name}: {self.question_agent.role}")
        print(f"   • {self.legal_agent.name}: {self.legal_agent.role} (도구 {len(self.legal_agent.tools)}개)")
        print(f"   • {self.precedent_agent.name}: {self.precedent_agent.role} (도구 {len(self.precedent_agent.tools)}개)")

    async def close_all_servers(self):
        """모든 MCP 서버 연결 종료"""
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

        if not mcp1_connected and not mcp2_connected:
            raise RuntimeError("Both MCP servers failed to connect.")

        print("\nInitializing Multi-Agent System...")
        orchestrator.initialize_agents()
        print(f"✓ Multi-Agent System initialized with {len(orchestrator.all_tools)} total tools\n")

        return orchestrator

    except Exception as e:
        raise RuntimeError(f"Failed to initialize multi-agent system: {e}") from e


async def run_multi_agent_conversation(orchestrator: MultiAgentOrchestrator, user_query: str) -> str:
    """멀티 에이전트 대화 실행: Agent A → Agent B and/or C → Agent A (통합)"""

    print(f"\n{'='*70}")
    print(f"🤖 Multi-Agent Processing Pipeline")
    print(f"{'='*70}\n")

    # Step 1: Agent A - 질문 분석
    print(f"📋 [Step 1] Agent A: 질문 분석")
    print(f"{'─'*70}")

    step1_start = time.time()
    question_response = await orchestrator.question_agent.process(user_query)
    step1_time = time.time() - step1_start

    if not question_response.success:
        print(f"❌ 질문 분석 실패: {question_response.content}")
        return question_response.content

    print(f"✅ 분석 완료 ({step1_time:.2f}초)")
    print(f"\n{question_response.content}\n")

    # JSON 파싱
    try:
        content = question_response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        analysis = json.loads(content)
        next_agents = analysis.get("next_agents", [])
        question_type = analysis.get("question_type", "general")
        queries = analysis.get("queries", {})

        print(f"🎯 판단 결과:")
        print(f"   질문 유형: {question_type}")
        print(f"   호출할 에이전트: {', '.join(next_agents) if next_agents else 'none'}")
        if queries:
            for agent, query in queries.items():
                print(f"   • {agent}: {query}")
        print()

    except json.JSONDecodeError:
        print(f"⚠️  JSON 파싱 실패, 기본 처리로 진행\n")
        next_agents = ["legal_agent"]
        queries = {"legal_agent": user_query}

    # Step 2: 전문 에이전트들 순차 실행
    if not next_agents or len(next_agents) == 0:
        # 일반 질문 - Agent A가 직접 답변 생성
        print(f"💬 [Final Answer] Agent A 직접 답변")
        print(f"{'─'*70}\n")

        try:
            messages = [
                {"role": "system", "content": """당신은 친절한 AI 어시스턴트입니다.
사용자의 질문에 대해 명확하고 친절하게 답변하세요.
법률이나 판례 관련 질문이 아닌 일반적인 대화에 자연스럽게 응답하세요."""},
                {"role": "user", "content": user_query}
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
            print(f"\n\n⏱️  답변 생성 시간: {stream_time:.2f}초")

            return collected_content

        except ContentFilterError as e:
            print(f"\n🚫 콘텐츠 필터링 차단: {', '.join(e.filtered_categories)}")
            return "죄송합니다. 답변이 콘텐츠 정책에 의해 차단되었습니다."
        except Exception as e:
            print(f"\n⚠️  답변 생성 실패: {e}")
            return question_response.content

    agent_results = {}
    step_num = 2

    for agent_name in next_agents:
        if agent_name not in ["legal_agent", "precedent_agent"]:
            continue

        if agent_name == "legal_agent":
            print(f"⚖️  [Step {step_num}] Agent B: 법률 전문가 처리")
            print(f"{'─'*70}")
            specialist_agent = orchestrator.legal_agent
            emoji = "⚖️"
        else:  # precedent_agent
            print(f"📚 [Step {step_num}] Agent C: 판례 전문가 처리")
            print(f"{'─'*70}")
            specialist_agent = orchestrator.precedent_agent
            emoji = "📚"

        query = queries.get(agent_name, user_query)
        print(f"질문: {query}\n")

        step_start = time.time()

        context = {
            "original_query": user_query,
            "analysis": question_response.content,
            "structured_query": query
        }

        response = await specialist_agent.process_with_tools(query, context)
        step_time = time.time() - step_start

        if response.success:
            print(f"\n✅ {emoji} 처리 완료 ({step_time:.2f}초)\n")
            agent_results[agent_name] = {
                "agent": specialist_agent.name,
                "query": query,
                "response": response.content,
                "time": step_time
            }
        else:
            print(f"\n❌ {emoji} 처리 실패: {response.content}\n")
            agent_results[agent_name] = {
                "agent": specialist_agent.name,
                "query": query,
                "response": f"[ERROR] {response.content}",
                "time": step_time
            }

        step_num += 1

    # 결과가 없으면 분석 결과만 반환
    if not agent_results:
        return question_response.content

    # Step 3: Agent A - 결과 통합 및 최종 답변 생성
    print(f"🔄 [Step {step_num}] Agent A: 결과 통합 및 최종 답변 생성")
    print(f"{'─'*70}\n")

    try:
        # 전문가 답변들을 구조화
        expert_answers = ""
        for agent_name, result in agent_results.items():
            expert_answers += f"\n\n[{result['agent']}의 답변]\n질문: {result['query']}\n답변: {result['response']}"

        messages = [
            {"role": "system", "content": """당신은 여러 전문가의 답변을 통합하여 사용자에게 최종 답변을 제공하는 코디네이터입니다.
각 전문가의 답변을 종합하여:
1. 사용자 질문에 대한 명확한 답변
2. 법률 전문가와 판례 전문가의 답변을 논리적으로 연결
3. 이해하기 쉽고 체계적인 설명

다음 구조로 답변하세요:
- 개요
- 법률적 설명 (법률 전문가 답변 기반)
- 실제 사례 (판례 전문가 답변 기반)
- 결론 및 시사점"""},
            {"role": "user", "content": f"""원본 질문: {user_query}

질문 분석:
{question_response.content}

전문가 답변들:
{expert_answers}

위 내용을 바탕으로 사용자에게 최종 답변을 제공해주세요."""}
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
        print(f"\n\n⏱️  통합 답변 생성 시간: {stream_time:.2f}초")

        return collected_content

    except ContentFilterError as e:
        print(f"\n🚫 콘텐츠 필터링 차단: {', '.join(e.filtered_categories)}")
        # 콘텐츠 필터링 시 원본 답변들 반환
        return "\n\n".join([f"[{r['agent']}]\n{r['response']}" for r in agent_results.values()])
    except Exception as e:
        print(f"\n⚠️  통합 실패, 개별 답변 반환: {e}")
        # 에러 시 원본 답변들 반환
        return "\n\n".join([f"[{r['agent']}]\n{r['response']}" for r in agent_results.values()])


async def main():
    print("="*70)
    print("🤖 Multi-Agent System: Question → Legal/Precedent Expert → Answer")
    print("="*70)

    orchestrator = None
    try:
        orchestrator = await initialize_multi_agent()
        print("\n✅ Multi-Agent System is ready!")
        print("Type 'quit' or 'exit' to stop.\n")

        while True:
            q = input("\n🧑 You> ").strip()
            if q.lower() in {"quit", "exit", "q"}:
                break
            if not q:
                continue

            try:
                conversation_start = time.time()
                ans = await run_multi_agent_conversation(orchestrator, q)
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
        if orchestrator:
            print("\n🔌 Closing all MCP server connections...")
            try:
                await orchestrator.close_all_servers()
                print("✓ All connections closed.")
            except asyncio.CancelledError:
                print("⚠️  Connection cleanup was cancelled")
            except Exception as e:
                print(f"⚠️  Error during cleanup: {e}")


if __name__ == "__main__":
    asyncio.run(main())

