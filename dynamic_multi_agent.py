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
    """Agent A: 질문 이해 및 라우팅 담당"""

    def __init__(self, aoai_wrapper: AzureOpenAIWrapper, available_agents: List[str], agent_tools_info: Dict[str, List[str]]):
        """
        Args:
            aoai_wrapper: Azure OpenAI 래퍼
            available_agents: 사용 가능한 에이전트 목록 (예: ["mcp1", "mcp2", "mcp3"])
            agent_tools_info: 각 에이전트의 도구 정보 {"mcp1": ["tool1", "tool2"], ...}
        """
        agents_str = ", ".join(available_agents)

        # 각 에이전트의 도구 정보를 문자열로 포맷
        tools_info_str = ""
        for agent, tools in agent_tools_info.items():
            tools_list = ", ".join(tools[:5])  # 최대 5개만 표시
            if len(tools) > 5:
                tools_list += f"... (총 {len(tools)}개)"
            tools_info_str += f"\n  - {agent}: {tools_list}"

        system_prompt = f"""당신은 질문 분석 및 라우팅 전문가입니다.
사용자의 질문을 분석하여 적절한 전문 에이전트에게 라우팅합니다.

**사용 가능한 에이전트 목록:**
{agents_str}

**각 에이전트가 제공하는 도구:**{tools_info_str}

**⚠️ 매우 중요 - execution_order 작성 규칙:**
1. execution_order에는 **반드시 에이전트 이름만** 사용하세요
2. 사용 가능한 에이전트 이름: {agents_str}
3. 도구 이름을 절대 사용하지 마세요

**에이전트 역할 추론 (도구를 보고 판단):**
각 에이전트가 제공하는 도구를 보고 어떤 역할을 하는지 추론한 후, **에이전트 이름**을 사용하세요.

**잘못된 예시 (절대 사용 금지):**
❌ execution_order: [["law-search", "precedent-search"]]  <- 도구 이름 사용
❌ execution_order: [["web-search"]]  <- 도구 이름 사용

**올바른 예시:**
✅ execution_order: [["{agents_str.split(', ')[0] if agents_str else 'mcp1'}", "{agents_str.split(', ')[1] if ', ' in agents_str else 'mcp2'}"]]  <- 에이전트 이름 사용
✅ execution_order: [["{agents_str.split(', ')[0] if agents_str else 'mcp1'}"]]  <- 에이전트 이름 사용

**중요 원칙:**
1. 질문이 여러 에이전트를 필요로 하면 **실행 순서**를 논리적으로 결정
2. **병렬 실행이 유리한 경우 여러 에이전트를 동시에 실행**할 수 있습니다
3. **나중 에이전트가 이전 에이전트 결과를 활용해야 하면 의존성을 명시하여 순차 실행**
4. 일반적인 대화는 에이전트를 호출하지 않음 (execution_order: [])

**병렬 vs 순차 실행 판단 기준:**

🔀 **병렬 실행 (동시에 독립적으로 검색):**
- 두 에이전트의 작업이 **서로 독립적**일 때
- 한 에이전트의 결과가 다른 에이전트의 입력으로 필요하지 않을 때
- 예: "12대 중과실은 뭐야?" → 법률 조문과 판례를 동시에 검색 가능
  - execution_order: [["agent1", "agent2"]]
  - 두 검색이 서로 독립적임

⏭️ **순차 실행 (첫 번째 결과를 두 번째가 활용):**
- 나중 에이전트가 **이전 에이전트의 결과를 참고**해야 할 때
- "찾아보고 ~해줘", "검색한 후 ~해줘", "바탕으로 ~해줘" 같은 표현이 있을 때
- 예: "최근 근로기준법 위반 사례를 찾아보고, 해당 법 조항을 알려줘"
  - execution_order: [["agent1"], ["agent2"]]
  - agent2는 agent1의 결과(위반 사례)를 보고 관련 법 조항을 검색해야 함
  - dependencies: {{"agent2": "agent1에서 찾은 위반 사례를 분석하여 해당 법 조항 검색"}}

**응답 형식 (JSON):**
{{
  "keywords": ["키워드1", "키워드2"],
  "question_type": "single|multiple|parallel|general",
  "execution_order": [["agent_name1"], ["agent_name2", "agent_name3"]],
  "queries": {{
    "agent_name1": "해당 에이전트에게 할 구체적인 질문",
    "agent_name2": "해당 에이전트에게 할 구체적인 질문"
  }},
  "dependencies": {{
    "agent_name": "이전 에이전트 결과 활용 방법 (선택사항)"
  }},
  "analysis": "질문 분석 및 실행 순서(순차/병렬) 결정 이유"
}}

**execution_order 작성 규칙:**
⚠️ CRITICAL: execution_order에는 반드시 실제 에이전트 이름만 사용하세요!
사용 가능: {agents_str}
사용 금지: 도구 이름 (law-search, precedent-search 등)

**형식 예시:**
- `[["{available_agents[0] if available_agents else 'mcp1'}"]]`: 단일 에이전트 실행
- `[["{available_agents[0] if available_agents else 'mcp1'}"], ["{available_agents[1] if len(available_agents) > 1 else 'mcp2'}"]]`: 순차 실행
- `[["{available_agents[0] if available_agents else 'mcp1'}", "{available_agents[1] if len(available_agents) > 1 else 'mcp2'}"]]`: 병렬 실행
- `[]`: 일반 대화 (에이전트 미호출)

**구체적인 예시 시나리오:**

1️⃣ **병렬 실행 예시:**
질문: "12대 중과실이 뭐야?"
분석: 법률 조문과 판례를 동시에 검색 가능 (독립적)
```json
{{
  "question_type": "parallel",
  "execution_order": [["{available_agents[0] if available_agents else 'mcp1'}", "{available_agents[1] if len(available_agents) > 1 else 'mcp2'}"]],
  "queries": {{
    "{available_agents[0] if available_agents else 'mcp1'}": "12대 중과실의 법률적 정의를 검색",
    "{available_agents[1] if len(available_agents) > 1 else 'mcp2'}": "12대 중과실 관련 판례를 검색"
  }},
  "dependencies": {{}},
  "analysis": "법률 조문과 판례는 독립적으로 검색 가능하므로 병렬 실행"
}}
```

2️⃣ **순차 실행 예시 (의존성 있음):**
질문: "최근 근로기준법 위반 사례를 찾아보고, 해당 법 조항을 알려줘"
분석: 첫 번째로 사례 검색 → 그 결과를 바탕으로 법 조항 검색 (의존적)
```json
{{
  "question_type": "multiple",
  "execution_order": [["{available_agents[0] if available_agents else 'mcp1'}"], ["{available_agents[1] if len(available_agents) > 1 else 'mcp2'}"]],
  "queries": {{
    "{available_agents[0] if available_agents else 'mcp1'}": "최근 근로기준법 위반 사례를 검색",
    "{available_agents[1] if len(available_agents) > 1 else 'mcp2'}": "위반 사례에 해당하는 근로기준법 조항을 검색"
  }},
  "dependencies": {{
    "{available_agents[1] if len(available_agents) > 1 else 'mcp2'}": "첫 번째 에이전트가 찾은 위반 사례를 분석하여 해당되는 법 조항 검색"
  }},
  "analysis": "사례를 먼저 찾은 후, 그 사례에 해당하는 법 조항을 검색해야 하므로 순차 실행"
}}
```"""
        super().__init__(
            name="QuestionUnderstandingAgent",
            role="질문 이해 및 라우팅",
            system_prompt=system_prompt,
            aoai_wrapper=aoai_wrapper
        )


class ToolBasedAgent(SpecializedAgent):
    """도구 기반 전문 에이전트 (범용)"""

    def __init__(self, name: str, role: str, aoai_wrapper: AzureOpenAIWrapper, tools: List[BaseTool]):
        system_prompt = f"""당신은 {role} 전문가입니다.
사용자의 질문에 대해 정확하고 전문적인 답변을 제공합니다.
필요시 제공된 도구를 사용하여 정보를 검색할 수 있습니다.

사용 가능한 도구: {len(tools)}개"""
        super().__init__(
            name=name,
            role=role,
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

        max_iterations = 10
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


class MultiAgentOrchestrator:
    """멀티 에이전트 오케스트레이터 - 동적 에이전트 관리"""

    def __init__(self):
        self.servers: Dict[str, MCPServerConnection] = {}
        self.all_tools: List[BaseTool] = []
        self.aoai_wrapper: Optional[AzureOpenAIWrapper] = None

        self.question_agent: Optional[QuestionUnderstandingAgent] = None
        self.specialist_agents: Dict[str, ToolBasedAgent] = {}  # MCP 서버별 에이전트

    def _normalize_url(self, url: str) -> str:
        """URL 정규화"""
        return url if url.endswith("/") else url + "/"

    def _prefix_tool(self, tool: BaseTool, prefix: str) -> BaseTool:
        """도구 이름에 prefix 추가"""
        class PrefixedTool(tool.__class__):
            @property
            def name(self):
                original = getattr(super(), "name", getattr(tool, "name", type(tool).__name__))
                return f"{prefix}__{original}"

        wrapped = PrefixedTool.__new__(PrefixedTool)
        wrapped.__dict__ = tool.__dict__.copy()
        return wrapped

    async def connect_mcp_server(self, server_name: str, base_url: str, auth_bearer: str = "") -> MCPServerConnection:
        """MCP 서버 연결 및 에이전트 자동 생성"""
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

            # 도구에 prefix 추가
            prefixed_tools = []
            for tool in tools:
                prefixed_tool = self._prefix_tool(tool, server_name)
                prefixed_tools.append(prefixed_tool)
                self.all_tools.append(prefixed_tool)

            # 에이전트 자동 생성 (초기화 후)
            # initialize_agents에서 처리

            return connection
        except Exception as e:
            error_msg = str(e)
            print(f"\n[Error] Failed to connect to {server_name}: {error_msg}")
            print(f" URL: {base_url}")
            if server_name in self.servers:
                del self.servers[server_name]
            raise RuntimeError(f"Failed to connect to MCP server {server_name} at {base_url}: {error_msg}") from e

    def initialize_agents(self):
        """개별 특화 에이전트들을 동적으로 초기화"""
        self.aoai_wrapper = AzureOpenAIWrapper(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        )

        # 각 서버별 도구 정보 수집
        agent_tools_info = {}
        for server_name in self.servers.keys():
            server_tools = [tool for tool in self.all_tools if getattr(tool, 'name', '').startswith(f'{server_name}__')]
            tool_names = [getattr(tool, 'name', '').replace(f'{server_name}__', '') for tool in server_tools]
            agent_tools_info[server_name] = tool_names

        # Agent A 초기화 (라우팅 에이전트) - 도구 정보 포함
        available_agents = list(self.servers.keys())
        self.question_agent = QuestionUnderstandingAgent(self.aoai_wrapper, available_agents, agent_tools_info)

        # 각 MCP 서버별로 에이전트 자동 생성
        print(f"\n✅ 에이전트 초기화 완료:")
        print(f"   • {self.question_agent.name}: {self.question_agent.role}")

        for server_name in self.servers.keys():
            # 해당 서버의 도구만 필터링
            server_tools = [tool for tool in self.all_tools if getattr(tool, 'name', '').startswith(f'{server_name}__')]

            # 에이전트 생성
            agent = ToolBasedAgent(
                name=f"{server_name.upper()}Agent",
                role=f"{server_name} 전문 서비스",
                aoai_wrapper=self.aoai_wrapper,
                tools=server_tools
            )
            self.specialist_agents[server_name] = agent
            print(f"   • {agent.name}: {agent.role} (도구 {len(server_tools)}개)")
            # 도구 목록 출력
            for tool in server_tools:
                tool_name = getattr(tool, 'name', '')
                print(f"      - {tool_name}")
                # tool_description = getattr(tool, 'description', '')
                # print(f"      - {tool_name} {tool_description}")

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
        self.specialist_agents.clear()


async def initialize_multi_agent() -> MultiAgentOrchestrator:
    """멀티 에이전트 시스템 초기화 - 동적 MCP 서버 연결"""
    orchestrator = MultiAgentOrchestrator()
    connected_servers = []

    try:
        # 환경 변수에서 MCP 서버 목록 동적 로드
        server_index = 1
        while True:
            url_key = f"MCP_SERVER_{server_index}_URL"
            bearer_key = f"MCP_SERVER_{server_index}_AUTH_BEARER"
            name_key = f"MCP_SERVER_{server_index}_NAME"

            server_url = os.environ.get(url_key, "")
            if not server_url:
                break

            server_bearer = os.environ.get(bearer_key, "")
            server_name = os.environ.get(name_key, f"mcp{server_index}")

            print(f"Connecting to MCP Server '{server_name}': {server_url}")
            try:
                await orchestrator.connect_mcp_server(server_name, server_url, server_bearer)
                print(f"✓ Connected to {server_name}")
                connected_servers.append(server_name)
            except Exception as e:
                print(f"✗ Failed to connect to {server_name}: {e}")

            server_index += 1

        if not connected_servers:
            raise RuntimeError("No MCP servers connected. Check your .env configuration.")

        print(f"\n✅ Connected to {len(connected_servers)} MCP server(s): {', '.join(connected_servers)}")
        print("\nInitializing Multi-Agent System...")
        orchestrator.initialize_agents()
        print(f"✓ Multi-Agent System initialized with {len(orchestrator.all_tools)} total tools\n")

        return orchestrator

    except Exception as e:
        raise RuntimeError(f"Failed to initialize multi-agent system: {e}") from e


async def run_multi_agent_conversation(orchestrator: MultiAgentOrchestrator, user_query: str) -> str:
    """멀티 에이전트 대화 실행"""

    print(f"\n{'='*70}")
    print(f"🤖 Multi-Agent Processing Pipeline")
    print(f"{'='*70}\n")

    # Step 1: Agent A - 질문 분석
    print(f"📋 [Step 1] Agent A: 질문 분석 및 라우팅")
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
        execution_order = analysis.get("execution_order", [])
        question_type = analysis.get("question_type", "general")
        queries = analysis.get("queries", {})
        dependencies = analysis.get("dependencies", {})

        print(f"🎯 판단 결과:")
        print(f"   질문 유형: {question_type}")

        # execution_order를 보기 좋게 출력 (병렬은 괄호로, 순차는 화살표로)
        if execution_order:
            execution_plan_str = " → ".join([
                f"({', '.join(group)})" if len(group) > 1 else group[0]
                for group in execution_order
            ])
            print(f"   실행 순서: {execution_plan_str}")
        else:
            print(f"   실행 순서: none")

        # 각 에이전트별 질문 출력
        if queries:
            flat_order = [agent for group in execution_order for agent in group]
            for agent in flat_order:
                if agent in queries:
                    print(f"   - {agent}: {queries[agent]}")
                    if agent in dependencies:
                        print(f"      └─ 의존성: {dependencies[agent]}")
        print()

    except json.JSONDecodeError:
        print(f"⚠️  JSON 파싱 실패, 기본 처리로 진행\n")
        execution_order = []
        queries = {}
        dependencies = {}

    # Step 2: 전문 에이전트들 순차 실행
    if not execution_order:
        # 일반 질문 - Agent A가 직접 답변
        print(f"💬 [Final Answer] Agent A 직접 답변")
        print(f"{'─'*70}\n")

        try:
            messages = [
                {"role": "system", "content": """당신은 친절한 AI 어시스턴트입니다.
사용자의 질문에 대해 명확하고 친절하게 답변하세요."""},
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

        except Exception as e:
            print(f"\n⚠️  답변 생성 실패: {e}")
            return question_response.content

    agent_results = {}
    previous_results = []
    step_num = 2

    # execution_order를 그룹별로 실행 (각 그룹 내부는 병렬, 그룹 간은 순차)
    for group_idx, agent_group in enumerate(execution_order):
        if not agent_group:
            continue

        # 단일 에이전트 (순차 실행)
        if len(agent_group) == 1:
            agent_name = agent_group[0]
            if agent_name not in orchestrator.specialist_agents:
                print(f"⚠️  에이전트 '{agent_name}' not found, skipping...")
                continue

            specialist_agent = orchestrator.specialist_agents[agent_name]
            print(f"🔧 [Step {step_num}] {specialist_agent.name} 처리 (순차)")
            print(f"{'─'*70}")

            query = queries.get(agent_name, user_query)
            dependency = dependencies.get(agent_name, "")

            print(f"질문: {query}")
            if dependency and previous_results:
                print(f"의존성: {dependency}")
            print()

            step_start = time.time()
            context = {
                "original_query": user_query,
                "analysis": question_response.content,
                "structured_query": query,
            }
            if previous_results:
                context["previous_agent_results"] = previous_results
                if dependency:
                    context["dependency_instruction"] = dependency

            response = await specialist_agent.process_with_tools(query, context)
            step_time = time.time() - step_start

            if response.success:
                print(f"\n✅ 처리 완료 ({step_time:.2f}초)\n")
                result_info = {
                    "agent": specialist_agent.name,
                    "agent_name": agent_name,
                    "query": query,
                    "response": response.content,
                    "time": step_time
                }
                agent_results[agent_name] = result_info
                previous_results.append(result_info)
            else:
                print(f"\n❌ 처리 실패: {response.content}\n")

        # 여러 에이전트 (병렬 실행)
        else:
            print(f"🚀 [Step {step_num}] {len(agent_group)}개 에이전트 병렬 처리: {', '.join(agent_group)}")
            print(f"{'─'*70}")

            tasks = []
            task_agent_names = []

            for agent_name in agent_group:
                if agent_name not in orchestrator.specialist_agents:
                    print(f"⚠️  에이전트 '{agent_name}' not found, skipping...")
                    continue

                specialist_agent = orchestrator.specialist_agents[agent_name]
                query = queries.get(agent_name, user_query)
                dependency = dependencies.get(agent_name, "")

                print(f"  - {specialist_agent.name}")
                print(f"    질문: {query}")
                if dependency and previous_results:
                    print(f"    의존성: {dependency}")

                context = {
                    "original_query": user_query,
                    "analysis": question_response.content,
                    "structured_query": query,
                }
                if previous_results:
                    context["previous_agent_results"] = previous_results
                    if dependency:
                        context["dependency_instruction"] = dependency

                tasks.append(specialist_agent.process_with_tools(query, context))
                task_agent_names.append(agent_name)

            if tasks:
                print()
                step_start = time.time()
                parallel_responses = await asyncio.gather(*tasks)
                step_time = time.time() - step_start

                print(f"✅ 병렬 처리 완료 ({step_time:.2f}초)\n")

                # 병렬 처리 결과를 순서대로 저장
                for agent_name, response in zip(task_agent_names, parallel_responses):
                    if response.success:
                        result_info = {
                            "agent": response.agent_name,
                            "agent_name": agent_name,
                            "query": queries.get(agent_name, user_query),
                            "response": response.content,
                            "time": step_time  # 병렬 실행은 전체 시간 사용
                        }
                        agent_results[agent_name] = result_info
                        previous_results.append(result_info)
                        print(f"  ✓ {response.agent_name}: 성공")
                    else:
                        print(f"  ✗ {response.agent_name}: 실패 - {response.content}")
                print()

        step_num += 1

    if not agent_results:
        return question_response.content

    # Step 3: Agent A - 결과 통합
    print(f"🔄 [Step {step_num}] Agent A: 결과 통합 및 최종 답변 생성")
    print(f"{'─'*70}\n")

    try:
        expert_answers = ""
        # execution_order를 flat하게 만들어서 순서대로 출력
        flat_execution_order = [agent for group in execution_order for agent in group]
        for i, agent_name in enumerate(flat_execution_order, 1):
            if agent_name in agent_results:
                result = agent_results[agent_name]
                expert_answers += f"\n\n[{i}단계: {result['agent']}의 답변]\n질문: {result['query']}\n답변: {result['response']}"

        messages = [
            {"role": "system", "content": """당신은 여러 전문가의 답변을 통합하여 최종 답변을 제공하는 코디네이터입니다.
각 전문가의 답변을 실행 순서대로 종합하여 명확하고 체계적인 답변을 제공하세요."""},
            {"role": "user", "content": f"""원본 질문: {user_query}

전문가 답변들 (실행 순서):
{expert_answers}

위 내용을 바탕으로 최종 답변을 제공해주세요."""}
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

    except Exception as e:
        print(f"\n⚠️  통합 실패: {e}")
        return "\n\n".join([f"[{r['agent']}]\n{r['response']}" for r in agent_results.values()])


async def main():
    print("="*70)
    print("🤖 Dynamic Multi-Agent System")
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
            except Exception as e:
                print(f"⚠️  Error during cleanup: {e}")


if __name__ == "__main__":
    asyncio.run(main())

