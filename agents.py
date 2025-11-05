# -*- coding: utf-8 -*-
"""
Agent Classes Module
다양한 에이전트 클래스 정의
"""

import os
import json
from typing import Any, Dict, List, Optional

from llm_client import LLMClient, ContentFilterError
from google.adk.tools import BaseTool


class AgentResponse:
    """에이전트 응답"""
    def __init__(self, content: str, metadata: Dict[str, Any], success: bool, agent_name: str):
        self.content = content
        self.metadata = metadata
        self.success = success
        self.agent_name = agent_name


class SpecializedAgent:
    """특화된 에이전트 기본 클래스"""

    def __init__(self, name: str, role: str, system_prompt: str, llm_client: LLMClient):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm_client = llm_client

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
            response = self.llm_client.chat_completion(messages, stream=False)
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
    """Agent A: 질문 이해 및 라우팅 담당 (A2A 1단계)"""

    def __init__(self, llm_client: LLMClient, available_agents: List[str], agent_descriptions: Dict[str, str]):
        """
        Args:
            llm_client: LLM 클라이언트
            available_agents: 사용 가능한 에이전트 목록
            agent_descriptions: 각 에이전트의 역할 설명 {"laws": "법률 검색 전문...", ...}
        """
        agents_str = ", ".join(available_agents)

        # 각 에이전트의 역할 설명을 문자열로 포맷
        agents_info_str = ""
        for agent, description in agent_descriptions.items():
            agents_info_str += f"\n  - **{agent}**: {description}"

        system_prompt = f"""당신은 Multi-Agent 시스템의 오케스트레이터입니다.
사용자의 질문을 분석하여 적절한 전문 에이전트에게 작업을 위임합니다.

**🤖 사용 가능한 에이전트:**{agents_info_str}

**⚠️ A2A (Agent-to-Agent) 1단계 원칙:**
1. 당신은 **에이전트의 역할과 설명만** 보고 라우팅합니다
2. 각 에이전트가 **어떤 도구를 가지고 있는지는 알 필요가 없습니다**
3. 각 에이전트는 **자율적으로** 자신의 도구를 선택합니다

**🎯 핵심 규칙: 가능하면 병렬 실행하세요!**

**실행 전략 결정 가이드:**

🔀 **병렬 실행 (PARALLEL) - 우선 고려!**
- 형식: execution_order: [["agent1", "agent2"]]
- 조건: 두 에이전트의 작업이 **서로 독립적**일 때
- 판단: agent2가 agent1의 결과를 **꼭 봐야 하나?** → NO면 병렬!
- 예시:
  * "12대 중과실이 뭐야?" → [["laws", "search"]] ✅
  * "근로기준법 설명하고 관련 뉴스 찾아줘" → [["laws", "search"]] ✅
  * "법률 정의와 판례 알려줘" → [["laws", "precedent"]] ✅

⏭️ **순차 실행 (SEQUENTIAL) - 의존성이 명확할 때만!**
- 형식: execution_order: [["agent1"], ["agent2"]]
- 조건: agent2가 agent1의 **구체적 결과를 반드시 필요**로 할 때
- 판단: "~를 찾고", "~한 후", "~를 바탕으로" 같은 **명시적 순서**가 있을 때
- **중요:** agent2의 query는 "이전 결과를 참고하여..." 형태로 작성
  * ❌ "앞서 찾은 위반 사례에 해당되는 법 조항..." (모호함)
  * ✅ "이전 에이전트가 찾은 근로기준법 위반 사례들을 분석하여, 각 사례에 해당하는 법 조항을 찾아주세요" (명확함)
- 예시:
  * "최근 판례를 찾고, 그 판례의 법률 조항을 알려줘" → [["precedent"], ["laws"]] ✅
    queries: {{
      "precedent": "최근 근로기준법 관련 판례를 찾아주세요",
      "laws": "이전에 찾은 판례에서 언급된 법률 조항들을 상세히 설명해주세요"
    }}
  * "이 사건의 관련 법률을 먼저 찾고, 그 법률의 위반 사례를 찾아줘" → [["laws"], ["search"]] ✅
    queries: {{
      "laws": "교통사고 관련 법률 조항을 찾아주세요",
      "search": "이전에 찾은 법률 조항들의 위반 사례를 검색해주세요"
    }}

💬 **단일 실행 (SINGLE)**
- 형식: execution_order: [["agent1"]]
- 조건: 한 에이전트만 명확히 필요할 때

❌ **일반 대화**
- 형식: execution_order: []
- 조건: 전문 지식이 필요 없는 일반 대화

**⚠️ 매우 중요 - execution_order 작성 규칙:**
1. execution_order에는 **반드시 에이전트 이름만** 사용
2. 사용 가능한 에이전트 이름: {agents_str}

**응답 형식 (JSON):**
{{
  "question_type": "parallel|sequential|single|general",
  "execution_order": [["agent1", "agent2"]] or [["agent1"], ["agent2"]] or [["agent1"]] or [],
  "queries": {{
    "agent1": "구체적인 질문",
    "agent2": "구체적인 질문"
  }},
  "dependencies": {{
    "agent_name": "의존성 설명 (선택사항)"
  }},
  "analysis": "질문 분석 및 실행 순서 결정 이유"
}}"""

        super().__init__(
            name="QuestionUnderstandingAgent",
            role="질문 이해 및 라우팅 (A2A 1단계)",
            system_prompt=system_prompt,
            llm_client=llm_client
        )


class ToolBasedAgent(SpecializedAgent):
    """도구 기반 전문 에이전트 (A2A Phase 2 지원)"""

    def __init__(self, name: str, role: str, llm_client: LLMClient, tools: List[BaseTool],
                 description: str = "", orchestrator=None):
        # 도구 정보 수집
        tools_info = []
        for tool in tools:
            tool_name = getattr(tool, 'name', type(tool).__name__)
            tool_input_schema = getattr(tool, 'input_schema', None)
            if tool_input_schema and 'properties' in tool_input_schema:
                params = list(tool_input_schema['properties'].keys())
                required = tool_input_schema.get('required', [])
                params_str = ', '.join([f"{p}{'*' if p in required else ''}" for p in params])
                tools_info.append(f"  - {tool_name}({params_str})")

        tools_detail = "\n".join(tools_info) if tools_info else "(도구 정보 없음)"

        system_prompt = f"""당신은 {role} 전문가입니다.

**역할:** {description}

**A2A Phase 2: 에이전트 간 협력**
당신은 자율적으로 다음을 결정할 수 있습니다:
1. **도구 사용**: 자신이 가진 도구를 사용하여 직접 작업 수행
2. **에이전트 협력**: 다른 에이전트의 도움이 필요하면 request_agent_help 도구를 사용

**중요: 도구 사용 시 반드시 아래의 정확한 파라미터 이름을 사용하세요!**

사용 가능한 도구: {len(tools)}개
{tools_detail}
(*표시는 필수 파라미터)

**협력 원칙:**
- 자신의 전문 영역 내 작업은 자신의 도구로 해결
- 다른 전문 영역의 작업이 필요하면 다른 에이전트에게 협력 요청
- request_agent_help 도구를 사용하여 다른 에이전트에게 작업 위임

**도구 파라미터 규칙:**
1. 도구 호출 시 스키마에 정의된 **정확한 파라미터 이름**을 사용하세요
2. 'keyword' 대신 'query', 'search_text' 등 실제 정의된 이름 사용
3. 필수 파라미터(*)는 반드시 포함"""

        super().__init__(
            name=name,
            role=role,
            system_prompt=system_prompt,
            llm_client=llm_client
        )
        self.tools = tools
        self.description = description
        self.orchestrator = orchestrator  # A2A Phase 2를 위한 오케스트레이터 참조

    def set_orchestrator(self, orchestrator):
        """오케스트레이터 설정 (A2A Phase 2)"""
        self.orchestrator = orchestrator

    async def process_with_tools(self, user_input: str, context: Optional[Dict[str, Any]] = None):
        """도구를 사용하여 처리 (A2A Phase 2: 에이전트 간 협력 지원)"""
        tools_for_openai = []

        print(f"\n  📋 [{self.name}] 사용 가능한 도구 Schema:", flush=True)

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
        
        # 🤝 A2A Phase 2: 다른 에이전트 협력 요청 도구 추가
        if self.orchestrator:
            available_agents = [name for name in self.orchestrator.specialist_agents.keys()
                              if name != self.name.replace("Agent", "").lower()]
            
            if available_agents:
                # 각 에이전트의 설명 수집
                agents_info = []
                for agent_name in available_agents:
                    agent = self.orchestrator.specialist_agents.get(agent_name)
                    if agent and hasattr(agent, 'description'):
                        agents_info.append(f"  - {agent_name}: {agent.description}")
                
                agents_info_str = "\n".join(agents_info) if agents_info else "\n".join([f"  - {name}" for name in available_agents])
                
                # 에이전트 협력 요청 도구 추가
                tools_for_openai.append({
                    "type": "function",
                    "function": {
                        "name": "request_agent_help",
                        "description": f"""다른 전문 에이전트에게 협력을 요청합니다.
자신의 전문 영역이 아닌 작업이 필요할 때 사용하세요.

사용 가능한 에이전트:
{agents_info_str}

예시: 법률 에이전트가 최신 뉴스를 검색해야 할 때 search 에이전트에게 요청""",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "target_agent": {
                                    "type": "string",
                                    "enum": available_agents,
                                    "description": "협력을 요청할 에이전트 이름"
                                },
                                "task": {
                                    "type": "string",
                                    "description": "요청할 작업 내용 (구체적으로)"
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "협력이 필요한 이유"
                                }
                            },
                            "required": ["target_agent", "task"]
                        },
                    },
                })
                print(f"    🤝 A2A Phase 2: request_agent_help 도구 활성화", flush=True)
                print(f"       협력 가능 에이전트: {', '.join(available_agents)}", flush=True)

        print(f"", flush=True)

        # 이전 에이전트 결과를 명확하게 포함
        enhanced_input = user_input

        if context:
            print(f"    🔍 Context keys: {list(context.keys())}", flush=True)

            if "previous_agent_results" in context:
                previous_results = context["previous_agent_results"]
                print(f"    📦 Previous results count: {len(previous_results)}", flush=True)

                if previous_results:
                    # 이전 결과를 읽기 쉬운 형태로 변환
                    previous_info = "\n\n**🔍 이전 에이전트가 찾은 정보:**\n"
                    for idx, result in enumerate(previous_results):
                        agent_name = result.get("agent", "Unknown")
                        response_content = result.get("response", "")
                        previous_info += f"\n[{agent_name}의 답변]\n{response_content}\n"

                        # 이전 결과 내용 미리보기 (첫 100자)
                        preview = response_content[:100].replace('\n', ' ')
                        if len(response_content) > 100:
                            preview += "..."
                        print(f"    📄 Previous Result {idx+1}: [{agent_name}] {preview}", flush=True)

                    # 의존성 지시사항이 있으면 추가
                    if "dependency_instruction" in context:
                        dependency = context["dependency_instruction"]
                        previous_info += f"\n**⚠️ 중요:** {dependency}\n"
                        previous_info += "위의 정보를 반드시 참고하여 답변하세요.\n"
                        print(f"    ⚠️  Dependency: {dependency}", flush=True)

                    enhanced_input = previous_info + "\n" + "─"*70 + f"\n\n**📌 현재 질문:**\n{user_input}"
                    print(f"    ✅ Enhanced input prepared ({len(enhanced_input)} chars)", flush=True)
                else:
                    print(f"    ⚠️  Previous results is empty", flush=True)
            else:
                print(f"    ⚠️  No previous_agent_results in context", flush=True)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": enhanced_input}
        ]

        max_iterations = int(os.environ.get("MAX_ITERATIONS", 10))
        for iteration in range(max_iterations):
            try:
                response = self.llm_client.chat_completion(messages, tools=tools_for_openai, stream=False)
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

                    # Tool 호출 input 출력
                    print(f"    🔍 [{tool_name}] Input arguments:", flush=True)
                    try:
                        args_str = json.dumps(args, indent=6, ensure_ascii=False)
                        print(f"{args_str}", flush=True)
                    except:
                        print(f"      {args}", flush=True)

                    # 🤝 A2A Phase 2: request_agent_help 처리
                    if tool_name == "request_agent_help":
                        target_agent_name = args.get("target_agent", "")
                        task = args.get("task", "")
                        reason = args.get("reason", "")
                        
                        print(f"    🤝 [{self.name}] → [{target_agent_name}] 협력 요청", flush=True)
                        print(f"       작업: {task}", flush=True)
                        if reason:
                            print(f"       이유: {reason}", flush=True)
                        
                        if self.orchestrator and target_agent_name in self.orchestrator.specialist_agents:
                            target_agent = self.orchestrator.specialist_agents[target_agent_name]
                            try:
                                # 다른 에이전트에게 작업 위임
                                agent_response = await target_agent.process_with_tools(task, context)
                                
                                if agent_response.success:
                                    tool_results.append({
                                        "tool_call_id": tc.id,
                                        "content": f"[{target_agent_name} 에이전트 응답]\n{agent_response.content}",
                                    })
                                    print(f"    ✅ [{target_agent_name}] 협력 완료", flush=True)
                                else:
                                    tool_results.append({
                                        "tool_call_id": tc.id,
                                        "content": f"Error: {target_agent_name} 에이전트 협력 실패 - {agent_response.content}",
                                    })
                                    print(f"    ❌ [{target_agent_name}] 협력 실패", flush=True)
                            except Exception as e:
                                tool_results.append({
                                    "tool_call_id": tc.id,
                                    "content": f"Error: 협력 요청 실패 - {str(e)}",
                                })
                                print(f"    ❌ 협력 요청 실패: {str(e)}", flush=True)
                        else:
                            tool_results.append({
                                "tool_call_id": tc.id,
                                "content": f"Error: Agent '{target_agent_name}' not found",
                            })
                            print(f"    ⚠️  Agent '{target_agent_name}' not found", flush=True)
                        continue

                    tool_found = False
                    for tool in self.tools:
                        current_tool_name = getattr(tool, 'name', type(tool).__name__)
                        if current_tool_name == tool_name:
                            tool_found = True
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

                                result_str = str(result)
                                result_preview = result_str[:500].replace('\n', ' ')
                                if len(result_str) > 500:
                                    result_preview += "..."
                                print(f"    📄 Result preview: {result_preview}", flush=True)
                                print(f"    ✅ Tool {tool_name} executed", flush=True)
                                break
                            except Exception as e:
                                print(f"    ❌ Tool {tool_name} failed: {str(e)}", flush=True)
                                tool_results.append({
                                    "tool_call_id": tc.id,
                                    "content": f"Error: {str(e)}",
                                })
                                break

                    # 도구를 찾지 못한 경우에도 결과 추가
                    if not tool_found:
                        print(f"    ⚠️  Tool '{tool_name}' not found", flush=True)
                        tool_results.append({
                            "tool_call_id": tc.id,
                            "content": f"Error: Tool '{tool_name}' not found",
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

