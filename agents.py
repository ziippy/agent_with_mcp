# -*- coding: utf-8 -*-
"""
Agent Classes Module
다양한 에이전트 클래스 정의
"""

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
    """Agent A: 질문 이해 및 라우팅 담당"""

    def __init__(self, llm_client: LLMClient, available_agents: List[str], agent_tools_info: Dict[str, List[str]]):
        agents_str = ", ".join(available_agents)

        # 각 에이전트의 도구 정보를 문자열로 포맷
        tools_info_str = ""
        for agent, tools in agent_tools_info.items():
            tools_list = ", ".join(tools[:5])
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

**병렬 vs 순차 실행 판단:**
- 병렬: 두 작업이 독립적 → execution_order: [["agent1", "agent2"]]
- 순차: 두 번째가 첫 번째 결과 필요 → execution_order: [["agent1"], ["agent2"]]

**응답 형식 (JSON):**
{{
  "keywords": ["키워드1", "키워드2"],
  "question_type": "single|multiple|parallel|general",
  "execution_order": [["agent_name1"], ["agent_name2"]],
  "queries": {{
    "agent_name1": "구체적인 질문",
    "agent_name2": "구체적인 질문"
  }},
  "dependencies": {{
    "agent_name": "의존성 설명 (선택사항)"
  }},
  "analysis": "질문 분석 및 실행 순서 결정 이유"
}}"""

        super().__init__(
            name="QuestionUnderstandingAgent",
            role="질문 이해 및 라우팅",
            system_prompt=system_prompt,
            llm_client=llm_client
        )


class ToolBasedAgent(SpecializedAgent):
    """도구 기반 전문 에이전트"""

    def __init__(self, name: str, role: str, llm_client: LLMClient, tools: List[BaseTool]):
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

**중요: 도구 사용 시 반드시 아래의 정확한 파라미터 이름을 사용하세요!**

사용 가능한 도구: {len(tools)}개
{tools_detail}
(*표시는 필수 파라미터)

**도구 파라미터 규칙:**
1. 스키마에 정의된 **정확한 파라미터 이름** 사용
2. 'keyword' 대신 'query', 'search_text' 등 실제 정의된 이름 사용
3. 필수 파라미터(*)는 반드시 포함

필요시 제공된 도구를 사용하여 정보를 검색할 수 있습니다."""

        super().__init__(
            name=name,
            role=role,
            system_prompt=system_prompt,
            llm_client=llm_client
        )
        self.tools = tools

    async def process_with_tools(self, user_input: str, context: Optional[Dict[str, Any]] = None):
        """도구를 사용하여 처리"""
        tools_for_openai = []

        print(f"\n  📋 [{self.name}] 사용 가능한 도구 Schema:", flush=True)

        for tool in self.tools:
            tool_name = getattr(tool, 'name', type(tool).__name__)
            tool_description = getattr(tool, 'description', '')
            tool_input_schema = getattr(tool, 'input_schema', None) or {"type": "object", "properties": {}}

            '''
            # Schema 상세 출력
            print(f"    • {tool_name}:", flush=True)
            print(f"      설명: {tool_description}", flush=True)
            if tool_input_schema and 'properties' in tool_input_schema:
                print(f"      파라미터:", flush=True)
                for param_name, param_info in tool_input_schema['properties'].items():
                    param_type = param_info.get('type', 'unknown')
                    param_desc = param_info.get('description', '')
                    param_enum = param_info.get('enum', None)
                    required = param_name in tool_input_schema.get('required', [])
                    req_str = " (필수)" if required else " (선택)"
                    enum_str = f" enum={param_enum}" if param_enum else ""
                    print(f"        - {param_name}: {param_type}{enum_str}{req_str} - {param_desc}", flush=True)
            '''

            tools_for_openai.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description or "",
                    "parameters": tool_input_schema,
                },
            })

        print(f"", flush=True)

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

