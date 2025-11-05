# -*- coding: utf-8 -*-
"""
Multi-Agent Orchestrator Module
멀티 에이전트 시스템 오케스트레이션
"""

import os
import json
import asyncio
import time
from typing import Optional

from llm_client import LLMClient
from agents import QuestionUnderstandingAgent, ToolBasedAgent
from mcp_manager import MCPManager


class MultiAgentOrchestrator:
    """멀티 에이전트 오케스트레이터"""

    def __init__(self):
        self.mcp_manager = MCPManager()
        self.llm_client: Optional[LLMClient] = None
        self.question_agent: Optional[QuestionUnderstandingAgent] = None
        self.specialist_agents = {}

    async def connect_mcp_server(self, server_name: str, base_url: str, auth_bearer: str = ""):
        """MCP 서버 연결"""
        return await self.mcp_manager.connect_mcp_server(server_name, base_url, auth_bearer)

    def initialize_agents(self):
        """에이전트 초기화"""
        # LLM 제공자 선택
        llm_provider = os.environ.get("LLM_PROVIDER", "azure").lower()

        # LLM 클라이언트 초기화
        if llm_provider == "azure":
            self.llm_client = LLMClient(
                provider="azure",
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            )
        elif llm_provider == "openai":
            self.llm_client = LLMClient(
                provider="openai",
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
                model=os.environ.get("OPENAI_MODEL", "gpt-4"),
            )
        elif llm_provider == "vllm":
            self.llm_client = LLMClient(
                provider="vllm",
                api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
                base_url=os.environ["VLLM_BASE_URL"],
                model=os.environ.get("VLLM_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
            )
        elif llm_provider == "google":
            self.llm_client = LLMClient(
                provider="google",
                api_key=os.environ.get("GEMINI_API_KEY"),
                base_url=os.environ.get("GEMINI_BASE_URL"),
                model=os.environ.get("GEMINI_MODEL", "gemini-1.5-pro"),
            )
        elif llm_provider == "anthropic":
            self.llm_client = LLMClient(
                provider="anthropic",
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                base_url=os.environ.get("ANTHROPIC_BASE_URL"),
                model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            )
        elif llm_provider == "xai":
            self.llm_client = LLMClient(
                provider="xai",
                api_key=os.environ.get("XAI_API_KEY"),
                base_url=os.environ.get("XAI_BASE_URL"),
                model=os.environ.get("XAI_MODEL", "grok-beta"),
            )
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {llm_provider}")

        # A2A 1단계: 각 서버별 에이전트 description 생성
        agent_descriptions = {}
        for server_name in self.mcp_manager.servers.keys():
            server_tools = [tool for tool in self.mcp_manager.all_tools
                          if getattr(tool, 'name', '').startswith(f'{server_name}__')]

            # .env에서 description이 있으면 사용
            description_key = f"MCP_SERVER_{list(self.mcp_manager.servers.keys()).index(server_name) + 1}_DESCRIPTION"
            env_description = os.environ.get(description_key, "")

            if env_description:
                description = env_description
            else:
                # 도구 이름에서 자동 생성
                description = self._generate_agent_description(server_name, server_tools)

            agent_descriptions[server_name] = description
            print(f"[INFO] {server_name} description: {description}")

        # Agent A 초기화 (A2A 1단계: description 기반)
        available_agents = list(self.mcp_manager.servers.keys())
        self.question_agent = QuestionUnderstandingAgent(
            self.llm_client, available_agents, agent_descriptions
        )

        # 전문 에이전트 생성
        print(f"\n✅ 에이전트 초기화 완료 (A2A 1단계):")
        print(f"   • {self.question_agent.name}: {self.question_agent.role}")

        for server_name in self.mcp_manager.servers.keys():
            server_tools = [tool for tool in self.mcp_manager.all_tools
                          if getattr(tool, 'name', '').startswith(f'{server_name}__')]

            agent = ToolBasedAgent(
                name=f"{server_name.upper()}Agent",
                role=f"{server_name} 전문 서비스",
                llm_client=self.llm_client,
                tools=server_tools
            )
            self.specialist_agents[server_name] = agent
            print(f"   • {agent.name}: {agent.role} (도구 {len(server_tools)}개)")

    def _generate_agent_description(self, server_name: str, server_tools) -> str:
        """도구들의 description을 기반으로 에이전트 설명 자동 생성 (A2A 1단계)"""
        if not server_tools:
            return f"{server_name} 도메인 전문 에이전트"

        # 각 도구의 description 수집
        tool_descriptions = []
        for tool in server_tools:
            tool_desc = getattr(tool, 'description', '')
            if tool_desc:
                # description이 너무 길면 첫 문장만 사용
                first_sentence = tool_desc.split('.')[0].strip()
                if first_sentence:
                    tool_descriptions.append(first_sentence)

        if not tool_descriptions:
            # description이 없으면 도구 이름 나열
            tool_names = [getattr(tool, 'name', '').replace(f'{server_name}__', '')
                         for tool in server_tools]
            tool_list = ', '.join(tool_names[:3])
            if len(tool_names) > 3:
                tool_list += f" 등 {len(tool_names)}개 도구"
            return f"{server_name} 전문 에이전트 ({tool_list}를 제공)"

        # 도구 description들을 통합
        if len(tool_descriptions) == 1:
            # 도구가 1개면 그대로 사용
            return f"{server_name} 전문 에이전트. {tool_descriptions[0]}을(를) 지원합니다."
        elif len(tool_descriptions) <= 3:
            # 도구가 2-3개면 모두 나열
            combined = ", ".join(tool_descriptions[:-1]) + f" 및 {tool_descriptions[-1]}"
            return f"{server_name} 전문 에이전트. {combined} 기능을 제공합니다."
        else:
            # 도구가 4개 이상이면 처음 3개만 + 개수
            combined = ", ".join(tool_descriptions[:3])
            remaining = len(tool_descriptions) - 3
            return f"{server_name} 전문 에이전트. {combined} 등 {len(tool_descriptions)}개 기능을 제공합니다."

    async def close_all_servers(self):
        """모든 MCP 서버 연결 종료"""
        await self.mcp_manager.close_all_servers()
        self.specialist_agents.clear()


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

        # 🔍 일관성 검증: execution_order와 question_type이 맞지 않으면 자동 수정
        if execution_order:
            # execution_order 구조 확인
            is_parallel = len(execution_order) == 1 and len(execution_order[0]) > 1
            is_sequential = len(execution_order) > 1

            # dependencies가 있으면 순차 실행이어야 함
            if dependencies and is_parallel:
                print(f"⚠️  일관성 오류 감지: dependencies가 있는데 parallel로 판단됨", flush=True)
                print(f"   자동 수정: parallel → sequential", flush=True)
                # 병렬을 순차로 변경
                if len(execution_order[0]) > 1:
                    execution_order = [[agent] for agent in execution_order[0]]
                    question_type = "sequential"

            # execution_order와 question_type 일치 여부 확인
            if is_parallel and question_type != "parallel":
                print(f"⚠️  일관성 오류 감지: execution_order는 parallel인데 question_type={question_type}", flush=True)
                print(f"   자동 수정: question_type → parallel", flush=True)
                question_type = "parallel"
            elif is_sequential and question_type == "parallel":
                print(f"⚠️  일관성 오류 감지: execution_order는 sequential인데 question_type=parallel", flush=True)
                print(f"   자동 수정: question_type → sequential", flush=True)
                question_type = "sequential"

        print(f"🎯 판단 결과:")
        print(f"   질문 유형: {question_type}")

        if execution_order:
            execution_plan_str = " → ".join([
                f"({', '.join(group)})" if len(group) > 1 else group[0]
                for group in execution_order
            ])
            print(f"   실행 순서: {execution_plan_str}")
        else:
            print(f"   실행 순서: none")

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

    # Step 2: 전문 에이전트들 실행
    if not execution_order:
        # 일반 질문 - Agent A가 직접 답변
        print(f"💬 [Final Answer] Agent A 직접 답변")
        print(f"{'─'*70}\n")

        try:
            messages = [
                {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
                {"role": "user", "content": user_query}
            ]

            stream_start = time.time()
            stream_response = orchestrator.llm_client.chat_completion(messages, stream=True)

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

    # execution_order를 그룹별로 실행
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
                    continue

                specialist_agent = orchestrator.specialist_agents[agent_name]
                query = queries.get(agent_name, user_query)

                context = {
                    "original_query": user_query,
                    "structured_query": query,
                }
                if previous_results:
                    context["previous_agent_results"] = previous_results

                tasks.append(specialist_agent.process_with_tools(query, context))
                task_agent_names.append(agent_name)

            if tasks:
                step_start = time.time()
                parallel_responses = await asyncio.gather(*tasks)
                step_time = time.time() - step_start

                print(f"✅ 병렬 처리 완료 ({step_time:.2f}초)\n")

                for agent_name, response in zip(task_agent_names, parallel_responses):
                    if response.success:
                        result_info = {
                            "agent": response.agent_name,
                            "agent_name": agent_name,
                            "query": queries.get(agent_name, user_query),
                            "response": response.content,
                            "time": step_time
                        }
                        agent_results[agent_name] = result_info
                        previous_results.append(result_info)

        step_num += 1

    if not agent_results:
        return question_response.content

    # Step 3: Agent A - 결과 통합
    print(f"🔄 [Step {step_num}] Agent A: 결과 통합 및 최종 답변 생성")
    print(f"{'─'*70}\n")

    try:
        expert_answers = ""
        flat_execution_order = [agent for group in execution_order for agent in group]
        for i, agent_name in enumerate(flat_execution_order, 1):
            if agent_name in agent_results:
                result = agent_results[agent_name]
                expert_answers += f"\n\n[{i}단계: {result['agent']}의 답변]\n질문: {result['query']}\n답변: {result['response']}"

        messages = [
            {"role": "system", "content": "당신은 여러 전문가의 답변을 통합하여 최종 답변을 제공하는 코디네이터입니다."},
            {"role": "user", "content": f"원본 질문: {user_query}\n\n전문가 답변들:{expert_answers}\n\n위 내용을 바탕으로 최종 답변을 제공해주세요."}
        ]

        stream_response = orchestrator.llm_client.chat_completion(messages, stream=True)

        collected_content = ""
        for chunk in stream_response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    print(delta.content, end="", flush=True)
                    collected_content += delta.content

        print(f"\n")
        return collected_content

    except Exception as e:
        print(f"\n⚠️  통합 실패: {e}")
        return "\n\n".join([f"[{r['agent']}]\n{r['response']}" for r in agent_results.values()])

