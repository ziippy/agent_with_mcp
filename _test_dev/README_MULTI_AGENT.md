# 🤖 Multi-Agent System with MCP

Agent A (질문 이해) → Agent B (법률 전문) / Agent C (판례 전문)

## 📋 아키텍처

```
User Query
    ↓
┌─────────────────────────────────────────────┐
│  Agent A: QuestionUnderstandingAgent        │
│  - 질문 분석 및 분류                          │
│  - 다음 에이전트 결정                         │
│  - 구조화된 쿼리 생성                         │
└─────────────────────────────────────────────┘
    ↓
    ├─→ legal_agent ──────────────────────────┐
    │                                          │
┌───┴──────────────────────────────────────┐  │
│  Agent B: LegalExpertAgent               │  │
│  - 법률 전문 답변                         │  │
│  - MCP Server 1 도구 사용                 │  │
│  - 법조문, 법률 용어 설명                  │  │
└──────────────────────────────────────────┘  │
                                              │
    ├─→ precedent_agent ──────────────────┐  │
    │                                      │  │
┌───┴──────────────────────────────────┐  │  │
│  Agent C: PrecedentExpertAgent       │  │  │
│  - 판례 검색 및 분석                  │  │  │
│  - MCP Server 2 도구 사용            │  │  │
│  - 판결 요지, 적용 법리 설명          │  │  │
└──────────────────────────────────────┘  │  │
    ↓                                     │  │
┌───────────────────────────────────────┐  │  │
│  Final Answer (Streaming)             │←─┴──┘
│  - 모든 에이전트 결과 종합             │
│  - 사용자 친화적 답변 생성             │
└───────────────────────────────────────┘
```

## 🚀 실행 방법

### 1. 환경 설정

`.env` 파일에 다음 설정:

```env
# Azure OpenAI 설정
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# MCP Server 1 (법률 도구)
MCP_SERVER_1_URL=http://localhost:8001
MCP_SERVER_1_AUTH_BEARER=

# MCP Server 2 (판례 도구)
MCP_SERVER_2_URL=http://localhost:8002
MCP_SERVER_2_AUTH_BEARER=
```

### 2. 실행

```bash
python multi_agent_system.py
```

## 💡 사용 예시

### 📋 법률 질문

```
🧑 You> 계약서에 명시되지 않은 조항을 요구받았는데 어떻게 해야 하나요?

[Step 1] Agent A: 질문 분석
✅ 분석 완료 (0.85초)
{
  "keywords": ["계약서", "명시되지 않은 조항"],
  "question_type": "legal",
  "next_agent": "legal_agent",
  "structured_query": "계약서에 명시되지 않은 조항의 법적 효력과 대응 방법",
  "analysis": "계약법 관련 법률 자문이 필요한 질문"
}

[Step 2] Agent B: 법률 전문가 처리
🔧 Tool calls: 2
  ✅ Tool mcp1__search_law executed
  ✅ Tool mcp1__get_legal_advice executed
✅ 처리 완료 (2.34초)

[Final Answer]
계약서에 명시되지 않은 조항은 법적 구속력이 없습니다...
```

### 📚 판례 검색

```
🧑 You> 부당해고 관련 판례를 알려주세요

[Step 1] Agent A: 질문 분석
✅ 분석 완료 (0.78초)
{
  "keywords": ["부당해고", "판례"],
  "question_type": "precedent",
  "next_agent": "precedent_agent",
  "structured_query": "부당해고 관련 주요 판례 검색",
  "analysis": "판례 검색이 필요한 질문"
}

[Step 2] Agent C: 판례 전문가 처리
🔧 Tool calls: 1
  ✅ Tool mcp2__search_precedent executed
✅ 처리 완료 (1.92초)

[Final Answer]
부당해고와 관련된 주요 판례는 다음과 같습니다...
```

## 🎯 주요 기능

### ✨ 특징

1. **지능형 라우팅**: Agent A가 질문을 분석하여 적절한 전문 에이전트로 자동 라우팅
2. **전문화된 도구**: 각 에이전트가 특화된 MCP 서버 도구 사용
3. **스트리밍 응답**: 최종 답변을 실시간으로 스트리밍 출력
4. **시간 측정**: 각 단계별 처리 시간 표시
5. **에러 처리**: 콘텐츠 필터링, API 오류 등 안전한 에러 처리

### 🛠️ 에이전트 상세

#### Agent A: QuestionUnderstandingAgent
- **역할**: 질문 이해 및 분석
- **출력**: JSON 형식의 구조화된 분석 결과
- **결정**: 다음 에이전트 선택 (legal_agent / precedent_agent / none)

#### Agent B: LegalExpertAgent
- **역할**: 법률 전문 답변
- **도구**: MCP Server 1 (법률 관련 도구)
- **기능**: 법조문 검색, 법률 자문, 법률 용어 설명

#### Agent C: PrecedentExpertAgent
- **역할**: 판례 검색 및 분석
- **도구**: MCP Server 2 (판례 관련 도구)
- **기능**: 판례 검색, 판결 요지 분석, 법리 설명

## 📊 출력 형식

```
======================================================================
🤖 Multi-Agent Processing Pipeline
======================================================================

📋 [Step 1] Agent A: 질문 분석
──────────────────────────────────────────────────────────────────────
✅ 분석 완료 (0.85초)

{분석 결과 JSON}

🎯 판단 결과:
   질문 유형: legal
   다음 에이전트: legal_agent
   구조화된 쿼리: ...

⚖️  [Step 2] Agent B: 법률 전문가 처리
──────────────────────────────────────────────────────────────────────
  🔧 [LegalExpertAgent] Tool calls: 1
    ✅ Tool mcp1__tool_name executed

✅ 처리 완료 (2.34초)

💬 [Final Answer]
──────────────────────────────────────────────────────────────────────

{스트리밍 최종 답변}

⏱️  스트리밍 시간: 0.42초

⏱️  총 소요 시간: 3.61초
```

## 🔧 커스터마이징

### 새로운 에이전트 추가

```python
class CustomAgent(SpecializedAgent):
    def __init__(self, aoai_wrapper: AzureOpenAIWrapper, tools: List[BaseTool]):
        system_prompt = """당신의 커스텀 프롬프트"""
        super().__init__(
            name="CustomAgent",
            role="커스텀 역할",
            system_prompt=system_prompt,
            aoai_wrapper=aoai_wrapper
        )
        self.tools = tools
```

### 시스템 프롬프트 수정

각 에이전트의 `__init__` 메서드에서 `system_prompt` 수정

## ⚙️ 기술 스택

- **Python 3.11+**
- **Azure OpenAI**: GPT-4o 모델
- **Google ADK**: MCP 도구 통합
- **MCP (Model Context Protocol)**: 외부 도구 연결
- **asyncio**: 비동기 처리

## 📝 비교: 기존 vs 멀티 에이전트

| 항목 | 기존 (agent_with_mcp.py) | 멀티 에이전트 (multi_agent_system.py) |
|------|-------------------------|-------------------------------------|
| 구조 | 단일 에이전트 | 3개 특화 에이전트 |
| 라우팅 | 없음 | Agent A가 자동 라우팅 |
| 도구 분리 | 모든 도구 공유 | 에이전트별 도구 분리 |
| 처리 방식 | 직접 처리 | 파이프라인 처리 |
| 가독성 | 보통 | 높음 |
| 확장성 | 낮음 | 높음 |
| 유지보수 | 어려움 | 쉬움 |

## 🚦 다음 단계

- [ ] Agent D 추가 (예: 법률 문서 작성 전문)
- [ ] 대화 히스토리 관리
- [ ] 데이터베이스 연동 (세션 저장)
- [ ] REST API 서버화
- [ ] 웹 UI 추가

## 📄 라이선스

MIT License

