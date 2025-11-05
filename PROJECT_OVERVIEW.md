# 🤖 Dynamic Multi-Agent System with MCP

## 📅 프로젝트 개요

**프로젝트명:** Agent with MCP (Model Context Protocol)  
**버전:** A2A Phase 1 (Agent-to-Agent Communication)  
**마지막 업데이트:** 2025-11-05

---

## 🎯 프로젝트 목적

**MCP(Model Context Protocol) 기반의 동적 멀티 에이전트 시스템**

- 여러 MCP 서버를 동적으로 연결
- 질문을 분석하여 적절한 에이전트에게 자동 라우팅
- 병렬/순차 실행을 지능적으로 결정
- 에이전트 간 자율적 협력 (A2A Phase 1)

---

## 📂 프로젝트 구조

```
agent_with_mcp/
├── 📦 Core Modules (실행 코드)
│   ├── llm_client.py              # LLM 클라이언트 (400줄)
│   ├── agents.py                  # 에이전트 클래스 (380줄)
│   ├── mcp_manager.py             # MCP 서버 관리 (130줄)
│   ├── orchestrator.py            # 멀티 에이전트 오케스트레이션 (350줄)
│   └── dynamic_multi_agent_v2.py  # 메인 진입점 (100줄)
│
├── 📚 Documentation (문서)
│   ├── PROJECT_OVERVIEW.md        # 프로젝트 전체 개요 ⭐
│   ├── MODULE_STRUCTURE.md        # 모듈 구조 및 분할 설명
│   ├── A2A_PHASE1_COMPLETE.md     # A2A 1단계 구현 가이드
│   ├── SEQUENTIAL_CONTEXT_FIX.md  # 순차 실행 시 이전 결과 전달
│   ├── ENUM_AUTO_CONVERSION.md    # Enum 타입 자동 변환
│   ├── MCP_SERVER_ERROR_HANDLING.md # MCP 서버 에러 처리
│   └── TOOLBASEDAGENT_PROMPT_FIX.md # 도구 사용 프롬프트 개선
│
├── 🗑️ Legacy (백업/참고용)
│   ├── dynamic_multi_agent.py     # 원본 파일 (백업)
│   ├── agent_to_agent.py          # A2A v1 (사용 안 함)
│   └── agent_to_agent_v2.py       # A2A v2 (개발 중단)
│
├── ⚙️ Configuration
│   ├── .env                       # 환경 변수 설정
│   └── requirements.txt           # Python 패키지 의존성
│
└── 📋 Others
    ├── README_DYNAMIC.md          # 기본 README
    └── LLM_CLIENT_GUIDE.md        # LLM 클라이언트 가이드
```

---

## 🏗️ 시스템 아키텍처

### A2A Phase 1 구조

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                            │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  Agent A (QuestionUnderstandingAgent)                       │
│  • 질문 분석 및 라우팅                                        │
│  • 에이전트 Description만 보고 판단 (A2A 1단계)               │
│  • 병렬/순차 실행 결정                                        │
└──────────────┬──────────────────────────────────────────────┘
               ↓
┌──────────────┴──────────────────────────────────────────────┐
│                                                               │
│  Agent B          Agent C          Agent D                   │
│  (ToolBasedAgent) (ToolBasedAgent) (ToolBasedAgent)         │
│                                                               │
│  • 자율적으로 도구 선택 및 실행                                │
│  • Agent A는 이들의 도구를 모름 (A2A 원칙)                     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────────┐
│                    MCP Tools                                 │
│  laws__law-search, search__web-search, etc.                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 핵심 개념

### 1. A2A Phase 1 (Agent-to-Agent Communication)

**원칙:**
- ✅ Agent A는 **에이전트의 역할(Description)만** 보고 라우팅
- ✅ Agent A는 **각 에이전트가 어떤 도구를 가졌는지 모름**
- ✅ 각 에이전트는 **자율적으로** 자신의 도구를 선택

**예시:**
```
Agent A가 보는 정보:
  - laws: "법률 검색 전문 에이전트. 법령, 조문을 검색하고 해석합니다."
  - search: "웹 검색 전문 에이전트. 뉴스, 정보를 검색합니다."

Agent A가 모르는 정보:
  - laws가 어떤 도구를 가졌는지 (law-search? statute-lookup?)
  - search가 어떤 도구를 가졌는지 (web-search? news-fetch?)
```

### 2. 동적 MCP 서버 연결

**.env 설정만으로 MCP 서버 추가:**
```env
MCP_SERVER_1_URL=http://localhost:8001/mcp-server-laws/
MCP_SERVER_1_NAME=laws
MCP_SERVER_1_DESCRIPTION=법률 검색 전문 에이전트

MCP_SERVER_2_URL=http://localhost:8002/mcp-server-search/
MCP_SERVER_2_NAME=search
MCP_SERVER_2_DESCRIPTION=웹 검색 전문 에이전트
```

코드 수정 없이 서버 추가 가능!

### 3. 병렬/순차 실행 자동 결정

**병렬 실행 (Parallel):**
```
질문: "12대 중과실이 뭐야?"
→ laws와 search 동시 실행 (독립적)
```

**순차 실행 (Sequential):**
```
질문: "최근 위반 사례를 찾고, 해당 법 조항을 알려줘"
→ search 먼저 실행 → 결과를 laws에게 전달
```

---

## 🎨 주요 모듈 설명

### 1. llm_client.py (LLM 클라이언트)

**역할:** LLM API 호출 통합 관리

**지원 LLM:**
- Azure OpenAI
- OpenAI
- vLLM (로컬)
- Google Gemini
- Anthropic Claude
- xAI Grok

**주요 기능:**
```python
client = LLMClient(
    provider="azure",  # or "openai", "vllm", "google", "anthropic", "xai"
    api_key="...",
    model="gpt-4"
)

response = client.chat_completion(messages, tools=tools)
```

---

### 2. agents.py (에이전트 클래스)

#### 2.1 QuestionUnderstandingAgent (Agent A)

**역할:** 질문 분석 및 라우팅

**입력:**
```python
QuestionUnderstandingAgent(
    llm_client=llm_client,
    available_agents=["laws", "search"],
    agent_descriptions={
        "laws": "법률 검색 전문 에이전트...",
        "search": "웹 검색 전문 에이전트..."
    }
)
```

**출력:**
```json
{
  "question_type": "parallel",
  "execution_order": [["laws", "search"]],
  "queries": {
    "laws": "12대 중과실 법률 조항",
    "search": "12대 중과실 설명과 사례"
  }
}
```

#### 2.2 ToolBasedAgent (Agent B/C/D)

**역할:** 도구 기반 전문 작업 수행

**특징:**
- MCP 도구 자동 로드
- Schema 기반 파라미터 검증
- 이전 에이전트 결과 참조
- 도구 실행 결과 반환

---

### 3. mcp_manager.py (MCP 관리)

**역할:** MCP 서버 연결 및 도구 관리

**주요 기능:**
```python
manager = MCPManager()

# MCP 서버 연결
await manager.connect_mcp_server(
    server_name="laws",
    base_url="http://localhost:8001/mcp-server-laws/",
    auth_bearer="token"
)

# 도구 자동 로드 및 prefix 추가
# law-search → laws__law-search
```

**Schema 자동 수정:**
- `raw_mcp_tool.inputSchema` 추출
- `input_schema` 속성 설정

---

### 4. orchestrator.py (오케스트레이션)

**역할:** 멀티 에이전트 시스템 조율

**주요 흐름:**
```python
1. 에이전트 초기화
   - LLM 클라이언트 설정
   - Agent A 생성 (Description 기반)
   - Agent B/C/D 생성 (도구 기반)

2. 질문 처리
   - Agent A: 질문 분석
   - Agent B/C/D: 병렬/순차 실행
   - Agent A: 결과 통합

3. 일관성 검증
   - execution_order vs question_type 확인
   - dependencies 존재 시 자동 sequential 변환
```

---

### 5. dynamic_multi_agent_v2.py (메인)

**역할:** 프로그램 진입점

**실행:**
```bash
python dynamic_multi_agent_v2.py
```

**초기화:**
```python
1. .env에서 MCP 서버 목록 로드
2. 각 서버에 연결
3. 에이전트 초기화
4. 대화 루프 시작
```

---

## 🚀 사용 방법

### 1. 환경 설정

**.env 파일 생성:**
```env
# LLM 설정
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4

# MCP 서버 1
MCP_SERVER_1_URL=http://localhost:8001/mcp-server-laws/
MCP_SERVER_1_NAME=laws
MCP_SERVER_1_AUTH_BEARER=your_token
MCP_SERVER_1_DESCRIPTION=법률 검색 전문 에이전트

# MCP 서버 2
MCP_SERVER_2_URL=http://localhost:8002/mcp-server-search/
MCP_SERVER_2_NAME=search
MCP_SERVER_2_DESCRIPTION=웹 검색 전문 에이전트

# 기타 설정
MAX_ITERATIONS=10
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 실행

```bash
python dynamic_multi_agent_v2.py
```

### 4. 사용 예시

```
🧑 You> 12대 중과실이 뭐야?

📋 [Step 1] Agent A: 질문 분석 및 라우팅
✅ 분석 완료

🎯 판단 결과:
   질문 유형: parallel
   실행 순서: (laws, search)

🚀 [Step 2] 2개 에이전트 병렬 처리
  🔧 [LAWSAgent] Tool calls: 1
    ✅ Tool laws__law-search executed
  
  🔧 [SEARCHAgent] Tool calls: 1
    ✅ Tool search__web-search executed

🔄 [Step 3] Agent A: 결과 통합 및 최종 답변 생성

12대 중과실은...
```

---

## 🔧 핵심 기능

### 1. 자동 일관성 검증

**문제:**
```json
{
  "question_type": "parallel",
  "execution_order": [["A", "B"]],
  "dependencies": {"B": "A의 결과 필요"}
}
```
❌ dependencies가 있는데 parallel → 모순!

**해결:**
```
⚠️  일관성 오류 감지: dependencies가 있는데 parallel로 판단됨
   자동 수정: parallel → sequential
   execution_order: [["A", "B"]] → [["A"], ["B"]]
```

### 2. 이전 결과 전달 (순차 실행)

**문제:**
```
질문: "앞서 찾은 사례에 해당되는 법 조항..."
```
❌ "앞서 찾은"만으로는 구체적인 값을 모름

**해결:**
```
🔍 이전 에이전트가 찾은 정보:

[SEARCHAgent의 답변]
최근 근로기준법 위반 사례:
1. A기업 - 연장근로 수당 미지급
2. B회사 - 부당해고
...

⚠️ 중요: search 결과를 바탕으로 해당 법 조항을 찾아야 함

──────────────────────────────────────────────────────

📌 현재 질문:
이전 에이전트가 찾은 사례들에 해당하는 법 조항을 찾아주세요
```

### 3. Tool Call ID 불일치 방지

**문제:**
```python
# LLM이 2개 도구 호출
tool_calls = [
    {"id": "call_1", "name": "tool_A"},
    {"id": "call_2", "name": "tool_B"}
]

# 첫 번째만 실행하고 두 번째 누락
tool_results = [
    {"tool_call_id": "call_1", "content": "..."}
    # ❌ call_2 누락!
]
```

**해결:**
```python
# 모든 tool_call에 대해 결과 보장
for tc in choice.tool_calls:
    tool_found = False
    # ... 도구 실행 ...
    
    if not tool_found:
        tool_results.append({
            "tool_call_id": tc.id,
            "content": "Error: Tool not found"
        })
```

### 4. Enum 타입 자동 변환

**문제:**
```python
# LLM이 정수 전달
args = {"search_type": 1}

# MCP 서버는 문자열 기대
schema = {"search_type": {"enum": ["1", "2"]}}
```

**해결:**
```python
# 자동 변환
if isinstance(args[key], (int, float)):
    args[key] = str(args[key])
# 1 → "1"
```

### 5. 범용 Description 자동 생성

**문제:**
```python
# 하드코딩
if 'law' in keywords:
    return "법률 검색 전문..."
```

**해결:**
```python
# 도구의 description 통합
tool_descriptions = [tool.description for tool in tools]
return f"{server_name} 전문 에이전트. {', '.join(tool_descriptions)} 기능 제공"
```

---

## 📊 성능 특성

### 병렬 vs 순차 실행

| 실행 방식 | 에이전트 수 | 평균 시간 | 예시 |
|-----------|------------|-----------|------|
| **병렬** | 2개 | ~5초 | "12대 중과실이 뭐야?" |
| **병렬** | 3개 | ~6초 | "법률, 판례, 뉴스 모두 찾아줘" |
| **순차** | 2개 | ~10초 | "사례 찾고 법 조항 알려줘" |
| **순차** | 3개 | ~15초 | "A→B→C 순서로" |

### 초기화 시간

- MCP 서버 연결: ~1초/서버
- 에이전트 초기화: <1초
- 총 초기화 시간: ~3-5초 (3개 서버 기준)

---

## ⚙️ 설정 가이드

### LLM 제공자 설정

#### Azure OpenAI
```env
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4
```

#### OpenAI
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4
```

#### Google Gemini
```env
LLM_PROVIDER=google
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-1.5-pro
```

#### Anthropic Claude
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

#### xAI Grok
```env
LLM_PROVIDER=xai
XAI_API_KEY=...
XAI_MODEL=grok-beta
```

#### vLLM (로컬)
```env
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=meta-llama/Llama-2-7b-chat-hf
```

---

## 🐛 문제 해결

### 1. MCP 서버 연결 실패

**증상:**
```
[Error] Failed to connect to laws: Connection refused
```

**해결:**
1. MCP 서버가 실행 중인지 확인
2. URL이 올바른지 확인 (`/` 끝에 주의)
3. Auth Bearer 토큰 확인

### 2. Tool Call ID 불일치

**증상:**
```
Error code: 400 - 'tool_call_id' of 'xxx' not found
```

**해결:** ✅ 이미 수정됨 (모든 tool_call에 대해 결과 보장)

### 3. Enum 검증 실패

**증상:**
```
ValidationError: Input should be '1' or '2' [type=enum, input_value=1]
```

**해결:** ✅ 이미 수정됨 (자동 타입 변환)

### 4. 순차 실행 시 이전 결과 누락

**증상:**
```
질문: "앞서 찾은 사례에 해당되는..."
→ 구체적인 사례 내용이 없음
```

**해결:** ✅ 이미 수정됨 (이전 결과를 포맷팅하여 명시적 전달)

---

## 📈 향후 계획

### A2A Phase 2 (개발 예정)

**목표:** Agent 간 Peer-to-Peer 협력

**기능:**
- Agent B가 Agent C에게 직접 협력 요청
- `request_agent_help` 도구 추가
- 다단계 에이전트 체인

**예시:**
```
User → Agent A → Agent B (법률 검색)
                    ↓ (협력 요청)
                 Agent C (판례 검색)
                    ↓ (협력 요청)
                 Agent D (뉴스 검색)
```

### 기타 개선 사항

- [ ] 스트리밍 응답 개선
- [ ] 에이전트 성능 모니터링
- [ ] 캐싱 메커니즘
- [ ] 에러 재시도 전략
- [ ] 로깅 시스템 개선

---

## 📚 참고 문서

### 주요 문서
1. **MODULE_STRUCTURE.md** - 모듈 구조 및 파일 분할
2. **A2A_PHASE1_COMPLETE.md** - A2A 1단계 구현 가이드
3. **LLM_CLIENT_GUIDE.md** - LLM 클라이언트 사용법

### 문제 해결 문서
1. **SEQUENTIAL_CONTEXT_FIX.md** - 순차 실행 시 context 전달
2. **ENUM_AUTO_CONVERSION.md** - Enum 타입 자동 변환
3. **MCP_SERVER_ERROR_HANDLING.md** - MCP 에러 처리
4. **TOOLBASEDAGENT_PROMPT_FIX.md** - 도구 사용 프롬프트 개선

---

## 🎓 핵심 원칙

### 1. A2A (Agent-to-Agent)
✅ Agent A는 Description만 보고 라우팅  
✅ 각 에이전트는 자율적으로 도구 선택  
✅ 느슨한 결합 (Loose Coupling)

### 2. 동적 확장성
✅ .env 설정만으로 서버 추가  
✅ 코드 수정 불필요  
✅ Description 자동 생성

### 3. 지능적 실행
✅ 병렬/순차 자동 결정  
✅ 일관성 자동 검증  
✅ 이전 결과 자동 전달

### 4. 견고성
✅ 에러 자동 처리  
✅ Tool Call ID 보장  
✅ 타입 자동 변환

---

## 🎯 결론

**A2A Phase 1 기반의 동적 멀티 에이전트 시스템이 완성되었습니다!**

- ✅ 5개 모듈로 깔끔하게 분리
- ✅ A2A 1단계 원칙 준수
- ✅ 동적 MCP 서버 연결
- ✅ 지능적 병렬/순차 실행
- ✅ 강력한 에러 처리

**이제 .env 파일만 수정하면 새로운 MCP 서버를 추가할 수 있습니다!** 🚀✨

---

**Made with ❤️ by Agent with MCP Team**  
**Last Updated:** 2025-11-05

