# 🤖 Dynamic Multi-Agent System

**범용 멀티 에이전트 시스템** - 어떤 도메인의 MCP 서버든 자동으로 에이전트 생성

## 🎯 주요 특징

### ✨ **도메인 독립적**
- ❌ 법률/판례 특화 → ✅ 범용 시스템
- MCP 서버만 연결하면 자동으로 에이전트 생성
- 도메인 지식 불필요

### 🔀 **병렬/순차 실행 지원**
- **병렬 실행**: 독립적인 작업을 동시 처리 (시간 절약)
  - 예: "12대 중과실은 뭐야?" → 법률 조문과 판례를 동시에 검색
  - `execution_order: [["agent1", "agent2"]]`
- **순차 실행**: 의존적인 작업을 단계별 처리 (정확도 향상)
  - 예: "최근 사례를 찾아보고, 해당 법 조항을 알려줘" → 사례 검색 후 법률 검색
  - `execution_order: [["agent1"], ["agent2"]]`
- **Agent A가 자동으로 실행 전략 결정**

### 🤖 **다양한 LLM 제공자 지원 (총 6개)**
- **Azure OpenAI** - 엔터프라이즈용 (기본)
- **OpenAI** - GPT-4o, GPT-4o-mini 등
- **vLLM** - 로컬 LLM 실행
- **Google Gemini** - OpenAI 호환 모드 지원 ✨
- **Anthropic Claude** - OpenAI 호환 모드 지원 ✨
- **xAI Grok** - 최신 AI 모델 ✨ NEW

### 🔄 **동적 에이전트 생성**
```
MCP Server 1 → Agent B (자동 생성)
MCP Server 2 → Agent C (자동 생성)
MCP Server 3 → Agent D (자동 생성)
MCP Server N → Agent X (자동 생성)
```

### 📋 **아키텍처**

```
User Query
    ↓
┌─────────────────────────────────────────┐
│  Agent A (라우팅)                        │
│  - 질문 분석                             │
│  - 각 MCP 서버의 도구 정보 확인 ✨       │
│  - 에이전트 역할 추론                    │
│  - 에이전트 선택 및 순서 결정             │
│  - 의존성 명시                           │
└─────────────────────────────────────────┘
    ↓
    ├─→ MCP1 Agent (동적 생성)
    │   └─ Tools: precedent-search, case-lookup, ...
    │
    ├─→ MCP2 Agent (동적 생성)
    │   └─ Tools: law-search, statute-lookup, ...
    │
    └─→ MCP3 Agent (동적 생성)
        └─ Tools: web-search, news-search, ...
    ↓
┌─────────────────────────────────────────┐
│  Agent A (통합)                          │
│  - 모든 결과 종합 (실행 순서 반영)       │
│  - 최종 답변 생성 (스트리밍)             │
└─────────────────────────────────────────┘
```

## 🚀 설정 및 실행

### 1. `.env` 설정

#### Option A: Azure OpenAI (기본값)

```env
# LLM 제공자 선택 (기본값: azure)
LLM_PROVIDER=azure

# Azure OpenAI 설정
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# MCP 서버 동적 설정
MCP_SERVER_1_NAME=mcp1
MCP_SERVER_1_URL=http://localhost:8001
MCP_SERVER_1_AUTH_BEARER=

MCP_SERVER_2_NAME=mcp2
MCP_SERVER_2_URL=http://localhost:8002
MCP_SERVER_2_AUTH_BEARER=
```

#### Option B: OpenAI

```env
# LLM 제공자 선택
LLM_PROVIDER=openai

# OpenAI 설정
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4

# MCP 서버 설정 (동일)
MCP_SERVER_1_NAME=mcp1
MCP_SERVER_1_URL=http://localhost:8001
```

#### Option C: vLLM (로컬)

```env
# LLM 제공자 선택
LLM_PROVIDER=vllm

# vLLM 설정
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_API_KEY=EMPTY

# MCP 서버 설정 (동일)
MCP_SERVER_1_NAME=mcp1
MCP_SERVER_1_URL=http://localhost:8001
```

#### Option D: Google Gemini ✨

```env
# LLM 제공자 선택
LLM_PROVIDER=google

# Google Gemini 설정
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash-exp
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai  # OpenAI 호환 모드 (권장)

# MCP 서버 설정 (동일)
MCP_SERVER_1_NAME=mcp1
MCP_SERVER_1_URL=http://localhost:8001
```

**💡 Tip**: `GEMINI_BASE_URL`을 설정하면 OpenAI 호환 모드로 동작하여 Tool Calling 등 고급 기능을 사용할 수 있습니다.

#### Option E: Anthropic Claude ✨

```env
# LLM 제공자 선택
LLM_PROVIDER=anthropic

# Anthropic Claude 설정
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_BASE_URL=  # OpenAI 호환 모드용 (선택 사항)

# MCP 서버 설정 (동일)
MCP_SERVER_1_NAME=mcp1
MCP_SERVER_1_URL=http://localhost:8001
```

#### Option F: xAI Grok ✨ NEW

```env
# LLM 제공자 선택
LLM_PROVIDER=xai

# xAI Grok 설정
XAI_API_KEY=xai-...
XAI_MODEL=grok-beta
XAI_BASE_URL=https://api.x.ai/v1

# MCP 서버 설정 (동일)
MCP_SERVER_1_NAME=mcp1
MCP_SERVER_1_URL=http://localhost:8001
```

**💡 더 많은 설정 옵션은 `.env.example` 파일을 참고하세요!**  
**📚 자세한 LLM 제공자 가이드는 `LLM_PROVIDERS_GUIDE.md`를 참고하세요!**  
**🔄 제공자 변경사항은 `LLM_PROVIDER_CHANGES.md`를 확인하세요!**

### 2. 실행

```bash
python dynamic_multi_agent.py
```

## 💡 사용 예시

### 예시 1: 법률 + 판례 도메인

```
MCP_SERVER_1_NAME=legal      # 법률 서비스
MCP_SERVER_2_NAME=precedent  # 판례 서비스

🧑 You> 중대재해처벌법 설명하고 최근 사례 찾아줘

Agent A: execution_order = ["legal", "precedent"]
  ↓
Step 2: legal Agent → 중대재해처벌법 설명
  ↓
Step 3: precedent Agent → 최근 사례 검색
  ↓
Step 4: Agent A → 통합 답변
```

### 예시 2: 병렬 실행 (독립적 작업)

```
MCP_SERVER_1_NAME=laws       # 법률 검색 서비스
MCP_SERVER_2_NAME=precedent  # 판례 검색 서비스

🧑 You> 12대 중과실은 뭐야?

Agent A: execution_order = [["laws", "precedent"]]  # 병렬 실행
  ↓
Step 2: [병렬 처리] laws Agent + precedent Agent 동시 실행
  - laws Agent → 12대 중과실 법률 정의 검색
  - precedent Agent → 12대 중과실 관련 판례 검색
  ↓
Step 3: Agent A → 두 결과를 통합하여 답변
```

### 예시 3: 순차 실행 (의존적 작업)

```
🧑 You> 최근 근로기준법 위반 사례를 찾아보고, 해당 법 조항을 알려줘

Agent A: execution_order = [["precedent"], ["laws"]]  # 순차 실행
  ↓
Step 2: precedent Agent → 최근 근로기준법 위반 사례 검색
  ↓
Step 3: laws Agent → 위반 사례에 해당하는 법 조항 검색 (의존성: precedent 결과 활용)
  ↓
Step 4: Agent A → 통합 답변
```

### 예시 4: 날씨 + 뉴스 도메인

```
MCP_SERVER_1_NAME=weather  # 날씨 서비스
MCP_SERVER_2_NAME=news     # 뉴스 서비스

🧑 You> 오늘 서울 날씨 알려주고 관련 뉴스도 찾아줘

Agent A: execution_order = [["weather", "news"]]  # 병렬 실행
  ↓
Step 2: [병렬 처리] weather Agent + news Agent 동시 실행
  ↓
Step 3: Agent A → 통합 답변
```

### 예시 5: 단일 에이전트

```
🧑 You> 오늘 날씨는?

Agent A: execution_order = ["weather"]
  ↓
Step 2: weather Agent → 날씨 조회
  ↓
Step 3: Agent A → 답변
```

### 예시 6: 일반 대화

```
🧑 You> 넌 누구야?

Agent A: execution_order = []
  ↓
Agent A 직접 답변 (에이전트 호출 없음)
```

## 🔑 핵심 개선 사항

| 항목 | 이전 (multi_agent_system.py) | 개선 후 (dynamic_multi_agent.py) |
|------|-------------------------------|----------------------------------|
| **에이전트 정의** | 하드코딩 (LegalAgent, PrecedentAgent) | 동적 생성 ✅ |
| **도메인** | 법률/판례 특화 | 범용 (모든 도메인) ✅ |
| **프롬프트** | 법률/판례 용어 포함 | 도메인 독립적 ✅ |
| **역할 인식** | 하드코딩된 역할 | 도구 기반 자동 추론 ✨ |
| **실행 전략** | 단순 순차 실행 | 병렬/순차 동적 결정 ✨ |
| **의존성 관리** | 없음 | 의존성 명시 및 컨텍스트 전달 ✨ |
| **확장성** | 수동 코드 수정 필요 | 환경 변수만 추가 ✅ |
| **MCP 서버 추가** | 코드 수정 | .env 설정만 ✅ |
| **LLM 제공자** | 3개 (Azure, OpenAI, vLLM) | 6개 (Google, Claude, xAI 추가) ✨ |

## 📊 동작 흐름

### 초기화 단계

```python
1. .env에서 MCP 서버 목록 로드
   MCP_SERVER_1_* → mcp1
   MCP_SERVER_2_* → mcp2
   MCP_SERVER_3_* → mcp3
   ...

2. 각 MCP 서버 연결
   - 도구 로드
   - 도구에 prefix 추가 (mcp1__tool1, mcp2__tool2)
   - 도구 정보 수집 ✨

3. 에이전트 자동 생성
   - Agent A (라우팅) - 각 서버의 도구 정보 포함 ✨
   - mcp1 → MCP1Agent (mcp1 도구 사용)
   - mcp2 → MCP2Agent (mcp2 도구 사용)
   - ...

출력 예시:
✅ 에이전트 초기화 완료:
   • QuestionUnderstandingAgent: 질문 이해 및 라우팅
   • MCP1Agent: mcp1 전문 서비스 (도구 3개)
      - mcp1__precedent-search
      - mcp1__case-lookup
      - mcp1__judgment-analysis
   • MCP2Agent: mcp2 전문 서비스 (도구 2개)
      - mcp2__law-search
      - mcp2__statute-lookup
```

### 실행 단계

```python
1. Agent A: 질문 분석
   - available_agents: ["laws", "precedent", "search"]
   - agent_tools_info: {"laws": ["law-search", ...], "precedent": [...], ...}
   - execution_order 및 실행 전략 결정
     * 병렬: [["agent1", "agent2"]]
     * 순차: [["agent1"], ["agent2"]]

2. 전문 에이전트 실행
   for group in execution_order:
       if len(group) == 1:
           # 순차 실행
           specialist_agent = orchestrator.specialist_agents[group[0]]
           response = await specialist_agent.process_with_tools(...)
       else:
           # 병렬 실행
           tasks = [agent.process_with_tools(...) for agent in group]
           responses = await asyncio.gather(*tasks)

3. Agent A: 결과 통합
   - 모든 에이전트 결과 종합 (실행 순서 반영)
   - 최종 답변 스트리밍
```

## 🎨 Agent A 프롬프트 (도구 기반 역할 추론)

```
당신은 질문 분석 및 라우팅 전문가입니다.

사용 가능한 에이전트 및 도구:
  - laws: law-search, statute-lookup, legal-interpretation
  - precedent: precedent-search, case-lookup, judgment-analysis
  - search: web-search, news-fetch, article-summary

**중요**: 각 에이전트가 가진 도구를 보고 어떤 역할을 하는지 추론하세요.
예: 
  - law-search, statute-lookup → 법률/조문 검색 서비스
  - precedent-search, case-lookup → 판례/사례 검색 서비스
  - web-search, news-fetch → 웹/뉴스 검색 서비스

질문 분석 후 적절한 에이전트 선택 및 실행 전략 결정:

🔀 **병렬 실행** (독립적 작업):
- execution_order: [["agent1", "agent2"]]
- 두 에이전트의 작업이 서로 독립적일 때
- 예: "12대 중과실은 뭐야?" → 법률 조문과 판례를 동시에 검색

⏭️ **순차 실행** (의존적 작업):
- execution_order: [["agent1"], ["agent2"]]
- 나중 에이전트가 이전 에이전트의 결과를 활용해야 할 때
- 예: "최근 사례를 찾아보고, 해당 법 조항을 알려줘" → 사례 검색 후 법률 검색

💬 **일반 대화**:
- execution_order: []
- 도구가 필요 없는 일반적인 대화

응답 형식 (JSON):
{
  "question_type": "single|multiple|parallel|general",
  "execution_order": [["agent1"], ["agent2"]] or [["agent1", "agent2"]],
  "queries": {
    "agent1": "구체적인 질문",
    "agent2": "구체적인 질문"
  },
  "dependencies": {
    "agent2": "agent1의 결과를 어떻게 활용할지" (순차 실행 시)
  },
  "analysis": "판단 이유"
}
```

## 🧪 테스트 시나리오

### 시나리오 1: 병렬 실행 (독립적 작업)

```bash
# .env
MCP_SERVER_1_NAME=laws
MCP_SERVER_2_NAME=precedent

# 테스트
You> 12대 중과실은 뭐야?
→ execution_order: [["laws", "precedent"]]  # 병렬 실행
→ 두 에이전트가 동시에 독립적으로 검색
```

### 시나리오 2: 순차 실행 (의존적 작업)

```bash
# .env
MCP_SERVER_1_NAME=precedent
MCP_SERVER_2_NAME=laws

# 테스트
You> 최근 부당해고 사례 찾고, 관련 법 조항 설명해줘
→ execution_order: [["precedent"], ["laws"]]  # 순차 실행
→ precedent가 먼저 사례를 찾고, laws가 그 결과를 바탕으로 법 조항 검색
```

### 시나리오 3: 3개 에이전트 (혼합)

```bash
# .env
MCP_SERVER_1_NAME=weather
MCP_SERVER_2_NAME=news
MCP_SERVER_3_NAME=stock

# 테스트 1: 병렬
You> 오늘 날씨, 관련 뉴스, 그리고 우산 회사 주식 정보 알려줘
→ execution_order: [["weather", "news", "stock"]]  # 3개 병렬

# 테스트 2: 순차 + 병렬
You> 오늘 날씨 확인하고, 그에 따른 뉴스와 주식 정보 알려줘
→ execution_order: [["weather"], ["news", "stock"]]  # weather 먼저, 그 다음 news+stock 병렬
```

## 🔧 커스터마이징

### 새로운 MCP 서버 추가

1. `.env`에 추가
```env
MCP_SERVER_4_NAME=finance
MCP_SERVER_4_URL=http://localhost:8004
MCP_SERVER_4_AUTH_BEARER=token_if_needed
```

2. 재시작
```bash
python dynamic_multi_agent.py
```

3. 자동으로 FinanceAgent 생성됨!

### 에이전트 역할 커스터마이징

`initialize_agents()` 메서드에서:

```python
# 특정 에이전트의 역할 수정
if server_name == "mcp1":
    role = "법률 전문 서비스"
elif server_name == "mcp2":
    role = "판례 검색 서비스"
else:
    role = f"{server_name} 전문 서비스"
```

## 📈 확장 가능성

- ✅ **MCP 서버**: 개수 무제한, .env 설정만으로 추가
- ✅ **도메인**: 제약 없음 (법률, 금융, 의료, 교육 등 모든 도메인)
- ✅ **에이전트**: 동적 생성, 코드 수정 불필요
- ✅ **역할 인식**: 도구 기반 자동 추론 ✨
- ✅ **실행 전략**: 병렬/순차 자동 결정 ✨
- ✅ **의존성**: 이전 결과 활용 및 컨텍스트 전달 ✨
- ✅ **라우팅**: 질문 기반 자동 에이전트 선택
- ✅ **LLM 제공자**: 6개 지원 (Azure, OpenAI, vLLM, Google, Claude, xAI) ✨

## 🆚 비교

| 기능 | multi_agent_system.py | dynamic_multi_agent.py |
|------|----------------------|------------------------|
| 도메인 | 법률/판례 고정 | 범용 ✅ |
| 에이전트 | LegalAgent, PrecedentAgent | 동적 생성 ✅ |
| 프롬프트 | 법률 용어 포함 | 도메인 독립적 ✅ |
| 역할 인식 | 하드코딩 | 도구 기반 추론 ✨ |
| 실행 전략 | 단순 순차 | 병렬/순차 동적 결정 ✨ |
| 의존성 관리 | 없음 | 명시적 의존성 및 컨텍스트 ✨ |
| MCP 추가 | 코드 수정 | .env만 ✅ |
| LLM 제공자 | 3개 | 6개 ✨ |
| 확장성 | 낮음 | 높음 ✅ |

## 📝 완료된 기능

- [x] 에이전트 선택 시 도구 목록 참고 ✅
- [x] 도구 기반 역할 자동 추론 ✅
- [x] 병렬/순차 실행 지원 ✅
- [x] 의존성 명시 및 컨텍스트 전달 ✅
- [x] 6개 LLM 제공자 지원 ✅
- [x] OpenAI 호환 모드 지원 (Google, Claude) ✅

## 🔗 관련 문서

- **`.env.example`** - 환경변수 설정 예시
- **`LLM_PROVIDERS_GUIDE.md`** - LLM 제공자 상세 가이드
- **`LLM_PROVIDER_CHANGES.md`** - 제공자 변경사항 (gemini→google, claude→anthropic, xAI 추가)
- **`multi_agent_system.md`** - 이전 버전 문서

## ⚡ 빠른 시작

```bash
# 1. 환경 설정
cp .env.example .env
# .env 파일 편집 (LLM_PROVIDER, MCP 서버 설정)

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 실행
python dynamic_multi_agent.py

# 4. 질문
🧑 You> 12대 중과실은 뭐야?
```

## 💡 팁

### OpenAI 호환 모드 활용
```env
# Google Gemini - OpenAI 호환 모드 (권장)
LLM_PROVIDER=google
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
```
→ Tool Calling, 스트리밍 등 고급 기능 지원

### 병렬 실행으로 속도 향상
- 독립적인 작업은 Agent A가 자동으로 병렬 처리
- 예: "법률 정의와 판례를 알려줘" → 두 에이전트 동시 실행

### 의존성 활용으로 정확도 향상
- 순차적 작업은 이전 결과를 다음 에이전트에 전달
- 예: "사례를 찾고, 해당 법률을 알려줘" → 사례 결과를 법률 검색에 활용
- [x] LLM 제공자 확장 (Azure OpenAI, OpenAI, vLLM) ✅
- [ ] 에이전트 설명 메타데이터 추가 (각 MCP 서버 설명)
- [ ] 에이전트 간 데이터 공유 최적화
- [ ] 웹 UI 추가
- [ ] 대화 히스토리 관리
- [ ] 캐싱 메커니즘 (반복 질문 최적화)

## 📄 라이선스

MIT License

