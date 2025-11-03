# 🤖 Dynamic Multi-Agent System

**범용 멀티 에이전트 시스템** - 어떤 도메인의 MCP 서버든 자동으로 에이전트 생성

## 🎯 주요 특징

### ✨ **도메인 독립적**
- ❌ 법률/판례 특화 → ✅ 범용 시스템
- MCP 서버만 연결하면 자동으로 에이전트 생성
- 도메인 지식 불필요

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

```env
# Azure OpenAI
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

# 필요한 만큼 추가 가능
MCP_SERVER_3_NAME=mcp3
MCP_SERVER_3_URL=http://localhost:8003
MCP_SERVER_3_AUTH_BEARER=
```

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

### 예시 2: 날씨 + 뉴스 도메인

```
MCP_SERVER_1_NAME=weather  # 날씨 서비스
MCP_SERVER_2_NAME=news     # 뉴스 서비스

🧑 You> 오늘 서울 날씨 알려주고 관련 뉴스도 찾아줘

Agent A: execution_order = ["weather", "news"]
  ↓
Step 2: weather Agent → 서울 날씨 조회
  ↓
Step 3: news Agent → 날씨 관련 뉴스 검색
  ↓
Step 4: Agent A → 통합 답변
```

### 예시 3: 단일 에이전트

```
🧑 You> 오늘 날씨는?

Agent A: execution_order = ["weather"]
  ↓
Step 2: weather Agent → 날씨 조회
  ↓
Step 3: Agent A → 답변
```

### 예시 4: 일반 대화

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
| **확장성** | 수동 코드 수정 필요 | 환경 변수만 추가 ✅ |
| **MCP 서버 추가** | 코드 수정 | .env 설정만 ✅ |
| **실행 순서** | 고정 또는 단순 | 동적 + 의존성 관리 ✅ |

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
   - available_agents: ["mcp1", "mcp2", "mcp3"]
   - execution_order 결정

2. 순차 실행
   for agent_name in execution_order:
       specialist_agent = orchestrator.specialist_agents[agent_name]
       response = await specialist_agent.process_with_tools(...)

3. Agent A: 결과 통합
   - 모든 에이전트 결과 종합
   - 최종 답변 스트리밍
```

## 🎨 Agent A 프롬프트 (도구 기반 역할 추론)

```
당신은 질문 분석 및 라우팅 전문가입니다.

사용 가능한 에이전트 및 도구:
  - mcp1: precedent-search, case-lookup, judgment-analysis
  - mcp2: law-search, statute-lookup, legal-interpretation
  - mcp3: web-search, news-fetch, article-summary

**중요**: 각 에이전트가 가진 도구를 보고 어떤 역할을 하는지 추론하세요.
예: 
  - precedent-search, case-lookup → 판례/사례 검색 서비스
  - law-search, statute-lookup → 법률/조문 검색 서비스
  - web-search, news-fetch → 웹/뉴스 검색 서비스

질문 분석 후 적절한 에이전트 선택:
- 단일: execution_order: ["mcp1"]
- 복합 (순차): execution_order: ["mcp1", "mcp2"]
- 복합 (역순): execution_order: ["mcp2", "mcp1"]
- 일반 대화: execution_order: []

실행 순서 결정 원칙:
1. 데이터 수집이 먼저 필요하면 해당 에이전트를 먼저 실행
2. 수집된 데이터를 분석/해석하는 에이전트는 나중에 실행
3. 의존성을 명시하여 이전 결과를 활용하도록 지시
```

## 🧪 테스트 시나리오

### 시나리오 1: 2개 에이전트 (법률 + 판례)

```bash
# .env
MCP_SERVER_1_NAME=legal
MCP_SERVER_2_NAME=precedent

# 테스트
You> 근로기준법 설명하고 관련 판례 찾아줘
→ execution_order: ["legal", "precedent"]
```

### 시나리오 2: 3개 에이전트 (날씨 + 뉴스 + 주식)

```bash
# .env
MCP_SERVER_1_NAME=weather
MCP_SERVER_2_NAME=news
MCP_SERVER_3_NAME=stock

# 테스트
You> 오늘 날씨, 관련 뉴스, 그리고 우산 회사 주식 정보 알려줘
→ execution_order: ["weather", "news", "stock"]
```

### 시나리오 3: 역순 실행 (판례 → 법률)

```bash
# 테스트
You> 최근 부당해고 사례 찾고, 관련 법 조항 설명해줘
→ execution_order: ["precedent", "legal"]
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

- ✅ MCP 서버 개수 무제한
- ✅ 도메인 제약 없음
- ✅ 동적 에이전트 생성
- ✅ 도구 기반 역할 자동 추론 ✨
- ✅ 자동 라우팅
- ✅ 순서 최적화
- ✅ 의존성 관리
- ✅ 컨텍스트 전달 (이전 결과 활용)

## 🆚 비교

| 기능 | multi_agent_system.py | dynamic_multi_agent.py |
|------|----------------------|------------------------|
| 도메인 | 법률/판례 고정 | 범용 ✅ |
| 에이전트 | LegalAgent, PrecedentAgent | 동적 생성 ✅ |
| 프롬프트 | 법률 용어 포함 | 도메인 독립적 ✅ |
| 역할 인식 | 하드코딩 | 도구 기반 추론 ✨ |
| MCP 추가 | 코드 수정 | .env만 ✅ |
| 확장성 | 낮음 | 높음 ✅ |

## 📝 다음 단계

- [x] 에이전트 선택 시 도구 목록 참고 ✅
- [x] 도구 기반 역할 자동 추론 ✅
- [ ] 에이전트 설명 메타데이터 추가 (각 MCP 서버 설명)
- [ ] 병렬 실행 지원 (독립적인 에이전트)
- [ ] 에이전트 간 데이터 공유 최적화
- [ ] 웹 UI 추가
- [ ] 대화 히스토리 관리
- [ ] 캐싱 메커니즘 (반복 질문 최적화)

## 📄 라이선스

MIT License

