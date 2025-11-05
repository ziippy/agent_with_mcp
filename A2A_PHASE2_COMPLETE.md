# 🤝 A2A Phase 2 구현 완료 - Peer-to-Peer Agent Communication

## 📅 날짜: 2025-11-05

---

## 🎯 A2A Phase 2란?

**Agent-to-Agent Peer-to-Peer Communication**

에이전트들이 서로 직접 협력을 요청할 수 있는 **자율적인 협업 시스템**

---

## 📊 Phase 1 vs Phase 2

### A2A Phase 1 (이전)

```
User → Agent A (라우팅)
         ↓
    Agent B, C, D (독립 작업)
```

**특징:**
- ✅ Agent A가 중앙 집중식 라우팅
- ❌ Agent B/C/D는 서로 협력 불가
- ❌ Top-Down 방식

### A2A Phase 2 (개선) ✨

```
User → Agent A (라우팅)
         ↓
    Agent B ←→ Agent C ←→ Agent D
    (Peer-to-Peer 협력)
```

**특징:**
- ✅ Agent A가 초기 라우팅
- ✅ Agent B/C/D가 **서로 협력 요청** 가능
- ✅ Peer-to-Peer 방식
- ✅ 더 유연하고 자율적

---

## 🔧 구현 내용

### 1. ToolBasedAgent 개선

**추가된 속성:**
```python
class ToolBasedAgent:
    def __init__(self, ..., description: str = "", orchestrator=None):
        self.description = description  # 에이전트 설명
        self.orchestrator = orchestrator  # 오케스트레이터 참조
```

**추가된 메서드:**
```python
def set_orchestrator(self, orchestrator):
    """오케스트레이터 설정 (A2A Phase 2)"""
    self.orchestrator = orchestrator
```

### 2. request_agent_help 가상 도구

**도구 정의:**
```python
{
    "type": "function",
    "function": {
        "name": "request_agent_help",
        "description": "다른 전문 에이전트에게 협력을 요청합니다",
        "parameters": {
            "type": "object",
            "properties": {
                "target_agent": {
                    "type": "string",
                    "enum": ["laws", "search", ...],
                    "description": "협력을 요청할 에이전트 이름"
                },
                "task": {
                    "type": "string",
                    "description": "요청할 작업 내용"
                },
                "reason": {
                    "type": "string",
                    "description": "협력이 필요한 이유"
                }
            },
            "required": ["target_agent", "task"]
        }
    }
}
```

### 3. 협력 요청 처리 로직

**process_with_tools에서 처리:**
```python
if tool_name == "request_agent_help":
    target_agent_name = args.get("target_agent")
    task = args.get("task")
    
    # 대상 에이전트에게 작업 위임
    target_agent = self.orchestrator.specialist_agents[target_agent_name]
    agent_response = await target_agent.process_with_tools(task, context)
    
    # 결과 반환
    tool_results.append({
        "tool_call_id": tc.id,
        "content": f"[{target_agent_name} 에이전트 응답]\n{agent_response.content}"
    })
```

---

## 🎨 동작 예시

### 시나리오: "교통사고 관련 법률을 찾고, 최근 뉴스도 검색해줘"

#### Phase 1 방식 (이전)

```
User → Agent A
         ↓
    [laws, search] 병렬 실행
         ↓
    Agent A가 결과 통합
```

**문제:**
- laws 에이전트가 "최근 뉴스도 필요하네..."라고 생각해도 직접 요청 불가

#### Phase 2 방식 (개선) ✨

```
User → Agent A
         ↓
    laws 에이전트 실행
         ↓
    laws: "법률은 찾았는데, 관련 뉴스가 필요해"
         ↓
    laws → request_agent_help(target="search", task="최근 교통사고 뉴스")
         ↓
    search 에이전트 실행
         ↓
    laws: 법률 + 뉴스를 통합하여 응답
```

**장점:**
- ✅ laws 에이전트가 **자율적으로** 협력 결정
- ✅ Agent A의 개입 없이 **Peer-to-Peer** 협력
- ✅ 더 유연하고 지능적

---

## 📋 실행 로그 예시

### Phase 2 협력 요청

```
🔧 [Step 2] LAWSAgent 처리
────────────────────────────────────────────────────────────────────

  📋 [LAWSAgent] 사용 가능한 도구 Schema:
    🤝 A2A Phase 2: request_agent_help 도구 활성화
       협력 가능 에이전트: search, precedent

  🔧 [LAWSAgent] Tool calls: 2
    🔍 [laws__law-search] Input arguments:
    {
      "query": "교통사고 법률",
      "search_type": "1"
    }
    ✅ Tool laws__law-search executed
    
    🔍 [request_agent_help] Input arguments:
    {
      "target_agent": "search",
      "task": "최근 교통사고 관련 뉴스를 검색해주세요",
      "reason": "법률 정보와 함께 최신 사례를 제공하기 위해"
    }
    🤝 [LAWSAgent] → [search] 협력 요청
       작업: 최근 교통사고 관련 뉴스를 검색해주세요
       이유: 법률 정보와 함께 최신 사례를 제공하기 위해
    
  📋 [SEARCHAgent] 사용 가능한 도구 Schema:
  
  🔧 [SEARCHAgent] Tool calls: 1
    🔍 [search__web-search] Input arguments:
    {
      "query": "최근 교통사고 뉴스"
    }
    ✅ Tool search__web-search executed
    
    ✅ [search] 협력 완료
    
✅ 처리 완료 (5.23초)
```

---

## 🔑 핵심 특징

### 1. 자율성 (Autonomy)

**에이전트가 스스로 판단:**
- "이 작업은 내 전문 영역이 아니네"
- "search 에이전트에게 도움을 요청해야겠어"
- `request_agent_help` 도구 호출

### 2. Peer-to-Peer

**중앙 집중식 → 분산형:**
- Phase 1: Agent A가 모든 것을 결정
- Phase 2: Agent B/C/D가 **서로 직접** 협력

### 3. 재귀적 협력

**다단계 협력 가능:**
```
User → Agent A
         ↓
    Agent B (laws)
         ↓ request_agent_help
    Agent C (search)
         ↓ request_agent_help
    Agent D (precedent)
```

### 4. 컨텍스트 전달

**협력 시 context 자동 전달:**
```python
agent_response = await target_agent.process_with_tools(task, context)
```
- 이전 결과를 자동으로 전달
- 협력 에이전트가 컨텍스트를 활용

---

## 🎯 사용 시나리오

### 시나리오 1: 법률 + 뉴스

**질문:** "교통사고 법률을 알려주고, 관련 최근 뉴스도 찾아줘"

**Phase 1:**
```
Agent A: laws, search 병렬 실행
```

**Phase 2:**
```
Agent A: laws 실행
  ↓
laws: law-search 실행
  ↓
laws: "뉴스도 필요하네"
  ↓
laws: request_agent_help(target="search")
  ↓
search: web-search 실행
  ↓
laws: 법률 + 뉴스 통합 응답
```

### 시나리오 2: 법률 → 판례 → 뉴스

**질문:** "교통사고 법률과 판례, 그리고 최신 뉴스를 모두 알려줘"

**Phase 2:**
```
Agent A: laws 실행
  ↓
laws: law-search 실행
  ↓
laws: request_agent_help(target="precedent", task="판례 검색")
  ↓
precedent: precedent-search 실행
  ↓
precedent: request_agent_help(target="search", task="뉴스 검색")
  ↓
search: web-search 실행
  ↓
laws: 법률 + 판례 + 뉴스 통합 응답
```

---

## 🔧 설정 방법

### .env 설정 (변경 없음)

```env
# LLM 설정
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=...

# MCP 서버
MCP_SERVER_1_URL=http://localhost:8001/
MCP_SERVER_1_NAME=laws

MCP_SERVER_2_URL=http://localhost:8002/
MCP_SERVER_2_NAME=search
```

### 코드 변경

- **agents.py**: ToolBasedAgent에 A2A Phase 2 기능 추가
- **orchestrator.py**: 에이전트 생성 시 orchestrator 전달

---

## 📊 Phase 1 vs Phase 2 비교

| 항목 | Phase 1 | Phase 2 |
|------|---------|---------|
| **라우팅** | Agent A 중앙 집중식 | Agent A + Peer-to-Peer |
| **협력 방식** | Top-Down | Peer-to-Peer |
| **자율성** | 낮음 | 높음 ✅ |
| **유연성** | 중간 | 높음 ✅ |
| **복잡도** | 낮음 | 중간 |
| **에이전트 간 통신** | 없음 | 있음 ✅ |

---

## ⚠️ 주의사항

### 1. 무한 루프 방지

**문제:**
```
Agent A → Agent B (협력 요청)
Agent B → Agent A (협력 요청)
Agent A → Agent B (협력 요청)
...
```

**해결:**
- MAX_ITERATIONS로 최대 반복 횟수 제한
- 협력 체인이 너무 길어지면 자동 중단

### 2. 컨텍스트 크기

**문제:**
- 협력할 때마다 context가 누적
- LLM context window 초과 가능

**해결:**
- 중요한 정보만 요약하여 전달
- context 크기 모니터링

---

## ✅ 테스트

### 테스트 케이스 1: 직접 협력

**질문:** "교통사고 법률을 찾고, 관련 뉴스도 검색해줘"

**예상 동작:**
1. Agent A → laws 실행
2. laws → law-search 실행
3. laws → request_agent_help(target="search")
4. search → web-search 실행
5. laws → 결과 통합 ✅

### 테스트 케이스 2: 다단계 협력

**질문:** "법률, 판례, 뉴스를 모두 찾아줘"

**예상 동작:**
1. Agent A → laws 실행
2. laws → request_agent_help(target="precedent")
3. precedent → request_agent_help(target="search")
4. 결과 역순으로 전달 ✅

---

## 🎓 결론

**A2A Phase 2 구현 완료!**

- ✅ 에이전트 간 Peer-to-Peer 협력
- ✅ `request_agent_help` 가상 도구
- ✅ 자율적 협력 결정
- ✅ 재귀적 협력 지원
- ✅ 컨텍스트 자동 전달

**이제 에이전트들이 서로 협력하며 복잡한 작업을 수행할 수 있습니다!** 🤝🎉

---

**A2A Phase 1 → Phase 2 업그레이드 완료!**

---

## 🚀 A2A Phase 3 로드맵

### Phase 3 예정: **Dynamic Team Formation & Memory**

```
User → Orchestrator
         ↓
    Agent A ←→ Agent B ←→ Agent C
         ↓
    [Shared Memory]
         ↓
    [Dynamic Team Formation]
```

**핵심 개선사항:**

#### 1. 공유 메모리 (Shared Memory)
- 모든 에이전트가 공통 메모리 공간 공유
- 중복 작업 방지
- 학습된 패턴 재사용

```python
class SharedMemory:
    def __init__(self):
        self.cache = {}  # 작업 결과 캐싱
        self.history = []  # 협력 이력

    async def get_or_execute(self, key, func):
        if key in self.cache:
            return self.cache[key]  # 캐시 히트
        result = await func()
        self.cache[key] = result
        return result
```

#### 2. 동적 팀 구성 (Dynamic Team Formation)
- 작업에 따라 최적의 에이전트 팀 자동 구성
- 에이전트가 다른 에이전트를 "추천"

```python
class TeamFormation:
    async def form_team(self, task: str) -> List[Agent]:
        team_plan = await llm.plan_team(task, available_agents)
        return [agents[name] for name in team_plan]
```

#### 3. 협상 프로토콜 (Negotiation Protocol)
- 에이전트 간 협상 메커니즘
- "내가 할게" vs "네가 더 적합해"

```python
class NegotiationProtocol:
    async def negotiate(self, agents: List[Agent], task: str):
        bids = await asyncio.gather(*[
            agent.bid_for_task(task) for agent in agents
        ])
        winner = max(bids, key=lambda b: b.confidence)
        return winner.agent
```

#### 4. Learning & Feedback Loop
- 협력 패턴 학습
- 성공/실패 피드백
- 최적의 협력 경로 자동 발견

---

## ⚠️ A2A Phase 2의 단점 및 개선 방안

### 1. 🔄 무한 루프 위험

**문제:**
```python
Agent A → Agent B (협력 요청)
Agent B → Agent A (다시 협력 요청)
Agent A → Agent B (또 협력 요청)
...
```

**현재 해결책:**
- ✅ MAX_ITERATIONS로 최대 반복 횟수 제한 (기본값: 10)
- ✅ 협력 체인이 길어지면 자동 중단

**Phase 3 개선:**
- Circular dependency 자동 감지
- 협력 이력 추적 및 중복 방지

---

### 2. 💰 비용 증가

**문제:**
- 협력할 때마다 추가 LLM API 호출 발생
- Phase 1: 1번 호출 → Phase 2: 3-5번 호출 가능
- API 비용 **3-5배 증가** 위험

**비용 비교:**
```
Phase 1:
  User → laws (1 call) = $0.01

Phase 2:
  User → laws (1 call)
    → laws → search (1 call)
      → search → precedent (1 call)
  Total = $0.03 (3배)
```

**현재 완화 방법:**
- 협력 필요성을 신중히 판단하도록 System Prompt 설계
- MAX_ITERATIONS로 제한

**Phase 3 개선:**
- ✅ Shared Memory로 결과 캐싱 → 중복 호출 방지
- ✅ 비용 임계값 설정 (예: 최대 $0.05)
- ✅ 비용 추적 및 알림

---

### 3. ⏱️ 응답 시간 증가

**문제:**
- 직렬 협력 시 대기 시간 누적
- laws → search → precedent (각 3초)
- Total: **9초** vs Phase 1의 **3초** (병렬)

**Phase 3 개선:**
- ✅ 병렬 협력 지원 (현재는 순차만)
- ✅ 비동기 처리 최적화
- ✅ 타임아웃 설정

```python
@timeout(30)  # 30초 제한
async def request_agent_help(...):
    ...
```

---

### 4. 🧩 컨텍스트 폭발 (Context Explosion)

**문제:**
```
Agent A context: 1000 tokens
  → Agent B context: 1000 + A's result (500) = 1500
    → Agent C context: 1500 + B's result (500) = 2000
      → LLM context limit 초과!
```

**현재 완화 방법:**
- 이전 결과를 요약하여 전달

**Phase 3 개선:**
- ✅ 자동 Context 요약 (Summarization)
- ✅ 중요 정보만 선별 전달
- ✅ Sliding window 방식

```python
def summarize_context(context: str) -> str:
    if len(context) > 2000:
        return llm.summarize(context, max_tokens=500)
    return context
```

---

### 5. 🎯 잘못된 에이전트 선택

**문제:**
- laws가 "뉴스가 필요해" → search 호출 ✅ (정상)
- laws가 "날씨가 필요해" → 적절한 에이전트 없음! ❌

**Phase 2의 한계:**
```python
"target_agent": {
    "enum": ["laws", "search", "precedent", ...]  # 고정된 목록
}
```
- 새로운 에이전트 추가 시 코드 수정 필요

**Phase 3 개선:**
```python
# Dynamic agent discovery
available_agents = orchestrator.discover_agents()
best_agent = llm.select_best_agent(task, available_agents)
```

---

### 6. 🤔 불필요한 협력

**문제:**
- 단순한 작업인데도 협력 요청
- "1+1은?" → calculator → (왜 search에게도 물어봐??)

**원인:**
- LLM이 과도하게 협력 선호
- "안전하게 다른 에이전트에게도 물어보자" 경향

**개선 방법:**
- ✅ System Prompt 최적화
- ✅ 협력 필요성 판단 기준 명확화
- ✅ Few-shot 예제 추가

---

### 7. 🐛 디버깅 어려움

**문제:**
```
User → Agent A → Agent B → Agent C
                ↓         ↓
             Error?    Error?
```
- 어디서 에러가 났는지 추적 어려움
- 협력 체인이 길수록 복잡

**현재 해결책:**
- ✅ 상세한 로깅 (🤝 협력 요청 표시)
- ✅ 각 단계별 결과 출력

**Phase 3 개선:**
- Tracing ID 추가
- Visualization 도구 (협력 그래프)
- 에러 위치 자동 추적

---

### 8. 💾 중복 작업

**문제:**
```
Agent A: law-search("교통사고")
Agent B: law-search("교통사고")  # 중복!
```
- 같은 작업을 여러 에이전트가 수행
- 비효율적

**Phase 3 해결:**
- ✅ Shared Memory로 결과 캐싱
- ✅ 작업 이력 공유
- ✅ 중복 감지 및 재사용

---

## 📊 Phase 비교표

| 항목 | Phase 1 | Phase 2 | Phase 3 (예정) |
|------|---------|---------|----------------|
| **협력 방식** | Top-Down | Peer-to-Peer | Dynamic Team |
| **메모리** | 없음 | 없음 | Shared Memory ✅ |
| **비용** | 낮음 ✅ | 중간 ⚠️ | 중간 (캐싱으로 절감) |
| **속도** | 빠름 ✅ | 느림 ⚠️ | 중간 (병렬 최적화) |
| **유연성** | 낮음 | 높음 ✅ | 매우 높음 ✅ |
| **자율성** | 낮음 | 높음 ✅ | 매우 높음 ✅ |
| **디버깅** | 쉬움 ✅ | 어려움 ⚠️ | 중간 (트레이싱) |
| **중복 방지** | 없음 | 없음 | 있음 ✅ |
| **학습 능력** | 없음 | 없음 | 있음 ✅ |

---

## 🎯 Phase 2 즉시 개선 가능한 사항

### 1. 협력 제한 강화

```python
# .env 추가
MAX_COOPERATION_DEPTH=3  # 최대 3단계 협력만 허용
```

### 2. 비용 모니터링

```python
def track_api_calls(agent_name, cost):
    total_cost += cost
    logger.info(f"💰 [{agent_name}] API cost: ${cost:.4f}")
    logger.info(f"💰 Total cost: ${total_cost:.4f}")
    
    if total_cost > MAX_COST:
        raise CostLimitExceeded()
```

### 3. Context 요약

```python
if context and len(context) > 2000:
    context = summarize_context(context)
```

### 4. 타임아웃 설정

```python
import asyncio

try:
    result = await asyncio.wait_for(
        target_agent.process_with_tools(task, context),
        timeout=30.0  # 30초 제한
    )
except asyncio.TimeoutError:
    return "협력 요청 시간 초과"
```

---

## 💡 결론

### A2A Phase 2의 트레이드오프:

**장점:**
- ✅ 유연성과 자율성 크게 향상
- ✅ 복잡한 작업 수행 가능
- ✅ Peer-to-Peer 협력

**단점:**
- ⚠️ 비용 증가 (3-5배)
- ⚠️ 응답 시간 증가
- ⚠️ 컨텍스트 관리 복잡
- ⚠️ 디버깅 어려움

**하지만:**

✅ Phase 3에서 대부분의 단점 해결 예정  
✅ Shared Memory로 중복 작업 방지  
✅ Dynamic Team Formation으로 최적화  
✅ 학습 메커니즘으로 지능적 협력

**Phase 2는 Peer-to-Peer의 첫 걸음이며, Phase 3에서 완성됩니다!** 🎯🚀

