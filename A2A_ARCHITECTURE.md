# 🤝 A2A (Agent-to-Agent) 아키텍처 설명

## 📋 개요

`dynamic_multi_agent.py`가 **진정한 A2A (Agent-to-Agent) 통신 구조**로 개선되었습니다.

---

## 🔄 변경 전 vs 변경 후

### ❌ 변경 전: 중앙 집중식 관리

```
Agent A
  ├─ 모든 MCP Tool의 상세 정보 확인
  ├─ "laws__law-search", "laws__statute-lookup" 등 미시적 관리
  ├─ Agent B에게 구체적인 도구 사용 지시
  └─ Agent B는 지시받은 대로만 실행
```

**문제점:**
- Agent A가 너무 많은 정보를 알아야 함
- 확장성 떨어짐 (새 도구 추가 시 Agent A 프롬프트 수정 필요)
- Agent B, C, D는 자율성이 없음

---

### ✅ 변경 후: A2A 자율 협력

```
Agent A (오케스트레이터)
  ├─ Agent B: "법률 검색 전문 에이전트" (역할만 파악)
  ├─ Agent C: "판례 검색 전문 에이전트" (역할만 파악)
  └─ Agent D: "웹 검색 전문 에이전트" (역할만 파악)
      ↓
사용자: "12대 중과실이 뭐야?"
      ↓
Agent A: "Agent B와 C에게 물어보면 되겠군"
      ↓
Agent B (자율 판단)
  ├─ 내 도구: law-search, statute-lookup, legal-interpretation
  ├─ 판단: "law-search를 사용하면 되겠어"
  └─ 실행: law-search("12대 중과실")
      ↓
Agent C (자율 판단)
  ├─ 내 도구: precedent-search, case-lookup
  ├─ 판단: "precedent-search를 사용하면 되겠어"
  └─ 실행: precedent-search("12대 중과실 판례")
      ↓
Agent A: 두 결과를 통합하여 최종 답변
```

**장점:**
- Agent A는 **고수준 라우팅만** 담당 (역할 기반)
- Agent B, C, D는 **자율적으로** 도구 선택
- 확장성 우수 (새 도구 추가 시 코드 수정 불필요)
- **진정한 A2A 통신**

---

## 🏗️ 아키텍처 상세

### 1. Agent A (QuestionUnderstandingAgent)

**역할:**
- 사용자 질문 분석
- 에이전트 역할 기반 라우팅
- 실행 전략 결정 (병렬/순차)

**입력 정보:**
```python
agent_descriptions = {
    "laws": "법률, 법령, 조문 검색 및 해석을 담당하는 법률 전문 에이전트",
    "precedent": "판례, 사례, 판결문 검색 및 분석을 담당하는 판례 전문 에이전트",
    "search": "웹 검색, 뉴스 검색, 정보 수집을 담당하는 검색 전문 에이전트"
}
```

**출력 (라우팅 계획):**
```json
{
  "execution_order": [["laws", "precedent"]],
  "queries": {
    "laws": "12대 중과실의 법률적 정의와 관련 조문을 찾아주세요",
    "precedent": "12대 중과실과 관련된 판례를 찾아주세요"
  }
}
```

**중요:** Agent A는 **도구 목록을 모릅니다!** 역할 설명만 보고 판단합니다.

---

### 2. Agent B, C, D (ToolBasedAgent)

**역할:**
- 자신의 전문 분야 작업 수행
- 자율적으로 도구 선택
- (향후) 다른 에이전트에게 협력 요청 가능

**구조:**
```python
class ToolBasedAgent:
    def __init__(self, name, role, description, tools, orchestrator):
        self.name = name  # "LAWSAgent"
        self.role = role  # "laws 전문 에이전트"
        self.description = description  # "법률, 법령, 조문 검색..."
        self.tools = tools  # [law-search, statute-lookup, ...]
        self.orchestrator = orchestrator  # A2A를 위한 참조
```

**작업 흐름:**
1. Agent A로부터 작업 지시 받음: "12대 중과실의 법률적 정의를 찾아주세요"
2. 자신의 도구 목록 확인: `[law-search, statute-lookup, legal-interpretation]`
3. LLM이 자율적으로 판단: "law-search를 사용하면 되겠다"
4. 도구 실행: `law-search("12대 중과실")`
5. 결과 반환

---

## 🎯 핵심 개선 사항

### 1. 에이전트 설명 자동 생성

**`_generate_agent_description()` 메서드:**
```python
def _generate_agent_description(self, server_name: str, tool_names: List[str]) -> str:
    """도구 이름을 기반으로 에이전트 역할 설명 자동 생성"""
    keywords = set()
    for tool_name in tool_names:
        parts = tool_name.lower().replace('-', ' ').split()
        keywords.update(parts)
    
    if 'law' in keywords or 'legal' in keywords:
        return "법률, 법령, 조문 검색 및 해석을 담당하는 법률 전문 에이전트"
    elif 'precedent' in keywords or 'case' in keywords:
        return "판례, 사례, 판결문 검색 및 분석을 담당하는 판례 전문 에이전트"
    # ... 기타 도메인
```

**효과:**
- MCP 서버를 추가하면 **자동으로 에이전트 설명 생성**
- 도구 이름만으로 역할 추론
- 수동 설정 불필요

---

### 2. Agent A 프롬프트 간소화

**변경 전:**
```
사용 가능한 에이전트 및 도구:
  - laws: law-search, statute-lookup, legal-interpretation, ... (총 15개)
  - precedent: precedent-search, case-lookup, judgment-analysis, ... (총 12개)
  - search: web-search, news-fetch, ... (총 8개)

각 에이전트가 제공하는 도구를 보고 어떤 역할을 하는지 추론하세요.
```

**변경 후:**
```
🤖 사용 가능한 에이전트:
  - laws: 법률, 법령, 조문 검색 및 해석을 담당하는 법률 전문 에이전트
  - precedent: 판례, 사례, 판결문 검색 및 분석을 담당하는 판례 전문 에이전트
  - search: 웹 검색, 뉴스 검색, 정보 수집을 담당하는 검색 전문 에이전트

⚠️ A2A 원칙:
- 에이전트의 역할과 설명만 보고 라우팅
- 각 에이전트가 어떤 도구를 가지고 있는지는 알 필요 없음
- 각 에이전트는 자율적으로 도구 선택
```

**효과:**
- 프롬프트 길이 대폭 감소
- 토큰 사용량 절감
- 더 명확한 역할 기반 라우팅

---

### 3. 에이전트 자율성 강화

**ToolBasedAgent 시스템 프롬프트:**
```
당신은 {role}입니다.

역할 설명: {description}

A2A (Agent-to-Agent) 기능:
1. 도구 사용: 자신이 가진 도구를 사용하여 직접 작업 수행
2. 에이전트 협력: 다른 에이전트의 도움이 필요하면 협력 요청

중요 원칙:
- 자신의 전문 영역 내 작업은 자신의 도구로 해결
- 다른 전문 영역의 작업이 필요하면 다른 에이전트에게 협력 요청
```

**효과:**
- 에이전트가 자율적으로 판단
- 필요시 다른 에이전트에게 협력 요청 가능 (향후 구현)
- 진정한 Multi-Agent 시스템

---

## 🚀 실행 흐름 예시

### 질문: "12대 중과실이 뭐야?"

```
[1] 사용자 → Agent A
    질문: "12대 중과실이 뭐야?"

[2] Agent A (라우팅)
    입력: agent_descriptions = {
        "laws": "법률 전문 에이전트",
        "precedent": "판례 전문 에이전트"
    }
    
    판단: "법률 정의와 판례가 필요하니 laws와 precedent 모두 호출"
    
    출력: {
        "execution_order": [["laws", "precedent"]],  # 병렬 실행
        "queries": {
            "laws": "12대 중과실의 법률적 정의를 찾아주세요",
            "precedent": "12대 중과실 관련 판례를 찾아주세요"
        }
    }

[3] Agent A → Agent B (laws) & Agent C (precedent) 병렬 실행
    
    Agent B (laws):
        작업 지시: "12대 중과실의 법률적 정의를 찾아주세요"
        도구 확인: [law-search, statute-lookup, legal-interpretation]
        자율 판단: "law-search를 사용하면 되겠다"
        실행: law-search("12대 중과실")
        결과: "12대 중과실은 도로교통법 제148조에 정의된..."
    
    Agent C (precedent):
        작업 지시: "12대 중과실 관련 판례를 찾아주세요"
        도구 확인: [precedent-search, case-lookup, judgment-analysis]
        자율 판단: "precedent-search를 사용하면 되겠다"
        실행: precedent-search("12대 중과실")
        결과: "대법원 2022다1234 판결에서..."

[4] Agent B & C → Agent A
    laws 결과: "법률적 정의..."
    precedent 결과: "관련 판례..."

[5] Agent A (통합)
    두 결과를 종합하여 최종 답변 생성
    
    출력: "12대 중과실은 도로교통법에서 정의된 중대한 과실로... 
           관련 판례로는 대법원 판결이 있습니다..."

[6] Agent A → 사용자
    최종 답변 전달
```

---

## 📊 비교 표

| 항목 | 변경 전 | 변경 후 (A2A) |
|------|---------|---------------|
| **Agent A 입력** | 모든 MCP Tool 목록 | 에이전트 역할 설명만 |
| **Agent A 판단 기준** | 도구 이름으로 미시적 판단 | 역할로 거시적 판단 |
| **Agent B/C/D 자율성** | 없음 (지시받은 대로만) | 높음 (도구 선택, 협력 요청) |
| **프롬프트 길이** | 길음 (모든 도구 나열) | 짧음 (역할 설명만) |
| **확장성** | 낮음 (도구 추가 시 수정) | 높음 (자동 설명 생성) |
| **토큰 사용량** | 높음 | 낮음 |
| **A2A 통신** | 부분적 | 완전한 A2A ✅ |

---

## 🔮 향후 개선 방향

### 1. 에이전트 간 직접 협력 (Full A2A)

현재는 오케스트레이터가 중재하지만, 향후 다음과 같이 개선 가능:

```python
# Agent B가 작업 중 Agent C의 도움이 필요한 경우
class ToolBasedAgent:
    async def process_with_tools(self, user_input):
        # ... 작업 수행 중 ...
        
        if self._need_help_from_other_agent():
            # 다른 에이전트에게 직접 협력 요청
            result = await self.orchestrator.request_agent_help(
                target_agent="precedent",
                request="이 법률과 관련된 판례를 찾아주세요"
            )
            # 결과를 받아서 작업 계속
```

### 2. 에이전트 간 대화 (Multi-turn A2A)

```
Agent B: "이 법률 해석이 맞는지 확인해줄래?"
Agent C: "네, 관련 판례를 확인해보겠습니다."
Agent C: "판례를 보니 그 해석이 맞습니다."
Agent B: "고마워, 그럼 이 내용으로 답변할게."
```

### 3. 동적 에이전트 생성

```python
# 특정 작업을 위한 임시 에이전트 생성
temp_agent = orchestrator.create_agent(
    role="특정 케이스 분석 전문",
    tools=["case-analyzer", "law-matcher"]
)
```

---

## 📝 요약

### 현재 구조: **Orchestrated A2A**

- Agent A가 고수준 라우팅 담당
- Agent B, C, D가 자율적으로 도구 선택
- 오케스트레이터가 결과 전달 중재

### 진정한 A2A로의 진화:

1. ✅ **Phase 1 (완료)**: 역할 기반 라우팅
2. 🔄 **Phase 2 (향후)**: 에이전트 간 직접 협력 요청
3. 🔮 **Phase 3 (미래)**: 에이전트 간 자유로운 대화

---

## 🎓 핵심 개념

### A2A (Agent-to-Agent) 란?

**정의:**
에이전트들이 서로 협력하여 복잡한 작업을 수행하는 아키텍처

**핵심 원칙:**
1. **자율성**: 각 에이전트가 자율적으로 판단하고 행동
2. **역할 기반**: 에이전트는 역할로 식별되며, 구체적인 구현은 숨김
3. **협력**: 에이전트 간 협력을 통해 더 복잡한 문제 해결

**현재 구현:**
- ✅ Agent A는 역할만 보고 라우팅
- ✅ Agent B, C, D는 자율적으로 도구 선택
- 🔄 (향후) 에이전트 간 직접 협력

---

이제 `dynamic_multi_agent.py`는 **진정한 A2A 아키텍처**를 구현하고 있습니다! 🎉

