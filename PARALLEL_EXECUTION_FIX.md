# 🔧 병렬 실행 문제 수정

## 문제 상황

```
🎯 판단 결과:
   질문 유형: parallel
   실행 순서: (law-search, precedent-search)
   - law-search: 교통사고 관련 12대 중과실에 대한 법률 조문...
   - precedent-search: 12대 중과실과 관련된 대표적인 판례...

🚀 [Step 2] 2개 에이전트 병렬 처리: law-search, precedent-search
⚠️  에이전트 'law-search' not found, skipping...
⚠️  에이전트 'precedent-search' not found, skipping...
```

## 원인 분석

Agent A(QuestionUnderstandingAgent)가 `execution_order`에 **도구 이름**을 넣고 있습니다.
- ❌ 잘못된 값: `law-search`, `precedent-search` (도구 이름)
- ✅ 올바른 값: `mcp1`, `mcp2`, `mcp3` (에이전트 이름)

## 수정 내용

### 1. Agent A 프롬프트 강화

```python
**⚠️ 매우 중요 - execution_order 작성 규칙:**
1. execution_order에는 **반드시 에이전트 이름만** 사용하세요
2. 사용 가능한 에이전트 이름: mcp1, mcp2, mcp3
3. 도구 이름을 절대 사용하지 마세요

**잘못된 예시 (절대 사용 금지):**
❌ execution_order: [["law-search", "precedent-search"]]  <- 도구 이름
❌ execution_order: [["web-search"]]  <- 도구 이름

**올바른 예시:**
✅ execution_order: [["mcp1", "mcp2"]]  <- 에이전트 이름
✅ execution_order: [["mcp1"]]  <- 에이전트 이름
```

### 2. 에이전트와 도구의 관계 명확화

```
시스템 구조:
┌─────────────┐
│  mcp1       │ <- 에이전트 이름 (execution_order에 사용)
│  └─ tools:  │
│     - law-search       <- 도구 이름 (execution_order에 사용 금지)
│     - statute-lookup   <- 도구 이름
└─────────────┘

┌─────────────┐
│  mcp2       │ <- 에이전트 이름 (execution_order에 사용)
│  └─ tools:  │
│     - precedent-search <- 도구 이름 (execution_order에 사용 금지)
│     - case-lookup      <- 도구 이름
└─────────────┘
```

## 기대 결과

수정 후 올바른 출력:

```
🎯 판단 결과:
   질문 유형: parallel
   실행 순서: (mcp1, mcp2)
   - mcp1: 교통사고 관련 12대 중과실에 대한 법률 조문...
   - mcp2: 12대 중과실과 관련된 대표적인 판례...

🚀 [Step 2] 2개 에이전트 병렬 처리: mcp1, mcp2
──────────────────────────────────────────────────────────────────────
  - MCP1Agent
    질문: 교통사고 관련 12대 중과실에 대한 법률 조문...
  - MCP2Agent
    질문: 12대 중과실과 관련된 대표적인 판례...

✅ 병렬 처리 완료 (3.45초)
  ✓ MCP1Agent: 성공
  ✓ MCP2Agent: 성공
```

## 테스트 방법

1. 시스템 재시작:
```bash
python dynamic_multi_agent.py
```

2. 테스트 질문:
```
You> 12대 중과실은 뭐야?
```

3. 확인 사항:
- ✅ `실행 순서: (mcp1, mcp2)` 형태로 에이전트 이름이 표시되는지 확인
- ✅ `⚠️ 에이전트 'xxx' not found` 에러가 발생하지 않는지 확인
- ✅ 병렬 처리가 정상적으로 완료되는지 확인

## 추가 개선 사항

만약 여전히 문제가 발생한다면, Agent A의 temperature를 더 낮추거나, few-shot 예시를 추가할 수 있습니다.

```python
# 옵션 1: temperature 낮추기
kwargs = {
    "model": self.deployment,
    "messages": messages,
    "temperature": 0.0,  # 0.2 -> 0.0으로 변경
}

# 옵션 2: few-shot 예시 추가
few_shot_examples = [
    {
        "role": "user",
        "content": "12대 중과실이 뭐야?"
    },
    {
        "role": "assistant",
        "content": '''```json
{
  "keywords": ["12대 중과실", "교통사고"],
  "question_type": "parallel",
  "execution_order": [["mcp1", "mcp2"]],
  "queries": {
    "mcp1": "12대 중과실의 법률적 정의를 검색해주세요",
    "mcp2": "12대 중과실 관련 판례를 검색해주세요"
  },
  "analysis": "법률 조문과 판례를 동시에 검색"
}
```'''
    }
]
```

