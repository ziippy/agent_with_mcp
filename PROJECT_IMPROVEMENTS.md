# 🚀 프로젝트 코드 개선 제안

## 📅 날짜: 2025-11-05

---

## ✅ 개선 사항 요약

1. **MCP 서버 동적 활성화/비활성화**
   - `.env` 파일에서 `MCP_SERVER_X_ENABLED` 플래그로 서버 연결 여부 제어

2. **LLM API 재시도 횟수 설정**
   - `.env` 파일에서 `LLM_MAX_RETRIES`로 API 호출 재시도 횟수 설정

3. **에이전트 최대 반복 횟수 설정**
   - `.env` 파일에서 `MAX_ITERATIONS`로 에이전트 최대 반복 횟수 설정

---

## 🔧 1. MCP 서버 동적 활성화/비활성화

### 문제점

- 특정 MCP 서버를 테스트에서 제외하려면 `.env` 파일에서 해당 라인을 주석 처리해야 함
- 불편하고 실수하기 쉬움

### 개선 내용

**dynamic_multi_agent_v2.py:**
```python
# .env에서 ENABLED 플래그 확인
enabled_key = f"MCP_SERVER_{server_index}_ENABLED"
is_enabled = os.environ.get(enabled_key, "true").lower() in ("true", "1", "yes")

if not is_enabled:
    print(f"MCP Server '{server_name}' is disabled. Skipping.")
    server_index += 1
    continue
```

**.env 설정 예시:**
```env
# 서버 1: 활성화
MCP_SERVER_1_URL=http://localhost:8001/
MCP_SERVER_1_NAME=laws
MCP_SERVER_1_ENABLED=true

# 서버 2: 비활성화
MCP_SERVER_2_URL=http://localhost:8002/
MCP_SERVER_2_NAME=search
MCP_SERVER_2_ENABLED=false
```

**장점:**
- ✅ 주석 처리 없이 서버를 켜고 끌 수 있음
- ✅ 테스트 및 디버깅 시 편리

---

## 🔧 2. LLM API 재시도 횟수 설정

### 문제점

- LLM API 호출 실패 시 재시도 횟수가 코드에 하드코딩되어 있음
- 네트워크가 불안정할 때 유연하게 대처하기 어려움

### 개선 내용

**llm_client.py:**
```python
class LLMClient:
    def __init__(self, ...):
        # .env에서 재시도 횟수 읽기 (기본값: 3)
        self.max_retries = int(os.environ.get("LLM_MAX_RETRIES", 3))
        
        # 각 클라이언트 초기화 시 max_retries 전달
        self.client = OpenAI(
            ...,
            max_retries=self.max_retries
        )
```

**.env 설정 예시:**
```env
# LLM API 호출 재시도 횟수
LLM_MAX_RETRIES=5
```

**장점:**
- ✅ 네트워크 환경에 따라 재시도 횟수 조절 가능
- ✅ 안정성 향상

---

## 🔧 3. 에이전트 최대 반복 횟수 설정

### 문제점

- 에이전트가 도구를 반복적으로 호출하는 최대 횟수가 코드에 하드코딩되어 있음
- 복잡한 작업 시 반복 횟수가 부족할 수 있음

### 개선 내용

**agents.py (ToolBasedAgent):**
```python
class ToolBasedAgent:
    async def process_with_tools(self, ...):
        # .env에서 최대 반복 횟수 읽기 (기본값: 10)
        max_iterations = int(os.environ.get("MAX_ITERATIONS", 10))
        
        for iteration in range(max_iterations):
            # ...
```

**.env 설정 예시:**
```env
# 에이전트 최대 반복 횟수
MAX_ITERATIONS=15
```

**장점:**
- ✅ 작업의 복잡도에 따라 반복 횟수 조절 가능
- ✅ 무한 루프 방지 및 유연성 확보

---

## 🎨 전체 개선 효과

### 1. 설정의 중앙화
- 주요 설정값(`.env`)을 한 곳에서 관리
- 코드 수정 없이 동작 변경 가능

### 2. 유연성 향상
- 서버 활성화/비활성화
- 재시도 횟수 조절
- 반복 횟수 조절

### 3. 안정성 및 견고성
- LLM API 호출 안정성 향상
- 테스트 및 디버깅 용이성

---

## 🚀 적용 방법

### 1. 코드 수정

- `dynamic_multi_agent_v2.py` - MCP 서버 활성화 로직 추가
- `llm_client.py` - `max_retries` 설정 추가
- `agents.py` - `max_iterations` 설정 추가

### 2. .env 파일 업데이트

```env
# 추가된 설정
LLM_MAX_RETRIES=5
MAX_ITERATIONS=15

# MCP 서버 설정
MCP_SERVER_1_URL=...
MCP_SERVER_1_NAME=...
MCP_SERVER_1_ENABLED=true

MCP_SERVER_2_URL=...
MCP_SERVER_2_NAME=...
MCP_SERVER_2_ENABLED=false
```

---

## ✅ 결론

**이제 프로젝트의 주요 동작을 `.env` 파일에서 쉽게 제어할 수 있습니다!**

- ✅ 서버 On/Off
- ✅ LLM 재시도 횟수
- ✅ 에이전트 반복 횟수

**코드가 더 유연하고 견고해졌습니다!** 🎉

