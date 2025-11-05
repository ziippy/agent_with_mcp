# -*- coding: utf-8 -*-
"""
LLM Client Module
범용 LLM 클라이언트 - Azure OpenAI, OpenAI, vLLM, Google Gemini, Anthropic Claude, xAI Grok 지원
"""
import os
from typing import Any, Dict, List, Optional
from openai import AzureOpenAI, OpenAI, BadRequestError

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class ContentFilterError(Exception):
    """LLM 콘텐츠 필터링 에러"""
    def __init__(self, filtered_categories: List[str], original_error: Exception):
        self.filtered_categories = filtered_categories
        self.original_error = original_error
        super().__init__(f"Content filtered: {', '.join(filtered_categories)}")


class LLMClient:
    """범용 LLM 클라이언트 - Azure OpenAI, OpenAI, vLLM, Google Gemini, Anthropic Claude, xAI Grok 지원"""

    def __init__(
        self,
        provider: str = "azure",  # "azure", "openai", "vllm", "google", "anthropic", "xai"
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        # Azure OpenAI 전용
        api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
    ):
        """
        LLM 클라이언트 초기화
        """
        self.provider = provider.lower()
        self.model = model or deployment
        self.max_retries = int(os.environ.get("LLM_MAX_RETRIES", 3))

        try:
            if self.provider == "azure":
                self.client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                    max_retries=self.max_retries,
                )
                self.model = deployment
                print(f"[LLM] Azure OpenAI initialized (deployment: {deployment})")

            elif self.provider == "openai":
                # OpenAI 초기화 - base_url이 없으면 기본값 사용
                init_kwargs = {
                    "api_key": api_key,
                    "max_retries": self.max_retries,
                }
                if base_url:  # base_url이 있을 때만 전달
                    init_kwargs["base_url"] = base_url

                self.client = OpenAI(**init_kwargs)

                # 디버깅 정보 출력
                print(f"[LLM] OpenAI initialized")
                print(f"[LLM]   Model: {model}")
                print(f"[LLM]   API Key: {api_key[:20]}..." if api_key else "[LLM]   API Key: None")
                print(f"[LLM]   Base URL: {base_url if base_url else 'default (https://api.openai.com/v1)'}")

            elif self.provider == "vllm":
                # vLLM은 OpenAI 호환 API를 제공
                # UTF-8 인코딩 문제 해결을 위한 명시적 설정
                import httpx
                import json as json_lib

                # UTF-8을 강제하는 커스텀 JSON serializer
                def utf8_json_serializer(obj):
                    return json_lib.dumps(obj, ensure_ascii=False).encode('utf-8')

                # UTF-8을 지원하는 HTTP 클라이언트 생성
                http_client = httpx.Client(
                    timeout=httpx.Timeout(60.0, connect=10.0),
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                    headers={
                        "Content-Type": "application/json; charset=utf-8",
                        "Accept": "application/json",
                        "Accept-Charset": "utf-8"
                    }
                )

                # httpx의 기본 JSON serializer를 UTF-8을 지원하는 것으로 교체
                # 하지만 OpenAI 클라이언트는 이를 직접 사용하지 않으므로
                # 메시지 전처리를 chat_completion에서 수행

                self.client = OpenAI(
                    api_key=api_key or "EMPTY",  # vLLM은 API 키가 필요없을 수 있음
                    base_url=base_url,
                    http_client=http_client,
                    default_headers={
                        "Content-Type": "application/json; charset=utf-8"
                    },
                    max_retries=self.max_retries,
                )

                # OpenAI 클라이언트 내부의 JSON serialization을 패치
                # 이것이 가장 확실한 UTF-8 인코딩 보장 방법
                import functools
                original_dumps = json_lib.dumps

                @functools.wraps(original_dumps)
                def utf8_dumps(*args, **kwargs):
                    # ensure_ascii=False를 기본값으로 설정
                    kwargs.setdefault('ensure_ascii', False)
                    return original_dumps(*args, **kwargs)

                # json.dumps를 패치 (vLLM 사용 시에만)
                json_lib.dumps = utf8_dumps

                print(f"[LLM] vLLM initialized (base_url: {base_url}, model: {model})")
                print(f"[LLM] UTF-8 encoding explicitly configured for vLLM")
                print(f"[LLM] JSON serialization patched for UTF-8 support")

            elif self.provider == "google":
                if genai is None:
                    raise ImportError("google-generativeai 패키지가 설치되지 않았습니다. 'pip install google-generativeai'를 실행하세요.")

                # base_url이 있으면 OpenAI 호환 모드 사용
                if base_url:
                    # Gemini OpenAI compatibility mode
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url=base_url,
                    )
                    self.model = model or "gemini-2.0-flash-exp"
                    self.use_openai_compat = True
                    print(f"[LLM] Google Gemini initialized (OpenAI compatibility mode)")
                    print(f"[LLM]   Model: {self.model}")
                    print(f"[LLM]   Base URL: {base_url}")
                else:
                    # 네이티브 Gemini SDK 사용
                    genai.configure(api_key=api_key)
                    self.client = genai.GenerativeModel(model or "gemini-1.5-pro")
                    self.model = model or "gemini-1.5-pro"
                    self.use_openai_compat = False
                    print(f"[LLM] Google Gemini initialized (Native SDK mode, model: {self.model})")

            elif self.provider == "anthropic":
                if anthropic is None:
                    raise ImportError("anthropic 패키지가 설치되지 않았습니다. 'pip install anthropic'를 실행하세요.")

                # base_url이 있으면 OpenAI 호환 모드 사용 가능
                if base_url:
                    # Anthropic도 OpenAI 호환 API를 제공할 수 있음
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url=base_url,
                    )
                    self.model = model or "claude-3-5-sonnet-20241022"
                    self.use_openai_compat = True
                    print(f"[LLM] Anthropic Claude initialized (OpenAI compatibility mode)")
                    print(f"[LLM]   Model: {self.model}")
                    print(f"[LLM]   Base URL: {base_url}")
                else:
                    # 네이티브 Anthropic SDK 사용
                    self.client = anthropic.Anthropic(api_key=api_key)
                    self.model = model or "claude-3-5-sonnet-20241022"
                    self.use_openai_compat = False
                    print(f"[LLM] Anthropic Claude initialized (Native SDK mode, model: {self.model})")

            elif self.provider == "xai":
                # xAI Grok은 OpenAI 호환 API 제공
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url or "https://api.x.ai/v1",
                    max_retries=self.max_retries,
                )
                self.model = model or "grok-beta"
                print(f"[LLM] xAI Grok initialized")
                print(f"[LLM]   Model: {self.model}")
                print(f"[LLM]   Base URL: {base_url or 'https://api.x.ai/v1'}")

            else:
                raise ValueError(f"Unsupported provider: {provider}. Use 'azure', 'openai', 'vllm', 'google', 'anthropic', or 'xai'")

        except Exception as e:
            print(f"[LLM] Initialization failed -> {e}")
            raise

    def chat_completion(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, stream: bool = False):
        """LLM Chat Completion 호출"""
        try:
            # Google Gemini (Native SDK)
            if self.provider == "google" and not getattr(self, 'use_openai_compat', False):
                return self._gemini_chat_completion(messages, tools, stream)

            # Anthropic Claude (Native SDK)
            elif self.provider == "anthropic" and not getattr(self, 'use_openai_compat', False):
                return self._claude_chat_completion(messages, tools, stream)

            # vLLM의 경우 한글 등 non-ASCII 문자 처리를 위한 추가 처리
            elif self.provider == "vllm":
                # 메시지를 JSON으로 직렬화한 후 다시 파싱하여 UTF-8 인코딩 보장
                import json as json_lib
                try:
                    # ensure_ascii=False로 UTF-8 문자 보존
                    messages_json = json_lib.dumps(messages, ensure_ascii=False)
                    messages = json_lib.loads(messages_json)
                except Exception as e:
                    print(f"[LLM] Warning: Failed to process messages for UTF-8: {e}")

            # OpenAI 호환 API (Azure, OpenAI, vLLM, Google OpenAI-compat, Anthropic OpenAI-compat, xAI)
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "stream": stream,
            }

            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            return self.client.chat.completions.create(**kwargs)

        except UnicodeEncodeError as e:
            # 인코딩 에러 처리
            error_msg = f"인코딩 에러: {str(e)}\n"
            error_msg += f"Provider: {self.provider}\n"
            error_msg += f"Model: {self.model}\n"
            error_msg += "vLLM 사용 시 모델이 UTF-8을 지원하는지 확인하세요."
            raise Exception(error_msg)

        except BadRequestError as e:
            error_str = str(e)

            # OpenAI API 에러 처리
            if hasattr(e, 'status_code'):
                if e.status_code == 429:
                    raise Exception(f"OpenAI API 할당량 초과 또는 Rate Limit: {error_str}\n"
                                  f"Provider: {self.provider}\n"
                                  f"Model: {self.model}\n"
                                  f"해결 방법:\n"
                                  f"1. API 키의 크레딧을 확인하세요\n"
                                  f"2. 올바른 API 키가 설정되었는지 확인하세요\n"
                                  f"3. Rate Limit인 경우 잠시 후 다시 시도하세요")
                elif e.status_code == 401:
                    raise Exception(f"OpenAI API 인증 실패: {error_str}\n"
                                  f"API 키가 올바른지 확인하세요")
                elif e.status_code == 404:
                    raise Exception(f"모델을 찾을 수 없음: {error_str}\n"
                                  f"Model: {self.model}\n"
                                  f"올바른 모델명을 사용하고 있는지 확인하세요\n"
                                  f"사용 가능한 모델: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo")

            # Azure OpenAI 콘텐츠 필터링 처리
            if 'content_filter' in error_str or 'content management policy' in error_str.lower():
                error_body = getattr(e, 'body', None)
                filtered_categories = []

                if error_body and isinstance(error_body, dict):
                    error_info = error_body.get('error', {})
                    inner_error = error_info.get('innererror', {})
                    filter_result = inner_error.get('content_filter_result', {})

                    for category, details in filter_result.items():
                        if isinstance(details, dict) and details.get('filtered'):
                            severity = details.get('severity', 'unknown')
                            filtered_categories.append(f"{category}={severity}")

                if not filtered_categories:
                    filtered_categories = ['content_filter']

                raise ContentFilterError(
                    filtered_categories=filtered_categories,
                    original_error=e
                )

            raise

        except Exception as e:
            # 기타 에러에 대한 디버깅 정보 추가
            if self.provider == "vllm":
                error_msg = f"vLLM 호출 실패: {str(e)}\n"
                error_msg += f"Base URL: {self.client.base_url}\n"
                error_msg += f"Model: {self.model}\n"
                error_msg += "vLLM 서버가 실행 중인지, 모델이 로드되었는지 확인하세요."
                raise Exception(error_msg) from e
            raise

    def _gemini_chat_completion(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, stream: bool = False):
        """Gemini API 호출"""
        # Gemini는 OpenAI와 다른 메시지 형식 사용
        # system 메시지를 분리하고 user/assistant 메시지만 전달
        system_instruction = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                chat_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                chat_messages.append({"role": "model", "parts": [msg["content"]]})

        # 새 모델 인스턴스 생성 (system instruction 포함)
        if system_instruction:
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_instruction
            )
        else:
            model = self.client

        # Tool calling은 Gemini에서 별도 처리 필요 (여기서는 기본 구현)
        response = model.generate_content(
            chat_messages[-1]["parts"] if chat_messages else "",
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )

        # OpenAI 형식으로 변환
        class GeminiResponse:
            def __init__(self, text):
                self.choices = [type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': text,
                        'role': 'assistant'
                    })()
                })()]

        return GeminiResponse(response.text)

    def _claude_chat_completion(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, stream: bool = False):
        """Claude API 호출"""
        # Claude는 system 메시지를 별도 파라미터로 받음
        system_message = None
        claude_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        kwargs = {
            "model": self.model,
            "messages": claude_messages,
            "max_tokens": 4096,
            "temperature": 0.2,
        }

        if system_message:
            kwargs["system"] = system_message

        # Tool calling은 Claude에서 별도 처리 필요 (여기서는 기본 구현)
        if tools:
            # Claude의 tool 형식으로 변환 필요
            pass

        response = self.client.messages.create(**kwargs)

        # OpenAI 형식으로 변환
        class ClaudeResponse:
            def __init__(self, content):
                self.choices = [type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': content,
                        'role': 'assistant'
                    })()
                })()]

        return ClaudeResponse(response.content[0].text)

