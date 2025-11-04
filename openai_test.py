import os, re, sys
from openai import OpenAI
from openai import APIError, RateLimitError, BadRequestError

from dotenv import load_dotenv
load_dotenv()

def mask(v):
    if not v: return None
    return v[:4] + "..." + v[-4:]

print("== ENV CHECK ==")
print("OPENAI_API_KEY:", mask(os.getenv("OPENAI_API_KEY")))
print("OPENAI_BASE_URL:", os.getenv("OPENAI_BASE_URL"))   # 있어선 안 됨(공식일 때)
print("OPENAI_PROJECT :", os.getenv("OPENAI_PROJECT"))    # n8n과 맞추면 Good
print("OPENAI_ORG     :", os.getenv("OPENAI_ORG"))        # 되도록 비우기 권장
print("OPENAI_MODEL   :", os.getenv("OPENAI_MODEL"))

# 공식 OpenAI: base_url 주지 않음
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    project=os.getenv("OPENAI_PROJECT") or None,
    # organization=os.getenv("OPENAI_ORG") or None,
)

def try_model(m):
    try:
        r = client.responses.create(
            model=m,
            input="Say hello in one word."
        )
        print(f"[OK] {m}: {r.output_text!r}")
    except RateLimitError as e:
        print(f"[429 RateLimit] {m} ->", e)
    except BadRequestError as e:
        print(f"[400 BadRequest] {m} ->", e)
    except APIError as e:
        print(f"[APIError {e.status_code}] {m} ->", getattr(e, "response", None))
    except Exception as e:
        print(f"[Other] {m} ->", repr(e))

print("\n== TEST CALLS ==")
try_model(os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
try_model("gpt-4o-mini")   # 대조용
