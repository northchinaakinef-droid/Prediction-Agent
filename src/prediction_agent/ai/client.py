from __future__ import annotations

import json
import os
from typing import Protocol, TypeVar
from urllib.request import Request, urlopen

from .schemas import AIAnalysis

T = TypeVar("T")


class LLMClient(Protocol):
    provider: str
    model: str
    def analyze(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T: ...


class OpenAICompatibleClient:
    def __init__(self, *, api_key: str, model: str, base_url: str, provider: str = "openai"):
        self.api_key, self.model = api_key, model
        self.provider = provider
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient | None":
        key = os.getenv("LLM_API_KEY")
        if not key:
            return None
        provider = os.getenv("LLM_PROVIDER", "openai").casefold()
        default_base = "https://api.deepseek.com" if provider == "deepseek" else "https://api.openai.com/v1"
        return cls(api_key=key, model=os.getenv("LLM_MODEL", "gpt-5-mini"),
                   base_url=os.getenv("LLM_BASE_URL", default_base), provider=provider)

    def analyze(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
        }
        request = Request(endpoint, data=json.dumps(payload).encode(), headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
        })
        with urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode())
        content = result["choices"][0]["message"]["content"]
        return schema.validate_payload(json.loads(content))  # type: ignore[attr-defined,no-any-return]


class AnthropicClient:
    def __init__(self, *, api_key: str, model: str, base_url: str = "https://api.anthropic.com/v1"):
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        self.provider = "anthropic"

    def analyze(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        payload = {
            "model": self.model, "max_tokens": 2000, "temperature": 0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        request = Request(f"{self.base_url}/messages", data=json.dumps(payload).encode(), headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json",
        })
        with urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode())
        content = result["content"][0]["text"]
        return schema.validate_payload(json.loads(content))  # type: ignore[attr-defined,no-any-return]


def client_from_env() -> LLMClient | None:
    key = os.getenv("LLM_API_KEY")
    if not key:
        return None
    provider = os.getenv("LLM_PROVIDER", "openai").casefold()
    if provider == "anthropic":
        return AnthropicClient(
            api_key=key, model=os.getenv("LLM_MODEL", "claude-sonnet-4-5"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.anthropic.com/v1"),
        )
    return OpenAICompatibleClient.from_env()
