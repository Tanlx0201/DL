"""LLM provider abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMProvider(ABC):
    """Base class cho provider."""

    def __init__(self, api_key: str = "", model: str = "") -> None:
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate(self, messages: list[dict], max_tokens: int, temperature: float) -> LLMResponse:
        """Sinh phản hồi cho chuỗi hội thoại."""

    def is_available(self) -> bool:
        return bool(self.api_key)


class OpenAIProvider(LLMProvider):
    def generate(self, messages: list[dict], max_tokens: int, temperature: float) -> LLMResponse:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model or "gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=45,
        )
        choice = resp.choices[0].message.content or ""
        usage = resp.usage
        return LLMResponse(
            text=choice,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )


class GeminiProvider(LLMProvider):
    def generate(self, messages: list[dict], max_tokens: int, temperature: float) -> LLMResponse:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model or "gemini-2.0-flash")
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        resp = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
        )
        text = getattr(resp, "text", "") or ""
        return LLMResponse(text=text)


class AnthropicProvider(LLMProvider):
    def generate(self, messages: list[dict], max_tokens: int, temperature: float) -> LLMResponse:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_messages = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        resp = client.messages.create(
            model=self.model or "claude-3-5-sonnet-20241022",
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=user_messages,
            timeout=45,
        )
        text = "".join([b.text for b in resp.content if hasattr(b, "text")])
        usage = resp.usage
        return LLMResponse(
            text=text,
            prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )


class OllamaProvider(LLMProvider):
    def __init__(self, api_key: str = "", model: str = "", base_url: str = "http://host.docker.internal:11434") -> None:
        super().__init__(api_key=api_key, model=model)
        self.base_url = base_url

    def is_available(self) -> bool:
        return bool(self.base_url)

    def generate(self, messages: list[dict], max_tokens: int, temperature: float) -> LLMResponse:
        payload = {
            "model": self.model or "llama3",
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        with httpx.Client(timeout=45) as client:
            resp = client.post(f"{self.base_url.rstrip('/')}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = data.get("message", {}).get("content", "")
        return LLMResponse(text=text)


PROVIDERS = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}

PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-pro"],
    "anthropic": ["claude-3-5-sonnet-20241022"],
    "ollama": ["llama3", "qwen2.5"],
}
