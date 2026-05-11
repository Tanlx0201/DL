"""RAG answer generation."""

from __future__ import annotations

import time

from app.config import AppConfig
from app.generation.llm_providers import LLMResponse, PROVIDERS, LLMProvider, OllamaProvider
from app.generation.prompt_templates import CONTEXT_TEMPLATE, SYSTEM_PROMPT, USER_TEMPLATE


class Generator:
    """Tạo prompt và gọi provider tương ứng, retry 1 lần khi timeout."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _build_context(self, search_results: list[dict]) -> str:
        lines = []
        for i, item in enumerate(search_results, start=1):
            score_pct = max(0.0, min(100.0, item["score"] * 100))
            lines.append(
                CONTEXT_TEMPLATE.format(
                    i=i,
                    chunk_text=item["text"],
                    source=item["source"],
                    page=item["page"],
                    chunk_index=item["chunk_index"],
                    score=f"{score_pct:.2f}",
                ).strip()
            )
        return "\n\n".join(lines)

    def _create_provider(self) -> tuple[str, str, LLMProvider]:
        provider_name = self.config.get("default_provider")
        model_name = self.config.get("default_model")
        cls = PROVIDERS.get(provider_name)
        if not cls:
            raise ValueError("Provider không được hỗ trợ.")

        if provider_name == "ollama":
            provider = OllamaProvider(
                model=model_name,
                base_url=self.config.get("ollama_base_url"),
            )
        else:
            provider = cls(
                api_key=self.config.get(f"{provider_name}_api_key"),
                model=model_name,
            )

        if not provider.is_available():
            raise ValueError("Provider chưa sẵn sàng. Vui lòng kiểm tra API key hoặc URL.")
        return provider_name, model_name, provider

    def answer(self, query: str, search_results: list[dict]) -> tuple[LLMResponse, int, str, str]:
        provider_name, model_name, provider = self._create_provider()

        context = self._build_context(search_results)
        user_content = USER_TEMPLATE.format(context=context, query=query)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        max_tokens = int(self.config.get("max_tokens"))
        temperature = float(self.config.get("temperature"))

        start = time.perf_counter()
        try:
            response = provider.generate(messages, max_tokens=max_tokens, temperature=temperature)
        except TimeoutError:
            response = provider.generate(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:
            if "timeout" in str(exc).lower():
                response = provider.generate(messages, max_tokens=max_tokens, temperature=temperature)
            else:
                raise
        latency_ms = int((time.perf_counter() - start) * 1000)
        return response, latency_ms, provider_name, model_name
