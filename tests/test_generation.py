import pytest

from app.generation.generator import Generator
from app.generation.llm_providers import LLMResponse


class DummyConfig:
    def __init__(self) -> None:
        self.values = {
            "default_provider": "openai",
            "default_model": "gpt-4o-mini",
            "openai_api_key": "x",
            "max_tokens": "64",
            "temperature": "0.1",
        }

    def get(self, key: str) -> str:
        return self.values.get(key, "")


class DummyProvider:
    def __init__(self) -> None:
        self.called = 0

    def is_available(self) -> bool:
        return True

    def generate(self, messages, max_tokens, temperature):
        self.called += 1
        if self.called == 1:
            raise TimeoutError("timeout")
        return LLMResponse(text="ok", prompt_tokens=1, completion_tokens=2)


def test_generator_retry_on_timeout(monkeypatch) -> None:
    cfg = DummyConfig()
    gen = Generator(cfg)
    provider = DummyProvider()

    def fake_create_provider():
        return "openai", "gpt-4o-mini", provider

    monkeypatch.setattr(gen, "_create_provider", fake_create_provider)
    res, latency, provider_name, model_name = gen.answer(
        "cau hoi",
        [
            {
                "score": 0.9,
                "text": "chunk",
                "source": "a.pdf",
                "page": 1,
                "chunk_index": 0,
            }
        ],
    )
    assert res.text == "ok"
    assert latency >= 0
    assert provider_name == "openai"
    assert model_name == "gpt-4o-mini"
    assert provider.called == 2


def test_generator_fails_when_provider_missing() -> None:
    cfg = DummyConfig()
    cfg.values["default_provider"] = "unknown"
    gen = Generator(cfg)
    with pytest.raises(ValueError):
        gen._create_provider()
