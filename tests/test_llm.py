import unittest
from types import SimpleNamespace
from unittest.mock import patch

import amr.llm as llm_module


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **payload: object) -> object:
        self.payloads.append(payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
            usage=SimpleNamespace(
                prompt_tokens=1200,
                prompt_tokens_details=SimpleNamespace(cached_tokens=1024, cache_write_tokens=0),
            ),
        )


class _FakeOpenAI:
    client: _FakeOpenAIClient

    def __new__(cls, **_: object) -> _FakeOpenAIClient:
        cls.client = _FakeOpenAIClient()
        return cls.client


class LanguageModelTests(unittest.TestCase):
    def _model(self, model_name: str) -> tuple[object, _FakeOpenAIClient]:
        openai_patch = patch.object(llm_module, "OpenAI", _FakeOpenAI)
        module_patch = patch.object(llm_module, "_openai_module", SimpleNamespace())
        openai_patch.start()
        module_patch.start()
        self.addCleanup(openai_patch.stop)
        self.addCleanup(module_patch.stop)
        model = llm_module.OpenAILanguageModel(
            llm_module.OpenAIConfig(model=model_name, api_key="test")
        )
        return model, _FakeOpenAI.client

    def test_legacy_model_uses_stable_cache_key_without_explicit_breakpoint(self) -> None:
        model, client = self._model("gpt-5-nano")

        model.complete(
            '{"title":"different"}',
            system_prompt="stable system prompt",
            prompt_cache_key="amr-unified-classifier-v2",
        )

        payload = client.payloads[0]
        self.assertEqual(payload["prompt_cache_key"], "amr-unified-classifier-v2")
        self.assertNotIn("prompt_cache_options", payload)
        self.assertEqual(payload["messages"][0]["content"], "stable system prompt")
        self.assertEqual(model.last_usage["cached_tokens"], 1024)
        self.assertEqual(model.prompt_cache_usage["cached_tokens"], 1024)

    def test_gpt_56_uses_explicit_breakpoint_after_system_prompt(self) -> None:
        model, client = self._model("gpt-5.6")

        model.complete(
            '{"title":"different"}',
            system_prompt="stable system prompt",
            prompt_cache_key="amr-unified-classifier-v2",
        )

        payload = client.payloads[0]
        self.assertEqual(payload["prompt_cache_options"], {"mode": "explicit"})
        system_content = payload["messages"][0]["content"]
        self.assertEqual(system_content[0]["text"], "stable system prompt")
        self.assertEqual(system_content[0]["prompt_cache_breakpoint"], {"mode": "explicit"})


if __name__ == "__main__":
    unittest.main()
