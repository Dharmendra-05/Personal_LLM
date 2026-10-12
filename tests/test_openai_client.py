"""Unit tests for OpenAICompatibleClient attribute assignment and payload construction."""

import unittest
from models.openai_compatible_client import OpenAICompatibleClient
from models.base import GenerationRequest


class TestOpenAICompatibleClient(unittest.TestCase):
    def test_client_attributes_stored_from_kwargs(self):
        client = OpenAICompatibleClient(
            model_name="test-model",
            base_url="https://api.example.com/v1",
            api_key="sk-test-secret",
            system_prompt="Custom system prompt",
            default_temperature=0.42,
        )
        self.assertEqual(client.system_prompt, "Custom system prompt")
        self.assertEqual(client.default_temperature, 0.42)
        self.assertEqual(client.api_key, "sk-test-secret")

    def test_build_payload_uses_default_temperature_and_system_prompt(self):
        client = OpenAICompatibleClient(
            model_name="test-model",
            base_url="https://api.example.com/v1",
            api_key="sk-test-secret",
            system_prompt="Default system prompt",
            default_temperature=0.55,
        )
        req = GenerationRequest(prompt="Hello world", temperature=0.7)
        payload = client._build_payload(req)

        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["temperature"], 0.55)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "Default system prompt")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"], "Hello world")


if __name__ == "__main__":
    unittest.main()
