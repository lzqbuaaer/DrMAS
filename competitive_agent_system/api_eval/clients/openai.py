from __future__ import annotations

from competitive_agent_system.api_eval.clients.openai_compatible import OpenAICompatibleChatClient


def build_openai_client(*, model: str, api_key_env: str = "OPENAI_API_KEY", base_url: str = "https://api.openai.com/v1", timeout: float = 120.0, max_http_retries: int = 3):
    return OpenAICompatibleChatClient(
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        timeout=timeout,
        max_http_retries=max_http_retries,
    )
