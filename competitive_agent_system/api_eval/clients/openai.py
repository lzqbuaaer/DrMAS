from __future__ import annotations

from urllib.parse import urlparse

from competitive_agent_system.api_eval.clients.openai_compatible import OpenAICompatibleChatClient
from competitive_agent_system.api_eval.clients.openai_responses import OpenAIResponsesClient


def build_openai_client(
    *,
    model: str,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str = "https://api.openai.com/v1",
    reasoning_effort: str | None = None,
    thinking_enabled: bool = False,
    timeout: float = 120.0,
    max_http_retries: int = 3,
):
    hostname = (urlparse(base_url).hostname or "").lower()
    if "apimart.ai" in hostname:
        return OpenAICompatibleChatClient(
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            timeout=timeout,
            max_http_retries=max_http_retries,
        )
    return OpenAIResponsesClient(
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        max_http_retries=max_http_retries,
    )
