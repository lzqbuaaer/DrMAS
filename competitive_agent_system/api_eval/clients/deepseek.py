from __future__ import annotations

from competitive_agent_system.api_eval.clients.openai_compatible import OpenAICompatibleChatClient


def build_deepseek_client(
    *,
    model: str,
    api_key_env: str = "DEEPSEEK_API_KEY",
    base_url: str = "https://api.deepseek.com",
    reasoning_effort: str | None = None,
    thinking_enabled: bool = False,
    timeout: float = 120.0,
    max_http_retries: int = 3,
):
    return OpenAICompatibleChatClient(
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        thinking_enabled=thinking_enabled,
        timeout=timeout,
        max_http_retries=max_http_retries,
    )
