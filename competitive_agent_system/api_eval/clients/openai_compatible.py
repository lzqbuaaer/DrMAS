from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from competitive_agent_system.api_eval.clients.base import ChatModelClient


class OpenAICompatibleChatClient(ChatModelClient):
    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        base_url: str,
        reasoning_effort: str | None = None,
        thinking_enabled: bool = False,
        timeout: float = 120.0,
        max_http_retries: int = 3,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.thinking_enabled = bool(thinking_enabled)
        self.timeout = float(timeout)
        self.max_http_retries = int(max_http_retries)

    def _get_api_key(self) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable {self.api_key_env}")
        return api_key

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        top_p: float | None,
        max_tokens: int,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.thinking_enabled:
            payload["extra_body"] = {"thinking": {"type": "enabled"}}
        data = json.dumps(payload).encode("utf-8")
        last_error = None

        for attempt in range(self.max_http_retries):
            request = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._get_api_key()}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                return parsed["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"HTTP {exc.code} for {url} model={self.model}: {error_body}"
                )
                if attempt + 1 >= self.max_http_retries:
                    break
                time.sleep(min(2 ** attempt, 8))
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_http_retries:
                    break
                time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(f"API generation failed after {self.max_http_retries} attempts: {last_error}") from last_error
