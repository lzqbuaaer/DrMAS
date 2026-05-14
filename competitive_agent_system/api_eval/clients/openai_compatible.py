from __future__ import annotations

import json
import os
import time

import requests

from competitive_agent_system.api_eval.clients.base import ChatModelClient


class OpenAICompatibleChatClient(ChatModelClient):
    _has_printed_debug_payload = False

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

    def _format_http_error(self, *, status_code: int, url: str, error_body: str) -> str:
        try:
            parsed = json.loads(error_body)
        except json.JSONDecodeError:
            return f"HTTP {status_code} for {url} model={self.model}: {error_body}"

        error_payload = parsed.get("error", parsed)
        if not isinstance(error_payload, dict):
            return f"HTTP {status_code} for {url} model={self.model}: {error_body}"

        message = error_payload.get("message")
        code = error_payload.get("code")
        param = error_payload.get("param")
        error_type = error_payload.get("type")

        details = [f"HTTP {status_code} for {url} model={self.model}"]
        if message:
            details.append(f"message={message}")
        if param:
            details.append(f"param={param}")
        if code:
            details.append(f"code={code}")
        if error_type:
            details.append(f"type={error_type}")
        details.append(f"raw={error_body}")
        return " | ".join(details)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.thinking_enabled:
            payload["extra_body"] = {"thinking": {"type": "enabled"}}
        last_error = None

        for attempt in range(self.max_http_retries):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._get_api_key()}",
            }
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                parsed = response.json()
                return parsed["choices"][0]["message"]["content"]
            except requests.exceptions.HTTPError as exc:
                response = exc.response
                status_code = response.status_code if response is not None else -1
                error_body = response.text if response is not None else str(exc)
                last_error = RuntimeError(
                    self._format_http_error(
                        status_code=status_code,
                        url=url,
                        error_body=error_body,
                    )
                )
                if attempt + 1 >= self.max_http_retries:
                    break
                time.sleep(min(2 ** attempt, 8))
            except (requests.exceptions.RequestException, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_http_retries:
                    break
                time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(f"API generation failed after {self.max_http_retries} attempts: {last_error}") from last_error
