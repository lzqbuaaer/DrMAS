from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from competitive_agent_system.api_eval.clients.base import ChatModelClient


class OpenAIResponsesClient(ChatModelClient):
    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        base_url: str,
        reasoning_effort: str | None = None,
        timeout: float = 120.0,
        max_http_retries: int = 3,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort or None
        self.timeout = float(timeout)
        self.max_http_retries = int(max_http_retries)

    def _get_api_key(self) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable {self.api_key_env}")
        return api_key

    def _extract_output_text(self, parsed: dict) -> str:
        texts: list[str] = []
        for item in parsed.get("output", []) or []:
            if item.get("type") != "message" or item.get("role") != "assistant":
                continue
            for content in item.get("content", []) or []:
                if content.get("type") == "output_text" and content.get("text"):
                    texts.append(str(content["text"]))
                elif content.get("type") == "refusal" and content.get("refusal"):
                    texts.append(str(content["refusal"]))
        text = "\n".join(texts).strip()
        if text:
            return text
        raise KeyError("No assistant output_text found in Responses API payload")

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        url = f"{self.base_url}/responses"
        payload = {
            "model": self.model,
            "input": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_output_tokens": int(max_tokens),
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
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
                return self._extract_output_text(parsed)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 >= self.max_http_retries:
                    break
                time.sleep(min(2**attempt, 8))

        raise RuntimeError(f"OpenAI Responses API generation failed after {self.max_http_retries} attempts: {last_error}") from last_error
