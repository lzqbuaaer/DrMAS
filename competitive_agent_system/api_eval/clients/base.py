from __future__ import annotations

from abc import ABC, abstractmethod


class ChatModelClient(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> str:
        raise NotImplementedError
