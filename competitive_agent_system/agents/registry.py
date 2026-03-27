from __future__ import annotations

from typing import Callable


class CompetitiveAgentRegistry:
    _REGISTRY: dict[str, Callable[..., "CompetitiveBaseAgent"]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(agent_cls: Callable[..., "CompetitiveBaseAgent"]):
            if name in cls._REGISTRY:
                raise ValueError(f"Competitive agent '{name}' already registered.")
            cls._REGISTRY[name] = agent_cls
            return agent_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls._REGISTRY:
            raise KeyError(f"Unknown competitive agent '{name}'. Registered: {list(cls._REGISTRY)}")
        return cls._REGISTRY[name](**kwargs)
