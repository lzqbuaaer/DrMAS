from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentPrivateState:
    plans_text: str = ""
    insights_text: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    failed: bool = False


@dataclass
class CompetitiveAction:
    raw_text: str
    price: float | None
    plans_text: str
    insights_text: str
    valid: bool
    error: str | None = None
    retry_count: int = 0


@dataclass
class CompetitiveStepResult:
    reward: float
    rewards_by_agent: dict[str, float]
    done: bool
    won: bool
    info: dict[str, Any]


@dataclass
class CompetitiveEpisodeSummary:
    metrics: dict[str, float]
    won: bool
    reason: str | None = None
