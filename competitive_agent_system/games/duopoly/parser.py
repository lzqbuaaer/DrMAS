from __future__ import annotations

import math
import re

from competitive_agent_system.games.base import CompetitiveAction


class DuopolyActionParser:
    def __init__(self, max_retries: int = 10):
        self.max_retries = max_retries

    def extract_block(self, text: str, tag: str) -> str | None:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
        if match is None:
            return None
        return match.group(1).strip()

    def extract_price(self, text: str) -> float | None:
        match = re.search(r"<PRICE>\s*([-+]?\d+(?:\.\d+)?)\s*</PRICE>", text, flags=re.DOTALL)
        if match is None:
            return None
        value = float(match.group(1))
        if not math.isfinite(value) or value < 0:
            return None
        return value

    def parse(self, text: str) -> CompetitiveAction:
        observations = self.extract_block(text, "OBSERVATIONS")
        plans_text = self.extract_block(text, "PLANS")
        insights_text = self.extract_block(text, "INSIGHTS")
        price = self.extract_price(text)

        valid = observations is not None and plans_text is not None and insights_text is not None and price is not None
        error = None if valid else "missing required tags or invalid price"
        return CompetitiveAction(
            raw_text=text,
            price=price,
            plans_text=plans_text or "",
            insights_text=insights_text or "",
            valid=valid,
            error=error,
        )
