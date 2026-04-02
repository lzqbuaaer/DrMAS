from __future__ import annotations

import math
import re

from competitive_agent_system.games.base import CompetitiveAction


class DuopolyActionParser:
    def __init__(self, max_retries: int = 10):
        self.max_retries = max_retries

    def extract_ceiling_from_observation(self, text: str) -> float | None:
        match = re.search(r"No customer would pay more than\s*([-+]?\d+(?:\.\d+)?)\.", text)
        if match is None:
            return None
        value = float(match.group(1))
        if not math.isfinite(value) or value < 0:
            return None
        return value

    def extract_block(self, text: str, tag: str) -> str | None:
        matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
        if not matches:
            return None
        return matches[-1].strip()

    def extract_price(self, text: str) -> float | None:
        matches = re.findall(r"<PRICE>\s*([-+]?\d+(?:\.\d+)?)\s*</PRICE>", text, flags=re.DOTALL)
        if not matches:
            return None
        value = float(matches[-1])
        if not math.isfinite(value) or value < 0:
            return None
        return value

    def parse(self, text: str, max_price: float | None = None) -> CompetitiveAction:
        observations = self.extract_block(text, "OBSERVATIONS")
        plans_text = self.extract_block(text, "PLANS")
        insights_text = self.extract_block(text, "INSIGHTS")
        price = self.extract_price(text)
        price_exceeds_ceiling = price is not None and max_price is not None and price > max_price

        valid = observations is not None and plans_text is not None and insights_text is not None and price is not None and not price_exceeds_ceiling
        if valid:
            error = None
        elif price_exceeds_ceiling:
            error = f"price exceeds ceiling {max_price}"
        else:
            error = "missing required tags or invalid price"
        return CompetitiveAction(
            raw_text=text,
            action_type="price",
            payload={} if price is None else {"price": price},
            plans_text=plans_text or "",
            insights_text=insights_text or "",
            valid=valid,
            error=error,
        )
