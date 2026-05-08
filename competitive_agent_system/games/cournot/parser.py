from __future__ import annotations

import math
import re

from competitive_agent_system.games.base import CompetitiveAction


def canonicalize_cournot_invalid_reason(reason: object) -> str | None:
    if reason is None:
        return None

    text = str(reason).strip()
    if not text:
        return None

    lowered = text.lower()
    canonical_map = [
        ("total quantity exceeds capacity", "capacity_exceeded"),
        ("missing required tags or invalid quantities", "missing_tags_or_invalid_quantities"),
        ("capacity exceeded by firm 1 agent", "capacity_exceeded_firm1"),
        ("capacity exceeded by firm 2 agent", "capacity_exceeded_firm2"),
        ("invalid output", "invalid_output"),
    ]
    for pattern, label in canonical_map:
        if pattern in lowered:
            return label

    sanitized = "".join(char if char.isalnum() else "_" for char in lowered)
    sanitized = "_".join(part for part in sanitized.split("_") if part)
    return sanitized[:80] or "unknown"


class CournotActionParser:
    def __init__(self, max_retries: int = 10):
        self.max_retries = max_retries

    def extract_total_units_from_observation(self, text: str) -> float | None:
        patterns = [
            r"Your total output across Product A and Product B must be at most\s*([-+]?\d+(?:\.\d+)?)\s*units\.",
            r"Producing more than\s*([-+]?\d+(?:\.\d+)?)\s*units of aggregate product will result in negative profits",
            r"Product_A\s*\+\s*Product_B\s*<=\s*([-+]?\d+(?:\.\d+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match is None:
                continue
            value = float(match.group(1))
            if not math.isfinite(value) or value < 0:
                return None
            return value
        return None

    def extract_block(self, text: str, tag: str) -> str | None:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
        if match is None:
            return None
        return match.group(1).strip()

    def extract_scalar(self, text: str, tag: str) -> float | None:
        match = re.search(rf"<{tag}>\s*([-+]?\d+(?:\.\d+)?)\s*</{tag}>", text, flags=re.DOTALL)
        if match is None:
            return None
        value = float(match.group(1))
        if not math.isfinite(value) or value < 0:
            return None
        return value

    def parse(self, text: str, total_units: float | None = None) -> CompetitiveAction:
        observations = self.extract_block(text, "OBSERVATIONS")
        plans_text = self.extract_block(text, "PLANS")
        insights_text = self.extract_block(text, "INSIGHTS")
        quantity_a = self.extract_scalar(text, "QUANTITY_A")
        quantity_b = self.extract_scalar(text, "QUANTITY_B")
        total_quantity = quantity_a + quantity_b if quantity_a is not None and quantity_b is not None else None

        total_exceeds_capacity = (
            quantity_a is not None
            and quantity_b is not None
            and total_units is not None
            and total_quantity is not None
            and total_quantity > total_units
        )

        valid = (
            observations is not None
            and plans_text is not None
            and insights_text is not None
            and quantity_a is not None
            and quantity_b is not None
            and not total_exceeds_capacity
        )
        if valid:
            error = None
        elif total_exceeds_capacity:
            error = (
                "total quantity exceeds capacity: "
                f"QUANTITY_A + QUANTITY_B = {total_quantity:.2f} > {total_units:.2f} "
            )
        else:
            error = "missing required tags or invalid quantities"

        payload = {}
        if quantity_a is not None:
            payload["quantity_a"] = quantity_a
        if quantity_b is not None:
            payload["quantity_b"] = quantity_b

        return CompetitiveAction(
            raw_text=text,
            action_type="quantities",
            payload=payload,
            plans_text=plans_text or "",
            insights_text=insights_text or "",
            valid=valid,
            error=error,
        )
