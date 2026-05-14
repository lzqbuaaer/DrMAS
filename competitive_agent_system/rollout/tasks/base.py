from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class BaseCompetitiveRolloutTaskHandler:
    def __init__(self, config, agent_ids: list[str], sanitize_path_component):
        self.config = config
        self.agent_ids = list(agent_ids)
        self.sanitize_path_component = sanitize_path_component

    @property
    def env_name(self) -> str:
        return str(self.config.env.env_name).lower()

    def get_eval_trace_extra_keys(self) -> tuple[str, ...]:
        return ()

    def build_common_step_trace(self, step_idx: int, info: dict, raw_text_by_agent: dict[str, str]) -> dict:
        return {
            "step": step_idx + 1,
            "data_source": info.get("data_source"),
            "profits_by_agent": info.get("profits_by_agent", {}),
            "failure_reason": info.get("failure_reason"),
            "invalid_by_agent": info.get("invalid_by_agent", {}),
            "retry_count_by_agent": info.get("retry_count_by_agent", {}),
            "raw_text_by_agent": raw_text_by_agent,
        }

    def build_step_trace(self, step_idx: int, info: dict, raw_text_by_agent: dict[str, str]) -> dict:
        return self.build_common_step_trace(step_idx=step_idx, info=info, raw_text_by_agent=raw_text_by_agent)

    def get_eval_summary_filename(self) -> str:
        return f"{self.sanitize_path_component(self.env_name)}_eval_summary.json"

    def iter_eval_payloads(self, dump_dir: str):
        summary_filename = self.get_eval_summary_filename()
        for json_path in sorted(Path(dump_dir).glob("*.json")):
            if json_path.name in {summary_filename, "run_config.json"}:
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                yield json.load(f)

    def build_summary_records(self, dump_dir: str) -> dict[str, list[dict]]:
        return {}

    def build_group_summary(self, data_source: str, records: list[dict], created_at: str) -> dict | None:
        return None

    def render_summary_artifacts(self, group_summary: dict, dump_dir: str, multiple_groups: bool) -> None:
        return None

    def log_eval_step(self, step_idx: int, infos: list[dict], active_masks) -> None:
        return None

    def mean_or_none(self, values: list[float]) -> float | None:
        if not values:
            return None
        return float(np.mean(values))

    def collect_agent_metric(self, records: list[dict], field_name: str, agent_id: str) -> list[float]:
        values = []
        for record in records:
            value = record.get(field_name, {}).get(agent_id)
            if value is not None:
                values.append(float(value))
        return values

    def collect_scalar_metric(self, records: list[dict], field_name: str) -> list[float]:
        values = []
        for record in records:
            value = record.get(field_name)
            if value is not None:
                values.append(float(value))
        return values

    def collect_product_metric(self, records: list[dict], field_name: str, product_key: str) -> list[float]:
        values = []
        for record in records:
            value = record.get(field_name, {}).get(product_key)
            if value is not None:
                values.append(float(value))
        return values

    def collect_agent_product_metric(self, records: list[dict], field_name: str, agent_id: str, product_key: str) -> list[float]:
        values = []
        for record in records:
            value = record.get(field_name, {}).get(agent_id, {}).get(product_key)
            if value is not None:
                values.append(float(value))
        return values
