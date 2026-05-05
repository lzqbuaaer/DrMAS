from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy
from competitive_agent_system.orchestras import CompetitiveTurnOrchestra
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn


class CompetitiveTrajectoryCollector:
    def __init__(self, config, wg_to_agents_mapping, tokenizers, processors=None):
        self.config = config
        self.tokenizers = tokenizers
        self.processors = processors
        self.eval_dump_dir = None

        agents_to_wg_mapping = {}
        for wg_id, agents in wg_to_agents_mapping.items():
            for agent in agents:
                agents_to_wg_mapping[agent["agent_id"]] = wg_id

        self.orchestra = CompetitiveTurnOrchestra(
            config=config,
            tokenizers=tokenizers,
            processors=processors,
            agents_to_wg_mapping=agents_to_wg_mapping,
        )

    def _sanitize_path_component(self, value: str) -> str:
        sanitized = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)
        return sanitized or "unknown"

    def _get_eval_dump_dir(self) -> str:
        if self.eval_dump_dir is None:
            task_name = self._sanitize_path_component(str(self.config.trainer.experiment_name))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.eval_dump_dir = os.path.join("eval_data", task_name, timestamp)
            os.makedirs(self.eval_dump_dir, exist_ok=True)
        return self.eval_dump_dir

    def _is_eval_only(self) -> bool:
        return bool(getattr(self.config.trainer, "val_only", False))

    def _get_env_name(self) -> str:
        return str(self.config.env.env_name).lower()

    def _get_agent_ids(self) -> list[str]:
        return list(self.config.agent.agent_ids)

    def _extract_terminal_infos(self, total_batch_list, total_infos) -> list[dict]:
        terminal_infos = []
        for batch_idx in range(len(total_batch_list)):
            terminal_info = {}
            for item_idx in reversed(range(len(total_batch_list[batch_idx]))):
                batch_item = total_batch_list[batch_idx][item_idx]
                if batch_item["active_masks"]:
                    terminal_info = total_infos[batch_idx][item_idx]
                    break
            terminal_infos.append(terminal_info)
        return terminal_infos

    def _get_task_env_cfg(self):
        return self.config.env.get(self._get_env_name(), None)

    def _get_task_rollout_metric_fields(self) -> list[str]:
        env_cfg = self._get_task_env_cfg()
        if env_cfg is None:
            return []
        rollout_metric_fields = env_cfg.get("rollout_metric_fields", [])
        return list(rollout_metric_fields) if rollout_metric_fields is not None else []

    def _get_task_eval_dump_fields(self):
        env_cfg = self._get_task_env_cfg()
        if env_cfg is None:
            return {}
        eval_dump_fields = env_cfg.get("eval_dump_fields", {})
        return eval_dump_fields if eval_dump_fields is not None else {}

    def _resolve_eval_dump_payload(self, terminal_info: dict) -> dict:
        resolved_payload = {}
        if not terminal_info:
            return resolved_payload

        agent_ids = self._get_agent_ids()
        eval_dump_fields = self._get_task_eval_dump_fields()

        for payload_key, spec in eval_dump_fields.items():
            if isinstance(spec, str):
                resolved_payload[payload_key] = terminal_info.get(spec)
                continue

            if not hasattr(spec, "get"):
                continue

            kind = spec.get("kind", "scalar")
            if kind == "agent_pair":
                resolved_payload[payload_key] = {
                    agent_ids[0]: terminal_info.get(spec.get("firm1")),
                    agent_ids[1]: terminal_info.get(spec.get("firm2")),
                }
            elif kind == "dict":
                fields = spec.get("fields", {})
                resolved_payload[payload_key] = {sub_key: terminal_info.get(source_key) for sub_key, source_key in fields.items()}
            elif kind == "scalar":
                resolved_payload[payload_key] = terminal_info.get(spec.get("field"))

        return resolved_payload

    def _get_task_eval_trace_extra_keys(self) -> tuple[str, ...]:
        env_name = self._get_env_name()
        if env_name == "duopoly":
            return ("p_monopoly", "p_nash", "ceiling")
        if env_name == "cournot":
            return ("monopoly_quantities", "nash_quantities", "total_units")
        return ()

    def _build_eval_trace_payload(
        self,
        idx: int,
        trace: list[dict],
        uid_batch,
        traj_uid,
        reset_infos,
        terminal_infos,
    ) -> dict:
        reset_info = reset_infos[idx] if idx < len(reset_infos) else {}
        terminal_info = terminal_infos[idx] if idx < len(terminal_infos) else {}
        first_step = trace[0]
        payload = {
            "uid": str(uid_batch[idx]),
            "traj_uid": str(traj_uid[idx]),
            "data_source": first_step["data_source"],
            "steps": trace,
            "reset_info": reset_info,
        }
        for key in self._get_task_eval_trace_extra_keys():
            if key in reset_info:
                payload[key] = reset_info.get(key)
            elif key in first_step:
                payload[key] = first_step.get(key)

        if terminal_info:
            payload.update(self._resolve_eval_dump_payload(terminal_info))
        return payload

    def _dump_eval_step_traces(self, step_traces, uid_batch, traj_uid, reset_infos, terminal_infos) -> None:
        dump_dir = self._get_eval_dump_dir()
        for idx, trace in enumerate(step_traces):
            if not trace:
                continue

            payload = self._build_eval_trace_payload(
                idx=idx,
                trace=trace,
                uid_batch=uid_batch,
                traj_uid=traj_uid,
                reset_infos=reset_infos,
                terminal_infos=terminal_infos,
            )

            filename = os.path.join(dump_dir, f"{traj_uid[idx]}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    def _build_common_step_trace(self, step_idx: int, info: dict, raw_text_by_agent: dict[str, str]) -> dict:
        return {
            "step": step_idx + 1,
            "data_source": info.get("data_source"),
            "profits_by_agent": info.get("profits_by_agent", {}),
            "failure_reason": info.get("failure_reason"),
            "invalid_by_agent": info.get("invalid_by_agent", {}),
            "retry_count_by_agent": info.get("retry_count_by_agent", {}),
            "raw_text_by_agent": raw_text_by_agent,
        }

    def _build_duopoly_step_trace(self, step_idx: int, info: dict, raw_text_by_agent: dict[str, str]) -> dict:
        payload = self._build_common_step_trace(step_idx=step_idx, info=info, raw_text_by_agent=raw_text_by_agent)
        payload.update(
            {
                "prices_by_agent": info.get("prices_by_agent", {}),
                "p_monopoly": info.get("p_monopoly"),
                "p_nash": info.get("p_nash"),
            }
        )
        return payload

    def _build_cournot_step_trace(self, step_idx: int, info: dict, raw_text_by_agent: dict[str, str]) -> dict:
        payload = self._build_common_step_trace(step_idx=step_idx, info=info, raw_text_by_agent=raw_text_by_agent)
        payload.update(
            {
                "quantities_by_agent": info.get("quantities_by_agent", {}),
                "market_prices": info.get("market_prices", {}),
                "monopoly_quantities": info.get("monopoly_quantities"),
                "nash_quantities": info.get("nash_quantities"),
            }
        )
        return payload

    def _build_step_trace_entry(self, step_idx: int, info: dict, raw_text_by_agent: dict[str, str]) -> dict:
        env_name = self._get_env_name()
        if env_name == "duopoly":
            return self._build_duopoly_step_trace(step_idx=step_idx, info=info, raw_text_by_agent=raw_text_by_agent)
        if env_name == "cournot":
            return self._build_cournot_step_trace(step_idx=step_idx, info=info, raw_text_by_agent=raw_text_by_agent)
        return self._build_common_step_trace(step_idx=step_idx, info=info, raw_text_by_agent=raw_text_by_agent)

    def _build_duopoly_group_summary(
        self,
        data_source: str,
        records: list[dict],
        created_at: str,
    ) -> dict:
        agent_ids = self._get_agent_ids()
        agent_1, agent_2 = agent_ids[0], agent_ids[1]

        valid_records = [record for record in records if record["valid"]]
        all_invalid_firm1 = [float(record.get("invalid_output_by_agent", {}).get(agent_1, 0.0)) for record in records]
        all_invalid_firm2 = [float(record.get("invalid_output_by_agent", {}).get(agent_2, 0.0)) for record in records]

        def _mean_or_none(values: list[float]) -> float | None:
            if not values:
                return None
            return float(np.mean(values))

        def _collect_agent_metric(records: list[dict], field_name: str, agent_id: str) -> list[float]:
            values = []
            for record in records:
                value = record.get(field_name, {}).get(agent_id)
                if value is not None:
                    values.append(float(value))
            return values

        def _collect_scalar_metric(records: list[dict], field_name: str) -> list[float]:
            values = []
            for record in records:
                value = record.get(field_name)
                if value is not None:
                    values.append(float(value))
            return values

        valid_profit_firm1 = _collect_agent_metric(valid_records, "tail20pct_avg_profit_by_agent", agent_1)
        valid_profit_firm2 = _collect_agent_metric(valid_records, "tail20pct_avg_profit_by_agent", agent_2)
        valid_price_firm1 = _collect_agent_metric(valid_records, "tail20pct_avg_price_by_agent", agent_1)
        valid_price_firm2 = _collect_agent_metric(valid_records, "tail20pct_avg_price_by_agent", agent_2)
        valid_consumer_surplus = _collect_scalar_metric(valid_records, "consumer_surplus_last20pct")

        tail20pct_price_points = []
        for record in valid_records:
            tail_window_size = int(record["tail20pct_window_size"] or 0)
            if tail_window_size <= 0:
                continue
            for step in record["steps"][-tail_window_size:]:
                prices_by_agent = step.get("prices_by_agent", {})
                if agent_1 not in prices_by_agent or agent_2 not in prices_by_agent:
                    continue
                if prices_by_agent.get(agent_1) is None or prices_by_agent.get(agent_2) is None:
                    continue
                tail20pct_price_points.append(
                    {
                        "traj_uid": record["traj_uid"],
                        "step": step.get("step"),
                        "firm1": float(prices_by_agent[agent_1]),
                        "firm2": float(prices_by_agent[agent_2]),
                    }
                )

        episodes = []
        for record in records:
            if record["valid"]:
                episode_payload = {
                    "traj_uid": record["traj_uid"],
                    "data_source": record["data_source"],
                    "valid": True,
                    "tail20pct_avg_profit_by_agent": record.get("tail20pct_avg_profit_by_agent"),
                    "tail20pct_avg_price_by_agent": record.get("tail20pct_avg_price_by_agent"),
                    "consumer_surplus_last20pct": record["consumer_surplus_last20pct"],
                    "invalid_output_by_agent": record.get("invalid_output_by_agent", {agent_1: 0.0, agent_2: 0.0}),
                }
            else:
                episode_payload = {
                    "traj_uid": record["traj_uid"],
                    "data_source": record["data_source"],
                    "valid": False,
                    "tail20pct_avg_profit_by_agent": None,
                    "tail20pct_avg_price_by_agent": None,
                    "consumer_surplus_last20pct": None,
                    "invalid_output_by_agent": record.get("invalid_output_by_agent", {agent_1: 0.0, agent_2: 0.0}),
                }
            episodes.append(episode_payload)

        p_monopoly = next((record["p_monopoly"] for record in records if record["p_monopoly"] is not None), None)
        p_nash = next((record["p_nash"] for record in records if record["p_nash"] is not None), None)

        return {
            "metadata": {
                "experiment_name": str(self.config.trainer.experiment_name),
                "env_name": str(self.config.env.env_name),
                "created_at": created_at,
                "data_source": data_source,
                "episode_count_total": len(records),
                "episode_count_valid": len(valid_records),
                "episode_count_invalid": len(records) - len(valid_records),
            },
            "benchmarks": {
                "p_monopoly": p_monopoly,
                "p_nash": p_nash,
            },
            "overall": {
                "tail20pct_avg_profit_by_agent": {
                    agent_1: _mean_or_none(valid_profit_firm1),
                    agent_2: _mean_or_none(valid_profit_firm2),
                },
                "tail20pct_avg_profit_mean": _mean_or_none(valid_profit_firm1 + valid_profit_firm2),
                "tail20pct_avg_price_by_agent": {
                    agent_1: _mean_or_none(valid_price_firm1),
                    agent_2: _mean_or_none(valid_price_firm2),
                },
                "tail20pct_avg_price_mean": _mean_or_none(valid_price_firm1 + valid_price_firm2),
                "consumer_surplus_last20pct": _mean_or_none(valid_consumer_surplus),
                "invalid_output_rate_by_agent": {
                    agent_1: _mean_or_none(all_invalid_firm1),
                    agent_2: _mean_or_none(all_invalid_firm2),
                },
                "invalid_output_rate_mean": _mean_or_none(all_invalid_firm1 + all_invalid_firm2),
            },
            "episodes": episodes,
            "tail20pct_price_points": tail20pct_price_points,
        }

    def _build_duopoly_record_from_eval_payload(self, payload: dict) -> dict | None:
        agent_ids = self._get_agent_ids()
        if len(agent_ids) < 2:
            return None
        agent_1, agent_2 = agent_ids[0], agent_ids[1]
        data_source = str(payload.get("data_source", "unknown"))
        invalid_output_by_agent = payload.get("invalid_output_by_agent", {})
        valid = not any(float(invalid_output_by_agent.get(agent_id, 0.0)) > 0.0 for agent_id in (agent_1, agent_2))
        return {
            "traj_uid": str(payload.get("traj_uid")),
            "data_source": data_source,
            "valid": valid,
            "tail20pct_window_size": payload.get("tail20pct_window_size"),
            "tail20pct_avg_profit_by_agent": payload.get("tail20pct_avg_profit_by_agent"),
            "tail20pct_avg_price_by_agent": payload.get("tail20pct_avg_price_by_agent"),
            "consumer_surplus_last20pct": payload.get("consumer_surplus_last20pct"),
            "invalid_output_by_agent": invalid_output_by_agent,
            "p_monopoly": payload.get("p_monopoly"),
            "p_nash": payload.get("p_nash"),
            "steps": payload.get("steps", []),
        }

    def _build_cournot_record_from_eval_payload(self, payload: dict) -> dict | None:
        agent_ids = self._get_agent_ids()
        if len(agent_ids) < 2:
            return None
        agent_1, agent_2 = agent_ids[0], agent_ids[1]
        data_source = str(payload.get("data_source", "unknown"))
        steps = payload.get("steps", [])
        invalid_by_agent = steps[-1].get("invalid_by_agent", {}) if steps else {}
        invalid_output_by_agent = {
            agent_1: float(invalid_by_agent.get(agent_1, 0.0)),
            agent_2: float(invalid_by_agent.get(agent_2, 0.0)),
        }
        valid = not any(value > 0.0 for value in invalid_output_by_agent.values())

        tail_window_size = int(payload.get("tail20pct_window_size") or 0)
        tail_steps = steps[-tail_window_size:] if tail_window_size > 0 else []

        def _mean_scalar(values: list[float]) -> float | None:
            if not values:
                return None
            return float(np.mean(values))

        def _mean_step_agent_metric(step_key: str, agent_id: str, sub_key: str | None = None) -> float | None:
            values = []
            for step in tail_steps:
                step_value = step.get(step_key, {}).get(agent_id)
                if step_value is None:
                    continue
                if sub_key is not None:
                    step_value = step_value.get(sub_key)
                if step_value is None:
                    continue
                values.append(float(step_value))
            return _mean_scalar(values)

        def _mean_step_market_metric(metric_key: str) -> float | None:
            values = []
            for step in tail_steps:
                step_value = step.get("market_prices", {}).get(metric_key)
                if step_value is None:
                    continue
                values.append(float(step_value))
            return _mean_scalar(values)

        return {
            "traj_uid": str(payload.get("traj_uid")),
            "data_source": data_source,
            "valid": valid,
            "tail20pct_window_size": payload.get("tail20pct_window_size"),
            "tail20pct_avg_profit_by_agent": {
                agent_1: _mean_step_agent_metric("profits_by_agent", agent_1),
                agent_2: _mean_step_agent_metric("profits_by_agent", agent_2),
            },
            "tail20pct_avg_quantity_by_agent": {
                agent_1: {
                    "product_a": _mean_step_agent_metric("quantities_by_agent", agent_1, "product_a"),
                    "product_b": _mean_step_agent_metric("quantities_by_agent", agent_1, "product_b"),
                },
                agent_2: {
                    "product_a": _mean_step_agent_metric("quantities_by_agent", agent_2, "product_a"),
                    "product_b": _mean_step_agent_metric("quantities_by_agent", agent_2, "product_b"),
                },
            },
            "tail20pct_avg_market_price": {
                "product_a": _mean_step_market_metric("product_a"),
                "product_b": _mean_step_market_metric("product_b"),
            },
            "consumer_surplus_last20pct": payload.get("consumer_surplus_last20pct"),
            "hhi_last20pct": payload.get("hhi_last20pct"),
            "invalid_output_by_agent": invalid_output_by_agent,
            "monopoly_quantities": payload.get("monopoly_quantities"),
            "nash_quantities": payload.get("nash_quantities"),
            "total_units": payload.get("total_units"),
            "alpha": payload.get("reset_info", {}).get("alpha"),
            "neg_inverse_beta": payload.get("reset_info", {}).get("neg_inverse_beta"),
            "steps": steps,
        }

    def _build_duopoly_summary_records(self, dump_dir: str) -> dict[str, list[dict]]:
        grouped_records: dict[str, list[dict]] = {}
        summary_filename = self._get_task_eval_summary_filename()
        for json_path in sorted(Path(dump_dir).glob("*.json")):
            if json_path.name == summary_filename:
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            record = self._build_duopoly_record_from_eval_payload(payload)
            if record is None:
                continue
            grouped_records.setdefault(record["data_source"], []).append(record)
        return grouped_records

    def _build_cournot_summary_records(self, dump_dir: str) -> dict[str, list[dict]]:
        grouped_records: dict[str, list[dict]] = {}
        summary_filename = self._get_task_eval_summary_filename()
        for json_path in sorted(Path(dump_dir).glob("*.json")):
            if json_path.name == summary_filename:
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            record = self._build_cournot_record_from_eval_payload(payload)
            if record is None:
                continue
            grouped_records.setdefault(record["data_source"], []).append(record)
        return grouped_records

    def _build_cournot_group_summary(
        self,
        data_source: str,
        records: list[dict],
        created_at: str,
    ) -> dict:
        agent_1, agent_2 = self._get_agent_ids()[:2]
        valid_records = [record for record in records if record["valid"]]

        def _mean_or_none(values: list[float]) -> float | None:
            if not values:
                return None
            return float(np.mean(values))

        def _collect_agent_metric(records: list[dict], field_name: str, agent_id: str) -> list[float]:
            values = []
            for record in records:
                value = record.get(field_name, {}).get(agent_id)
                if value is not None:
                    values.append(float(value))
            return values

        def _collect_agent_product_metric(records: list[dict], field_name: str, agent_id: str, product_key: str) -> list[float]:
            values = []
            for record in records:
                value = record.get(field_name, {}).get(agent_id, {}).get(product_key)
                if value is not None:
                    values.append(float(value))
            return values

        def _collect_product_metric(records: list[dict], field_name: str, product_key: str) -> list[float]:
            values = []
            for record in records:
                value = record.get(field_name, {}).get(product_key)
                if value is not None:
                    values.append(float(value))
            return values

        valid_profit_firm1 = _collect_agent_metric(valid_records, "tail20pct_avg_profit_by_agent", agent_1)
        valid_profit_firm2 = _collect_agent_metric(valid_records, "tail20pct_avg_profit_by_agent", agent_2)
        valid_qty_1a = _collect_agent_product_metric(valid_records, "tail20pct_avg_quantity_by_agent", agent_1, "product_a")
        valid_qty_1b = _collect_agent_product_metric(valid_records, "tail20pct_avg_quantity_by_agent", agent_1, "product_b")
        valid_qty_2a = _collect_agent_product_metric(valid_records, "tail20pct_avg_quantity_by_agent", agent_2, "product_a")
        valid_qty_2b = _collect_agent_product_metric(valid_records, "tail20pct_avg_quantity_by_agent", agent_2, "product_b")
        valid_price_a = _collect_product_metric(valid_records, "tail20pct_avg_market_price", "product_a")
        valid_price_b = _collect_product_metric(valid_records, "tail20pct_avg_market_price", "product_b")
        valid_cs_a = _collect_product_metric(valid_records, "consumer_surplus_last20pct", "product_a")
        valid_cs_b = _collect_product_metric(valid_records, "consumer_surplus_last20pct", "product_b")
        valid_cs_total = _collect_product_metric(valid_records, "consumer_surplus_last20pct", "total")
        valid_hhi_a = _collect_product_metric(valid_records, "hhi_last20pct", "product_a")
        valid_hhi_b = _collect_product_metric(valid_records, "hhi_last20pct", "product_b")
        valid_hhi_mean = _collect_product_metric(valid_records, "hhi_last20pct", "mean")

        invalid_rate_firm1 = [float(record.get("invalid_output_by_agent", {}).get(agent_1, 0.0)) for record in records]
        invalid_rate_firm2 = [float(record.get("invalid_output_by_agent", {}).get(agent_2, 0.0)) for record in records]

        tail20pct_quantity_points = []
        for record in valid_records:
            tail_window_size = int(record.get("tail20pct_window_size") or 0)
            if tail_window_size <= 0:
                continue
            for step in record.get("steps", [])[-tail_window_size:]:
                quantities = step.get("quantities_by_agent", {})
                market_prices = step.get("market_prices", {})
                q1 = quantities.get(agent_1, {})
                q2 = quantities.get(agent_2, {})
                if (
                    q1.get("product_a") is None
                    or q1.get("product_b") is None
                    or q2.get("product_a") is None
                    or q2.get("product_b") is None
                ):
                    continue
                tail20pct_quantity_points.append(
                    {
                        "traj_uid": record["traj_uid"],
                        "step": step.get("step"),
                        "firm1_product_a": float(q1["product_a"]),
                        "firm1_product_b": float(q1["product_b"]),
                        "firm2_product_a": float(q2["product_a"]),
                        "firm2_product_b": float(q2["product_b"]),
                        "market_price_a": market_prices.get("product_a"),
                        "market_price_b": market_prices.get("product_b"),
                    }
                )

        episodes = []
        for record in records:
            if record["valid"]:
                episode_payload = {
                    "traj_uid": record["traj_uid"],
                    "data_source": record["data_source"],
                    "valid": True,
                    "tail20pct_avg_profit_by_agent": record.get("tail20pct_avg_profit_by_agent"),
                    "tail20pct_avg_quantity_by_agent": record.get("tail20pct_avg_quantity_by_agent"),
                    "tail20pct_avg_market_price": record.get("tail20pct_avg_market_price"),
                    "consumer_surplus_last20pct": record.get("consumer_surplus_last20pct"),
                    "hhi_last20pct": record.get("hhi_last20pct"),
                    "invalid_output_by_agent": record.get("invalid_output_by_agent", {agent_1: 0.0, agent_2: 0.0}),
                }
            else:
                episode_payload = {
                    "traj_uid": record["traj_uid"],
                    "data_source": record["data_source"],
                    "valid": False,
                    "tail20pct_avg_profit_by_agent": None,
                    "tail20pct_avg_quantity_by_agent": None,
                    "tail20pct_avg_market_price": None,
                    "consumer_surplus_last20pct": None,
                    "hhi_last20pct": None,
                    "invalid_output_by_agent": record.get("invalid_output_by_agent", {agent_1: 0.0, agent_2: 0.0}),
                }
            episodes.append(episode_payload)

        monopoly_quantities = next((record["monopoly_quantities"] for record in records if record.get("monopoly_quantities") is not None), None)
        nash_quantities = next((record["nash_quantities"] for record in records if record.get("nash_quantities") is not None), None)
        total_units = next((record["total_units"] for record in records if record.get("total_units") is not None), None)
        alpha = next((record["alpha"] for record in records if record.get("alpha") is not None), None)
        neg_inverse_beta = next((record["neg_inverse_beta"] for record in records if record.get("neg_inverse_beta") is not None), None)

        return {
            "metadata": {
                "experiment_name": str(self.config.trainer.experiment_name),
                "env_name": str(self.config.env.env_name),
                "created_at": created_at,
                "data_source": data_source,
                "episode_count_total": len(records),
                "episode_count_valid": len(valid_records),
                "episode_count_invalid": len(records) - len(valid_records),
            },
            "benchmarks": {
                "monopoly_quantities": monopoly_quantities,
                "nash_quantities": nash_quantities,
                "total_units": total_units,
                "alpha": alpha,
                "neg_inverse_beta": neg_inverse_beta,
            },
            "overall": {
                "tail20pct_avg_profit_by_agent": {
                    agent_1: _mean_or_none(valid_profit_firm1),
                    agent_2: _mean_or_none(valid_profit_firm2),
                },
                "tail20pct_avg_profit_mean": _mean_or_none(valid_profit_firm1 + valid_profit_firm2),
                "tail20pct_avg_quantity_by_agent": {
                    agent_1: {
                        "product_a": _mean_or_none(valid_qty_1a),
                        "product_b": _mean_or_none(valid_qty_1b),
                    },
                    agent_2: {
                        "product_a": _mean_or_none(valid_qty_2a),
                        "product_b": _mean_or_none(valid_qty_2b),
                    },
                },
                "tail20pct_avg_market_price": {
                    "product_a": _mean_or_none(valid_price_a),
                    "product_b": _mean_or_none(valid_price_b),
                },
                "consumer_surplus_last20pct": {
                    "product_a": _mean_or_none(valid_cs_a),
                    "product_b": _mean_or_none(valid_cs_b),
                    "total": _mean_or_none(valid_cs_total),
                },
                "hhi_last20pct": {
                    "product_a": _mean_or_none(valid_hhi_a),
                    "product_b": _mean_or_none(valid_hhi_b),
                    "mean": _mean_or_none(valid_hhi_mean),
                },
                "invalid_output_rate_by_agent": {
                    agent_1: _mean_or_none(invalid_rate_firm1),
                    agent_2: _mean_or_none(invalid_rate_firm2),
                },
                "invalid_output_rate_mean": _mean_or_none(invalid_rate_firm1 + invalid_rate_firm2),
            },
            "episodes": episodes,
            "tail20pct_quantity_points": tail20pct_quantity_points,
        }

    def _build_task_summary_records(self, dump_dir: str) -> dict[str, list[dict]]:
        if self._get_env_name() == "duopoly":
            return self._build_duopoly_summary_records(dump_dir=dump_dir)
        if self._get_env_name() == "cournot":
            return self._build_cournot_summary_records(dump_dir=dump_dir)
        return {}

    def _build_task_group_summary(self, data_source: str, records: list[dict], created_at: str) -> dict | None:
        if self._get_env_name() == "duopoly":
            return self._build_duopoly_group_summary(
                data_source=data_source,
                records=records,
                created_at=created_at,
            )
        if self._get_env_name() == "cournot":
            return self._build_cournot_group_summary(
                data_source=data_source,
                records=records,
                created_at=created_at,
            )
        return None

    def _render_task_summary_artifacts(self, group_summary: dict, dump_dir: str, multiple_groups: bool) -> None:
        if self._get_env_name() == "duopoly":
            from competitive_agent_system.games.duopoly.plotting import plot_tail20pct_price_scatter

            data_source = group_summary.get("metadata", {}).get("data_source", "unknown")
            suffix = "" if not multiple_groups else f"__{self._sanitize_path_component(data_source)}"
            scatter_path = os.path.join(dump_dir, f"duopoly_tail20pct_price_scatter{suffix}.png")
            plot_tail20pct_price_scatter(group_summary, scatter_path)
        elif self._get_env_name() == "cournot":
            from competitive_agent_system.games.cournot.plotting import plot_tail20pct_quantity_scatter

            data_source = group_summary.get("metadata", {}).get("data_source", "unknown")
            suffix = "" if not multiple_groups else f"__{self._sanitize_path_component(data_source)}"
            scatter_path = os.path.join(dump_dir, f"cournot_tail20pct_quantity_scatter{suffix}.png")
            plot_tail20pct_quantity_scatter(group_summary, scatter_path)

    def _get_task_eval_summary_filename(self) -> str:
        env_name = self._get_env_name()
        if env_name == "duopoly":
            return "duopoly_eval_summary.json"
        if env_name == "cournot":
            return "cournot_eval_summary.json"
        return f"{self._sanitize_path_component(env_name)}_eval_summary.json"

    def _dump_task_eval_summary(self) -> None:
        created_at = datetime.now().isoformat(timespec="seconds")
        dump_dir = self._get_eval_dump_dir()
        grouped_records = self._build_task_summary_records(dump_dir=dump_dir)

        if not grouped_records:
            return

        overall_payload = {
            "metadata": {
                "experiment_name": str(self.config.trainer.experiment_name),
                "env_name": str(self.config.env.env_name),
                "created_at": created_at,
                "data_source_count": len(grouped_records),
            },
            "groups": [],
        }

        multiple_groups = len(grouped_records) > 1
        for data_source, records in grouped_records.items():
            group_summary = self._build_task_group_summary(
                data_source=data_source,
                records=records,
                created_at=created_at,
            )
            if group_summary is None:
                continue
            overall_payload["groups"].append(group_summary)
            self._render_task_summary_artifacts(
                group_summary=group_summary,
                dump_dir=dump_dir,
                multiple_groups=multiple_groups,
            )

        summary_path = os.path.join(dump_dir, self._get_task_eval_summary_filename())
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(overall_payload, f, ensure_ascii=False, indent=2)

    def finalize_eval_artifacts(self) -> None:
        self._dump_task_eval_summary()

    def _log_eval_step_progress(self, step_idx: int, infos: list[dict]) -> None:
        batch_prices = [info.get("prices_by_agent", {}) for info in infos]
        if any(prices for prices in batch_prices):
            print(f"[competitive eval] step={step_idx} batch_prices={batch_prices}")
        else:
            batch_quantities = [info.get("quantities_by_agent", {}) for info in infos]
            print(f"[competitive eval] step={step_idx} batch_quantities={batch_quantities}")

    def _get_agent_specific_episode_reward(self, terminal_info: dict, agent_id: str, fallback_reward: float) -> float:
        agent_ids = self._get_agent_ids()
        if len(agent_ids) >= 2:
            if agent_id == agent_ids[0]:
                return float(terminal_info.get("train_reward/firm1", fallback_reward))
            if agent_id == agent_ids[1]:
                return float(terminal_info.get("train_reward/firm2", fallback_reward))
        return float(fallback_reward)

    def gather_rollout_data(self, total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, terminal_infos) -> DataProto:
        effective_batch = []
        for bs in range(len(total_batch_list)):
            terminal_info = terminal_infos[bs] if bs < len(terminal_infos) else {}
            for data in total_batch_list[bs]:
                if data["active_masks"]:
                    episode_reward = self._get_agent_specific_episode_reward(
                        terminal_info=terminal_info,
                        agent_id=data["agent_id"],
                        fallback_reward=episode_rewards[bs],
                    )
                    data["episode_rewards"] = np.asarray(episode_reward, dtype=np.float32)
                    data["episode_lengths"] = np.asarray(episode_lengths[bs], dtype=np.float32)
                    data["tool_callings"] = np.asarray(tool_callings[bs], dtype=np.float32)
                    data["pass"] = np.asarray(success["success_rate"][bs], dtype=np.float32)
                    if terminal_info:
                        for key in self._get_task_rollout_metric_fields():
                            if key in terminal_info:
                                data[key] = terminal_info[key]
                    effective_batch.append(data)

        return DataProto.from_single_dict(data=collate_fn(effective_batch))

    def vanilla_multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg, envs, effective_rollout_n: int, dump_eval_traces: bool = False):
        batch_size = len(gen_batch.batch)
        obs, reset_infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop("env_kwargs", None))
        self.orchestra.reset()

        uid_batch = []
        for i in range(batch_size):
            if effective_rollout_n <= 0 or i % effective_rollout_n == 0:
                uid = str(uuid.uuid4())
            uid_batch.append(uid)
        uid_batch = np.array(uid_batch, dtype=object)

        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        step_traces = [[] for _ in range(batch_size)]
        log_eval_progress = dump_eval_traces and self._is_eval_only()

        for step_idx in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            # if log_eval_progress:
            #     print(f"[competitive eval] step={step_idx + 1} entering_run_turn active_runs={int(np.count_nonzero(active_masks))}")
            actions_by_agent, multiagent_batch_buffer = self.orchestra.run_turn(
                gen_batch=gen_batch,
                env_obs=obs,
                actor_rollout_wgs=actor_rollout_wg,
                active_masks=active_masks,
                step=step_idx + 1,
            )
            # if log_eval_progress:
            #     print(f"[competitive eval] step={step_idx + 1} finished_run_turn")
            next_obs, rewards, dones, infos = envs.step(actions_by_agent)
            # if log_eval_progress:
            #     print(f"[competitive eval] step={step_idx + 1} finished_env_step")

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)

            tool_callings[active_masks] += np.array([info.get("tool_calling", 0.0) for info in infos], dtype=np.float32)[active_masks]
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            for i in range(batch_size):
                if not active_masks[i]:
                    continue
                raw_text_by_agent = {
                    agent_id: actions_by_agent[agent_id][i].raw_text for agent_id in self._get_agent_ids()
                }
                step_traces[i].append(self._build_step_trace_entry(step_idx=step_idx, info=infos[i], raw_text_by_agent=raw_text_by_agent))

            for data in multiagent_batch_buffer:
                agent_id, agent_batch = data["agent_id"], data["batch"]
                agent_batch.non_tensor_batch["agent_id"] = np.array([agent_id for _ in range(batch_size)], dtype=object)
                agent_batch.non_tensor_batch["uid"] = uid_batch
                agent_batch.non_tensor_batch["traj_uid"] = traj_uid
                agent_batch.non_tensor_batch["rewards"] = torch_to_numpy(rewards, is_object=True)
                agent_batch.non_tensor_batch["active_masks"] = torch_to_numpy(active_masks, is_object=True)
                agent_batch_list = to_list_of_dict(agent_batch)
                for i in range(batch_size):
                    if agent_batch_list[i]["agent_active_mask"]:
                        total_batch_list[i].append(agent_batch_list[i])
                        total_infos[i].append(infos[i])

            if log_eval_progress:
                self._log_eval_step_progress(step_idx=step_idx + 1, infos=infos)

            is_done = np.logical_or(is_done, dones)
            obs = next_obs
            if is_done.all():
                break

        success = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        terminal_infos = self._extract_terminal_infos(total_batch_list=total_batch_list, total_infos=total_infos)
        if dump_eval_traces:
            self._dump_eval_step_traces(
                step_traces=step_traces,
                uid_batch=uid_batch,
                traj_uid=traj_uid,
                reset_infos=reset_infos,
                terminal_infos=terminal_infos,
            )
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, terminal_infos

    def multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg, envs, is_train: bool = True) -> DataProto:
        if is_train:
            effective_rollout_n = self.config.env.rollout.n
            gen_batch = gen_batch.repeat(repeat_times=effective_rollout_n, interleave=True)
        else:
            val_rollout_n = getattr(self.config.env.rollout, "val_n", None)
            if val_rollout_n is not None and val_rollout_n > 1:
                effective_rollout_n = val_rollout_n
                gen_batch = gen_batch.repeat(repeat_times=effective_rollout_n, interleave=True)
            else:
                effective_rollout_n = 1

        total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings, terminal_infos = self.vanilla_multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            effective_rollout_n=effective_rollout_n,
            dump_eval_traces=not is_train,
        )

        return self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=total_tool_callings,
            terminal_infos=terminal_infos,
        )
