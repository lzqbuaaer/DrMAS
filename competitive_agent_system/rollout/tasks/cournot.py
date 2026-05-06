from __future__ import annotations

import os

import numpy as np

from competitive_agent_system.games.cournot.plotting import plot_tail20pct_quantity_scatter
from competitive_agent_system.rollout.tasks.base import BaseCompetitiveRolloutTaskHandler


class CournotRolloutTaskHandler(BaseCompetitiveRolloutTaskHandler):
    def get_eval_trace_extra_keys(self) -> tuple[str, ...]:
        return ("monopoly_quantities", "nash_quantities", "total_units")

    def build_step_trace(self, step_idx: int, info: dict, raw_text_by_agent: dict[str, str]) -> dict:
        payload = self.build_common_step_trace(step_idx=step_idx, info=info, raw_text_by_agent=raw_text_by_agent)
        payload.update(
            {
                "quantities_by_agent": info.get("quantities_by_agent", {}),
                "market_prices": info.get("market_prices", {}),
                "monopoly_quantities": info.get("monopoly_quantities"),
                "nash_quantities": info.get("nash_quantities"),
            }
        )
        return payload

    def get_eval_summary_filename(self) -> str:
        return "cournot_eval_summary.json"

    def build_record_from_eval_payload(self, payload: dict) -> dict | None:
        if len(self.agent_ids) < 2:
            return None
        agent_1, agent_2 = self.agent_ids[:2]
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

        def mean_scalar(values: list[float]) -> float | None:
            if not values:
                return None
            return float(np.mean(values))

        def mean_step_agent_metric(step_key: str, agent_id: str, sub_key: str | None = None) -> float | None:
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
            return mean_scalar(values)

        def mean_step_market_metric(metric_key: str) -> float | None:
            values = []
            for step in tail_steps:
                step_value = step.get("market_prices", {}).get(metric_key)
                if step_value is None:
                    continue
                values.append(float(step_value))
            return mean_scalar(values)

        return {
            "traj_uid": str(payload.get("traj_uid")),
            "data_source": data_source,
            "valid": valid,
            "tail20pct_window_size": payload.get("tail20pct_window_size"),
            "tail20pct_avg_profit_by_agent": {
                agent_1: mean_step_agent_metric("profits_by_agent", agent_1),
                agent_2: mean_step_agent_metric("profits_by_agent", agent_2),
            },
            "tail20pct_avg_quantity_by_agent": {
                agent_1: {
                    "product_a": mean_step_agent_metric("quantities_by_agent", agent_1, "product_a"),
                    "product_b": mean_step_agent_metric("quantities_by_agent", agent_1, "product_b"),
                },
                agent_2: {
                    "product_a": mean_step_agent_metric("quantities_by_agent", agent_2, "product_a"),
                    "product_b": mean_step_agent_metric("quantities_by_agent", agent_2, "product_b"),
                },
            },
            "tail20pct_avg_market_price": {
                "product_a": mean_step_market_metric("product_a"),
                "product_b": mean_step_market_metric("product_b"),
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

    def build_summary_records(self, dump_dir: str) -> dict[str, list[dict]]:
        grouped_records: dict[str, list[dict]] = {}
        for payload in self.iter_eval_payloads(dump_dir):
            record = self.build_record_from_eval_payload(payload)
            if record is None:
                continue
            grouped_records.setdefault(record["data_source"], []).append(record)
        return grouped_records

    def build_group_summary(self, data_source: str, records: list[dict], created_at: str) -> dict:
        agent_1, agent_2 = self.agent_ids[:2]
        valid_records = [record for record in records if record["valid"]]

        valid_profit_firm1 = self.collect_agent_metric(valid_records, "tail20pct_avg_profit_by_agent", agent_1)
        valid_profit_firm2 = self.collect_agent_metric(valid_records, "tail20pct_avg_profit_by_agent", agent_2)
        valid_qty_1a = self.collect_agent_product_metric(valid_records, "tail20pct_avg_quantity_by_agent", agent_1, "product_a")
        valid_qty_1b = self.collect_agent_product_metric(valid_records, "tail20pct_avg_quantity_by_agent", agent_1, "product_b")
        valid_qty_2a = self.collect_agent_product_metric(valid_records, "tail20pct_avg_quantity_by_agent", agent_2, "product_a")
        valid_qty_2b = self.collect_agent_product_metric(valid_records, "tail20pct_avg_quantity_by_agent", agent_2, "product_b")
        valid_price_a = self.collect_product_metric(valid_records, "tail20pct_avg_market_price", "product_a")
        valid_price_b = self.collect_product_metric(valid_records, "tail20pct_avg_market_price", "product_b")
        valid_cs_a = self.collect_product_metric(valid_records, "consumer_surplus_last20pct", "product_a")
        valid_cs_b = self.collect_product_metric(valid_records, "consumer_surplus_last20pct", "product_b")
        valid_cs_total = self.collect_product_metric(valid_records, "consumer_surplus_last20pct", "total")
        valid_hhi_a = self.collect_product_metric(valid_records, "hhi_last20pct", "product_a")
        valid_hhi_b = self.collect_product_metric(valid_records, "hhi_last20pct", "product_b")
        valid_hhi_mean = self.collect_product_metric(valid_records, "hhi_last20pct", "mean")

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
                    agent_1: self.mean_or_none(valid_profit_firm1),
                    agent_2: self.mean_or_none(valid_profit_firm2),
                },
                "tail20pct_avg_profit_mean": self.mean_or_none(valid_profit_firm1 + valid_profit_firm2),
                "tail20pct_avg_quantity_by_agent": {
                    agent_1: {
                        "product_a": self.mean_or_none(valid_qty_1a),
                        "product_b": self.mean_or_none(valid_qty_1b),
                    },
                    agent_2: {
                        "product_a": self.mean_or_none(valid_qty_2a),
                        "product_b": self.mean_or_none(valid_qty_2b),
                    },
                },
                "tail20pct_avg_market_price": {
                    "product_a": self.mean_or_none(valid_price_a),
                    "product_b": self.mean_or_none(valid_price_b),
                },
                "consumer_surplus_last20pct": {
                    "product_a": self.mean_or_none(valid_cs_a),
                    "product_b": self.mean_or_none(valid_cs_b),
                    "total": self.mean_or_none(valid_cs_total),
                },
                "hhi_last20pct": {
                    "product_a": self.mean_or_none(valid_hhi_a),
                    "product_b": self.mean_or_none(valid_hhi_b),
                    "mean": self.mean_or_none(valid_hhi_mean),
                },
                "invalid_output_rate_by_agent": {
                    agent_1: self.mean_or_none(invalid_rate_firm1),
                    agent_2: self.mean_or_none(invalid_rate_firm2),
                },
                "invalid_output_rate_mean": self.mean_or_none(invalid_rate_firm1 + invalid_rate_firm2),
            },
            "episodes": episodes,
            "tail20pct_quantity_points": tail20pct_quantity_points,
        }

    def render_summary_artifacts(self, group_summary: dict, dump_dir: str, multiple_groups: bool) -> None:
        data_source = group_summary.get("metadata", {}).get("data_source", "unknown")
        suffix = "" if not multiple_groups else f"__{self.sanitize_path_component(data_source)}"
        scatter_path = os.path.join(dump_dir, f"cournot_tail20pct_quantity_scatter{suffix}.png")
        plot_tail20pct_quantity_scatter(group_summary, scatter_path)

    def log_eval_step(self, step_idx: int, infos: list[dict], active_masks) -> None:
        step_payload = []
        for idx, info in enumerate(infos):
            if not active_masks[idx]:
                continue
            step_payload.append(
                {
                    "data_source": info.get("data_source"),
                    "quantities_by_agent": info.get("quantities_by_agent", {}),
                }
            )
        print(f"[competitive eval] step={step_idx} active_runs={len(step_payload)} batch_quantities={step_payload}")
