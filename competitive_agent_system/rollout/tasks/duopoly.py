from __future__ import annotations

import os

from competitive_agent_system.games.duopoly.plotting import plot_tail20pct_price_scatter
from competitive_agent_system.rollout.tasks.base import BaseCompetitiveRolloutTaskHandler


class DuopolyRolloutTaskHandler(BaseCompetitiveRolloutTaskHandler):
    def get_eval_trace_extra_keys(self) -> tuple[str, ...]:
        return ("p_monopoly", "p_nash", "ceiling")

    def build_step_trace(self, step_idx: int, info: dict, raw_text_by_agent: dict[str, str]) -> dict:
        payload = self.build_common_step_trace(step_idx=step_idx, info=info, raw_text_by_agent=raw_text_by_agent)
        payload.update(
            {
                "prices_by_agent": info.get("prices_by_agent", {}),
                "p_monopoly": info.get("p_monopoly"),
                "p_nash": info.get("p_nash"),
            }
        )
        return payload

    def get_eval_summary_filename(self) -> str:
        return "duopoly_eval_summary.json"

    def build_record_from_eval_payload(self, payload: dict) -> dict | None:
        if len(self.agent_ids) < 2:
            return None
        agent_1, agent_2 = self.agent_ids[:2]
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
        all_invalid_firm1 = [float(record.get("invalid_output_by_agent", {}).get(agent_1, 0.0)) for record in records]
        all_invalid_firm2 = [float(record.get("invalid_output_by_agent", {}).get(agent_2, 0.0)) for record in records]
        valid_profit_firm1 = self.collect_agent_metric(valid_records, "tail20pct_avg_profit_by_agent", agent_1)
        valid_profit_firm2 = self.collect_agent_metric(valid_records, "tail20pct_avg_profit_by_agent", agent_2)
        valid_price_firm1 = self.collect_agent_metric(valid_records, "tail20pct_avg_price_by_agent", agent_1)
        valid_price_firm2 = self.collect_agent_metric(valid_records, "tail20pct_avg_price_by_agent", agent_2)
        valid_consumer_surplus = self.collect_scalar_metric(valid_records, "consumer_surplus_last20pct")

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
                    agent_1: self.mean_or_none(valid_profit_firm1),
                    agent_2: self.mean_or_none(valid_profit_firm2),
                },
                "tail20pct_avg_profit_mean": self.mean_or_none(valid_profit_firm1 + valid_profit_firm2),
                "tail20pct_avg_price_by_agent": {
                    agent_1: self.mean_or_none(valid_price_firm1),
                    agent_2: self.mean_or_none(valid_price_firm2),
                },
                "tail20pct_avg_price_mean": self.mean_or_none(valid_price_firm1 + valid_price_firm2),
                "consumer_surplus_last20pct": self.mean_or_none(valid_consumer_surplus),
                "invalid_output_rate_by_agent": {
                    agent_1: self.mean_or_none(all_invalid_firm1),
                    agent_2: self.mean_or_none(all_invalid_firm2),
                },
                "invalid_output_rate_mean": self.mean_or_none(all_invalid_firm1 + all_invalid_firm2),
            },
            "episodes": episodes,
            "tail20pct_price_points": tail20pct_price_points,
        }

    def render_summary_artifacts(self, group_summary: dict, dump_dir: str, multiple_groups: bool) -> None:
        data_source = group_summary.get("metadata", {}).get("data_source", "unknown")
        suffix = "" if not multiple_groups else f"__{self.sanitize_path_component(data_source)}"
        scatter_path = os.path.join(dump_dir, f"duopoly_tail20pct_price_scatter{suffix}.png")
        plot_tail20pct_price_scatter(group_summary, scatter_path)

    def log_eval_step(self, step_idx: int, infos: list[dict], active_masks) -> None:
        step_payload = []
        for idx, info in enumerate(infos):
            if not active_masks[idx]:
                continue
            step_payload.append(
                {
                    "data_source": info.get("data_source"),
                    "prices_by_agent": info.get("prices_by_agent", {}),
                    "profits_by_agent": info.get("profits_by_agent", {}),
                    "failure_reason": info.get("failure_reason"),
                }
            )
        print(f"[competitive eval] step={step_idx} active_runs={len(step_payload)} batch_prices={step_payload}")
