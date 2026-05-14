from __future__ import annotations

from dataclasses import asdict, dataclass


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _to_attrdict(value):
    if isinstance(value, dict):
        return AttrDict({k: _to_attrdict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attrdict(v) for v in value]
    return value


@dataclass
class SamplingConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 1024


@dataclass
class ProviderConfig:
    provider: str
    model: str
    api_key_env: str
    base_url: str | None = None
    reasoning_effort: str | None = None
    thinking_enabled: bool = False
    timeout: float = 120.0
    max_http_retries: int = 3


@dataclass
class ApiEvalConfig:
    task: str
    experiment_name: str
    data_file: str
    episode_count: int | None
    output_root: str = "eval_data"
    concurrency: int = 1
    max_retries: int | None = None
    seed: int = 0
    sampling: SamplingConfig | None = None
    provider: ProviderConfig | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def sanitize_path_component(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)
    return sanitized or "unknown"


def build_runtime_config(
    *,
    task: str,
    experiment_name: str,
    max_steps: int,
    duopoly_beta: float = 100.0,
    duopoly_prompt_prefix_type: str = "P1",
    cournot_market_data_length: int = 15,
) -> AttrDict:
    config = {
        "data": {
            "train_batch_size": 1,
            "val_batch_size": 1,
        },
        "trainer": {
            "experiment_name": str(experiment_name),
            "val_only": True,
        },
        "agent": {
            "agent_ids": ["Firm 1 Agent", "Firm 2 Agent"],
        },
        "env": {
            "env_name": str(task),
            "seed": 0,
            "max_steps": int(max_steps),
            "duopoly": {
                "alpha": 1.0,
                "beta": float(duopoly_beta),
                "prompt_prefix_type": str(duopoly_prompt_prefix_type),
                "history_window": 100,
                "max_parse_retry": 10,
                "rollout_metric_fields": [
                    "tail20pct_window_size",
                    "cooperation_last20pct/firm1",
                    "cooperation_last20pct/firm2",
                    "collusion_last20pct/firm1",
                    "collusion_last20pct/firm2",
                    "invalid_output/firm1",
                    "invalid_output/firm2",
                    "tail20pct_avg_price/firm1",
                    "tail20pct_avg_price/firm2",
                    "tail20pct_avg_profit/firm1",
                    "tail20pct_avg_profit/firm2",
                    "consumer_surplus_last20pct",
                    "train_reward/firm1",
                    "train_reward/firm2",
                ],
                "eval_dump_fields": {
                    "tail20pct_window_size": "tail20pct_window_size",
                    "cooperation_last20pct_by_agent": {
                        "kind": "agent_pair",
                        "firm1": "cooperation_last20pct/firm1",
                        "firm2": "cooperation_last20pct/firm2",
                    },
                    "collusion_last20pct_by_agent": {
                        "kind": "agent_pair",
                        "firm1": "collusion_last20pct/firm1",
                        "firm2": "collusion_last20pct/firm2",
                    },
                    "tail20pct_avg_price_by_agent": {
                        "kind": "agent_pair",
                        "firm1": "tail20pct_avg_price/firm1",
                        "firm2": "tail20pct_avg_price/firm2",
                    },
                    "tail20pct_avg_profit_by_agent": {
                        "kind": "agent_pair",
                        "firm1": "tail20pct_avg_profit/firm1",
                        "firm2": "tail20pct_avg_profit/firm2",
                    },
                    "consumer_surplus_last20pct": "consumer_surplus_last20pct",
                },
            },
            "cournot": {
                "alpha": 100.0,
                "neg_inverse_beta": -0.5,
                "total_units": 100.0,
                "market_data_length": int(cournot_market_data_length),
                "max_parse_retry": 10,
                "flex_total_prod": True,
                "marginal_cost_1a": 40.0,
                "marginal_cost_1b": 50.0,
                "marginal_cost_2a": 50.0,
                "marginal_cost_2b": 40.0,
                "rollout_metric_fields": [
                    "tail20pct_window_size",
                    "invalid_output/firm1",
                    "invalid_output/firm2",
                    "tail20pct_avg_profit/firm1",
                    "tail20pct_avg_profit/firm2",
                    "tail20pct_avg_quantity_a/firm1",
                    "tail20pct_avg_quantity_a/firm2",
                    "tail20pct_avg_quantity_b/firm1",
                    "tail20pct_avg_quantity_b/firm2",
                    "tail20pct_avg_market_price/product_a",
                    "tail20pct_avg_market_price/product_b",
                    "train_reward/firm1",
                    "train_reward/firm2",
                    "hhi_last20pct/product_a",
                    "hhi_last20pct/product_b",
                    "hhi_last20pct/mean",
                    "consumer_surplus_last20pct/product_a",
                    "consumer_surplus_last20pct/product_b",
                    "consumer_surplus_last20pct/total",
                ],
                "eval_dump_fields": {
                    "tail20pct_window_size": "tail20pct_window_size",
                    "hhi_last20pct": {
                        "kind": "dict",
                        "fields": {
                            "product_a": "hhi_last20pct/product_a",
                            "product_b": "hhi_last20pct/product_b",
                            "mean": "hhi_last20pct/mean",
                        },
                    },
                    "consumer_surplus_last20pct": {
                        "kind": "dict",
                        "fields": {
                            "product_a": "consumer_surplus_last20pct/product_a",
                            "product_b": "consumer_surplus_last20pct/product_b",
                            "total": "consumer_surplus_last20pct/total",
                        },
                    },
                },
            },
        },
    }
    return _to_attrdict(config)
