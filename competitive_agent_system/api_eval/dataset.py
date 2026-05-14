from __future__ import annotations

import ast
import json

import pandas as pd


def _coerce_env_kwargs(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            return dict(json.loads(value))
        except json.JSONDecodeError:
            return dict(ast.literal_eval(value))
    raise TypeError(f"Unsupported env_kwargs type: {type(value)!r}")


def load_env_kwargs_from_parquet(path: str, episode_count: int | None = None) -> list[dict]:
    df = pd.read_parquet(path)
    if episode_count is not None:
        df = df.head(int(episode_count))
    env_kwargs_list = [_coerce_env_kwargs(value) for value in df["env_kwargs"].tolist()]
    return env_kwargs_list
