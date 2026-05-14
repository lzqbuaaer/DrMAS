from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def create_output_dir(output_root: str, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / experiment_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def dump_episode_json(output_dir: Path, traj_uid: str, payload: dict) -> None:
    path = output_dir / f"{traj_uid}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_run_config(output_dir: Path, payload: dict) -> None:
    path = output_dir / "run_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
