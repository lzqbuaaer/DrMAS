from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", help="Path to eval_data/<experiment>/<timestamp>")
    parser.add_argument("--task", choices=["duopoly", "cournot"], default=None)
    parser.add_argument("--experiment-name", default=None)
    return parser.parse_args()


def load_run_config(eval_dir: Path) -> dict:
    run_config_path = eval_dir / "run_config.json"
    if not run_config_path.exists():
        return {}
    return json.loads(run_config_path.read_text(encoding="utf-8"))


def infer_metadata(eval_dir: Path, args) -> tuple[str, str, list[str], int]:
    payload = load_run_config(eval_dir)
    runtime_config = payload.get("runtime_config", {})
    eval_config = payload.get("eval_config", {})

    task = args.task or runtime_config.get("env_name") or eval_config.get("task")
    if not task:
        raise ValueError("Unable to infer task. Please pass --task explicitly.")

    experiment_name = (
        args.experiment_name
        or eval_config.get("experiment_name")
        or runtime_config.get("experiment_name")
        or eval_dir.parent.name
    )
    agent_ids = runtime_config.get("agent_ids") or ["Firm 1 Agent", "Firm 2 Agent"]
    max_steps = int(runtime_config.get("max_steps", 20))
    return str(task), str(experiment_name), list(agent_ids), max_steps


def main():
    args = parse_args()

    from competitive_agent_system.api_eval.config import build_runtime_config, sanitize_path_component
    from competitive_agent_system.api_eval.tasks import create_task_adapter

    eval_dir = Path(args.eval_dir).expanduser().resolve()
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval directory does not exist: {eval_dir}")
    if not eval_dir.is_dir():
        raise NotADirectoryError(f"Eval path is not a directory: {eval_dir}")

    task, experiment_name, agent_ids, max_steps = infer_metadata(eval_dir, args)
    runtime_config = build_runtime_config(
        task=task,
        experiment_name=experiment_name,
        max_steps=max_steps,
    )
    task_adapter = create_task_adapter(
        task=task,
        config=runtime_config,
        agent_ids=agent_ids,
        sanitize_path_component=sanitize_path_component,
    )
    task_adapter.finalize_artifacts(eval_dir)
    print(f"Regenerated summary and plots in {eval_dir}")


if __name__ == "__main__":
    main()
