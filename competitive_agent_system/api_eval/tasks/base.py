from __future__ import annotations

import json
from pathlib import Path

from competitive_agent_system.rollout.task_registry import create_task_handler


class ApiEvalTaskAdapter:
    def __init__(self, config, agent_ids: list[str], sanitize_path_component):
        self.config = config
        self.agent_ids = list(agent_ids)
        self.sanitize_path_component = sanitize_path_component
        self.task_handler = create_task_handler(
            config=config,
            agent_ids=self.agent_ids,
            sanitize_path_component=sanitize_path_component,
        )

    def build_env(self):
        raise NotImplementedError

    def build_parser(self):
        raise NotImplementedError

    def build_parse_kwargs(self, observation: str, parser) -> dict:
        raise NotImplementedError

    def get_eval_dump_fields(self):
        env_cfg = self.config.env.get(str(self.config.env.env_name).lower(), {})
        return env_cfg.get("eval_dump_fields", {}) or {}

    def resolve_eval_dump_payload(self, terminal_info: dict) -> dict:
        resolved_payload = {}
        if not terminal_info:
            return resolved_payload

        for payload_key, spec in self.get_eval_dump_fields().items():
            if isinstance(spec, str):
                resolved_payload[payload_key] = terminal_info.get(spec)
                continue
            if not hasattr(spec, "get"):
                continue

            kind = spec.get("kind", "scalar")
            if kind == "agent_pair":
                resolved_payload[payload_key] = {
                    self.agent_ids[0]: terminal_info.get(spec.get("firm1")),
                    self.agent_ids[1]: terminal_info.get(spec.get("firm2")),
                }
            elif kind == "dict":
                fields = spec.get("fields", {})
                resolved_payload[payload_key] = {sub_key: terminal_info.get(source_key) for sub_key, source_key in fields.items()}
            elif kind == "scalar":
                resolved_payload[payload_key] = terminal_info.get(spec.get("field"))
        return resolved_payload

    def build_episode_payload(
        self,
        *,
        uid: str,
        traj_uid: str,
        reset_info: dict,
        terminal_info: dict,
        step_traces: list[dict],
    ) -> dict:
        first_step = step_traces[0] if step_traces else {}
        payload = {
            "uid": uid,
            "traj_uid": traj_uid,
            "data_source": first_step.get("data_source", reset_info.get("data_source")),
            "steps": step_traces,
            "reset_info": reset_info,
        }
        for key in self.task_handler.get_eval_trace_extra_keys():
            if key in reset_info:
                payload[key] = reset_info.get(key)
            elif key in first_step:
                payload[key] = first_step.get(key)
        payload.update(self.resolve_eval_dump_payload(terminal_info))
        return payload

    def finalize_artifacts(self, output_dir: Path) -> None:
        created_at = __import__("datetime").datetime.now().isoformat(timespec="seconds")
        grouped_records = self.task_handler.build_summary_records(str(output_dir))
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
            group_summary = self.task_handler.build_group_summary(
                data_source=data_source,
                records=records,
                created_at=created_at,
            )
            if group_summary is None:
                continue
            overall_payload["groups"].append(group_summary)
            self.task_handler.render_summary_artifacts(
                group_summary=group_summary,
                dump_dir=str(output_dir),
                multiple_groups=multiple_groups,
            )

        summary_path = output_dir / self.task_handler.get_eval_summary_filename()
        summary_path.write_text(json.dumps(overall_payload, ensure_ascii=False, indent=2), encoding="utf-8")


class DuopolyApiEvalTaskAdapter(ApiEvalTaskAdapter):
    def build_env(self):
        from competitive_agent_system.environments.env_manager import DuopolyArenaEnv

        return DuopolyArenaEnv(self.config, self.agent_ids)

    def build_parser(self):
        from competitive_agent_system.games.duopoly import DuopolyActionParser

        return DuopolyActionParser(max_retries=int(self.config.env.duopoly.max_parse_retry))

    def build_parse_kwargs(self, observation: str, parser) -> dict:
        return {"max_price": parser.extract_ceiling_from_observation(observation)}


class CournotApiEvalTaskAdapter(ApiEvalTaskAdapter):
    def build_env(self):
        from competitive_agent_system.environments.env_manager import CournotArenaEnv

        return CournotArenaEnv(self.config, self.agent_ids)

    def build_parser(self):
        from competitive_agent_system.games.cournot.parser import CournotActionParser

        return CournotActionParser(max_retries=int(self.config.env.cournot.max_parse_retry))

    def build_parse_kwargs(self, observation: str, parser) -> dict:
        return {"total_units": parser.extract_total_units_from_observation(observation)}


def create_task_adapter(task: str, config, agent_ids: list[str], sanitize_path_component):
    normalized_task = str(task).lower()
    if "duopoly" in normalized_task:
        return DuopolyApiEvalTaskAdapter(config=config, agent_ids=agent_ids, sanitize_path_component=sanitize_path_component)
    if "cournot" in normalized_task:
        return CournotApiEvalTaskAdapter(config=config, agent_ids=agent_ids, sanitize_path_component=sanitize_path_component)
    raise ValueError(f"Unsupported API eval task '{task}'")
