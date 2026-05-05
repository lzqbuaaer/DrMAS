from __future__ import annotations

from competitive_agent_system.rollout.tasks.base import BaseCompetitiveRolloutTaskHandler
from competitive_agent_system.rollout.tasks.cournot import CournotRolloutTaskHandler
from competitive_agent_system.rollout.tasks.duopoly import DuopolyRolloutTaskHandler


def create_task_handler(config, agent_ids: list[str], sanitize_path_component) -> BaseCompetitiveRolloutTaskHandler:
    env_name = str(config.env.env_name).lower()
    if env_name == "duopoly":
        return DuopolyRolloutTaskHandler(config=config, agent_ids=agent_ids, sanitize_path_component=sanitize_path_component)
    if env_name == "cournot":
        return CournotRolloutTaskHandler(config=config, agent_ids=agent_ids, sanitize_path_component=sanitize_path_component)
    return BaseCompetitiveRolloutTaskHandler(config=config, agent_ids=agent_ids, sanitize_path_component=sanitize_path_component)
