from __future__ import annotations

import importlib

import numpy as np

from competitive_agent_system.agents.registry import CompetitiveAgentRegistry
from competitive_agent_system.games.duopoly import DuopolyActionParser
from agent_system.multi_turn_rollout.utils import to_list_of_dict
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn


class CompetitiveTurnOrchestra:
    def __init__(self, config, tokenizers, processors, agents_to_wg_mapping):
        importlib.import_module("competitive_agent_system.agents.pricing")

        self.config = config
        self.tokenizers = tokenizers
        self.processors = processors
        self.agents_to_wg_mapping = agents_to_wg_mapping
        self.agent_ids = list(config.agent.agent_ids)

        self.agents = {
            name: CompetitiveAgentRegistry.create(
                name=name,
                wg_id=agents_to_wg_mapping[name],
                tokenizer=tokenizers[agents_to_wg_mapping[name]],
                processor=processors[agents_to_wg_mapping[name]],
                config=config,
            )
            for name in self.agent_ids
        }
        self.parser = DuopolyActionParser(max_retries=int(config.env.duopoly.max_parse_retry))

    def reset(self):
        for agent in self.agents.values():
            agent.reset()

    def _call_single_agent(self, agent_id: str, gen_batch: DataProto, obs_texts: list[str], actor_rollout_wg, active_masks: np.ndarray, step: int):
        attempts = np.zeros(len(obs_texts), dtype=np.int32)
        remaining = active_masks.copy()
        saved_rows = None
        max_prices = [self.parser.extract_ceiling_from_observation(obs_text) for obs_text in obs_texts]
        saved_actions = [self.parser.parse("", max_price=max_prices[idx]) for idx in range(len(obs_texts))]

        while remaining.any() and np.any(attempts[remaining] < self.parser.max_retries):
            batch, text_responses = self.agents[agent_id].call(
                gen_batch=gen_batch,
                obs_texts=obs_texts,
                actor_rollout_wg=actor_rollout_wg,
                agent_active_mask=remaining,
                step=step,
            )
            row_list = to_list_of_dict(batch)

            if saved_rows is None:
                saved_rows = row_list
            next_remaining = remaining.copy()
            for idx in range(len(obs_texts)):
                if not remaining[idx]:
                    continue

                attempts[idx] += 1
                parsed = self.parser.parse(text_responses[idx], max_price=max_prices[idx])
                parsed.retry_count = int(attempts[idx])

                row_list[idx]["is_action_valid"] = parsed.valid
                saved_rows[idx] = row_list[idx]
                saved_actions[idx] = parsed
                next_remaining[idx] = (not parsed.valid) and (attempts[idx] < self.parser.max_retries)

            remaining = next_remaining

        final_batch = DataProto.from_single_dict(collate_fn(saved_rows))
        final_batch.non_tensor_batch["is_action_valid"] = np.array([action.valid for action in saved_actions], dtype=bool)
        final_batch.non_tensor_batch["env_step"] = np.array([step] * len(saved_actions), dtype=object)
        return final_batch, saved_actions

    def run_turn(self, gen_batch: DataProto, env_obs, actor_rollout_wgs, active_masks: np.ndarray, step: int):
        per_agent_text = env_obs["per_agent_text"]
        actions_by_agent = {}
        multiagent_batch_buffer = []

        for agent_id in self.agent_ids:
            wg_id = self.agents_to_wg_mapping[agent_id]
            batch, actions = self._call_single_agent(
                agent_id=agent_id,
                gen_batch=gen_batch,
                obs_texts=per_agent_text[agent_id],
                actor_rollout_wg=actor_rollout_wgs[wg_id],
                active_masks=active_masks,
                step=step,
            )
            actions_by_agent[agent_id] = actions
            multiagent_batch_buffer.append({"agent_id": agent_id, "batch": batch})

        return actions_by_agent, multiagent_batch_buffer
