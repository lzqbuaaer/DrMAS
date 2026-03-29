from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

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

    def _dump_eval_step_traces(self, step_traces, uid_batch, traj_uid, reset_infos) -> None:
        dump_dir = self._get_eval_dump_dir()
        for idx, trace in enumerate(step_traces):
            if not trace:
                continue

            reset_info = reset_infos[idx] if idx < len(reset_infos) else {}
            first_step = trace[0]
            payload = {
                "uid": str(uid_batch[idx]),
                "traj_uid": str(traj_uid[idx]),
                "data_source": first_step["data_source"],
                "p_monopoly": first_step["p_monopoly"],
                "p_nash": first_step["p_nash"],
                "steps": trace,
                "reset_info": reset_info,
            }

            filename = os.path.join(dump_dir, f"{traj_uid[idx]}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    def _log_eval_step_progress(self, step_idx: int, infos: list[dict], dones: np.ndarray) -> None:
        batch_prices = [info.get("prices_by_agent", {}) for info in infos]
        print(f"[duopoly eval] step={step_idx} batch_prices={batch_prices}")

    def gather_rollout_data(self, total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings) -> DataProto:
        effective_batch = []
        for bs in range(len(total_batch_list)):
            for data in total_batch_list[bs]:
                if data["active_masks"]:
                    data["episode_rewards"] = episode_rewards[bs]
                    data["episode_lengths"] = episode_lengths[bs]
                    data["tool_callings"] = tool_callings[bs]
                    data["pass"] = success["success_rate"][bs]
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

        for step_idx in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            actions_by_agent, multiagent_batch_buffer = self.orchestra.run_turn(
                gen_batch=gen_batch,
                env_obs=obs,
                actor_rollout_wgs=actor_rollout_wg,
                active_masks=active_masks,
                step=step_idx + 1,
            )
            next_obs, rewards, dones, infos = envs.step(actions_by_agent)

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
                    agent_id: actions_by_agent[agent_id][i].raw_text for agent_id in self.config.agent.agent_ids
                }
                step_traces[i].append(
                    {
                        "step": step_idx + 1,
                        "data_source": infos[i].get("data_source"),
                        "prices_by_agent": infos[i].get("prices_by_agent", {}),
                        "p_monopoly": infos[i].get("p_monopoly"),
                        "p_nash": infos[i].get("p_nash"),
                        "failure_reason": infos[i].get("failure_reason"),
                        "invalid_by_agent": infos[i].get("invalid_by_agent", {}),
                        "retry_count_by_agent": infos[i].get("retry_count_by_agent", {}),
                        "raw_text_by_agent": raw_text_by_agent,
                    }
                )

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

            if dump_eval_traces:
                self._log_eval_step_progress(step_idx=step_idx + 1, infos=infos, dones=dones)

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
        if dump_eval_traces:
            self._dump_eval_step_traces(step_traces=step_traces, uid_batch=uid_batch, traj_uid=traj_uid, reset_infos=reset_infos)
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

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

        total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings = self.vanilla_multi_turn_loop(
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
        )
