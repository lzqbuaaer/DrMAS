from __future__ import annotations

import uuid

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

    def vanilla_multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg, envs, effective_rollout_n: int):
        batch_size = len(gen_batch.batch)
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop("env_kwargs", None))
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
        )

        return self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=total_tool_callings,
        )
