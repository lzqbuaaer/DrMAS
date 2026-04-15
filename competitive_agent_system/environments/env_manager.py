from __future__ import annotations

import asyncio
import concurrent.futures
from collections import defaultdict
from typing import Any

import numpy as np

from competitive_agent_system.games.base import CompetitiveAction
from competitive_agent_system.games.duopoly import DuopolyGameSpec, DuopolyMetricComputer, DuopolyObservationBuilder


class DuopolyArenaEnv:
    def __init__(self, config, agent_ids: list[str]):
        self.config = config
        self.agent_ids = list(agent_ids)
        self.observer = DuopolyObservationBuilder(self.agent_ids)
        self.metric_computer = DuopolyMetricComputer()
        self.game = DuopolyGameSpec(config, self.agent_ids)
        self._finished = False
        self._terminal_info = None

    def _build_observations(self) -> dict[str, str]:
        if self._finished:
            return {agent_id: "" for agent_id in self.agent_ids}
        return {agent_id: self.observer.build_observation(agent_id, self.game) for agent_id in self.agent_ids}

    def reset(self, extras: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        extras = dict(extras)
        extras["periods"] = int(self.config.env.max_steps)
        self.game.reset(extras)
        self._finished = False
        self._terminal_info = {
            "data_source": self.game.data_source,
            "p_monopoly": self.game.p_monopoly,
            "p_nash": self.game.p_nash,
            "ceiling": self.game.ceiling,
            "won": False,
            "tool_calling": 0.0,
        }
        return self._build_observations(), {
            "data_source": self.game.data_source,
            "p_monopoly": self.game.p_monopoly,
            "p_nash": self.game.p_nash,
            "ceiling": self.game.ceiling,
        }

    def step(self, actions_by_agent: dict[str, Any]):
        if self._finished:
            return self._build_observations(), 0.0, True, self._terminal_info

        step_result = self.game.step(actions_by_agent)
        info = dict(step_result.info)
        if step_result.done:
            self._finished = True
            summary = self.metric_computer.finalize(self.game)
            info.update(summary.metrics)
            info["won"] = summary.won
            self._terminal_info = info

        return self._build_observations(), step_result.reward, step_result.done, info

    def close(self) -> None:
        pass


class CournotArenaEnv:
    def __init__(self, config, agent_ids: list[str]):
        from competitive_agent_system.games.cournot.metrics import CournotMetricComputer
        from competitive_agent_system.games.cournot.observer import CournotObservationBuilder
        from competitive_agent_system.games.cournot.spec import CournotGameSpec

        self.config = config
        self.agent_ids = list(agent_ids)
        self.observer = CournotObservationBuilder(self.agent_ids)
        self.metric_computer = CournotMetricComputer()
        self.game = CournotGameSpec(config, self.agent_ids)
        self._finished = False
        self._terminal_info = None

    def _build_observations(self) -> dict[str, str]:
        if self._finished:
            return {agent_id: "" for agent_id in self.agent_ids}
        return {agent_id: self.observer.build_observation(agent_id, self.game) for agent_id in self.agent_ids}

    def reset(self, extras: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        extras = dict(extras)
        extras["periods"] = int(self.config.env.max_steps)
        self.game.reset(extras)
        self._finished = False
        self._terminal_info = {
            "data_source": self.game.data_source,
            "monopoly_quantities": self.game.monopoly_quantities,
            "nash_quantities": self.game.nash_quantities,
            "total_units": self.game.total_units,
            "won": False,
            "tool_calling": 0.0,
        }
        return self._build_observations(), {
            "data_source": self.game.data_source,
            "monopoly_quantities": self.game.monopoly_quantities,
            "nash_quantities": self.game.nash_quantities,
            "total_units": self.game.total_units,
        }

    def step(self, actions_by_agent: dict[str, Any]):
        if self._finished:
            return self._build_observations(), 0.0, True, self._terminal_info

        step_result = self.game.step(actions_by_agent)
        info = dict(step_result.info)
        if step_result.done:
            self._finished = True
            summary = self.metric_computer.finalize(self.game)
            info.update(summary.metrics)
            info["won"] = summary.won
            self._terminal_info = info

        return self._build_observations(), step_result.reward, step_result.done, info

    def close(self) -> None:
        pass


class DuopolyMultiProcessEnv:
    def __init__(self, config, seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True):
        self.config = config
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train

        self.envs = [DuopolyArenaEnv(config, list(config.agent.agent_ids)) for _ in range(self.batch_size)]
        max_workers = min(self.batch_size, 128)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def _sync_reset(self, env, kwargs):
        return env.reset(kwargs)

    def _sync_step(self, env, action):
        return env.step(action)

    def reset(self, kwargs: list[dict[str, Any]]):
        pad_n = self.batch_size - len(kwargs)
        dummy_kw = {
            "alpha": self.config.env.duopoly.alpha,
            "prompt_prefix_type": self.config.env.duopoly.prompt_prefix_type,
            "periods": self.config.env.max_steps,
            "history_window": self.config.env.duopoly.history_window,
            "seed": 0,
            "data_source": "duopoly_dummy",
        }
        padded_kwargs = list(kwargs) + [dummy_kw] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        tasks = [self._loop.run_in_executor(self._executor, self._sync_reset, env, kw) for env, kw in zip(self.envs, padded_kwargs)]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, info_list = map(list, zip(*results))
        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]
        if not self.is_train and bool(getattr(self.config.trainer, "val_only", False)):
            batch_stats = [
                {
                    "data_source": info.get("data_source"),
                    "p_monopoly": info.get("p_monopoly"),
                    "p_nash": info.get("p_nash"),
                    "ceiling": info.get("ceiling"),
                }
                for info in info_list
            ]
            print(f"[duopoly eval] batch_init={batch_stats}")
        return obs_list, info_list

    def step(self, actions: list[dict[str, Any]]):
        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [
            {
                agent_id: CompetitiveAction(
                    raw_text="",
                    action_type="invalid",
                    error="padded_action",
                )
                for agent_id in self.config.agent.agent_ids
            }
            for _ in range(pad_n)
        ]
        valid_mask = [True] * len(actions) + [False] * pad_n

        tasks = [self._loop.run_in_executor(self._executor, self._sync_step, env, action) for env, action in zip(self.envs, padded_actions)]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, reward_list, done_list, info_list = map(list, zip(*results))
        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid_mask) if keep]
        done_list = [d for d, keep in zip(done_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]
        return obs_list, reward_list, done_list, info_list

    def close(self):
        if getattr(self, "_closed", False):
            return
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True

    def __del__(self):
        self.close()


class CournotMultiProcessEnv:
    def __init__(self, config, seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True):
        self.config = config
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train

        self.envs = [CournotArenaEnv(config, list(config.agent.agent_ids)) for _ in range(self.batch_size)]
        max_workers = min(self.batch_size, 128)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def _sync_reset(self, env, kwargs):
        return env.reset(kwargs)

    def _sync_step(self, env, action):
        return env.step(action)

    def reset(self, kwargs: list[dict[str, Any]]):
        pad_n = self.batch_size - len(kwargs)
        dummy_kw = {
            "alpha": self.config.env.cournot.alpha,
            "neg_inverse_beta": self.config.env.cournot.neg_inverse_beta,
            "total_units": self.config.env.cournot.total_units,
            "market_data_length": self.config.env.cournot.market_data_length,
            "prompt_prefix_type": self.config.env.cournot.prompt_prefix_type,
            "seed": 0,
            "data_source": "cournot_dummy",
        }
        padded_kwargs = list(kwargs) + [dummy_kw] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        tasks = [self._loop.run_in_executor(self._executor, self._sync_reset, env, kw) for env, kw in zip(self.envs, padded_kwargs)]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, info_list = map(list, zip(*results))
        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]
        if not self.is_train and bool(getattr(self.config.trainer, "val_only", False)):
            batch_stats = [
                {
                    "data_source": info.get("data_source"),
                    "total_units": info.get("total_units"),
                    "monopoly_quantities": info.get("monopoly_quantities"),
                    "nash_quantities": info.get("nash_quantities"),
                }
                for info in info_list
            ]
            print(f"[cournot eval] batch_init={batch_stats}")
        return obs_list, info_list

    def step(self, actions: list[dict[str, Any]]):
        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [
            {
                agent_id: CompetitiveAction(
                    raw_text="",
                    action_type="invalid",
                    error="padded_action",
                )
                for agent_id in self.config.agent.agent_ids
            }
            for _ in range(pad_n)
        ]
        valid_mask = [True] * len(actions) + [False] * pad_n

        tasks = [self._loop.run_in_executor(self._executor, self._sync_step, env, action) for env, action in zip(self.envs, padded_actions)]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, reward_list, done_list, info_list = map(list, zip(*results))
        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid_mask) if keep]
        done_list = [d for d, keep in zip(done_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]
        return obs_list, reward_list, done_list, info_list

    def close(self):
        if getattr(self, "_closed", False):
            return
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True

    def __del__(self):
        self.close()


class CompetitiveEnvironmentManagerBase:
    def __init__(self, envs, config):
        self.envs = envs
        self.config = config

    def _stack_observations(self, obs_list: list[dict[str, str]]):
        agent_ids = list(self.config.agent.agent_ids)
        return {
            "per_agent_text": {agent_id: [obs[agent_id] for obs in obs_list] for agent_id in agent_ids},
            "image": None,
            "anchor": None,
        }

    def reset(self, kwargs):
        obs, infos = self.envs.reset(kwargs=kwargs)
        return self._stack_observations(obs), infos

    def step(self, actions_by_agent: dict[str, list[Any]]):
        batch_size = len(next(iter(actions_by_agent.values())))
        action_list = []
        for idx in range(batch_size):
            action_list.append({agent_id: actions_by_agent[agent_id][idx] for agent_id in actions_by_agent})
        next_obs, rewards, dones, infos = self.envs.step(action_list)
        return self._stack_observations(next_obs), np.array(rewards), np.array(dones), infos

    def success_evaluator(self, *args, **kwargs) -> dict[str, np.ndarray]:
        total_infos = kwargs["total_infos"]
        total_batch_list = kwargs["total_batch_list"]
        success = defaultdict(list)
        for batch_idx in range(len(total_batch_list)):
            for i in reversed(range(len(total_batch_list[batch_idx]))):
                batch_item = total_batch_list[batch_idx][i]
                if batch_item["active_masks"]:
                    info = total_infos[batch_idx][i]
                    won_value = float(info.get("won", False))
                    success["success_rate"].append(won_value)
                    data_source = info.get("data_source", "unknown")
                    success[f"{data_source}_success_rate"].append(won_value)
                    break

        return {key: np.array(value) for key, value in success.items()}


class DuopolyEnvironmentManager(CompetitiveEnvironmentManagerBase):
    pass


class CournotEnvironmentManager(CompetitiveEnvironmentManagerBase):
    pass


def build_duopoly_envs(config, seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True):
    return DuopolyMultiProcessEnv(config=config, seed=seed, env_num=env_num, group_n=group_n, is_train=is_train)


def build_cournot_envs(config, seed: int = 0, env_num: int = 1, group_n: int = 1, is_train: bool = True):
    return CournotMultiProcessEnv(config=config, seed=seed, env_num=env_num, group_n=group_n, is_train=is_train)


def make_envs(config):
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")

    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    val_group_n = getattr(config.env.rollout, "val_n", 1)

    env_name = config.env.env_name.lower()
    if "duopoly" in env_name:
        _envs = build_duopoly_envs(config=config, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_duopoly_envs(config=config, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=val_group_n, is_train=False)
        envs = DuopolyEnvironmentManager(_envs, config)
        val_envs = DuopolyEnvironmentManager(_val_envs, config)
    elif "cournot" in env_name:
        _envs = build_cournot_envs(config=config, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_cournot_envs(config=config, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=val_group_n, is_train=False)
        envs = CournotEnvironmentManager(_envs, config)
        val_envs = CournotEnvironmentManager(_val_envs, config)
    else:
        raise ValueError(f"Unsupported competitive environment '{config.env.env_name}'")
    return envs, val_envs
