from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

from competitive_agent_system.api_eval.config import sanitize_path_component
from competitive_agent_system.api_eval.dump import create_output_dir, dump_episode_json, dump_run_config
from competitive_agent_system.api_eval.retry import build_retry_feedback
from competitive_agent_system.api_eval.tasks import create_task_adapter


class ApiEvalRunner:
    def __init__(self, *, runtime_config, eval_config, client, env_kwargs_list: list[dict]):
        self.runtime_config = runtime_config
        self.eval_config = eval_config
        self.client = client
        self.env_kwargs_list = list(env_kwargs_list)
        self.agent_ids = list(runtime_config.agent.agent_ids)
        self.task_adapter = create_task_adapter(
            task=eval_config.task,
            config=runtime_config,
            agent_ids=self.agent_ids,
            sanitize_path_component=sanitize_path_component,
        )
        self.executor = ThreadPoolExecutor(max_workers=max(1, len(self.agent_ids))) if self.eval_config.parallel_agents else None

    def _build_messages(self, observation: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": observation},
        ]

    def _generate_with_retry(self, *, agent_id: str, observation: str, parser):
        parse_kwargs = self.task_adapter.build_parse_kwargs(observation=observation, parser=parser)
        max_retries = int(self.eval_config.max_retries or parser.max_retries)
        retry_observation = observation
        last_text = ""
        parsed = parser.parse("", **parse_kwargs)

        for attempt in range(max_retries):
            last_text = self.client.generate(
                self._build_messages(retry_observation),
                temperature=self.eval_config.sampling.temperature,
                top_p=self.eval_config.sampling.top_p,
                max_tokens=self.eval_config.sampling.max_tokens,
            )
            parsed = parser.parse(last_text, **parse_kwargs)
            parsed.retry_count = attempt + 1
            if parsed.valid:
                return last_text, parsed

            if attempt + 1 < max_retries:
                retry_observation = observation + "\n\n" + build_retry_feedback(
                    task=self.eval_config.task,
                    agent_id=agent_id,
                    parsed_action=parsed,
                    parse_kwargs=parse_kwargs,
                )

        return last_text, parsed

    def _run_episode(self, env_kwargs: dict, episode_idx: int) -> dict:
        env = self.task_adapter.build_env()
        parser = self.task_adapter.build_parser()
        uid = str(uuid.uuid4())
        traj_uid = str(uuid.uuid4())
        step_traces: list[dict] = []
        observations, reset_info = env.reset(env_kwargs)
        terminal_info = {}

        try:
            for step_idx in range(int(self.runtime_config.env.max_steps)):
                actions_by_agent = {}
                raw_text_by_agent = {}
                prompt_char_length_by_agent = {}
                if self.eval_config.parallel_agents:
                    future_by_agent = {}
                    for agent_id in self.agent_ids:
                        prompt_char_length_by_agent[agent_id] = len(observations[agent_id])
                        if episode_idx == 0 and step_idx == 0:
                            print(f"[api eval debug] episode=1 step=1 agent={agent_id} prompt_begin")
                            print(observations[agent_id])
                            print(f"[api eval debug] episode=1 step=1 agent={agent_id} prompt_end")
                        future_by_agent[agent_id] = self.executor.submit(
                            self._generate_with_retry,
                            agent_id=agent_id,
                            observation=observations[agent_id],
                            parser=parser,
                        )
                    for agent_id in self.agent_ids:
                        raw_text, parsed = future_by_agent[agent_id].result()
                        actions_by_agent[agent_id] = parsed
                        raw_text_by_agent[agent_id] = raw_text
                else:
                    for agent_id in self.agent_ids:
                        prompt_char_length_by_agent[agent_id] = len(observations[agent_id])
                        if episode_idx == 0 and step_idx == 0:
                            print(f"[api eval debug] episode=1 step=1 agent={agent_id} prompt_begin")
                            print(observations[agent_id])
                            print(f"[api eval debug] episode=1 step=1 agent={agent_id} prompt_end")
                        raw_text, parsed = self._generate_with_retry(
                            agent_id=agent_id,
                            observation=observations[agent_id],
                            parser=parser,
                        )
                        actions_by_agent[agent_id] = parsed
                        raw_text_by_agent[agent_id] = raw_text

                observations, reward, done, info = env.step(actions_by_agent)
                step_trace = self.task_adapter.task_handler.build_step_trace(
                    step_idx=step_idx,
                    info=info,
                    raw_text_by_agent=raw_text_by_agent,
                )
                step_trace["prompt_char_length_by_agent"] = prompt_char_length_by_agent
                step_traces.append(step_trace)
                prices_by_agent = info.get("prices_by_agent")
                print(
                    f"[api eval] step={step_idx + 1} "
                    f"data_source={info.get('data_source')} "
                    f"{'' if prices_by_agent is None else f'prices={prices_by_agent} '}"
                    f"prompt_chars={prompt_char_length_by_agent}"
                )
                terminal_info = dict(info)
                if done:
                    break
        finally:
            env.close()

        return self.task_adapter.build_episode_payload(
            uid=uid,
            traj_uid=traj_uid,
            reset_info=reset_info,
            terminal_info=terminal_info,
            step_traces=step_traces,
        )

    def run(self):
        output_dir = create_output_dir(
            output_root=self.eval_config.output_root,
            experiment_name=self.eval_config.experiment_name,
        )
        dump_run_config(
            output_dir,
            {
                "runtime_config": {
                    "env_name": str(self.runtime_config.env.env_name),
                    "max_steps": int(self.runtime_config.env.max_steps),
                    "agent_ids": list(self.agent_ids),
                },
                "eval_config": self.eval_config.to_dict(),
            },
        )

        for episode_idx, env_kwargs in enumerate(self.env_kwargs_list):
            payload = self._run_episode(env_kwargs=env_kwargs, episode_idx=episode_idx)
            dump_episode_json(output_dir=output_dir, traj_uid=payload["traj_uid"], payload=payload)

        try:
            self.task_adapter.finalize_artifacts(output_dir)
            return output_dir
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
