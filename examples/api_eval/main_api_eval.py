from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_client(provider_config: ProviderConfig):
    from competitive_agent_system.api_eval.clients import build_deepseek_client, build_openai_client

    provider = provider_config.provider.lower()
    if provider == "deepseek":
        return build_deepseek_client(
            model=provider_config.model,
            api_key_env=provider_config.api_key_env,
            base_url=provider_config.base_url or "https://api.deepseek.com",
            reasoning_effort=provider_config.reasoning_effort,
            thinking_enabled=provider_config.thinking_enabled,
            timeout=provider_config.timeout,
            max_http_retries=provider_config.max_http_retries,
        )
    if provider == "openai":
        return build_openai_client(
            model=provider_config.model,
            api_key_env=provider_config.api_key_env,
            base_url=provider_config.base_url or "https://api.openai.com/v1",
            reasoning_effort=provider_config.reasoning_effort,
            thinking_enabled=provider_config.thinking_enabled,
            timeout=provider_config.timeout,
            max_http_retries=provider_config.max_http_retries,
        )
    raise ValueError(f"Unsupported provider '{provider_config.provider}'")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["duopoly", "cournot"])
    parser.add_argument("--provider", required=True, choices=["deepseek", "openai"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--episode-count", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output-root", default="eval_data")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--thinking-enabled", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-http-retries", type=int, default=3)
    parser.add_argument("--max-parse-retries", type=int, default=None)
    parser.add_argument("--duopoly-beta", type=float, default=100.0)
    parser.add_argument("--duopoly-prompt-prefix-type", default="P1")
    parser.add_argument("--cournot-market-data-length", type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()

    from competitive_agent_system.api_eval.config import ApiEvalConfig, ProviderConfig, SamplingConfig, build_runtime_config
    from competitive_agent_system.api_eval.dataset import load_env_kwargs_from_parquet
    from competitive_agent_system.api_eval.runner import ApiEvalRunner

    runtime_config = build_runtime_config(
        task=args.task,
        experiment_name=args.experiment_name,
        max_steps=args.max_steps,
        duopoly_beta=args.duopoly_beta,
        duopoly_prompt_prefix_type=args.duopoly_prompt_prefix_type,
        cournot_market_data_length=args.cournot_market_data_length,
    )
    env_kwargs_list = load_env_kwargs_from_parquet(args.data_file, args.episode_count)

    provider_config = ProviderConfig(
        provider=args.provider,
        model=args.model,
        api_key_env=args.api_key_env or ("DEEPSEEK_API_KEY" if args.provider == "deepseek" else "OPENAI_API_KEY"),
        base_url=args.base_url or ("https://api.deepseek.com" if args.provider == "deepseek" else "https://api.openai.com/v1"),
        reasoning_effort=args.reasoning_effort,
        thinking_enabled=args.thinking_enabled,
        timeout=args.timeout,
        max_http_retries=args.max_http_retries,
    )
    eval_config = ApiEvalConfig(
        task=args.task,
        experiment_name=args.experiment_name,
        data_file=args.data_file,
        episode_count=args.episode_count,
        output_root=args.output_root,
        max_retries=args.max_parse_retries,
        sampling=SamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        ),
        provider=provider_config,
    )
    client = build_client(provider_config)
    runner = ApiEvalRunner(
        runtime_config=runtime_config,
        eval_config=eval_config,
        client=client,
        env_kwargs_list=env_kwargs_list,
    )
    output_dir = runner.run()
    print(f"Saved API eval artifacts to {output_dir}")


if __name__ == "__main__":
    main()
