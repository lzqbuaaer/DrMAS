from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--message", default="Hello")
    parser.add_argument("--system-message", default="You are a helpful assistant.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-http-retries", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    from competitive_agent_system.api_eval.clients.openai import build_openai_client

    client = build_openai_client(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        reasoning_effort=args.reasoning_effort,
        timeout=args.timeout,
        max_http_retries=args.max_http_retries,
    )
    messages = [
        {"role": "system", "content": args.system_message},
        {"role": "user", "content": args.message},
    ]

    started_at = time.time()
    response_text = client.generate(
        messages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    elapsed = time.time() - started_at

    print(f"model={args.model}")
    print(f"base_url={args.base_url}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print("response:")
    print(response_text)


if __name__ == "__main__":
    main()
