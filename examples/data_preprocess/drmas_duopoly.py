import argparse
import os

import pandas as pd


def build_rows(split: str, seeds: range, alphas: list[float], prompt_prefix_types: list[str]):
    rows = []
    index = 0
    for alpha in alphas:
        for prompt_prefix_type in prompt_prefix_types:
            for seed in seeds:
                data_source = f"duopoly_alpha_{str(alpha).replace('.', '_')}_{prompt_prefix_type.lower()}"
                rows.append(
                    {
                        "data_source": data_source,
                        "prompt": [{"role": "system", "content": "You are a helpful and harmless assistant."}],
                        "ability": "competitive_pricing",
                        "reward_model": {"style": "rule"},
                        "extra_info": {
                            "split": split,
                            "index": index,
                        },
                        "env_kwargs": {
                            "alpha": alpha,
                            "prompt_prefix_type": prompt_prefix_type,
                            "periods": 300,
                            "history_window": 100,
                            "seed": int(seed),
                            "data_source": data_source,
                        },
                    }
                )
                index += 1
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/drmas_duopoly")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[1.0, 3.2, 10.0],
        help="Alpha values to include, e.g. --alphas 1 3.2 10",
    )
    parser.add_argument(
        "--prompt_prefix_types",
        nargs="+",
        default=["P1", "P2"],
        help="Prompt prefix types to include, e.g. --prompt_prefix_types P1 P2",
    )
    parser.add_argument("--seed_start", type=int, default=0, help="Starting seed value for all splits")
    parser.add_argument("--train_seed_count", type=int, default=64, help="Number of seeds to generate for train.parquet")
    parser.add_argument("--test_seed_count", type=int, default=16, help="Number of seeds to generate for test.parquet")
    parser.add_argument("--test_sampled_seed_count", type=int, default=4, help="Number of seeds to generate for test_sampled.parquet")
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_seeds = range(args.seed_start, args.seed_start + args.train_seed_count)
    test_seeds = range(args.seed_start, args.seed_start + args.test_seed_count)
    test_sampled_seeds = range(args.seed_start, args.seed_start + args.test_sampled_seed_count)

    train_df = pd.DataFrame(build_rows("train", train_seeds, args.alphas, args.prompt_prefix_types))
    test_df = pd.DataFrame(build_rows("test", test_seeds, args.alphas, args.prompt_prefix_types))
    test_sampled_df = pd.DataFrame(build_rows("test", test_sampled_seeds, args.alphas, args.prompt_prefix_types))

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"), index=False)
    test_sampled_df.to_parquet(os.path.join(local_dir, "test_sampled.parquet"), index=False)

    print(f"Saved train.parquet with {len(train_df)} rows")
    print(f"Saved test.parquet with {len(test_df)} rows")
    print(f"Saved test_sampled.parquet with {len(test_sampled_df)} rows")
    print(f"alphas={args.alphas}")
    print(f"prompt_prefix_types={args.prompt_prefix_types}")
    print(
        "seed_counts="
        f"train:{args.train_seed_count}, "
        f"test:{args.test_seed_count}, "
        f"test_sampled:{args.test_sampled_seed_count}"
    )
