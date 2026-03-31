import argparse
import os

import pandas as pd


def build_rows(
    split: str,
    seeds: range,
    prompt_prefix_types: list[str],
    alphas: list[float],
    total_units_list: list[float],
    marginal_cost_1a: float,
    marginal_cost_1b: float,
    marginal_cost_2a: float,
    marginal_cost_2b: float,
    market_data_length: int,
):
    rows = []
    index = 0
    for alpha in alphas:
        neg_inverse_beta = -1.0 / 2.0
        for total_units in total_units_list:
            for prompt_prefix_type in prompt_prefix_types:
                for seed in seeds:
                    data_source = (
                        f"cournot_alpha_{str(alpha).replace('.', '_')}_"
                        f"units_{str(total_units).replace('.', '_')}_"
                        f"{prompt_prefix_type.lower()}"
                    )
                    rows.append(
                        {
                            "data_source": data_source,
                            "prompt": [{"role": "system", "content": "You are a helpful and harmless assistant."}],
                            "ability": "competitive_allocation",
                            "reward_model": {"style": "rule"},
                            "extra_info": {"split": split, "index": index},
                            "env_kwargs": {
                                "alpha": alpha,
                                "neg_inverse_beta": neg_inverse_beta,
                                "total_units": total_units,
                                "market_data_length": market_data_length,
                                "prompt_prefix_type": prompt_prefix_type,
                                "marginal_cost_1a": marginal_cost_1a,
                                "marginal_cost_1b": marginal_cost_1b,
                                "marginal_cost_2a": marginal_cost_2a,
                                "marginal_cost_2b": marginal_cost_2b,
                                "periods": 50,
                                "seed": int(seed),
                                "data_source": data_source,
                            },
                        }
                    )
                    index += 1
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/drmas_cournot")
    parser.add_argument("--prompt_prefix_types", nargs="+", default=["P1", "P2"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[100.0])
    parser.add_argument("--total_units_list", nargs="+", type=float, default=[100.0])
    parser.add_argument("--marginal_cost_1a", type=float, default=40.0)
    parser.add_argument("--marginal_cost_1b", type=float, default=50.0)
    parser.add_argument("--marginal_cost_2a", type=float, default=50.0)
    parser.add_argument("--marginal_cost_2b", type=float, default=40.0)
    parser.add_argument("--market_data_length", type=int, default=30)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--train_seed_count", type=int, default=64)
    parser.add_argument("--test_seed_count", type=int, default=16)
    parser.add_argument("--test_sampled_seed_count", type=int, default=4)
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_seeds = range(args.seed_start, args.seed_start + args.train_seed_count)
    test_seeds = range(args.seed_start, args.seed_start + args.test_seed_count)
    test_sampled_seeds = range(args.seed_start, args.seed_start + args.test_sampled_seed_count)

    train_df = pd.DataFrame(
        build_rows(
            "train",
            train_seeds,
            args.prompt_prefix_types,
            args.alphas,
            args.total_units_list,
            args.marginal_cost_1a,
            args.marginal_cost_1b,
            args.marginal_cost_2a,
            args.marginal_cost_2b,
            args.market_data_length,
        )
    )
    test_df = pd.DataFrame(
        build_rows(
            "test",
            test_seeds,
            args.prompt_prefix_types,
            args.alphas,
            args.total_units_list,
            args.marginal_cost_1a,
            args.marginal_cost_1b,
            args.marginal_cost_2a,
            args.marginal_cost_2b,
            args.market_data_length,
        )
    )
    test_sampled_df = pd.DataFrame(
        build_rows(
            "test",
            test_sampled_seeds,
            args.prompt_prefix_types,
            args.alphas,
            args.total_units_list,
            args.marginal_cost_1a,
            args.marginal_cost_1b,
            args.marginal_cost_2a,
            args.marginal_cost_2b,
            args.market_data_length,
        )
    )

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"), index=False)
    test_sampled_df.to_parquet(os.path.join(local_dir, "test_sampled.parquet"), index=False)

    print(f"Saved train.parquet with {len(train_df)} rows")
    print(f"Saved test.parquet with {len(test_df)} rows")
    print(f"Saved test_sampled.parquet with {len(test_sampled_df)} rows")
