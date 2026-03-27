import argparse
import os

import pandas as pd


def build_rows(split: str, seeds: range):
    rows = []
    index = 0
    for alpha in [1.0, 3.2, 10.0]:
        for prompt_prefix_type in ["P1", "P2"]:
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
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_df = pd.DataFrame(build_rows("train", range(64)))
    test_df = pd.DataFrame(build_rows("test", range(16)))
    test_sampled_df = pd.DataFrame(build_rows("test", range(4)))

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"), index=False)
    test_sampled_df.to_parquet(os.path.join(local_dir, "test_sampled.parquet"), index=False)
