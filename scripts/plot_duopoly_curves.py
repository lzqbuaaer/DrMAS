import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data["steps"]
    p_monopoly = data.get("p_monopoly", None)
    p_nash = data.get("p_nash", None)

    first_step = steps[0]
    agent_names = list(first_step["prices_by_agent"].keys())
    if len(agent_names) != 2:
        raise ValueError(f"预期是 2 个 agent，但检测到 {len(agent_names)} 个: {agent_names}")

    agent1, agent2 = agent_names

    rounds = []
    prices_1, prices_2 = [], []
    profits_1, profits_2 = [], []
    retries_1, retries_2 = [], []

    for step in steps:
        rounds.append(step["step"])
        prices_1.append(step["prices_by_agent"].get(agent1, None))
        prices_2.append(step["prices_by_agent"].get(agent2, None))
        profits_1.append(step["profits_by_agent"].get(agent1, None))
        profits_2.append(step["profits_by_agent"].get(agent2, None))
        retries_1.append(step["retry_count_by_agent"].get(agent1, None))
        retries_2.append(step["retry_count_by_agent"].get(agent2, None))

    return {
        "rounds": rounds,
        "agent1": agent1,
        "agent2": agent2,
        "prices_1": prices_1,
        "prices_2": prices_2,
        "profits_1": profits_1,
        "profits_2": profits_2,
        "retries_1": retries_1,
        "retries_2": retries_2,
        "p_monopoly": p_monopoly,
        "p_nash": p_nash,
    }


def plot_price_curve(data, save_path):
    rounds = data["rounds"]
    agent1 = data["agent1"]
    agent2 = data["agent2"]

    plt.figure(figsize=(12, 6))
    plt.plot(rounds, data["prices_1"], marker="o", linestyle="None", markersize=4, label=f"{agent1} price")
    plt.plot(rounds, data["prices_2"], marker="s", linestyle="None", markersize=4, label=f"{agent2} price")

    if data["p_monopoly"] is not None:
        plt.axhline(
            y=data["p_monopoly"],
            linestyle="--",
            linewidth=1.8,
            label=f"p_monopoly = {data['p_monopoly']:.4f}"
        )

    if data["p_nash"] is not None:
        plt.axhline(
            y=data["p_nash"],
            linestyle=":",
            linewidth=1.8,
            label=f"p_nash = {data['p_nash']:.4f}"
        )

    plt.xlabel("Round")
    plt.ylabel("Price")
    plt.title("Price Curves of Agents")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_profit_curve(data, save_path):
    rounds = data["rounds"]
    agent1 = data["agent1"]
    agent2 = data["agent2"]

    plt.figure(figsize=(12, 6))
    plt.plot(rounds, data["profits_1"], marker="o", linestyle="None", markersize=4, label=f"{agent1} profit")
    plt.plot(rounds, data["profits_2"], marker="s", linestyle="None", markersize=4, label=f"{agent2} profit")

    plt.xlabel("Round")
    plt.ylabel("Profit")
    plt.title("Profit Curves of Agents")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_retry_curve(data, save_path):
    rounds = data["rounds"]
    agent1 = data["agent1"]
    agent2 = data["agent2"]

    plt.figure(figsize=(12, 6))
    plt.plot(rounds, data["retries_1"], marker="o", linestyle="None", markersize=4, label=f"{agent1} retry count")
    plt.plot(rounds, data["retries_2"], marker="s", linestyle="None", markersize=4, label=f"{agent2} retry count")

    plt.xlabel("Round")
    plt.ylabel("Retry Count")
    plt.title("Retry Count Curves of Agents")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="输入 JSON 文件路径"
    )
    args = parser.parse_args()

    json_path = Path(args.json_path).resolve()
    out_dir = json_path.parent
    stem = json_path.stem

    data = load_data(json_path)

    price_path = out_dir / f"{stem}_price_curve.png"
    profit_path = out_dir / f"{stem}_profit_curve.png"
    retry_path = out_dir / f"{stem}_retry_curve.png"

    plot_price_curve(data, price_path)
    plot_profit_curve(data, profit_path)
    plot_retry_curve(data, retry_path)

    print("绘图完成，输出文件：")
    print(price_path)
    print(profit_path)
    print(retry_path)


if __name__ == "__main__":
    main()
