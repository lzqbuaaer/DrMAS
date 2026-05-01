from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_tail20pct_price_scatter(summary: dict, save_path: str) -> None:
    points = summary.get("tail20pct_price_points", [])
    benchmarks = summary.get("benchmarks", {})
    p_monopoly = benchmarks.get("p_monopoly")
    p_nash = benchmarks.get("p_nash")
    metadata = summary.get("metadata", {})

    xs = [point["firm1"] for point in points]
    ys = [point["firm2"] for point in points]

    plt.figure(figsize=(7.5, 7.5))

    if xs and ys:
        plt.scatter(xs, ys, s=20, alpha=0.6, edgecolors="none", label="Tail 20% valid points")
    else:
        plt.text(0.5, 0.5, "No valid tail points", ha="center", va="center", transform=plt.gca().transAxes)

    candidates = list(xs) + list(ys)
    if p_monopoly is not None:
        candidates.append(float(p_monopoly))
    if p_nash is not None:
        candidates.append(float(p_nash))

    if candidates:
        axis_min = min(candidates)
        axis_max = max(candidates)
        span = max(axis_max - axis_min, 1e-6)
        padding = 0.08 * span
        lower = axis_min - padding
        upper = axis_max + padding
    else:
        lower, upper = 0.0, 1.0

    line_x = [lower, upper]
    plt.plot(line_x, line_x, linestyle="--", linewidth=1.2, color="gray", label="x = y")

    if p_nash is not None:
        plt.axvline(float(p_nash), linestyle=":", linewidth=1.2, color="#1f77b4", label=f"pNash = {float(p_nash):.4f}")
        plt.axhline(float(p_nash), linestyle=":", linewidth=1.2, color="#1f77b4")

    if p_monopoly is not None:
        plt.axvline(float(p_monopoly), linestyle="-.", linewidth=1.2, color="#d62728", label=f"pM = {float(p_monopoly):.4f}")
        plt.axhline(float(p_monopoly), linestyle="-.", linewidth=1.2, color="#d62728")

    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.xlabel("Firm 1 Price")
    plt.ylabel("Firm 2 Price")
    plt.title(
        "Duopoly Tail 20% Price Scatter\n"
        f"{metadata.get('data_source', 'unknown')} | valid episodes={metadata.get('episode_count_valid', 0)}"
    )
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()
