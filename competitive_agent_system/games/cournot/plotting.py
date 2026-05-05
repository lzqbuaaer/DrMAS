from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_tail20pct_quantity_scatter(summary: dict, save_path: str) -> None:
    points = summary.get("tail20pct_quantity_points", [])
    benchmarks = summary.get("benchmarks", {})
    metadata = summary.get("metadata", {})

    firm1_a = [point["firm1_product_a"] for point in points]
    firm2_a = [point["firm2_product_a"] for point in points]
    firm1_b = [point["firm1_product_b"] for point in points]
    firm2_b = [point["firm2_product_b"] for point in points]

    monopoly = benchmarks.get("monopoly_quantities") or {}
    nash = benchmarks.get("nash_quantities") or {}
    agent_keys = list(monopoly.keys()) or list(nash.keys())
    agent_1 = agent_keys[0] if len(agent_keys) >= 1 else "Firm 1 Agent"
    agent_2 = agent_keys[1] if len(agent_keys) >= 2 else "Firm 2 Agent"

    monopoly_a = (
        monopoly.get(agent_1, {}).get("product_a"),
        monopoly.get(agent_2, {}).get("product_a"),
    )
    monopoly_b = (
        monopoly.get(agent_1, {}).get("product_b"),
        monopoly.get(agent_2, {}).get("product_b"),
    )
    nash_a = (
        nash.get(agent_1, {}).get("product_a"),
        nash.get(agent_2, {}).get("product_a"),
    )
    nash_b = (
        nash.get(agent_1, {}).get("product_b"),
        nash.get(agent_2, {}).get("product_b"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    panel_specs = [
        (axes[0], firm1_a, firm2_a, monopoly_a, nash_a, "Product A"),
        (axes[1], firm1_b, firm2_b, monopoly_b, nash_b, "Product B"),
    ]

    for ax, xs, ys, monopoly_point, nash_point, title in panel_specs:
        if xs and ys:
            ax.scatter(xs, ys, s=18, alpha=0.6, edgecolors="none", label="Tail 20% valid points")
        else:
            ax.text(0.5, 0.5, "No valid tail points", ha="center", va="center", transform=ax.transAxes)

        candidates = list(xs) + list(ys)
        for point in (monopoly_point, nash_point):
            if point[0] is not None:
                candidates.append(float(point[0]))
            if point[1] is not None:
                candidates.append(float(point[1]))

        if candidates:
            axis_min = min(candidates)
            axis_max = max(candidates)
            span = max(axis_max - axis_min, 1e-6)
            padding = 0.08 * span
            lower = axis_min - padding
            upper = axis_max + padding
        else:
            lower, upper = 0.0, 1.0

        ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.1, color="gray", label="x = y")

        if nash_point[0] is not None and nash_point[1] is not None:
            ax.scatter(
                [float(nash_point[0])],
                [float(nash_point[1])],
                marker="X",
                s=90,
                color="#1f77b4",
                label="Nash",
                zorder=5,
            )

        if monopoly_point[0] is not None and monopoly_point[1] is not None:
            ax.scatter(
                [float(monopoly_point[0])],
                [float(monopoly_point[1])],
                marker="*",
                s=140,
                color="#d62728",
                label="Monopoly",
                zorder=6,
            )

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_xlabel("Firm 1 Quantity")
        ax.set_ylabel("Firm 2 Quantity")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle(
        "Cournot Tail 20% Quantity Scatter\n"
        f"{metadata.get('data_source', 'unknown')} | valid episodes={metadata.get('episode_count_valid', 0)}"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
