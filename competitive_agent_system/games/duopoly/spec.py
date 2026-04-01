from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from competitive_agent_system.games.base import AgentPrivateState, CompetitiveAction, CompetitiveEpisodeSummary, CompetitiveStepResult


def _maximize_scalar(func, low: float, high: float, rounds: int = 3, num_points: int = 2000) -> float:
    for _ in range(rounds):
        grid = np.linspace(low, high, num_points)
        values = np.array([func(x) for x in grid], dtype=np.float64)
        best_idx = int(values.argmax())
        if best_idx == 0:
            left, right = grid[0], grid[1]
        elif best_idx == len(grid) - 1:
            left, right = grid[-2], grid[-1]
        else:
            left, right = grid[best_idx - 1], grid[best_idx + 1]
        low, high = left, right
    return float((low + high) / 2.0)


@lru_cache(maxsize=None)
def solve_symmetric_p_monopoly(alpha: float, a: float, a0: float, mu: float, beta: float, c: float) -> float:
    cost = alpha * c

    def total_profit(price: float) -> float:
        util = math.exp((a - price / alpha) / mu)
        denom = 2.0 * util + math.exp(a0 / mu)
        q = beta * util / denom
        return 2.0 * (price - cost) * q

    return _maximize_scalar(total_profit, low=max(0.0, cost), high=max(10.0 * alpha, cost + 1.0))


@lru_cache(maxsize=None)
def solve_symmetric_p_nash(alpha: float, a: float, a0: float, mu: float, beta: float, c: float) -> float:
    cost = alpha * c

    def best_response(opponent_price: float) -> float:
        def own_profit(price: float) -> float:
            util_i = math.exp((a - price / alpha) / mu)
            util_j = math.exp((a - opponent_price / alpha) / mu)
            denom = util_i + util_j + math.exp(a0 / mu)
            q = beta * util_i / denom
            return (price - cost) * q

        return _maximize_scalar(own_profit, low=max(0.0, cost), high=max(10.0 * alpha, cost + 1.0))

    price = max(cost + alpha, cost + 1.0)
    for _ in range(25):
        next_price = best_response(price)
        if abs(next_price - price) < 1e-4:
            break
        price = next_price
    return float(price)


class DuopolyGameSpec:
    def __init__(self, config, agent_ids: list[str]):
        if len(agent_ids) != 2:
            raise ValueError(f"Duopoly expects exactly 2 agents, got {agent_ids}")

        self.config = config
        self.agent_ids = agent_ids
        self.a1 = 2.0
        self.a2 = 2.0
        self.a0 = 0.0
        self.mu = 0.25
        self.beta = float(config.env.duopoly.beta)
        self.c1 = 1.0
        self.c2 = 1.0

        self.private_states: dict[str, AgentPrivateState] = {}
        self.round_idx = 0
        self.max_periods = 0
        self.history_window = 100
        self.alpha = 1.0
        self.prompt_prefix_type = "P1"
        self.data_source = "duopoly"
        self.seed = 0
        self.ceiling = 0.0
        self.p_nash = 0.0
        self.p_monopoly = 0.0
        self.failed = False

        self.cumulative_profit_by_agent: dict[str, float] = {}
        self.last_step_profit_by_agent: dict[str, float] = {}
        self.last_step_quantity_by_agent: dict[str, float] = {}
        self.last_step_price_by_agent: dict[str, float] = {}
        self.price_history_by_agent: dict[str, list[float]] = {}
        self.consumer_surplus_history: list[float] = []

    def reset(self, scenario: dict[str, object]) -> None:
        self.alpha = float(scenario.get("alpha", self.config.env.duopoly.alpha))
        self.beta = float(scenario.get("beta", self.config.env.duopoly.beta))
        self.prompt_prefix_type = str(scenario.get("prompt_prefix_type", self.config.env.duopoly.prompt_prefix_type))
        self.max_periods = int(scenario.get("periods", self.config.env.max_steps))
        self.history_window = int(scenario.get("history_window", self.config.env.duopoly.history_window))
        self.data_source = str(scenario.get("data_source", f"duopoly_alpha_{self.alpha}_{self.prompt_prefix_type.lower()}"))
        self.seed = int(scenario.get("seed", self.config.env.seed))
        rng = np.random.RandomState(self.seed)

        self.private_states = {agent_id: AgentPrivateState() for agent_id in self.agent_ids}
        self.round_idx = 0
        self.failed = False
        self.cumulative_profit_by_agent = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.last_step_profit_by_agent = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.last_step_quantity_by_agent = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.last_step_price_by_agent = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.price_history_by_agent = {agent_id: [] for agent_id in self.agent_ids}
        self.consumer_surplus_history = []

        self.p_monopoly = solve_symmetric_p_monopoly(self.alpha, self.a1, self.a0, self.mu, self.beta, self.c1)
        self.p_nash = solve_symmetric_p_nash(self.alpha, self.a1, self.a0, self.mu, self.beta, self.c1)
        self.ceiling = float(rng.uniform(1.5, 2.5) * self.p_monopoly)

    def build_public_state(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "data_source": self.data_source,
            "history_window": self.history_window,
            "p_monopoly": self.p_monopoly,
            "p_nash": self.p_nash,
            "ceiling": self.ceiling,
            "round_idx": self.round_idx,
            "max_periods": self.max_periods,
        }

    def compute_demands(self, p1: float, p2: float) -> tuple[float, float]:
        util1 = math.exp((self.a1 - p1 / self.alpha) / self.mu)
        util2 = math.exp((self.a2 - p2 / self.alpha) / self.mu)
        outside = math.exp(self.a0 / self.mu)
        denom = util1 + util2 + outside
        return self.beta * util1 / denom, self.beta * util2 / denom

    def compute_profits(self, p1: float, p2: float, q1: float, q2: float) -> tuple[float, float]:
        pi1 = (p1 - self.alpha * self.c1) * q1
        pi2 = (p2 - self.alpha * self.c2) * q2
        return pi1, pi2

    def compute_consumer_surplus(self, p1: float, p2: float) -> float:
        util1 = math.exp((self.a1 - p1 / self.alpha) / self.mu)
        util2 = math.exp((self.a2 - p2 / self.alpha) / self.mu)
        outside = math.exp(self.a0 / self.mu)
        inclusive_value = math.log(util1 + util2 + outside) - math.log(outside)
        return float(self.beta * self.mu * inclusive_value)

    def _mark_failure(self, actions: dict[str, CompetitiveAction], reason: str) -> CompetitiveStepResult:
        self.failed = True
        for agent_id in self.agent_ids:
            self.private_states[agent_id].failed = True

        rewards_by_agent = {agent_id: 0.0 for agent_id in self.agent_ids}
        info = {
            "data_source": self.data_source,
            "won": False,
            "failure_reason": reason,
            "rewards_by_agent": rewards_by_agent,
            "profits_by_agent": rewards_by_agent,
            "quantities_by_agent": {agent_id: 0.0 for agent_id in self.agent_ids},
            "prices_by_agent": {agent_id: actions[agent_id].payload.get("price") for agent_id in self.agent_ids},
            "p_monopoly": self.p_monopoly,
            "p_nash": self.p_nash,
            "invalid_by_agent": {agent_id: not actions[agent_id].valid for agent_id in self.agent_ids},
            "retry_count_by_agent": {agent_id: actions[agent_id].retry_count for agent_id in self.agent_ids},
            "tool_calling": 0.0,
        }
        return CompetitiveStepResult(reward=0.0, rewards_by_agent=rewards_by_agent, done=True, won=False, info=info)

    def _tail_window_size(self) -> int:
        if self.round_idx <= 0:
            return 0
        return max(1, int(math.ceil(self.round_idx * 0.2)))

    def _mean_tail_alignment(self, agent_id: str, target_price: float) -> float:
        prices = self.price_history_by_agent.get(agent_id, [])
        if not prices:
            return 0.0
        tail_window = self._tail_window_size()
        tail_prices = prices[-tail_window:]
        scores = [1.0 / (1.0 + abs(price - target_price) / self.alpha) for price in tail_prices]
        return float(np.mean(scores))

    def _mean_tail_consumer_surplus(self) -> float:
        if not self.consumer_surplus_history:
            return 0.0
        tail_window = self._tail_window_size()
        return float(np.mean(self.consumer_surplus_history[-tail_window:]))

    def step(self, actions: dict[str, CompetitiveAction]) -> CompetitiveStepResult:
        invalid_agents = [agent_id for agent_id, action in actions.items() if not action.valid]
        if invalid_agents:
            return self._mark_failure(actions, f"invalid outputs from {invalid_agents}")

        agent_1, agent_2 = self.agent_ids
        p1 = float(actions[agent_1].payload["price"])
        p2 = float(actions[agent_2].payload["price"])

        q1, q2 = self.compute_demands(p1, p2)
        pi1, pi2 = self.compute_profits(p1, p2, q1, q2)

        self.round_idx += 1

        self.last_step_price_by_agent = {agent_1: p1, agent_2: p2}
        self.last_step_quantity_by_agent = {agent_1: q1, agent_2: q2}
        self.last_step_profit_by_agent = {agent_1: pi1, agent_2: pi2}
        self.price_history_by_agent[agent_1].append(p1)
        self.price_history_by_agent[agent_2].append(p2)
        self.consumer_surplus_history.append(self.compute_consumer_surplus(p1, p2))
        self.cumulative_profit_by_agent[agent_1] += pi1
        self.cumulative_profit_by_agent[agent_2] += pi2

        self.private_states[agent_1].plans_text = actions[agent_1].plans_text
        self.private_states[agent_1].insights_text = actions[agent_1].insights_text
        self.private_states[agent_2].plans_text = actions[agent_2].plans_text
        self.private_states[agent_2].insights_text = actions[agent_2].insights_text

        self.private_states[agent_1].history.append(
            {"round": self.round_idx, "my_price": p1, "competitor_price": p2, "my_quantity": q1, "my_profit": pi1}
        )
        self.private_states[agent_2].history.append(
            {"round": self.round_idx, "my_price": p2, "competitor_price": p1, "my_quantity": q2, "my_profit": pi2}
        )

        done = self.round_idx >= self.max_periods
        won = done and not self.failed
        rewards_by_agent = {agent_1: pi1, agent_2: pi2}
        info = {
            "data_source": self.data_source,
            "won": won,
            "failure_reason": None,
            "rewards_by_agent": rewards_by_agent,
            "profits_by_agent": rewards_by_agent,
            "quantities_by_agent": {agent_1: q1, agent_2: q2},
            "prices_by_agent": {agent_1: p1, agent_2: p2},
            "p_monopoly": self.p_monopoly,
            "p_nash": self.p_nash,
            "invalid_by_agent": {agent_id: False for agent_id in self.agent_ids},
            "retry_count_by_agent": {agent_id: actions[agent_id].retry_count for agent_id in self.agent_ids},
            "tool_calling": 0.0,
        }
        reward = float(np.mean([pi1, pi2]))
        return CompetitiveStepResult(reward=reward, rewards_by_agent=rewards_by_agent, done=done, won=won, info=info)

    def summarize_episode(self) -> CompetitiveEpisodeSummary:
        agent_1, agent_2 = self.agent_ids
        cooperation_firm1 = self._mean_tail_alignment(agent_1, self.p_nash)
        cooperation_firm2 = self._mean_tail_alignment(agent_2, self.p_nash)
        collusion_firm1 = self._mean_tail_alignment(agent_1, self.p_monopoly)
        collusion_firm2 = self._mean_tail_alignment(agent_2, self.p_monopoly)
        metrics = {
            "cumulative_profit/firm1": self.cumulative_profit_by_agent[agent_1],
            "cumulative_profit/firm2": self.cumulative_profit_by_agent[agent_2],
            "mean_cumulative_profit": float(np.mean(list(self.cumulative_profit_by_agent.values()))),
            "last_price/firm1": self.last_step_price_by_agent[agent_1],
            "last_price/firm2": self.last_step_price_by_agent[agent_2],
            "p_monopoly": self.p_monopoly,
            "p_nash": self.p_nash,
            "tail20pct_window_size": float(self._tail_window_size()),
            "cooperation_last20pct/firm1": cooperation_firm1,
            "cooperation_last20pct/firm2": cooperation_firm2,
            "collusion_last20pct/firm1": collusion_firm1,
            "collusion_last20pct/firm2": collusion_firm2,
            "consumer_surplus_last20pct": self._mean_tail_consumer_surplus(),
        }
        return CompetitiveEpisodeSummary(metrics=metrics, won=not self.failed and self.round_idx == self.max_periods, reason=None if not self.failed else "invalid_output")
