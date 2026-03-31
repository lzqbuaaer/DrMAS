from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from competitive_agent_system.games.base import AgentPrivateState, CompetitiveAction, CompetitiveEpisodeSummary, CompetitiveStepResult


class CournotGameSpec:
    def __init__(self, config, agent_ids: list[str]):
        if len(agent_ids) != 2:
            raise ValueError(f"Cournot expects exactly 2 agents, got {agent_ids}")

        self.config = config
        self.agent_ids = list(agent_ids)

        self.private_states: dict[str, AgentPrivateState] = {}
        self.round_idx = 0
        self.max_periods = 0
        self.market_data_length = 30
        self.prompt_prefix_type = "P1"
        self.data_source = "cournot"
        self.seed = 0
        self.failed = False

        self.alpha = 100.0
        self.neg_inverse_beta = -0.5
        self.total_units = 100.0
        self.flex_total_prod = True

        self.cost_by_agent: dict[str, dict[str, float]] = {}
        self.cumulative_profit_by_agent: dict[str, float] = {}
        self.last_step_profit_by_agent: dict[str, float] = {}
        self.last_step_quantities_by_agent: dict[str, dict[str, float]] = {}
        self.last_market_prices: dict[str, float] = {}
        self.monopoly_quantities: dict[str, dict[str, float]] = {}
        self.nash_quantities: dict[str, dict[str, float]] = {}

    def reset(self, scenario: dict[str, object]) -> None:
        self.alpha = float(scenario.get("alpha", self.config.env.cournot.alpha))
        self.neg_inverse_beta = float(scenario.get("neg_inverse_beta", self.config.env.cournot.neg_inverse_beta))
        self.total_units = float(scenario.get("total_units", self.config.env.cournot.total_units))
        self.market_data_length = int(scenario.get("market_data_length", self.config.env.cournot.market_data_length))
        self.prompt_prefix_type = str(scenario.get("prompt_prefix_type", self.config.env.cournot.prompt_prefix_type))
        self.max_periods = int(scenario.get("periods", self.config.env.max_steps))
        self.data_source = str(scenario.get("data_source", f"cournot_alpha_{str(self.alpha).replace('.', '_')}_{self.prompt_prefix_type.lower()}"))
        self.seed = int(scenario.get("seed", self.config.env.seed))
        self.flex_total_prod = bool(scenario.get("flex_total_prod", self.config.env.cournot.flex_total_prod))

        agent_1, agent_2 = self.agent_ids
        self.cost_by_agent = {
            agent_1: {
                "product_a": float(scenario.get("marginal_cost_1a", self.config.env.cournot.marginal_cost_1a)),
                "product_b": float(scenario.get("marginal_cost_1b", self.config.env.cournot.marginal_cost_1b)),
            },
            agent_2: {
                "product_a": float(scenario.get("marginal_cost_2a", self.config.env.cournot.marginal_cost_2a)),
                "product_b": float(scenario.get("marginal_cost_2b", self.config.env.cournot.marginal_cost_2b)),
            },
        }

        self.private_states = {agent_id: AgentPrivateState() for agent_id in self.agent_ids}
        self.round_idx = 0
        self.failed = False
        self.cumulative_profit_by_agent = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.last_step_profit_by_agent = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.last_step_quantities_by_agent = {
            agent_id: {"product_a": 0.0, "product_b": 0.0} for agent_id in self.agent_ids
        }
        self.last_market_prices = {"product_a": 0.0, "product_b": 0.0}
        self.monopoly_quantities = self.solve_monopolist_quantities()
        self.nash_quantities = self.solve_cournot_nash_equilibrium()

    def build_public_state(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "neg_inverse_beta": self.neg_inverse_beta,
            "total_units": self.total_units,
            "market_data_length": self.market_data_length,
            "data_source": self.data_source,
            "round_idx": self.round_idx,
            "max_periods": self.max_periods,
            "monopoly_quantities": self.monopoly_quantities,
            "nash_quantities": self.nash_quantities,
        }

    def get_price_from_quantity(self, total_quantity: float) -> float:
        return self.alpha + self.neg_inverse_beta * total_quantity

    def compute_price_and_profit(self, my_quantity: float, competitor_quantity: float, marginal_cost: float) -> tuple[float, float]:
        price = self.get_price_from_quantity(my_quantity + competitor_quantity)
        profit = my_quantity * (price - marginal_cost)
        return price, profit

    def _capacity_is_valid(self, quantity_a: float, quantity_b: float) -> bool:
        total_quantity = quantity_a + quantity_b
        if self.flex_total_prod:
            return total_quantity <= self.total_units + 1e-8
        return abs(total_quantity - self.total_units) <= 1e-8

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
            "quantities_by_agent": {
                agent_id: {
                    "product_a": actions[agent_id].payload.get("quantity_a"),
                    "product_b": actions[agent_id].payload.get("quantity_b"),
                }
                for agent_id in self.agent_ids
            },
            "market_prices": {"product_a": None, "product_b": None},
            "monopoly_quantities": self.monopoly_quantities,
            "nash_quantities": self.nash_quantities,
            "invalid_by_agent": {agent_id: not actions[agent_id].valid for agent_id in self.agent_ids},
            "retry_count_by_agent": {agent_id: actions[agent_id].retry_count for agent_id in self.agent_ids},
            "tool_calling": 0.0,
        }
        return CompetitiveStepResult(reward=0.0, rewards_by_agent=rewards_by_agent, done=True, won=False, info=info)

    def step(self, actions: dict[str, CompetitiveAction]) -> CompetitiveStepResult:
        invalid_agents = [agent_id for agent_id, action in actions.items() if not action.valid]
        if invalid_agents:
            return self._mark_failure(actions, f"invalid outputs from {invalid_agents}")

        agent_1, agent_2 = self.agent_ids
        q1a = float(actions[agent_1].payload["quantity_a"])
        q1b = float(actions[agent_1].payload["quantity_b"])
        q2a = float(actions[agent_2].payload["quantity_a"])
        q2b = float(actions[agent_2].payload["quantity_b"])

        if not self._capacity_is_valid(q1a, q1b):
            actions[agent_1].valid = False
            return self._mark_failure(actions, f"capacity exceeded by {agent_1}")
        if not self._capacity_is_valid(q2a, q2b):
            actions[agent_2].valid = False
            return self._mark_failure(actions, f"capacity exceeded by {agent_2}")

        price_a_1, profit_a_1 = self.compute_price_and_profit(q1a, q2a, self.cost_by_agent[agent_1]["product_a"])
        price_b_1, profit_b_1 = self.compute_price_and_profit(q1b, q2b, self.cost_by_agent[agent_1]["product_b"])
        price_a_2, profit_a_2 = self.compute_price_and_profit(q2a, q1a, self.cost_by_agent[agent_2]["product_a"])
        price_b_2, profit_b_2 = self.compute_price_and_profit(q2b, q1b, self.cost_by_agent[agent_2]["product_b"])

        pi1 = profit_a_1 + profit_b_1
        pi2 = profit_a_2 + profit_b_2

        self.round_idx += 1
        self.last_market_prices = {"product_a": price_a_1, "product_b": price_b_1}
        self.last_step_quantities_by_agent = {
            agent_1: {"product_a": q1a, "product_b": q1b},
            agent_2: {"product_a": q2a, "product_b": q2b},
        }
        self.last_step_profit_by_agent = {agent_1: pi1, agent_2: pi2}
        self.cumulative_profit_by_agent[agent_1] += pi1
        self.cumulative_profit_by_agent[agent_2] += pi2

        self.private_states[agent_1].plans_text = actions[agent_1].plans_text
        self.private_states[agent_1].insights_text = actions[agent_1].insights_text
        self.private_states[agent_2].plans_text = actions[agent_2].plans_text
        self.private_states[agent_2].insights_text = actions[agent_2].insights_text

        self.private_states[agent_1].history.append(
            {
                "round": self.round_idx,
                "my_quantity_a": q1a,
                "my_quantity_b": q1b,
                "competitor_quantity_a": q2a,
                "competitor_quantity_b": q2b,
                "market_price_a": price_a_1,
                "market_price_b": price_b_1,
                "my_profit": pi1,
            }
        )
        self.private_states[agent_2].history.append(
            {
                "round": self.round_idx,
                "my_quantity_a": q2a,
                "my_quantity_b": q2b,
                "competitor_quantity_a": q1a,
                "competitor_quantity_b": q1b,
                "market_price_a": price_a_2,
                "market_price_b": price_b_2,
                "my_profit": pi2,
            }
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
            "quantities_by_agent": {
                agent_1: {"product_a": q1a, "product_b": q1b},
                agent_2: {"product_a": q2a, "product_b": q2b},
            },
            "market_prices": {"product_a": price_a_1, "product_b": price_b_1},
            "monopoly_quantities": self.monopoly_quantities,
            "nash_quantities": self.nash_quantities,
            "invalid_by_agent": {agent_id: False for agent_id in self.agent_ids},
            "retry_count_by_agent": {agent_id: actions[agent_id].retry_count for agent_id in self.agent_ids},
            "tool_calling": 0.0,
        }
        reward = float(np.mean([pi1, pi2]))
        return CompetitiveStepResult(reward=reward, rewards_by_agent=rewards_by_agent, done=done, won=won, info=info)

    def _flatten_quantities(self, quantities: dict[str, dict[str, float]]) -> np.ndarray:
        agent_1, agent_2 = self.agent_ids
        return np.array(
            [
                quantities[agent_1]["product_a"],
                quantities[agent_1]["product_b"],
                quantities[agent_2]["product_a"],
                quantities[agent_2]["product_b"],
            ],
            dtype=np.float64,
        )

    def solve_monopolist_quantities(self) -> dict[str, dict[str, float]]:
        agent_1, agent_2 = self.agent_ids
        c1 = self.cost_by_agent[agent_1]
        c2 = self.cost_by_agent[agent_2]
        kappa = self.total_units

        def total_profit(q: np.ndarray) -> float:
            q1a, q1b, q2a, q2b = q
            price_a = self.get_price_from_quantity(q1a + q2a)
            price_b = self.get_price_from_quantity(q1b + q2b)
            return (
                (price_a - c1["product_a"]) * q1a
                + (price_b - c1["product_b"]) * q1b
                + (price_a - c2["product_a"]) * q2a
                + (price_b - c2["product_b"]) * q2b
            )

        if self.flex_total_prod:
            constraints = (
                {"type": "ineq", "fun": lambda q: kappa - q[0] - q[1]},
                {"type": "ineq", "fun": lambda q: kappa - q[2] - q[3]},
            )
        else:
            constraints = (
                {"type": "eq", "fun": lambda q: kappa - q[0] - q[1]},
                {"type": "eq", "fun": lambda q: kappa - q[2] - q[3]},
            )

        bounds = ((0.0, kappa), (0.0, kappa), (0.0, kappa), (0.0, kappa))
        initial_guess = np.array([kappa / 2.0, kappa / 2.0, kappa / 2.0, kappa / 2.0], dtype=np.float64)
        result = minimize(lambda q: -total_profit(q), initial_guess, method="SLSQP", bounds=bounds, constraints=constraints)
        if not result.success:
            result_x = initial_guess
        else:
            result_x = result.x

        return {
            agent_1: {"product_a": float(result_x[0]), "product_b": float(result_x[1])},
            agent_2: {"product_a": float(result_x[2]), "product_b": float(result_x[3])},
        }

    def _best_response(self, own_costs: dict[str, float], other_quantities: np.ndarray) -> np.ndarray:
        kappa = self.total_units

        def own_profit(q: np.ndarray) -> float:
            qa, qb = q
            price_a = self.get_price_from_quantity(qa + other_quantities[0])
            price_b = self.get_price_from_quantity(qb + other_quantities[1])
            return (price_a - own_costs["product_a"]) * qa + (price_b - own_costs["product_b"]) * qb

        constraint = {"type": "ineq" if self.flex_total_prod else "eq", "fun": lambda q: kappa - q[0] - q[1]}
        bounds = ((0.0, kappa), (0.0, kappa))
        initial_guess = np.array([kappa / 2.0, kappa / 2.0], dtype=np.float64)
        result = minimize(lambda q: -own_profit(q), initial_guess, method="SLSQP", bounds=bounds, constraints=constraint)
        if not result.success:
            return initial_guess
        return result.x

    def solve_cournot_nash_equilibrium(self, max_iterations: int = 100, tolerance: float = 1e-8) -> dict[str, dict[str, float]]:
        agent_1, agent_2 = self.agent_ids
        q1 = np.array([self.total_units / 2.0, self.total_units / 2.0], dtype=np.float64)
        q2 = np.array([self.total_units / 2.0, self.total_units / 2.0], dtype=np.float64)

        for _ in range(max_iterations):
            new_q1 = self._best_response(self.cost_by_agent[agent_1], q2)
            new_q2 = self._best_response(self.cost_by_agent[agent_2], new_q1)
            if np.all(np.abs(new_q1 - q1) < tolerance) and np.all(np.abs(new_q2 - q2) < tolerance):
                q1, q2 = new_q1, new_q2
                break
            q1, q2 = new_q1, new_q2

        return {
            agent_1: {"product_a": float(q1[0]), "product_b": float(q1[1])},
            agent_2: {"product_a": float(q2[0]), "product_b": float(q2[1])},
        }

    def summarize_episode(self) -> CompetitiveEpisodeSummary:
        agent_1, agent_2 = self.agent_ids

        final_quantities = self._flatten_quantities(self.last_step_quantities_by_agent)
        monopoly_quantities = self._flatten_quantities(self.monopoly_quantities)
        nash_quantities = self._flatten_quantities(self.nash_quantities)

        metrics = {
            "cumulative_profit/firm1": self.cumulative_profit_by_agent[agent_1],
            "cumulative_profit/firm2": self.cumulative_profit_by_agent[agent_2],
            "mean_cumulative_profit": float(np.mean(list(self.cumulative_profit_by_agent.values()))),
            "distance_to_monopoly": float(np.linalg.norm(final_quantities - monopoly_quantities)),
            "distance_to_nash": float(np.linalg.norm(final_quantities - nash_quantities)),
            "last_market_price/product_a": self.last_market_prices["product_a"],
            "last_market_price/product_b": self.last_market_prices["product_b"],
        }
        return CompetitiveEpisodeSummary(metrics=metrics, won=not self.failed and self.round_idx == self.max_periods, reason=None if not self.failed else "invalid_output")
