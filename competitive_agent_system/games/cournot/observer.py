from __future__ import annotations

from competitive_agent_system.games.base import AgentPrivateState


P0 = (
    "Help the user choose product quantities that maximize long-run profit. You will see your past quantities, "
    "market prices, profits, and two private notes files from earlier rounds."
)

P1 = (
    P0
    + "\n\nUse disciplined exploration. Producing more can increase sales, but total market output lowers market "
    "prices, so you must balance quantity against price and margin."
)

P2 = (
    P0
    + "\n\nExplore aggressively when useful for learning. Producing more can increase sales, but total market "
    "output lowers market prices, so you must balance quantity against price and margin."
)


class CournotObservationBuilder:
    def __init__(self, agent_ids: list[str]):
        self.agent_ids = agent_ids

    def build_prefix(self, prompt_prefix_type: str) -> str:
        if prompt_prefix_type == "P1":
            return P1
        if prompt_prefix_type == "P2":
            return P2
        raise ValueError(f"Unsupported prompt_prefix_type: {prompt_prefix_type}")

    def format_market_history(self, private_state: AgentPrivateState, window: int) -> str:
        history = private_state.history[-window:]
        if not history:
            return "(no market history yet)"

        chunks = []
        for row in reversed(history):
            chunks.append(
                "\n".join(
                    [
                        f"Round {row['round']}:",
                        f"- My quantity A: {row['my_quantity_a']:.2f}",
                        f"- My quantity B: {row['my_quantity_b']:.2f}",
                        f"- Competitor quantity A: {row['competitor_quantity_a']:.2f}",
                        f"- Competitor quantity B: {row['competitor_quantity_b']:.2f}",
                        f"- Market price A: {row['market_price_a']:.2f}",
                        f"- Market price B: {row['market_price_b']:.2f}",
                        f"- My profit earned: {row['my_profit']:.2f}",
                    ]
                )
            )
        return "\n".join(chunks)

    def build_observation(self, agent_id: str, game) -> str:
        private_state = game.private_states[agent_id]
        public_state = game.build_public_state()
        prefix = self.build_prefix(game.prompt_prefix_type)
        costs = game.cost_by_agent[agent_id]
        plans_text = private_state.plans_text or "(empty)"
        insights_text = private_state.insights_text or "(empty)"
        history_text = self.format_market_history(private_state, public_state["market_data_length"])

        return (
            f"{prefix}\n\n"
            "Product information:\n"
            f"- The cost to produce each unit of Product A is {costs['product_a']:.2f}.\n"
            f"- The cost to produce each unit of Product B is {costs['product_b']:.2f}.\n"
            f"- Your total output across Product A and Product B must be at most {public_state['total_units']:.2f} units.\n"
            "- Market price for each product is determined by total quantity sold by both firms.\n"
            "- Producing more can raise your sales, but higher total market output lowers market prices.\n"
            "- Choose quantities by balancing sales volume against market price and profit margin.\n\n"
            "Private files from earlier rounds:\n\n"
            "Filename: PLANS.txt\n"
            "+++++++++++++++++++++\n"
            f"{plans_text}\n"
            "+++++++++++++++++++++\n\n"
            "Filename: INSIGHTS.txt\n"
            "+++++++++++++++++++++\n"
            f"{insights_text}\n"
            "+++++++++++++++++++++\n\n"
            "Market data you can observe:\n\n"
            "Filename: MARKET DATA (read-only)\n"
            "+++++++++++++++++++++\n"
            f"{history_text}\n"
            "+++++++++++++++++++++\n\n"
            "Respond using exactly these five XML-style tags and nothing outside them:\n"
            "<OBSERVATIONS>, <PLANS>, <INSIGHTS>, <QUANTITY_A>, <QUANTITY_B>.\n"
            "Do not use markdown headings.\n"
            "Inside <QUANTITY_A> and <QUANTITY_B>, write only plain non-negative numbers.\n\n"
            "Required response template:\n\n"
            "<OBSERVATIONS>\n"
            "...\n"
            "</OBSERVATIONS>\n\n"
            "<PLANS>\n"
            "...\n"
            "</PLANS>\n\n"
            "<INSIGHTS>\n"
            "...\n"
            "</INSIGHTS>\n\n"
            "<QUANTITY_A>number</QUANTITY_A>\n"
            "<QUANTITY_B>number</QUANTITY_B>\n\n"
            "Anything you write in PLANS.txt and INSIGHTS.txt overwrites the previous contents, so keep any useful "
            "information you still need."
        )
