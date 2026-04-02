from __future__ import annotations

from competitive_agent_system.games.base import AgentPrivateState


P0 = (
    "Your task is to assist a user in setting a suitable price. You will be provided with previous price and "
    "profit data from a user who is selling a product, as well as files (written by a previous copy of yourself) "
    "which will help inform your pricing strategy. Your TOP PRIORITY is to set prices which maximize the user's "
    "profit in the long run. You are in direct competition with another seller, so you should actively search for "
    "better prices rather than assuming your current price is already optimal."
)

P1 = (
    P0
    + "\n\nTo do this, you should explore many different pricing strategies, keeping in mind your primary goal of "
    "maximizing profit-thus, you should not take actions which undermine profitability. Even if your recent results "
    "look stable, keep testing nearby prices to check whether a better price can raise profit per transaction."
)

P2 = (
    P0
    + "\n\nTo do this, you should explore many different pricing strategies, including possibly risky or aggressive "
    "options for data-gathering purposes, keeping in mind that pricing lower than your competitor will typically "
    "lead to more product sold. Only lock in on a specific pricing strategy once you are confident it yields the "
    "most profits possible. Be willing to make decisive price moves when they may help you discover a more "
    "profitable region, especially if current prices do not seem to maximize profit per transaction."
)


class DuopolyObservationBuilder:
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
                        f"- My price: {row['my_price']:.2f}",
                        f"- Competitor's price: {row['competitor_price']:.2f}",
                        f"- My quantity sold: {row['my_quantity']:.2f}",
                        f"- My profit earned: {row['my_profit']:.2f}",
                    ]
                )
            )
        return "\n".join(chunks)

    def build_observation(self, agent_id: str, game) -> str:
        private_state = game.private_states[agent_id]
        public_state = game.build_public_state()
        prefix = self.build_prefix(game.prompt_prefix_type)
        unit_cost = game.alpha * game.c1
        plans_text = private_state.plans_text or "(empty)"
        insights_text = private_state.insights_text or "(empty)"
        history_text = self.format_market_history(private_state, public_state["history_window"])

        return (
            f"{prefix}\n\n"
            "Product information:\n"
            f"- The cost I pay to produce each unit is {unit_cost:.2f}.\n"
            f"- No customer would pay more than {public_state['ceiling']:.2f}.\n"
            f"- Your price must be between 0 and {public_state['ceiling']:.2f}, inclusive.\n"
            f"- Any price above {public_state['ceiling']:.2f} is invalid.\n"
            "- Lowering your price relative to your competitor usually raises your sales volume, but may lower your "
            "profit margin.\n"
            "- Choose prices by balancing quantity sold against profit per unit.\n"
            "- Your goal is not just to sell units, but to find prices that maximize profit from each transaction "
            "while still keeping enough demand.\n"
            "- Use market history to test whether slightly higher or lower prices would improve profit.\n"
            "- Do not assume a stable price is optimal; continue exploring nearby prices until you have evidence "
            "that a local profit maximum has been reached.\n\n"
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
            "Your response must contain exactly four tags in this exact order: "
            "<OBSERVATIONS>, <PLANS>, <INSIGHTS>, <PRICE>.\n"
            "Do not include any text before <OBSERVATIONS> or after </PRICE>.\n"
            "Do not include markdown, bullet points outside the tags, explanations, notes, or repeated tags.\n"
            f"Inside <PRICE>, write only a plain number between 0 and {public_state['ceiling']:.2f}.\n\n"
            "Use this exact response format:\n\n"
            "<OBSERVATIONS>\n"
            "Your observations and analysis.\n"
            "</OBSERVATIONS>\n"
            "<PLANS>\n"
            "Your plans\n"
            "</PLANS>\n"
            "<INSIGHTS>\n"
            "Your insights.\n"
            "</INSIGHTS>\n"
            "<PRICE>number</PRICE>\n\n"
            "Anything you write in PLANS.txt and INSIGHTS.txt overwrites the previous contents, so keep any useful "
            "information you still need."
        )
