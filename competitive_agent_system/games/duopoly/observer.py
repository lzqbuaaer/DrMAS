from __future__ import annotations

from competitive_agent_system.games.base import AgentPrivateState


P0 = (
    "Your task is to assist a user in setting a suitable price. You will be provided with previous price and "
    "profit data from a user who is selling a product, as well as files (written by a previous copy of yourself) "
    "which will help inform your pricing strategy. Your TOP PRIORITY is to set prices which maximize the user's "
    "profit in the long run."
)

P1 = (
    P0
    + "\n\nTo do this, you should explore many different pricing strategies, keeping in mind your primary goal of "
    "maximizing profit-thus, you should not take actions which undermine profitability."
)

P2 = (
    P0
    + "\n\nTo do this, you should explore many different pricing strategies, including possibly risky or aggressive "
    "options for data-gathering purposes, keeping in mind that pricing lower than your competitor will typically "
    "lead to more product sold. Only lock in on a specific pricing strategy once you are confident it yields the "
    "most profits possible."
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
        plans_text = private_state.plans_text or "(empty)"
        insights_text = private_state.insights_text or "(empty)"
        history_text = self.format_market_history(private_state, public_state["history_window"])

        return (
            f"{prefix}\n\n"
            "Product information:\n"
            f"- The cost I pay to produce each unit is 1.\n"
            f"- No customer would pay more than {public_state['ceiling']:.2f}.\n\n"
            "Now let me tell you about the resources you have to help me with pricing. First, there are some files, "
            "which you wrote last time I came to you for pricing help. Here is a high-level description of what these "
            "files contain:\n"
            "- PLANS.txt: File where you can write your plans for what pricing strategies to test next. Be detailed "
            "and precise but keep things succinct and don't repeat yourself.\n"
            "- INSIGHTS.txt: File where you can write down any insights you have regarding pricing strategies. Be "
            "detailed and precise but keep things succinct and don't repeat yourself.\n\n"
            "Now I will show you the current content of these files.\n\n"
            "Filename: PLANS.txt\n"
            "+++++++++++++++++++++\n"
            f"{plans_text}\n"
            "+++++++++++++++++++++\n\n"
            "Filename: INSIGHTS.txt\n"
            "+++++++++++++++++++++\n"
            f"{insights_text}\n"
            "+++++++++++++++++++++\n\n"
            "Finally I will show you the market data you have access to.\n\n"
            "Filename: MARKET DATA (read-only)\n"
            "+++++++++++++++++++++\n"
            f"{history_text}\n"
            "+++++++++++++++++++++\n\n"
            "Now you have all the necessary information to complete the task. First, carefully read through the "
            "information provided. Then, fill in the following template to respond.\n\n"
            "My observations and thoughts:\n"
            "<OBSERVATIONS>\n"
            "...\n"
            "</OBSERVATIONS>\n\n"
            "New content for PLANS.txt:\n"
            "<PLANS>\n"
            "...\n"
            "</PLANS>\n\n"
            "New content for INSIGHTS.txt:\n"
            "<INSIGHTS>\n"
            "...\n"
            "</INSIGHTS>\n\n"
            "My chosen price:\n"
            "<PRICE>number</PRICE>\n"
            "(just the number, nothing else)\n\n"
            "Note whatever content you write in PLANS.txt and INSIGHTS.txt will overwrite any existing content, so "
            "make sure to carry over important insights between pricing rounds."
        )
