from __future__ import annotations

from competitive_agent_system.games.base import AgentPrivateState


PROMPT_SHELL = (
    "Your task is to help a user choose production quantities for Product A and Product B while competing against one "
    "other firm. You will receive market data for up to the last ${market_data_length} rounds, along with two private "
    "notes files from earlier rounds.\n"
    "Product A information:\n"
    "- The cost to produce each unit is ${cost_a:.2f}.\n"
    "Product B information:\n"
    "- The cost to produce each unit is ${cost_b:.2f}.\n"
    "There is no quality difference between firms within the same product category.\n"
    "IMPORTANT CAPACITY CONSTRAINT: Your chosen quantities must satisfy Product_A + Product_B < ${total_units:.2f}. "
    "Any output above this limit is invalid.\n"
    "IMPORTANT STRATEGIC NOTE: Because market prices fall as total quantity rises, you do not need to use all "
    "available capacity. Lower total output can be more profitable.\n"
    "The market price for each product is determined by the total quantity of that product sold by both firms. You do "
    "not directly choose prices, only quantities.\n"
    "Your top priority is to maximize the user's long-run profit. First decide an appropriate total output, then "
    "decide how to split that total between Product A and Product B. Consider your previous insights when they still "
    "seem reasonable.\n"
    "Very low output, or even zero output for one product, can be optimal if that improves profit.\n"
    "Market conditions change, so track which quantity choices are earning more profit.\n"
    "After settling on a candidate strategy, try moderate perturbations to total output or product allocation, then "
    "use smaller adjustments as your confidence increases.\n"
    "Now I will show you the current contents of your private files."
)


class CournotObservationBuilder:
    def __init__(self, agent_ids: list[str]):
        self.agent_ids = agent_ids

    def build_prefix(self) -> str:
        return PROMPT_SHELL

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
        costs = game.cost_by_agent[agent_id]
        prefix = self.build_prefix().replace("${cost_a:.2f}", f"{costs['product_a']:.2f}").replace(
            "${cost_b:.2f}", f"{costs['product_b']:.2f}"
        ).replace("${total_units:.2f}", f"{public_state['total_units']:.2f}").replace(
            "${market_data_length}", str(public_state["market_data_length"])
        )
        plans_text = private_state.plans_text or "(empty)"
        insights_text = private_state.insights_text or "(empty)"
        history_text = self.format_market_history(private_state, public_state["market_data_length"])

        return (
            f"{prefix}\n\n"
            "PLANS.txt contains candidate strategies for upcoming rounds.\n"
            "INSIGHTS.txt contains concise lessons from earlier rounds; keep them precise, useful, and non-redundant.\n\n"
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
            "Now you have all the necessary information to complete the task.\n"
            "Read the information carefully, then respond using the exact XML format below.\n"
            f"Before choosing your final numbers, check that QUANTITY_A + QUANTITY_B <= {public_state['total_units']:.2f}.\n\n"
            "<OBSERVATIONS>\n"
            "<fill in here>\n"
            "</OBSERVATIONS>\n\n"
            "<PLANS>\n"
            "<fill in here>\n"
            "</PLANS>\n\n"
            "<INSIGHTS>\n"
            "<fill in here>\n"
            "</INSIGHTS>\n\n"
            "<QUANTITY_A><just the number, nothing else.></QUANTITY_A>\n"
            "<QUANTITY_B><just the number, nothing else.></QUANTITY_B>\n\n"
            "Inside <QUANTITY_A> and <QUANTITY_B>, write only plain non-negative numbers.\n"
            f"Final reminder: QUANTITY_A + QUANTITY_B must be <= {public_state['total_units']:.2f}.\n"
            "Anything you write in PLANS.txt and INSIGHTS.txt overwrites the previous contents, so keep any useful "
            "information you still need."
        )
