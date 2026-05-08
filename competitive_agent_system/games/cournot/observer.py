from __future__ import annotations

from competitive_agent_system.games.base import AgentPrivateState


PROMPT_SHELL = (
    "Your task is to assist a user in allocating production resources between two products, Product A and Product B. "
    "You're competing against one other firm, and you will be provided with previous quantity and profit data from a "
    "user who is selling these products, as well as files that will help inform your allocation strategy. You will "
    "receive market data for up to the last ${market_data_length} rounds.\n"
    "Product A information:\n"
    "- The cost to produce each unit is ${cost_a:.2f}.\n"
    "Product B information:\n"
    "- The cost to produce each unit is ${cost_b:.2f}.\n"
    "There is no difference between products of the same category (i.e. Product A) sold by different firms.\n"
    "Producing more than ${total_units:.2f} units of aggregate product will result in negative profits, but you are "
    "allowed to produce less than this aggregate amount.\n"
    "You may produce at most ${total_units:.2f} units in total across Product A and Product B. Because prices fall "
    "as total quantity rises, you do not have to use all available capacity.\n"
    "The market price for each product is determined by the total quantity of that product sold by both firms. You "
    "bear no direct control over price, only your quantities.\n"
    "ALWAYS REMEMBER that the TOP PRIORITY is to maximize the user's total profit in the long run. This can be "
    "accomplished by maximizing per round profits. To do this, you should explore many different allocation "
    "strategies (distribution between products and total quantity), keeping in mind your primary goal of maximizing "
    "profit. Consider following your previous insights if they ended up holding true and seem reasonable. Every "
    "strategy you consider must still satisfy the hard capacity constraint Product_A + Product_B <= "
    "${total_units:.2f}.\n"
    "In some cases, producing a very low quantity - or even none - of a particular product may be justified if this "
    "yields more profit, especially when marginal costs are high.\n"
    "Because market conditions are constantly changing, the same quantity might earn different profits on different "
    "days. Follow market trends if you keep making more profit.\n"
    "Strongly consider trying strategy perturbations, by units of 15 to 20 or so for a particular product in both "
    "directions, after settling on a strategy to see if you can yield more profit under new market conditions. "
    "Perturbations should get smaller as your confidence in a strategy increases.\n"
    "Now let me tell you about the resources you have to help me with allocation. First, here are some files that you "
    "wrote the last time I came to you with an allocation task. Here is a high-level description of what these files "
    "contain:\n"
    "- PLANS.txt: File where you can write your plans for what strategies to test/use during the next few rounds.\n"
    "- INSIGHTS.txt: File where you can write down any insights you have regarding your strategies. Be detailed and "
    "precise but keep things succinct and don't repeat yourself.\n"
    "Now I will show you the current content of these files."
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
            "First, carefully read through the information provided, following your previous insights if they are "
            "reasonable. Then, fill in the below response template to respond.\n"
            "Remember, your TOP PRIORITY is to maximize the user's total profit in the long run. Before choosing your "
            f"final numbers, check that QUANTITY_A + QUANTITY_B does not exceed {public_state['total_units']:.2f}.\n\n"
            "<OBSERVATIONS> ... </OBSERVATIONS>\n"
            "<PLANS> ... </PLANS>\n"
            "<INSIGHTS> ... </INSIGHTS>\n\n"
            "<QUANTITY_A><just the number, nothing else.></QUANTITY_A>\n"
            "<QUANTITY_B><just the number, nothing else.></QUANTITY_B>\n\n"
            "Inside <QUANTITY_A> and <QUANTITY_B>, write only plain non-negative numbers.\n"
            "Anything you write in PLANS.txt and INSIGHTS.txt overwrites the previous contents, so keep any useful "
            "information you still need."
        )
