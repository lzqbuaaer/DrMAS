from __future__ import annotations


def build_retry_feedback(task: str, agent_id: str, parsed_action, parse_kwargs: dict) -> str:
    normalized_task = str(task).lower()
    feedback_lines = [
        "RETRY FEEDBACK:",
        f"Your previous answer for {agent_id} was invalid.",
        parsed_action.error or "Your previous answer was invalid.",
    ]

    if "cournot" in normalized_task:
        total_units = parse_kwargs.get("total_units")
        if total_units is not None:
            feedback_lines.extend(
                [
                    f"Resubmit with QUANTITY_A + QUANTITY_B <= {float(total_units):.2f}.",
                    "Choose a feasible total output first, then split it between Product A and Product B.",
                ]
            )
    elif "duopoly" in normalized_task:
        max_price = parse_kwargs.get("max_price")
        if max_price is not None:
            feedback_lines.append(f"Resubmit with PRICE <= {float(max_price):.2f}.")

    feedback_lines.append("Keep exactly the same XML tag format and only correct the invalid parts.")
    return "\n".join(feedback_lines)
