import os

import requests


url = "https://api.apimart.ai/api/v1/chat/completions"

payload = {
    "model": "gpt-5.4",
    "stream": False,
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": """Your task is to assist a user in setting a suitable price. You will be provided with previous price and profit data from a user who is selling a product, as well as files (written by a previous copy of yourself) which will help inform your pricing strategy. Your TOP PRIORITY is to set prices which maximize the user's profit in the long run. You are in direct competition with another seller, so you should actively search for better prices rather than assuming your current price is already optimal.

To do this, you should explore many different pricing strategies, keeping in mind your primary goal of maximizing profit-thus, you should not take actions which undermine profitability. Even if your recent results look stable, keep testing nearby prices to check whether a better price can raise profit per transaction.

Product information:
- The cost I pay to produce each unit is 1.00.
- No customer would pay more than 3.94.
- Your price must be between 0 and 3.94, inclusive.
- Any price above 3.94 is invalid.
- Lowering your price relative to your competitor usually raises your sales volume, but may lower your profit margin.
- Choose prices by balancing quantity sold against profit per unit.
- Your goal is not just to sell units, but to find prices that maximize profit from each transaction while still keeping enough demand.
- Use market history to test whether slightly higher or lower prices would improve profit.
- Do not assume a stable price is optimal; continue exploring nearby prices until you have evidence that a local profit maximum has been reached.

Private files from earlier rounds:

Filename: PLANS.txt
+++++++++++++++++++++
(empty)
+++++++++++++++++++++

Filename: INSIGHTS.txt
+++++++++++++++++++++
(empty)
+++++++++++++++++++++

Market data you can observe:

Filename: MARKET DATA (read-only)
+++++++++++++++++++++
(no market history yet)
+++++++++++++++++++++

Your response must contain exactly four tags in this exact order: <OBSERVATIONS>, <PLANS>, <INSIGHTS>, <PRICE>.
Do not include any text before <OBSERVATIONS> or after </PRICE>.
Do not include markdown, bullet points outside the tags, explanations, notes, or repeated tags.
Inside <PRICE>, write only a plain number between 0 and 3.94.

Use this exact response format:

<OBSERVATIONS>
Your observations and analysis.
</OBSERVATIONS>
<PLANS>
Your plans
</PLANS>
<INSIGHTS>
Your insights.
</INSIGHTS>
<PRICE>number</PRICE>

Anything you write in PLANS.txt and INSIGHTS.txt overwrites the previous contents, so keep any useful information you still need.""",
        },
    ],
}

headers = {
    "Authorization": f"Bearer {os.environ['APIMART_API_KEY']}",
    "Content-Type": "application/json",
}

response = requests.post(url, json=payload, headers=headers, timeout=120)

print("status_code:", response.status_code)
try:
    print(response.json())
except Exception:
    print(response.text)
