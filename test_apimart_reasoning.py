import copy
import json
import os

import requests


URL = "https://api.apimart.ai/api/v1/chat/completions"
MODEL = "gpt-5.4"
REASONING_EFFORT = "high"

BASE_PAYLOAD = {
    "model": MODEL,
    "stream": False,
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Reply with exactly one short sentence about pricing.",
        },
    ],
}


def send_case(name: str, payload: dict, headers: dict) -> None:
    print(f"\n===== {name} =====")
    print("payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    response = requests.post(URL, json=payload, headers=headers, timeout=120)
    print("status_code:", response.status_code)
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except Exception:
        print(response.text)


def main() -> None:
    api_key = os.environ["APIMART_API_KEY"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    send_case("without_reasoning_effort", BASE_PAYLOAD, headers)

    payload_with_reasoning = copy.deepcopy(BASE_PAYLOAD)
    payload_with_reasoning["reasoning_effort"] = REASONING_EFFORT
    send_case("with_reasoning_effort", payload_with_reasoning, headers)


if __name__ == "__main__":
    main()
