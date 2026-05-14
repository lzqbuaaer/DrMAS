set -x

MODEL="gpt-4.1-mini"
API_KEY_ENV="OPENAI_API_KEY"
BASE_URL="https://api.apimart.ai/api/v1"
REASONING_EFFORT=""
TEMPERATURE=0.6
TOP_P=0.95
MAX_TOKENS=512
SYSTEM_MESSAGE="You are a helpful assistant."
USER_MESSAGE="Hello"

python3 examples/api_eval/test_gpt_api.py \
  --model "$MODEL" \
  --api-key-env "$API_KEY_ENV" \
  --base-url "$BASE_URL" \
  --reasoning-effort "$REASONING_EFFORT" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-tokens "$MAX_TOKENS" \
  --system-message "$SYSTEM_MESSAGE" \
  --message "$USER_MESSAGE"
