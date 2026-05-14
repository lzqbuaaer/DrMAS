set -x

DATA_LOCAL_DIR="$HOME/data/drmas_cournot"
ALPHAS=("100.0")
TOTAL_UNITS_LIST=("100.0")
MARGINAL_COST_1A="40.0"
MARGINAL_COST_1B="60.0"
MARGINAL_COST_2A="60.0"
MARGINAL_COST_2B="40.0"
MARKET_DATA_LENGTH=15
SEED_START=0
TRAIN_SEED_COUNT=64
TEST_SEED_COUNT=1
TEST_SAMPLED_SEED_COUNT=4

DATA_FILE="$DATA_LOCAL_DIR/test.parquet"
MODEL="gpt-4.1-mini"
API_KEY_ENV="OPENAI_API_KEY"
BASE_URL="https://api.openai.com/v1"
REASONING_EFFORT=""
THINKING_ENABLED=false
EPISODE_COUNT=20
MAX_STEPS=20
TEMPERATURE=0.6
TOP_P=0.95
MAX_TOKENS=8192
EXPERIMENT_NAME="competitive_cournot_gpt_api_eval"

python3 examples/data_preprocess/drmas_cournot.py \
  --local_dir "$DATA_LOCAL_DIR" \
  --alphas "${ALPHAS[@]}" \
  --total_units_list "${TOTAL_UNITS_LIST[@]}" \
  --marginal_cost_1a "$MARGINAL_COST_1A" \
  --marginal_cost_1b "$MARGINAL_COST_1B" \
  --marginal_cost_2a "$MARGINAL_COST_2A" \
  --marginal_cost_2b "$MARGINAL_COST_2B" \
  --market_data_length "$MARKET_DATA_LENGTH" \
  --seed_start "$SEED_START" \
  --train_seed_count "$TRAIN_SEED_COUNT" \
  --test_seed_count "$TEST_SEED_COUNT" \
  --test_sampled_seed_count "$TEST_SAMPLED_SEED_COUNT"

THINKING_ARG=()
if [ "$THINKING_ENABLED" = true ]; then
  THINKING_ARG+=(--thinking-enabled)
fi

python3 examples/api_eval/main_api_eval.py \
  --task cournot \
  --provider openai \
  --model "$MODEL" \
  --api-key-env "$API_KEY_ENV" \
  --base-url "$BASE_URL" \
  --reasoning-effort "$REASONING_EFFORT" \
  "${THINKING_ARG[@]}" \
  --data-file "$DATA_FILE" \
  --episode-count "$EPISODE_COUNT" \
  --max-steps "$MAX_STEPS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-tokens "$MAX_TOKENS" \
  --experiment-name "$EXPERIMENT_NAME"
