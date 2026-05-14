set -x

DATA_LOCAL_DIR="$HOME/data/drmas_duopoly"
ALPHAS=("1.0")
BETA="10000"
PROMPT_PREFIX_TYPES=("P1")
SEED_START=0
TRAIN_SEED_COUNT=512
TEST_SEED_COUNT=1
TEST_SAMPLED_SEED_COUNT=4

MODEL="gpt-4.1-mini"
API_KEY_ENV="OPENAI_API_KEY"
BASE_URL="https://api.apimart.ai/api/v1"
REASONING_EFFORT=""
THINKING_ENABLED=false
RETRIES=3
PARALLEL_AGENTS=false
EPISODE_COUNT="$TEST_SEED_COUNT"
MAX_STEPS=20
EXPERIMENT_NAME="competitive_duopoly_gpt_api_eval"
DATA_FILE="$DATA_LOCAL_DIR/test.parquet"

python3 examples/data_preprocess/drmas_duopoly.py \
  --local_dir "$DATA_LOCAL_DIR" \
  --alphas "${ALPHAS[@]}" \
  --prompt_prefix_types "${PROMPT_PREFIX_TYPES[@]}" \
  --seed_start "$SEED_START" \
  --train_seed_count "$TRAIN_SEED_COUNT" \
  --test_seed_count "$TEST_SEED_COUNT" \
  --test_sampled_seed_count "$TEST_SAMPLED_SEED_COUNT"

THINKING_ARG=()
if [ "$THINKING_ENABLED" = true ]; then
  THINKING_ARG+=(--thinking-enabled)
fi
PARALLEL_ARG=()
if [ "$PARALLEL_AGENTS" = true ]; then
  PARALLEL_ARG+=(--parallel-agents)
else
  PARALLEL_ARG+=(--serial-agents)
fi

python3 examples/api_eval/main_api_eval.py \
  --task duopoly \
  --provider openai \
  --model "$MODEL" \
  --api-key-env "$API_KEY_ENV" \
  --base-url "$BASE_URL" \
  --reasoning-effort "$REASONING_EFFORT" \
  --max-http-retries "$RETRIES" \
  "${THINKING_ARG[@]}" \
  "${PARALLEL_ARG[@]}" \
  --data-file "$DATA_FILE" \
  --episode-count "$EPISODE_COUNT" \
  --max-steps "$MAX_STEPS" \
  --duopoly-beta "$BETA" \
  --duopoly-prompt-prefix-type "${PROMPT_PREFIX_TYPES[0]}" \
  --experiment-name "$EXPERIMENT_NAME"
