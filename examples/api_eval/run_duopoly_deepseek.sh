set -x

DATA_LOCAL_DIR="$HOME/data/drmas_duopoly"
ALPHAS=("1.0")
BETA="10000"
PROMPT_PREFIX_TYPES=("P1")
SEED_START=0
TRAIN_SEED_COUNT=512
TEST_SEED_COUNT=1
TEST_SAMPLED_SEED_COUNT=4

MODEL="deepseek-chat"
API_KEY_ENV="DEEPSEEK_API_KEY"
BASE_URL="https://api.deepseek.com"
EPISODE_COUNT="$TEST_SEED_COUNT"
MAX_STEPS=20
TEMPERATURE=0.6
TOP_P=0.95
MAX_TOKENS=1024
EXPERIMENT_NAME="competitive_duopoly_deepseek_api_eval"
DATA_FILE="$DATA_LOCAL_DIR/test.parquet"

python3 examples/data_preprocess/drmas_duopoly.py \
  --local_dir "$DATA_LOCAL_DIR" \
  --alphas "${ALPHAS[@]}" \
  --prompt_prefix_types "${PROMPT_PREFIX_TYPES[@]}" \
  --seed_start "$SEED_START" \
  --train_seed_count "$TRAIN_SEED_COUNT" \
  --test_seed_count "$TEST_SEED_COUNT" \
  --test_sampled_seed_count "$TEST_SAMPLED_SEED_COUNT"

python3 examples/api_eval/main_api_eval.py \
  --task duopoly \
  --provider deepseek \
  --model "$MODEL" \
  --api-key-env "$API_KEY_ENV" \
  --base-url "$BASE_URL" \
  --data-file "$DATA_FILE" \
  --episode-count "$EPISODE_COUNT" \
  --max-steps "$MAX_STEPS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-tokens "$MAX_TOKENS" \
  --duopoly-beta "$BETA" \
  --duopoly-prompt-prefix-type "${PROMPT_PREFIX_TYPES[0]}" \
  --experiment-name "$EXPERIMENT_NAME"
