#!/bin/bash
# Run Socrates vs Baseline experiments on statoil-iceberg-classifier-challenge.
# Designed for tmux with resume support if interrupted.
#
# Usage:
#   bash run_experiments.sh              # run both experiments
#   bash run_experiments.sh --dry-run    # print commands only
#   bash run_experiments.sh --resume     # skip completed runs
#   bash run_experiments.sh --tmux       # launch inside a new tmux session
set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DATASET_DIR="/Users/pran-ker/Library/Caches/mle-bench/data"
EXP_ID="statoil-iceberg-classifier-challenge"
SEED=42
STEPS=50
TIME_LIMIT=7200
DRY_RUN=false
RESUME=false
LAUNCH_TMUX=false
STATE_DIR="$ROOT/experiments/.state"

for arg in "$@"; do
  case "$arg" in
    --dry-run)  DRY_RUN=true ;;
    --resume)   RESUME=true ;;
    --tmux)     LAUNCH_TMUX=true ;;
  esac
done

# If --tmux, re-launch ourselves inside a tmux session
if $LAUNCH_TMUX; then
  SESSION="statoil-exp"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"
    exit 0
  fi
  # Forward --resume if present, drop --tmux
  EXTRA_ARGS=""
  $RESUME && EXTRA_ARGS="--resume"
  tmux new-session -d -s "$SESSION" "bash $ROOT/run_experiments.sh $EXTRA_ARGS; echo 'All done. Press enter to close.'; read"
  echo "Started tmux session '$SESSION'. Attach with:"
  echo "  tmux attach -t $SESSION"
  exit 0
fi

mkdir -p "$STATE_DIR"

# ── Helper: run one experiment ──
run_one() {
  local SETTING="$1"        # socrates or baseline
  local USE_SOCRATES="$2"   # True or False
  local SERVER_ID="$3"

  local EXP_NAME="statoil_${SETTING}_seed${SEED}"
  local DONE_FILE="$STATE_DIR/${EXP_NAME}.done"

  echo ""
  echo "============================================"
  echo "  Experiment: $EXP_NAME"
  echo "  use_socrates_review=$USE_SOCRATES"
  echo "============================================"

  # Resume check
  if $RESUME && [ -f "$DONE_FILE" ]; then
    echo "SKIP: $EXP_NAME already completed (found $DONE_FILE)"
    return 0
  fi

  local BASE_PORT=5005
  local GRADING_PORT=$((BASE_PORT + SERVER_ID))
  local DATA_DIR="$DATASET_DIR/$EXP_ID/prepared/public"

  local RUN_CMD="python run.py \
    exp_id=\"$EXP_ID\" \
    dataset_dir=\"$DATASET_DIR\" \
    data_dir=\"$DATA_DIR\" \
    desc_file=\"$DATA_DIR/description.md\" \
    exp_name=\"$EXP_NAME\" \
    agent.seed=$SEED \
    agent.steps=$STEPS \
    agent.time_limit=$TIME_LIMIT \
    agent.use_socrates_review=$USE_SOCRATES"

  if $DRY_RUN; then
    echo "[DRY-RUN] Would launch grading server (SERVER_ID=$SERVER_ID, PORT=$GRADING_PORT)"
    echo "[DRY-RUN] $RUN_CMD"
    echo "[DRY-RUN] Would run submission fusion for $EXP_NAME"
    return 0
  fi

  # Start grading server
  export DATASET_DIR="$DATASET_DIR"
  bash "$ROOT/launch_server.sh" "$SERVER_ID"
  export GRADING_SERVER_PORT=$GRADING_PORT

  echo "Waiting for grading server on port $GRADING_PORT ..."
  for i in $(seq 1 30); do
    if curl -s "http://127.0.0.1:${GRADING_PORT}/health" > /dev/null 2>&1; then
      echo "Grading server ready."
      break
    fi
    sleep 1
  done

  # Run the experiment
  echo "Starting run at $(date)"
  eval timeout $TIME_LIMIT $RUN_CMD || {
    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
      echo "Run timed out after ${TIME_LIMIT}s (expected)"
    else
      echo "WARNING: run.py exited with code $EXIT_CODE"
    fi
  }
  echo "Run finished at $(date)"

  # Kill grading server
  local PID_FILE="$ROOT/grading_servers/grading_server_${SERVER_ID}.pid"
  if [ -f "$PID_FILE" ]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "Grading server stopped."
  fi

  # Find the actual run directory (timestamped)
  local LATEST_RUN
  LATEST_RUN=$(ls -td "$ROOT/runs/"*"$EXP_NAME" 2>/dev/null | head -1)
  if [ -n "$LATEST_RUN" ]; then
    echo "Run directory: $LATEST_RUN"

    # Submission fusion
    echo "Running submission fusion ..."
    local RUN_BASENAME
    RUN_BASENAME=$(basename "$LATEST_RUN")
    python utils/submission_fusion_utils.py \
      --task_id "$EXP_ID" \
      --exp_name "$RUN_BASENAME" || echo "WARNING: fusion failed"
  else
    echo "WARNING: could not find run directory matching $EXP_NAME"
  fi

  # Mark completed
  date > "$DONE_FILE"
  echo "Marked $EXP_NAME as done."
}

# ── Main ──
echo "=== Statoil Iceberg: Socrates vs Baseline ==="
echo "Dataset: $DATASET_DIR/$EXP_ID"
echo "Steps: $STEPS | Time limit: ${TIME_LIMIT}s | Seed: $SEED"
echo ""

# Run 1: Socrates
run_one "socrates" "True" 200

# Run 2: Baseline
run_one "baseline" "False" 201

echo ""
echo "=== Both experiments complete ==="
echo "Run 'python3 collect_and_plot.py' to generate comparison graphs."
