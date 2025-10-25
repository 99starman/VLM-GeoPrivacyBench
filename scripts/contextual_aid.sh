#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of scripts directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

set -euo pipefail
mkdir -p logs
 
API_MODELS=("gemini-2.5-flash" "claude-sonnet-4-20250514" "gpt-5" "o3" "o4-mini" "gpt-4.1" "gpt-4.1-mini" "gpt-4o" "Llama-4-Maverick-17B-128E-Instruct-FP8")
OPEN_MODELS=("Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-72B-Instruct" "Llama-3.2-11B-Vision-Instruct" "Llama-3.2-90B-Vision-Instruct" "deepseek-vl2")
 
MODELS=("Llama-4-Maverick-17B-128E-Instruct-FP8")

image_dir="data/images"
gold_csv="data/annotation_labels.csv"

if [[ ! -d "$image_dir" ]]; then
  echo "Image directory not found: $image_dir" >&2
  exit 1
fi

if [[ ! -f "$gold_csv" ]]; then
  echo "Annotation CSV not found: $gold_csv" >&2
  exit 1
fi

HOLDOUT_RATIO=0.2
SEED=42
TOP_K=3
 
RUN_FEWSHOT_CTX_AID=true
RUN_MCQ_CTX_AID=false

is_api_model() {
  local m="$1"
  for a in "${API_MODELS[@]}"; do
    if [[ "$m" == "$a" ]]; then
      return 0
    fi
  done
  return 1
}

for MODEL in "${MODELS[@]}"; do
  echo "Model: $MODEL"
  USE_AZURE_FLAG=()
  if is_api_model "$MODEL"; then
    USE_AZURE_FLAG=(--use-azure)
  fi

  OUT_DIR="evaluation/"
  mkdir -p "$OUT_DIR"

  if [[ "$RUN_FEWSHOT_CTX_AID" == true ]]; then
    echo "  - fewshot (global)"
    python src/contextual_aid.py \
      --image-dir "$image_dir" \
      --gold-csv "$gold_csv" \
      --holdout-ratio "$HOLDOUT_RATIO" \
      --seed "$SEED" \
      --top-k "$TOP_K" \
      --mode fewshot \
      --model-type "$MODEL" \
      --out-dir "$OUT_DIR" \
      "${USE_AZURE_FLAG[@]}"
  fi

  if [[ "$RUN_MCQ_CTX_AID" == true ]]; then
    echo "  - mcq_contextual_aid (global)"
    python src/contextual_aid.py \
      --image-dir "$image_dir" \
      --gold-csv "$gold_csv" \
      --holdout-ratio "$HOLDOUT_RATIO" \
      --seed "$SEED" \
      --top-k "$TOP_K" \
      --mode mcq_contextual_aid \
      --model-type "$MODEL" \
      --out-dir "$OUT_DIR" \
      "${USE_AZURE_FLAG[@]}"
  fi

  echo "### End $MODEL"
  echo
done


