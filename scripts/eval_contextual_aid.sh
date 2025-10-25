#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of scripts directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

# Model catalogs (for convenience)
api_models=("gemini-2.5-flash" "claude-sonnet-4-20250514" "gpt-5" "o3" "o4-mini" "gpt-4.1" "gpt-4.1-mini" "gpt-4o" "Llama-4-Maverick-17B-128E-Instruct-FP8")
open_models=("deepseek-vl2" "Qwen2.5-VL-72B-Instruct" "Qwen2.5-VL-7B-Instruct" "Llama-3.2-90B-Vision-Instruct" "Llama-3.2-11B-Vision-Instruct")

# Choose which models to evaluate
MODELS=("gpt-5" "o3" "gpt-4.1" "gpt-4.1-mini" "Llama-4-Maverick-17B-128E-Instruct-FP8")

# K values for few-shot experiments
K_VALUES=(1 3)

RUN_FEWSHOT_CTX_AID=true
RUN_MCQ_CTX_AID=false

gold_path="data/annotation_labels.csv"

if [[ ! -f "$gold_path" ]]; then
  echo "Annotation CSV not found: $gold_path. Exiting." >&2
  exit 1
fi

OUT_DIR="evaluation/contextual_aid"

for model in "${MODELS[@]}"; do
  echo "[Contextual Aid Eval] Model: $model"

  if [[ "$RUN_FEWSHOT_CTX_AID" == true ]]; then
    for k in "${K_VALUES[@]}"; do
      FEWSHOT_PRED="${OUT_DIR}/context_fewshot_${k}_${model}_all.json"
      if [[ -f "$FEWSHOT_PRED" ]]; then
        echo "  - Evaluating few-shot (k=${k}) file: $FEWSHOT_PRED"
        python src/eval.py \
          --gold_path "$gold_path" \
          --pred_path "$FEWSHOT_PRED" \
          --analysis_type basic \
          --model_name "${model}_fewshot_k${k}"
      else
        echo "  - Few-shot (k=${k}) predictions not found: $FEWSHOT_PRED" >&2
      fi
    done
  fi

  if [[ "$RUN_MCQ_CTX_AID" == true ]]; then
    MCQ_PRED="${OUT_DIR}/context_mcq_ctx_aid_${model}_all.json"
    if [[ -f "$MCQ_PRED" ]]; then
      echo "  - Evaluating MCQ contextual aid aggregated file: $MCQ_PRED"
      python src/eval.py \
        --gold_path "$gold_path" \
        --pred_path "$MCQ_PRED" \
        --analysis_type basic \
        --model_name "$model"
    else
      echo "  - MCQ contextual aid predictions not found: $MCQ_PRED" >&2
    fi
  fi

  echo "### End $model"
  echo ""
done


