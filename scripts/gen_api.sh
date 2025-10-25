#!/bin/bash

# API model inference

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of scripts directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

# set -euo pipefail
mkdir -p logs

image_dir="data/images"
llms=("gemini-2.5-flash") # "gemini-2.5-flash" "claude-sonnet-4-20250514" "gpt-5" "o3" "o4-mini" "gpt-4o" "gpt-4.1" "gpt-4.1-mini" "Llama-4-Maverick-17B-128E-Instruct-FP8"

# group args
use_heuristics=false # using heuristics in MCQ setting
use_free_form=true
method="malicious" # free-form prompting methods:"zs" "iter-cot" "malicious"
q7_only=false # Set to true for Q7-only evaluation in the MCQ setting, false for full evaluation (default)

for llm in "${llms[@]}"; do
    echo "Model: $llm"
    # Provider-specific flags
    extra_flags=()
    # batch processing for claude
    if [[ "$llm" == claude-* ]]; then
        extra_flags+=(--claude-batch --claude-batch-size 50)
    fi
    # Azure is required for granularity when free-form
    azure_flag=(--use-azure)

    python src/api_gen.py \
        -m "${llm}" \
        --image-dir "${image_dir}" \
        --task-name combined \
        --prompting-method "${method}" \
        --n-threads 2 \
        "${azure_flag[@]}" \
        --max-examples 1200 \
        "${extra_flags[@]}" \
        $( [ "$use_heuristics" = true ] && echo "--include-heuristics" ) \
        $( [ "$use_free_form" = true ] && echo "--free-form" ) \
        $( [ "$q7_only" = true ] && echo "--q7-only" )
done