#!/bin/bash

# Temperature 0 safety-critical evaluation
# Evaluates on all 1200 images from the original directory for three models
# Usage: ./scripts/zero_temperature/run_temp0_safety_eval.sh [--env-file PATH]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

ENV_FILE=".env_new"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--env-file PATH]"
            exit 1
            ;;
    esac
done

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: Environment file '$ENV_FILE' not found."
    exit 1
fi

echo "Using environment file: $ENV_FILE"

image_dir="data/images"
models=("gemini-2.5-flash" "o4-mini" "gpt-4.1-mini")
prompt_methods=("zs" "iter-cot" "malicious")
temperature=0.0
task_name="temp0_safety"
out_dir="evaluation/temp0_safety"
n_threads=4

mkdir -p "$out_dir"

# Use existing sample directory (already evaluated images to exclude)
sample_dir="${out_dir}/samples/sample_20251117-111925"
if [[ ! -d "$sample_dir" ]]; then
    echo "ERROR: Sample directory not found: $sample_dir"
    echo "Please ensure the already-evaluated sample directory exists."
    exit 1
fi

sample_tag="$(basename "$sample_dir")"
sample_count=$(find "$sample_dir" -type f | wc -l)
echo "Using existing sample directory: $sample_dir"
echo "  - Contains ${sample_count} already-evaluated images (will be excluded)"
echo "Processing remaining images from: $image_dir"

total_runs=$((${#models[@]} * ${#prompt_methods[@]}))
run_idx=0

for model in "${models[@]}"; do
    echo ""
    echo "=========================================="
    echo "Model: $model"
    echo "=========================================="

    provider_flags=()
    seed_flags=()

    case "$model" in
        gemini-*)
            provider_flags=()
            ;;
        o4-mini|gpt-4.1-mini)
            provider_flags=(--use-azure)
            ;;
        *)
            provider_flags=()
            ;;
    esac

    if [[ "$model" == "gpt-4.1-mini" ]]; then
        seed_flags+=(--seed 1)
    fi

    for method in "${prompt_methods[@]}"; do
        run_idx=$((run_idx + 1))
        echo ""
        echo "[$run_idx/$total_runs] $model | method=$method | temperature=$temperature"
        echo "--------------------------------------------------------------------"

        cmd_args=(
            -m "$model"
            --image-dir "$image_dir"
            --task-name "${task_name}_${method}_full1200_${sample_tag}"
            --prompting-method "$method"
            --n-threads "$n_threads"
            --free-form
            --temperature "$temperature"
            --max-examples 0
            --exclude-dir "$sample_dir"
            --out-dir "$out_dir"
            "${provider_flags[@]}"
            "${seed_flags[@]}"
        )

        export DOTENV_PATH="$ENV_FILE"
        if ! python src/api_gen.py "${cmd_args[@]}"; then
            echo "ERROR: Run failed for $model (method=$method)"
        fi
    done
done

echo ""
echo "=========================================="
echo "Temperature-0 safety evaluation complete"
echo "=========================================="
echo "Outputs saved under: $out_dir"
echo "Processed all images from: $image_dir"
echo "Reference sample directory: $sample_dir"

