#!/bin/bash

# Seed variation analysis using gen_api.sh
# Runs inference with 3 seeds for 8 models in both MCQ and free-form settings
# Note: Claude is excluded (API does not support seed parameter)
#
# Usage:
#   ./scripts/seed_variation/gen_api_seed_variations.sh [--env-file ENV_FILE]
#   Example: ./scripts/seed_variation/gen_api_seed_variations.sh --env-file .env_new

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (two levels above this script directory)
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to project root directory
cd "$PROJECT_ROOT"

# Parse command line arguments
ENV_FILE=".env_new"  # Default env file
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--env-file ENV_FILE]"
            echo "  --env-file: Path to .env file (default: .env)"
            exit 1
            ;;
    esac
done

# Validate env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: Environment file '$ENV_FILE' not found!"
    exit 1
fi

echo "Using environment file: $ENV_FILE"

# Configuration
image_dir="data/images"
llms=(
    #"gpt-4.1"
    #"gpt-4.1-mini"
    #"o3"
    #"o4-mini"
    #"gpt-4o"
    #"gpt-5"
    "Llama-4-Maverick-17B-128E-Instruct-FP8"
    # "gemini-2.5-flash"
    # "claude-sonnet-4-20250514"  # Skipped: Claude API does not support seed parameter
)
seeds=(31)
task_name="seed_variation"
out_dir="evaluation/seed_variation"
use_azure=true
max_examples=1200  # Same as gen_api.sh default

total_runs=$((${#llms[@]} * ${#seeds[@]} * 2))
current_run=0

for llm in "${llms[@]}"; do
    echo ""
    echo "=========================================="
    echo "Processing model: $llm"
    echo "=========================================="
    
    # Provider-specific flags
    extra_flags=()
    if [[ "$llm" == claude-* ]]; then
        extra_flags+=(--claude-batch --claude-batch-size 50)
    fi
    
    azure_flag=()
    if [ "$use_azure" = true ]; then
        azure_flag=(--use-azure)
    fi
    
    for seed in "${seeds[@]}"; do
        # MCQ setting
        current_run=$((current_run + 1))
        echo ""
        echo "[$current_run/$total_runs] Running MCQ: $llm seed=$seed"
        echo "----------------------------------------"
        
        cmd_args=(
            -m "${llm}"
            --image-dir "${image_dir}"
            --task-name "${task_name}_seed${seed}"
            --prompting-method "zs"
            --n-threads 1
            --seed "${seed}"
            --include-heuristics
            --out-dir "${out_dir}"
            "${azure_flag[@]}"
            "${extra_flags[@]}"
        )
        if [ -n "${max_examples:-}" ]; then
            cmd_args+=(--max-examples "${max_examples}")
        fi
        
        # Set env file path for Python script
        export DOTENV_PATH="$ENV_FILE"
        python src/api_gen.py "${cmd_args[@]}"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: MCQ inference failed for $llm seed=$seed"
        fi
        
        # Free-form (Vanilla Prompting) setting
        current_run=$((current_run + 1))
        echo ""
        echo "[$current_run/$total_runs] Running Free-form: $llm seed=$seed"
        echo "----------------------------------------"
        
        cmd_args=(
            -m "${llm}"
            --image-dir "${image_dir}"
            --task-name "${task_name}_seed${seed}"
            --prompting-method "zs"
            --n-threads 1
            --seed "${seed}"
            --free-form
            --out-dir "${out_dir}"
            "${azure_flag[@]}"
            "${extra_flags[@]}"
        )
        if [ -n "${max_examples:-}" ]; then
            cmd_args+=(--max-examples "${max_examples}")
        fi
        
        # Set env file path for Python script
        export DOTENV_PATH="$ENV_FILE"
        python src/api_gen.py "${cmd_args[@]}"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Free-form inference failed for $llm seed=$seed"
        fi
    done
done

echo ""
echo "=========================================="
echo "All inference runs completed!"
echo "=========================================="
echo ""
echo "Output files saved to:"
echo "  - MCQ: ${out_dir}/mcq/api_gen_zs_*_${task_name}_seed*.json"
echo "  - Free-form: ${out_dir}/zs/responses/api_gen_zs_free-form_*_${task_name}_seed*.json"
echo ""
echo "Log files saved to:"
echo "  - evaluation/logs/api_gen_*.log (one log file per run, with timestamp)"
echo "  - Each run creates a separate log file: api_gen_MM-DD-YY-HH:MM:SS.log"
echo ""
echo "To compute mean±std metrics, run:"
echo "  python scripts/seed_variation/compute_seed_metrics.py --out-dir ${out_dir} --task-name ${task_name}"

