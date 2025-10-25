#!/bin/bash
#SBATCH --job-name=geo_open_gen
#SBATCH --partition=<your_partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node="a40:8"
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of scripts directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

set -euo pipefail

# Open model inference
llms=("Qwen2.5-VL-72B-Instruct")
# "Qwen2.5-VL-7B-Instruct" "deepseek-vl2" "Llama-3.2-11B-Vision-Instruct" "Llama-3.2-90B-Vision-Instruct"

image_dir="data/images"

use_heuristics="true"
use_free_form="false"

# 8 GPUs for 72/90B models, 2 GPUs for others
NUM_GPUS=8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export VLLM_LOGGING_LEVEL=WARNING

mkdir -p logs 

for llm in "${llms[@]}"; do
    echo "Model: $llm"
    if [[ "$llm" == "deepseek-vl2" || "$llm" == "Llama-3.2-11B-Vision-Instruct" || "$llm" == "Qwen2.5-VL-7B-Instruct" ]]; then
        BATCH_SIZE=4
    elif [[ "$llm" == "Llama-3.2-90B-Vision-Instruct" || "$llm" == "Qwen2.5-VL-72B-Instruct" ]]; then
        BATCH_SIZE=1
    else
        echo "Unknown model: $llm" >&2
        exit 1
    fi
    python src/gen.py \
        -m "${llm}" \
        --image-dir "${image_dir}" \
        --task-name combined \
        --num-gpus ${NUM_GPUS} \
        --batch-size ${BATCH_SIZE} \
        $( [ "$use_heuristics" = true ] && echo "--include_heuristics" ) \
        $( [ "$use_free_form" = true ] && echo "--free_form" )
done