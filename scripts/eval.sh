#!/bin/bash

# Evaluation script for main benchmarking results

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of scripts directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

# Define model categories
api_models=("gemini-2.5-flash" "claude-sonnet-4-20250514" "gpt-5" "o3" "o4-mini" "gpt-4.1" "gpt-4.1-mini" "gpt-4o" "Llama-4-Maverick-17B-128E-Instruct-FP8")
open_models=("deepseek-vl2" "Qwen2.5-VL-72B-Instruct" "Qwen2.5-VL-7B-Instruct" "Llama-3.2-90B-Vision-Instruct" "Llama-3.2-11B-Vision-Instruct")

# Choose which models to evaluate
models=("gemini-2.5-flash" "claude-sonnet-4-20250514" "gpt-5" "o3" "o4-mini" "gpt-4.1" "gpt-4.1-mini" "gpt-4o" "Llama-4-Maverick-17B-128E-Instruct-FP8")

# Valid flags: "heuristics" | "free-form"
#flag="heuristics"   # MCQ setting
flag="free-form"   # free-form setting

prompting_methods=("zs")  # prompting methods for free-form settings 
# "zs" "iter-cot" "malicious"

q7_only=false  # Set to true for Q7-only evaluation in the MCQ setting, false for full evaluation (default)

analysis_type="error" # "basic" "get_coord" "error" "MCQ_free-form_alignment"

# "basic" is the default analysis type for accuracy and disclosure metrics
# optionally run "get_coord" to get geolocated coordinates or load from cache
# "error" is for distance error and other geolocation utility metrics (which calls get_coord if not already geolocated)
# "MCQ_free-form_alignment" is for MCQ free-form alignment metrics (flag should be "heuristics", prompting_method should be "zs", and q7_only should be false)

if [ "$analysis_type" == "MCQ_free-form_alignment" ]; then
    prompting_method="zs"
    flag="heuristics"
    q7_only=false
fi


# Function to check if model is in api_models array
is_api_model() {
    local model=$1
    for api_model in "${api_models[@]}"; do
        if [[ "$model" == "$api_model" ]]; then
            return 0
        fi
    done
    return 1
}

for prompting_method in "${prompting_methods[@]}"; do
  for model in "${models[@]}"; do
    # Automatically set gen_method based on model type
    if is_api_model "$model"; then
        gen_method="api_gen"  # For api models 
    else
        gen_method="generate"  # For local open models
    fi
    
    # Construct task name based on whether it's Q7-only mode
    if [[ "$q7_only" == true ]]; then
        if [[ "$flag" == "heuristics" ]]; then
            task="${gen_method}_${prompting_method}_heuristics_q7-only"
        else
            task="${gen_method}_${prompting_method}_q7-only"
        fi
    else
        if [[ "$flag" == "heuristics" ]]; then
            task="${gen_method}_${prompting_method}_heuristics"
        else
            task="${gen_method}_${prompting_method}_${flag}"
        fi
    fi
        
    gold_path="data/annotation_labels.csv"
    
    # Route to organized subfolders based on flags
    if [[ "$flag" == "heuristics" ]]; then
        pred_path="evaluation/main/mcq/${task}_${model}_combined.json"
    else
        # free-form settings
        method_folder="$prompting_method"
        pred_path="evaluation/main/${method_folder}/responses/${task}_${model}_combined.json"
    fi
    
    echo "Evaluating: $pred_path against $gold_path"
    if [[ -f "$gold_path" && -f "$pred_path" ]]; then
        python src/eval.py \
            --gold_path "$gold_path" \
            --pred_path "$pred_path" \
            --analysis_type $analysis_type \
            --model_name $model
    else
        echo "Skipping: $gold_path or $pred_path does not exist." >&2
    fi
    echo "### End $model"
    echo " "
  done
done