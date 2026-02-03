#!/bin/bash

# Re-run granularity label mapping using grok-4 on existing free-form results
# Looks for JSON files with ~640 samples and re-maps Q7-label using grok-4
# Usage: ./scripts/grok-4_judge/run_grok4_granularity_mapping.sh [--env-file PATH] [--input-dir DIR]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

ENV_FILE=".env_new"
INPUT_DIR="evaluation"
OUTPUT_SUFFIX="_grok4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--env-file PATH] [--input-dir DIR]"
            exit 1
            ;;
    esac
done

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: Environment file '$ENV_FILE' not found."
    exit 1
fi

echo "Using environment file: $ENV_FILE"
echo "Searching for free-form JSON files in: $INPUT_DIR"

# Find all free-form JSON files and check their size
found_files=()
while IFS= read -r file; do
    count=$(python3 -c "import json; print(len(json.load(open('$file'))))" 2>/dev/null || echo "0")
    if [[ "$count" -ge 600 && "$count" -le 680 ]]; then
        found_files+=("$file|$count")
        echo "Found: $file ($count entries)"
    fi
done < <(find "$INPUT_DIR" -type f -name "*free-form*.json" 2>/dev/null)

if [[ ${#found_files[@]} -eq 0 ]]; then
    echo "WARNING: No JSON files with ~640 entries found."
    echo "Searching for any free-form JSON files..."
    while IFS= read -r file; do
        count=$(python3 -c "import json; print(len(json.load(open('$file'))))" 2>/dev/null || echo "0")
        if [[ "$count" -gt 0 ]]; then
            found_files+=("$file|$count")
            echo "Found: $file ($count entries)"
        fi
    done < <(find "$INPUT_DIR" -type f -name "*free-form*.json" 2>/dev/null)
fi

if [[ ${#found_files[@]} -eq 0 ]]; then
    echo "ERROR: No free-form JSON files found in $INPUT_DIR"
    exit 1
fi

echo ""
echo "=========================================="
echo "Processing ${#found_files[@]} file(s) with grok-4-fast-reasoning"
echo "=========================================="

for file_info in "${found_files[@]}"; do
    IFS='|' read -r input_file count <<< "$file_info"
    echo ""
    echo "Processing: $input_file ($count entries)"
    echo "----------------------------------------"
    
    # Generate output filename
    input_path=$(dirname "$input_file")
    input_basename=$(basename "$input_file" .json)
    output_file="${input_path}/${input_basename}${OUTPUT_SUFFIX}.json"
    
    # Run granularity remapping
    export DOTENV_PATH="$ENV_FILE"
    python3 scripts/grok-4_judge/rerun_granularity_grok4.py \
        --input-file "$input_file" \
        --output-file "$output_file" \
        --judge-model "grok-4-fast-reasoning" \
        --env-file "$ENV_FILE"
    
    if [[ $? -eq 0 ]]; then
        echo "✓ Successfully processed: $output_file"
    else
        echo "✗ Failed to process: $input_file"
    fi
done

echo ""
echo "=========================================="
echo "Output files saved with suffix: ${OUTPUT_SUFFIX}"

