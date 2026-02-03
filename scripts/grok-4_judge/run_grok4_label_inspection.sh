#!/bin/bash

# Re-run granularity label mapping using grok-4-fast-reasoning on label_inspection samples
# Processes the 640 samples from label_inspection_gpt-4.1-mini.json
# Usage: ./scripts/grok-4_judge/run_grok4_label_inspection.sh [--env-file PATH] [--input-file PATH] [--output-file PATH]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

ENV_FILE=".env_new"
INPUT_FILE="/nethome/ryang396/flash/GeoPrivGuard/benchmark/experiments/results/all/label_inspection_gpt-4.1-mini.json"
OUTPUT_FILE="/nethome/ryang396/flash/GeoPrivGuard/benchmark/experiments/results/all/label_inspection_grok4-fast-reasoning.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --input-file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--env-file PATH] [--input-file PATH] [--output-file PATH]"
            exit 1
            ;;
    esac
done

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: Environment file '$ENV_FILE' not found."
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file '$INPUT_FILE' not found."
    exit 1
fi

echo "Using environment file: $ENV_FILE"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"

# Count samples in input file
sample_count=$(python3 -c "import json; data = json.load(open('$INPUT_FILE')); print(len(data.get('samples', [])))" 2>/dev/null || echo "0")
echo "Processing $sample_count samples with grok-4-fast-reasoning"

echo ""
echo "=========================================="
echo "Running granularity mapping with grok-4-fast-reasoning"
echo "=========================================="

# Run granularity remapping
export DOTENV_PATH="$ENV_FILE"
python3 scripts/grok-4_judge/rerun_granularity_label_inspection.py \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --judge-model "grok-4-fast-reasoning" \
    --env-file "$ENV_FILE"

if [[ $? -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "Granularity remapping complete"
    echo "=========================================="
    echo "Output saved to: $OUTPUT_FILE"
else
    echo ""
    echo "ERROR: Failed to process samples"
    exit 1
fi

