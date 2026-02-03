#!/bin/bash

# Wrapper script to run seed variation analysis
# This runs inference with 3 seeds for 9 models in both MCQ and free-form settings

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (two levels above this script directory)
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to project root directory
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Seed Variation Analysis"
echo "=========================================="
echo "This will run inference with seeds [1, 2, 3]"
echo "for 9 models in both MCQ and free-form settings"
echo "Total runs: 9 models × 3 seeds × 2 settings = 54"
echo "=========================================="
echo ""

# Configuration
IMAGE_DIR="data/images"
GOLD_FILE="data/annotation_labels.csv"
TASK_NAME="seed_variation"
OUT_DIR="evaluation/seed_variation"
USE_AZURE=true

# Optional: limit examples for testing (comment out for full run)
# MAX_EXAMPLES=50

# Build command
CMD="python scripts/seed_variation/run_seed_variations.py"
CMD="$CMD --image-dir $IMAGE_DIR"
CMD="$CMD --gold-file $GOLD_FILE"
CMD="$CMD --task-name $TASK_NAME"
CMD="$CMD --out-dir $OUT_DIR"

if [ "$USE_AZURE" = true ]; then
    CMD="$CMD --use-azure"
fi

# Uncomment for testing with limited examples
# if [ -n "$MAX_EXAMPLES" ]; then
#     CMD="$CMD --max-examples $MAX_EXAMPLES"
# fi

echo "Running command:"
echo "$CMD"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Run the script
$CMD

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "Check the report at: $OUT_DIR/seed_variation_report.txt"
echo "=========================================="

