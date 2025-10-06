#!/bin/bash
# Auto-run all 4 strategies sequentially
# This script will run overnight and complete all classification experiments

CONFIG="config/full_ceas08_gpt41mini_v1.yaml"
STRATEGIES=("within_group" "cross_group" "real_fixed_random_synthetic" "full_random")

echo "=========================================="
echo "Starting full classification experiment"
echo "Date: $(date)"
echo "Config: $CONFIG"
echo "Strategies: ${STRATEGIES[@]}"
echo "=========================================="

for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Starting strategy: $strategy"
    echo "Start time: $(date)"
    echo "=========================================="

    poetry run python scripts/step4_classification.py \
        --config "$CONFIG" \
        --strategy "$strategy" \
        --resume

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ Strategy $strategy completed successfully"
        echo "Completion time: $(date)"
    else
        echo "✗ Strategy $strategy failed with exit code $exit_code"
        echo "Failure time: $(date)"
        echo "Continuing to next strategy..."
    fi

    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All strategies completed!"
echo "End time: $(date)"
echo "=========================================="
