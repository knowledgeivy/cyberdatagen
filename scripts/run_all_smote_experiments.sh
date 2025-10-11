#!/bin/bash
# Run all SMOTE experiments for 20 groups with 2 strategies
# Total: 20 groups × 2 strategies = 40 tasks

CONFIG="config/full_ceas08_smote_v1.yaml"
VARIANT="smote"

echo "=========================================="
echo "Starting SMOTE Baseline Experiments"
echo "Total: 20 groups × 2 strategies = 40 tasks"
echo "=========================================="
echo ""

# Counter for progress tracking
total_tasks=40
completed=0

# Loop through all groups and strategies
for strategy in within_group cross_group; do
    echo "=========================================="
    echo "Strategy: $strategy"
    echo "=========================================="

    for group_id in {0..19}; do
        completed=$((completed + 1))
        echo ""
        echo "[$completed/$total_tasks] Running: Strategy=$strategy, Group=$group_id"
        echo "------------------------------------------"

        python scripts/run_smote_experiments.py \
            --config "$CONFIG" \
            --variant "$VARIANT" \
            --strategy "$strategy" \
            --group_id "$group_id"

        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed: $strategy - Group $group_id"
        else
            echo "✗ Failed: $strategy - Group $group_id"
            echo "Continuing with next task..."
        fi

        echo "------------------------------------------"
    done
done

echo ""
echo "=========================================="
echo "All SMOTE experiments completed!"
echo "Completed: $completed/$total_tasks tasks"
echo "=========================================="
echo ""
echo "Results saved in: output/full_experiments/ceas08_smote/results/"
echo "Next step: Run statistical analysis to aggregate results"
