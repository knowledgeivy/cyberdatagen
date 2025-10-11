#!/bin/bash
# Run all SMOTE experiments in parallel
# Uses 4 concurrent workers to speed up execution

CONFIG="config/full_ceas08_smote_v1.yaml"
VARIANT="smote"
MAX_PARALLEL=4  # Number of concurrent jobs

echo "=========================================="
echo "Starting SMOTE Baseline Experiments (PARALLEL)"
echo "Total: 20 groups × 2 strategies = 40 tasks"
echo "Concurrent workers: $MAX_PARALLEL"
echo "=========================================="
echo ""

# Function to run a single experiment
run_experiment() {
    local strategy=$1
    local group_id=$2

    echo "[$(date +%H:%M:%S)] Starting: $strategy - Group $group_id"

    python scripts/run_smote_experiments.py \
        --config "$CONFIG" \
        --variant "$VARIANT" \
        --strategy "$strategy" \
        --group_id "$group_id" \
        > "logs/smote_${strategy}_group${group_id}.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✓ Completed: $strategy - Group $group_id"
    else
        echo "[$(date +%H:%M:%S)] ✗ Failed: $strategy - Group $group_id"
    fi
}

# Export function for parallel execution
export -f run_experiment
export CONFIG VARIANT

# Counter for progress
total=0
completed=0

# Run all experiments with limited parallelism
for strategy in within_group cross_group; do
    echo "=========================================="
    echo "Strategy: $strategy"
    echo "=========================================="

    # Process groups in batches of MAX_PARALLEL
    for ((i=0; i<20; i+=MAX_PARALLEL)); do
        # Start MAX_PARALLEL jobs
        for ((j=0; j<MAX_PARALLEL && i+j<20; j++)); do
            group_id=$((i+j))
            run_experiment "$strategy" "$group_id" &
            total=$((total + 1))
        done

        # Wait for this batch to complete
        wait

        # Update progress
        completed=$((i + MAX_PARALLEL))
        if [ $completed -gt 20 ]; then
            completed=20
        fi
        echo ""
        echo "Progress: $completed/20 groups completed for $strategy"
        echo ""
    done
done

echo ""
echo "=========================================="
echo "All SMOTE experiments completed!"
echo "Total tasks: 40"
echo "=========================================="
echo ""
echo "Results saved in: output/full_experiments/ceas08_smote/results/"
