#!/bin/bash
# SMOTE Baseline Experiments - Parallel Execution
# Runs SMOTE/ADASYN experiments for all groups in parallel

set -e

CONFIG="config/full_ceas08_smote_v1.yaml"
VARIANT="smote"  # or "adasyn"
N_GROUPS=20
MAX_PARALLEL=8  # Maximum parallel jobs

echo "=========================================="
echo "SMOTE Baseline Experiments - Parallel"
echo "Configuration: ${CONFIG}"
echo "Variant: ${VARIANT}"
echo "Groups: ${N_GROUPS}"
echo "Max Parallel: ${MAX_PARALLEL}"
echo "Start Time: $(date)"
echo "=========================================="

# Function to run experiment for one group and strategy
run_experiment() {
    local group_id=$1
    local strategy=$2

    echo "[Group ${group_id} - ${strategy}] Starting..."

    python scripts/run_smote_experiments.py \
        --config ${CONFIG} \
        --variant ${VARIANT} \
        --strategy ${strategy} \
        --group_id ${group_id} \
        --resume

    if [ $? -eq 0 ]; then
        echo "[Group ${group_id} - ${strategy}] ✓ Completed"
    else
        echo "[Group ${group_id} - ${strategy}] ✗ Failed"
    fi
}

# Export function for parallel execution
export -f run_experiment
export CONFIG VARIANT

# Strategy 1: Within-group
echo ""
echo "=========================================="
echo "Strategy: within_group"
echo "=========================================="

for GROUP_ID in $(seq 0 $((N_GROUPS-1))); do
    # Wait if too many jobs running
    while [ $(jobs -r | wc -l) -ge ${MAX_PARALLEL} ]; do
        sleep 1
    done

    run_experiment ${GROUP_ID} "within_group" &
done

# Wait for all within_group jobs to complete
wait
echo "✓ within_group strategy completed"

# Strategy 2: Cross-group
echo ""
echo "=========================================="
echo "Strategy: cross_group"
echo "=========================================="

for GROUP_ID in $(seq 0 $((N_GROUPS-1))); do
    # Wait if too many jobs running
    while [ $(jobs -r | wc -l) -ge ${MAX_PARALLEL} ]; do
        sleep 1
    done

    run_experiment ${GROUP_ID} "cross_group" &
done

# Wait for all cross_group jobs to complete
wait
echo "✓ cross_group strategy completed"

echo ""
echo "=========================================="
echo "All SMOTE experiments completed!"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Merge results: python scripts/merge_smote_results.py"
echo "2. Run analysis: python scripts/step5_statistical_analysis.py --config ${CONFIG}"
echo "3. Generate plots: python scripts/step6_visualization.py --config ${CONFIG}"
