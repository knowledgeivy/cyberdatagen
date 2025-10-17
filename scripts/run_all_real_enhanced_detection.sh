#!/bin/bash
# Run All Real-Enhanced Detection Experiments (Parallel)
# Training: Real ham (900) + Real spam (100, 150, or 200)
# Testing: Real ham (900) + Synthetic spam (100% from target LLM)

# Note: Not using 'set -e' to allow parallel jobs to complete even if some fail

CONFIG="config/full_ceas08_gpt41mini_v1.yaml"
OUTPUT_DIR="output/real_enhanced_detection"

# Experimental factors
TESTING_METHODS=("gpt41mini" "claude35haiku")
TESTING_PROMPTS=("original" "strong" "weak")
TESTING_STRATEGIES=("within_group" "cross_group")
REAL_SPAM_COUNTS=(100 150 200)

# Parallel execution settings
MAX_PARALLEL_JOBS=6  # Number of experiments to run in parallel

# Total configurations: 2 × 3 × 2 × 3 = 36
# Total experiments: 36 × 20 groups × 1 classifier (SVM only) = 720

echo "========================================"
echo "Real-Enhanced Detection Experiments (Parallel)"
echo "========================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Experimental Factors:"
echo "  Testing Methods: ${TESTING_METHODS[@]}"
echo "  Testing Prompts: ${TESTING_PROMPTS[@]}"
echo "  Testing Strategies: ${TESTING_STRATEGIES[@]}"
echo "  Real Spam Counts: ${REAL_SPAM_COUNTS[@]}"
echo ""
echo "Total Configurations: 36"
echo "Total Experiments: 720 (36 configs × 20 groups × 1 classifier)"
echo "Parallel Jobs: $MAX_PARALLEL_JOBS"
echo "========================================"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR/results"
mkdir -p "$OUTPUT_DIR/logs"

# Start time
START_TIME=$(date +%s)

# Counter
total_configs=0
completed_configs=0
failed_configs=0
skipped_configs=0

# Job tracking
declare -a PIDS=()
declare -a JOB_NAMES=()

# Function to wait for any job to complete
wait_for_slot() {
    while [ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                echo "  → Job completed: ${JOB_NAMES[$i]}"
                unset PIDS[$i]
                unset JOB_NAMES[$i]
            fi
        done
        PIDS=("${PIDS[@]}")  # Reindex array
        JOB_NAMES=("${JOB_NAMES[@]}")
        sleep 2
    done
}

echo "Running experiments in parallel (max $MAX_PARALLEL_JOBS concurrent jobs)..."
echo ""

# Loop through all combinations
for testing_method in "${TESTING_METHODS[@]}"; do
    for testing_prompt in "${TESTING_PROMPTS[@]}"; do
        for testing_strategy in "${TESTING_STRATEGIES[@]}"; do
            for real_spam_count in "${REAL_SPAM_COUNTS[@]}"; do
                total_configs=$((total_configs + 1))

                # Check if results already exist
                results_file="$OUTPUT_DIR/results/${testing_method}_${testing_prompt}_${testing_strategy}_count${real_spam_count}_real_enhanced_results.json"

                if [ -f "$results_file" ]; then
                    echo "[$total_configs/36] ⚠️  Skipping (exists): ${testing_method}-${testing_prompt}-${testing_strategy}-count${real_spam_count}"
                    skipped_configs=$((skipped_configs + 1))
                    continue
                fi

                # Wait for available slot
                wait_for_slot

                JOB_NAME="${testing_method}-${testing_prompt}-${testing_strategy}-count${real_spam_count}"
                LOG_FILE="$OUTPUT_DIR/logs/${JOB_NAME}.log"

                echo "[$total_configs/36] Launching: $JOB_NAME"

                # Run in background
                (
                    python scripts/real_enhanced_detection_experiment.py \
                        --testing_method "$testing_method" \
                        --testing_prompt "$testing_prompt" \
                        --testing_strategy "$testing_strategy" \
                        --real_spam_count $real_spam_count \
                        --config "$CONFIG" \
                        --output_dir "$OUTPUT_DIR" \
                        > "$LOG_FILE" 2>&1

                    if [ $? -eq 0 ]; then
                        echo "✓ $JOB_NAME completed successfully" >> "$OUTPUT_DIR/logs/summary.log"
                    else
                        echo "✗ $JOB_NAME failed" >> "$OUTPUT_DIR/logs/summary.log"
                    fi
                ) &

                # Track the job
                PIDS+=($!)
                JOB_NAMES+=("$JOB_NAME")
                echo "  → Job started (PID: $!)"
            done
        done
    done
done

# Wait for all remaining jobs to complete
echo ""
echo "========================================"
echo "Waiting for all jobs to complete..."
echo "========================================"

for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "All parallel jobs completed!"

# End time and duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Check completion status
TOTAL_JOBS=$(wc -l < "$OUTPUT_DIR/logs/summary.log" 2>/dev/null || echo "0")
SUCCESS_JOBS=$(grep -c "✓" "$OUTPUT_DIR/logs/summary.log" 2>/dev/null || echo "0")
FAILED_JOBS=$(grep -c "✗" "$OUTPUT_DIR/logs/summary.log" 2>/dev/null || echo "0")

# Final summary
echo ""
echo "========================================"
echo "All Real-Enhanced Detection Experiments Completed!"
echo "========================================"
echo ""
echo "Execution Summary:"
echo "  - Total configurations: $total_configs (2 methods × 3 prompts × 2 strategies × 3 counts)"
echo "  - Skipped (existing): $skipped_configs"
echo "  - Parallel jobs: $MAX_PARALLEL_JOBS"
echo "  - Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Job Status:"
echo "  - Successful: $SUCCESS_JOBS"
echo "  - Failed: $FAILED_JOBS"
echo "  - Total: $TOTAL_JOBS"
echo ""
echo "Output Locations:"
echo "  - Results: $OUTPUT_DIR/results/"
echo "  - Logs: $OUTPUT_DIR/logs/"
echo "  - Summary: $OUTPUT_DIR/logs/summary.log"
echo ""
echo "Next Steps:"
echo "1. Check logs for any failures: cat $OUTPUT_DIR/logs/summary.log"
echo "2. Run analysis: python scripts/analyze_real_enhanced_detection.py"
echo "3. Generate visualizations: python scripts/visualize_real_enhanced_detection.py"
echo "4. Update paper with results"
echo ""
