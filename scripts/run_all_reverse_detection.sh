#!/bin/bash
# Batch runner for cross-model reverse detection experiments (with parallel execution)
# Total: 12 configurations × 3 ratios × 20 groups × 2 classifiers = 1,440 experiments

# Note: Not using 'set -e' to allow parallel jobs to complete even if some fail

echo "=========================================="
echo "Cross-Model Reverse Detection Experiments"
echo "=========================================="
echo ""

# Configuration
OUTPUT_DIR="output/reverse_detection"
CONFIG_GPT="config/full_ceas08_gpt41mini_v1.yaml"
CONFIG_CLAUDE="config/full_ceas08_claude35haiku_v1.yaml"

# Parallel execution settings
MAX_PARALLEL_JOBS=6  # Number of experiments to run in parallel

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Start time
START_TIME=$(date +%s)

# Counter
TOTAL_CONFIGS=36  # 12 configs × 3 ratios
CURRENT_CONFIG=0

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

#######################################
# GPT-4.1-mini testing (trained on Claude)
#######################################
echo "=========================================="
echo "GPT-4.1-mini Testing (trained on Claude)"
echo "=========================================="

for PROMPT in original strong weak; do
    for STRATEGY in within_group cross_group; do
        for RATIO in 0 50 100; do
            ((CURRENT_CONFIG++))

            # Wait for available slot
            wait_for_slot

            JOB_NAME="GPT-$PROMPT-$STRATEGY-r$RATIO"
            LOG_FILE="$OUTPUT_DIR/logs/${JOB_NAME}.log"

            echo ""
            echo "[$CURRENT_CONFIG/$TOTAL_CONFIGS] Launching: $JOB_NAME (training on Claude)"

            # Run in background
            (
                python scripts/reverse_detection_experiment.py \
                    --testing_method gpt41mini \
                    --testing_prompt $PROMPT \
                    --testing_strategy $STRATEGY \
                    --training_ratio $RATIO \
                    --config $CONFIG_GPT \
                    --classifiers svm random_forest \
                    --output_dir $OUTPUT_DIR \
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

#######################################
# Claude-3.5-Haiku testing (trained on GPT)
#######################################
echo ""
echo "=========================================="
echo "Claude-3.5-Haiku Testing (trained on GPT)"
echo "=========================================="

for PROMPT in original strong weak; do
    for STRATEGY in within_group cross_group; do
        for RATIO in 0 50 100; do
            ((CURRENT_CONFIG++))

            # Wait for available slot
            wait_for_slot

            JOB_NAME="Claude-$PROMPT-$STRATEGY-r$RATIO"
            LOG_FILE="$OUTPUT_DIR/logs/${JOB_NAME}.log"

            echo ""
            echo "[$CURRENT_CONFIG/$TOTAL_CONFIGS] Launching: $JOB_NAME (training on GPT)"

            # Run in background
            (
                python scripts/reverse_detection_experiment.py \
                    --testing_method claude35haiku \
                    --testing_prompt $PROMPT \
                    --testing_strategy $STRATEGY \
                    --training_ratio $RATIO \
                    --config $CONFIG_CLAUDE \
                    --classifiers svm random_forest \
                    --output_dir $OUTPUT_DIR \
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

# Wait for all remaining jobs to complete
echo ""
echo "=========================================="
echo "Waiting for all jobs to complete..."
echo "=========================================="

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

echo ""
echo "=========================================="
echo "All Cross-Model Reverse Detection Experiments Completed!"
echo "=========================================="
echo ""
echo "Execution Summary:"
echo "  - Total configurations: $TOTAL_CONFIGS (2 methods × 3 prompts × 2 strategies × 3 ratios)"
echo "  - Total experiments: 1,440 (36 configs × 20 groups × 2 classifiers)"
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
echo "Cross-model Pairing:"
echo "  - GPT testing → trained on Claude synthetic data"
echo "  - Claude testing → trained on GPT synthetic data"
echo ""
echo "Next Steps:"
echo "1. Check logs for any failures: cat $OUTPUT_DIR/logs/summary.log"
echo "2. Verify all 36 result files exist in $OUTPUT_DIR/results/"
echo "3. Validate cross-model pairing in results"
echo "4. Run analysis script: python scripts/analyze_reverse_detection.py"
echo "5. Generate plots and integrate into paper"
echo ""
