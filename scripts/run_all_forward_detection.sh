#!/bin/bash
# Run All Forward Detection Experiments (Parallel)
# Main experiments: GPT-4.1-mini and Claude-3.5-Haiku
# Strategy: cross_group only (optimized)
# Classifier: SVM with linear kernel only

# Configuration
OUTPUT_DIR_GPT="output/full_experiments/ceas08_gpt41mini/results"
OUTPUT_DIR_CLAUDE="output/full_experiments/ceas08_claude35haiku/results"
CONFIG_GPT="config/full_ceas08_gpt41mini_v1.yaml"
CONFIG_CLAUDE="config/full_ceas08_claude35haiku_v1.yaml"

# Experimental factors
PROMPTS=("original" "strong" "weak")
STRATEGY="cross_group"  # Using only cross_group for best performance

# Parallel execution settings
MAX_PARALLEL_JOBS=6  # Number of experiments to run in parallel

# Total configurations: 2 models × 3 prompts × 1 strategy = 6
# Total experiments: 6 × 20 groups × 11 ratios = 1,320 (SVM only)

echo "========================================"
echo "Forward Detection Experiments (Parallel)"
echo "========================================"
echo "Models: GPT-4.1-mini, Claude-3.5-Haiku"
echo "Prompts: ${PROMPTS[@]}"
echo "Strategy: $STRATEGY (cross_group only)"
echo "Classifier: SVM with linear kernel"
echo ""
echo "Total Configurations: 6"
echo "Total Experiments: 1,320 (6 configs × 20 groups × 11 ratios)"
echo "Parallel Jobs: $MAX_PARALLEL_JOBS"
echo "========================================"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR_GPT"
mkdir -p "$OUTPUT_DIR_CLAUDE"
mkdir -p "logs/full_experiments/ceas08_gpt41mini"
mkdir -p "logs/full_experiments/ceas08_claude35haiku"

# Start time
START_TIME=$(date +%s)

# Counter
TOTAL_CONFIGS=6
CURRENT_CONFIG=0
completed_configs=0
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
                ((completed_configs++))
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
# GPT-4.1-mini Experiments
#######################################
echo "========================================"
echo "GPT-4.1-mini Experiments"
echo "========================================"

for PROMPT in "${PROMPTS[@]}"; do
    ((CURRENT_CONFIG++))

    # Check if all results for this config exist
    all_exist=true
    for GROUP_ID in {0..19}; do
        results_file="$OUTPUT_DIR_GPT/full_ceas08_gpt41mini_v1_${PROMPT}_${STRATEGY}_group${GROUP_ID}_results.json"
        if [ ! -f "$results_file" ]; then
            all_exist=false
            break
        fi
    done

    if [ "$all_exist" = true ]; then
        echo "[$CURRENT_CONFIG/$TOTAL_CONFIGS] ⚠️  Skipping (all groups exist): GPT-${PROMPT}-${STRATEGY}"
        ((skipped_configs++))
        continue
    fi

    # Wait for available slot
    wait_for_slot

    JOB_NAME="GPT-${PROMPT}-${STRATEGY}"
    LOG_FILE="logs/full_experiments/ceas08_gpt41mini/${JOB_NAME}.log"

    echo "[$CURRENT_CONFIG/$TOTAL_CONFIGS] Launching: $JOB_NAME (20 groups × 11 ratios = 220 experiments)"

    # Run in background - process all groups sequentially within this job
    (
        for GROUP_ID in {0..19}; do
            # Check if this group already exists
            results_file="$OUTPUT_DIR_GPT/full_ceas08_gpt41mini_v1_${PROMPT}_${STRATEGY}_group${GROUP_ID}_results.json"
            if [ -f "$results_file" ]; then
                echo "  Group $GROUP_ID already exists, skipping..." >> "$LOG_FILE"
                continue
            fi

            echo "  Processing Group $GROUP_ID..." >> "$LOG_FILE"
            python scripts/step4_classification.py \
                --config "$CONFIG_GPT" \
                --strategy "$STRATEGY" \
                --prompt "$PROMPT" \
                --group_id $GROUP_ID \
                --classifiers svm \
                --resume \
                >> "$LOG_FILE" 2>&1

            if [ $? -eq 0 ]; then
                echo "  ✓ Group $GROUP_ID completed" >> "$LOG_FILE"
            else
                echo "  ✗ Group $GROUP_ID failed" >> "$LOG_FILE"
            fi
        done

        # Summary
        if [ $? -eq 0 ]; then
            echo "✓ $JOB_NAME completed successfully" >> "logs/forward_detection_summary.log"
        else
            echo "✗ $JOB_NAME failed" >> "logs/forward_detection_summary.log"
        fi
    ) &

    # Track the job
    PIDS+=($!)
    JOB_NAMES+=("$JOB_NAME")
    echo "  → Job started (PID: $!)"
done

#######################################
# Claude-3.5-Haiku Experiments
#######################################
echo ""
echo "========================================"
echo "Claude-3.5-Haiku Experiments"
echo "========================================"

for PROMPT in "${PROMPTS[@]}"; do
    ((CURRENT_CONFIG++))

    # Check if all results for this config exist
    all_exist=true
    for GROUP_ID in {0..19}; do
        results_file="$OUTPUT_DIR_CLAUDE/full_ceas08_claude35haiku_v1_${PROMPT}_${STRATEGY}_group${GROUP_ID}_results.json"
        if [ ! -f "$results_file" ]; then
            all_exist=false
            break
        fi
    done

    if [ "$all_exist" = true ]; then
        echo "[$CURRENT_CONFIG/$TOTAL_CONFIGS] ⚠️  Skipping (all groups exist): Claude-${PROMPT}-${STRATEGY}"
        ((skipped_configs++))
        continue
    fi

    # Wait for available slot
    wait_for_slot

    JOB_NAME="Claude-${PROMPT}-${STRATEGY}"
    LOG_FILE="logs/full_experiments/ceas08_claude35haiku/${JOB_NAME}.log"

    echo "[$CURRENT_CONFIG/$TOTAL_CONFIGS] Launching: $JOB_NAME (20 groups × 11 ratios = 220 experiments)"

    # Run in background - process all groups sequentially within this job
    (
        for GROUP_ID in {0..19}; do
            # Check if this group already exists
            results_file="$OUTPUT_DIR_CLAUDE/full_ceas08_claude35haiku_v1_${PROMPT}_${STRATEGY}_group${GROUP_ID}_results.json"
            if [ -f "$results_file" ]; then
                echo "  Group $GROUP_ID already exists, skipping..." >> "$LOG_FILE"
                continue
            fi

            echo "  Processing Group $GROUP_ID..." >> "$LOG_FILE"
            python scripts/step4_classification.py \
                --config "$CONFIG_CLAUDE" \
                --strategy "$STRATEGY" \
                --prompt "$PROMPT" \
                --group_id $GROUP_ID \
                --classifiers svm \
                --resume \
                >> "$LOG_FILE" 2>&1

            if [ $? -eq 0 ]; then
                echo "  ✓ Group $GROUP_ID completed" >> "$LOG_FILE"
            else
                echo "  ✗ Group $GROUP_ID failed" >> "$LOG_FILE"
            fi
        done

        # Summary
        if [ $? -eq 0 ]; then
            echo "✓ $JOB_NAME completed successfully" >> "logs/forward_detection_summary.log"
        else
            echo "✗ $JOB_NAME failed" >> "logs/forward_detection_summary.log"
        fi
    ) &

    # Track the job
    PIDS+=($!)
    JOB_NAMES+=("$JOB_NAME")
    echo "  → Job started (PID: $!)"
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
TOTAL_JOBS=$(wc -l < "logs/forward_detection_summary.log" 2>/dev/null || echo "0")
SUCCESS_JOBS=$(grep -c "✓" "logs/forward_detection_summary.log" 2>/dev/null || echo "0")
FAILED_JOBS=$(grep -c "✗" "logs/forward_detection_summary.log" 2>/dev/null || echo "0")

# Final summary
echo ""
echo "========================================"
echo "All Forward Detection Experiments Completed!"
echo "========================================"
echo ""
echo "Execution Summary:"
echo "  - Total configurations: $TOTAL_CONFIGS (2 models × 3 prompts × 1 strategy)"
echo "  - Strategy: cross_group only (best performance)"
echo "  - Classifier: SVM with linear kernel"
echo "  - Skipped (existing): $skipped_configs"
echo "  - Completed: $completed_configs"
echo "  - Parallel jobs: $MAX_PARALLEL_JOBS"
echo "  - Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Job Status:"
echo "  - Successful: $SUCCESS_JOBS"
echo "  - Failed: $FAILED_JOBS"
echo "  - Total: $TOTAL_JOBS"
echo ""
echo "Output Locations:"
echo "  - GPT Results: $OUTPUT_DIR_GPT"
echo "  - Claude Results: $OUTPUT_DIR_CLAUDE"
echo "  - Logs: logs/full_experiments/"
echo "  - Summary: logs/forward_detection_summary.log"
echo ""
echo "Next Steps:"
echo "1. Check logs for any failures: cat logs/forward_detection_summary.log"
echo "2. Run reverse detection: ./scripts/run_all_reverse_detection.sh"
echo "3. Run real-enhanced detection: ./scripts/run_all_real_enhanced_detection.sh"
echo "4. Generate analysis and plots"
echo ""
