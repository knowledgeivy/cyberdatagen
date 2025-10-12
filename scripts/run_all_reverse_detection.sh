#!/bin/bash
# Batch runner for all reverse detection experiments
# Total: 14 configurations × 20 groups × 2 classifiers = 560 experiments

set -e  # Exit on error

echo "=========================================="
echo "Reverse Detection Experiment - Batch Runner"
echo "=========================================="
echo ""

# Configuration
OUTPUT_DIR="output/reverse_detection"
CONFIG_GPT="configs/ceas08_gpt41mini_config.yaml"
CONFIG_CLAUDE="configs/ceas08_claude35haiku_config.yaml"
CONFIG_SMOTE="configs/ceas08_smote_config.yaml"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start time
START_TIME=$(date +%s)

# Counter
TOTAL_CONFIGS=14
CURRENT_CONFIG=0

echo "Running experiments for all methods, prompts, and strategies..."
echo ""

#######################################
# GPT-4.1-mini experiments (6 configs)
#######################################
echo "=========================================="
echo "Running GPT-4.1-mini experiments..."
echo "=========================================="

for PROMPT in original strong weak; do
    for STRATEGY in within_group cross_group; do
        ((CURRENT_CONFIG++))
        echo ""
        echo "[$CURRENT_CONFIG/$TOTAL_CONFIGS] GPT-4.1-mini | Prompt: $PROMPT | Strategy: $STRATEGY"
        echo "----------------------------------------"

        python scripts/reverse_detection_experiment.py \
            --method gpt41mini \
            --prompt $PROMPT \
            --strategy $STRATEGY \
            --config $CONFIG_GPT \
            --classifiers svm random_forest \
            --output_dir $OUTPUT_DIR

        if [ $? -eq 0 ]; then
            echo "✓ Completed successfully"
        else
            echo "✗ Failed"
            exit 1
        fi
    done
done

#######################################
# Claude-3.5-Haiku experiments (6 configs)
#######################################
echo ""
echo "=========================================="
echo "Running Claude-3.5-Haiku experiments..."
echo "=========================================="

for PROMPT in original strong weak; do
    for STRATEGY in within_group cross_group; do
        ((CURRENT_CONFIG++))
        echo ""
        echo "[$CURRENT_CONFIG/$TOTAL_CONFIGS] Claude-3.5-Haiku | Prompt: $PROMPT | Strategy: $STRATEGY"
        echo "----------------------------------------"

        python scripts/reverse_detection_experiment.py \
            --method claude35haiku \
            --prompt $PROMPT \
            --strategy $STRATEGY \
            --config $CONFIG_CLAUDE \
            --classifiers svm random_forest \
            --output_dir $OUTPUT_DIR

        if [ $? -eq 0 ]; then
            echo "✓ Completed successfully"
        else
            echo "✗ Failed"
            exit 1
        fi
    done
done

#######################################
# SMOTE experiments (2 configs)
#######################################
echo ""
echo "=========================================="
echo "Running SMOTE experiments..."
echo "=========================================="

for STRATEGY in within_group cross_group; do
    ((CURRENT_CONFIG++))
    echo ""
    echo "[$CURRENT_CONFIG/$TOTAL_CONFIGS] SMOTE | Strategy: $STRATEGY"
    echo "----------------------------------------"

    python scripts/reverse_detection_experiment.py \
        --method smote \
        --strategy $STRATEGY \
        --config $CONFIG_SMOTE \
        --classifiers svm random_forest \
        --output_dir $OUTPUT_DIR

    if [ $? -eq 0 ]; then
        echo "✓ Completed successfully"
    else
        echo "✗ Failed"
        exit 1
    fi
done

# End time and duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "All Reverse Detection Experiments Completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/results/"
echo "Total configurations: $TOTAL_CONFIGS"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Next steps:"
echo "1. Verify all result files exist"
echo "2. Run analysis script: python scripts/analyze_reverse_detection.py"
echo "3. Generate plots and integrate into paper"
echo ""
