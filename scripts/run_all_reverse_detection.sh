#!/bin/bash
# Batch runner for cross-model reverse detection experiments
# Total: 12 configurations × 3 ratios × 20 groups × 2 classifiers = 1,440 experiments

set -e  # Exit on error

echo "=========================================="
echo "Cross-Model Reverse Detection Experiments"
echo "=========================================="
echo ""

# Configuration
OUTPUT_DIR="output/reverse_detection"
CONFIG_GPT="configs/ceas08_gpt41mini_config.yaml"
CONFIG_CLAUDE="configs/ceas08_claude35haiku_config.yaml"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start time
START_TIME=$(date +%s)

# Counter
TOTAL_CONFIGS=36  # 12 configs × 3 ratios
CURRENT_CONFIG=0

echo "Running experiments for all LLM methods, prompts, strategies, and training ratios..."
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
            echo ""
            echo "[$CURRENT_CONFIG/$TOTAL_CONFIGS] Testing: GPT-4.1-mini | Prompt: $PROMPT | Strategy: $STRATEGY | Ratio: $RATIO%"
            echo "Training source: Claude-3.5-Haiku (opposite model)"
            echo "----------------------------------------"

            python scripts/reverse_detection_experiment.py \
                --testing_method gpt41mini \
                --testing_prompt $PROMPT \
                --testing_strategy $STRATEGY \
                --training_ratio $RATIO \
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
            echo ""
            echo "[$CURRENT_CONFIG/$TOTAL_CONFIGS] Testing: Claude-3.5-Haiku | Prompt: $PROMPT | Strategy: $STRATEGY | Ratio: $RATIO%"
            echo "Training source: GPT-4.1-mini (opposite model)"
            echo "----------------------------------------"

            python scripts/reverse_detection_experiment.py \
                --testing_method claude35haiku \
                --testing_prompt $PROMPT \
                --testing_strategy $STRATEGY \
                --training_ratio $RATIO \
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
done

# End time and duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "All Cross-Model Reverse Detection Experiments Completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/results/"
echo "Total configurations: $TOTAL_CONFIGS (12 configs × 3 ratios)"
echo "Total experiments: 1,440 (36 configs × 20 groups × 2 classifiers)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Cross-model pairing:"
echo "  GPT testing → trained on Claude"
echo "  Claude testing → trained on GPT"
echo ""
echo "Next steps:"
echo "1. Verify all 36 result files exist"
echo "2. Validate cross-model pairing in results"
echo "3. Run analysis script: python scripts/analyze_reverse_detection.py"
echo "4. Generate plots and integrate into paper"
echo ""
