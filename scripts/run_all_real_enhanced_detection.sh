#!/bin/bash
# Run All Real-Enhanced Detection Experiments
# Training: Real ham (900) + Real spam (100, 200, or 300)
# Testing: Real ham (900) + Synthetic spam (100% from target LLM)

CONFIG="configs/ceas08_config.yaml"
OUTPUT_DIR="output/real_enhanced_detection"

# Experimental factors
TESTING_METHODS=("gpt41mini" "claude35haiku")
TESTING_PROMPTS=("original" "strong" "weak")
TESTING_STRATEGIES=("within_group" "cross_group")
REAL_SPAM_COUNTS=(100 200 300)

# Total configurations: 2 × 3 × 2 × 3 = 36
# Total experiments: 36 × 20 groups × 2 classifiers = 1,440

echo "========================================"
echo "Real-Enhanced Detection Experiments"
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
echo "Total Experiments: 1,440 (36 configs × 20 groups × 2 classifiers)"
echo "========================================"
echo ""

# Counter
total_configs=0
completed_configs=0
failed_configs=0

# Loop through all combinations
for testing_method in "${TESTING_METHODS[@]}"; do
    for testing_prompt in "${TESTING_PROMPTS[@]}"; do
        for testing_strategy in "${TESTING_STRATEGIES[@]}"; do
            for real_spam_count in "${REAL_SPAM_COUNTS[@]}"; do
                total_configs=$((total_configs + 1))

                echo ""
                echo "========================================"
                echo "Configuration $total_configs/36"
                echo "========================================"
                echo "Testing Method: $testing_method"
                echo "Testing Prompt: $testing_prompt"
                echo "Testing Strategy: $testing_strategy"
                echo "Real Spam Count: $real_spam_count"
                echo "========================================"
                echo ""

                # Check if results already exist
                results_file="$OUTPUT_DIR/results/${testing_method}_${testing_prompt}_${testing_strategy}_count${real_spam_count}_real_enhanced_results.json"

                if [ -f "$results_file" ]; then
                    echo "⚠️  Results already exist: $results_file"
                    echo "Skipping this configuration. Delete the file to re-run."
                    completed_configs=$((completed_configs + 1))
                    continue
                fi

                # Run experiment
                python scripts/real_enhanced_detection_experiment.py \
                    --testing_method "$testing_method" \
                    --testing_prompt "$testing_prompt" \
                    --testing_strategy "$testing_strategy" \
                    --real_spam_count $real_spam_count \
                    --config "$CONFIG" \
                    --output_dir "$OUTPUT_DIR"

                if [ $? -eq 0 ]; then
                    echo "✅ Configuration $total_configs completed successfully"
                    completed_configs=$((completed_configs + 1))
                else
                    echo "❌ Configuration $total_configs FAILED"
                    failed_configs=$((failed_configs + 1))

                    # Ask user if they want to continue
                    read -p "Continue with next configuration? (y/n) " -n 1 -r
                    echo
                    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                        echo "Stopping experiments."
                        exit 1
                    fi
                fi

                # Progress summary
                echo ""
                echo "Progress: $completed_configs completed, $failed_configs failed out of $total_configs run"
                echo ""
            done
        done
    done
done

# Final summary
echo ""
echo "========================================"
echo "All Real-Enhanced Detection Experiments Completed!"
echo "========================================"
echo "Total Configurations: $total_configs"
echo "Completed: $completed_configs"
echo "Failed: $failed_configs"
echo "========================================"
echo ""

if [ $failed_configs -eq 0 ]; then
    echo "✅ All experiments completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run analysis: python scripts/analyze_real_enhanced_detection.py"
    echo "2. Generate visualizations: python scripts/visualize_real_enhanced_detection.py"
    echo "3. Update paper with results"
else
    echo "⚠️  Some experiments failed. Please check logs in $OUTPUT_DIR/logs/"
    exit 1
fi
