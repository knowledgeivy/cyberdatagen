#!/bin/bash
# Check progress of SMOTE experiments by counting completed result files

RESULTS_DIR="output/full_experiments/ceas08_smote/results"

echo "=========================================="
echo "SMOTE Experiments Progress"
echo "=========================================="
echo ""

# Count result files for each strategy
within_count=$(ls -1 "$RESULTS_DIR"/*_within_group_*.json 2>/dev/null | wc -l | tr -d ' ')
cross_count=$(ls -1 "$RESULTS_DIR"/*_cross_group_*.json 2>/dev/null | wc -l | tr -d ' ')
total_count=$((within_count + cross_count))

echo "Within-group strategy: $within_count/20 groups completed"
echo "Cross-group strategy:  $cross_count/20 groups completed"
echo ""
echo "Total progress: $total_count/40 tasks completed"
echo ""

# Calculate percentage
percentage=$((total_count * 100 / 40))
echo "Overall progress: $percentage%"
echo ""

# Show most recent result files
echo "Most recent completions:"
ls -lt "$RESULTS_DIR"/*.json 2>/dev/null | head -5 | awk '{print $9}' | xargs -I {} basename {}

echo ""
echo "=========================================="
