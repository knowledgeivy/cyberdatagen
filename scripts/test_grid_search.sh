#!/bin/bash
# Test Grid Search Script - Quick Test with Small Parameter Set
# Uses default data path (group 0, r0 baseline)

echo "========================================"
echo "Testing Hyperparameter Grid Search"
echo "========================================"
echo "This is a quick test to verify the grid search code works correctly."
echo "Using: group 0, r0 baseline training data"
echo "Cross-validation: 3 folds (faster than 5)"
echo ""

# Test with a smaller subset first
echo "Running grid search (this may take 5-10 minutes)..."
echo ""

python scripts/hyperparameter_grid_search.py \
    --data_path data/full_experiments/ceas08_gpt41mini/datasets/within_group/train_original_r0_g0_t0.csv.gz \
    --output_dir output/hyperparameter_search/test \
    --cv 3 \
    --classifiers svm random_forest

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Grid Search Test Completed Successfully!"
    echo "========================================"
    echo ""
    echo "Results saved to: output/hyperparameter_search/test/"
    echo ""
    echo "Check the following files:"
    echo "  - grid_search_results.json (detailed results)"
    echo "  - best_hyperparameters.json (best params for each classifier)"
    echo "  - svm_grid_search_heatmap.png (SVM visualization)"
    echo "  - rf_grid_search_heatmap.png (RF visualization)"
    echo "  - hyperparameter_comparison.csv (comparison table)"
    echo ""
    echo "Next step: Review results and decide if you want to run full grid search with 5-fold CV"
else
    echo ""
    echo "❌ Grid Search Test FAILED"
    echo "Check logs at: output/hyperparameter_search/test/grid_search.log"
    exit 1
fi
