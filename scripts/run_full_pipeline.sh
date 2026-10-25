#!/bin/bash
# Full Pipeline: NAS Search + Train Both Approaches

echo "=========================================="
echo "Heart-Lung Sound Separation - Full Pipeline"
echo "=========================================="

# 1. Run NAS search for both approaches
echo -e "\n[1/4] Running NAS search for Approach 1: Raw Audio..."
python3 nas_search/run_nas_search.py --approach raw_audio --search_epochs 50

echo -e "\n[2/4] Running NAS search for Approach 2: Feature-based..."
python3 nas_search/run_nas_search.py --approach feature_based --search_epochs 50

# 2. Train best models on full dataset
echo -e "\n[3/4] Training both approaches on full dataset..."
python3 training/train_both_approaches.py

# 3. Summary
echo -e "\n[4/4] Pipeline Complete!"
echo "Models saved in models/ directory"
echo "STM32 code in stm32_code/ directory"
echo "=========================================="
