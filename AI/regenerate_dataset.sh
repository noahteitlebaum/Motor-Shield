#!/bin/bash
# Quick setup script to regenerate dataset with leak-free configuration

set -euo pipefail

echo "Motor Shield - Leak-Free Dataset Generation"
echo "=============================================="
echo ""

# Check if we're in the right directory
if [ ! -d "src/core" ]; then
    echo "ERROR: Must run from AI/ directory"
    exit 1
fi

VENV_PATH="../.venv/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at ../.venv"
    exit 1
fi

# Activate project venv as requested
source "$VENV_PATH"
PYTHON_BIN="$(which python)"
echo "Using Python: $PYTHON_BIN"

# Backup old dataset if it exists
if [ -f "artifacts/dataset.npz" ]; then
    echo "Backing up old dataset..."
    mv artifacts/dataset.npz artifacts/dataset_old_$(date +%Y%m%d_%H%M%S).npz
fi

echo "Generating leak-free dataset..."
echo "   - Train: 10 augmentations (full pipeline)"
echo "   - Val: 2 augmentations (light noise only)"
echo "   - Test: 0 augmentations (original data)"
echo ""

if "$PYTHON_BIN" src/core/generate_dataset.py \
    --train_augmentations 10 \
    --val_augmentations 2 \
    --test_augmentations 0 \
    --output artifacts/dataset.npz; then
    echo ""
    echo "PASS: Dataset generated successfully!"
    echo ""
    echo "Running validation checks..."
    if "$PYTHON_BIN" src/validation/validate_no_leakage.py; then
        echo ""
        echo "SUCCESS: Dataset is ready for training."
        echo ""
        echo "Next steps:"
        echo "  1. Train model:"
        echo "     python train_model.py --model_type improved --epochs 50"
        echo ""
        echo "  2. Review training metrics for healthy train/test gap (<5%)"
        echo ""
        echo "  3. Check docs/data_leakage_prevention.md for detailed info"
    else
        echo ""
        echo "WARNING: Validation found issues. Please review output above."
    fi
else
    echo ""
    echo "ERROR: Dataset generation failed. Check errors above."
    exit 1
fi
