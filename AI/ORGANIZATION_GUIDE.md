# Motor Shield AI - Organization Summary

## Latest Change: train_model.py Moved to Root!

The main entry point for training has been moved from `src/core/` to the **AI root directory** for easier access. This is the primary script users interact with, so it makes sense to have it at the top level.

```bash
# Old way (still works from src/core/)
python src/core/train_model.py

# New way (recommended!)
python train_model.py
```

## What Changed

The AI folder has been reorganized from a flat structure into a well-organized module-based layout for better maintainability and clarity.

## New File Organization

### Before (Flat Structure)
```
AI/
├── train_model.py
├── generate_dataset.py
├── inference.py
├── validate_no_leakage.py
├── visualize_results.py
├── ... 11 more scripts
├── artifacts/
├── files/
└── docs/
```

### After (Organized Structure)
```
AI/
├── train_model.py            # Moved to root for easy access!
├── regenerate_dataset.sh
├── src/                      # All source code organized by function
│   ├── core/                 # Core ML pipeline
│   │   ├── generate_dataset.py
│   │   ├── inference.py
│   │   ├── transformer_model.py
│   │   └── (train_model.py moved to root)
│   ├── analysis/             # Data analysis & visualization
│   │   ├── visualize_results.py
│   │   ├── visualize_features.py
│   │   ├── ensemble_model.py
│   │   └── inspect_data.py
│   │
│   └── validation/           # Data validation & testing
│       ├── validate_no_leakage.py
│       ├── check_leakage.py
│       └── inference_test.py
│
├── data/
│   ├── raw/files/            # Raw CSV data
│   └── processed/artifacts/  # Processed datasets & models
│
├── outputs/                  # Prediction results & outputs
├── backups/datasets/         # Old dataset versions
├── docs/                     # Documentation
├── FOLDER_STRUCTURE.md       # Detailed folder guide
└── ... configuration files
```

## Key Improvements

 **Better Discoverability** - Scripts grouped by purpose (training, analysis, validation)
 **Cleaner Root** - No clutter from 11 scripts in root directory
 **Logical Organization** - Easy to find what you need
 **Version Control** - Backups and outputs separated
 **Scalability** - Easy to add new scripts to appropriate folders

## How to Use Scripts (Updated Paths)

### Core Training Pipeline
```bash
# Generate dataset
python src/core/generate_dataset.py --train_augmentations 10 --val_augmentations 2 --test_augmentations 0

# Train model (now in root for easy access!)
python train_model.py --dataset artifacts/dataset.npz --epochs 50

# Run inference
python src/core/inference.py --csv files/Healthy/BLDC_Healthy_0.1Nm.csv
```

### Analysis & Visualization
```bash
# Visualize training results
python src/analysis/visualize_results.py

# Analyze features
python src/analysis/visualize_features.py

# Run ensemble experiments
python src/analysis/ensemble_model.py
```

### Validation & Testing
```bash
# Comprehensive leakage validation
python src/validation/validate_no_leakage.py

# Test on dataset split
python src/validation/inference_test.py
```

## File Mapping Reference

| Old Path | New Path | Purpose |
|----------|----------|---------|
| train_model.py | **train_model.py** (root) | **Main entry point - moved to root** |
| generate_dataset.py | src/core/generate_dataset.py | Dataset generation |
| inference.py | src/core/inference.py | Real-world inference |
| transformer_model.py | src/core/transformer_model.py | Transformer architecture |
| validate_no_leakage.py | src/validation/validate_no_leakage.py | Leakage detection |
| check_leakage.py | src/validation/check_leakage.py | Quick leakage check |
| inference_test.py | src/validation/inference_test.py | Dataset testing |
| visualize_results.py | src/analysis/visualize_results.py | Result visualization |
| visualize_features.py | src/analysis/visualize_features.py | Feature analysis |
| inspect_data.py | src/analysis/inspect_data.py | Data inspection |
| ensemble_model.py | src/analysis/ensemble_model.py | Ensemble learning |
| regenerate_dataset.sh | regenerate_dataset.sh | Dataset script |
| README.md | README.md | Main documentation |

## Data File Organization

### Raw Data
- **Location:** data/raw/files/
- **Structure:**
  - `BLDC_basedata/` - Base operation
  - `Healthy/` - 10 healthy motor files
  - `Advanced/` - Advanced fault scenarios
    - ControlSwitch/ (10 files)
    - InterTurn/ (10 files)
    - OpenCircuit/ (various files)

### Processed Data
- **Location:** data/processed/artifacts/
- **Main:** `dataset.npz` - Training dataset
- **Models:** 
  - `improved/` - Best CNN model
  - `transformer/` - Transformer model (optional)

### Outputs & Backups
- **Location:** outputs/ - Prediction results
- **Backups:** backups/datasets/ - Old dataset versions

## Import Path Changes

### If you see ImportError

This may happen if a script can't find modules. The organization is fixed with sys.path manipulation in analysis scripts:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
```

**This is already applied to:**
- src/analysis/visualize_results.py
- src/analysis/ensemble_model.py

## Running from Different Directories

All scripts should be run from the AI directory:
```bash
cd /Users/pratikngupta/Developer/Motor-Shield/AI
python src/core/train_model.py ...
```

Or with full path:
```bash
python /Users/pratikngupta/Developer/Motor-Shield/AI/src/core/train_model.py ...
```

## Benefits This Organization Provides

1. **Clarity** - Know where to find each functionality
2. **Maintenance** - Similar code grouped together
3. **Scalability** - Easy to add new analysis, validation, or training scripts
4. **Collaboration** - Team members know where to look/add code
5. **Version Control** - Cleaner git history (backups/outputs not constantly changing root)
6. **Deployment** - src/ can be packaged separately for production

## Quick Reference

**Need to...**

- **Train a model?** → `python train_model.py`
- **Generate dataset?** → `python src/core/generate_dataset.py`
- **Do inference?** → `python src/core/inference.py`
- **Validate data?** → `python src/validation/validate_no_leakage.py`
- **Visualize results?** → `python src/analysis/visualize_results.py`
- **Analyze features?** → `python src/analysis/visualize_features.py`
- **Test on dataset?** → `python src/validation/inference_test.py`
- **View outputs?** → `outputs/` folder
- **See old versions?** → `backups/datasets/` folder

See `FOLDER_STRUCTURE.md` for detailed folder documentation!
