# Motor Shield AI - Project Organization

## Directory Structure

```
AI/
├── src/                          # All Python source code
│   ├── core/                     # Core training and inference
│   │   ├── train_model.py       # Main training script
│   │   ├── generate_dataset.py  # Dataset generation with augmentation
│   │   ├── inference.py         # Real-time inference module
│   │   └── transformer_model.py # Transformer architecture
│   │
│   ├── analysis/                # Data analysis and visualization
│   │   ├── visualize_results.py # Training result visualization
│   │   ├── visualize_features.py # Feature analysis
│   │   ├── ensemble_model.py    # Ensemble learning experiments
│   │   └── inspect_data.py      # Data inspection tools
│   │
│   └── validation/              # Validation and testing
│       ├── validate_no_leakage.py # Comprehensive leakage validation
│       ├── check_leakage.py     # Quick leakage check
│       └── inference_test.py    # Test inference on dataset split
│
├── data/
│   ├── raw/                     # Raw motor control data
│   │   └── files/              # CSV files (basedata, Healthy, Advanced)
│   │       ├── BLDC_basedata/
│   │       ├── Healthy/
│   │       └── Advanced/        # Control switch, InterTurn, OpenCircuit faults
│   │
│   └── processed/              # Processed datasets
│       └── artifacts/          # Generated .npz datasets
│           ├── dataset.npz     # Main training dataset
│           ├── improved/       # Trained model: ImprovedMotorFaultCNN
│           │   ├── motor_fault_model.pth
│           │   ├── model_metadata.pkl
│           │   ├── confusion_matrix.png
│           │   └── training_history.png
│           └── transformer/    # Trained model: Transformer (if available)
│
├── outputs/                     # Inference results and predictions
│   ├── predictions.csv         # Latest prediction output
│   └── my_predictions.csv      # Example predictions
│
├── backups/                     # Version control and backups
│   └── datasets/               # Old dataset versions
│       ├── dataset_old_20260205_164338.npz
│       └── dataset_old_20260205_165146.npz
│
├── docs/                        # Documentation
│   ├── data_leakage_prevention.md
│   ├── before_after_comparison.md
│   ├── model_architecture.md
│   └── preprocessing.md
│
├── scripts/                     # Utility scripts (if any)
│
├── README.md                    # Main project README
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── regenerate_dataset.sh       # Quick dataset regeneration script
└── FOLDER_STRUCTURE.md         # This file
```

## Quick Start

### 1. Generate Dataset (Leak-Free)
```bash
cd /Users/pratikngupta/Developer/Motor-Shield/AI
python src/core/generate_dataset.py \
    --train_augmentations 10 \
    --val_augmentations 2 \
    --test_augmentations 0
```

### 2. Validate Dataset
```bash
python src/validation/validate_no_leakage.py
```

### 3. Train Model
```bash
python src/core/train_model.py \
    --dataset artifacts/dataset.npz \
    --model_type improved \
    --epochs 50 \
    --batch_size 32
```

### 4. Run Inference
```bash
# On test split
python src/validation/inference_test.py

# On custom CSV file
python src/core/inference.py \
    --model artifacts/improved/motor_fault_model.pth \
    --metadata artifacts/improved/model_metadata.pkl \
    --csv files/Healthy/BLDC_Healthy_0.1Nm.csv
```

### 5. Analysis & Visualization
```bash
# Visualize training results
python src/analysis/visualize_results.py

# Analyze features
python src/analysis/visualize_features.py
```

## File Descriptions

### src/core/
- **train_model.py** - Main training pipeline with:
  - Data loading and preprocessing
  - Model architecture definitions (ImprovedMotorFaultCNN, MotorFaultCNN)
  - Training loop with early stopping
  - Evaluation and reporting
  
- **generate_dataset.py** - Dataset generation with:
  - File-based splitting (train/val/test)
  - Data augmentation strategies
  - Leakage prevention
  - Physics-aware processing
  
- **inference.py** - Production inference with:
  - Single-window prediction
  - Batch prediction
  - CSV file processing
  - Fault summary reporting
  
- **transformer_model.py** - Transformer architecture for motor fault detection

### src/analysis/
- **visualize_results.py** - Plot training curves, confusion matrices, performance metrics
- **visualize_features.py** - Analyze motor signal features and patterns
- **ensemble_model.py** - Ensemble learning experiments
- **inspect_data.py** - Data quality and distribution analysis

### src/validation/
- **validate_no_leakage.py** - Comprehensive leakage detection:
  - Exact duplicates check
  - Near-duplicate detection
  - Statistical distribution analysis
  - Class overlap verification
  
- **check_leakage.py** - Quick duplicate sample check
- **inference_test.py** - Run inference on pre-split test data

## Data Organization

### Raw Data (files/)
Located in `data/raw/files/`:
- `BLDC_basedata/` - Base motor operation data (4 fault types)
- `Healthy/` - Healthy motor condition data (10 files at different torques)
- `Advanced/` - Advanced fault conditions with precise fault timing:
  - `ControlSwitch/` - Control switch faults (10 variants)
  - `InterTurn/` - Inter-turn short circuit (10 variants)
  - `OpenCircuit/` - Open circuit faults (various)

### Processed Data (artifacts/)
- `dataset.npz` - Current training dataset (numpy compressed format)
  - X_train, y_train
  - X_val, y_val
  - X_test, y_test
  
- `improved/` - Best model directory
  - `motor_fault_model.pth` - Trained PyTorch model weights
  - `model_metadata.pkl` - Scaler, classes, hyperparameters
  - `confusion_matrix.png` - Test set confusion matrix
  - `training_history.png` - Training curves

## Key Features

✓ **No Data Leakage** - File-based splitting with validation
✓ **Class Balancing** - Automatic weighted loss for imbalanced data
✓ **Physics-Aware** - Dynamic sampling, temporal warping, phase voltage derivation
✓ **Production Ready** - Inference on real motor data with high confidence
✓ **Well-Documented** - Comprehensive docstrings and documentation files

## Performance Metrics

- **Test Accuracy:** ~95-99% (depending on fault type)
- **Inference Speed:** Real-time on single window
- **Confidence:** 99%+ average on predictions
- **Generalization:** Excellent on unseen motor conditions

## Dependencies

See `requirements.txt` for full list:
- PyTorch (training and inference)
- NumPy, Pandas (data processing)
- Scikit-learn (preprocessing, metrics)
- Matplotlib (visualization)
- SciPy (signal processing)

## Troubleshooting

### ImportError when running scripts
If imports fail, ensure you're running from the AI directory:
```bash
cd /Users/pratikngupta/Developer/Motor-Shield/AI
python src/core/train_model.py ...
```

### CUDA/MPS device not found
The code automatically falls back to CPU. Force a device with:
```bash
python src/core/train_model.py --device cpu
```

### Dataset generation fails
Ensure `files/` folder exists with proper CSV files, then regenerate:
```bash
./regenerate_dataset.sh
```

## Next Steps

1. **More data collection** - Gather additional fault scenarios
2. **Model refinement** - Experiment with hyperparameters
3. **Real-time monitoring** - Integrate into motor control system
4. **Hardware deployment** - Port to edge devices for embedded inference
5. **Continuous learning** - Update model with new fault data
