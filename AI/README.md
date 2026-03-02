# AI Module Configuration & Usage Guide

This document details the configuration options and usage patterns for the Motor Fault Detection AI module.

## 1. Training (`train_model.py`)

The `train_model.py` script is the single entry point for training all supported model architectures (CNNs and Transformer).

### Usage

```bash
python AI/train_model.py [ARGUMENTS]
```

### General Arguments

| Argument       | Default                 | Description                                                                                                                                            |
| -------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--dataset`    | `artifacts/dataset.npz` | Path to the preprocessed dataset file.                                                                                                                 |
| `--output_dir` | `artifacts`             | Directory where models and training artifacts will be saved. Models are saved in subdirectories based on `model_type` (e.g., `artifacts/transformer`). |
| `--device`     | Auto-detect             | Compute device to use: `cpu`, `cuda`, `mps` (Mac). Default behavior prefers CUDA > MPS > CPU.                                                          |
| `--epochs`     | `20`                    | Number of training epochs.                                                                                                                             |
| `--batch_size` | `32`                    | Batch size for training.                                                                                                                               |
| `--lr`         | `0.001`                 | Learning rate for the AdamW optimizer.                                                                                                                 |
| `--patience`   | `15`                    | Early stopping patience (epochs without improvement).                                                                                                  |
| `--dropout`    | `0.5`                   | Dropout rate applied to model layers.                                                                                                                  |

### Model Selection

| Argument       | Choices                                       | Description                     |
| -------------- | --------------------------------------------- | ------------------------------- |
| `--model_type` | `improved` (default), `simple`, `transformer` | Selects the model architecture. |

### Transformer-Specific Arguments

These arguments only apply when `--model_type transformer` is used.

| Argument            | Default | Description                                                                  |
| ------------------- | ------- | ---------------------------------------------------------------------------- |
| `--d_model`         | `128`   | The number of expected features in the encoder inputs (embedding dimension). |
| `--nhead`           | `4`     | The number of heads in the multiheadattention models.                        |
| `--num_layers`      | `3`     | The number of sub-encoder-layers in the encoder.                             |
| `--dim_feedforward` | `256`   | The dimension of the feedforward network model.                              |

### Examples

**Train default Improved CNN:**

```bash
python AI/train_model.py --model_type improved
```

**Train Transformer on CPU with batch size 64:**

```bash
python AI/train_model.py --model_type transformer --device cpu --batch_size 64
```

**Train Transformer with custom architecture:**

```bash
python AI/train_model.py --model_type transformer --d_model 64 --num_layers 6 --nhead 8
```

---

## 2. Inference (`inference.py`)

The `inference.py` script is used to load a trained model and make predictions on new data.

### Usage

```bash
python AI/inference.py [ARGUMENTS]
```

### Arguments

| Argument        | Default           | Description                                                          |
| --------------- | ----------------- | -------------------------------------------------------------------- |
| `--model`       | Required          | Path to the trained model file (`.pth`).                             |
| `--metadata`    | Required          | Path to the model metadata file (`.pkl`) corresponding to the model. |
| `--csv`         | `None`            | Path to a CSV file containing raw motor data to predict on.          |
| `--window_size` | `200`             | Size of the rolling window for prediction (must match training).     |
| `--stride`      | `10`              | Stride step between windows.                                         |
| `--output`      | `predictions.csv` | Output file for prediction results.                                  |
| `--device`      | Auto-detect       | Compute device: `cpu`, `cuda`, `mps`.                                |

### Example

```bash
python AI/inference.py \
  --model artifacts/transformer/motor_fault_model.pth \
  --metadata artifacts/transformer/model_metadata.pkl \
  --csv files/Healthy/data_log.csv \
  --device cpu
```

---

## 3. Ensemble (`ensemble_model.py`)

The `ensemble_model.py` script allows training multiple models and combining their predictions for higher robustness.

### Usage

```bash
python AI/ensemble_model.py [ARGUMENTS]
```

### Arguments

| Argument        | Default     | Description                                                                 |
| --------------- | ----------- | --------------------------------------------------------------------------- |
| `--mode`        | `train`     | Operation mode: `train` (train new ensemble) or `evaluate` (test existing). |
| `--n_models`    | `5`         | Number of models to train for the ensemble.                                 |
| `--voting`      | `soft`      | Voting method: `soft` (average probabilities) or `hard` (majority vote).    |
| `--model_paths` | `None`      | List of model paths to load for evaluation mode.                            |
| `--device`      | Auto-detect | Compute device.                                                             |

### Example

**Train an ensemble of 5 models:**

```bash
python AI/ensemble_model.py --mode train --n_models 5 --output_dir artifacts/ensemble
```

**Evaluate an existing ensemble:**

```bash
python AI/ensemble_model.py --mode evaluate \
  --model_paths artifacts/ensemble/model_1.pth artifacts/ensemble/model_2.pth \
  --output_dir artifacts/ensemble
```
