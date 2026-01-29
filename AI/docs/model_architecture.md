# Model selection and architecture details

This document explains the model options and the exact architecture used in the latest training run. Code lives in [processing/train_model.py](processing/train_model.py).

## 1) Model options

### Simple CNN (baseline)
- Lightweight 1D CNN for fast iteration and low compute.
- Use when you need very fast inference or a quick baseline.

### Improved CNN (current default)
- Residual blocks with batch normalization
- Channel attention for feature reweighting
- Dual pooling (avg + max) before the classifier

The improved model is defined in `ImprovedMotorFaultCNN` in [processing/train_model.py](processing/train_model.py).

## 2) Improved CNN layer layout (current default)

Input shape: (batch, 6 channels, 200 time steps)

1) Initial block
- Conv1d: 6 → 64, kernel 7, padding 3
- BatchNorm1d + ReLU
- MaxPool1d (stride 2)

2) Residual blocks + attention
- ResBlock: 64 → 128, stride 2 + ChannelAttention(128)
- ResBlock: 128 → 256, stride 2 + ChannelAttention(256)
- ResBlock: 256 → 512 + ChannelAttention(512)

3) Global pooling and classifier
- AdaptiveAvgPool1d + AdaptiveMaxPool1d (concatenate)
- Linear 1024 → 256 → 128 → num_classes
- BatchNorm + ReLU + Dropout between FC layers

## 3) Training configuration (latest run)

Run: 10 epochs, batch size 32, learning rate 0.001

Training components:
- Optimizer: AdamW
- LR scheduler: ReduceLROnPlateau
- Early stopping: patience 15
- Gradient clipping: max norm 1.0
- Loss: CrossEntropyLoss

## 4) Real results from the latest run

From the last training run in this workspace:
- Train: 30,261 samples
- Validation: 6,484 samples
- Test: 6,485 samples
- Improved model parameters: 2,113,860

Classification report (test set):
- Faulty_Control_Switch: precision 1.00, recall 1.00, f1 1.00 (support 1,206)
- Faulty_Open_Circuit: precision 1.00, recall 1.00, f1 1.00 (support 1,206)
- Faulty_Short_Circuit: precision 1.00, recall 1.00, f1 1.00 (support 1,131)
- Healthy: precision 1.00, recall 1.00, f1 1.00 (support 2,942)
- Overall accuracy: 1.00

## 5) Output artifacts

Model and evaluation outputs are stored under the artifacts folder by default:
- artifacts/motor_fault_model.pth
- artifacts/model_metadata.pkl
- artifacts/confusion_matrix.png
- artifacts/training_history.png
- artifacts/visualizations (detailed plots)

## 6) How to choose a model

- Use the improved model for accuracy-focused workflows.
- Use the simple model when you need faster iteration or lower compute.

Switch models with the --model_type flag in [processing/train_model.py](processing/train_model.py).
