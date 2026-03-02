# Model Architectures

This document details the neural network architectures available for motor fault detection.

## 1. Improved CNN (Default)

The **ImprovedMotorFaultCNN** uses residual connections and channel attention to capture complex patterns in the multi-channel time-series data.

### Architecture Diagram

```mermaid
graph TD
    Input["Input (Batch, 6, 200)"] --> Conv1["Initial Conv (7x7, 64)"]
    Conv1 --> BN1["BatchNorm + ReLU"]
    BN1 --> MP1["MaxPool (Stride 2)"]

    MP1 --> RB1["ResBlock 1 (128)"]
    RB1 --> CA1["Channel Attention"]

    CA1 --> RB2["ResBlock 2 (256)"]
    RB2 --> CA2["Channel Attention"]

    CA2 --> RB3["ResBlock 3 (512)"]
    RB3 --> CA3["Channel Attention"]

    CA3 --> DualPool["Global Avg + Max Pool"]
    DualPool --> FC["Classifier (1024 → 256 → 128 → 4)"]
    FC --> Output["Fault Prediction"]
```

---

## 2. Transformer

The **MotorFaultTransformer** applies self-attention across the time dimension, allowing the model to focus on specific temporal events (spikes, shifts) more effectively than traditional convolutions.

### Architecture Diagram

```mermaid
graph TD
    Input["Input (Batch, 6, 200)"] --> Trans["Transpose (Batch, 200, 6)"]
    Trans --> Proj["Input Projection (6 → d_model)"]
    Proj --> PE["Positional Encoding"]

    subgraph "Encoder Stack (N Layers)"
        PE --> Attn["Multi-Head Self-Attention"]
        Attn --> FFN["Feed-Forward Network"]
    end

    FFN --> GAP["Global Average Pooling"]
    GAP --> Classifier["Classifier (d_model → 64 → 4)"]
    Classifier --> Output["Fault Prediction"]
```

---

## 3. Comparison Summary

| Feature         | Improved CNN                       | Transformer                      |
| --------------- | ---------------------------------- | -------------------------------- |
| **Core Layer**  | Residual 1D Conv                   | Multi-Head Attention             |
| **Parameters**  | ~2.1 Million                       | ~0.4 Million (Configurable)      |
| **Strength**    | Excellent local feature extraction | Captures long-range dependencies |
| **Suitability** | General motor fault patterns       | Complex, non-periodic transients |

---
