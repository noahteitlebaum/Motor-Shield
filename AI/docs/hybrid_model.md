# HybridCNNTransformer Architecture

## Diagram
```mermaid
---
config:
    layout: dagre
---
flowchart LR
 subgraph STEM["1D CNN STEM (Feature Extractor)"]
        direction TB
                Conv1["Conv1d k7 s2 p3\nOut: (48, 100)"]
                BN1["BatchNorm1d (48)"]
                GELU1["GELU"]
                Conv2["Conv1d k5 s2 p2\nOut: (128, 50)"]
                BN2["BatchNorm1d (128)"]
                GELU2["GELU"]
    end
 subgraph BLOCK["Transformer Layer (×3)"]
                MHA["Self-Attention nhead=4, d_model=128\n+ Add & LayerNorm"]
                FFN["FFN 128→256→128 + Dropout p=encoder_dropout\n+ Add & LayerNorm"]
    end
 subgraph TRANSFORMER["TRANSFORMER ENCODER"]
        direction TB
                PE["Positional Encoding + Dropout p=encoder_dropout\nOut: (50, 128)"]
                BLOCK
    end
 subgraph HEAD["CLASSIFIER HEAD"]
        direction TB
                Pool["Concat(AvgPool, MaxPool)\nShape: (Batch, 256)"]
                Lin1["Linear 256→128 + GELU + Dropout p=classifier_dropout"]
                Lin2["Linear 128→4 (logits)"]
    end
        Input[/INPUT (6, 200)/] --> Conv1
        Conv1 --> BN1 --> GELU1 --> Conv2 --> BN2 --> GELU2 --> ShapeConv{"Transpose\n(50, 128)"}
        ShapeConv --> PE
        PE --> MHA --> FFN --> Pool
        Pool --> Lin1 --> Lin2 --> Output(["Class probabilities (4)"])

        classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px
        classDef inputNode fill:#fff,stroke:#333,stroke-width:2px
        classDef outputNode fill:#333,color:#fff,stroke:#333,stroke-width:2px
        class Input inputNode
        class Output outputNode
```

## Hybrid CNN–Transformer Model — Detailed Explanation

### 1) Problem setting
- Multivariate time-series classification: input $200\times 6$, output 4 classes.
- Current split: Train 25{,}938 (class counts: 4{,}774 / 4{,}334 / 4{,}829 / 12{,}001), Val 2{,}358, Test 786.
- Goal: capture local transient signatures and long-range dependencies without leakage.

### 2) Architectural logic
Signal $\rightarrow$ CNN stem (local patterns, downsample to 50 tokens) $\rightarrow$ positional encoding $\rightarrow$ 3-layer transformer encoder $\rightarrow$ concat(avg,max) pooling $\rightarrow$ MLP head.

### 3) Components (code-accurate)
- **CNN stem:** Conv1d $6\to48$ ($k=7,s=2,p=3$), Conv1d $48\to128$ ($k=5,s=2,p=2$) with BatchNorm+GELU. Length: $200\to100\to50$.
- **Tokenization:** transpose to $(\text{batch}, 50, 128)$.
- **Positional encoding:** sinusoidal, dropout $p=0.15$ (encoder\_dropout).
- **Transformer encoder:** 3 layers, $d_{model}=128$, $n_{head}=4$, FFN $128\to256\to128$, dropout $p=0.15$, residual + LayerNorm.
- **Pooling:** concat(global avg, global max) $\Rightarrow 256$-D.
- **Classifier:** Linear $256\to128$ (GELU + dropout $p=0.25$), Linear $128\to4$ logits.

### 4) Training setup
- Optimizer: AdamW, $\text{lr}=3\times10^{-4}$, weight decay $1\times10^{-4}$.
- Scheduler: cosine with 1-epoch warmup, $\text{min\_lr\_ratio}=0.1$.
- Epochs: 5, batch size: 64, patience: 15.
- Loss: Cross-entropy with class weights (computed from train counts) and label smoothing $\epsilon=0.02$.
- Device: MPS (macOS). DataLoader workers: 0 on MPS to avoid overhead.

### 5) Leakage controls
- Scaling: fit StandardScaler on train only; apply to val/test.
- Split integrity: generation pipeline runs exact/near-duplicate checks, class overlap, and distribution checks (see `src/validation/validate_no_leakage.py`).
- Quick training-time sanity: no NaN/Inf; sample-based duplicate check passes.

### 6) Mathematical intuition
- **Attention:** $\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$ over 50 tokens captures long-range relations at reduced cost ($200\to50$ yields $16\times$ cheaper $O(n^2)$).
- **Pooling fusion:** $h = [\text{avg}(Z);\,\text{max}(Z)]$ stabilizes evidence across channels/time.
- **Label smoothing:** $y' = (1-\epsilon)y + \epsilon/4$ improves calibration and reduces overconfidence.

### 7) Performance (latest run)
- Val/Test: 100% top-1, macro-F1 1.0, macro-AUC 1.0 (see `artifacts/hybrid/metrics.json`).
- Training history and LR trace in `artifacts/hybrid/training_history.{png,csv}` show smooth cosine decay and convergence by epoch ~4–5.

### 8) Design rationale vs baselines
- **vs Improved CNN:** adds temporal self-attention for global context while keeping CNN locality and efficiency.
- **vs Plain Transformer:** CNN stem shortens sequence (50 tokens) and injects inductive bias, improving stability and compute.

### 9) Strengths and limitations
- **Strengths:**
        - Compact: ~464k parameters; fits on modest hardware (MPS/CPU) with batch 64.
        - Efficient: CNN stem shrinks length 200→50, making attention ~16× cheaper than full-length transformers.
        - Stable optimization: cosine LR + warmup; gradient clipping; AdamW with weight decay.
        - Robustness aids: label smoothing 0.02 + class weights mitigate imbalance and overconfidence.
        - Leak-safe preprocessing: scaler fit on train only; duplicate/distribution checks baked into pipeline.
        - Strong generalization: perfect macro metrics on held-out test in current run.
        - Clear inductive biases: locality (CNN), global context (attention), invariance (avg/max pooling).
- **Limitations:**
        - Attention cost is still $O(n^2)$ on 50 tokens; for longer windows, consider Performer/Linear attention or further downsampling.
        - No calibrated uncertainty head yet; add temperature scaling or MC dropout for deployment.
        - Needs ablations to quantify each component (stem, pooling choice, smoothing, scheduler).

### 11) How the model works (step-by-step)
1. **Input prep:** 200×6 window scaled with train-fitted StandardScaler, then transposed to channels-first for CNN.
2. **CNN stem:** Two strided convs extract local temporal-frequency patterns and reduce length to 50 while expanding channels to 128.
3. **Tokenization:** Transpose to (batch, 50, 128) so each timestep is a token embedding.
4. **Positional encoding:** Sinusoidal PE injected (dropout 0.15) to encode order without extra params.
5. **Transformer encoder (3 layers):**
         - Multi-head self-attention (4 heads) relates any timestep to any other across the 50-token sequence.
         - Feedforward 128→256→128 with dropout 0.15 refines token features.
         - Residual + LayerNorm stabilize depth and gradients.
6. **Pooling fusion:** Global average and max pool across time, then concatenate to a 256-D summary capturing both dominant and subtle signals.
7. **Classifier head:** Linear 256→128 with GELU + dropout 0.25, then Linear 128→4 logits; softmax yields class probabilities.
8. **Loss & reweighting:** Cross-entropy with class weights (derived from train counts) and label smoothing (0.02) to address imbalance and calibration.
9. **Scheduler:** Cosine decay with 1-epoch warmup; learning rate smoothly anneals, reducing late-epoch oscillations.
10. **Selection:** Best checkpoint by validation accuracy; metadata, metrics JSON, confusion matrix, and training curves saved per run.

### 10) Suggested future additions
- Ablations (drop stem / change pooling / vary depth), attention visualizations, confidence calibration (temperature scaling), and domain-shift tests.
