# Data preprocessing and noise/augmentation

This document describes the exact preprocessing pipeline and noise/augmentation steps used in the processing module. Code lives in [processing/generate_dataset.py](processing/generate_dataset.py) and the train-time scaling happens in [processing/train_model.py](processing/train_model.py).

## 1) Raw inputs

Each CSV file contains columns:
- time
- Vdc
- da, db, dc (duty cycles)
- Ia, Ib, Ic (phase currents)

## 2) Derived phase voltages

Phase voltages are derived from duty cycles and DC bus voltage:

$$
va_N = (2da - 1) \cdot \frac{Vdc}{2},\quad
vb_N = (2db - 1) \cdot \frac{Vdc}{2},\quad
vc_N = (2dc - 1) \cdot \frac{Vdc}{2}
$$

$$
\bar{v}_n = \frac{va_N + vb_N + vc_N}{3}
$$

$$
va = va_N - \bar{v}_n,\quad vb = vb_N - \bar{v}_n,\quad vc = vc_N - \bar{v}_n
$$

These are computed in `DataProcessor.derive_phase_voltages()` in [processing/generate_dataset.py](processing/generate_dataset.py).

## 3) Windowing

Sampling and window parameters (from [processing/generate_dataset.py](processing/generate_dataset.py)):
- Sample rate: 5,000 Hz
- Window length: 40 ms → 200 samples
- Overlap: 50% → stride of 100 samples

Windows are extracted as sliding segments of shape (200, 6) with features:
- Ia, Ib, Ic, va, vb, vc

### Labeling logic
- Healthy files: all windows labeled Healthy
- Faulty files: windows before fault time labeled Healthy, windows after fault time labeled Faulty_*
- Windows that cross the fault time are discarded

## 4) Noise and augmentation

All augmentations are applied per file during dataset generation. Base augmentations are always applied; advanced ones are probabilistic.

### Base augmentations (always applied)
1) Gaussian noise
- Currents: 3–5% per-sample noise
- Voltage: 0.5–1% per-sample noise

2) Gain variation (per file)
- Currents: ±1–3%
- Voltage: ±1–3%

3) Offset drift (linear drift over time)
- Currents: ±1–3% of mean magnitude
- Voltage: ±1–2% of mean magnitude

### Advanced augmentations (probabilistic)
1) Random spikes
- Injects occasional transient spikes into current channels

2) Harmonic distortion
- Adds small 3rd and 5th harmonics to currents

3) Time warping (approx.)
- Applies slight temporal distortion to mimic timing variations

These are implemented in `DataAugmentor` methods in [processing/generate_dataset.py](processing/generate_dataset.py).

## 5) Scaling and train/val/test split

Scaling happens after dataset creation, during training:
- 70/15/15 split (train/val/test) in `preprocess_data()` in [processing/train_model.py](processing/train_model.py)
- StandardScaler is fit on the training set only
- Each sample is scaled per feature (flattened to (N×T, F), then reshaped)

## 6) Output artifacts

By default, the generated dataset is saved as:
- artifacts/dataset.npz

You can override the output location with the --output flag in [processing/generate_dataset.py](processing/generate_dataset.py).
