# Motor Shield

<div align="center">

<img src="frontend/public/MotorShieldLogo.png" alt="Motor Shield Logo" width="200"/>

**AI-Powered Motor Fault Detection & Health Monitoring System**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Next.js](https://img.shields.io/badge/Next.js-15.5-black)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue)](https://www.typescriptlang.org/)

[Features](#features) • [Installation](#installation) • [Training](#training-the-model) • [Usage](#usage) • [Documentation](#documentation)

</div>

---

## Overview

Motor Shield is a full-stack intelligent motor health monitoring system under development. It uses deep learning to detect and classify motor faults, featuring advanced AI models and an intuitive web dashboard designed to provide early fault detection, preventing costly equipment failures and downtime.

### Fault Detection Capabilities

Motor Shield can detect and classify the following motor conditions:

- **Healthy**: Normal motor operation
- **Open Circuit Fault**: Broken or disconnected motor phases
- **Inter-Turn Short Circuit**: Winding insulation breakdown
- **Control Switch Fault**: Controller or switching component failures

## Project Status

> **Note**: This project is currently under active development. The frontend dashboard and AI/ML backend are being developed independently and are **not yet connected**. The backend ML models are fully functional for training and inference via command-line tools, while the frontend provides UI components and visualization framework.

**Current State:**
- AI/ML Backend: Fully functional (training, inference, validation)
- Frontend Dashboard: UI components and pages implemented
- API Integration: In progress (backend API server and frontend integration pending)
- Real-time Data Streaming: Planned

## Features

### AI/ML Backend

- **Multiple Neural Architectures**: CNN with Residual Connections, Transformer-based models, and Ensemble methods
- **Real-time Fault Detection**: Processes motor sensor data with sliding window approach
- **High Accuracy**: 95%+ classification accuracy across fault types
- **Data Leakage Prevention**: Robust train/validation/test splits at the file level
- **Scalable Training Pipeline**: Support for CPU, CUDA, and Apple Silicon (MPS)

### Frontend Dashboard

- **Modern UI Components**: Dashboard, alerts, and analytics pages built with Next.js 15
- **Responsive Design**: Modern interface built with TailwindCSS and HeroUI
- **Dark Mode Support**: Theme switching for better user experience
- **Visualization Framework**: Ready for real-time motor health data integration
- **Coming Soon**: Live data integration with backend API

## Architecture

### Current Architecture (Development)

```
┌─────────────────────────────────────────────────────────────┐
│                Frontend (Next.js) - Port 3000               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Dashboard   │  │   Alerts     │  │   Analytics  │     │
│  │     (UI)     │  │     (UI)     │  │     (UI)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘

                    [Integration Layer - TBD]
                         │ (Coming Soon)
                         │ REST API / WebSocket

┌─────────────────────────────────────────────────────────────┐
│              AI Backend (Python CLI Tools)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Dataset    │→ │  ML Training │→ │  Inference   │     │
│  │  Generation  │  │  (PyTorch)   │  │   (CLI)      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Planned Architecture (Production)

Once integrated, the system will include:
- FastAPI/Flask backend server exposing ML inference endpoints
- WebSocket support for real-time motor data streaming
- Frontend API client for data fetching and visualization
- Database for storing motor health history and alerts

## Prerequisites

### Frontend Requirements
- **Node.js**: 18.x or higher
- **npm/pnpm/yarn**: Latest version

### Backend/AI Requirements
- **Python**: 3.8 or higher
- **pip**: Latest version
- **CUDA** (optional): For GPU acceleration

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/noahteitlebaum/Motor-Shield.git
cd Motor-Shield
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
# or
pnpm install
# or
yarn install

# For pnpm users: add to .npmrc if not already present
echo "public-hoist-pattern[]=*@heroui/*" >> .npmrc

# Run development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

### 3. Backend/AI Setup

```bash
# Navigate to AI directory
cd AI

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Training the Model

Motor Shield provides flexible training options with multiple neural architectures.

### Quick Start

```bash
# Generate dataset from raw CSV files
cd AI
python src/core/generate_dataset.py

# Train default Improved CNN model
python train_model.py --model_type improved --epochs 20 --batch_size 32
```

### Training Options

#### 1. **Improved CNN (Recommended)**

Best for: High accuracy with efficient training time

```bash
python train_model.py \
  --model_type improved \
  --epochs 20 \
  --batch_size 32 \
  --lr 0.001 \
  --dropout 0.5 \
  --device cuda
```

#### 2. **Transformer Model**

Best for: Capturing long-range temporal dependencies

```bash
python train_model.py \
  --model_type transformer \
  --d_model 128 \
  --nhead 4 \
  --num_layers 3 \
  --dim_feedforward 256 \
  --epochs 20 \
  --batch_size 32 \
  --device cuda
```

#### 3. **Ensemble Model**

Best for: Maximum accuracy through model averaging

```bash
python src/analysis/ensemble_model.py \
  --mode train \
  --n_models 5 \
  --output_dir artifacts/ensemble
```

### Training Arguments

| Argument       | Default                 | Description                                         |
|----------------|-------------------------|-----------------------------------------------------|
| `--dataset`    | `artifacts/dataset.npz` | Path to preprocessed dataset                        |
| `--output_dir` | `artifacts`             | Directory for saving models and artifacts           |
| `--model_type` | `improved`              | Model architecture: `improved`, `simple`, `transformer` |
| `--epochs`     | `20`                    | Number of training epochs                           |
| `--batch_size` | `32`                    | Training batch size                                 |
| `--lr`         | `0.001`                 | Learning rate                                       |
| `--dropout`    | `0.5`                   | Dropout rate                                        |
| `--patience`   | `15`                    | Early stopping patience                             |
| `--device`     | Auto-detect             | Compute device: `cpu`, `cuda`, `mps`                |

### Device Selection

The training script automatically selects the best available device:

```bash
# Force CPU training
python train_model.py --device cpu

# Use NVIDIA GPU (CUDA)
python train_model.py --device cuda

# Use Apple Silicon GPU (M1/M2/M3)
python train_model.py --device mps
```

### Dataset Generation

If you need to regenerate the dataset from raw CSV files:

```bash
# Place CSV files in AI/files/ directory following this structure:
# AI/files/
# ├── Healthy/
# │   └── *.csv
# └── Faulty/
#     ├── OpenCircuit/
#     │   └── *.csv
#     ├── InterTurn/
#     │   └── *.csv
#     └── ControlSwitch/
#         └── *.csv

# Generate dataset
python src/core/generate_dataset.py \
  --input_dir files \
  --output artifacts/dataset.npz \
  --window_size 200 \
  --stride 10
```

### Monitoring Training

Training outputs include:
- **Progress bars**: Real-time epoch progress
- **Metrics**: Training/validation loss and accuracy
- **Best model**: Auto-saved when validation loss improves
- **Artifacts**: Saved in `artifacts/<model_type>/`

Example output:
```
Epoch 1/20: 100%|██████████| 156/156 [00:15<00:00, 10.02it/s]
Train Loss: 0.4521, Train Acc: 83.45%
Val Loss: 0.3012, Val Acc: 89.23%
✓ New best model saved!
```

## Usage

### Making Predictions

```bash
# Run inference on new motor data
python src/core/inference.py \
  --model artifacts/improved/motor_fault_model.pth \
  --metadata artifacts/improved/model_metadata.pkl \
  --csv path/to/motor_data.csv \
  --output predictions.csv
```

### Validation & Analysis

```bash
# Validate dataset for data leakage
python src/validation/validate_no_leakage.py

# Visualize features
python src/analysis/visualize_features.py

# Analyze training results
python src/analysis/visualize_results.py --artifact_dir artifacts/improved
```

## Project Structure

```
Motor-Shield/
├── frontend/                 # Next.js web application
│   ├── app/                 # Next.js 15 app directory
│   │   ├── Dashboard/       # Dashboard page
│   │   ├── LearnMore/       # Information pages
│   │   ├── MeetTheTeam/     # Team information
│   │   └── components/      # React components
│   ├── components/          # Shared UI components
│   ├── config/              # Configuration files
│   ├── public/              # Static assets
│   └── styles/              # Global styles
│
├── AI/                      # Python ML backend
│   ├── src/
│   │   ├── core/           # Core training & inference
│   │   │   ├── generate_dataset.py
│   │   │   ├── train_model.py
│   │   │   ├── inference.py
│   │   │   └── transformer_model.py
│   │   ├── analysis/       # Model analysis tools
│   │   │   ├── ensemble_model.py
│   │   │   ├── visualize_features.py
│   │   │   └── visualize_results.py
│   │   └── validation/     # Validation utilities
│   │       ├── validate_no_leakage.py
│   │       └── check_leakage.py
│   ├── files/              # Training data (CSV)
│   ├── docs/               # Documentation
│   ├── artifacts/          # Trained models & outputs
│   ├── requirements.txt    # Python dependencies
│   └── README.md          # AI module documentation
│
├── .gitignore
├── LICENSE
└── README.md               # This file
```

## Technologies Used

### Frontend
- [Next.js 15](https://nextjs.org/) - React framework with App Router
- [TypeScript](https://www.typescriptlang.org/) - Type-safe JavaScript
- [TailwindCSS](https://tailwindcss.com/) - Utility-first CSS framework
- [HeroUI](https://heroui.com/) - Modern React component library
- [Framer Motion](https://www.framer.com/motion/) - Animation library

### Backend/AI
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [NumPy](https://numpy.org/) - Numerical computing
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [scikit-learn](https://scikit-learn.org/) - Machine learning utilities
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Data visualization

## Documentation

Detailed documentation is available in the following locations:

- **AI Module**: [AI/README.md](AI/README.md)
- **Model Architecture**: [AI/docs/model_architecture.md](AI/docs/model_architecture.md)
- **Data Preprocessing**: [AI/docs/preprocessing.md](AI/docs/preprocessing.md)
- **Data Leakage Prevention**: [AI/docs/data_leakage_prevention.md](AI/docs/data_leakage_prevention.md)

## Development

### Frontend Development

```bash
cd frontend

# Run dev server with Turbopack
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linter
npm run lint
```

### Backend Development

```bash
cd AI

# Run tests
python -m pytest tests/

# Check for data leakage
python src/validation/validate_no_leakage.py

# Regenerate dataset
./regenerate_dataset.sh
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Team

Visit our [Meet The Team](frontend/app/MeetTheTeam) page to learn more about the contributors.

## Acknowledgments

- Motor fault detection dataset and research methodology
- Open-source community for amazing tools and libraries
- All contributors who have helped improve this project

---

<div align="center">

[Report Bug](https://github.com/noahteitlebaum/Motor-Shield/issues) • [Request Feature](https://github.com/noahteitlebaum/Motor-Shield/issues)

</div>
