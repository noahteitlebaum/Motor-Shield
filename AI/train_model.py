import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import argparse
import pickle
from pathlib import Path

# Add src/core to path for transformer import
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'core'))

# from train_model import ImprovedMotorFaultCNN, MotorFaultCNN # Removed self-import
# Import Transformer (delayed import to avoid circular dependency issues if any remain, though they shouldn't)
# Actually, since we cleaned transformer_model.py, we can import directly.
# However, let's keep it clean.
try:
    from transformer_model import MotorFaultTransformer
except ImportError:
    pass # Will handle later or assuming it's available

# Set device

def get_device(device_name=None):
    """Get the appropriate device (cpu, cuda, mps)"""
    if device_name:
        return torch.device(device_name)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# Default device determination (will be overridden by main if args provided)
# Kept for backward compatibility if imported without calling get_device
device = get_device()

def load_data(filepath):
    """Load pre-split dataset from npz"""
    data = np.load(filepath)
    return {
        'X_train': data['X_train'], 'y_train': data['y_train'],
        'X_val': data['X_val'], 'y_val': data['y_val'],
        'X_test': data['X_test'], 'y_test': data['y_test']
    }

def preprocess_data_v2(data_dict):
    """Preprocess already split data with validation checks"""
    print("\n" + "="*60)
    print("PREPROCESSING & VALIDATION")
    print("="*60)
    
    # 1. Encode Labels (using all labels to ensure consistent mapping)
    le = LabelEncoder()
    all_y = np.concatenate([data_dict['y_train'], data_dict['y_val'], data_dict['y_test']])
    le.fit(all_y)
    
    y_train = le.transform(data_dict['y_train'])
    y_val = le.transform(data_dict['y_val'])
    y_test = le.transform(data_dict['y_test'])
    classes = le.classes_
    print(f"\nClasses found ({len(classes)}): {list(classes)}")
    
    # Print class distribution
    print("\nClass Distribution:")
    for split_name, y_data in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        unique, counts = np.unique(y_data, return_counts=True)
        print(f"  {split_name}:")
        for idx, count in zip(unique, counts):
            print(f"    {classes[idx]}: {count} ({100*count/len(y_data):.1f}%)")

    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']

    # 2. Data Validation
    print("\nData Shape Validation:")
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    
    print(f"  Train: {X_train.shape} ({X_train.nbytes / 1024**2:.1f} MB)")
    print(f"  Val:   {X_val.shape} ({X_val.nbytes / 1024**2:.1f} MB)")
    print(f"  Test:  {X_test.shape} ({X_test.nbytes / 1024**2:.1f} MB)")
    
    # Check for NaN or Inf
    for name, X in [('Train', X_train), ('Val', X_val), ('Test', X_test)]:
        if np.any(np.isnan(X)):
            raise ValueError(f"{name} data contains NaN values!")
        if np.any(np.isinf(X)):
            raise ValueError(f"{name} data contains Inf values!")
    print("  PASS: No NaN or Inf values detected")
    
    # Quick leakage check (sample-based)
    print("\nQuick Leakage Check:")
    X_train_flat_check = X_train.reshape(len(X_train), -1)
    X_test_flat_check = X_test.reshape(len(X_test), -1)
    n_check = min(50, len(X_test_flat_check))
    duplicates = sum(1 for i in range(n_check) 
                     if np.any(np.all(X_train_flat_check == X_test_flat_check[i], axis=1)))
    if duplicates == 0:
        print(f"  PASS: No exact duplicates found (checked {n_check} test samples)")
    else:
        print(f"  WARNING: Found {duplicates} exact duplicates!")
    
    # 3. Scale Data (Fit on Train ONLY, Transform others)
    print("\nScaling Data:")
    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_val = X_val.shape[0]
    n_samples_test = X_test.shape[0]
    
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    
    # Fit scaler on TRAINING data only to prevent leakage
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    print(f"  Scaler fitted on train data only")
    print(f"  Feature means: {scaler.mean_[:3]}... (first 3)")
    print(f"  Feature stds:  {scaler.scale_[:3]}... (first 3)")
    
    X_train_scaled = X_train_flat.reshape(n_samples_train, n_timesteps, n_features)
    X_val_scaled = X_val_flat.reshape(n_samples_val, n_timesteps, n_features)
    X_test_scaled = X_test_flat.reshape(n_samples_test, n_timesteps, n_features)
    
    # 4. Transpose for PyTorch (N, L, C) -> (N, C, L)
    X_train_scaled = X_train_scaled.transpose(0, 2, 1)
    X_val_scaled = X_val_scaled.transpose(0, 2, 1)
    X_test_scaled = X_test_scaled.transpose(0, 2, 1)
    
    # Convert to Tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    print("\nPASS: Preprocessing completed successfully")
    print("="*60 + "\n")
    
    return X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t, classes, scaler

class ResidualBlock1D(nn.Module):
    """Residual block with batch normalization"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, in_channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1)
        return x * out

class ImprovedMotorFaultCNN(nn.Module):
    """Enhanced CNN with residual connections and attention mechanism"""
    def __init__(self, num_classes, dropout=0.5):
        super(ImprovedMotorFaultCNN, self).__init__()
        # Input (Batch, 6, 200)
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2) # -> (32, 100)
        )
        
        # Residual blocks
        self.res_block1 = ResidualBlock1D(32, 64, stride=2) # -> (64, 50)
        self.attention1 = ChannelAttention(64)
        
        self.res_block2 = ResidualBlock1D(64, 128, stride=2) # -> (128, 25)
        self.attention2 = ChannelAttention(128)
        
        self.res_block3 = ResidualBlock1D(128, 256) # -> (256, 25)
        self.attention3 = ChannelAttention(256)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2, 128),  # *2 for avg and max pooling concat
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.res_block1(x)
        x = self.attention1(x)
        
        x = self.res_block2(x)
        x = self.attention2(x)
        
        x = self.res_block3(x)
        x = self.attention3(x)
        
        # Combine avg and max pooling
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        x = self.classifier(x)
        return x

class MotorFaultCNN(nn.Module):
    """Original simple CNN """
    def __init__(self, num_classes):
        super(MotorFaultCNN, self).__init__()
        # Input (Batch, 6, 200)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # -> (64, 100)
            
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # -> (32, 50)
            
            nn.AdaptiveAvgPool1d(1) # -> (32, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, test_loader, epochs=100, learning_rate=0.001, 
                patience=15, model_path='motor_fault_model.pth', device=None, class_weights=None):
    if device is None:
        device = get_device()
        
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"Using class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    train_losses = []
    val_accuracies = []
    test_accuracies = []
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Verbose logging (reduced frequency for efficiency)
            if (i + 1) % 50 == 0 or (i + 1) == len(train_loader):
                print(f"  Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_accuracies.append(val_acc)
        
        # Test phase
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        test_accuracies.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        # Early stopping and checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc
            }, model_path)
            print(f"Saved best model with val_acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
    return train_losses, val_accuracies, test_accuracies

def evaluate_model(model, test_loader, classes, filename='confusion_matrix.png', device=None):
    if device is None:
        device = get_device()
        
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\nClassification Report:")
    # Explicitly pass labels to handle cases where test set is missing some classes (e.g. rare faults)
    label_indices = np.arange(len(classes))
    print(classification_report(all_labels, all_preds, target_names=classes, labels=label_indices))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=label_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Confusion matrix saved to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='artifacts/dataset.npz', help='Path to dataset.npz')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--model_type', type=str, default='improved', choices=['simple', 'improved', 'transformer'],
                        help='Model architecture type')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for improved model/transformer')
    
    # Transformer specific args
    parser.add_argument('--d_model', type=int, default=128, help='Transformer: Dimension of model')
    parser.add_argument('--nhead', type=int, default=4, help='Transformer: Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Transformer: Number of encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='Transformer: FFN dimension')
    
    parser.add_argument('--output_dir', type=str, default='artifacts', help='Output directory for models and plots')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, mps)')
    args = parser.parse_args()

    # Determine device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create model-specific subdirectory
    output_dir = args.output_dir
    model_dir = os.path.join(output_dir, args.model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'motor_fault_model.pth')
    metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
    confusion_matrix_path = os.path.join(model_dir, 'confusion_matrix.png')
    training_history_path = os.path.join(model_dir, 'training_history.png')
    
    print(f"Artifacts will be saved to: {model_dir}")

    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    print(f"Dataset path: {args.dataset}")
    
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    
    data_dict = load_data(args.dataset)
    
    # Preprocess with validation
    X_train, X_val, X_test, y_train, y_val, y_test, classes, scaler = preprocess_data_v2(data_dict)
    
    print(f"\nFinal tensor sizes:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Optimize DataLoader workers for macOS (num_workers=0 often faster on Mac)
    num_workers = 0 if device.type == 'mps' else 2
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    # Select model
    # Get input channels from X_train (Batch, Channels, Length)
    input_channels = X_train.shape[1]
    max_len = X_train.shape[2]

    if args.model_type == 'improved':
        model = ImprovedMotorFaultCNN(num_classes=len(classes), dropout=args.dropout).to(device)
        print(f"Using ImprovedMotorFaultCNN with {sum(p.numel() for p in model.parameters()):,} parameters")
    elif args.model_type == 'transformer':
        from transformer_model import MotorFaultTransformer
        model = MotorFaultTransformer(
            num_classes=len(classes),
            input_channels=input_channels, 
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            max_len=max_len
        ).to(device)
        print(f"Using MotorFaultTransformer with {sum(p.numel() for p in model.parameters()):,} parameters")
    else:
        model = MotorFaultCNN(num_classes=len(classes)).to(device)
        print(f"Using MotorFaultCNN with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("\nTraining model...")
    print("\n" + "="*50)
    print("Model Architecture:")
    print(model)
    print("="*50)
    
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("="*50 + "\n")

    print("="*50 + "\n")

    # Compute Class Weights to handle imbalanced data
    print("\n" + "="*60)
    print("CLASS WEIGHT COMPUTATION")
    print("="*60)
    
    unique_y = np.unique(y_train.cpu().numpy())
    print(f"Unique classes in y_train: {unique_y}")
    
    # Calculate weights: n_samples / (n_classes * n_samples_j)
    y_train_np = y_train.cpu().numpy()
    class_counts = np.bincount(y_train_np)
    total_samples = len(y_train_np)
    n_classes = len(unique_y)
    
    weights = total_samples / (n_classes * class_counts)
    class_weights_t = torch.tensor(weights, dtype=torch.float32)
    
    print("\nClass weights:")
    for i, (cls, weight) in enumerate(zip(classes, class_weights_t)):
        print(f"  {cls}: {weight:.4f} (count: {class_counts[i]:,})")
    print("="*60 + "\n")
    
    train_losses, val_accs, test_accs = train_model(
        model, train_loader, val_loader, test_loader, 
        epochs=args.epochs, learning_rate=args.lr, patience=args.patience,
        model_path=model_path,
        device=device,
        class_weights=class_weights_t
    )
    
    # Load best model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nBest model - Val Acc: {checkpoint['val_acc']:.2f}%, Test Acc: {checkpoint['test_acc']:.2f}%")
    
    print("\nEvaluating model...")
    evaluate_model(model, test_loader, classes, filename=confusion_matrix_path, device=device)
    
    # Save metadata
    metadata = {
        'classes': classes,
        'scaler': scaler,
        'model_type': args.model_type,
        'input_shape': (X_train.shape[2], X_train.shape[1]), # (Length, Channels)
        'hyperparameters': vars(args)
    }
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Model metadata saved to {metadata_path}")
    
    # Plot training history
    plot_training_history(train_losses, val_accs, test_accs, output_path=training_history_path)

def plot_training_history(train_losses, val_accs, test_accs, output_path='training_history.png'):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.plot(test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Model Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Training history saved to {output_path}")

if __name__ == "__main__":
    main()
