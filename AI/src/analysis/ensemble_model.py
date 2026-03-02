"""
Ensemble Model for Motor Fault Detection
Combines multiple models for improved accuracy and robustness
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict
from pathlib import Path
import pickle

from train_model import ImprovedMotorFaultCNN, MotorFaultCNN


class ModelEnsemble:
    """
    Ensemble of multiple models with voting mechanism
    """
    
    def __init__(self, model_paths: List[str], metadata_path: str, 
                 voting_method: str = 'soft', device=None):
        """
        Initialize ensemble
        
        Args:
            model_paths: List of paths to model checkpoints
            metadata_path: Path to model metadata
            voting_method: 'soft' (probability averaging) or 'hard' (majority vote)
            device: Device to run inference on
        """
        self.model_paths = [Path(p) for p in model_paths]
        self.voting_method = voting_method
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                     ('mps' if torch.backends.mps.is_available() else 'cpu'))
        else:
            self.device = torch.device(device)
        
        print(f"Initializing ensemble with {len(model_paths)} models")
        print(f"Using device: {self.device}")
        print(f"Voting method: {voting_method}")
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.classes = metadata['classes']
        self.scaler = metadata['scaler']
        self.num_classes = len(self.classes)
        
        # Load models
        self.models = []
        self._load_models()
        
    def _load_models(self):
        """Load all models in the ensemble"""
        for i, model_path in enumerate(self.model_paths):
            if not model_path.exists():
                print(f"Warning: Model not found: {model_path}")
                continue
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Try to determine model type from checkpoint
            state_dict = checkpoint['model_state_dict']
            
            # Simple heuristic: improved model has more parameters
            if 'res_block1.conv1.weight' in state_dict:
                model = ImprovedMotorFaultCNN(num_classes=self.num_classes)
                model_type = 'improved'
            elif 'transformer_encoder.layers.0.self_attn.in_proj_weight' in state_dict:
                # Naive loading (defaults), ideally should read from metadata if we save per-model metadata in ensemble
                # But ensemble assumes all models are same? Or mix?
                # The current code assumes one metadata file for the whole ensemble -> one scaling, one class map.
                # If we mix models, we might need hyperparameters per model.
                # For now let's assume we can load with default params or try to guess.
                # Actually, `ModelEnsemble` takes model *paths*.
                # If we want to support transformer, we need to import it.
                from transformer_model import MotorFaultTransformer
                
                # Check for hyperparameters in metadata if we can
                # But metadata is loaded once.
                # Let's check if 'hyperparameters' is impactful.
                # If we don't have them, we might fail unless defaults match.
                # Let's assume defaults for now or standard config.
                model = MotorFaultTransformer(num_classes=self.num_classes) 
                # Note: If train_model used different d_model, this will fail on load_state_dict.
                model_type = 'transformer'
            else:
                model = MotorFaultCNN(num_classes=self.num_classes)
                model_type = 'simple'
            
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
            print(f"  Loaded model {i+1}/{len(self.model_paths)}: {model_type}")
    
    def predict(self, X: torch.Tensor, return_probabilities: bool = False) -> Dict:
        """
        Predict using ensemble
        
        Args:
            X: Input tensor (batch_size, channels, length)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with ensemble predictions
        """
        X = X.to(self.device)
        
        # Collect predictions from all models
        all_outputs = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(X)
                probabilities = torch.softmax(outputs, dim=1)
                all_outputs.append(probabilities)
        
        # Stack predictions
        all_outputs = torch.stack(all_outputs)  # (n_models, batch_size, n_classes)
        
        if self.voting_method == 'soft':
            # Average probabilities
            ensemble_probs = torch.mean(all_outputs, dim=0)
            predicted_classes = torch.argmax(ensemble_probs, dim=1)
            confidences = torch.max(ensemble_probs, dim=1)[0]
        else:  # hard voting
            # Get predicted class from each model
            model_predictions = torch.argmax(all_outputs, dim=2)  # (n_models, batch_size)
            
            # Majority vote
            predicted_classes = []
            confidences = []
            
            for i in range(X.size(0)):
                votes = model_predictions[:, i]
                unique, counts = torch.unique(votes, return_counts=True)
                majority_class = unique[torch.argmax(counts)]
                confidence = counts.max().float() / len(self.models)
                
                predicted_classes.append(majority_class)
                confidences.append(confidence)
            
            predicted_classes = torch.stack(predicted_classes)
            confidences = torch.tensor(confidences, device=self.device)
            ensemble_probs = None  # Not meaningful for hard voting
        
        # Convert to CPU and numpy
        predicted_classes = predicted_classes.cpu().numpy()
        confidences = confidences.cpu().numpy()
        
        result = {
            'predicted_classes': predicted_classes,
            'confidences': confidences
        }
        
        if return_probabilities and ensemble_probs is not None:
            result['probabilities'] = ensemble_probs.cpu().numpy()
        
        return result
    
    def predict_single(self, window: torch.Tensor) -> Dict:
        """
        Predict for a single window
        
        Args:
            window: Single window tensor (1, channels, length) or (channels, length)
            
        Returns:
            Prediction dictionary
        """
        if window.dim() == 2:
            window = window.unsqueeze(0)
        
        result = self.predict(window, return_probabilities=True)
        
        # Extract single prediction
        single_result = {
            'predicted_class': self.classes[result['predicted_classes'][0]],
            'confidence': float(result['confidences'][0]),
            'class_index': int(result['predicted_classes'][0])
        }
        
        if 'probabilities' in result:
            single_result['probabilities'] = {
                cls: float(result['probabilities'][0, i])
                for i, cls in enumerate(self.classes)
            }
        
        return single_result
    
    def evaluate(self, test_loader):
        """
        Evaluate ensemble on test set
        
        Args:
            test_loader: DataLoader for test set
            
        Returns:
            Accuracy and predictions
        """
        all_predictions = []
        all_labels = []
        
        for inputs, labels in test_loader:
            result = self.predict(inputs)
            all_predictions.extend(result['predicted_classes'])
            all_labels.extend(labels.numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = np.mean(all_predictions == all_labels)
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }


def train_ensemble_models(dataset_path: str, n_models: int = 5, 
                         epochs: int = 100, base_seed: int = 42,
                         output_dir: str = 'artifacts', device=None):
    """
    Train multiple models with different initialization for ensemble
    
    Args:
        dataset_path: Path to dataset
        n_models: Number of models to train
        epochs: Training epochs per model
        base_seed: Base random seed
    """
    from train_model import load_data, preprocess_data, train_model, get_device
    from torch.utils.data import TensorDataset, DataLoader
    
    print("Loading data...")
    X, y = load_data(dataset_path)
    
    os.makedirs(output_dir, exist_ok=True)
    model_paths = []
    
    for i in range(n_models):
        print(f"\n{'='*60}")
        print(f"Training model {i+1}/{n_models}")
        print('='*60)
        
        # Set seed for reproducibility but different for each model
        seed = base_seed + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Preprocess with different split (due to different seed)
        X_train, X_val, X_test, y_train, y_val, y_test, classes, scaler = preprocess_data(X, y)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        if device is None:
            device = get_device()
        
        model = ImprovedMotorFaultCNN(num_classes=len(classes)).to(device)
        
        # Train
        model_path = os.path.join(output_dir, f'ensemble_model_{i+1}.pth')
        train_model(model, train_loader, val_loader, test_loader, 
                   epochs=epochs, model_path=model_path, device=device)
        
        model_paths.append(model_path)
        
        # Save metadata for first model only
        if i == 0:
            import pickle
            metadata = {
                'classes': classes,
                'scaler': scaler,
                'model_type': 'improved',
                'input_shape': (X.shape[1], X.shape[2])
            }
            metadata_path = os.path.join(output_dir, 'model_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
    
    print(f"\n{'='*60}")
    print("Ensemble training complete!")
    print(f"Trained models: {model_paths}")
    print('='*60)
    
    return model_paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or use ensemble models')
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train',
                       help='Train new ensemble or evaluate existing')
    parser.add_argument('--dataset', type=str, default='artifacts/dataset.npz',
                       help='Path to dataset')
    parser.add_argument('--n_models', type=int, default=5,
                       help='Number of models in ensemble')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs per model')
    parser.add_argument('--model_paths', nargs='+',
                       help='Paths to existing models for evaluation')
    parser.add_argument('--voting', choices=['soft', 'hard'], default='soft',
                       help='Voting method')
    parser.add_argument('--output_dir', type=str, default='artifacts',
                       help='Output directory for ensemble artifacts')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu, cuda, mps)')
    args = parser.parse_args()
    
    if args.mode == 'train':
        model_paths = train_ensemble_models(
            args.dataset, 
            n_models=args.n_models,
            epochs=args.epochs,
            output_dir=args.output_dir,
            device=args.device
        )
    else:  # evaluate
        if not args.model_paths:
            print("Error: --model_paths required for evaluation")
            exit(1)
        
        # Load test data
        from train_model import load_data, preprocess_data
        from torch.utils.data import TensorDataset, DataLoader
        
        X, y = load_data(args.dataset)
        X_train, X_val, X_test, y_train, y_val, y_test, classes, scaler = preprocess_data(X, y)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create ensemble
        metadata_path = os.path.join(args.output_dir, 'model_metadata.pkl')
        ensemble = ModelEnsemble(
            args.model_paths,
            metadata_path,
            voting_method=args.voting,
            device=args.device
        )
        
        # Evaluate
        print("\nEvaluating ensemble...")
        results = ensemble.evaluate(test_loader)
        print(f"Ensemble Accuracy: {results['accuracy']*100:.2f}%")
