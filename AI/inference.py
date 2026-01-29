"""
Motor Fault Detection - Inference Module
Provides real-time and batch inference capabilities for motor fault detection
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class MotorFaultPredictor:
    """
    Real-time motor fault detection predictor
    """
    
    def __init__(self, model_path: str = 'artifacts/motor_fault_model.pth', 
                 metadata_path: str = 'artifacts/model_metadata.pkl',
                 device: Optional[str] = None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to trained model checkpoint
            metadata_path: Path to model metadata (scaler, classes, etc.)
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                     ('mps' if torch.backends.mps.is_available() else 'cpu'))
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load metadata
        self._load_metadata()
        
        # Load model
        self._load_model()
        
    def _load_metadata(self):
        """Load model metadata"""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.classes = metadata['classes']
        self.scaler = metadata['scaler']
        self.model_type = metadata.get('model_type', 'simple')
        self.input_shape = metadata.get('input_shape', (200, 6))
        self.hyperparameters = metadata.get('hyperparameters', {})
        
        print(f"Loaded metadata: {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
        
    def _load_model(self):
        """Load trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Import model architectures (assuming they're in the same directory)
        from train_model import ImprovedMotorFaultCNN, MotorFaultCNN
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Initialize model
        if self.model_type == 'improved':
            self.model = ImprovedMotorFaultCNN(num_classes=len(self.classes))
        elif self.model_type == 'transformer':
            from transformer_model import MotorFaultTransformer
            # These params should ideally come from metadata, but for now we'll assume defaults or try to extract
            # If metadata has hyperparameters, use them
            hyperparams = getattr(self, 'hyperparameters', {})
            # If hyperparams is dict (from pickle), use it. Metadata might be dict or object depending on how it was loaded
            # Based on previous code: metadata is a dict
            
            # Re-read metadata to be safe about hyperparameters structure if available
            # We already loaded metadata into self.classes etc.
            # Let's see if we saved hyperparameters in train_model
            # Yes: 'hyperparameters': vars(args)
            
            # Need to access the raw metadata dict again or store it in _load_metadata
            # Let's check _load_metadata
            pass # See next chunk for _load_metadata update
            
            # Assuming we updated _load_metadata to store self.hyperparameters
            d_model = self.hyperparameters.get('d_model', 128)
            nhead = self.hyperparameters.get('nhead', 4)
            num_layers = self.hyperparameters.get('num_layers', 3)
            dim_feedforward = self.hyperparameters.get('dim_feedforward', 256)
            
            self.model = MotorFaultTransformer(
                num_classes=len(self.classes),
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward
            )
        else:
            self.model = MotorFaultCNN(num_classes=len(self.classes))
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded {self.model_type} model from {self.model_path}")
        
    def preprocess_window(self, window_data: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single window for inference
        
        Args:
            window_data: Shape (n_timesteps, n_features) - raw window data
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Ensure correct shape
        if window_data.shape != self.input_shape:
            raise ValueError(f"Expected shape {self.input_shape}, got {window_data.shape}")
        
        # Flatten for scaling
        n_timesteps, n_features = window_data.shape
        window_flat = window_data.reshape(-1, n_features)
        
        # Scale
        window_scaled = self.scaler.transform(window_flat)
        
        # Reshape back and transpose to (C, L)
        window_scaled = window_scaled.reshape(n_timesteps, n_features).T
        
        # Convert to tensor and add batch dimension
        window_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0)
        
        return window_tensor
    
    def predict_window(self, window_data: np.ndarray, 
                      return_probabilities: bool = False) -> Dict:
        """
        Predict fault class for a single window
        
        Args:
            window_data: Shape (n_timesteps, n_features) - raw window data
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess
        window_tensor = self.preprocess_window(window_data)
        window_tensor = window_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(window_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
        
        result = {
            'predicted_class': self.classes[predicted_class_idx],
            'confidence': confidence,
            'class_index': predicted_class_idx
        }
        
        if return_probabilities:
            result['probabilities'] = {
                cls: probabilities[0, i].item() 
                for i, cls in enumerate(self.classes)
            }
        
        return result
    
    def predict_batch(self, windows: np.ndarray, 
                     batch_size: int = 32) -> List[Dict]:
        """
        Predict fault classes for multiple windows
        
        Args:
            windows: Shape (n_windows, n_timesteps, n_features)
            batch_size: Batch size for inference
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        n_windows = len(windows)
        
        for i in range(0, n_windows, batch_size):
            batch_end = min(i + batch_size, n_windows)
            batch_windows = windows[i:batch_end]
            
            # Preprocess batch
            batch_tensors = []
            for window in batch_windows:
                tensor = self.preprocess_window(window)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
            
            # Collect results
            for j in range(len(batch_windows)):
                results.append({
                    'predicted_class': self.classes[predicted_classes[j].item()],
                    'confidence': confidences[j].item(),
                    'class_index': predicted_classes[j].item()
                })
        
        return results
    
    def predict_from_csv(self, csv_path: str, window_size: int = 200, 
                        stride: int = 100) -> pd.DataFrame:
        """
        Predict faults from a CSV file
        
        Args:
            csv_path: Path to CSV file with motor data
            window_size: Number of samples per window
            stride: Step size between windows
            
        Returns:
            DataFrame with predictions for each window
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        # Derive phase voltages
        df = self._derive_phase_voltages(df)
        
        # Extract windows
        windows, time_ranges = self._extract_windows(df, window_size, stride)
        
        # Predict
        predictions = self.predict_batch(windows)
        
        # Create results DataFrame
        results_df = pd.DataFrame(predictions)
        results_df['start_time'] = [t[0] for t in time_ranges]
        results_df['end_time'] = [t[1] for t in time_ranges]
        results_df['window_index'] = range(len(predictions))
        
        return results_df
    
    def _derive_phase_voltages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive phase voltages from duty cycles"""
        Vdc = df['Vdc']
        da = df['da']
        db = df['db']
        dc = df['dc']
        
        vaN = (2 * da - 1) * Vdc / 2
        vbN = (2 * db - 1) * Vdc / 2
        vcN = (2 * dc - 1) * Vdc / 2
        
        vn = (vaN + vbN + vcN) / 3
        
        df['va'] = vaN - vn
        df['vb'] = vbN - vn
        df['vc'] = vcN - vn
        
        return df
    
    def _extract_windows(self, df: pd.DataFrame, window_size: int, 
                        stride: int) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Extract overlapping windows from dataframe"""
        features = ['Ia', 'Ib', 'Ic', 'va', 'vb', 'vc']
        data = df[features].values
        times = df['time'].values
        
        windows = []
        time_ranges = []
        
        num_windows = (len(df) - window_size) // stride + 1
        
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            
            window = data[start_idx:end_idx]
            windows.append(window)
            
            time_ranges.append((times[start_idx], times[end_idx-1]))
        
        return np.array(windows), time_ranges
    
    def get_fault_summary(self, predictions: List[Dict]) -> Dict:
        """
        Generate summary statistics from predictions
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Summary statistics dictionary
        """
        if not predictions:
            return {}
        
        # Count predictions
        class_counts = {}
        for pred in predictions:
            cls = pred['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Overall confidence
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        
        # Most common prediction
        most_common = max(class_counts.items(), key=lambda x: x[1])
        
        healthy_count = class_counts.get('Healthy', 0)
        summary = {
            'total_windows': len(predictions),
            'class_distribution': class_counts,
            'average_confidence': avg_confidence,
            'most_common_prediction': most_common[0],
            'most_common_count': most_common[1],
            'most_common_percentage': 100 * most_common[1] / len(predictions),
            'healthy_count': healthy_count,
            'healthy_percentage': 100 * healthy_count / len(predictions)
        }
        
        return summary


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Motor Fault Detection Inference')
    parser.add_argument('--csv', type=str, help='Input CSV file path')
    parser.add_argument('--model', type=str, default='artifacts/motor_fault_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--metadata', type=str, default='artifacts/model_metadata.pkl',
                       help='Path to model metadata')
    parser.add_argument('--window_size', type=int, default=200,
                       help='Window size in samples')
    parser.add_argument('--stride', type=int, default=100,
                       help='Stride between windows')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file path')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run inference on (cpu, cuda, mps)')
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MotorFaultPredictor(
        model_path=args.model,
        metadata_path=args.metadata,
        device=args.device
    )
    
    if args.csv:
        print(f"\nProcessing file: {args.csv}")
        
        # Predict
        results_df = predictor.predict_from_csv(
            args.csv,
            window_size=args.window_size,
            stride=args.stride
        )
        
        # Print summary
        predictions = results_df.to_dict('records')
        summary = predictor.get_fault_summary(predictions)

        # Add summary stats to each row for easy CSV inspection
        results_df['healthy_percentage'] = summary['healthy_percentage']
        results_df['healthy_count'] = summary['healthy_count']
        results_df['total_windows'] = summary['total_windows']

        # Save results
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
        
        print("\n=== Prediction Summary ===")
        print(f"Total windows analyzed: {summary['total_windows']}")
        print(f"Average confidence: {summary['average_confidence']:.2%}")
        print(f"\nMost common prediction: {summary['most_common_prediction']}")
        print(f"  - Count: {summary['most_common_count']}")
        print(f"  - Percentage: {summary['most_common_percentage']:.1f}%")
        print(f"\nHealthy percentage: {summary['healthy_percentage']:.1f}%")
        print("\nClass distribution:")
        for cls, count in summary['class_distribution'].items():
            pct = 100 * count / summary['total_windows']
            print(f"  {cls}: {count} ({pct:.1f}%)")
    else:
        print("\nPredictor initialized successfully!")
        print("Use --csv argument to process a CSV file")
        print(f"Example: python inference.py --csv files/Healthy/BLDC_Healthy_0.1Nm.csv")


if __name__ == "__main__":
    main()
