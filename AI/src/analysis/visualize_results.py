"""
Comprehensive Visualization Suite for Motor Fault Detection Neural Network
Generates graphs, confusion matrices, ROC curves, and detailed analysis plots
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import pickle
from train_model import load_data, preprocess_data, ImprovedMotorFaultCNN, MotorFaultCNN

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

device = torch.device('cuda' if torch.cuda.is_available() else 
                     ('mps' if torch.backends.mps.is_available() else 'cpu'))


class ModelVisualizer:
    """Comprehensive visualization suite for neural network analysis"""
    
    def __init__(self, model_path='motor_fault_model.pth', 
                 metadata_path='model_metadata.pkl',
                 dataset_path='dataset.npz',
                 output_dir='visualizations'):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📊 Initializing visualizer...")
        print(f"Output directory: {self.output_dir}")
        
        # Load metadata
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        self.classes = metadata['classes']
        self.scaler = metadata['scaler']
        self.model_type = metadata.get('model_type', 'improved')
        
        # Load model
        self._load_model()
        
        # Load data
        self._load_data()
        
    def _load_model(self):
        """Load trained model"""
        checkpoint = torch.load(self.model_path, map_location=device)
        
        if self.model_type == 'improved':
            self.model = ImprovedMotorFaultCNN(num_classes=len(self.classes))
        else:
            self.model = MotorFaultCNN(num_classes=len(self.classes))
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print(f"✓ Loaded {self.model_type} model")
        
    def _load_data(self):
        """Load and preprocess dataset"""
        X, y = load_data(self.dataset_path)
        X_train, X_val, X_test, y_train, y_val, y_test, classes, scaler = preprocess_data(X, y)
        
        self.test_dataset = TensorDataset(X_test, y_test)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        
        self.X_test = X_test
        self.y_test = y_test
        print(f"✓ Loaded dataset: {len(X_test)} test samples")
        
    def get_predictions(self):
        """Get model predictions and probabilities"""
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_preds), np.array(all_probs), np.array(all_labels)
    
    def plot_confusion_matrix(self, save=True):
        """Generate enhanced confusion matrix"""
        print("\n📈 Generating confusion matrix...")
        
        preds, probs, labels = self.get_predictions()
        cm = confusion_matrix(labels, preds)
        
        # Create figure with multiple views
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 1. Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix - Counts', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # 2. Normalized (percentages)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='RdYlGn', 
                   xticklabels=self.classes, yticklabels=self.classes,
                   ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
        axes[1].set_title('Confusion Matrix - Normalized (%)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'confusion_matrix_detailed.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        plt.close()
        
        return cm
    
    def plot_classification_metrics(self, save=True):
        """Generate comprehensive classification metrics visualization"""
        print("\n📊 Generating classification metrics...")
        
        preds, probs, labels = self.get_predictions()
        
        # Calculate per-class metrics
        precision = precision_score(labels, preds, average=None)
        recall = recall_score(labels, preds, average=None)
        f1 = f1_score(labels, preds, average=None)
        
        # Create bar plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        x = np.arange(len(self.classes))
        width = 0.25
        
        # 1. Precision, Recall, F1 comparison
        axes[0, 0].bar(x - width, precision, width, label='Precision', alpha=0.8)
        axes[0, 0].bar(x, recall, width, label='Recall', alpha=0.8)
        axes[0, 0].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        axes[0, 0].set_ylabel('Score', fontsize=12)
        axes[0, 0].set_title('Per-Class Metrics Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(self.classes, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].set_ylim([0, 1.1])
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            axes[0, 0].text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
            axes[0, 0].text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
            axes[0, 0].text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Class distribution (support)
        support = np.bincount(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.classes)))
        bars = axes[0, 1].bar(self.classes, support, color=colors, alpha=0.8)
        axes[0, 1].set_ylabel('Number of Samples', fontsize=12)
        axes[0, 1].set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(np.arange(len(self.classes)))
        axes[0, 1].set_xticklabels(self.classes, rotation=45, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(height)}',
                          ha='center', va='bottom', fontsize=10)
        
        # 3. Accuracy per class
        accuracy_per_class = []
        for i, cls in enumerate(self.classes):
            mask = labels == i
            if mask.sum() > 0:
                acc = accuracy_score(labels[mask], preds[mask])
                accuracy_per_class.append(acc * 100)
            else:
                accuracy_per_class.append(0)
        
        bars = axes[1, 0].barh(self.classes, accuracy_per_class, color=colors, alpha=0.8)
        axes[1, 0].set_xlabel('Accuracy (%)', fontsize=12)
        axes[1, 0].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlim([0, 105])
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracy_per_class)):
            axes[1, 0].text(acc + 1, bar.get_y() + bar.get_height()/2.,
                          f'{acc:.1f}%',
                          ha='left', va='center', fontsize=10)
        
        # 4. Overall metrics summary
        overall_acc = accuracy_score(labels, preds) * 100
        overall_prec = precision_score(labels, preds, average='weighted') * 100
        overall_rec = recall_score(labels, preds, average='weighted') * 100
        overall_f1 = f1_score(labels, preds, average='weighted') * 100
        
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score (%)': [overall_acc, overall_prec, overall_rec, overall_f1]
        }
        
        axes[1, 1].axis('off')
        table_data = [[m, f'{s:.2f}%'] for m, s in zip(metrics_data['Metric'], metrics_data['Score (%)'])]
        table = axes[1, 1].table(cellText=table_data,
                                colLabels=['Metric', 'Score'],
                                cellLoc='center',
                                loc='center',
                                colWidths=[0.4, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 3)
        
        # Style the table
        for i in range(len(table_data) + 1):
            if i == 0:
                table[(i, 0)].set_facecolor('#4CAF50')
                table[(i, 1)].set_facecolor('#4CAF50')
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            else:
                table[(i, 0)].set_facecolor('#E8F5E9')
                table[(i, 1)].set_facecolor('#E8F5E9')
        
        axes[1, 1].set_title('Overall Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'classification_metrics.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_roc_curves(self, save=True):
        """Generate ROC curves for each class"""
        print("\n📈 Generating ROC curves...")
        
        preds, probs, labels = self.get_predictions()
        
        # Binarize labels for multi-class ROC
        labels_bin = label_binarize(labels, classes=range(len(self.classes)))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot ROC curve for each class
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.classes)))
        
        for i, (cls, color) in enumerate(zip(self.classes, colors)):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            axes[0].plot(fpr, tpr, color=color, lw=2, 
                        label=f'{cls} (AUC = {roc_auc:.3f})')
        
        axes[0].plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate', fontsize=12)
        axes[0].set_title('ROC Curves - Multi-Class', fontsize=14, fontweight='bold')
        axes[0].legend(loc="lower right", fontsize=10)
        axes[0].grid(alpha=0.3)
        
        # Plot micro-average ROC curve
        fpr_micro, tpr_micro, _ = roc_curve(labels_bin.ravel(), probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        axes[1].plot(fpr_micro, tpr_micro, color='deeppink', lw=3,
                    label=f'Micro-average (AUC = {roc_auc_micro:.3f})')
        axes[1].plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate', fontsize=12)
        axes[1].set_ylabel('True Positive Rate', fontsize=12)
        axes[1].set_title('ROC Curve - Micro-Average', fontsize=14, fontweight='bold')
        axes[1].legend(loc="lower right", fontsize=12)
        axes[1].grid(alpha=0.3)
        axes[1].fill_between(fpr_micro, tpr_micro, alpha=0.2, color='deeppink')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'roc_curves.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_precision_recall_curves(self, save=True):
        """Generate Precision-Recall curves"""
        print("\n📈 Generating Precision-Recall curves...")
        
        preds, probs, labels = self.get_predictions()
        labels_bin = label_binarize(labels, classes=range(len(self.classes)))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.ravel()
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.classes)))
        
        for i, (cls, color) in enumerate(zip(self.classes, colors)):
            precision, recall, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
            pr_auc = auc(recall, precision)
            
            axes[i].plot(recall, precision, color=color, lw=3, 
                        label=f'PR curve (AUC = {pr_auc:.3f})')
            axes[i].fill_between(recall, precision, alpha=0.2, color=color)
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('Recall', fontsize=12)
            axes[i].set_ylabel('Precision', fontsize=12)
            axes[i].set_title(f'Precision-Recall: {cls}', fontsize=13, fontweight='bold')
            axes[i].legend(loc="lower left", fontsize=10)
            axes[i].grid(alpha=0.3)
            
            # Add baseline
            baseline = labels_bin[:, i].sum() / len(labels_bin)
            axes[i].axhline(y=baseline, color='red', linestyle='--', lw=2, 
                           label=f'Baseline ({baseline:.3f})')
            axes[i].legend(loc="lower left", fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'precision_recall_curves.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_prediction_confidence(self, save=True):
        """Visualize prediction confidence distribution"""
        print("\n📊 Generating confidence distribution plots...")
        
        preds, probs, labels = self.get_predictions()
        confidences = np.max(probs, axis=1)
        correct = (preds == labels)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall confidence distribution
        axes[0, 0].hist(confidences[correct], bins=50, alpha=0.7, 
                       label='Correct', color='green', edgecolor='black')
        axes[0, 0].hist(confidences[~correct], bins=50, alpha=0.7, 
                       label='Incorrect', color='red', edgecolor='black')
        axes[0, 0].set_xlabel('Confidence', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=12)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Confidence by predicted class
        for i, cls in enumerate(self.classes):
            mask = preds == i
            if mask.sum() > 0:
                data = confidences[mask]
                data_range = data.max() - data.min()
                if data_range < 0.001:
                    n_bins = 3
                elif data_range < 0.01:
                    n_bins = 5
                elif data_range < 0.05:
                    n_bins = 10
                else:
                    n_bins = min(30, max(5, int(mask.sum() ** 0.5)))
                axes[0, 1].hist(data, bins=n_bins, alpha=0.6, 
                              label=cls, edgecolor='black')
        axes[0, 1].set_xlabel('Confidence', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Confidence Distribution by Predicted Class', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Mean confidence per class
        mean_conf_per_class = []
        for i in range(len(self.classes)):
            mask = preds == i
            if mask.sum() > 0:
                mean_conf_per_class.append(confidences[mask].mean())
            else:
                mean_conf_per_class.append(0)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.classes)))
        bars = axes[1, 0].bar(self.classes, mean_conf_per_class, color=colors, 
                             alpha=0.8, edgecolor='black')
        axes[1, 0].set_ylabel('Mean Confidence', fontsize=12)
        axes[1, 0].set_title('Average Confidence per Predicted Class', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(np.arange(len(self.classes)))
        axes[1, 0].set_xticklabels(self.classes, rotation=45, ha='right')
        axes[1, 0].set_ylim([0, 1.1])
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        for bar, conf in zip(bars, mean_conf_per_class):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{conf:.3f}',
                          ha='center', va='bottom', fontsize=10)
        
        # 4. Confidence vs Accuracy scatter
        axes[1, 1].scatter(confidences[correct], np.ones(correct.sum()), 
                          alpha=0.3, s=10, c='green', label='Correct')
        axes[1, 1].scatter(confidences[~correct], np.zeros((~correct).sum()), 
                          alpha=0.3, s=10, c='red', label='Incorrect')
        axes[1, 1].set_xlabel('Confidence', fontsize=12)
        axes[1, 1].set_ylabel('Correctness (1=Correct, 0=Incorrect)', fontsize=12)
        axes[1, 1].set_title('Prediction Confidence vs Correctness', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=12)
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_ylim([-0.1, 1.1])
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'confidence_analysis.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_error_analysis(self, save=True):
        """Detailed error analysis"""
        print("\n🔍 Generating error analysis...")
        
        preds, probs, labels = self.get_predictions()
        errors = preds != labels
        
        if errors.sum() == 0:
            print("⚠️  No errors found! Perfect predictions!")
            return
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Error distribution by true class
        ax1 = fig.add_subplot(gs[0, :2])
        error_per_class = []
        for i in range(len(self.classes)):
            mask = labels == i
            if mask.sum() > 0:
                error_rate = errors[mask].sum() / mask.sum() * 100
                error_per_class.append(error_rate)
            else:
                error_per_class.append(0)
        
        colors = ['red' if e > 10 else 'orange' if e > 5 else 'green' 
                 for e in error_per_class]
        bars = ax1.bar(self.classes, error_per_class, color=colors, 
                      alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Error Rate (%)', fontsize=12)
        ax1.set_title('Error Rate by True Class', fontsize=14, fontweight='bold')
        ax1.set_xticklabels(self.classes, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5% threshold')
        ax1.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
        ax1.legend()
        
        for bar, err in zip(bars, error_per_class):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{err:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Error statistics table
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        total_errors = errors.sum()
        total_samples = len(errors)
        overall_error_rate = total_errors / total_samples * 100
        
        stats_data = [
            ['Total Samples', f'{total_samples}'],
            ['Correct Predictions', f'{total_samples - total_errors}'],
            ['Errors', f'{total_errors}'],
            ['Error Rate', f'{overall_error_rate:.2f}%'],
            ['Accuracy', f'{100 - overall_error_rate:.2f}%']
        ]
        
        table = ax2.table(cellText=stats_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        for i in range(len(stats_data) + 1):
            if i == 0:
                table[(i, 0)].set_facecolor('#2196F3')
                table[(i, 1)].set_facecolor('#2196F3')
                table[(i, 0)].set_text_props(weight='bold', color='white')
                table[(i, 1)].set_text_props(weight='bold', color='white')
            else:
                table[(i, 0)].set_facecolor('#E3F2FD')
                table[(i, 1)].set_facecolor('#E3F2FD')
        
        # 3. Confusion pairs (most common misclassifications)
        ax3 = fig.add_subplot(gs[1, :])
        error_pairs = []
        for true_idx in range(len(self.classes)):
            for pred_idx in range(len(self.classes)):
                if true_idx != pred_idx:
                    count = ((labels == true_idx) & (preds == pred_idx)).sum()
                    if count > 0:
                        error_pairs.append((self.classes[true_idx], 
                                          self.classes[pred_idx], count))
        
        if error_pairs:
            error_pairs.sort(key=lambda x: x[2], reverse=True)
            top_errors = error_pairs[:10]  # Top 10 error pairs
            
            labels_str = [f'{true}\n→\n{pred}' for true, pred, _ in top_errors]
            counts = [count for _, _, count in top_errors]
            
            bars = ax3.barh(range(len(top_errors)), counts, color='coral', 
                           alpha=0.8, edgecolor='black')
            ax3.set_yticks(range(len(top_errors)))
            ax3.set_yticklabels(labels_str, fontsize=10)
            ax3.set_xlabel('Number of Misclassifications', fontsize=12)
            ax3.set_title('Top 10 Misclassification Pairs (True → Predicted)', 
                         fontsize=14, fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
            
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax3.text(count + 0.5, bar.get_y() + bar.get_height()/2.,
                        f'{count}',
                        ha='left', va='center', fontsize=10, fontweight='bold')
        
        # 4-6. Per-class error distribution heatmaps (if space)
        error_conf_matrix = confusion_matrix(labels, preds)
        np.fill_diagonal(error_conf_matrix, 0)  # Remove correct predictions
        
        ax4 = fig.add_subplot(gs[2, :])
        sns.heatmap(error_conf_matrix, annot=True, fmt='d', cmap='Reds',
                   xticklabels=self.classes, yticklabels=self.classes,
                   ax=ax4, cbar_kws={'label': 'Error Count'})
        ax4.set_title('Error Distribution Matrix (Excluding Diagonal)', 
                     fontsize=14, fontweight='bold')
        ax4.set_ylabel('True Label', fontsize=12)
        ax4.set_xlabel('Predicted Label', fontsize=12)
        
        if save:
            filepath = self.output_dir / 'error_analysis.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*70)
        print("🎨 GENERATING COMPREHENSIVE NEURAL NETWORK VISUALIZATIONS")
        print("="*70)
        
        self.plot_confusion_matrix()
        self.plot_classification_metrics()
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_prediction_confidence()
        self.plot_error_analysis()
        
        print("\n" + "="*70)
        print(f"✅ ALL VISUALIZATIONS COMPLETE!")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print("="*70)
        
        # List all generated files
        print("\n📊 Generated files:")
        for file in sorted(self.output_dir.glob('*.png')):
            print(f"  • {file.name}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive neural network visualizations')
    parser.add_argument('--model', type=str, default='../../artifacts/improved/motor_fault_model.pth',
                       help='Path to trained model')
    parser.add_argument('--metadata', type=str, default='../../artifacts/improved/model_metadata.pkl',
                       help='Path to model metadata')
    parser.add_argument('--dataset', type=str, default='../../artifacts/dataset.npz',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='../../outputs/visualizations',
                       help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ModelVisualizer(
        model_path=args.model,
        metadata_path=args.metadata,
        dataset_path=args.dataset,
        output_dir=args.output_dir
    )
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
