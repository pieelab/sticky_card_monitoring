"""
Simple model evaluation on test split.

Evaluates trained models on held-out test data and computes:
- Overall accuracy
- Per-class metrics (accuracy, precision, recall, F1)
- Confusion matrix
- Confidence calibration

Usage:
    python evaluate_models_simple.py \\
        -m models/binary.pt \\
        -t path/to/test/crops \\
        -n 2 \\
        --class_names arthropod debris \\
        -o results

    python evaluate_models_simple.py \\
        -m models/multi.pt \\
        -t path/to/test/crops \\
        -n 4 \\
        --class_names SWD_male SWD_parasitoid SBW unidentified \\
        -o results \\
        -r dinov2_vitb14 \\
        -a
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Import from label_crops
# Note: Adjust path as needed based on your project structure
sys.path.insert(0, str(Path(__file__).parent))
try:
    from label_crops import (
        SegmentClassifier, extract_info, calculate_mean_std_npb,
        OrigResNet50, SizeResNet50, my_resnet50,
        OrigDINOv2, SizeDINOv2
    )
except ImportError:
    print("Warning: Could not import from label_crops.py")
    print("Make sure label_crops.py is in the same directory or in Python path")


class SimpleModelEvaluator:
    """Simple evaluation on test split using existing model code."""
    
    def __init__(self, model_path, test_dir, num_classes, device='cuda',
                 class_names=None, arch='resnet50', size_aware=False):
        """
        Parameters
        ----------
        model_path : str
            Path to model checkpoint
        test_dir : str
            Path to test directory (subdirectories are class labels)
        num_classes : int
            Number of classes
        device : str
            Device to run on
        class_names : list of str
            Names of classes (in order)
        arch : str
            Model architecture
        size_aware : bool
            Whether model is size-aware
        """
        self.model_path = model_path
        self.test_dir = Path(test_dir)
        self.num_classes = num_classes
        self.device = device
        self.arch = arch
        self.size_aware = size_aware
        
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.class_names_from_dirs = None  # Will be populated by _load_test_data
        
        # Load model
        self.model = self._load_model()
        self.mean_npb = None
        self.std_npb = None
        
        # Load test data
        self.test_data = self._load_test_data()
    
    def _load_model(self):
        """Load model from checkpoint"""
        print(f"Loading model from {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract state_dict and metadata
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            self.mean_npb = checkpoint.get('mean_npb', 0.0)
            self.std_npb = checkpoint.get('std_npb', 1.0)
            print(f"  ✓ Loaded enhanced checkpoint")
            if self.mean_npb != 0.0:
                print(f"  NPB: mean={self.mean_npb:.2f}, std={self.std_npb:.2f}")
        else:
            state_dict = checkpoint
            print(f"  ✓ Loaded legacy checkpoint")
        
        # Reconstruct model architecture
        if self.arch == 'resnet50':
            model = my_resnet50(
                weights=None,
                original=(not self.size_aware)
            )
            model.fc = nn.Sequential(
                nn.Linear(
                    in_features=model.fc.in_features + int(self.size_aware),
                    out_features=256
                ),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=256, out_features=self.num_classes)
            )
        elif self.arch.startswith('dinov2'):
            if self.size_aware:
                model = SizeDINOv2(self.num_classes, model_name=self.arch)
            else:
                model = OrigDINOv2(self.num_classes, model_name=self.arch)
        else:
            raise ValueError(f"Unknown architecture: {self.arch}")
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        print("  ✓ Model ready for inference")
        return model
    
    def _load_test_data(self):
        """Load test images and labels from directory structure"""
        print(f"\nLoading test data from {self.test_dir}...")
        
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")
        
        test_data = []
        
        # Find class subdirectories
        class_dirs = sorted([d for d in self.test_dir.iterdir() if d.is_dir()])
        
        print("\nDirectory to Class Mapping (Alphabetical Order):")
        dir_names = [d.name for d in class_dirs]
        
        # Extract class names from directories (alphabetical order)
        self.class_names_from_dirs = dir_names
        self.class_names = dir_names  # Override provided class_names with actual directory names
        
        for class_idx, class_dir in enumerate(class_dirs):
            dir_name = class_dir.name
            
            # Use directory index as the label (matches alphabetical order)
            label = class_idx
            actual_class_name = dir_name
            
            # Find all images
            image_files = sorted(
                list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            )
            
            for img_path in image_files:
                test_data.append({
                    'path': img_path,
                    'label': label,
                    'class_name': actual_class_name
                })
            
            print(f"  [{label}] '{dir_name}': {len(image_files)} images")
        
        print(f"  Total: {len(test_data)} images\n")
        return test_data
    
    def _load_and_process_image(self, image_path):
        """Load and preprocess image"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        
        # Convert to tensor
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # Normalize (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.to(self.device)
    
    def _extract_npb_from_filename(self, filename):
        """Extract NPB from filename if available
        
        Parameters
        ----------
        filename : str or Path
            Filename like 'crop_000000_width_height_npb_npa.png'
        """
        try:
            # Convert to string if Path object
            if hasattr(filename, 'stem'):
                filename_str = filename.stem
            else:
                # Remove extension from string
                filename_str = filename.rsplit('.', 1)[0] if '.' in filename else filename
            
            # Filename format: crop_000000_width_height_npb_npa
            parts = filename_str.split('_')
            if len(parts) >= 5:
                npb = float(parts[-2])
                return npb
        except (ValueError, IndexError, AttributeError):
            pass
        return None
    
    def run_evaluation(self):
        """Run inference on test set and compute metrics"""
        print("Running inference on test set...")
        
        all_predictions = []
        all_confidences = []
        all_labels = []
        prediction_details = []
        
        # Process each test image
        for test_sample in tqdm(self.test_data, desc="Evaluating"):
            img_path = test_sample['path']
            true_label = test_sample['label']
            true_class = test_sample['class_name']
            
            # Load and preprocess image
            img_tensor = self._load_and_process_image(img_path)
            
            # Prepare model input
            model_input = {"img": img_tensor.unsqueeze(0)}
            
            # Add NPB if size-aware
            if self.size_aware:
                npb = self._extract_npb_from_filename(img_path.name)
                if npb is not None and self.std_npb is not None:
                    npb_norm = (npb - self.mean_npb) / self.std_npb
                else:
                    npb_norm = 0.0
                model_input["npb"] = torch.tensor(
                    [npb_norm], dtype=torch.float32
                ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(model_input)
                probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Get prediction
            pred_label = torch.argmax(logits, dim=1).item()
            confidence = float(probs[0, pred_label].item())
            pred_class = self.class_names[pred_label]
            
            # Store results
            all_predictions.append(pred_label)
            all_confidences.append(confidence)
            all_labels.append(true_label)
            
            # Store detailed prediction
            class_probs = {self.class_names[i]: float(probs[0, i].item())
                          for i in range(self.num_classes)}
            
            prediction_details.append({
                'image': str(img_path.name),
                'true_label': true_class,
                'predicted_label': pred_class,
                'confidence': confidence,
                'correct': pred_label == true_label,
                'class_probabilities': class_probs
            })
        
        # Compute metrics
        results = self._compute_metrics(
            all_predictions, all_labels, all_confidences, prediction_details
        )
        
        return results, prediction_details
    
    def _compute_metrics(self, predictions, labels, confidences, prediction_details):
        """Compute evaluation metrics"""
        predictions = np.array(predictions)
        labels = np.array(labels)
        confidences = np.array(confidences)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'test_dir': str(self.test_dir),
            'num_test_samples': len(labels),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'architecture': self.arch,
            'size_aware': self.size_aware,
        }
        
        # Overall accuracy
        overall_accuracy = accuracy_score(labels, predictions)
        results['overall_accuracy'] = float(overall_accuracy)
        
        # Per-class metrics
        # Compute overall metrics with average=None to get per-class scores
        prec_all = precision_score(labels, predictions, average=None, zero_division=0, 
                                   labels=list(range(self.num_classes)))
        rec_all = recall_score(labels, predictions, average=None, zero_division=0,
                              labels=list(range(self.num_classes)))
        f1_all = f1_score(labels, predictions, average=None, zero_division=0,
                         labels=list(range(self.num_classes)))
        
        per_class = {}
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = labels == class_idx
            class_count = int(class_mask.sum())
            
            if class_count > 0:
                # Accuracy for this class: (correct predictions / total predictions for this class)
                class_correct = (predictions[class_mask] == class_idx).sum()
                acc = float(class_correct) / class_count if class_count > 0 else 0.0
                
                per_class[class_name] = {
                    'count': class_count,
                    'accuracy': float(acc),
                    'precision': float(prec_all[class_idx]),
                    'recall': float(rec_all[class_idx]),
                    'f1': float(f1_all[class_idx]),
                }
        
        results['per_class_metrics'] = per_class
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        results['confusion_matrix'] = cm.tolist()
        
        # Confidence statistics by predicted class
        confidence_stats = {}
        for pred_idx in range(self.num_classes):
            pred_mask = predictions == pred_idx
            if pred_mask.sum() > 0:
                class_confidences = confidences[pred_mask]
                confidence_stats[self.class_names[pred_idx]] = {
                    'mean': float(np.mean(class_confidences)),
                    'std': float(np.std(class_confidences)),
                    'min': float(np.min(class_confidences)),
                    'max': float(np.max(class_confidences)),
                    'count': int(pred_mask.sum()),
                }
        
        results['confidence_statistics'] = confidence_stats
        
        return results
    
    def save_results(self, results, prediction_details, output_dir):
        """Save results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to {output_dir}...")
        
        # Save JSON
        results_file = output_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ Results: {results_file}")
        
        # Save confusion matrix
        try:
            cm = np.array(results['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                         display_labels=self.class_names)
            disp.plot(ax=ax, cmap='Blues')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            cm_file = output_dir / 'confusion_matrix.png'
            plt.savefig(cm_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Confusion matrix: {cm_file}")
        except Exception as e:
            print(f"  ⚠ Could not save confusion matrix: {e}")
        
        # Save per-class performance
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Per-Class Performance Metrics', fontsize=14)
            
            metrics_names = ['accuracy', 'precision', 'recall', 'f1']
            for ax, metric in zip(axes.flat, metrics_names):
                values = [results['per_class_metrics'].get(cn, {}).get(metric, 0)
                         for cn in self.class_names]
                ax.bar(self.class_names, values, color='steelblue', edgecolor='black')
                ax.set_ylabel(metric.capitalize())
                ax.set_ylim([0, 1])
                ax.axhline(y=np.mean([v for v in values if v > 0]), 
                          color='red', linestyle='--', label='Mean')
                ax.legend()
                ax.set_title(f'{metric.capitalize()} by Class')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            perf_file = output_dir / 'per_class_metrics.png'
            plt.savefig(perf_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Per-class metrics: {perf_file}")
        except Exception as e:
            print(f"  ⚠ Could not save per-class metrics: {e}")
        
        # Save CSV predictions
        try:
            import csv
            csv_file = output_dir / 'predictions.csv'
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'image', 'true_label', 'predicted_label', 'confidence', 'correct'
                ] + [f'prob_{cn}' for cn in self.class_names])
                writer.writeheader()
                
                for pred in prediction_details:
                    row = {
                        'image': pred['image'],
                        'true_label': pred['true_label'],
                        'predicted_label': pred['predicted_label'],
                        'confidence': f"{pred['confidence']:.4f}",
                        'correct': pred['correct'],
                    }
                    for cn in self.class_names:
                        row[f'prob_{cn}'] = f"{pred['class_probabilities'][cn]:.4f}"
                    writer.writerow(row)
            
            print(f"  ✓ Predictions: {csv_file}")
        except Exception as e:
            print(f"  ⚠ Could not save predictions CSV: {e}")
        
        # Print summary
        self._print_summary(results)
    
    def _print_summary(self, results):
        """Print evaluation summary"""
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        print(f"Test Samples: {results['num_test_samples']}")
        print(f"Classes: {len(self.class_names)}")
        print(f"Architecture: {self.arch}")
        if self.size_aware:
            print(f"Size-Aware: Yes")
        print()
        
        print("Per-Class Performance:")
        print(f"{'Class':<25} {'Count':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<10}")
        print("-" * 80)
        
        for class_name in self.class_names:
            if class_name in results['per_class_metrics']:
                metrics = results['per_class_metrics'][class_name]
                print(f"{class_name:<25} {metrics['count']:<8} "
                      f"{metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                      f"{metrics['recall']:<12.4f} {metrics['f1']:<10.4f}")
        
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained model on test split',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('-t', '--test_dir', type=str, required=True,
                        help='Path to test directory (with class subdirectories)')
    parser.add_argument('-n', '--num_classes', type=int, required=True,
                        help='Number of classes')
    parser.add_argument('--class_names', type=str, nargs='+', 
                        help='Class names in order (example: --class_names SBW SWD_male SWD_parasitoid unidentified)')
    parser.add_argument('-o', '--output_dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('-r', '--arch', type=str, default='resnet50',
                        help='Model architecture')
    parser.add_argument('-a', '--size_aware', action='store_true',
                        help='Whether model is size-aware')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SimpleModelEvaluator(
        model_path=args.model,
        test_dir=args.test_dir,
        num_classes=args.num_classes,
        device=args.device,
        class_names=args.class_names,
        arch=args.arch,
        size_aware=args.size_aware
    )
    
    # Run evaluation
    results, prediction_details = evaluator.run_evaluation()
    
    # Save results
    evaluator.save_results(results, prediction_details, args.output_dir)
    
    print("✓ Evaluation complete!")


if __name__ == '__main__':
    main()