"""
SWD Detection Inference Script

Runs inference on flatbug-generated crops using a two-tier classifier:
1. Binary classifier: arthropod vs debris
2. Multi-class classifier: SWD_male, SWD_parasitoid, SBW, unidentified_arthropod

Maps predictions back to original scans and produces annotated output images.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2

# Import the SegmentClassifier from your training code
# Adjust the import path as needed for your project structure
sys.path.insert(0, r'C:\Users\ALANalysis\sticky_card_monitoring')
from src.label_crops import SegmentClassifier


class SWDAnnotationPipeline:
    """
    Pipeline for running inference on flatbug crops and annotating original scans.
    """
    
    # Color map for different classes
    CLASS_COLORS = {
        'Small_black_weevil': (0, 0, 255),      # Blue
        'SWD_male': (255, 0, 0),           # Red
        'SWD_parasitoid': (0, 255, 0),     # Green
        'Unidentified_Arthropod': (255, 255, 0),     # Yellow
        'debris': (128, 128, 128)          # Gray
    }
    
    # Class mappings for both stages
    BINARY_CLASSES = {0: 'debris', 1: 'arthropod'}
    MULTI_CLASSES = {
        0: 'Small_black_weevil',
        1: 'SWD_male',
        2: 'SWD_parasitoid',
        3: 'Unidentified_Arthropod'
    }
    
    def __init__(self, binary_model_path, multi_model_path, crops_dir, 
                 scans_dir, output_dir, device='cuda' if torch.cuda.is_available() else 'cpu',
                 annotate_classes=None, arch='resnet50', size_aware=False,
                 multi_only=False, binary_only=False):
        """
        Initialize the annotation pipeline.
        
        Parameters
        ----------
        binary_model_path : str
            Path to binary classifier model checkpoint
        multi_model_path : str
            Path to multi-class classifier model checkpoint
        crops_dir : str
            Path to flatbug-generated crops directory
        scans_dir : str
            Path to original scans directory
        output_dir : str
            Path to save annotated scans
        device : str
            'cuda' or 'cpu'
        annotate_classes : list of str, optional
            List of class names to annotate. If None, all classes are annotated.
            Example: ['SWD_male', 'SWD_parasitoid', 'SBW'] (skip 'unidentified')
        arch : str, optional
            Model architecture used during training. 
            Options: 'resnet50', 'dinov2_vitb14', 'dinov2_vitl14', etc.
            Default: 'resnet50'. Must match the architecture used to train your models.
        size_aware : bool, optional
            Set to True if your models were trained with size awareness (-a flag).
            Default: False. Must match the training configuration.
        multi_only : bool, optional
            If True, skip binary classification and run only multi-class model.
            Assumes all crops are arthropods. Useful for testing/debugging.
            Default: False.
        binary_only : bool, optional
            If True, run only binary classification (arthropod vs debris).
            Useful for testing/debugging the first stage.
            Default: False.
        """
        self.binary_model_path = binary_model_path
        self.multi_model_path = multi_model_path
        self.crops_dir = Path(crops_dir)
        self.scans_dir = Path(scans_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        self.arch = arch
        self.size_aware = size_aware
        self.multi_only = multi_only
        self.binary_only = binary_only
        
        # Initialize mean_npb/std_npb - will be loaded from checkpoint in _load_model()
        # We set dummy values here; they'll be overwritten if checkpoint contains NPB values
        if size_aware:
            self.mean_npb = 0.0
            self.std_npb = 1.0
        else:
            self.mean_npb = None
            self.std_npb = None
        
        # Set default: all classes if not specified
        if annotate_classes is None:
            self.annotate_classes = list(self.CLASS_COLORS.keys())
        else:
            self.annotate_classes = annotate_classes
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models (only load what's needed based on mode)
        print("Loading models...")
        if multi_only:
            print("  Mode: Multi-class only (skipping binary classifier)")
            self.binary_classifier = None
            self.multi_classifier = self._load_model(multi_model_path, num_classes=4)
        elif binary_only:
            print("  Mode: Binary only (skipping multi-class classifier)")
            self.binary_classifier = self._load_model(binary_model_path, num_classes=2)
            self.multi_classifier = None
        else:
            print("  Mode: Normal (both classifiers)")
            self.binary_classifier = self._load_model(binary_model_path, num_classes=2)
            self.multi_classifier = self._load_model(multi_model_path, num_classes=4)
        
        # Load COCO metadata
        print("Loading COCO metadata...")
        self.coco_data = self._load_coco_metadata()
        
        # Create lookup tables
        self.image_id_to_annotations = self._build_annotation_lookup()
        self.crop_to_image_mapping = self._build_crop_to_image_mapping()
        
    def _load_model(self, checkpoint_path, num_classes):
        """
        Load a trained SegmentClassifier model.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to saved model checkpoint
        num_classes : int
            Number of classes (2 for binary, 4 for multi-class)
            
        Returns
        -------
        classifier : SegmentClassifier
            Loaded classifier in eval mode
        """
        # Create a temporary classifier instance for model loading
        # Use dummy parameters since we only need the model structure
        classifier = SegmentClassifier(
            id="inference",
            data_dir=str(self.crops_dir),
            num_classes=num_classes,
            device=self.device,
            mean_npb=self.mean_npb,
            std_npb=self.std_npb,
            optim=2,
            Transform=None,
            sample=False,
            loss_weights=False,
            batch_size=32,
            num_workers=0,
            lr=1e-4,
            stop_early=False,
            freeze_backbone=False
        )
        
        # Load model architecture (size_aware is determined by mean_npb/std_npb)
        classifier.load_inference_model(backbone=self.arch)
        
        # Load weights and NPB values from checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle both enhanced checkpoint format (dict with metadata) and legacy format (state_dict only)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # Enhanced checkpoint format with NPB values
                state_dict = checkpoint['state_dict']
                
                # Extract NPB values if available
                if 'mean_npb' in checkpoint and 'std_npb' in checkpoint:
                    self.mean_npb = checkpoint['mean_npb']
                    self.std_npb = checkpoint['std_npb']
                    classifier.mean_npb = self.mean_npb
                    classifier.std_npb = self.std_npb
                    print(f"  ✓ Loaded NPB from checkpoint: mean={self.mean_npb:.2f}, std={self.std_npb:.2f}")
                else:
                    # Enhanced format but no NPB values (shouldn't happen with new training script)
                    print(f"  ⚠ Checkpoint format enhanced but NPB values missing")
            else:
                # Legacy state_dict only format
                state_dict = checkpoint
                print(f"  ⚠ Legacy checkpoint format (no NPB values) - predictions may be incorrect!")
            
            classifier.model.load_state_dict(state_dict)
            print(f"  Loaded model weights from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        classifier.model.eval()
        return classifier
    
    def _load_coco_metadata(self):
        """
        Load COCO format metadata from coco_instances.json.
        
        Returns
        -------
        dict : COCO format data
        """
        coco_path = self.crops_dir / 'coco_instances.json'
        if not coco_path.exists():
            raise FileNotFoundError(f"COCO metadata not found: {coco_path}")
        
        with open(coco_path, 'r') as f:
            return json.load(f)
    
    def _build_annotation_lookup(self):
        """
        Build a lookup table mapping image IDs to their annotations.
        
        Returns
        -------
        dict : {image_id: [annotations]}
        """
        lookup = defaultdict(list)
        for annotation in self.coco_data.get('annotations', []):
            image_id = annotation['image_id']
            lookup[image_id].append(annotation)
        return lookup
    
    def _build_crop_to_image_mapping(self):
        """
        Build mapping from crop folder names to image IDs and filenames.
        
        Matches crop folders with COCO image entries. COCO filenames may have
        extensions (.jpg) but crop folders don't, so we strip extensions.
        
        Returns
        -------
        dict : {crop_folder_name: (image_id, image_filename, scan_name)}
        """
        mapping = {}
        
        # Get list of actual crop folders
        crop_folders = {d.name: d for d in self.crops_dir.iterdir() if d.is_dir() and (d / 'crops').exists()}
        
        print(f"\nMatching {len(crop_folders)} crop folders with COCO metadata...")
        
        # Build mapping from COCO images
        matched_count = 0
        for image_info in self.coco_data.get('images', []):
            image_id = image_info['id']
            filename = image_info['file_name']
            
            # Strip file extension (.jpg, .png, etc.) to get folder name
            # COCO: "ARR_ADM-NS-23960-B014.jpg" -> "ARR_ADM-NS-23960-B014"
            scan_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            
            # Check if this folder exists
            if scan_name in crop_folders:
                mapping[scan_name] = (image_id, filename, scan_name)
                matched_count += 1
            else:
                # Try alternative extraction methods as fallback
                possible_names = []
                
                # Method: Extract from path if filename has directory
                if '/' in filename:
                    # "path/to/ARR_ADM-NS-23960-B014.jpg" -> "ARR_ADM-NS-23960-B014"
                    base = filename.split('/')[-1]
                    base_no_ext = base.rsplit('.', 1)[0] if '.' in base else base
                    possible_names.append(base_no_ext)
                    
                    # Also try directory name
                    possible_names.append(filename.split('/')[-2])
                
                # Check fallback names
                for possible_name in possible_names:
                    if possible_name in crop_folders:
                        mapping[possible_name] = (image_id, filename, possible_name)
                        matched_count += 1
                        break
        
        print(f"✓ Matched {matched_count} of {len(self.coco_data.get('images', []))} COCO images with crop folders")
        
        if matched_count == 0:
            print("\n⚠ WARNING: No crop folders matched with COCO metadata!")
            print("Debugging info:")
            print(f"  Crop folder names (first 3):")
            for name in sorted(crop_folders.keys())[:3]:
                print(f"    - {name}")
            print(f"  COCO filenames (first 3):")
            for img in self.coco_data.get('images', [])[:3]:
                print(f"    - {img['file_name']}")
        
        return mapping
    
    def _get_transform(self):
        """
        Get the inference transform (same as used in training).
        """
        return transforms.Compose([
            transforms.ToImage(),
            transforms.Resize(size=(224, 224), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_crop(self, crop_path):
        """
        Load and transform a crop image for inference.
        
        Parameters
        ----------
        crop_path : str or Path
            Path to crop image
            
        Returns
        -------
        torch.Tensor : Transformed image tensor
        """
        img = Image.open(crop_path).convert('RGB')
        transform = self._get_transform()
        return transform(img).unsqueeze(0)  # Add batch dimension
    
    def _extract_metadata_from_filename(self, crop_path):
        """
        Extract size metadata from rescaled crop filename.
        
        Rescaled crop filenames have format: <name>_<width>_<height>_<npb>_<npa>.png
        Example: crop_000000_1024_768_45234_11245.png
        
        Parameters
        ----------
        crop_path : str or Path
            Path to crop image
            
        Returns
        -------
        dict
            Dictionary with keys: width, height, npb, npa
            Returns None if metadata couldn't be extracted
        """
        try:
            filename = Path(crop_path).stem  # Get filename without extension
            parts = filename.split('_')
            
            # Format: ..._<width>_<height>_<npb>_<npa>
            # The last 4 parts should be: width, height, npb, npa
            if len(parts) >= 4:
                try:
                    width = int(parts[-4])
                    height = int(parts[-3])
                    npb = int(parts[-2])      # Number of pixels before resizing
                    npa = int(parts[-1])      # Number of pixels after resizing
                    
                    return {
                        'width': width,
                        'height': height,
                        'npb': npb,
                        'npa': npa
                    }
                except (ValueError, IndexError):
                    return None
            return None
        except Exception:
            return None
    
    @torch.no_grad()
    def _classify_crop(self, crop_path):
        """
        Run inference on a crop image through classifiers.
        
        Handles both size-aware and non-size-aware models.
        Supports multi_only mode (skip binary) or binary_only mode (skip multi).
        For size-aware models, extracts npb from the rescaled crop filename.
        
        Parameters
        ----------
        crop_path : str or Path
            Path to crop image
            
        Returns
        -------
        dict : {
            'binary_class': str,
            'binary_confidence': float,
            'multi_class': str (if binary != debris),
            'multi_confidence': float
        }
        """
        try:
            # Load and prepare crop
            img_tensor = self._load_crop(crop_path).to(self.device)
            
            # Extract metadata from filename if size_aware
            npb_tensor = None
            if self.size_aware:
                metadata = self._extract_metadata_from_filename(crop_path)
                if metadata:
                    npb = float(metadata['npb'])
                    # Normalize npb using mean and std
                    # If mean_npb/std_npb are 0/1, this just returns the raw value
                    npb_norm = (npb - self.mean_npb) / self.std_npb if self.std_npb != 0 else npb
                    npb_tensor = torch.tensor([npb_norm], dtype=torch.float32).to(self.device)
                else:
                    # If we can't extract npb, use 0 as default
                    npb_tensor = torch.tensor([0.0], dtype=torch.float32).to(self.device)
            
            result = {
                'binary_class': None,
                'binary_confidence': None,
                'multi_class': None,
                'multi_confidence': None
            }
            
            # Mode: multi_only - skip binary classification
            if self.multi_only:
                # Assume all inputs are arthropods
                result['binary_class'] = 'arthropod'
                result['binary_confidence'] = 1.0  # Confidence = 1.0 (assumed)
                
                # Run multi-class classification directly
                multi_input = {"img": img_tensor}
                if self.size_aware and npb_tensor is not None:
                    multi_input["npb"] = npb_tensor
                
                multi_logits = self.multi_classifier.model(multi_input)
                multi_probs = torch.nn.functional.softmax(multi_logits, dim=1)
                multi_pred = torch.argmax(multi_logits, dim=1).item()
                multi_confidence = multi_probs[0, multi_pred].item()
                result['multi_class'] = self.MULTI_CLASSES[multi_pred]
                result['multi_confidence'] = multi_confidence
                
                return result
            
            # Normal flow or binary_only mode: run binary classification first
            binary_input = {"img": img_tensor}
            if self.size_aware and npb_tensor is not None:
                binary_input["npb"] = npb_tensor
            
            binary_logits = self.binary_classifier.model(binary_input)
            binary_probs = torch.nn.functional.softmax(binary_logits, dim=1)
            binary_pred = torch.argmax(binary_logits, dim=1).item()
            binary_confidence = binary_probs[0, binary_pred].item()
            binary_class = self.BINARY_CLASSES[binary_pred]
            
            result['binary_class'] = binary_class
            result['binary_confidence'] = binary_confidence
            
            # If binary_only mode, skip multi-class
            if self.binary_only:
                return result
            
            # Normal flow: run multi-class classification (only if arthropod)
            if binary_class == 'arthropod':
                multi_input = {"img": img_tensor}
                if self.size_aware and npb_tensor is not None:
                    multi_input["npb"] = npb_tensor
                
                multi_logits = self.multi_classifier.model(multi_input)
                multi_probs = torch.nn.functional.softmax(multi_logits, dim=1)
                multi_pred = torch.argmax(multi_logits, dim=1).item()
                multi_confidence = multi_probs[0, multi_pred].item()
                result['multi_class'] = self.MULTI_CLASSES[multi_pred]
                result['multi_confidence'] = multi_confidence
            
            return result
            
        except Exception as e:
            print(f"Error classifying {crop_path}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'binary_class': 'error',
                'binary_confidence': 0.0,
                'multi_class': None,
                'multi_confidence': None
            }
    
    def _get_final_class(self, prediction):
        """
        Determine final class label from both stage predictions.
        
        Parameters
        ----------
        prediction : dict
            Prediction dictionary from _classify_crop
            
        Returns
        -------
        str : Final class label
        """
        if prediction['binary_class'] == 'debris':
            return 'debris'
        else:
            return prediction['multi_class'] if prediction['multi_class'] else 'unidentified'
    
    def _draw_bbox(self, image, bbox, label, confidence):
        """
        Draw a bounding box with label on an image.
        
        Parameters
        ----------
        image : PIL.Image
            Image to draw on
        bbox : list
            [x, y, width, height] in COCO format
        label : str
            Class label
        confidence : float
            Confidence score
        """
        draw = ImageDraw.Draw(image)
        
        # Convert COCO format [x, y, w, h] to corner coordinates
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        # Get color for this class
        color = self.CLASS_COLORS.get(label, (255, 255, 255))
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        
        # Draw label
        label_text = f"{label}\n{confidence:.2f}"
        try:
            # Try to use a nice font if available
            font = ImageFont.truetype("arial.ttf", 30)
        except:
            font = ImageFont.load_default()
        
        # Draw text background
        text_bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 20), label_text, fill=(255, 255, 255), font=font)
    
    def run_inference(self, confidence_threshold=0.5):
        """
        Run inference on all crops and produce annotated scans.
        
        Parameters
        ----------
        confidence_threshold : float
            Minimum confidence to include annotation
            
        Returns
        -------
        dict : Summary statistics
        """
        predictions_by_scan = defaultdict(list)
        
        # Iterate through crop folders
        crop_folders = [d for d in self.crops_dir.iterdir() if d.is_dir() and (d / 'crops').exists()]
        
        print(f"\nRunning inference on {len(crop_folders)} scans...")
        
        for crop_folder in tqdm(crop_folders, desc="Processing scans"):
            scan_name = crop_folder.name
            crops_path = crop_folder / 'crops'
            
            if not crops_path.exists():
                continue
            
            # Get image info from COCO data
            if scan_name not in self.crop_to_image_mapping:
                print(f"Warning: {scan_name} not found in COCO metadata")
                continue
            
            image_id, image_filename, _ = self.crop_to_image_mapping[scan_name]
            annotations = self.image_id_to_annotations[image_id]
            
            # Classify each crop
            for idx, annotation in enumerate(annotations):
                crop_filename = f"crop_{idx:06d}.png"  # Adjust based on your naming
                crop_path = crops_path / crop_filename
                
                # Try to find the actual crop file
                crop_files = list(crops_path.glob(f"*{idx:06d}*"))
                if not crop_files:
                    crop_files = list(crops_path.glob("*.png"))
                    if idx < len(crop_files):
                        crop_path = crop_files[idx]
                    else:
                        continue
                else:
                    crop_path = crop_files[0]
                
                if not crop_path.exists():
                    continue
                
                # Run inference
                prediction = self._classify_crop(crop_path)
                final_class = self._get_final_class(prediction)
                confidence = prediction['multi_confidence'] if prediction['multi_class'] else prediction['binary_confidence']
                
                # Store prediction with annotation data
                predictions_by_scan[scan_name].append({
                    'annotation_id': annotation['id'],
                    'bbox': annotation['bbox'],
                    'class': final_class,
                    'confidence': confidence,
                    'binary_class': prediction['binary_class'],
                    'binary_confidence': prediction['binary_confidence'],
                    'multi_class': prediction['multi_class'],
                    'multi_confidence': prediction['multi_confidence']
                })
        
        # Load original scans and draw annotations
        print("\nCreating annotated scans...")
        self._annotate_scans(predictions_by_scan, confidence_threshold)
        
        # Generate summary statistics
        stats = self._generate_statistics(predictions_by_scan)
        
        return stats
    
    def _annotate_scans(self, predictions_by_scan, confidence_threshold):
        """
        Load original scans and draw predictions as bounding boxes.
        
        Parameters
        ----------
        predictions_by_scan : dict
            {scan_name: [predictions]}
        confidence_threshold : float
            Minimum confidence to draw
        """
        # Increase PIL's maximum image size limit for high-resolution scans
        # Default limit is ~179 million pixels, but scientific images can exceed this
        # Setting to None removes the limit entirely (safe for legitimate high-res images)
        Image.MAX_IMAGE_PIXELS = None
        
        for scan_name, predictions in tqdm(predictions_by_scan.items(), desc="Annotating scans"):
            # Get scan information
            if scan_name not in self.crop_to_image_mapping:
                continue
            
            _, image_filename, _ = self.crop_to_image_mapping[scan_name]
            
            # Try to find the scan file - might be with or without extension
            scan_path = None
            
            # Method 1: Use the exact filename from COCO
            potential_path = self.scans_dir / image_filename
            if potential_path.exists():
                scan_path = potential_path
            
            # Method 2: Try to find by scan name (might have different extension)
            if not scan_path:
                # Look for any image file matching the scan name
                for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    potential_path = self.scans_dir / f"{scan_name}{ext}"
                    if potential_path.exists():
                        scan_path = potential_path
                        break
            
            # Method 3: Check in subdirectories (in case scans are organized by folder)
            if not scan_path:
                for potential_path in self.scans_dir.rglob(f"{scan_name}*"):
                    if potential_path.is_file() and potential_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                        scan_path = potential_path
                        break
            
            if not scan_path:
                continue
            
            # Load original scan
            image = Image.open(scan_path).convert('RGB')
            
            # Draw annotations (only for classes in annotate_classes list)
            for pred in predictions:
                if pred['confidence'] >= confidence_threshold and pred['class'] in self.annotate_classes:
                    self._draw_bbox(image, pred['bbox'], pred['class'], pred['confidence'])
            
            # Save annotated image
            output_path = self.output_dir / f"annotated_{scan_name}.jpg"
            image.save(output_path, quality=95)
    
    def _generate_statistics(self, predictions_by_scan):
        """
        Generate summary statistics from predictions.
        
        Parameters
        ----------
        predictions_by_scan : dict
            {scan_name: [predictions]}
            
        Returns
        -------
        dict : Statistics summary
        """
        stats = {
            'total_scans': len(predictions_by_scan),
            'total_detections': sum(len(preds) for preds in predictions_by_scan.values()),
            'annotated_detections': 0,
            'class_counts': defaultdict(int),
            'annotated_class_counts': defaultdict(int),
            'confidence_stats': {},
            'annotate_classes': self.annotate_classes,
            'timestamp': datetime.now().isoformat()
        }
        
        all_predictions = []
        for predictions in predictions_by_scan.values():
            all_predictions.extend(predictions)
        
        # Count by class
        for pred in all_predictions:
            stats['class_counts'][pred['class']] += 1
            # Count annotated classes separately
            if pred['class'] in self.annotate_classes:
                stats['annotated_class_counts'][pred['class']] += 1
                stats['annotated_detections'] += 1
        
        # Confidence statistics
        for class_name in self.CLASS_COLORS.keys():
            class_preds = [p['confidence'] for p in all_predictions if p['class'] == class_name]
            if class_preds:
                stats['confidence_stats'][class_name] = {
                    'mean': float(np.mean(class_preds)),
                    'std': float(np.std(class_preds)),
                    'min': float(np.min(class_preds)),
                    'max': float(np.max(class_preds))
                }
        
        return stats
    
    def save_results(self, stats, results_dir=None):
        """
        Save inference results and statistics to JSON.
        
        Parameters
        ----------
        stats : dict
            Statistics dictionary
        results_dir : str, optional
            Directory to save results (default: output_dir)
        """
        if results_dir is None:
            results_dir = self.output_dir
        
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdict to regular dict for JSON serialization
        stats_json = {
            'total_scans': stats['total_scans'],
            'total_detections': stats['total_detections'],
            'annotated_detections': stats['annotated_detections'],
            'class_counts': dict(stats['class_counts']),
            'annotated_class_counts': dict(stats['annotated_class_counts']),
            'annotate_classes': stats['annotate_classes'],
            'skipped_classes': [c for c in stats['class_counts'].keys() if c not in stats['annotate_classes']],
            'confidence_stats': stats['confidence_stats'],
            'timestamp': stats['timestamp']
        }
        
        # Save statistics
        stats_path = results_dir / 'inference_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats_json, f, indent=2)
        
        print(f"\nResults saved to {results_dir}")
        print(f"Statistics: {stats_json}")


def main():
    parser = argparse.ArgumentParser(description="Run SWD detection inference on flatbug crops")
    parser.add_argument('-b', '--binary_model', type=str, required=True,
                        help='Path to binary classifier model checkpoint')
    parser.add_argument('-m', '--multi_model', type=str, required=True,
                        help='Path to multi-class classifier model checkpoint')
    parser.add_argument('-c', '--crops_dir', type=str, required=True,
                        help='Path to flatbug crops directory (containing coco_instances.json)')
    parser.add_argument('-s', '--scans_dir', type=str, required=True,
                        help='Path to original scans directory')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Output directory for annotated scans')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='Confidence threshold for annotations (default: 0.5)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu), default: auto')
    parser.add_argument('--arch', type=str, default='resnet50',
                        help='Model architecture used during training. Options: resnet50, dinov2_vitb14, dinov2_vitl14, etc. '
                             'Default: resnet50. Must match the architecture used to train your models.')
    parser.add_argument('--size_aware', action='store_true',
                        help='Use this flag if your models were trained with size awareness (-a flag). '
                             'Required if training used: -a or --size_aware')
    parser.add_argument('--annotate_classes', type=str, nargs='+', default=None,
                        help='Classes to annotate (e.g., SWD_male SWD_parasitoid SBW). '
                             'If not specified, all classes are annotated. '
                             'Use this to skip unidentified arthropods: --annotate_classes SWD_male SWD_parasitoid SBW')
    parser.add_argument('--multi_only', action='store_true',
                        help='Skip binary classification and run only multi-class model. '
                             'Assumes all crops are arthropods. Useful for testing/debugging.')
    parser.add_argument('--binary_only', action='store_true',
                        help='Run only binary classification (arthropod vs debris). '
                             'Useful for testing/debugging the first stage.')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create pipeline
    pipeline = SWDAnnotationPipeline(
        binary_model_path=args.binary_model,
        multi_model_path=args.multi_model,
        crops_dir=args.crops_dir,
        scans_dir=args.scans_dir,
        output_dir=args.output_dir,
        device=device,
        annotate_classes=args.annotate_classes,
        arch=args.arch,
        size_aware=args.size_aware,
        multi_only=args.multi_only,
        binary_only=args.binary_only
    )
    
    # Run inference
    stats = pipeline.run_inference(confidence_threshold=args.threshold)
    
    # Save results
    pipeline.save_results(stats)


if __name__ == '__main__':
    main()