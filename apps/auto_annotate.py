"""
Automatic Annotation Script
Processes a folder of images and generates annotations in YOLO and COCO formats
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

# Import project modules
from src.models import ModelManager
from src.detection import DogDetectionClassifier
from src.annotations import AnnotationExporter


def setup_directories(output_folder):
    """Create necessary output directories"""
    output_path = Path(output_folder)
    
    # Create main output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different formats
    yolo_dir = output_path / "yolo_annotations"
    coco_dir = output_path / "coco_annotations"
    
    yolo_dir.mkdir(exist_ok=True)
    coco_dir.mkdir(exist_ok=True)
    
    return output_path, yolo_dir, coco_dir


def get_image_files(input_folder):
    """Get all image files from input folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    input_path = Path(input_folder)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    for file_path in input_path.rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return sorted(image_files)


def process_images(image_files, detection_pipeline, confidence_threshold=0.5):
    """Process all images and collect detection results"""
    detections_dict = {}
    
    print(f"Processing {len(image_files)} images...")
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Get detections using batch processing capability
            detections = detection_pipeline.detect_and_classify(
                image_path,
                confidence_threshold=confidence_threshold
            )
            
            # Store results
            relative_path = str(image_path.name)
            detections_dict[relative_path] = {
                'image_path': str(image_path),
                'image_size': image.size,  # (width, height)
                'detections': detections
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    return detections_dict


def main():
    parser = argparse.ArgumentParser(description='Automatic Dog Breed Annotation Tool')
    parser.add_argument('input_folder', type=str, help='Path to folder containing images')
    parser.add_argument('output_folder', type=str, help='Path to output folder for annotations')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--format', type=str, choices=['yolo', 'coco', 'both'], default='both',
                       help='Output format (default: both)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                       help='Device to use for inference (default: cuda)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist.")
        sys.exit(1)
    
    if args.confidence < 0.1 or args.confidence > 1.0:
        print("Error: Confidence threshold must be between 0.1 and 1.0")
        sys.exit(1)
    
    print("=== Automatic Dog Breed Annotation Tool ===")
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Output format: {args.format}")
    print(f"Device: {args.device}")
    print()
    
    try:
        # Setup output directories
        print("Setting up output directories...")
        output_path, yolo_dir, coco_dir = setup_directories(args.output_folder)
        
        # Initialize models
        print("Loading models...")
        model_manager = ModelManager(device=args.device)
        model_manager.load_all_models()
        
        # Get required models
        yolo_model = model_manager.get_model('yolo')
        classifier_model = model_manager.get_model('transfer_full')
        label_encoder = model_manager.get_label_encoder()
        
        # Initialize detection pipeline
        detection_pipeline = DogDetectionClassifier(
            yolo_model=yolo_model,
            classifier_model=classifier_model,
            label_encoder=label_encoder,
            device=args.device
        )
        
        # Initialize annotation exporter
        annotation_exporter = AnnotationExporter(label_encoder)
        
        # Get image files
        print("Scanning for images...")
        image_files = get_image_files(args.input_folder)
        
        if not image_files:
            print("No image files found in the input folder.")
            sys.exit(1)
        
        print(f"Found {len(image_files)} image files.")
        
        # Process images
        detections_dict = process_images(
            image_files, 
            detection_pipeline, 
            confidence_threshold=args.confidence
        )
        
        if not detections_dict:
            print("No valid detections found.")
            sys.exit(1)
        
        # Export annotations
        print("\nExporting annotations...")
        
        if args.format in ['yolo', 'both']:
            print("Exporting YOLO format...")
            annotation_exporter.export_yolo_format(detections_dict, str(yolo_dir))
            print(f"YOLO annotations saved to: {yolo_dir}")
        
        if args.format in ['coco', 'both']:
            print("Exporting COCO format...")
            coco_file = coco_dir / "annotations.json"
            annotation_exporter.export_coco_format(detections_dict, str(coco_file))
            print(f"COCO annotations saved to: {coco_file}")
        
        # Generate summary
        total_images = len(detections_dict)
        total_detections = sum(len(data['detections']) for data in detections_dict.values() if data['detections'])
        
        print(f"\n=== Summary ===")
        print(f"Processed images: {total_images}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections/total_images:.2f}")
        
        # Save processing summary
        summary_file = output_path / "processing_summary.json"
        summary_data = {
            "input_folder": args.input_folder,
            "output_folder": args.output_folder,
            "confidence_threshold": args.confidence,
            "format": args.format,
            "device": args.device,
            "total_images": total_images,
            "total_detections": total_detections,
            "average_detections_per_image": total_detections/total_images if total_images > 0 else 0,
            "image_files": [str(f) for f in image_files]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Processing summary saved to: {summary_file}")
        print("\n✅ Annotation process completed successfully!")
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()