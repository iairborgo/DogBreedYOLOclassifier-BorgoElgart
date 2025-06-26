"""
Annotation export utilities for YOLO and COCO formats
"""
import json
import os
from datetime import datetime
from PIL import Image



class AnnotationExporter:
    """Export annotations in YOLO and COCO formats"""
    
    def __init__(self, label_encoder):
        self.label_encoder = label_encoder
        self.class_mapping = {breed: idx for idx, breed in enumerate(label_encoder.classes_)}
        
    def export_yolo_format(self, detections_dict, output_folder, parent_folder=None):
        """
        Export annotations in YOLO format (.txt files)
        
        Args:
            detections_dict: Dictionary with image paths as keys and detection results as values
            output_folder: Folder to save annotation files
        """
        os.makedirs(output_folder, exist_ok=True)
        
        for image_path, results in detections_dict.items():
            # Get image dimensions
            try:
                image_path = os.path.join(parent_folder, image_path) if parent_folder else image_path
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")
                continue
            
            # Create annotation file
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            annotation_file = os.path.join(output_folder, f"{base_name}.txt")
            
            with open(annotation_file, 'w') as f:
                for detection in results.get('detections', []):
                    bbox = detection['bbox']  # [x1, y1, x2, y2]
                    breed = detection['breed']
                    
                    # Convert to YOLO format (class x_center y_center width height)
                    # All values normalized to [0, 1]
                    x1, y1, x2, y2 = bbox
                    x_center = (x1 + x2) / 2.0 / img_width
                    y_center = (y1 + y2) / 2.0 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # Get class index
                    class_id = self.class_mapping.get(breed, 0)
                    
                    # Write YOLO format line
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Create classes.txt file
        classes_file = os.path.join(output_folder, "classes.txt")
        with open(classes_file, 'w') as f:
            for breed in self.label_encoder.classes_:
                f.write(f"{breed}\n")
        
        print(f"anotaciones YOLO exportadas a {output_folder}")
        
    def export_coco_format(self, detections_dict, output_file, parent_folder=None):
        """
        Export annotations in COCO format (.json file)
        
        Args:
            detections_dict: Dictionary with image paths as keys and detection results as values
            output_file: Path to output JSON file
        """
        coco_data = {
            "info": {
                "description": "Dog Breed Detection Dataset",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Auto-annotation Pipeline",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Create categories
        for idx, breed in enumerate(self.label_encoder.classes_):
            coco_data["categories"].append({
                "id": idx + 1,  # COCO categories start from 1
                "name": breed,
                "supercategory": "dog"
            })
        
        image_id = 1
        annotation_id = 1
        
        for image_path, results in detections_dict.items():
            # Get image info
            try:
                image_path = os.path.join(parent_folder, image_path) if parent_folder else image_path
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")
                continue
            
            # Add image info
            image_info = {
                "id": image_id,
                "width": img_width,
                "height": img_height,
                "file_name": os.path.basename(image_path),
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            }
            coco_data["images"].append(image_info)
            
            # Add annotations for this image
            for detection in results.get('detections', []):
                bbox = detection['bbox']  # [x1, y1, x2, y2]
                breed = detection['breed']
                confidence = detection.get('confidence', 1.0)
                
                # Convert to COCO format (x, y, width, height)
                x1, y1, x2, y2 = bbox
                coco_bbox = [x1, y1, x2 - x1, y2 - y1]
                area = (x2 - x1) * (y2 - y1)
                
                # Get category ID
                category_id = self.class_mapping.get(breed, 0) + 1  # COCO categories start from 1
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [],
                    "area": area,
                    "bbox": coco_bbox,
                    "iscrowd": 0,
                    "score": confidence  # Custom field for detection confidence
                }
                
                coco_data["annotations"].append(annotation)
                annotation_id += 1
            
            image_id += 1
        
        # Save COCO JSON file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"anotaciones COCO exportadas a {output_file}")
        
    def export_both_formats(self, detections_dict, output_folder):
        """Export annotations in both YOLO and COCO formats"""
        # Create subfolders
        yolo_folder = os.path.join(output_folder, "yolo")
        coco_file = os.path.join(output_folder, "coco", "annotations.json")
        
        # Export both formats
        self.export_yolo_format(detections_dict, yolo_folder)
        self.export_coco_format(detections_dict, coco_file)
        
        print(f"Anotaciones en {output_folder}")
        print(f"  - YOLO: {yolo_folder}")
        print(f"  - COCO: {coco_file}")


def create_annotation_exporter(label_encoder):
    """Factory function to create annotation exporter"""
    return AnnotationExporter(label_encoder)