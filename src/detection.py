"""
Object detection and breed classification pipeline
"""
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import numpy as np


class DogDetectionClassifier:
    """Complete pipeline for dog detection and breed classification"""
    
    def __init__(self, yolo_model, classifier_model, label_encoder, device='cuda'):
        self.yolo_model = yolo_model
        self.classifier_model = classifier_model
        self.label_encoder = label_encoder
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Image transforms for classification
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def detect_and_classify(self, image_path_or_pil, confidence_threshold=0.5):
        """
        Detect dogs and classify their breeds
        
        Args:
            image_path_or_pil: Path to image file or PIL Image object
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            PIL Image with bounding boxes and breed labels
        """
        # Load image
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil).convert('RGB')
        else:
            img = image_path_or_pil.convert('RGB')
        
        # Get YOLO detections
        detections = self.yolo_model(img)
        
        # Prepare for drawing
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(30)
        
        detection_results = []
        
        # Process each detection
        for i, box in enumerate(detections[0].boxes.xyxy):
            # Check if it's a dog (class 16 in COCO)
            if int(detections[0].boxes.cls[i]) == 16:
                confidence = float(detections[0].boxes.conf[i])
                
                if confidence >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    
                    # Crop the detected dog
                    cropped_img = img.crop((x1, y1, x2, y2))
                    
                    # Classify the breed
                    breed = self._classify_breed(cropped_img)
                    
                    # Store detection result
                    detection_results.append({
                        'bbox': [x1, y1, x2, y2],
                        'breed': breed,
                        'confidence': confidence
                    })
                    
                    # Draw bounding box and label
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                    label_text = f"{breed} ({confidence:.2f})"
                    
                    # Get the bounding box of the text
                    bbox = draw.textbbox((x1, y1), label_text, font=font)
                    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    
                    # Ensure the text does not go out of bounds
                    label_x = x1
                    label_y = y1 - text_height - 5
                    
                    # If the label goes out of bounds on the right
                    if label_x + text_width > img.width:
                        label_x = img.width - text_width - 5
                    
                    # If the label goes above the image
                    if label_y < 0:
                        label_y = y1 + 5
                    
                    # Draw label text
                    draw.text((label_x, label_y), label_text, fill="green", font=font)

        return img, detection_results
    
    def _classify_breed(self, cropped_image):
        """Classify the breed of a cropped dog image"""
        # Transform image for model
        img_tensor = self.transform(cropped_image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.classifier_model(img_tensor)
            _, predicted = torch.max(output, 1)
            predicted = predicted.item()
            
            # Fix for missing American Spaniel class
            if predicted > 3:
                predicted = predicted - 1
                
            predicted_label = self.label_encoder.classes_[predicted]
            
        return predicted_label
    
    def detect_only(self, image_path_or_pil, confidence_threshold=0.5):
        """Only detect dogs without classification"""
        # Load image
        if isinstance(image_path_or_pil, str):
            img = Image.open(image_path_or_pil).convert('RGB')
        else:
            img = image_path_or_pil.convert('RGB')
        
        # Get YOLO detections
        detections = self.yolo_model(img)
        
        detection_results = []
        
        # Process each detection
        for i, box in enumerate(detections[0].boxes.xyxy):
            # Check if it's a dog (class 16 in COCO)
            if int(detections[0].boxes.cls[i]) == 16:
                confidence = float(detections[0].boxes.conf[i])
                
                if confidence >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    
                    detection_results.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': 'dog'
                    })
        
        return detection_results
