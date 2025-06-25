"""
Model loading and initialization utilities
"""
import torch
import torch.nn as nn
import pickle
import fiftyone.zoo as foz
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder


def load_pretrained_resnet50():
    """Load ResNet50 pretrained model from fiftyone zoo"""
    return foz.load_zoo_model('resnext50-32x4d-imagenet-torch')


def load_classification_models(transfer_path='data/models/transfer_model_full.pth',
                             custom_path='data/models/custom_model_full.pth',
                             device='cuda'):
    """Load both transfer learning and custom classification models"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load models
    transfer_model = torch.load(transfer_path, weights_only=False, map_location=device)
    custom_model = torch.load(custom_path, weights_only=False, map_location=device)
    
    # Set to evaluation mode
    transfer_model.eval()
    custom_model.eval()
    
    return transfer_model, custom_model


def load_label_encoder(encoder_path='data/models/label_encoder.pkl'):
    """Load the trained label encoder"""
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder


def create_embedding_models(transfer_model, custom_model, device='cuda'):
    """Create embedding versions of the models (without classification head)"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create embedding models by removing the classification layers
    custom_embedding = nn.Sequential(*list(custom_model.children())[:-4])
    transfer_embedding = nn.Sequential(*list(transfer_model.children())[:-1])
    
    custom_embedding.to(device)
    custom_embedding.eval()
    
    transfer_embedding.to(device)
    transfer_embedding.eval()
    
    return custom_embedding, transfer_embedding


def load_yolo_model(model_path='data/models/yolo12n.onnx'):
    """Load YOLO model for object detection"""
    return YOLO(model_path)


def labels_to_indices(labels_tuple, label_encoder):
    """Convert string labels to indices using the label encoder"""
    return torch.tensor([label_encoder.transform([label])[0] for label in labels_tuple])


class ModelManager:
    """Centralized model management class"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.label_encoder = None
        
    def load_all_models(self):
        """Load all models and components"""
        print("Loading pretrained ResNet50...")
        self.models['resnet50'] = load_pretrained_resnet50()
        
        print("Loading classification models...")
        self.models['transfer'], self.models['custom'] = load_classification_models(device=self.device)
        
        print("Loading label encoder...")
        self.label_encoder = load_label_encoder()
        
        print("Creating embedding models...")
        self.models['custom_embedding'], self.models['transfer_embedding'] = create_embedding_models(
            self.models['transfer'], self.models['custom'], device=self.device
        )
        
        print("Loading YOLO model...")
        self.models['yolo'] = load_yolo_model()
        
        print("All models loaded successfully!")
        
    def get_model(self, model_name):
        """Get a specific model"""
        return self.models.get(model_name)
        
    def get_label_encoder(self):
        """Get the label encoder"""
        return self.label_encoder