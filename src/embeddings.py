"""
Embedding generation utilities for dog breed classification.
Handles creating embeddings from trained models and pre-trained models.
"""

import torch
import torch.nn as nn
from torchvision import transforms
import fiftyone.zoo as foz
from PIL import Image
import numpy as np


class EmbeddingExtractor:
    """Class for extracting embeddings from different models."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.transform = self._get_transform()
        
    def _get_transform(self):
        """Get the standard transform for image preprocessing."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_from_resnet50_pretrained(self, image, model=None):
        """
        Extract embeddings using pre-trained ResNet50 from FiftyOne zoo.
        
        Args:
            image: PIL Image or path to image
            model: Pre-loaded ResNet50 model (optional)
            
        Returns:
            numpy array: Embedding vector
        """
        if model is None:
            model = foz.load_zoo_model('resnext50-32x4d-imagenet-torch')
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            
        return model.embed(image)
    
    def extract_from_custom_model(self, image, model):
        """
        Extract embeddings from custom CNN model (before final layers).
        
        Args:
            image: PIL Image or path to image
            model: Custom embedding model (without final classification layers)
            
        Returns:
            numpy array: Embedding vector
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract embedding
        model.eval()
        with torch.no_grad():
            embedding = model(img_tensor)
            
        return embedding.squeeze().cpu().numpy()
    
    def extract_from_transfer_model(self, image, model):
        """
        Extract embeddings from transfer learning model (before final layer).
        
        Args:
            image: PIL Image or path to image
            model: Transfer learning embedding model (without final classification layer)
            
        Returns:
            numpy array: Embedding vector
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract embedding
        model.eval()
        with torch.no_grad():
            embedding = model(img_tensor)
            
        return embedding.squeeze().cpu().numpy()
    
    def batch_extract(self, images, model, model_type='custom'):
        """
        Extract embeddings from multiple images in batch.
        
        Args:
            images: List of PIL Images or image paths
            model: The embedding model to use
            model_type: Type of model ('custom', 'transfer', or 'resnet50')
            
        Returns:
            list: List of embedding vectors
        """
        embeddings = []
        
        for image in images:
            if model_type == 'resnet50':
                embedding = self.extract_from_resnet50_pretrained(image, model)
            elif model_type == 'transfer':
                embedding = self.extract_from_transfer_model(image, model)
            else:  # custom
                embedding = self.extract_from_custom_model(image, model)
                
            embeddings.append(embedding)
            
        return embeddings


def create_embedding_extractor(device='cuda'):
    """
    Factory function to create an EmbeddingExtractor instance.
    
    Args:
        device: Device to use for computations ('cuda' or 'cpu')
        
    Returns:
        EmbeddingExtractor: Configured embedding extractor
    """
    return EmbeddingExtractor(device=device)


def extract_embeddings_from_model(model, model_type='custom', device='cuda'):
    """
    Create embedding extraction function for a specific model.
    
    Args:
        model: The model to extract embeddings from
        model_type: Type of model ('custom', 'transfer', or 'resnet50')
        device: Device to use for computations
        
    Returns:
        function: Function that takes an image and returns embeddings
    """
    extractor = EmbeddingExtractor(device=device)
    
    if model_type == 'resnet50':
        return lambda image: extractor.extract_from_resnet50_pretrained(image, model)
    elif model_type == 'transfer':
        return lambda image: extractor.extract_from_transfer_model(image, model)
    else:  # custom
        return lambda image: extractor.extract_from_custom_model(image, model)


def prepare_embedding_models(transfer_model, custom_model, device='cuda'):
    """
    Prepare embedding models by removing final classification layers.
    
    Args:
        transfer_model: Full transfer learning model
        custom_model: Full custom model
        device: Device to move models to
        
    Returns:
        tuple: (custom_embedding_model, transfer_embedding_model)
    """
    # Create embedding models by removing final layers
    custom_embedding = nn.Sequential(*list(custom_model.children())[:-4])
    transfer_embedding = nn.Sequential(*list(transfer_model.children())[:-1])
    
    # Move to device and set to eval mode
    custom_embedding.to(device)
    custom_embedding.eval()
    
    transfer_embedding.to(device)
    transfer_embedding.eval()
    
    return custom_embedding, transfer_embedding


def compute_similarity(embedding1, embedding2, metric='cosine'):
    """
    Compute similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        metric: Similarity metric ('cosine', 'euclidean', 'dot')
        
    Returns:
        float: Similarity score
    """
    if isinstance(embedding1, torch.Tensor):
        embedding1 = embedding1.cpu().numpy()
    if isinstance(embedding2, torch.Tensor):
        embedding2 = embedding2.cpu().numpy()
        
    if metric == 'cosine':
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)
    
    elif metric == 'euclidean':
        # Euclidean distance (lower is more similar)
        return -np.linalg.norm(embedding1 - embedding2)
    
    elif metric == 'dot':
        # Dot product similarity
        return np.dot(embedding1, embedding2)
    
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


class EmbeddingComparator:
    """Class for comparing embeddings from different models."""
    
    def __init__(self, models_dict, device='cuda'):
        """
        Initialize with a dictionary of models.
        
        Args:
            models_dict: Dictionary with model names as keys and models as values
            device: Device to use for computations
        """
        self.models = models_dict
        self.extractor = EmbeddingExtractor(device=device)
        
    def compare_embeddings(self, image, metric='cosine'):
        """
        Compare embeddings from all models for a single image.
        
        Args:
            image: PIL Image or path to image
            metric: Similarity metric to use
            
        Returns:
            dict: Dictionary with model names as keys and embeddings as values
        """
        embeddings = {}
        
        for model_name, (model, model_type) in self.models.items():
            if model_type == 'resnet50':
                embedding = self.extractor.extract_from_resnet50_pretrained(image, model)
            elif model_type == 'transfer':
                embedding = self.extractor.extract_from_transfer_model(image, model)
            else:  # custom
                embedding = self.extractor.extract_from_custom_model(image, model)
                
            embeddings[model_name] = embedding
            
        return embeddings
    
    def compute_cross_similarities(self, embeddings, metric='cosine'):
        """
        Compute similarities between all pairs of embeddings.
        
        Args:
            embeddings: Dictionary of embeddings from compare_embeddings
            metric: Similarity metric to use
            
        Returns:
            dict: Dictionary of similarity scores between model pairs
        """
        similarities = {}
        model_names = list(embeddings.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                similarity = compute_similarity(
                    embeddings[model1], 
                    embeddings[model2], 
                    metric=metric
                )
                similarities[f"{model1}_vs_{model2}"] = similarity
                
        return similarities