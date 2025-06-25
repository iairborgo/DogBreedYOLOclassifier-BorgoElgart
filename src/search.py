"""
Vector similarity search functionality using ChromaDB
"""
import chromadb
from PIL import Image
from collections import Counter
import torch
from torchvision import transforms


class VectorSearch:
    """Vector similarity search using ChromaDB"""
    
    def __init__(self, chroma_path='data/chroma/chroma/content/chroma'):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collections = {}
        self._load_collections()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_collections(self):
        """Load all available collections"""
        try:
            self.collections['train'] = self.client.get_collection(name='dog_train')
            print("Loaded 'dog_train' collection")
        except:
            print("Warning: 'dog_train' collection not found")
            
        try:
            self.collections['custom'] = self.client.get_collection(name='custom_collection')
            print("Loaded 'custom_collection' collection")
        except:
            print("Warning: 'custom_collection' collection not found")
            
        try:
            self.collections['transfer'] = self.client.get_collection(name='transfer_collection')
            print("Loaded 'transfer_collection' collection")
        except:
            print("Warning: 'transfer_collection' collection not found")
    
    def search_with_resnet50(self, image, model, n_results=10):
        """Search using ResNet50 embeddings"""
        if 'train' not in self.collections:
            raise ValueError("Train collection not available")
            
        # Generate embedding
        embedding = model.embed(image)
        
        # Search in collection
        results = self.collections['train'].query(
            query_embeddings=embedding,
            n_results=n_results
        )
        
        return results
    
    def search_with_custom_model(self, image, model, device, n_results=10):
        """Search using custom model embeddings"""
        if 'custom' not in self.collections:
            raise ValueError("Custom collection not available")
            
        # Prepare image tensor
        img_tensor = self.transform(image).unsqueeze(0).to(device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = model(img_tensor).squeeze().tolist()
        
        # Search in collection
        results = self.collections['custom'].query(
            query_embeddings=embedding,
            n_results=n_results
        )
        
        return results
    
    def search_with_transfer_model(self, image, model, device, n_results=10):
        """Search using transfer learning model embeddings"""
        if 'transfer' not in self.collections:
            raise ValueError("Transfer collection not available")
            
        # Prepare image tensor
        img_tensor = self.transform(image).unsqueeze(0).to(device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = model(img_tensor).squeeze().tolist()
        
        # Search in collection
        results = self.collections['transfer'].query(
            query_embeddings=embedding,
            n_results=n_results
        )
        
        return results
    
    def majority_vote(self, results):
        """Determine the most frequent label from search results"""
        try:
            # Extract labels from metadata
            labels = [meta['label'] for meta in results['metadatas'][0] if 'label' in meta]
            
            if not labels:
                return "No labels found in results"
            
            # Count frequency of each label
            counter = Counter(labels)
            most_common_label = counter.most_common(1)[0][0]
            
            return most_common_label
        except Exception as e:
            return f"Error determining majority vote: {str(e)}"
    
    def load_similar_images(self, results):
        """Load images from search results"""
        similar_images = []
        
        try:
            for original_filepath in results['documents'][0]:

                # Fix for old stored paths
                parts = original_filepath.split('/')
                breed = parts[-2]
                filename = parts[-1]
                filepath = f"data/images/{breed}/{filename}"
                
                try:
                    img = Image.open(filepath)
                    similar_images.append(img)
                except Exception as e:
                    print(f"Error loading image {filepath}: {e}")
                    continue
                    
            return similar_images
        except Exception as e:
            print(f"Error processing results: {e}")
            return []


def search_similar_images(image, search_engine, model, model_type='resnet50', device='cuda', n_results=10):
    """
    Unified function to search for similar images
    
    Args:
        image: PIL Image to search for
        search_engine: VectorSearch instance
        model: Model to use for embedding generation
        model_type: Type of model ('resnet50', 'custom', 'transfer')
        device: Device to use for computation
        n_results: Number of results to return
    """
    if model_type == 'resnet50':
        results = search_engine.search_with_resnet50(image, model, n_results)
    elif model_type == 'custom':
        results = search_engine.search_with_custom_model(image, model, device, n_results)
    elif model_type == 'transfer':
        results = search_engine.search_with_transfer_model(image, model, device, n_results)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load similar images
    similar_images = search_engine.load_similar_images(results)
    
    # Get majority vote
    predicted_breed = search_engine.majority_vote(results)
    
    return similar_images, predicted_breed, results