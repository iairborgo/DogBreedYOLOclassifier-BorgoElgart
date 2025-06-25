"""
Gradio App 1: Similarity Search using Vector Database
Based on the original demo from the notebook
"""

import gradio as gr
import sys
import os
from PIL import Image

# Add parent directory to path to import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import ModelManager
from src.search import VectorSearch
from src.embeddings import EmbeddingExtractor


def initialize_components():
    """Initialize all required components for the app"""
    # Initialize model manager
    model_manager = ModelManager()
    model_manager.load_all_models()
    
    # Initialize vector search
    vector_search = VectorSearch()
    
    # Initialize embedding extractor
    embedding_extractor = EmbeddingExtractor()
    
    # Get pretrained ResNet50 model
    resnet50_model = model_manager.get_model('resnet50_pretrained')
    
    return model_manager, vector_search, embedding_extractor, resnet50_model


def search_similar_images(image_input):
    """
    Main function to search for similar images using ResNet50 embeddings
    """
    if image_input is None:
        return "Por favor, sube una imagen primero.", [], ""
    
    try:
        # Search for similar images using ResNet50
        results = vector_search.search_with_resnet50(
            image_input, 
            resnet50_model, 
            n_results=10
        )
        
        # Load similar images
        similar_images = vector_search.load_similar_images(results)
        
        # Get majority vote for breed
        detected_breed = vector_search.majority_vote(results)
        
        if not similar_images:
            return "No se encontraron imágenes similares.", [], ""
        
        message = f"Se encontraron {len(similar_images)} imágenes similares:"
        return message, similar_images, detected_breed
        
    except Exception as e:
        return f"Error procesando la imagen: {str(e)}", [], ""


def create_gallery_images(images):
    """Helper function to format images for gallery"""
    if not images:
        return []
    return images


# Initialize components
try:
    model_manager, vector_search, embedding_extractor, resnet50_model = initialize_components()
except Exception as e:
    print(f"Error initializing components: {e}")
    print("Make sure you have extracted chroma.zip and have all model files in data/models/")
    sys.exit(1)


# Create Gradio interface
with gr.Blocks(title="Búsqueda de Imágenes Similares", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🔍 Búsqueda de Imágenes Similares")
    gr.Markdown("Arrastra o selecciona una imagen para encontrar imágenes similares en la base de datos usando ResNet50 preentrenado.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Image input component
            image_input = gr.Image(
                label="Sube tu imagen aquí",
                type="pil",
                height=300
            )
            
            # Search button
            search_btn = gr.Button("🔍 Buscar Imágenes Similares", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            # Status message
            message_output = gr.Textbox(
                label="Estado",
                interactive=False,
                lines=2
            )
            
            # Gallery for results
            gallery_output = gr.Gallery(
                label="Imágenes Similares Encontradas",
                show_label=True,
                elem_id="gallery",
                columns=5,
                rows=2,
                height="auto"
            )
            
            # Detected breed output
            breed_output = gr.Textbox(
                label="Raza Detectada (voto mayoritario)",
                interactive=False
            )
    
    # Connect button to function
    search_btn.click(
        fn=search_similar_images,
        inputs=[image_input],
        outputs=[message_output, gallery_output, breed_output]
    )
    
    # Auto-search when image is uploaded
    image_input.change(
        fn=search_similar_images,
        inputs=[image_input],
        outputs=[message_output, gallery_output, breed_output]
    )
    
    gr.Markdown("### Instrucciones")
    gr.Markdown("- Sube una imagen de un perro")
    gr.Markdown("- El sistema buscará imágenes similares usando embeddings de ResNet50")
    gr.Markdown("- Se mostrará la raza detectada por voto mayoritario")


if __name__ == "__main__":
    demo.launch(debug=True, share=True)