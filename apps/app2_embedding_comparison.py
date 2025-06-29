"""
Gradio App 2: Embedding Comparison between Custom and Transfer Learning Models
Based on demo2 from the notebook
"""

import gradio as gr
import sys
import os
from PIL import Image
import torch

# Add parent directory to path to import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import ModelManager
from src.search import VectorSearch


def initialize_components():
    """Initialize all required components for the app"""
    # Initialize model manager
    model_manager = ModelManager()
    model_manager.load_all_models()
    
    # Initialize vector search
    vector_search = VectorSearch()
    
    # Get models
    custom_model = model_manager.get_model('custom_embedding')
    transfer_model = model_manager.get_model('transfer_embedding')
    
    return model_manager, vector_search, custom_model, transfer_model


def search_with_selected_model(image_input, model_selected):
    """
    Search for similar images using the selected model
    """
    if image_input is None:
        return "Por favor, sube una imagen primero.", [], ""
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_selected == "Modelo Custom":
            # Search using custom model
            results = vector_search.search_with_custom_model(
                image_input,
                custom_model,
                device=device,
                n_results=10
            )
        else:  # ResNet18 Transfer Learning
            # Search using transfer model
            results = vector_search.search_with_transfer_model(
                image_input,
                transfer_model,
                device=device,
                n_results=10
            )
        
        # Load similar images
        similar_images = vector_search.load_similar_images(results)
        
        # Get majority vote for breed
        detected_breed = vector_search.majority_vote(results)
        
        if not similar_images:
            return f"No se encontraron imágenes similares con {model_selected}.", [], ""
        
        message = f"Búsqueda con {model_selected}: Se encontraron {len(similar_images)} imágenes similares"
        return message, similar_images, detected_breed
        
    except Exception as e:
        return f"Error procesando la imagen: {str(e)}", [], ""


# Initialize components
try:
    model_manager, vector_search, custom_model, transfer_model = initialize_components()
except Exception as e:
    print(f"Error initializing components: {e}")
    print("Make sure you have extracted chroma.zip and have all model files in data/models/")
    sys.exit(1)


# Create Gradio interface
with gr.Blocks(title="Comparación de Embeddings", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🔬 Comparación de Embeddings")
    gr.Markdown("Compara los resultados de búsqueda entre el modelo custom y ResNet18 con Transfer Learning.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Image input component
            image_input = gr.Image(
                label="Sube tu imagen aquí",
                type="pil",
                height=300
            )
            
            # Model selector
            model_selector = gr.Dropdown(
                choices=["Modelo Custom", "ResNet18 Transfer Learning"],
                value="Modelo Custom",
                label="Selecciona el modelo para la búsqueda",
                info="Elige qué modelo usar para generar los embeddings"
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
            
            # Detected breed
            breed_output = gr.Textbox(
                label="Raza Detectada (voto mayoritario)",
                interactive=False
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
    
    # Connect button to function
    search_btn.click(
        fn=search_with_selected_model,
        inputs=[image_input, model_selector],
        outputs=[message_output, gallery_output, breed_output]
    )
    
    # Auto-search when image is uploaded or model is changed
    image_input.change(
        fn=search_with_selected_model,
        inputs=[image_input, model_selector],
        outputs=[message_output, gallery_output, breed_output]
    )
    
    model_selector.change(
        fn=search_with_selected_model,
        inputs=[image_input, model_selector],
        outputs=[message_output, gallery_output, breed_output]
    )
    
    gr.Markdown("### Instrucciones")
    gr.Markdown("- Sube una imagen de un perro")
    gr.Markdown("- Selecciona el modelo que quieres usar para la búsqueda")
    gr.Markdown("- **Modelo Custom**: Red neuronal entrenada desde cero")
    gr.Markdown("- **ResNet18 Transfer Learning**: ResNet18 con fine-tuning")
    gr.Markdown("- Compara los resultados entre ambos modelos")


if __name__ == "__main__":
    demo.launch(debug=True, share=True)