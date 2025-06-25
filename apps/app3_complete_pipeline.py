"""
Gradio App 3: Complete Pipeline - YOLO Detection + Breed Classification
Based on demo3 from the notebook
"""

import gradio as gr
import sys
import os
import tempfile
from PIL import Image
import torch

# Add parent directory to path to import src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import ModelManager
from src.detection import DogDetectionClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def initialize_components():
    """Initialize all required components for the app"""
    # Initialize model manager
    model_manager = ModelManager()
    model_manager.load_all_models()
    
    # Get required models
    yolo_model = model_manager.get_model('yolo')
    classifier_model = model_manager.get_model('transfer')  # Best model from Stage 2
    label_encoder = model_manager.get_label_encoder()
    
    # Initialize detection pipeline
    detection_pipeline = DogDetectionClassifier(
        yolo_model=yolo_model,
        classifier_model=classifier_model,
        label_encoder=label_encoder,
        device=device
    )
    
    return detection_pipeline


def process_complete_image(image_input, confidence_threshold=0.5):
    """
    Process image with complete pipeline: YOLO detection + breed classification
    """
    if image_input is None:
        return "Por favor, sube una imagen primero.", None
    
    try:
        # Process image with detection pipeline
        processed_image, _ = detection_pipeline.detect_and_classify(
            image_input, 
            confidence_threshold=confidence_threshold
        )
        
        message = "✅ Procesamiento completado: Se detectaron perros y se clasificaron sus razas"
        return message, processed_image
        
    except Exception as e:
        return f"❌ Error procesando la imagen: {str(e)}", None


def process_detection_only(image_input, confidence_threshold=0.5):
    """
    Process image with detection only (no classification)
    """
    if image_input is None:
        return "Por favor, sube una imagen primero.", None
    
    try:
        # Process image with detection only
        processed_image, _ = detection_pipeline.detect_only(
            image_input,
            confidence_threshold=confidence_threshold
        )
        
        message = "✅ Detección completada: Se detectaron perros (sin clasificación)"
        return message, processed_image
        
    except Exception as e:
        return f"❌ Error procesando la imagen: {str(e)}", None


# Initialize components
try:
    detection_pipeline = initialize_components()
except Exception as e:
    print(f"Error initializing components: {e}")
    print("Make sure you have all model files in data/models/")
    sys.exit(1)


# Create Gradio interface
with gr.Blocks(title="Pipeline Completo YOLO + Clasificador", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🐕 Pipeline Completo: Detección y Clasificación de Razas de Perros")
    gr.Markdown("""
    **Flujo de trabajo:**
    1. Sube una imagen que contenga uno o más perros
    2. YOLO detecta automáticamente todos los perros en la imagen
    3. Para cada perro detectado, se recorta la imagen
    4. El clasificador predice la raza de cada perro
    5. Se muestra la imagen original con bounding boxes y etiquetas de razas
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Image input component
            image_input = gr.Image(
                label="📤 Sube una imagen con perros",
                type="pil",
                height=400
            )
            
            # Confidence threshold slider
            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.5,
                step=0.1,
                label="Umbral de Confianza",
                info="Nivel mínimo de confianza para las detecciones"
            )
            
            # Processing buttons
            with gr.Row():
                process_btn = gr.Button(
                    "🚀 Detectar y Clasificar",
                    variant="primary",
                    size="lg"
                )
                detect_btn = gr.Button(
                    "🔍 Solo Detectar",
                    variant="secondary",
                    size="lg"
                )
        
        with gr.Column(scale=1):
            # Status message
            message_output = gr.Textbox(
                label="📊 Estado del Procesamiento",
                interactive=False,
                lines=2
            )
            
            # Result image
            image_result = gr.Image(
                label="🎯 Resultado: Imagen con Detecciones y Clasificaciones",
                height=400
            )
    
    # Connect buttons to functions
    process_btn.click(
        fn=process_complete_image,
        inputs=[image_input, confidence_slider],
        outputs=[message_output, image_result]
    )
    
    detect_btn.click(
        fn=process_detection_only,
        inputs=[image_input, confidence_slider],
        outputs=[message_output, image_result]
    )
    
    # Auto-process when image is uploaded
    image_input.change(
        fn=process_complete_image,
        inputs=[image_input, confidence_slider],
        outputs=[message_output, image_result]
    )
    
    # Additional information
    with gr.Accordion("ℹ️ Información Adicional", open=False):
        gr.Markdown("### Modelos Utilizados")
        gr.Markdown("- **YOLO**: Detección de objetos (perros)")
        gr.Markdown("- **ResNet18**: Clasificación de razas (transfer learning)")
        gr.Markdown("- **Conjunto de datos**: 70 razas de perros")
        
        gr.Markdown("### Parámetros")
        gr.Markdown("- **Umbral de Confianza**: Nivel mínimo para considerar una detección válida")
        gr.Markdown("- **Clases**: Solo se detectan perros (clase 16 en COCO)")
        
        gr.Markdown("### Salida")
        gr.Markdown("- Bounding boxes verdes alrededor de cada perro detectado")
        gr.Markdown("- Etiqueta con la raza clasificada encima de cada bounding box")


if __name__ == "__main__":
    demo.launch(debug=True, share=True)