# Dog Breed Classification Pipeline

## Project Structure

```
COMPUTERVISION-TPFINAL-BORGOELGART/
├── README.md
├── requirements.txt
├── data/
│   ├── chroma/
│   │   └── chroma.zip (to be extracted)
│   └── models/
│       ├── transfer_model_full.pth
│       ├── custom_model_full.pth
│       ├── label_encoder.pkl
│       └── yolo12n.onnx
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── search.py
│   ├── detection.py
│   └── annotations.py
├── apps/
│   ├── __init__.py
│   ├── app1_similarity_search.py
│   ├── app2_embedding_comparison.py
│   └── app3_complete_pipeline.py
└── auto_annotate.py
```

## Instalacion

1. Clonar repositorio
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Moverse hasta la carpeta del proyecto
4. Ejecutar el script de descarga de datos:
   ```bash
   python prepare_project.py
   ```

## Uso

### ES IMPORTANTE EJECUTARLO DESDE LA CARPETA RAIZ

### App Busqueda Similitud Vectorial (Etapa 1)
```bash
python apps/app1_similarity_search.py
```

### App Embeddings (Etapa 2)
```bash
python apps/app2_embedding_comparison.py
```

### App Pipeline YOLO (Stage 3)
```bash
python apps/app3_complete_pipeline.py
```

### Script con Anotaciones Automaticas
```bash
python auto_annotate.py /path/to/images /path/to/output
```
Este tiene 3 parametros opcionales
- --confidence: para cambiar el % de confianza con el queres que YOLO detecte los bounding box
- --format [yolo, coco, both]: Dependiendo que formato queres que se 
- --device [cuda, cpu]


