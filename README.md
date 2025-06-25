# Dog Breed Classification Pipeline

## Project Structure

```
dog-breed-classification/
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
│   ├── embeddings.py
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

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Extract the database:
   ```bash
   cd data/chroma
   unzip chroma.zip
   ```

## Usage

### Similarity Search App (Stage 1)
```bash
python apps/app1_similarity_search.py
```

### Embedding Comparison App (Stage 2)
```bash
python apps/app2_embedding_comparison.py
```

### Complete Pipeline App (Stage 3)
```bash
python apps/app3_complete_pipeline.py
```

### Auto-annotation Script
```bash
python auto_annotate.py --input_folder /path/to/images --output_folder /path/to/annotations
```

## Features

- **Similarity Search**: Find similar dog images using ResNet50 embeddings
- **Custom vs Transfer Learning**: Compare embeddings from custom CNN vs ResNet18
- **Complete Pipeline**: YOLO detection + breed classification
- **Auto-annotation**: Batch processing with YOLO and COCO format export

## Model Performance

- **NDCG@10**: 0.9635 (validation set)
- **Transfer Learning Model**: Best accuracy on validation
- **ONNX Optimization**: ~100% speed improvement for YOLO inference
