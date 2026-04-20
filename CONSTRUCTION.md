# 🏗️ Estado de Construcción - Dynamic Video Summarizer

Este archivo detalla el progreso técnico del sistema, mapeando el código desarrollado con la arquitectura definida en `ARCHITECTURE.rtf`.

## 📈 Resumen de Progreso (Pipeline Base)
- [x] **Módulo 1: Detección y División** (100%)
- [x] **Módulo 2: Extracción de Características** (100%)
- [x] **Módulo 3: Redes de Predicción** (100%)
- [x] **Módulo 4: Selección y Construcción** (100%)
- [x] **Entrenamiento y Evaluación** (100% - Pipeline implementado)

---

## 📂 Mapeo de Archivos por Módulo

### 🔹 Módulo 1: Shot Boundary Detection & Splitting
*Responsabilidad: Descomponer el video original en unidades lógicas.*
- **`src/data_loader/video_processor.py`**: 
    - Extrae frames a 2 FPS (RGB).
    - Extrae audio a 16kHz (Mono).
- **`src/data_loader/kts_segmentation.py`**: 
    - Algoritmo KTS para generar límites de "shots" ($s_1, s_2, ..., s_n$).

### 🔹 Módulo 2: Feature Extraction
*Responsabilidad: Convertir datos crudos en tensores matemáticos profundos.*
- **`src/feature_extraction/visual_extractor.py`**: 
    - **Modelos:** ResNet50 + InceptionV3.
    - **Salida:** Vector de 4096 dimensiones por frame (Early Fusion).
- **`src/feature_extraction/audio_extractor.py`**: 
    - **Modelos:** VGGish + MFCC + Mel-Spectrogram.
    - **Salida:** Vector de 296 dimensiones (Fusionado).
- **`src/feature_extraction/__init__.py`**: Pipeline coordinador que alinea audio y video.

### 🔹 Módulo 3: Score Prediction (ANNs)
*Responsabilidad: Evaluar la importancia de cada frame.*
- **`src/models/visual_ann.py`**: ANN de 3 capas para puntaje visual.
- **`src/models/audio_ann.py`**: ANN profunda (2048-1024-2048-1024) para puntaje de audio.
- **`src/models/fusion_network.py`**: 
    - **`FusionScorer`**: Calcula $AV\_Score_j = (V\_Score + A\_Score) / 2$.
    - **`ScorePredictor`**: Genera el puntaje final por shot promediando sus frames.

### 🔹 Módulo 4: Key Shots Extraction & Construction
*Responsabilidad: Filtrar shots y renderizar el video resumen.*
- **`src/summarization/video_summarizer.py`**: 
    - Implementa $Score_{th} = \frac{1}{n} \sum AV\_Score_{s_i}$.
    - Selección de shots mediante $AV\_Score_{s_i} \geq Score_{th}$.
    - Renderizado final MP4 usando `moviepy`.

### 🔹 Módulo 5: Entrenamiento y Evaluación
*Responsabilidad: Entrenar las redes y evaluar métricas de calidad.*
- **`src/config.py`**: Configuración centralizada (paths, hiperparámetros, device).
- **`src/data_loader/dataset.py`**: 
    - PyTorch `Dataset` que carga features pre-extraídos desde archivos `.h5`.
    - Soporta splits de entrenamiento/validación/test.
- **`scripts/extract_features.py`**: 
    - Extracción batch de features de videos a archivos `h5py`.
    - Cachea ResNet50 + InceptionV3 + VGGish para evitar re-procesamiento.
- **`scripts/train.py`**: 
    - Bucle de entrenamiento con MSE Loss.
    - Early stopping, LR scheduler, checkpoints (best + latest).
- **`scripts/evaluate.py`**: 
    - Métricas: F-score, Precision, Recall.
    - Selección de frames top-k vs ground truth.
    - Exporta resultados a JSON.

---

## 🛠️ Requisitos Técnicos instalados
- **Core:** `torch`, `torchvision`, `torchaudio`.
- **Multimedia:** `opencv-python`, `moviepy`, `librosa`.
- **Data:** `numpy`, `h5py`.
- **Training:** `tqdm` (para progress bars en scripts).

---

## 🚀 Flujo de Entrenamiento Completo

### 1. Preparar Dataset
Coloca videos en `data/raw/videos/` y anotaciones (ground truth scores) en los archivos `.h5`.

### 2. Extraer Features
```bash
uv run python scripts/extract_features.py \
    --input_dir data/raw/videos \
    --output_dir data/features
```

### 3. Crear Splits
Crear archivos de texto con IDs de videos:
- `data/train_split.txt`
- `data/val_split.txt`
- `data/test_split.txt`

### 4. Entrenar
```bash
uv run python scripts/train.py \
    --features_dir data/features \
    --train_split data/train_split.txt \
    --val_split data/val_split.txt \
    --epochs 50
```

### 5. Evaluar
```bash
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --split data/test_split.txt \
    --features_dir data/features
```

---

## 🚀 Próximos Pasos (Pendientes)
1. **`data/raw/`**: Cargar videos del dataset TVSum/SumMe para pruebas reales.
2. **Integrar anotaciones**: Conectar ground truth scores de TVSum/SumMe a los archivos `.h5`.
3. **Entrenamiento real**: Ejecutar pipeline completo con datos reales.

---
**Última actualización:** 2026-04-20
