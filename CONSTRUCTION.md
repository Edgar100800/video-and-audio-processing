# 🏗️ Estado de Construcción - Dynamic Video Summarizer

Este archivo detalla el progreso técnico del sistema, mapeando el código desarrollado con la arquitectura definida en `ARCHITECTURE.rtf`.

## 📈 Resumen de Progreso (Pipeline Base)
- [x] **Módulo 1: Detección y División** (100%)
- [x] **Módulo 2: Extracción de Características** (100%)
- [x] **Módulo 3: Redes de Predicción** (100%)
- [x] **Módulo 4: Selección y Construcción** (100%)
- [ ] **Entrenamiento y Evaluación** (En espera)

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

---

## 🛠️ Requisitos Técnicos instalados
- **Core:** `torch`, `torchvision`, `torchaudio`.
- **Multimedia:** `opencv-python`, `moviepy`, `librosa`.
- **Data:** `numpy`, `h5py`.

---

## 🚀 Próximos Pasos (Pendientes)
1. **`scripts/train.py`**: Implementar bucle de entrenamiento con Loss MSE.
2. **`scripts/evaluate.py`**: Implementar F-score, Precision y Recall.
3. **`data/raw/`**: Cargar videos del dataset TVSum/SumMe para pruebas reales.

---
**Última actualización:** $(date)
