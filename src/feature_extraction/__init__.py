import torch
import numpy as np
from .visual_extractor import VisualDeepExtractor
from .audio_extractor import AudioDeepExtractor

class FeatureExtractionPipeline:
    def __init__(self):
        self.visual_extractor = VisualDeepExtractor()
        self.audio_extractor = AudioDeepExtractor()

    def process_shots(self, shots_data):
        """
        Toma la salida del Módulo 1 (lista de diccionarios de shots)
        y añade los vectores característicos a cada shot.
        """
        print("\n🚀 Iniciando procesamiento de características profundas...")
        processed_shots = []

        for shot in shots_data:
            shot_id = shot["shot_id"]
            frames = shot["frames"]
            audio = shot["audio"]
            
            print(f"Procesando Shot {shot_id} ({len(frames)} frames)...")

            # Extracción Visual (Vector por cada frame)
            # Dimensiones: (Num_Frames_en_Shot, 4096)
            visual_features = self.visual_extractor.extract_frame_features(frames)

            # Extracción de Audio (Un vector global para el fragmento de audio del shot)
            # Dimensiones: (Dimensión_Audio_Fusionada,)
            audio_features = self.audio_extractor.extract_audio_features(audio)

            # Para la red de predicción de frames del paper, 
            # necesitamos alinear el vector de audio global a cada frame del shot.
            # Duplicamos el vector de audio para que cada frame tenga el mismo contexto de sonido.
            aligned_audio_features = np.tile(audio_features, (len(frames), 1))

            processed_shots.append({
                "shot_id": shot_id,
                "visual_features": visual_features, # Shape: (N, 4096)
                "audio_features": aligned_audio_features # Shape: (N, Dim_Audio)
            })

        print("✅ Procesamiento de características finalizado.")
        return processed_shots

if __name__ == "__main__":
    # Datos simulados para probar el script
    mock_shots_data = [
        {
            "shot_id": 0,
            "frames": np.random.randint(0, 255, (5, 224, 224, 3), dtype=np.uint8), # 5 frames
            "audio": np.random.randn(16000 * 2) # 2 segundos de audio
        }
    ]

    # Ejecutar pipeline
    pipeline = FeatureExtractionPipeline()
    final_features = pipeline.process_shots(mock_shots_data)

    print("\n--- RESUMEN DE TENSORES ---")
    print(f"Vector Visual por frame shape: {final_features[0]['visual_features'].shape}")
    print(f"Vector Audio alineado por frame shape: {final_features[0]['audio_features'].shape}")
