import numpy as np

class KTSSegmentor:
    def __init__(self, max_shots=50):
        self.max_shots = max_shots

    def segment_into_shots(self, frame_features):
        """
        Divide el video en 'shots' (escenas).
        
        Args:
            frame_features: Un tensor/array de shape (num_frames, feature_dim).
                            Usualmente se extrae con un modelo ligero como GoogLeNet/ResNet
                            antes de correr el KTS para que el algoritmo sepa dónde la 
                            imagen cambió drásticamente.
        Returns:
            boundaries: Array de índices indicando dónde empieza y termina cada shot.
        """
        print("Calculando matriz de similitud y puntos de cambio (KTS)...")
        
        # ------------------------------------------------------------------
        # AQUÍ SE INTEGRA EL CÓDIGO MATEMÁTICO DE KTS (cpd_auto)
        # La implementación clásica usa la librería 'pinafore/kts'
        # m = int(np.ceil(len(frame_features) / 2.0)) 
        # K = np.dot(frame_features, frame_features.T)
        # cps, scores = cpd_auto(K, self.max_shots, 1)
        # boundaries = np.concatenate(([0], cps, [len(frame_features)]))
        # ------------------------------------------------------------------
        
        # Simulación de boundaries temporales para probar la estructura:
        # Asumimos cortes cada 10 frames (5 segundos a 2 FPS)
        num_frames = frame_features.shape[0]
        if num_frames == 0:
            return np.array([0])
            
        simulated_cps = np.arange(10, num_frames, 10) 
        boundaries = np.concatenate(([0], simulated_cps, [num_frames]))
        
        print(f"✅ Video segmentado en {len(boundaries)-1} shots.")
        return boundaries

    def split_frames_and_audio(self, frames, audio_array, boundaries, fps=2.0, sample_rate=16000):
        """
        Agrupa los frames y el audio en diccionarios correspondientes a cada shot.
        """
        shots = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            
            # Recortar frames
            shot_frames = frames[start_idx:end_idx]
            
            # Recortar audio correspondiente (tiempo en segundos)
            start_time = start_idx / fps
            end_time = end_idx / fps
            
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            shot_audio = None
            if audio_array is not None:
                # Asegurar que no nos pasamos del largo del audio
                actual_end_sample = min(end_sample, len(audio_array))
                actual_start_sample = min(start_sample, actual_end_sample)
                shot_audio = audio_array[actual_start_sample:actual_end_sample]
            
            shots.append({
                "shot_id": i,
                "frames": shot_frames,
                "audio": shot_audio,
                "start_frame": start_idx,
                "end_frame": end_idx
            })
            
        return shots
