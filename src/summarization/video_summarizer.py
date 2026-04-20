"""
video_summarizer.py
Pipeline Unificado - Módulo 4: Selección de Shots y Generación de Video
Basado en las Ecuaciones 1 y 2 de Gamal El-Nagar et al. (2024)
"""

import numpy as np
from moviepy import ImageSequenceClip, AudioArrayClip
import os

class DynamicVideoSummarizer:
    def __init__(self, target_fps=2.0, audio_sr=16000):
        """
        Maneja la lógica de selección y la exportación multimedia.
        """
        self.fps = target_fps
        self.audio_sr = audio_sr

    def select_key_shots(self, scored_shots):
        """
        Aplica las Ecuaciones 1 y 2 del paper para filtrar los shots importantes.
        scored_shots: Lista de diccionarios que sale del Módulo 3.
        """
        print("\n🔍 Evaluando importancia de los shots...")
        
        if not scored_shots:
            print("⚠️ No hay shots para evaluar.")
            return []

        # Ecuación 1: Calcular el Score_th (Promedio de todos los AV_Score)
        all_scores = [shot["shot_score"] for shot in scored_shots]
        score_th = np.mean(all_scores)
        print(f"Umbral de selección calculado (Score_th): {score_th:.4f}")

        # Ecuación 2: Selección
        selected_shots = []
        discarded_count = 0
        
        for shot in scored_shots:
            if shot["shot_score"] >= score_th:
                selected_shots.append(shot)
            else:
                discarded_count += 1
                
        print(f"✅ Shots seleccionados: {len(selected_shots)} | ❌ Shots descartados: {discarded_count}")
        return selected_shots

    def build_summary_video(self, selected_shots, output_path="output_summary.mp4"):
        """
        Ensambla el Dynamic Summarized Video (DSV) final.
        Toma los arrays de frames y audio, los concatena y exporta.
        """
        print(f"\n🎬 Construyendo el video resumen: {output_path}...")
        
        if not selected_shots:
            print("⚠️ No hay shots seleccionados para construir el video.")
            return

        # 1. Recopilar todos los frames y audios en orden
        # Aseguramos que se mantenga el orden temporal original
        sorted_shots = sorted(selected_shots, key=lambda x: x["shot_id"])
        
        final_frames = []
        final_audio = []
        
        for shot in sorted_shots:
            if shot.get("frames") is not None:
                # extend añade los elementos de la lista/array uno por uno
                final_frames.extend(shot["frames"])
            
            if shot.get("audio") is not None:
                final_audio.extend(shot["audio"])

        if not final_frames:
            print("⚠️ Faltan los datos visuales (frames) en los shots.")
            return

        # 2. Construir el Clip de Video
        # MoviePy espera una lista de arrays RGB
        video_clip = ImageSequenceClip(final_frames, fps=self.fps)

        # 3. Construir el Clip de Audio (si existe)
        if len(final_audio) > 0:
            audio_np = np.array(final_audio)
            
            # Normalizar audio para evitar recortes (clipping)
            max_amp = np.max(np.abs(audio_np))
            if max_amp > 1.0:
                audio_np = audio_np / max_amp
                
            # MoviePy espera el audio en formato (N, 1) para mono o (N, 2) para estéreo
            if len(audio_np.shape) == 1:
                audio_np = audio_np.reshape(-1, 1)
            
            audio_clip = AudioArrayClip(audio_np, fps=self.audio_sr)
            
            # Recortar el audio a la duración exacta del video para evitar desincronización
            audio_clip = audio_clip.with_duration(video_clip.duration)
            video_clip = video_clip.with_audio(audio_clip)

        # 4. Exportar
        # Usamos libx264 para máxima compatibilidad MP4 y aac para audio
        video_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac" if len(final_audio) > 0 else None,
            fps=self.fps,
            logger=None # Apaga los logs verbosos de MoviePy
        )
        print(f"🎊 ¡Video resumen exportado exitosamente! Duración: {video_clip.duration:.2f} segundos.")

if __name__ == "__main__":
    # Simulación para prueba rápida
    frames_shot_1 = [np.full((224, 224, 3), (255, 0, 0), dtype=np.uint8)] * 10 
    frames_shot_2 = [np.full((224, 224, 3), (0, 255, 0), dtype=np.uint8)] * 10 
    
    audio_fake = np.random.uniform(-0.5, 0.5, 16000 * 5)

    mock_scored_shots = [
        {"shot_id": 0, "shot_score": 0.3, "frames": frames_shot_1, "audio": audio_fake},
        {"shot_id": 1, "shot_score": 0.9, "frames": frames_shot_2, "audio": audio_fake}
    ]

    summarizer = DynamicVideoSummarizer(target_fps=2.0)
    selected = summarizer.select_key_shots(mock_scored_shots)
    summarizer.build_summary_video(selected, "test_dsv_output.mp4")
