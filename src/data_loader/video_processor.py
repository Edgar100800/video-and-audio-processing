import cv2
import os
import numpy as np
import librosa
from moviepy import VideoFileClip

class VideoProcessor:
    def __init__(self, target_fps=2.0):
        """
        Procesador inicial basado en Gamal El-Nagar et al. (2024).
        Reduce la redundancia bajando el framerate a target_fps (2 fps).
        """
        self.target_fps = target_fps

    def extract_frames(self, video_path, output_dir=None):
        """
        Extrae frames del video a 2 FPS y los guarda en disco o memoria.
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return np.array([])

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0:
            print(f"Error: FPS original es 0 en {video_path}")
            return np.array([])
            
        frame_interval = max(1, int(original_fps / self.target_fps))
        
        frames_list = []
        count = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Solo guardamos el frame si coincide con nuestro intervalo (2 FPS)
            if count % frame_interval == 0:
                # Convertir BGR (OpenCV) a RGB (PyTorch standard)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame_rgb)
                
                if output_dir:
                    frame_name = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(frame_name, frame) # OpenCV usa BGR para guardar
                
                saved_count += 1
                
            count += 1

        cap.release()
        print(f"✅ Extraídos {saved_count} frames a {self.target_fps} FPS del video {os.path.basename(video_path)}.")
        return np.array(frames_list)

    def extract_audio(self, video_path, output_audio_path):
        """
        Extrae la pista de audio completa del video a 16kHz (ideal para VGGish).
        """
        try:
            video = VideoFileClip(video_path)
            if video.audio is None:
                print("⚠️ El video no tiene pista de audio (Manejado como tensor vacío posteriormente).")
                return None
            
            video.audio.write_audiofile(output_audio_path, fps=16000, logger=None)
            
            # Cargamos con librosa para tener el array de numpy listo
            y, sr = librosa.load(output_audio_path, sr=16000)
            return y
        except Exception as e:
            print(f"Error extrayendo audio: {e}")
            return None
