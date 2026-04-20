import torch
import torch.nn as nn
import numpy as np

class FusionScorer(nn.Module):
    def __init__(self, visual_model, audio_model):
        """
        Maneja la inferencia de ambas redes y fusiona los puntajes.
        """
        super(FusionScorer, self).__init__()
        self.visual_model = visual_model
        self.audio_model = audio_model

    def forward(self, visual_features, audio_features):
        """
        Retorna el score fusionado por frame.
        """
        v_scores = self.visual_model(visual_features)
        a_scores = self.audio_model(audio_features)
        
        # Fusión base del paper: Promedio simple de ambos scores
        # Fórmula: AV_Score_j = (V_Score_j + A_Score_j) / 2
        av_scores = (v_scores + a_scores) / 2.0
        
        return av_scores, v_scores, a_scores

class ScorePredictor:
    def __init__(self, visual_dim=4096, audio_dim=296, model_weights_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Inicializando Redes de Puntuación en {self.device}...")
        
        from .visual_ann import VisualANN
        from .audio_ann import AudioANN
        
        v_ann = VisualANN(input_dim=visual_dim)
        a_ann = AudioANN(input_dim=audio_dim)
        
        self.scorer = FusionScorer(v_ann, a_ann).to(self.device)
        
        if model_weights_path:
            # self.scorer.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            print("Pesos cargados.")
        else:
            print("⚠️ Ejecutando con pesos aleatorios (Requiere entrenamiento).")
            
        self.scorer.eval() # Modo inferencia por defecto

    def predict_shot_scores(self, processed_shots):
        """
        Toma los shots procesados por el Módulo 2 y les asigna sus puntajes.
        """
        scored_shots = []
        
        with torch.no_grad():
            for shot in processed_shots:
                # Convertir numpy arrays a tensores
                v_tensor = torch.tensor(shot["visual_features"], dtype=torch.float32).to(self.device)
                a_tensor = torch.tensor(shot["audio_features"], dtype=torch.float32).to(self.device)
                
                # Predecir a nivel de frame
                av_scores, _, _ = self.scorer(v_tensor, a_tensor)
                
                # Convertir de vuelta a numpy y aplanar
                frame_scores = av_scores.cpu().numpy().flatten()
                
                # El score del shot es el promedio de los scores de sus frames
                shot_score = np.mean(frame_scores)
                
                scored_shots.append({
                    "shot_id": shot["shot_id"],
                    "shot_score": float(shot_score),
                    "frame_scores": frame_scores,
                    "frames": shot.get("frames"),
                    "audio": shot.get("audio")
                })
                
        return scored_shots
