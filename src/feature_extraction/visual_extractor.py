import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class VisualDeepExtractor:
    def __init__(self):
        """
        Extractor de características visuales basado en Gamal El-Nagar et al. (2024).
        Utiliza ResNet50 e InceptionV3 con Early Fusion (Concatenación).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Inicializando Visual Extractor en {self.device}...")

        # Cargar ResNet50 (Se usa la capa penúltima para obtener el vector de 2048)
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1]) # Quitamos la capa FC
        self.resnet.to(self.device).eval()

        # Cargar InceptionV3
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.fc = nn.Identity() # Quitamos la capa de clasificación
        self.inception = inception.to(self.device).eval()

        # Transformaciones
        self.transform_resnet = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_inception = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_frame_features(self, frames):
        """
        Aplica fusión temprana (Early Fusion) concatenando los tensores de ambas redes.
        frames: Array de numpy con shape (N, H, W, C)
        """
        resnet_features = []
        inception_features = []

        with torch.no_grad():
            for frame in frames:
                # Preprocesar para cada modelo
                t_res = self.transform_resnet(frame).unsqueeze(0).to(self.device)
                t_inc = self.transform_inception(frame).unsqueeze(0).to(self.device)

                # Inferir ResNet: Salida (1, 2048, 1, 1) -> aplanar a (2048)
                out_res = self.resnet(t_res).squeeze()
                
                # Inferir Inception: Salida (1, 2048) -> aplanar a (2048)
                out_inc = self.inception(t_inc)
                if isinstance(out_inc, tuple): # InceptionV3 puede retornar tuple en train mode
                    out_inc = out_inc[0]
                out_inc = out_inc.squeeze()

                resnet_features.append(out_res.cpu().numpy())
                inception_features.append(out_inc.cpu().numpy())

        # Fusión temprana (Concatenación) -> Resultado: Vector de 4096 dimensiones por frame
        if len(resnet_features) == 0:
            return np.array([]).reshape(0, 4096)
            
        fused_visual_features = np.concatenate([np.array(resnet_features), np.array(inception_features)], axis=1)
        return fused_visual_features
