import torch
import torch.nn as nn

class VisualANN(nn.Module):
    def __init__(self, input_dim=4096):
        """
        Red densa para predecir la importancia visual.
        Basado en Gamal El-Nagar et al. (2024).
        """
        super(VisualANN, self).__init__()
        # El paper menciona: MaxPooling -> Flatten -> Linear(128) -> Out(1)
        # Dado que ya tenemos vectores de características (1D por frame):
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU() # Los scores de importancia son positivos
        )

    def forward(self, x):
        # x shape: (Batch/Frames, input_dim)
        return self.network(x)
