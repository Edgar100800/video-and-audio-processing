import torch
import torch.nn as nn

class AudioANN(nn.Module):
    def __init__(self, input_dim=296):
        """
        Red profunda para predecir la importancia del audio.
        El paper detalla una progresión de capas: 2048 -> 1024 -> 2048 -> 1024.
        """
        super(AudioANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3), # Recomendado para evitar sobreajuste en capas tan grandes
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: (Batch/Frames, input_dim)
        return self.network(x)
