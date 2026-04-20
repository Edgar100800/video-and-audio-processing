import torch
import numpy as np
import librosa

class AudioDeepExtractor:
    def __init__(self, target_sr=16000):
        """
        Extrae representaciones de audio. Se integra torchvggish a través de torch.hub.
        """
        self.sr = target_sr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎵 Inicializando Audio Extractor en {self.device}...")
        
        try:
            # Cargar VGGish desde PyTorch Hub
            self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish', verbose=False)
            self.vggish.to(self.device).eval()
            self.has_vggish = True
        except Exception as e:
            print(f"⚠️ No se pudo cargar VGGish ({e}). Se usarán solo MFCC y Mel-Spectrogram.")
            self.has_vggish = False

    def extract_audio_features(self, audio_array):
        """
        Combina MFCC, Mel-Spectrogram y VGGish Embeddings.
        Retorna un vector 1D que representa el segmento de audio completo.
        """
        if audio_array is None or len(audio_array) == 0:
            # MFCC (40) + Mel (128) + VGGish (128) = 296 dimensiones
            return np.zeros(296) 

        # 1. Mel-Frequency Cepstral Coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=audio_array, sr=self.sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0) # Promedio a lo largo del tiempo

        # 2. Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=self.sr)
        mel_spec_mean = np.mean(mel_spec.T, axis=0)

        # 3. VGGish Embeddings
        vggish_embed = np.zeros(128)
        if self.has_vggish:
            with torch.no_grad():
                try:
                    # torchvggish espera un path o un numpy array con preprocesado específico
                    # Usamos el método forward interno si es posible
                    embeddings = self.vggish(audio_array, self.sr)
                    if len(embeddings.shape) > 1:
                        vggish_embed = embeddings.mean(axis=0)
                    else:
                        vggish_embed = embeddings
                except Exception as e:
                    print(f"Error en VGGish: {e}")
                    pass

        # Concatenar características de audio en un solo vector
        fused_audio_features = np.concatenate([mfccs_mean, mel_spec_mean, vggish_embed])
        return fused_audio_features
