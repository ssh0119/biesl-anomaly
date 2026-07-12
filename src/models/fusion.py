import torch
import torch.nn as nn


class LateFusionModel(nn.Module):
    """Combines separate ECG and PPG encoders via late fusion (concat embeddings + MLP head).

    Late fusion (rather than stacking raw channels) is used because ECG and PPG differ enough
    in morphology that a shared early conv stack wastes capacity; this also lets each branch
    warm-start from the corresponding single-modality model's weights.
    """

    def __init__(self, ecg_encoder: nn.Module, ppg_encoder: nn.Module, num_classes: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.ppg_encoder = ppg_encoder
        fused_dim = ecg_encoder.embedding_dim + ppg_encoder.embedding_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: dict):
        ecg_feat = self.ecg_encoder.forward_features(inputs["ecg"])
        ppg_feat = self.ppg_encoder.forward_features(inputs["ppg"])
        fused = torch.cat([ecg_feat, ppg_feat], dim=1)
        return self.head(fused)
