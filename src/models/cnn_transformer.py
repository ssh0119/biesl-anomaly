import math

import torch
import torch.nn as nn


class ConvPatchEmbed(nn.Module):
    """Strided 1D conv stack turning a raw waveform into a sequence of local patch embeddings."""

    def __init__(self, in_channels: int = 1, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, d_model, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x).transpose(1, 2)  # (B, C, L) -> (B, L', d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CNNTransformer(nn.Module):
    """CNN patch embedding + Transformer encoder. Stage 2 (rhythm-type classification) backbone.

    The conv front-end captures local (single-beat) morphology; self-attention over the
    resulting patch sequence captures the beat-to-beat rhythm pattern (irregularity, rate,
    pauses) that distinguishes e.g. AF/AFLT/SVT from conduction blocks or paced rhythms.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = ConvPatchEmbed(in_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.embedding_dim = d_model
        self.classifier = nn.Linear(d_model, num_classes)

    def forward_features(self, x):
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_enc(tokens)
        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0])

    def forward(self, x):
        return self.classifier(self.forward_features(x))
