import torch.nn as nn

from src.data.labels import STAGE2_CLASSES
from src.models.cnn_transformer import CNNTransformer
from src.models.fusion import LateFusionModel
from src.models.resnet1d import ResNet1D

# Stage 1 (binary anomaly detection) uses ResNet1D: cheap, sample-efficient, runs on every segment.
# Stage 2 (rhythm-type classification) uses CNN+Transformer: only runs on flagged anomalies, and
# needs beat-to-beat attention to separate rhythm-pattern classes (AF/AFLT/SVT) from morphology-only
# classes (conduction blocks, paced rhythms).
STAGE_BACKBONES = {1: ResNet1D, 2: CNNTransformer}


def num_classes_for_stage(stage: int) -> int:
    return 2 if stage == 1 else len(STAGE2_CLASSES)


def build_model(stage: int, modality: str) -> nn.Module:
    assert stage in (1, 2)
    assert modality in ("ecg", "ppg", "both")
    backbone_cls = STAGE_BACKBONES[stage]
    num_classes = num_classes_for_stage(stage)

    if modality in ("ecg", "ppg"):
        return backbone_cls(in_channels=1, num_classes=num_classes)

    ecg_encoder = backbone_cls(in_channels=1, num_classes=num_classes)
    ppg_encoder = backbone_cls(in_channels=1, num_classes=num_classes)
    return LateFusionModel(ecg_encoder, ppg_encoder, num_classes=num_classes)
