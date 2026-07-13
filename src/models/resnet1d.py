import torch.nn as nn


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """ResNet18-style 1D CNN. Used as the Stage 1 (binary anomaly detection) backbone."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 64,
        blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        stages = []
        in_ch = base_channels
        for i, n_blocks in enumerate(blocks_per_stage):
            out_ch = base_channels * (2**i)
            stride = 1 if i == 0 else 2
            stage_blocks = [BasicBlock1D(in_ch, out_ch, stride=stride)]
            stage_blocks += [BasicBlock1D(out_ch, out_ch, stride=1) for _ in range(n_blocks - 1)]
            stages.append(nn.Sequential(*stage_blocks))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_dim = in_ch
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.pool(x).squeeze(-1)

    def forward(self, x):
        return self.classifier(self.dropout(self.forward_features(x)))
