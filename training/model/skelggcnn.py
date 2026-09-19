
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.3):
        super().__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.branch3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)

        self.branch_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Conv2d(4 * out_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = F.relu(self.branch1x1(x))

        branch3x3 = F.relu(self.branch3x3_1(x))
        branch3x3 = F.relu(self.branch3x3_2(branch3x3))

        branch5x5 = F.relu(self.branch5x5_1(x))
        branch5x5 = F.relu(self.branch5x5_2(branch5x5))

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = F.relu(self.branch_pool(branch_pool))

        output = torch.cat((branch1x1, branch3x3, branch5x5, branch_pool), dim=1)
        output = F.relu(self.conv(output))
        output = self.dropout(output)
        return x + output


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(self.upsample(x)))


class SkeletonGG_CNN(nn.Module):
    """Four-output SkelGG-CNN.

    The fourth head retains the name ``width`` for compatibility with existing
    checkpoints. It learns either skeleton gradients or grasp widths according
    to the loader's ``target`` setting.
    """

    def __init__(
        self,
        input_channels: int = 3,
        filters: Sequence[int] = (32, 64, 128, 256, 512),
        dropout_prob: float = 0.3,
    ):
        super().__init__()
        if len(filters) != 5:
            raise ValueError("filters must contain exactly five channel sizes")

        self.down1 = DownBlock(input_channels, filters[0], dropout_prob)
        self.down2 = DownBlock(filters[0], filters[1], dropout_prob)
        self.down3 = DownBlock(filters[1], filters[2], dropout_prob)
        self.down4 = DownBlock(filters[2], filters[3], dropout_prob)

        self.bn = nn.BatchNorm2d(filters[3])
        self.conv = nn.Conv2d(filters[3], filters[4], kernel_size=3, padding=1)
        self.bottleneck = nn.Conv2d(filters[4], filters[4], kernel_size=1)
        self.inception_resnet = InceptionResNetBlock(
            filters[4], filters[4], dropout_prob=dropout_prob
        )

        self.up1 = UpBlock(filters[4], filters[3])
        self.up2 = UpBlock(filters[3], filters[2])
        self.up3 = UpBlock(filters[2], filters[1])
        self.up4 = UpBlock(filters[1], filters[0])

        self.position = nn.Conv2d(filters[0], 1, kernel_size=3, padding=1)
        self.cosine = nn.Conv2d(filters[0], 1, kernel_size=3, padding=1)
        self.sin = nn.Conv2d(filters[0], 1, kernel_size=3, padding=1)
        self.width = nn.Conv2d(filters[0], 1, kernel_size=3, padding=1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.bn(x)
        x = F.relu(self.conv(x))
        x = F.relu(self.bottleneck(x))
        x = self.inception_resnet(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        return self.position(x), self.cosine(x), self.sin(x), self.width(x)

    def compute_loss(
        self, xc: torch.Tensor, yc: Sequence[torch.Tensor]
    ) -> Dict[str, object]:
        y_pos, y_cos, y_sin, y_target = yc
        pos_pred, cos_pred, sin_pred, target_pred = self(xc)

        losses = {
            "position": F.smooth_l1_loss(pos_pred, y_pos),
            "cosine": F.smooth_l1_loss(cos_pred, y_cos),
            "sine": F.smooth_l1_loss(sin_pred, y_sin),
            "target": F.smooth_l1_loss(target_pred, y_target),
        }
        return {"loss": sum(losses.values()), "losses": losses}

