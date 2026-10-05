"""YOLOv1 风格的单阶段检测网络。

输入 128x128，经过 4 次下采样得到 S x S (默认 8x8) 的特征图。
每个格子输出 B 个框 (tx, ty, tw, th, conf) 和 C 个类别 logits，
所以输出形状是 [N, S, S, B*5 + C]。

和原版 YOLOv1 的区别：
- 用全卷积的 1x1 检测头代替原论文的全连接层，参数少很多；
- 加了 BatchNorm，训练更稳定。
"""

import torch
from torch import nn


def conv_bn(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.1, inplace=True),
    )


class MiniYOLO(nn.Module):
    def __init__(self, num_classes: int, num_boxes: int = 2, width: int = 16):
        super().__init__()
        self.C = num_classes
        self.B = num_boxes
        w = width
        self.backbone = nn.Sequential(
            conv_bn(3, w), nn.MaxPool2d(2),                              # 64
            conv_bn(w, w * 2), nn.MaxPool2d(2),                          # 32
            conv_bn(w * 2, w * 4), conv_bn(w * 4, w * 2, 1), conv_bn(w * 2, w * 4),
            nn.MaxPool2d(2),                                             # 16
            conv_bn(w * 4, w * 8), conv_bn(w * 8, w * 4, 1), conv_bn(w * 4, w * 8),
            nn.MaxPool2d(2),                                             # 8
            conv_bn(w * 8, w * 8), conv_bn(w * 8, w * 8),
        )
        self.head = nn.Conv2d(w * 8, self.B * 5 + self.C, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.backbone(x))      # [N, B*5+C, S, S]
        return out.permute(0, 2, 3, 1).contiguous()  # [N, S, S, B*5+C]


def split_output(out: torch.Tensor, B: int):
    """把原始输出拆成框参数和类别 logits，并把框参数映射到合法范围。

    返回:
        xy:     [N, S, S, B, 2]  框中心相对所在格子左上角的偏移，0~1
        sqrtwh: [N, S, S, B, 2]  框宽高（相对整张图）的平方根，0~1
        conf:   [N, S, S, B]     置信度，0~1
        cls:    [N, S, S, C]     类别 logits
    """
    N, S, _, _ = out.shape
    boxes = out[..., : B * 5].view(N, S, S, B, 5).sigmoid()
    return boxes[..., 0:2], boxes[..., 2:4], boxes[..., 4], out[..., B * 5 :]


def to_image_boxes(xy: torch.Tensor, sqrtwh: torch.Tensor) -> torch.Tensor:
    """把格子内偏移 + sqrt(wh) 转成整张图坐标下的 (x1, y1, x2, y2)，均归一化到 0~1。"""
    S = xy.shape[1]
    ii, jj = torch.meshgrid(
        torch.arange(S, device=xy.device), torch.arange(S, device=xy.device), indexing="ij"
    )
    # xy: [N, S, S, B, 2]，最后一维是 (x, y)，x 对应列 j，y 对应行 i
    cx = (jj[None, :, :, None] + xy[..., 0]) / S
    cy = (ii[None, :, :, None] + xy[..., 1]) / S
    wh = sqrtwh**2
    return torch.stack(
        [cx - wh[..., 0] / 2, cy - wh[..., 1] / 2, cx + wh[..., 0] / 2, cy + wh[..., 1] / 2], -1
    )
