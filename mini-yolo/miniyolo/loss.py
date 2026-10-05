"""YOLOv1 损失函数。

对每个有物体的格子，从 B 个预测框中选出和真实框 IoU 最大的那个作为"负责"框：
- 坐标损失:  λ_coord * [(x-x̂)² + (y-ŷ)² + (√w-√ŵ)² + (√h-√ĥ)²]
- 置信度:    (conf - IoU)²            （负责框，目标是它和真实框的 IoU）
- 无物体:    λ_noobj * conf²          （其余所有框）
- 分类:      交叉熵                    （论文里用的是 MSE，这里换成 CE，收敛更快）
"""

import torch
import torch.nn.functional as F
from torch import nn

from .model import split_output, to_image_boxes
from .utils import box_iou


class YOLOLoss(nn.Module):
    def __init__(self, num_boxes: int = 2, lambda_coord: float = 5.0, lambda_noobj: float = 0.5):
        super().__init__()
        self.B = num_boxes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, out: torch.Tensor, target: torch.Tensor):
        """
        out:    [N, S, S, B*5+C] 网络原始输出
        target: [N, S, S, 5]     (x_off, y_off, w, h, class_id)，class_id=-1 表示空格子
        """
        N = out.shape[0]
        xy, sqrtwh, conf, cls_logits = split_output(out, self.B)

        obj = target[..., 4] >= 0                                    # [N, S, S]
        t_xy = target[..., 0:2].unsqueeze(3)                         # [N, S, S, 1, 2]
        t_sqrtwh = target[..., 2:4].clamp(min=0).sqrt().unsqueeze(3)

        with torch.no_grad():
            pred_boxes = to_image_boxes(xy, sqrtwh)                  # [N, S, S, B, 4]
            gt_boxes = to_image_boxes(t_xy, t_sqrtwh)                # [N, S, S, 1, 4]
            ious = box_iou(pred_boxes, gt_boxes.expand_as(pred_boxes))  # [N, S, S, B]
            best = ious.argmax(-1, keepdim=True)                     # [N, S, S, 1]
            resp = torch.zeros_like(conf, dtype=torch.bool).scatter_(-1, best, True)
            resp &= obj.unsqueeze(-1)                                # [N, S, S, B]
            best_iou = ious.gather(-1, best).squeeze(-1)             # [N, S, S]

        r = resp
        coord = ((xy - t_xy) ** 2).sum(-1)[r].sum() + ((sqrtwh - t_sqrtwh) ** 2).sum(-1)[r].sum()
        conf_obj = ((conf[r] - best_iou[obj]) ** 2).sum()
        conf_noobj = (conf[~r] ** 2).sum()
        if obj.any():
            cls = F.cross_entropy(cls_logits[obj], target[..., 4][obj].long(), reduction="sum")
        else:
            cls = out.sum() * 0

        parts = {
            "coord": self.lambda_coord * coord / N,
            "obj": conf_obj / N,
            "noobj": self.lambda_noobj * conf_noobj / N,
            "cls": cls / N,
        }
        return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}
