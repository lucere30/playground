"""IoU、NMS、解码和 mAP 评估。全部手写，不依赖 torchvision。"""

import torch

from .model import split_output, to_image_boxes


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """逐元素 IoU。a、b 形状相同 [..., 4]，格式 (x1, y1, x2, y2)。"""
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[..., 2:] - a[..., :2]).clamp(min=0).prod(-1)
    area_b = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1)
    return inter / (area_a + area_b - inter + 1e-9)


def pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """两两 IoU。a: [N, 4], b: [M, 4] -> [N, M]。"""
    return box_iou(a[:, None, :], b[None, :, :])


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """贪心非极大值抑制，返回保留下来的下标（按分数从高到低）。"""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = pairwise_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long)


def batched_nms(boxes, scores, labels, iou_thr):
    """按类别分别做 NMS：把不同类别的框平移到互不重叠的区域，再统一做一次 NMS。"""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    offset = labels.to(boxes.dtype)[:, None] * 2.0
    return nms(boxes + offset, scores, iou_thr)


@torch.no_grad()
def decode(out: torch.Tensor, B: int, conf_thr: float = 0.25, iou_thr: float = 0.45):
    """把网络输出解码成每张图的检测结果。

    返回 list，每个元素是 dict(boxes=[K,4] 归一化 xyxy, scores=[K], labels=[K])。
    最终分数 = 框置信度 x 类别概率（和 YOLOv1 论文一致）。
    """
    xy, sqrtwh, conf, cls_logits = split_output(out, B)
    boxes = to_image_boxes(xy, sqrtwh).clamp(0, 1)          # [N, S, S, B, 4]
    cls_prob = cls_logits.softmax(-1)                        # [N, S, S, C]
    best_prob, best_cls = cls_prob.max(-1)                   # [N, S, S]
    scores = conf * best_prob[..., None]                     # [N, S, S, B]
    labels = best_cls[..., None].expand_as(scores)

    results = []
    for n in range(out.shape[0]):
        b = boxes[n].reshape(-1, 4)
        s = scores[n].reshape(-1)
        l = labels[n].reshape(-1)
        m = s > conf_thr
        b, s, l = b[m], s[m], l[m]
        keep = batched_nms(b, s, l, iou_thr)
        results.append(dict(boxes=b[keep], scores=s[keep], labels=l[keep]))
    return results


def targets_to_boxes(target: torch.Tensor):
    """把 encode_targets 产生的 [S, S, 5] 网格还原成 (boxes xyxy, labels)。"""
    S = target.shape[0]
    mask = target[..., 4] >= 0
    ii, jj = mask.nonzero(as_tuple=True)
    t = target[mask]
    cx = (jj + t[:, 0]) / S
    cy = (ii + t[:, 1]) / S
    w, h = t[:, 2], t[:, 3]
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
    return boxes, t[:, 4].long()


def average_precision(recall: torch.Tensor, precision: torch.Tensor) -> float:
    """VOC 2010+ 风格的全点插值 AP。"""
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
    # 让 precision 单调不增
    for i in range(mpre.numel() - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])
    idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=True)[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def mean_average_precision(preds, gts, num_classes: int, iou_thr: float = 0.5):
    """计算 mAP@iou_thr。

    preds: decode() 的输出列表
    gts:   list of (boxes, labels)
    返回 (mAP, 每类 AP 列表)
    """
    aps = []
    for c in range(num_classes):
        records = []  # (score, image_idx, box)
        n_gt = 0
        gt_by_img = {}
        for k, (gb, gl) in enumerate(gts):
            g = gb[gl == c]
            gt_by_img[k] = (g, torch.zeros(len(g), dtype=torch.bool))
            n_gt += len(g)
        for k, p in enumerate(preds):
            m = p["labels"] == c
            for box, score in zip(p["boxes"][m], p["scores"][m]):
                records.append((float(score), k, box))
        if n_gt == 0:
            continue
        records.sort(key=lambda r: -r[0])
        tp = torch.zeros(len(records))
        for r, (_, k, box) in enumerate(records):
            g, matched = gt_by_img[k]
            if len(g) == 0:
                continue
            ious = pairwise_iou(box[None], g)[0]
            best = int(ious.argmax())
            if ious[best] >= iou_thr and not matched[best]:
                matched[best] = True
                tp[r] = 1
        ctp = tp.cumsum(0)
        recall = ctp / n_gt
        precision = ctp / torch.arange(1, len(records) + 1)
        aps.append(average_precision(recall, precision) if records else 0.0)
    return (sum(aps) / len(aps) if aps else 0.0), aps
