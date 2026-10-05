import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniyolo import (  # noqa: E402
    CLASSES,
    MiniYOLO,
    ShapesDataset,
    YOLOLoss,
    box_iou,
    decode,
    encode_targets,
    make_sample,
    mean_average_precision,
    nms,
    targets_to_boxes,
)


def test_box_iou_basic():
    a = torch.tensor([0.0, 0.0, 2.0, 2.0])
    assert torch.isclose(box_iou(a, a), torch.tensor(1.0))
    b = torch.tensor([1.0, 1.0, 3.0, 3.0])  # 交集 1，并集 7
    assert torch.isclose(box_iou(a, b), torch.tensor(1 / 7))
    c = torch.tensor([5.0, 5.0, 6.0, 6.0])
    assert box_iou(a, c) == 0


def test_nms_suppresses_overlaps():
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=torch.float)
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_thr=0.5)
    assert keep.tolist() == [0, 2]


def test_make_sample_is_deterministic_and_valid():
    img1, t1 = make_sample(42)
    img2, t2 = make_sample(42)
    assert t1 == t2 and img1.tobytes() == img2.tobytes()
    assert 1 <= len(t1) <= 4
    cells = set()
    for c, cx, cy, w, h in t1:
        assert 0 <= c < len(CLASSES)
        assert 0 <= cx - w / 2 and cx + w / 2 <= 1
        assert 0 <= cy - h / 2 and cy + h / 2 <= 1
        cell = (int(cy * 8), int(cx * 8))
        assert cell not in cells  # 每个格子最多一个物体
        cells.add(cell)


def test_encode_decode_targets_roundtrip():
    tg = [(1, 0.30, 0.70, 0.2, 0.25), (3, 0.81, 0.12, 0.1, 0.1)]
    enc = encode_targets(tg, grid=8, num_classes=4)
    boxes, labels = targets_to_boxes(enc)
    expected = torch.tensor(
        [[c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] for c, cx, cy, w, h in tg]
    )
    # targets_to_boxes 按格子行优先顺序返回
    order = expected[:, 2].argsort()
    assert labels.tolist() == expected[order, 0].long().tolist()
    assert torch.allclose(boxes, expected[order, 1:], atol=1e-6)


def test_model_output_shape():
    model = MiniYOLO(num_classes=4, num_boxes=2, width=8)
    out = model(torch.rand(2, 3, 128, 128))
    assert out.shape == (2, 8, 8, 2 * 5 + 4)


def test_loss_decreases_when_overfitting_one_batch():
    torch.manual_seed(0)
    ds = ShapesDataset(8)
    imgs = torch.stack([ds[i][0] for i in range(8)])
    targets = torch.stack([ds[i][1] for i in range(8)])
    model = MiniYOLO(num_classes=4, width=8)
    crit = YOLOLoss()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss, _ = crit(model(imgs), targets)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first * 0.3


def test_map_perfect_and_empty_predictions():
    gt_boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.9, 0.9]])
    gt_labels = torch.tensor([0, 1])
    perfect = [dict(boxes=gt_boxes, scores=torch.tensor([0.9, 0.8]), labels=gt_labels)]
    mAP, aps = mean_average_precision(perfect, [(gt_boxes, gt_labels)], num_classes=2)
    assert abs(mAP - 1.0) < 1e-6 and len(aps) == 2

    empty = [dict(boxes=torch.zeros(0, 4), scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long))]
    mAP, _ = mean_average_precision(empty, [(gt_boxes, gt_labels)], num_classes=2)
    assert mAP == 0.0


def test_decode_returns_per_image_results():
    model = MiniYOLO(num_classes=4, width=8).eval()
    with torch.no_grad():
        dets = decode(model(torch.rand(3, 3, 128, 128)), B=2, conf_thr=0.0)
    assert len(dets) == 3
    for d in dets:
        assert d["boxes"].shape[1] == 4
        assert len(d["boxes"]) == len(d["scores"]) == len(d["labels"])
        assert (d["boxes"] >= 0).all() and (d["boxes"] <= 1).all()
