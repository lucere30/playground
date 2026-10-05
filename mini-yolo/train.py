"""训练 Mini-YOLO。

    python train.py                 # 默认配置，CPU 上几分钟
    python train.py --epochs 60     # 训练更久，mAP 更高
"""

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from miniyolo import CLASSES, MiniYOLO, ShapesDataset, YOLOLoss, decode, mean_average_precision, targets_to_boxes

VAL_SEED_OFFSET = 1_000_000  # 验证集用不同的随机种子区间，保证和训练集不重叠


@torch.no_grad()
def evaluate(model, loader, B, device):
    model.eval()
    preds, gts = [], []
    for imgs, targets in loader:
        out = model(imgs.to(device)).cpu()
        preds += decode(out, B, conf_thr=0.01)  # 低阈值，让 PR 曲线完整
        gts += [targets_to_boxes(t) for t in targets]
    return mean_average_precision(preds, gts, len(CLASSES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train-size", type=int, default=5000)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--boxes", type=int, default=2, help="每个格子预测的框数 B")
    ap.add_argument("--width", type=int, default=16, help="网络通道宽度")
    ap.add_argument("--out", default="weights/miniyolo.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    train_ds = ShapesDataset(args.train_size, augment=True)
    val_ds = ShapesDataset(args.val_size, seed_offset=VAL_SEED_OFFSET)
    print(f"生成数据集: train={len(train_ds)} val={len(val_ds)} ({time.time() - t0:.1f}s)")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=128)

    model = MiniYOLO(len(CLASSES), num_boxes=args.boxes, width=args.width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f}M  设备: {device}")

    criterion = YOLOLoss(num_boxes=args.boxes)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    total_steps = args.epochs * len(train_dl)
    warmup = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
    )

    best_map = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        sums, n = {}, 0
        for imgs, targets in train_dl:
            out = model(imgs.to(device))
            loss, parts = criterion(out, targets.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        parts_str = " ".join(f"{k}={v / n:.3f}" for k, v in sums.items())
        msg = f"epoch {epoch:3d}/{args.epochs}  loss={sum(sums.values()) / n:.3f} ({parts_str})  {time.time() - t0:.1f}s"

        if epoch % 5 == 0 or epoch == args.epochs:
            mAP, aps = evaluate(model, val_dl, args.boxes, device)
            per_cls = " ".join(f"{c}={a:.3f}" for c, a in zip(CLASSES, aps))
            msg += f"\n            val mAP@0.5={mAP:.4f}  [{per_cls}]"
            if mAP > best_map:
                best_map = mAP
                torch.save(
                    dict(model=model.state_dict(), classes=CLASSES, boxes=args.boxes, width=args.width, mAP=mAP),
                    out_path,
                )
                msg += f"  -> 保存到 {out_path}"
        print(msg, flush=True)

    print(f"训练完成，最佳 val mAP@0.5 = {best_map:.4f}")


if __name__ == "__main__":
    main()
