"""用训练好的 Mini-YOLO 做检测，并把结果画出来。

    python detect.py                          # 随机生成 16 张新图，输出 outputs/demo.png
    python detect.py --images a.png b.jpg     # 检测自己的图片（会缩放到 128x128）
    python detect.py --show-gt                # 同时画出真实框（虚线）
"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from miniyolo import IMG_SIZE, MiniYOLO, decode, image_to_tensor, make_sample

COLORS = [(255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 204, 0)]
TEST_SEED_OFFSET = 9_000_000
SCALE = 3  # 输出图片放大倍数，方便看清


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = MiniYOLO(len(ckpt["classes"]), num_boxes=ckpt["boxes"], width=ckpt["width"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def draw_dashed_rect(draw, box, color, dash=6):
    x0, y0, x1, y1 = box
    for a, b, horizontal, fixed in [(x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)]:
        p = a
        while p < b:
            q = min(p + dash, b)
            draw.line([(p, fixed), (q, fixed)] if horizontal else [(fixed, p), (fixed, q)], fill=color, width=1)
            p += dash * 2


def render(img, det, classes, gt=None):
    big = img.resize((IMG_SIZE * SCALE, IMG_SIZE * SCALE), Image.NEAREST)
    draw = ImageDraw.Draw(big)
    s = IMG_SIZE * SCALE
    if gt:
        for c, cx, cy, w, h in gt:
            draw_dashed_rect(draw, ((cx - w / 2) * s, (cy - h / 2) * s, (cx + w / 2) * s, (cy + h / 2) * s), (255, 255, 255))
    for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
        x0, y0, x1, y1 = (box * s).tolist()
        color = COLORS[int(label) % len(COLORS)]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        text = f"{classes[int(label)]} {float(score):.2f}"
        tw = draw.textlength(text)
        ty = y0 - 14 if y0 >= 14 else y1
        draw.rectangle((x0, ty, x0 + tw + 6, ty + 14), fill=color)
        draw.text((x0 + 3, ty + 1), text, fill=(0, 0, 0))
    return big


def make_grid(images, cols):
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    pad = 6
    grid = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + pad) + pad), (30, 30, 30))
    for k, im in enumerate(images):
        grid.paste(im, (pad + (k % cols) * (w + pad), pad + (k // cols) * (h + pad)))
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/miniyolo.pt")
    ap.add_argument("--images", nargs="*", help="要检测的图片路径；不给就随机生成测试图")
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--show-gt", action="store_true")
    ap.add_argument("--out", default="outputs/demo.png")
    args = ap.parse_args()

    model, ckpt = load_model(args.weights)
    classes = ckpt["classes"]

    if args.images:
        samples = [(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)), None) for p in args.images]
    else:
        samples = [make_sample(TEST_SEED_OFFSET + args.seed * 1000 + k) for k in range(args.num)]

    batch = torch.stack([image_to_tensor(img) for img, _ in samples])
    with torch.no_grad():
        dets = decode(model(batch), ckpt["boxes"], conf_thr=args.conf, iou_thr=args.iou)

    rendered = []
    for (img, gt), det in zip(samples, dets):
        names = ", ".join(f"{classes[int(l)]}({float(s):.2f})" for l, s in zip(det["labels"], det["scores"]))
        print(f"{len(det['boxes'])} 个物体: {names}")
        rendered.append(render(img, det, classes, gt if args.show_gt else None))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_grid(rendered, cols=min(4, len(rendered))).save(out)
    print(f"结果已保存到 {out}")


if __name__ == "__main__":
    main()
