"""合成"几何图形"检测数据集。

每张图是一个带渐变和噪声背景的 RGB 图像，上面随机画 1~MAX_OBJECTS 个图形：
圆形、正方形、三角形、十字。标注格式为 (class_id, cx, cy, w, h)，均为 0~1 的归一化坐标。

因为 YOLOv1 每个网格只负责一个物体，生成时保证每个格子里最多只有一个物体中心。
"""

import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

CLASSES = ["circle", "square", "triangle", "cross"]
IMG_SIZE = 128
MAX_OBJECTS = 4
MIN_SIZE, MAX_SIZE = 18, 48


def _background(rng: random.Random, size: int) -> Image.Image:
    c1 = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
    c2 = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
    t = np.linspace(0, 1, size, dtype=np.float32)
    grad = t[:, None] if rng.random() < 0.5 else t[None, :]
    img = c1 * (1 - grad[..., None]) + c2 * grad[..., None]
    img = np.broadcast_to(img, (size, size, 3)).copy()
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    img += nrng.normal(0, 12, img.shape).astype(np.float32)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _contrasting_color(rng: random.Random, bg: tuple) -> tuple:
    # 保证图形和背景有足够的颜色差，否则人眼都看不清
    while True:
        c = tuple(rng.randint(0, 255) for _ in range(3))
        if sum(abs(a - b) for a, b in zip(c, bg)) > 180:
            return c


def _draw_shape(draw: ImageDraw.ImageDraw, cls: int, box: tuple, color: tuple) -> None:
    x0, y0, x1, y1 = box
    if cls == 0:  # circle
        draw.ellipse(box, fill=color)
    elif cls == 1:  # square
        draw.rectangle(box, fill=color)
    elif cls == 2:  # triangle
        draw.polygon([((x0 + x1) / 2, y0), (x1, y1), (x0, y1)], fill=color)
    elif cls == 3:  # cross
        w, h = x1 - x0, y1 - y0
        t = max(3, int(min(w, h) * 0.3))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        draw.rectangle((x0, cy - t / 2, x1, cy + t / 2), fill=color)
        draw.rectangle((cx - t / 2, y0, cx + t / 2, y1), fill=color)


def _overlap(a: tuple, b: tuple) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / min(area(a), area(b))


def make_sample(seed: int, grid: int = 8, size: int = IMG_SIZE):
    """生成一张图片和它的标注。同一个 seed 总是生成同一张图。"""
    rng = random.Random(seed)
    img = _background(rng, size)
    bg = img.getpixel((size // 2, size // 2))
    draw = ImageDraw.Draw(img)

    boxes, labels, used_cells = [], [], set()
    n = rng.randint(1, MAX_OBJECTS)
    for _ in range(n * 20):  # 多试几次，放不下就算了
        if len(boxes) == n:
            break
        cls = rng.randrange(len(CLASSES))
        s = rng.randint(MIN_SIZE, MAX_SIZE)
        w = s
        h = s if cls != 2 else int(s * rng.uniform(0.8, 1.0))
        x0 = rng.randint(0, size - w)
        y0 = rng.randint(0, size - h)
        box = (x0, y0, x0 + w, y0 + h)
        cell = (int((y0 + h / 2) / size * grid), int((x0 + w / 2) / size * grid))
        if cell in used_cells or any(_overlap(box, b) > 0.15 for b in boxes):
            continue
        used_cells.add(cell)
        _draw_shape(draw, cls, box, _contrasting_color(rng, bg))
        boxes.append(box)
        labels.append(cls)

    targets = [
        (c, (b[0] + b[2]) / 2 / size, (b[1] + b[3]) / 2 / size, (b[2] - b[0]) / size, (b[3] - b[1]) / size)
        for c, b in zip(labels, boxes)
    ]
    return img, targets


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def encode_targets(targets, grid: int, num_classes: int) -> torch.Tensor:
    """把标注编码成 [S, S, 5] 的网格张量: (x_offset, y_offset, w, h, class_id)。

    class_id = -1 表示这个格子里没有物体。
    """
    t = torch.zeros(grid, grid, 5)
    t[..., 4] = -1
    for cls, cx, cy, w, h in targets:
        j = min(int(cx * grid), grid - 1)
        i = min(int(cy * grid), grid - 1)
        t[i, j] = torch.tensor([cx * grid - j, cy * grid - i, w, h, cls])
    return t


class ShapesDataset(Dataset):
    """预先生成并缓存在内存里的合成数据集。"""

    def __init__(self, n: int, seed_offset: int = 0, grid: int = 8, augment: bool = False):
        self.grid = grid
        self.augment = augment
        self.images, self.targets = [], []
        for k in range(n):
            img, tg = make_sample(seed_offset + k, grid=grid)
            self.images.append(image_to_tensor(img))
            self.targets.append(tg)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, tg = self.images[idx], list(self.targets[idx])
        if self.augment and random.random() < 0.5:  # 水平翻转
            img = img.flip(-1)
            tg = [(c, 1 - cx, cy, w, h) for c, cx, cy, w, h in tg]
        if self.augment and random.random() < 0.5:  # 垂直翻转（三角形会倒过来，但框和类别不变）
            img = img.flip(-2)
            tg = [(c, cx, 1 - cy, w, h) for c, cx, cy, w, h in tg]
        if self.augment:  # 颜色抖动
            img = (img * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1)).clamp(0, 1)
        # 翻转后物体中心可能刚好落在格子边界上，encode_targets 会处理
        return img, encode_targets(tg, self.grid, len(CLASSES))


def denormalize_box(cx, cy, w, h, size=IMG_SIZE):
    return (
        (cx - w / 2) * size,
        (cy - h / 2) * size,
        (cx + w / 2) * size,
        (cy + h / 2) * size,
    )


__all__ = [
    "CLASSES",
    "IMG_SIZE",
    "ShapesDataset",
    "make_sample",
    "image_to_tensor",
    "encode_targets",
    "denormalize_box",
]
