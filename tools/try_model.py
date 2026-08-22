"""Chạy detector lên ảnh bất kỳ và vẽ ra xem nó thấy gì.

    python3 tools/try_model.py --images anh1.jpg anh2.jpg --out thu.jpg
    python3 tools/try_model.py --dir /duong/dan/thu-muc --n 6

Không đi qua pipeline — đây là ĐẦU RA THÔ của model, chưa có cổng đổi, chưa có
dwell, chưa có chốt. Dùng để xem model *thật sự* nhận ra cái gì, tách khỏi mọi
tầng lọc phía sau.

Chạy đúng cách pipeline gọi nó: cắt ô 320px, phóng 2x, gộp NMS. Gọi khác đi
(ném cả khung vào 640px) thì kết quả tụt hẳn — đo được 29% so với 75%.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from core.verify import RegionVerifier   # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def tiles(img, tile=320, overlap=0.5):
    h, w = img.shape[:2]
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), step)) or [0]
    ys = list(range(0, max(1, h - tile + 1), step)) or [0]
    if xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    if ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    return [(x, y) for y in ys for x in xs]


def detect(model, img, conf, tile=320, upscale=2.0):
    out = []
    crops, offs = [], []
    for x, y in tiles(img, tile):
        c = img[y:y + tile, x:x + tile]
        if c.size == 0:
            continue
        crops.append(cv2.resize(c, None, fx=upscale, fy=upscale,
                                interpolation=cv2.INTER_CUBIC))
        offs.append((x, y))
    if not crops:
        return out
    for (ox, oy), r in zip(offs, model.predict(crops, conf=conf, verbose=False)):
        for b in r.boxes:
            q = b.xyxy[0].tolist()
            out.append(((ox + q[0] / upscale, oy + q[1] / upscale,
                         ox + q[2] / upscale, oy + q[3] / upscale), float(b.conf)))
    # NMS gộp phát hiện trùng ở phần chồng lấn
    if not out:
        return out
    bx = np.array([o[0] for o in out], np.float32)
    sc = np.array([o[1] for o in out], np.float32)
    area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
    order, keep = sc.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(bx[i, 0], bx[rest, 0]); yy1 = np.maximum(bx[i, 1], bx[rest, 1])
        xx2 = np.minimum(bx[i, 2], bx[rest, 2]); yy2 = np.minimum(bx[i, 3], bx[rest, 3])
        it = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = rest[it / (area[i] + area[rest] - it + 1e-9) <= 0.5]
    return [out[i] for i in keep]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--dir")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--config", default="config/day_cfg.yaml")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--cell", type=int, default=640, help="kích thước ô hiển thị")
    ap.add_argument("--out", default="try_model.jpg")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    v = RegionVerifier(cfg.get("verify", {}))
    conf = args.conf if args.conf is not None else v.conf
    model = v._model()
    print(f"model {v.weights} · conf {conf} · ô 320px phóng {v.upscale}x")

    paths = list(args.images)
    if args.dir:
        paths += [os.path.join(args.dir, f) for f in sorted(os.listdir(args.dir))
                  if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")]
    paths = paths[:args.n]

    panels = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  bỏ qua (không đọc được): {p}")
            continue
        dets = detect(model, img, conf, upscale=v.upscale)
        vis = img.copy()
        for (x1, y1, x2, y2), s in dets:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(vis, f"{s:.2f}", (int(x1), max(11, int(y1) - 4)),
                        FONT, .45, (0, 0, 255), 1, cv2.LINE_AA)
        sizes = [f"{int(max(b[2]-b[0], b[3]-b[1]))}px" for b, _ in dets[:4]]
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(vis, f"{os.path.basename(p)[:28]}  {len(dets)} phat hien"
                         + ("  " + " ".join(sizes) if sizes else ""),
                    (6, 17), FONT, .5,
                    (0, 220, 120) if dets else (0, 165, 255), 1, cv2.LINE_AA)
        print(f"  {os.path.basename(p):32s} {len(dets)} phát hiện"
              + (f"  conf {max(s for _, s in dets):.2f} cao nhất" if dets else ""))
        panels.append(cv2.resize(vis, (args.cell, int(args.cell * 0.75))))

    if panels:
        cols = 2 if len(panels) <= 4 else 3
        while len(panels) % cols:
            panels.append(np.zeros_like(panels[0]))
        cv2.imwrite(args.out, np.vstack([np.hstack(panels[i:i + cols])
                                         for i in range(0, len(panels), cols)]))
        print(f"\nlưu {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
