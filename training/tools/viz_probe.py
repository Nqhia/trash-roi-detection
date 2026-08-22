"""Vẽ ra ĐÚNG thứ probe.py đã đo: bắt được gì, bỏ sót gì, báo nhầm gì.

    python3 tools/viz_probe.py --backend owlv2 --size 24
    python3 tools/viz_probe.py --backend owlv2 --clean          # chỉ khung sạch

Dùng chung hàm `detect_in_zone` với probe.py — nếu vẽ bằng đường khác thì ảnh
sẽ không còn là bằng chứng cho con số nữa.

  XANH LÁ  = rác thật (ground truth)
  ĐỎ       = model đoán
  nhãn góc = TRUNG (IoU >= 0.3 với GT) / MISS / FP
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import build_detector, detect_in_zone   # noqa: E402
from tools.probe import (SOURCES, WIN_H, WIN_W, ZONE, clean_frames,  # noqa: E402
                         coco_items, iou, make_scene)

FONT = cv2.FONT_HERSHEY_SIMPLEX
POLY = [(int(p[0] * WIN_W), int(p[1] * WIN_H)) for p in ZONE]


def annotate(scene, gt, dets, tag, color, sub=""):
    im = scene.copy()
    if gt is not None:
        cv2.rectangle(im, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])),
                      (60, 255, 60), 2)
        cv2.putText(im, "rac that", (int(gt[0]), max(11, int(gt[1]) - 4)),
                    FONT, .42, (60, 255, 60), 1, cv2.LINE_AA)
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d.box]
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(im, f"{d.label} {d.score:.2f}", (x1, min(WIN_H - 4, y2 + 13)),
                    FONT, .42, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.rectangle(im, (0, 0), (WIN_W, 22), (0, 0, 0), -1)
    cv2.putText(im, f"{tag}   {sub}", (7, 16), FONT, .5, color, 1, cv2.LINE_AA)
    return im


def sheet(tiles, cols=3):
    if not tiles:
        return None
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="owlv2",
                    choices=["yoloworld", "owlv2", "coco"])
    ap.add_argument("--size", type=int, default=24, help="cỡ vật (px) mô phỏng")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--each", type=int, default=6, help="số ảnh mỗi loại")
    ap.add_argument("--scan", type=int, default=60, help="duyệt tối đa bao nhiêu vật")
    ap.add_argument("--clean", action="store_true", help="chỉ vẽ FP trên khung sạch")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    det = build_detector(args.backend, conf=args.conf)

    if args.clean:
        tiles = []
        for f in clean_frames(30):
            dets = detect_in_zone(det, f, POLY, tile_px=320, upscale=3.0)
            if dets:
                tiles.append(annotate(f, None, dets, "FP - khung nay KHONG CO RAC",
                                      (0, 0, 255), f"{len(dets)} phat hien"))
            if len(tiles) >= args.each:
                break
        img = sheet(tiles)
        p = os.path.join(args.out, f"viz_fp_{args.backend}.jpg")
        cv2.imwrite(p, img)
        print(f"luu {p}  ({len(tiles)} khung sach co bao nham)")
        return 0

    random.seed(0)
    items = []
    for name, annf, imgd, ren in SOURCES:
        if not os.path.exists(annf):
            continue
        got = coco_items(annf, imgd, ren)
        random.shuffle(got)
        items += [(name, p, b, mw) for p, bs, mw in got[:80] for b in bs[:1]]
    random.shuffle(items)

    hits, misses = [], []
    seen = 0
    for name, path, bbox, meta_w in items:
        if seen >= args.scan or (len(hits) >= args.each and len(misses) >= args.each):
            break
        img = cv2.imread(path)
        if img is None:
            continue
        k = img.shape[1] / meta_w
        scene, gt = make_scene(img, [v * k for v in bbox], args.size)
        if scene is None:
            continue
        seen += 1
        dets = detect_in_zone(det, scene, POLY, tile_px=320, upscale=3.0)
        ok = any(iou(d.box, gt) >= 0.3 for d in dets)
        gt_px = int(max(gt[2] - gt[0], gt[3] - gt[1]))
        if ok and len(hits) < args.each:
            hits.append(annotate(scene, gt, dets, "TRUNG", (60, 255, 60),
                                 f"{name}  vat {gt_px}px"))
        elif not ok and len(misses) < args.each:
            misses.append(annotate(scene, gt, dets, "MISS", (0, 165, 255),
                                   f"{name}  vat {gt_px}px  "
                                   f"{len(dets)} phat hien deu sai cho"))

    for tag, tl in (("hit", hits), ("miss", misses)):
        img = sheet(tl)
        if img is None:
            print(f"khong co anh nao thuoc loai {tag}")
            continue
        p = os.path.join(args.out, f"viz_{tag}_{args.backend}_{args.size}px.jpg")
        cv2.imwrite(p, img)
        print(f"luu {p}  ({len(tl)} anh)")
    print(f"duyet {seen} vat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
