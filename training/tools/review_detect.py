"""Soi BẰNG MẮT: model bắt được gì, trượt gì, bắt nhầm gì — trên khung thật.

    python3 tools/review_detect.py --models trash_yolo11n ctrl11n --out ../test_cases/22_soi

Vẽ đúng ba loại, để nhìn là phân biệt được ngay:
  XANH LÁ  vật thật ĐƯỢC bắt
  ĐỎ       vật thật BỊ TRƯỢT
  VÀNG     hộp detector không trúng vật thật nào (bắt nhầm)

Đo ở chế độ tầng xác nhận thực sự chạy: gom ô nóng thành vùng 160-320px, phóng
lên 640px rồi mới hỏi model — không phải chạy trên toàn khung.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../training
PKG = os.path.dirname(HERE)                                          # .../trash_pipeline
# Thu tu QUAN TRONG: training/ co mot thu muc `core` rieng che mat
# trash_pipeline/core. Chen PKG SAU CUNG de no nam dau sys.path.
sys.path.insert(0, os.path.join(PKG, "tools"))
sys.path.insert(0, PKG)

from core.verify import RegionVerifier              # noqa: E402
from core.grid import build_grid, poly_to_px        # noqa: E402
from run_test_cases import SITE, FULL_POLY, gt_boxes        # noqa: E402


def inter(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["trash_yolo11n", "ctrl11n"])
    ap.add_argument("--confs", nargs="+", type=float, default=[0.20, 0.03])
    ap.add_argument("--config", default="config/day_cfg.yaml")
    ap.add_argument("--out", default="../test_cases/22_soi")
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(PKG, a.config), encoding="utf-8"))
    out = os.path.abspath(os.path.join(PKG, "test_cases", "22_soi"))
    os.makedirs(out, exist_ok=True)

    ref = np.load(os.path.join(SITE, "_ref.npy"))
    frames = []
    for fn in sorted(f for f in os.listdir(SITE) if f.startswith("frame_")):
        im = cv2.imread(os.path.join(SITE, fn))
        if im is not None and im.shape == ref.shape and len(gt_boxes(im, ref)) >= 3:
            frames.append((fn, im))
    warm = int(cfg["decide"]["dwell_scans"]) + int(cfg["confirm"]["m"])
    frames = frames[warm:warm + a.frames]
    h, w = ref.shape[:2]
    grid = build_grid(poly_to_px(FULL_POLY, w, h), w, h, cell_px=48, overlap=0.5)

    for mname, conf in zip(a.models, a.confs):
        wt = os.path.join("models", mname + ".pt")
        if not os.path.exists(os.path.join(PKG, wt)):
            print(f"  bỏ {mname}: không thấy file")
            continue
        v = RegionVerifier(dict(cfg["verify"], weights=wt, conf=conf))
        tiles = []
        n_hit = n_gt = n_fp = 0
        for fn, im in frames:
            gts = gt_boxes(im, ref)
            hot = [c for c in grid.cells
                   if any(inter((c.x1, c.y1, c.x2, c.y2), g) for g in gts)]
            _keep, boxes = v.verify(im, hot)
            vis = im.copy()
            hits = 0
            for g in gts:
                ok = any(inter(b[:4], g) for b in boxes)
                hits += ok
                col = (0, 220, 0) if ok else (0, 0, 255)
                cv2.rectangle(vis, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])),
                              col, 3)
            for b in boxes:
                if not any(inter(b[:4], g) for g in gts):
                    n_fp += 1
                    cv2.rectangle(vis, (int(b[0]), int(b[1])),
                                  (int(b[2]), int(b[3])), (0, 210, 255), 2)
                    if len(b) > 4:
                        cv2.putText(vis, f"{b[4]:.2f}", (int(b[0]), max(12, int(b[1]) - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 210, 255), 2)
            n_hit += hits
            n_gt += len(gts)
            vis = cv2.resize(vis, (560, 420))
            cv2.rectangle(vis, (0, 0), (560, 22), (0, 0, 0), -1)
            cv2.putText(vis, f"{mname} c={conf} · bat {hits}/{len(gts)}",
                        (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            tiles.append(vis)
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        g = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        p = os.path.join(out, f"{mname}.jpg")
        cv2.imwrite(p, g)
        print(f"  {mname:14s} conf {conf}: bắt {n_hit}/{n_gt} · {n_fp} hộp nhầm -> {p}")
    print("\nXANH = bắt được · ĐỎ = trượt · VÀNG = bắt nhầm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
