"""Soi BẰNG MẮT trên NHIỀU CẢNH khác nhau, so hai model cạnh nhau.

    python3 training/tools/review_wide.py --models trash_yolo11n ctrl11n --confs 0.20 0.03

Vì sao cần: bộ eco cũ có 6 khung nhưng chỉ là 6 khung LIÊN TIẾP của MỘT cảnh với
3-4 vật. Một vật bị trượt bị đếm thành 6 lỗi. Số mẫu độc lập thật là 3-4, không
phải 21 — và toàn bộ kết luận "model nào tốt hơn" đã đứng trên đó.

Bộ này lấy mỗi ảnh một cảnh khác nhau, từ 6 nguồn, trong đó 4 nguồn chưa model
nào được train trên.

Vẽ:
  XANH LÁ  vật thật ĐƯỢC bắt
  ĐỎ       vật thật BỊ TRƯỢT
  VÀNG     hộp không trúng vật thật nào (bắt nhầm)

Đo ở chế độ tầng xác nhận chạy: cắt vùng quanh vật sao cho vật chiếm 6-40% ô,
phóng lên 640px — không chạy trên toàn khung.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../training
PKG = os.path.dirname(HERE)                                          # .../trash_pipeline
sys.path.insert(0, HERE)
sys.path.insert(0, PKG)

# Nap theo DUONG DAN TUYET DOI: goi `tools` cua trash_pipeline va cua training
# trung ten, nen `from tools.bench_suite import ...` se tro nham cho.
import importlib.util                                    # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "bench_suite", os.path.join(HERE, "tools", "bench_suite.py"))
_bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bs)
sources, centre_in = _bs.sources, _bs.centre_in


def draw(img, gts, boxes, tag):
    vis = img.copy()
    hit = fp = 0
    for g in gts:
        ok = any(centre_in(b, g) for b in boxes)
        hit += ok
        cv2.rectangle(vis, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])),
                      (0, 220, 0) if ok else (0, 0, 255), 3)
    for b in boxes:
        if not any(centre_in(b, g) for g in gts):
            fp += 1
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (0, 210, 255), 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(vis, f"{tag} · bat {hit}/{len(gts)} · nham {fp}", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    return vis, hit, fp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["trash_yolo11n", "ctrl11n"])
    ap.add_argument("--confs", nargs="+", type=float, default=[0.20, 0.03])
    ap.add_argument("--per-src", type=int, default=3, help="số cảnh mỗi nguồn")
    ap.add_argument("--side", type=int, default=640)
    a = ap.parse_args()

    from ultralytics import YOLO
    out = os.path.join(PKG, "test_cases", "23_soi_rong")
    os.makedirs(out, exist_ok=True)
    models = []
    for n, c in zip(a.models, a.confs):
        p = os.path.join(PKG, "models", n + ".pt")
        if os.path.exists(p):
            models.append((n, c, YOLO(p)))
        else:
            print(f"  bỏ {n}: không thấy")
    rng = np.random.default_rng(0)

    tally = {n: [0, 0, 0] for n, _, _ in models}      # bat, tong, nham
    rows = []
    for sname, items, held in sources():
        random.Random(1).shuffle(items)
        taken = 0
        for ip, bs in items:
            if taken >= a.per_src:
                break
            img = cv2.imread(ip)
            if img is None:
                continue
            H, W = img.shape[:2]
            px = [(b[0] * W, b[1] * H, b[2] * W, b[3] * H) for b in bs]
            g0 = max(px, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
            gw, gh = g0[2] - g0[0], g0[3] - g0[1]
            if gw < 6 or gh < 6:
                continue
            need = float(np.clip(max(gw, gh) / rng.uniform(0.10, 0.30), 32, min(W, H)))
            cx, cy = (g0[0] + g0[2]) / 2, (g0[1] + g0[3]) / 2
            x0 = int(np.clip(cx - need / 2, 0, max(0, W - need)))
            y0 = int(np.clip(cy - need / 2, 0, max(0, H - need)))
            crop = img[y0:int(y0 + need), x0:int(x0 + need)]
            if crop.size == 0 or min(crop.shape[:2]) < 24:
                continue
            inp = cv2.resize(crop, (a.side, a.side), interpolation=cv2.INTER_CUBIC)
            sx, sy = a.side / crop.shape[1], a.side / crop.shape[0]
            gts = []
            for q in px:
                r = ((q[0] - x0) * sx, (q[1] - y0) * sy,
                     (q[2] - x0) * sx, (q[3] - y0) * sy)
                if r[2] > 4 and r[3] > 4 and r[0] < a.side - 4 and r[1] < a.side - 4:
                    gts.append(r)
            if not gts:
                continue
            taken += 1
            pair = []
            for mn, mc, mdl in models:
                res = mdl.predict([inp], conf=mc, verbose=False)[0]
                boxes = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
                vis, hit, fp = draw(inp, gts, boxes,
                                    f"{mn} {mc} · {sname}{' [HELD]' if held else ''}")
                tally[mn][0] += hit
                tally[mn][1] += len(gts)
                tally[mn][2] += fp
                pair.append(cv2.resize(vis, (400, 400)))
            rows.append(np.hstack(pair))
    if rows:
        k = 2
        grid = np.vstack([np.hstack(rows[i:i + k]) if len(rows[i:i + k]) == k
                          else np.hstack(rows[i:i + k] + [np.zeros_like(rows[0])])
                          for i in range(0, len(rows), k)])
        p = os.path.join(out, "so_sanh.jpg")
        cv2.imwrite(p, grid)
        print(f"\n{len(rows)} cảnh -> {p}")
    print(f"\n{'model':16s} {'bắt được':>14s} {'bắt nhầm':>9s}")
    for n, (h, t, f) in tally.items():
        print(f"  {n:14s} {h:4d}/{t:<5d} {100*h/max(1,t):5.1f}% {f:9d}")
    print("\nXANH = bắt được · ĐỎ = trượt · VÀNG = bắt nhầm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
