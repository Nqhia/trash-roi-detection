"""Hiệu chuẩn `decide.litter_thr` theo TỪNG site. Chạy sau khi có âm của site.

Vì sao bắt buộc: ngưỡng 0.6 là con số mặc định vô nghĩa. Sau khi model có ô âm
của chính camera đó, điểm của NỀN tụt rất sâu (đo được: max 0.105) trong khi rác
vẫn 0.25-0.92. Giữ 0.6 là vứt đi phần lớn khoảng cách đó — và mất recall ở đúng
chỗ đắt nhất là vật nhỏ:

    thr 0.60 -> rác 22px bắt được 20% số lượt
    thr 0.15 -> rác 22px bắt được 100%, FP vẫn 0

Cách dùng: quay vùng lúc CHẮC CHẮN SẠCH (mọi khung giờ càng tốt), rồi

    python3 tools/calibrate.py --source sach.mp4 --zone zone.json

Nó chấm mọi ô của mọi lượt, lấy phân vị cao nhất của nền rồi đề xuất ngưỡng có
biên an toàn. KHÔNG cần nhãn.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grid import build_grid, occluded_ids, poly_to_px    # noqa: E402
from core.scorers import build_scorer                          # noqa: E402
from tools.run_video import YoloBoxes, iter_frames             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video vùng lúc SẠCH")
    ap.add_argument("--zone", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--yolo", help="weights YOLO để bỏ ô có người/xe")
    ap.add_argument("--margin", type=float, default=1.5,
                    help="ngưỡng = phân vị nền x margin (mặc định 1.5)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.zone, encoding="utf-8") as f:
        poly = json.load(f)["points"]

    g = cfg.get("grid", {})
    cell_px, overlap = int(g.get("cell_px", 64)), float(g.get("overlap", 0.5))
    scorer = build_scorer(cfg.get("scorer", {}))
    boxes_of = YoloBoxes(args.yolo) if args.yolo else (lambda _f: [])
    interval = float(cfg.get("scan", {}).get("interval_s", 30))

    grid, scores, n = None, [], 0
    for t, frame in iter_frames(args.source, interval):
        h, w = frame.shape[:2]
        if grid is None:
            grid = build_grid(poly_to_px(poly, w, h), w, h, cell_px, overlap)
            print(f"lưới {len(grid)} ô (cell={cell_px})")
        occ = occluded_ids(grid.cells, boxes_of(frame), 0.3)
        cells = [c for c in grid.cells if c.id not in occ]
        scores += scorer.score([frame[c.y1:c.y2, c.x1:c.x2] for c in cells])
        n += 1
        if n % 20 == 0:
            print(f"  {n} lượt, {len(scores)} lần chấm")
        if args.limit and n >= args.limit:
            break

    if len(scores) < 200:
        print(f"! chỉ {len(scores)} lần chấm — quá ít để hiệu chuẩn. Cần >= 2.000,")
        print("  và nên trải qua nhiều khung giờ (sáng/trưa/chiều/đêm/IR).")
    s = np.array(scores)
    print(f"\n{len(s)} lần chấm trên vùng SẠCH ({n} lượt)")
    for q in (50, 90, 99, 99.9):
        print(f"  phân vị {q:5.1f}%: {np.percentile(s, q):.4f}")
    print(f"  cao nhất    : {s.max():.4f}")

    thr = float(min(0.9, max(0.05, s.max() * args.margin)))
    print(f"\n  ĐỀ XUẤT  decide.litter_thr: {thr:.2f}")
    print(f"           (= max nền {s.max():.3f} x {args.margin})")
    for t_ in (thr, 0.3, 0.6):
        print(f"    thr {t_:.2f} -> {int((s >= t_).sum())} ô nền vượt "
              f"({(s >= t_).sum()/max(1, n):.2f} ô/lượt)")

    if s.max() > 0.5:
        print("\n  ! Nền vẫn cho điểm cao — model chưa có đủ ô âm của site này.")
        print("    Thu thêm âm rồi train lại TRƯỚC, đừng nâng ngưỡng để che.")
    print("\n  Nhớ đo lại sau khi đổi mùa / lắp đèn / sơn lại mặt đường.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
