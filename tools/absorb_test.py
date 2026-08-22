"""Lỗ NUỐT RÁC: nền bị vứt lúc trong vùng đang có rác -> rác thành nền, im luôn.

    python3 tools/absorb_test.py

Đây là kiểu hỏng tệ nhất vì nó IM LẶNG — không cảnh báo, không log, hệ thống
trông vẫn khoẻ. Ba đường dẫn tới chỗ vứt nền:

  A. camera bị hích quá `scene_shift.thr_px`   -> tự động, không ai bấm gì
  B. vận hành bấm CHỐT LẠI NỀN                 -> chủ động
  C. đổi sáng toàn cục (rạng đông)             -> nạp lại nền theo guard

C vốn đã không nuốt (guard chỉ nạp lại mô tả, không xoá bộ đếm). A và B thì có.

Cách bịt, đo bằng chính file này:
  A: nắn khung về mốc TRƯỚC khi xét vứt nền — `_stabilize` bù được tới 40px còn
     ngưỡng vứt mới 12px, nên phần lớn cú hích nắn lại được, không cần vứt gì.
  B: sau khi vứt nền, quét detector khắp vùng `dwell` lượt. Cổng đổi mù vì mất
     nền, detector thì không cần nền.
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

from core.pipeline import ZoneTrashDetector   # noqa: E402
from core.scorers import build_scorer         # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
SITE = os.path.join(ROOT, "data", "site", "pos_raw")
FULL_POLY = [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]]


def knock(im: np.ndarray, px: int) -> np.ndarray:
    """Hích camera `px` pixel. BORDER_REPLICATE chứ không để viền đen — viền đen
    tự nó là một mảng 'đổi' khổng lồ, đo ra số đẹp mà chẳng nói gì về lỗ này."""
    M = np.float32([[1, 0, px], [0, 1, px * 0.4]])
    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


def run(cfg: dict, ref: np.ndarray, trash: np.ndarray, mode: str,
        n: int = 15, shift_px: int = 30) -> tuple:
    """-> (số ô nóng lớn nhất trong n lượt, số lượt bị vứt nền)"""
    det = ZoneTrashDetector(cfg, build_scorer({"kind": "constant", "value": 0.0}),
                            camera_id=f"absorb_{mode}", zone_id="z")
    t = 0.0
    for _ in range(4):                        # dựng nền từ khung SẠCH
        det.scan(ref, FULL_POLY, [], now=t)
        t += 30.0
    det.scan(trash, FULL_POLY, [], now=t)     # rác xuất hiện
    t += 30.0
    if mode == "B":
        det.reset_background()                # vận hành chốt lại nền ĐÚNG LÚC CÓ RÁC
    best, n_reset = 0, 0
    for i in range(n):
        fr = knock(trash, shift_px) if mode == "A" else trash
        res = det.scan(fr, FULL_POLY, [], now=t)
        t += 30.0
        n_reset += int(getattr(res, "ref_reset", False))
        best = max(best, len(res.hot))
    return best, n_reset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/day_cfg.yaml")
    ap.add_argument("--shift-px", type=int, default=30)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg["clutter"] = dict(cfg.get("clutter", {}), enabled=False)

    ref = np.load(os.path.join(SITE, "_ref.npy"))
    trash = None
    for fn in sorted(f for f in os.listdir(SITE) if f.startswith("frame_")):
        im = cv2.imread(os.path.join(SITE, fn))
        if im is not None and im.shape == ref.shape:
            d = cv2.absdiff(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
            if float((d > 40).mean()) > 0.002:     # khung có rác rõ
                trash = im
                break
    if trash is None:
        print("khong tim duoc khung co rac trong", SITE)
        return 1

    print(f"config {args.config} · hích {args.shift_px}px · vùng toàn khung\n")
    cases = [
        ("KIỂM CHỨNG  không vứt nền, rác nằm yên", "ctl"),
        (f"A  camera bị hích {args.shift_px}px lúc có rác", "A"),
        ("B  vận hành CHỐT LẠI NỀN lúc có rác", "B"),
    ]
    bad = 0
    for label, mode in cases:
        best, n_reset = run(cfg, ref, trash, mode, shift_px=args.shift_px)
        ok = best > 0
        bad += 0 if ok else 1
        print(f"  {label:46s} ô nóng={best:3d}  vứt nền={n_reset}  "
              f"{'ok' if ok else '*** NUỐT MẤT RÁC ***'}")
    print("\n" + ("Không ca nào nuốt rác." if not bad else f"{bad} ca còn nuốt rác."))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
