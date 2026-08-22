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
# Vùng bị sửa trên UI: cùng cảnh, khác vân tay lưới -> nền cũ bị bỏ.
EDITED_POLY = [[0.05, 0.02], [0.98, 0.02], [0.98, 0.95], [0.05, 0.95]]


def knock(im: np.ndarray, px: int) -> np.ndarray:
    """Hích camera `px` pixel. BORDER_REPLICATE chứ không để viền đen — viền đen
    tự nó là một mảng 'đổi' khổng lồ, đo ra số đẹp mà chẳng nói gì về lỗ này."""
    M = np.float32([[1, 0, px], [0, 1, px * 0.4]])
    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)


def cloud(im: np.ndarray, frac: float = 0.7) -> np.ndarray:
    """Bóng mây trùm 70% khung — cái THỰC SỰ bắn guard đổi sáng toàn cục.

    Đổi sáng ĐỀU thì không bắn được: mô tả ô đã trừ trung bình nên cộng 55 vào
    cả khung làm mô tả gần như không đổi (đo được 88/360 ô đổi, guard 0/10).
    Phải đổi TƯƠNG PHẢN mới chạm tới guard — bóng mây, gamma, đèn pha.
    """
    o = im.astype(np.float32)
    o[:, :int(im.shape[1] * frac)] *= 0.45
    return np.clip(o, 0, 255).astype(np.uint8)


def run(cfg: dict, ref: np.ndarray, trash: np.ndarray, mode: str,
        n: int = 15, shift_px: int = 30) -> tuple:
    """-> (số ô nóng lớn nhất, số lượt vứt nền, số lượt guard bắn)"""
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
    best, n_reset, n_guard = 0, 0, 0
    for i in range(n):
        if mode == "A":
            fr = knock(trash, shift_px)
        elif mode == "C":
            fr = cloud(trash)
        else:
            fr = trash
        poly = EDITED_POLY if mode == "D" else FULL_POLY
        res = det.scan(fr, poly, [], now=t)
        t += 30.0
        n_reset += int(getattr(res, "ref_reset", False))
        n_guard += int(res.global_change)
        best = max(best, len(res.hot))
    return best, n_reset, n_guard


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
        ("C  bóng mây trùm 70% khung lúc có rác", "C"),
        ("D  vận hành SỬA VÙNG trên UI lúc có rác", "D"),
    ]
    bad = 0
    for label, mode in cases:
        best, n_reset, n_guard = run(cfg, ref, trash, mode, shift_px=args.shift_px)
        ok = best > 0
        bad += 0 if ok else 1
        print(f"  {label:46s} ô nóng={best:3d}  vứt nền={n_reset}  "
              f"guard={n_guard:2d}/15  {'ok' if ok else '*** NUỐT MẤT RÁC ***'}")
    print("\n" + ("Không ca nào nuốt rác." if not bad else f"{bad} ca còn nuốt rác."))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
