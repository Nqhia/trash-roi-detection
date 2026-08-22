"""So model o DIEM LAM VIEC TUONG DUONG, khong so o cung mot nguong conf.

    python3 tools/bench_sweep.py --models a.pt repo/id ...

Vi sao can: `bench_models.py` so tat ca o conf 0,10 va xep sharktide dau bang
voi 100% recall — nhung no rai 118 hop len 6 khung chua 20 vat. Cac model hieu
chinh khac nhau, nen mot nguong chung khong noi len dieu gi. Model A o conf 0,1
co the tuong duong model B o conf 0,4.

Cach so dung: quet conf cho TUNG model, roi doc hai lat cat
  * cung do NHIEU  — o muc ~4 hop thua (bang model hien tai), recall bao nhieu
  * cung RECALL    — o muc 90% recall, phai chiu bao nhieu hop thua

Do la duong cong danh doi thuc su cua tung model.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

from bench_models import detect, load                      # noqa: E402
from run_test_cases import ABODA, SITE, gt_boxes, hits     # noqa: E402

CONFS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--clean", type=int, default=12)
    args = ap.parse_args()

    ref = np.load(f"{SITE}/_ref.npy")
    eco = []
    for fn in sorted(f for f in os.listdir(SITE) if f.startswith("frame_")):
        im = cv2.imread(os.path.join(SITE, fn))
        if im is not None and im.shape == ref.shape:
            g = gt_boxes(im, ref)
            if len(g) >= 3:
                eco.append((im, g))
    eco = eco[11:11 + args.frames]
    n_gt = sum(len(g) for _, g in eco)

    clean = []
    for vid in ("video1", "video3", "video6"):
        cap = cv2.VideoCapture(f"{ABODA}/{vid}.avi")
        k = 0
        while len(clean) < args.clean * 3:
            ok, f = cap.read()
            if not ok:
                break
            if k % 40 == 0:
                clean.append(f)
            k += 1
        cap.release()
    clean = clean[:args.clean]
    print(f"{len(eco)} khung eco / {n_gt} vat · {len(clean)} khung sach\n")

    summary = []
    for spec in args.models:
        try:
            m, name = load(spec)
        except Exception as e:
            print(f"{spec}: NAP LOI {str(e)[:50]}\n")
            continue
        short = name.split(":")[0].split("/")[-1][:28]
        print(f"=== {short} ({len(m.names)} lop) ===")
        print(f"  {'conf':>5s} {'bat duoc':>12s} {'hop thua':>9s} {'sach':>7s}")
        curve = []
        for c in CONFS:
            hit = nbox = 0
            for im, gts in eco:
                b = detect(m, im, c)
                hit += sum(hits(b, g) for g in gts)
                nbox += len(b)
            nfp = sum(len(detect(m, f, c)) for f in clean) / max(1, len(clean))
            extra = max(0, nbox - n_gt)
            curve.append((c, hit, extra, nfp))
            print(f"  {c:5.2f} {hit:4d}/{n_gt:<3d} {100*hit/n_gt:4.0f}% "
                  f"{extra:8d} {nfp:7.1f}")

        # lat cat 1: o muc <=4 hop thua (bang model hien tai), recall cao nhat
        a = max((x for x in curve if x[2] <= 4), key=lambda x: x[1], default=None)
        # lat cat 2: o muc >=90% recall, it hop thua nhat
        b = min((x for x in curve if x[1] >= 0.9 * n_gt), key=lambda x: x[2],
                default=None)
        summary.append((short, a, b))
        print()

    print("=" * 72)
    print("DIEM LAM VIEC TUONG DUONG")
    print(f"  {'model':30s} {'<=4 hop thua':>22s} {'>=90% recall':>18s}")
    print("  " + "-" * 70)
    for short, a, b in summary:
        sa = f"conf {a[0]:.2f} -> {100*a[1]/n_gt:.0f}%" if a else "khong dat duoc"
        sb = f"conf {b[0]:.2f} -> {b[2]} hop thua" if b else "khong dat duoc"
        print(f"  {short:30s} {sa:>22s} {sb:>18s}")
    print("\n  Cot 1: chiu cung do nhieu nhu model hien tai thi bat duoc bao nhieu.")
    print("  Cot 2: muon dat 90% recall thi phai chiu bao nhieu hop thua.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
