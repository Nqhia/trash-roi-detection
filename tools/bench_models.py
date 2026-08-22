"""So NHIEU model rac bang CUNG mot thuoc do.

    python3 tools/bench_models.py --hf esapzoi/litter-detection-yolov8 ...

Bat ky model nao ultralytics nap duoc deu so duoc voi model hien tai, tren dung
hai tap kiem cua `run_test_cases.py`:

  * khung eco   — camera EcoVision, rac that do nguoi vut. Nhan lay bang tru nen.
  * chuoi sach  — 3 video ABODA co nguoi di, KHONG co rac.

Model ngoai co the co NHIEU LOP (chai, lon, giay...). Bai toan nay khong can
phan loai nen gop TAT CA lop lam mot — cong bang voi model 1 lop cua ta, va
dung voi yeu cau "vung trong co vat la thi bao".

Chay cung cach pipeline goi: cat o 320px, phong 2x, NMS. Goi khac di (nem ca
khung vao 640px) thi ket qua tut hang — do duoc 29% so voi 75%.
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

from run_test_cases import ABODA, FULL_POLY, SITE, gt_boxes, hits  # noqa: E402


def tiles_of(w, h, tile, overlap=0.5):
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), step)) or [0]
    ys = list(range(0, max(1, h - tile + 1), step)) or [0]
    if xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    if ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    return [(x, y) for y in ys for x in xs]


def detect(model, img, conf, tile=320, upscale=2.0):
    crops, offs = [], []
    for x, y in tiles_of(img.shape[1], img.shape[0], tile):
        c = img[y:y + tile, x:x + tile]
        if c.size:
            crops.append(cv2.resize(c, None, fx=upscale, fy=upscale,
                                    interpolation=cv2.INTER_CUBIC))
            offs.append((x, y))
    if not crops:
        return []
    out = []
    for (ox, oy), r in zip(offs, model.predict(crops, conf=conf, verbose=False)):
        for b in r.boxes:
            q = b.xyxy[0].tolist()
            out.append(((ox + q[0] / upscale, oy + q[1] / upscale,
                         ox + q[2] / upscale, oy + q[3] / upscale), float(b.conf)))
    if not out:
        return []
    bx = np.array([o[0] for o in out], np.float32)
    sc = np.array([o[1] for o in out], np.float32)
    area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
    order, keep = sc.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(bx[i, 0], bx[rest, 0]); yy1 = np.maximum(bx[i, 1], bx[rest, 1])
        xx2 = np.minimum(bx[i, 2], bx[rest, 2]); yy2 = np.minimum(bx[i, 3], bx[rest, 3])
        it = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = rest[it / (area[i] + area[rest] - it + 1e-9) <= 0.5]
    return [out[i][0] for i in keep]


def load(spec):
    """spec = duong dan .pt, hoac 'repo/id' tren HuggingFace."""
    from ultralytics import YOLO
    if os.path.exists(spec):
        return YOLO(spec), spec
    from huggingface_hub import hf_hub_download, list_repo_files
    fs = [f for f in list_repo_files(spec) if f.endswith(".pt")]
    if not fs:
        raise FileNotFoundError(f"{spec}: khong co file .pt")
    pref = [f for f in fs if "best" in f] or fs
    p = hf_hub_download(spec, pref[0])
    return YOLO(p), f"{spec}:{pref[0]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="duong dan .pt hoac repo HuggingFace")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--clean", type=int, default=12)
    args = ap.parse_args()

    # ---- du lieu ----
    ref = np.load(f"{SITE}/_ref.npy")
    eco = []
    for fn in sorted(f for f in os.listdir(SITE) if f.startswith("frame_")):
        im = cv2.imread(os.path.join(SITE, fn))
        if im is not None and im.shape == ref.shape:
            g = gt_boxes(im, ref)
            if len(g) >= 3:
                eco.append((im, g))
    eco = eco[11:11 + args.frames]          # bo qua phan dau nhu run_test_cases
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

    print(f"{len(eco)} khung eco / {n_gt} vat that · {len(clean)} khung sach\n")
    print(f"  {'model':40s} {'lop':>4s}  {'bat duoc':>10s}  {'hop':>5s} {'hop thua':>8s}  {'hop/khung sach':>8s}")
    print("  " + "-" * 92)

    for spec in args.models:
        try:
            m, name = load(spec)
        except Exception as e:
            print(f"  {spec:46s}  NAP LOI: {str(e)[:40]}")
            continue
        ncls = len(m.names)
        hit = n_box = 0
        for im, gts in eco:
            b = detect(m, im, args.conf)
            hit += sum(hits(b, g) for g in gts)
            n_box += len(b)
        nfp = sum(len(detect(m, f, args.conf)) for f in clean)
        # PHAI co cot nay. Do moi recall thi mot model rai hop khap noi se an
        # diem cao ma khong he "nhin thay" gi: sharktide ra 20 hop tren khung co
        # 4 vat va trung het — do la van de precision doi lot recall.
        extra = max(0, n_box - n_gt)
        print(f"  {name[:40]:40s} {ncls:4d}  {hit:3d}/{n_gt:<3d} {100*hit/max(1,n_gt):3.0f}%"
              f"  {n_box:5d} {extra:8d}  {nfp/max(1,len(clean)):8.1f}")
    print("\n  bat duoc = tren khung eco (rac that) · hop thua = hop khong trung vat nao")
    print("  Model rai hop khap noi an diem recall cao ma khong nhin thay gi — phai")
    print("  doc CA HAI cot. Model nhieu lop duoc gop het lam mot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
