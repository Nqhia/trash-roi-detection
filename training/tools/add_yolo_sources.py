"""Them cac bo nhan YOLO (chup GAN) vao tap tile, co THU NHO theo co muc tieu.

    python3 tools/add_yolo_sources.py --out data/tiles

Chay SAU make_tiles_data.py, ghi them vao cung thu muc.

VI SAO PHAI THU NHO
-------------------
Pipeline khong bao gio chay model tren toan khung: no cat vung 160-320px quanh
cho doi roi phong len 640px. Nen vat 44px trong vung 240px chiem 18% anh vao
model. Do la dai lam viec that: khoang 6-40%.

Cac bo chup gan (Drinking Waste, bottles_wild, bagbottle) co vat chiem 130-330%
neu quy ve vung 240px — tuc mot vat chiem ca vung. De nguyen la day model nhan
vat KHONG LOT trong o cat, dieu khong bao gio xay ra luc chay.

Thu nho anh nguon la phep bien doi THAT (lay mau lai pixel that), khong phai du
lieu dung. Ta chon he so sao cho vat roi vao giua dai, roi lot nen cho du o.

GIOI HAN phai noi ro: vat 400px thu ve 40px SAC NET hon vat 40px chup that bang
cam CCTV — khong nhieu cam bien, khong nhoe chuyen dong, khong vo nen JPEG. Nen
co tuy chon --degrade de them lai dung ba thu do.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from tools.make_tiles_data import split_of                    # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
K = f"{ROOT}/data/kaggle"

# (ten, thu muc anh, thu muc nhan, lop giu lai hoac None = tat ca)
YOLO_SRC = [
    ("drinking",  f"{K}/drinking/Images_of_Waste/YOLO_imgs",
                  f"{K}/drinking/Images_of_Waste/YOLO_imgs", None),
    ("bottleswild", f"{K}/Plastic Bottle Image Dataset/train/images",
                  f"{K}/Plastic Bottle Image Dataset/train/labels", None),
    ("bagbottle", f"{K}/bagbottle/dataset/images/train",
                  f"{K}/bagbottle/dataset/labels/train", None),
    ("recyc",     f"{K}/recyc/Final Data/images/train",
                  f"{K}/recyc/Final Data/labels/train", None),
]


def degrade(img, rng):
    """Them lai thu ma phep thu nho da lam mat: nhieu cam bien, nhoe, vo nen."""
    if rng.random() < 0.5:
        k = rng.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.7:
        n = rng.normal(0, rng.uniform(2, 7), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        q = int(rng.uniform(35, 75))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/tiles")
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--upscale", type=float, default=2.0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--target-lo", type=float, default=0.08, help="canh vat / canh o, min")
    ap.add_argument("--target-hi", type=float, default=0.35, help="canh vat / canh o, max")
    ap.add_argument("--per-img", type=int, default=2, help="so o sinh ra moi anh nguon")
    ap.add_argument("--degrade", action="store_true", default=True)
    ap.add_argument("--max-per-src", type=int, default=0)
    a = ap.parse_args()

    side = int(a.tile * a.upscale)
    rng = np.random.default_rng(0)
    random.seed(0)
    for sp in ("train", "val"):
        for k in ("images", "labels"):
            os.makedirs(os.path.join(a.out, sp, k), exist_ok=True)

    grand = 0
    for name, imgd, labd, keep in YOLO_SRC:
        if not os.path.isdir(imgd):
            print(f"  {name:12s} khong co thu muc, bo qua")
            continue
        files = sorted(glob.glob(os.path.join(imgd, "*.jpg")) +
                       glob.glob(os.path.join(imgd, "*.png")))
        if a.max_per_src and len(files) > a.max_per_src:
            random.shuffle(files)
            files = files[:a.max_per_src]
        n_tile = n_box = n_skip = 0
        for ip in files:
            lp = os.path.join(labd, os.path.splitext(os.path.basename(ip))[0] + ".txt")
            if not os.path.exists(lp):
                continue
            img = cv2.imread(ip)
            if img is None:
                continue
            h, w = img.shape[:2]
            bs = []
            for line in open(lp):
                p = line.split()
                if len(p) < 5:
                    continue
                if keep is not None and int(p[0]) not in keep:
                    continue
                cx, cy, bw, bh = (float(x) for x in p[1:5])
                bs.append((cx * w, cy * h, bw * w, bh * h))
            if not bs:
                continue
            sp = split_of(f"{name}:{os.path.basename(ip)}", a.val_frac)
            for j in range(a.per_img):
                # CAT QUANH VAT ROI MOI PHONG — khong thu nho ca anh.
                #
                # Ban dau toi thu nho toan anh cho vat dat co muc tieu. Sai: vat
                # von chiem 60% anh nguon nen ca anh co lai con be xiu, va 90% o
                # la nen xam vo nghia. Soi mat thay ngay: vat ti hon giua bien
                # xam. Model se hoc "rac = vat nho giua nen phang" — dieu khong
                # bao gio dung o camera that.
                #
                # Cat mot vung rong `canh_vat / ty_le_muc_tieu` roi phong vua o:
                # vat dat dung co MA o van day boi canh that.
                bi = int(rng.integers(0, len(bs)))
                cx0, cy0, bw0, bh0 = bs[bi]
                tgt = rng.uniform(a.target_lo, a.target_hi)
                need = max(bw0, bh0) / tgt                 # canh vung can cat
                need = float(min(need, min(w, h)))         # khong vuot anh nguon
                need = max(need, 24.0)
                jx = rng.uniform(-0.18, 0.18) * need
                jy = rng.uniform(-0.18, 0.18) * need
                x0 = float(np.clip(cx0 + jx - need / 2, 0, max(0.0, w - need)))
                y0 = float(np.clip(cy0 + jy - need / 2, 0, max(0.0, h - need)))
                crop = img[int(y0):int(y0 + need), int(x0):int(x0 + need)]
                if crop.shape[0] < 16 or crop.shape[1] < 16:
                    continue
                canvas = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
                if a.degrade:
                    canvas = degrade(canvas, rng)
                sx = side / crop.shape[1]
                sy = side / crop.shape[0]
                lines = []
                for cx, cy, bw, bh in bs:
                    nx, ny = (cx - x0) * sx, (cy - y0) * sy
                    nw, nh = bw * sx, bh * sy
                    x1, y1 = nx - nw / 2, ny - nh / 2
                    x2, y2 = nx + nw / 2, ny + nh / 2
                    x1, y1 = max(0.0, x1), max(0.0, y1)
                    x2, y2 = min(float(side), x2), min(float(side), y2)
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        continue
                    # bo hop chi con lot mot mau ngoai ria
                    if (x2 - x1) * (y2 - y1) < 0.25 * nw * nh:
                        continue
                    lines.append(f"0 {(x1+x2)/2/side:.6f} {(y1+y2)/2/side:.6f} "
                                 f"{(x2-x1)/side:.6f} {(y2-y1)/side:.6f}")
                if not lines:
                    n_skip += 1
                    continue
                stem = f"{name}_{os.path.splitext(os.path.basename(ip))[0]}_{j}"
                cv2.imwrite(os.path.join(a.out, sp, "images", stem + ".jpg"), canvas,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(os.path.join(a.out, sp, "labels", stem + ".txt"), "w") as f:
                    f.write("\n".join(lines) + "\n")
                n_tile += 1
                n_box += len(lines)
        print(f"  {name:12s} {len(files):6d} anh nguon -> {n_tile:6d} o, "
              f"{n_box:6d} hop (bo {n_skip} o vi hop qua nho)")
        grand += n_tile
    print(f"\nthem {grand} o vao {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
