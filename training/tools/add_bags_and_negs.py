"""Nhan nhom TUI trong TACO + tron anh am HIEN TRUONG vao tap tile.

    python3 tools/add_bags_and_negs.py --out data/tiles26 --bag-rep 4

Hai lo do duoc, ca hai deu ung voi ca that da fail:

TUI. Ca that duy nhat co rac that tren camera that la mot TUI NI LONG trang buoc
nut — he thong khong bao trong 4,5 gio. Dem trong TACO: "Garbage bag" chi 31 hop,
ca nhom tui/mang ~850 tren 26k. Cac bo chai/lon khong co tui nao. Nhan nhom nay
len de model duoc nhin nhieu lan hon; khong tao du lieu moi.

ANH AM. Cac bo cong khai deu ngoai troi, nen anh am sinh ra tu chung la nhua
duong, co, cat. Bat nham do duoc o hien truong lai la chan ghe, mep vach kinh,
chan tuong, gach — khong bo nao co. 2393 anh am dao tu chinh camera do (goc cu,
nhung be mat van con o goc moi) la thu tri dung benh.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from tools.make_tiles_data import split_of                   # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TACO_ANN = f"{ROOT}/data/taco/annotations.json"
TACO_IMG = f"{ROOT}/data/taco/images"
SITE_NEG = os.path.join(HERE, "data", "site_neg")
BAG_WORDS = ("bag", "wrapper", "film", "crisp packet")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/tiles26")
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--upscale", type=float, default=2.0)
    ap.add_argument("--bag-rep", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args()
    side = int(a.tile * a.upscale)
    rng = np.random.default_rng(1)
    for sp in ("train", "val"):
        for k in ("images", "labels"):
            os.makedirs(os.path.join(a.out, sp, k), exist_ok=True)

    # ---------- 1) TUI tu TACO ----------
    ann = json.load(open(TACO_ANN))
    cats = {c["id"]: c for c in ann["categories"]}
    bag_ids = {cid for cid, c in cats.items()
               if any(w in (c["name"] + " " + c.get("supercategory", "")).lower()
                      for w in BAG_WORDS)}
    print(f"lop tui: {sorted(cats[c]['name'] for c in bag_ids)}")
    meta = {im["id"]: im for im in ann["images"]}
    by = {}
    for an in ann["annotations"]:
        if an["category_id"] in bag_ids:
            by.setdefault(an["image_id"], []).append(an["bbox"])
    n_tile = n_box = 0
    for iid, raw in by.items():
        p = os.path.join(TACO_IMG, meta[iid]["file_name"].replace("/", "_"))
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        k = w / float(meta[iid]["width"])          # anh dia nho hon nhan
        bs = [(bx * k, byy * k, bw * k, bh * k) for bx, byy, bw, bh in raw
              if bw * k >= 4 and bh * k >= 4]
        if not bs:
            continue
        sp = split_of(f"TACObag:{meta[iid]['file_name']}", a.val_frac)
        for j in range(a.bag_rep):
            # cat o quanh MOT cai tui, vi tri ngau nhien -> moi lan mot boi canh
            bx, byy, bw, bh = bs[rng.integers(0, len(bs))]
            cx, cy = bx + bw / 2, byy + bh / 2
            t = a.tile
            x0 = int(np.clip(cx - t / 2 + rng.integers(-t // 4, t // 4), 0, max(0, w - t)))
            y0 = int(np.clip(cy - t / 2 + rng.integers(-t // 4, t // 4), 0, max(0, h - t)))
            crop = img[y0:y0 + t, x0:x0 + t]
            if crop.shape[0] < 32 or crop.shape[1] < 32:
                continue
            out = cv2.resize(crop, (side, side), interpolation=cv2.INTER_CUBIC)
            sx, sy = side / crop.shape[1], side / crop.shape[0]
            lines = []
            for ox, oy, ow, oh in bs:
                x1, y1 = (ox - x0) * sx, (oy - y0) * sy
                x2, y2 = (ox + ow - x0) * sx, (oy + oh - y0) * sy
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(side), x2), min(float(side), y2)
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                lines.append(f"0 {(x1+x2)/2/side:.6f} {(y1+y2)/2/side:.6f} "
                             f"{(x2-x1)/side:.6f} {(y2-y1)/side:.6f}")
            if not lines:
                continue
            stem = f"TACObag_{iid}_{j}"
            cv2.imwrite(os.path.join(a.out, sp, "images", stem + ".jpg"), out,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(a.out, sp, "labels", stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            n_tile += 1
            n_box += len(lines)
    print(f"  tui TACO: {len(by)} anh nguon -> {n_tile} o, {n_box} hop (x{a.bag_rep})")

    # ---------- 2) anh am hien truong ----------
    n_neg = 0
    if os.path.isdir(SITE_NEG):
        fs = sorted(os.listdir(SITE_NEG))
        for f in fs:
            img = cv2.imread(os.path.join(SITE_NEG, f))
            if img is None or min(img.shape[:2]) < 24:
                continue
            # lot vao giua o, nen xam nhu ban ghep cua ultralytics
            s = min(side / img.shape[1], side / img.shape[0], 3.0)
            sm = cv2.resize(img, (max(8, int(img.shape[1] * s)),
                                  max(8, int(img.shape[0] * s))),
                            interpolation=cv2.INTER_CUBIC)
            canvas = np.full((side, side, 3), 114, np.uint8)
            ox = rng.integers(0, max(1, side - sm.shape[1] + 1))
            oy = rng.integers(0, max(1, side - sm.shape[0] + 1))
            canvas[oy:oy + sm.shape[0], ox:ox + sm.shape[1]] = sm
            sp = split_of(f"siteneg:{f}", a.val_frac)
            # KHONG ghi file nhan -> ultralytics coi la anh nen
            cv2.imwrite(os.path.join(a.out, sp, "images", f"siteneg_{n_neg:05d}.jpg"),
                        canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n_neg += 1
        print(f"  anh am hien truong: {n_neg} o (khong nhan -> anh nen)")
    else:
        print(f"  khong thay {SITE_NEG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
