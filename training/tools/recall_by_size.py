"""Recall theo CO VAT (px). Tra loi: train them co cuu duoc khong?

    python3 tools/recall_by_size.py --weights ../models/trash_yolo11n.pt

Neu vat truot don het o day nho (<20px) thi train them KHONG cuu duoc — thieu
pixel chu khong thieu du lieu, va cach sua la dat camera gan hon / tang do phan
giai, khong phai gan them nhan. Neu truot rai deu moi cо co so noi la model yeu.

Cham theo tam hop (centre_in) chu khong IoU: pipeline chi can biet "co vat o o
nay khong", khong can hop khop khit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

ROOT = "/mnt/c/Users/kangh/Documents/TrashDataset"
WADE = f"{ROOT}/data/wade/wade-ai/Trash_Detection/trash/dataset"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.eval_datasets import centre_in, detect          # noqa: E402
from tools.make_tiles_data import clip_of, split_of        # noqa: E402

SETS = [
    ("TACO", f"{ROOT}/data/taco/annotations.json", f"{ROOT}/data/taco/images",
     lambda fn: fn.replace("/", "_")),
    ("RoLID", f"{ROOT}/data/rolid/RoLID-11K/validation.json",
     f"{ROOT}/data/rolid/RoLID-11K/val_images", lambda fn: fn.split("/")[-1]),
    ("UAVVaste", f"{ROOT}/data/uavvaste/annotations.json",
     f"{ROOT}/data/uavvaste/images", lambda fn: fn.split("/")[-1]),
    ("Wade", f"{WADE}/train_wade_ai.json", f"{WADE}/train",
     lambda fn: fn.split("/")[-1]),
]

# Chai nhua o CCTV 1080p: cao ~25cm. Cam cao 4m nhin xa 10m -> ~27px.
BUCKETS = [(0, 12), (12, 20), (20, 32), (32, 48), (48, 80), (80, 10 ** 9)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="../models/trash_yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-side", type=int, default=1280)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    print(f"conf {args.conf} · val chia theo clip · toi da {args.n} anh/bo\n")

    tally = {b: [0, 0] for b in BUCKETS}          # bucket -> [trung, tong]
    for name, annf, imgd, ren in SETS:
        if not os.path.exists(annf):
            continue
        a = json.load(open(annf))
        meta = {im["id"]: im for im in a["images"]}
        by = {}
        for an in a["annotations"]:
            by.setdefault(an["image_id"], []).append(an["bbox"])
        ids = [i for i in by
               if os.path.exists(os.path.join(imgd, ren(meta[i]["file_name"])))
               and split_of(clip_of(name, meta[i]["file_name"]), 0.15) == "val"]
        ids.sort()
        loc = {b: [0, 0] for b in BUCKETS}
        nimg = 0
        for iid in ids:
            if nimg >= args.n:
                break
            m = meta[iid]
            img = cv2.imread(os.path.join(imgd, ren(m["file_name"])))
            if img is None:
                continue
            h, w = img.shape[:2]
            k = w / float(m["width"])
            s = min(1.0, args.max_side / max(h, w))
            if s < 1.0:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            gts = [((bx * k) * s, (byy * k) * s,
                    (bx + bw) * k * s, (byy + bh) * k * s)
                   for bx, byy, bw, bh in by[iid]]
            gts = [g for g in gts if g[2] - g[0] >= 2 and g[3] - g[1] >= 2]
            if not gts:
                continue
            boxes = detect(model, img, args.conf)
            nimg += 1
            for g in gts:
                side = max(g[2] - g[0], g[3] - g[1])
                b = next(b for b in BUCKETS if b[0] <= side < b[1])
                ok = 1 if any(centre_in(x, g) for x in boxes) else 0
                loc[b][0] += ok
                loc[b][1] += 1
                tally[b][0] += ok
                tally[b][1] += 1
        cols = []
        for lo, hi in BUCKETS:
            h_, t = loc[(lo, hi)]
            lbl = f"{lo}-{hi}" if hi < 10 ** 9 else f">{lo}"
            cols.append(f"{lbl}:{100*h_/t:3.0f}%({t})" if t else f"{lbl}:  -")
        print(f"  {name:10s} " + "  ".join(cols))

    print("\n  canh vat (px)   recall   so vat")
    for lo, hi in BUCKETS:
        h_, t = tally[(lo, hi)]
        lbl = f"{lo}-{hi}" if hi < 10 ** 9 else f">{lo}"
        if t:
            print(f"  {lbl:>13s}   {100*h_/t:5.1f}%   {t:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
