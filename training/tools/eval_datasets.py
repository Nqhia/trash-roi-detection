"""Danh gia model tren NHIEU bo rac, tach ro bo nao model DA THAY khi train.

    python3 tools/eval_datasets.py --weights ../models/trash_yolo11n.pt

Cau hoi can tra loi: co can train them khong. Muon tra loi thi phai tach ba
loai du lieu ra, vi chung noi ba dieu khac nhau:

  DA TRAIN      TACO / RoLID / UAVVaste / Wade phan train
                -> con so o day la HOC THUOC, khong dung de ket luan
  VAL cung bo   phan val chia theo CLIP (khung lien tiep cung video khong bi
                ném sang hai phia) -> tong quat hoa TRONG MIEN
  CHUA HE THAY  GINI (907 anh bai rac, chua bao gio dua vao train) va anh bai
                rac Wade bi `--max-box-frac 0.40` loai bo
                -> tong quat hoa SANG MIEN KHAC, day moi la cau tra loi

GINI gan nhan o muc VUNG (mot hop phu ca dong rac) chu khong phai tung vat,
nen cham "co phat hien nao nam trong vung do khong", khong cham IoU. Cham IoU
o day se ket luan sai hoan toan.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys

import cv2
import numpy as np

ROOT = "/mnt/c/Users/kangh/Documents/TrashDataset"
WADE = f"{ROOT}/data/wade/wade-ai/Trash_Detection/trash/dataset"
GINI = f"{ROOT}/data/gini/spotgarbage-GINI-master/spotgarbage"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.make_tiles_data import clip_of, split_of        # noqa: E402

COCO = [
    ("TACO", f"{ROOT}/data/taco/annotations.json", f"{ROOT}/data/taco/images",
     lambda fn: fn.replace("/", "_")),
    ("RoLID", f"{ROOT}/data/rolid/RoLID-11K/validation.json",
     f"{ROOT}/data/rolid/RoLID-11K/val_images", lambda fn: fn.split("/")[-1]),
    ("RoLIDtr", f"{ROOT}/data/rolid/RoLID-11K/training.json",
     f"{ROOT}/data/rolid/RoLID-11K/train_images", lambda fn: fn.split("/")[-1]),
    ("UAVVaste", f"{ROOT}/data/uavvaste/annotations.json",
     f"{ROOT}/data/uavvaste/images", lambda fn: fn.split("/")[-1]),
    ("WadeTr", f"{WADE}/train_wade_ai.json", f"{WADE}/train",
     lambda fn: fn.split("/")[-1]),
]


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
            out.append((ox + q[0] / upscale, oy + q[1] / upscale,
                        ox + q[2] / upscale, oy + q[3] / upscale))
    return out


def centre_in(b, g):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return g[0] <= cx <= g[2] and g[1] <= cy <= g[3]


def eval_coco(model, name, annf, imgd, ren, conf, n, split, max_side=1280):
    """-> (so vat trung, tong vat, so hop, so anh)"""
    if not os.path.exists(annf):
        return None
    a = json.load(open(annf))
    meta = {im["id"]: im for im in a["images"]}
    by = {}
    for an in a["annotations"]:
        by.setdefault(an["image_id"], []).append(an["bbox"])
    ids = [i for i in by if os.path.exists(os.path.join(imgd, ren(meta[i]["file_name"])))]
    if split != "all":
        ids = [i for i in ids
               if split_of(clip_of(name, meta[i]["file_name"]), 0.15) == split]
    random.seed(0)
    random.shuffle(ids)
    hit = tot = nbox = nimg = 0
    for iid in ids:
        if nimg >= n:
            break
        m = meta[iid]
        img = cv2.imread(os.path.join(imgd, ren(m["file_name"])))
        if img is None:
            continue
        h, w = img.shape[:2]
        k = w / float(m["width"])
        # Thu ve co CCTV thuc te: anh 4032px thi vat chiem 70% khung, khong bao
        # gio gap o camera treo cao.
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gts = [((bx * k) * s, (byy * k) * s, (bx + bw) * k * s, (byy + bh) * k * s)
               for bx, byy, bw, bh in by[iid]]
        gts = [g for g in gts if g[2] - g[0] >= 3 and g[3] - g[1] >= 3]
        if not gts:
            continue
        b = detect(model, img, conf)
        nimg += 1
        nbox += len(b)
        for g in gts:
            tot += 1
            hit += any(centre_in(x, g) for x in b)
    return hit, tot, nbox, nimg


def eval_gini(model, conf, n, max_side=1280):
    """GINI: nhan o muc VUNG. Cham 'co phat hien nao trong vung do khong'."""
    csvf = f"{GINI}/garbage-queried-images.csv"
    imgd = f"{GINI}/garbage-queried-images"
    if not os.path.exists(csvf):
        return None
    # anh nam trong thu muc con theo TRUY VAN -> map ten tran -> duong dan
    fmap = {}
    for dp, _dn, fs in os.walk(imgd):
        for f in fs:
            fmap.setdefault(f, os.path.join(dp, f))
    rows = []
    with open(csvf, newline="", encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            try:
                x1, y1, x2, y2 = (int(r["startX"]), int(r["startY"]),
                                  int(r["endX"]), int(r["endY"]))
            except (ValueError, KeyError, TypeError):
                continue           # dong khong co rac -> bo qua
            p = fmap.get(r["image"])
            if p:
                rows.append((p, (x1, y1, x2, y2)))
    random.seed(0)
    random.shuffle(rows)
    hit = tot = nbox = 0
    for p, g in rows[:n]:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        g = tuple(v * s for v in g)
        b = detect(model, img, conf)
        tot += 1
        nbox += len(b)
        hit += any(centre_in(x, g) for x in b)
    return hit, tot, nbox, tot


def eval_wade_piles(model, conf, n, min_frac=0.40, max_side=1280):
    """Anh bai rac Wade da bi --max-box-frac loai khoi tap train."""
    out = []
    for sp in ("train", "val"):
        p = f"{WADE}/{sp}_wade_ai.json"
        if not os.path.exists(p):
            continue
        a = json.load(open(p))
        meta = {im["id"]: im for im in a["images"]}
        by = {}
        for an in a["annotations"]:
            by.setdefault(an["image_id"], []).append(an["bbox"])
        for iid, boxes in by.items():
            m = meta[iid]
            big = [b for b in boxes
                   if max(b[2], b[3]) >= min_frac * max(m["width"], m["height"])]
            fp = os.path.join(WADE, sp, m["file_name"].split("/")[-1])
            if big and os.path.exists(fp):
                out.append((fp, big, float(m["width"])))
    random.seed(0)
    random.shuffle(out)
    hit = tot = nbox = 0
    for fp, boxes, mw in out[:n]:
        img = cv2.imread(fp)
        if img is None:
            continue
        h, w = img.shape[:2]
        k = w / mw
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gts = [((b[0] * k) * s, (b[1] * k) * s,
                (b[0] + b[2]) * k * s, (b[1] + b[3]) * k * s) for b in boxes]
        b = detect(model, img, conf)
        tot += 1
        nbox += len(b)
        hit += any(centre_in(x, g) for g in gts for x in b)
    return hit, tot, nbox, tot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="../models/trash_yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    from ultralytics import YOLO
    m = YOLO(args.weights)
    print(f"model {args.weights} · conf {args.conf} · toi da {args.n} anh moi bo\n")

    def row(lbl, r, note=""):
        if not r:
            print(f"  {lbl:24s} khong co du lieu")
            return
        hit, tot, nbox, nimg = r
        extra = max(0, nbox - tot)
        print(f"  {lbl:24s} {hit:4d}/{tot:<5d} {100*hit/max(1,tot):5.1f}%  "
              f"{nbox:5d} hop ({extra:4d} thua)  {nimg:4d} anh   {note}")

    print("### DA TRAIN — con so nay la HOC THUOC, khong ket luan gi")
    print(f"  {'bo':24s} {'trung/tong':>11s}  {'':>6s} {'hop':>18s} {'anh':>8s}")
    for name, annf, imgd, ren in COCO:
        row(name, eval_coco(m, name, annf, imgd, ren, args.conf, args.n, "train"))

    print("\n### VAL chia theo CLIP — tong quat hoa TRONG MIEN")
    for name, annf, imgd, ren in COCO:
        row(name, eval_coco(m, name, annf, imgd, ren, args.conf, args.n, "val"))

    print("\n### CHUA HE THAY — day moi la cau tra loi")
    row("GINI (bai rac)", eval_gini(m, args.conf, args.n), "nhan muc VUNG")
    row("Wade bai rac (da loai)", eval_wade_piles(m, args.conf, args.n),
        "nhan muc CA DONG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
