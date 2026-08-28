"""Soi nhan bang MAT + do co vat theo THUOC DUNG, cho moi bo du lieu.

    python3 tools/review_data.py --out ../test_cases/20_review

Hai viec, va viec thu nhat quan trong hon:

1. XUAT ANH CO HOP de nguoi nhin. Nhan tu dong tin duoc la chuyen chua bao gio
   dung o day: nhan TACO tung lech 1,6x lam recall do ra 0% o MOI co, va nhan
   ABODA tu do tung ra hop phu ca khung hinh o 3/11 video.

2. Do co vat theo TY LE ANH DUA VAO MODEL, khong phai ty le khung goc. Pipeline
   cat vung 160-320px roi phong len 640px, nen mot vat 44px trong vung 160px
   chiem 27,5% anh vao — chu khong phai 3,4% khung 1280. Do bang thuoc khung goc
   la sai he quy chieu gap 4 lan, va no da lam toi loai nham 5 bo du lieu.
"""
from __future__ import annotations
import argparse, json, os, random, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
D = f"{ROOT}/data"
K = f"{D}/kaggle"

def coco_src(name, annf, imgd, ren):
    if not os.path.exists(annf): return None
    a = json.load(open(annf)); meta = {i["id"]: i for i in a["images"]}
    by = {}
    for an in a["annotations"]:
        by.setdefault(an["image_id"], []).append(an["bbox"])
    items = []
    for iid, boxes in by.items():
        p = os.path.join(imgd, ren(meta[iid]["file_name"]))
        if os.path.exists(p):
            # Toa do hop nam trong he cua METADATA, con anh tren dia thuong nho
            # hon. Do duoc tren TACO: 399/400 anh lech, ty le 0,16x den 0,62x.
            # Tron hai he la sai co vat toi 6 lan — chinh cai bay tung lam
            # recall do ra 0% o MOI co truoc day.
            items.append((p, [(b[0], b[1], b[0]+b[2], b[1]+b[3]) for b in boxes],
                          float(meta[iid]["width"]), float(meta[iid]["height"])))
    return name, items

def yolo_src(name, imgd, labd, keep=None):
    """Nhan YOLO da chuan hoa 0-1 nen khong dinh bay lech ty le."""
    if not os.path.isdir(imgd): return None
    items = []
    for f in os.listdir(imgd):
        st = os.path.splitext(f)[0]
        lp = os.path.join(labd, st + ".txt")
        ip = os.path.join(imgd, f)
        if not os.path.exists(lp): continue
        im_w = im_h = None
        bs = []
        for line in open(lp):
            p = line.split()
            if len(p) < 5: continue
            if keep is not None and int(p[0]) not in keep: continue
            cx, cy, w, h = map(float, p[1:5])
            bs.append((cx-w/2, cy-h/2, cx+w/2, cy+h/2))   # chuan hoa
        if bs:
            items.append((ip, bs, None, None))
    return name, items

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../test_cases/20_review")
    ap.add_argument("--per", type=int, default=6)
    a = ap.parse_args()
    out = os.path.abspath(os.path.join(HERE, a.out))
    os.makedirs(out, exist_ok=True)

    W = f"{D}/wade/wade-ai/Trash_Detection/trash/dataset"
    HH = {14, 15, 21, 26, 17, 18, 16, 19, 20, 23}      # lop rac trong household-trash
    srcs = [
        coco_src("TACO", f"{D}/taco/annotations.json", f"{D}/taco/images", lambda x: x.replace("/","_")),
        coco_src("RoLID", f"{D}/rolid/RoLID-11K/training.json", f"{D}/rolid/RoLID-11K/train_images", lambda x: x.split("/")[-1]),
        coco_src("UAVVaste", f"{D}/uavvaste/annotations.json", f"{D}/uavvaste/images", lambda x: x.split("/")[-1]),
        coco_src("Wade", f"{W}/train_wade_ai.json", f"{W}/train", lambda x: x.split("/")[-1]),
        yolo_src("household", f"{K}/images/train", f"{K}/labels/train", HH),
        yolo_src("bottles_wild", f"{K}/Plastic Bottle Image Dataset/train/images",
                 f"{K}/Plastic Bottle Image Dataset/train/labels"),
        yolo_src("drinking", f"{K}/drinking/Images_of_Waste/YOLO_imgs",
                 f"{K}/drinking/Images_of_Waste/YOLO_imgs"),
    ]
    print(f"{'bo':14s} {'anh':>6s} {'hop':>7s}   ty le vat / ANH VAO MODEL (vung 240px)")
    for s in srcs:
        if s is None: continue
        name, items = s
        if not items:
            print(f"{name:14s} khong co du lieu"); continue
        random.seed(0); random.shuffle(items)
        # do ty le: canh vat / canh vung 240px (vung dien hinh)
        fr = []
        for p, bs, mw, mh in items[:400]:
            if mw is None:                       # YOLO: da chuan hoa 0-1
                im = cv2.imread(p)
                if im is None: continue
                mh, mw = im.shape[:2]
                bs = [(b[0]*mw, b[1]*mh, b[2]*mw, b[3]*mh) for b in bs]
            k = 1280.0 / max(mw, mh)             # quy ve khung CCTV 1280
            for b in bs:
                fr.append(max(b[2]-b[0], b[3]-b[1]) * k / 240.0)
        fr.sort()
        med = fr[len(fr)//2] if fr else 0
        inband = sum(1 for x in fr if 0.06 <= x <= 0.40) / max(1, len(fr))
        print(f"{name:14s} {len(items):6d} {sum(len(x[1]) for x in items):7d}   "
              f"trung vi {100*med:5.1f}%  ·  trong dai 6-40%: {100*inband:4.0f}%")
        # xuat anh soi mat
        tiles = []
        for p, bs, mw, mh in items[:a.per]:
            im = cv2.imread(p)
            if im is None: continue
            h, w = im.shape[:2]
            sc = 1.0 if mw is None else w / mw    # bu lech anh dia vs metadata
            for b in bs:
                x1, y1, x2, y2 = ((b[0]*w, b[1]*h, b[2]*w, b[3]*h) if mw is None
                                  else (b[0]*sc, b[1]*sc, b[2]*sc, b[3]*sc))
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), max(2, w//400))
            im = cv2.resize(im, (420, 320))
            cv2.rectangle(im, (0,0), (420,20), (0,0,0), -1)
            cv2.putText(im, f"{name} · {len(bs)} hop", (4,15), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
            tiles.append(im)
        if tiles:
            while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
            g = np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)])
            cv2.imwrite(f"{out}/{name}.jpg", g)
    print(f"\n-> anh soi mat: {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
