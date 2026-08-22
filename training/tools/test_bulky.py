"""Model co thay VAT TO khong: tui rac, bai rac, thung carton.

    python3 tools/test_bulky.py

Bo train lech nang ve mau vun: 59% nhan la vat duoi 2% khung, rieng RoLID
(68% bo) thi 83%. Cac lop cong kenh rat mong — Garbage bag 31 mau, Corrugated
carton 64, Pizza box 3.

Phep thu nay dung dung nhung anh BAI RAC cua Wade da bi `--max-box-frac 0.40`
loai khoi tap train, cong voi anh TACO thuoc cac lop cong kenh. Model chua he
thay chung luc train, nen day la phep do sach ve kha nang tong quat hoa sang
vat to.

Do "co thay khong", khong do IoU: nhan cua Wade la mot hop phu ca dong rac, con
model duoc day de khoanh tung vat. Trung tam phat hien nam trong hop that la
tinh dung — dung IoU o day se ket luan sai hoan toan.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import build_detector, detect_in_zone   # noqa: E402

ROOT = "/mnt/c/Users/kangh/Documents/TrashDataset"
WADE = f"{ROOT}/data/wade/wade-ai/Trash_Detection/trash/dataset"
BULKY = {"Garbage bag", "Corrugated carton", "Pizza box", "Single-use carrier bag",
         "Paper bag", "Glass bottle", "Other carton", "Meal carton", "Drink carton",
         "Polypropylene bag", "Plastic film", "Disposable food container"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def wade_piles(min_frac=0.40):
    """Anh Wade co nhan mux CA DONG — chinh la anh bai rac da bi loai."""
    out = []
    for split in ("train", "val"):
        p = f"{WADE}/{split}_wade_ai.json"
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
            if not big:
                continue
            fp = os.path.join(WADE, split, m["file_name"].split("/")[-1])
            if os.path.exists(fp):
                out.append(("Wade-bai-rac", fp, big, float(m["width"])))
    return out


def taco_bulky():
    p = f"{ROOT}/data/taco/annotations.json"
    if not os.path.exists(p):
        return []
    a = json.load(open(p))
    cats = {c["id"]: c["name"] for c in a["categories"]}
    meta = {im["id"]: im for im in a["images"]}
    by = {}
    for an in a["annotations"]:
        if cats.get(an["category_id"]) in BULKY:
            by.setdefault(an["image_id"], []).append(an["bbox"])
    out = []
    for iid, boxes in by.items():
        m = meta[iid]
        fp = f"{ROOT}/data/taco/images/" + m["file_name"].replace("/", "_")
        if os.path.exists(fp):
            out.append(("TACO-cong-kenh", fp, boxes, float(m["width"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights",
                    default="runs/detect/runs/tiles_audited/weights/best.pt")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--upscale", type=float, default=2.0)
    ap.add_argument("--out", default="viz_bulky.jpg")
    args = ap.parse_args()

    det = build_detector("trained", weights=args.weights, conf=args.conf)
    groups = {}
    for tag, fp, boxes, mw in wade_piles() + taco_bulky():
        groups.setdefault(tag, []).append((fp, boxes, mw))
    random.seed(0)

    panels = []
    for tag, items in groups.items():
        random.shuffle(items)
        hit = tot = 0
        for fp, boxes, mw in items[:args.n]:
            img = cv2.imread(fp)
            if img is None:
                continue
            h, w = img.shape[:2]
            k = w / mw
            gts = [(b[0] * k, b[1] * k, (b[0] + b[2]) * k, (b[1] + b[3]) * k)
                   for b in boxes]
            # Thu ve co CCTV thuc te truoc khi do: anh goc 4032px thi mot tui rac
            # chiem 70% khung, khong bao gio gap o CCTV. Ep canh dai ve 1280.
            s = min(1.0, 1280.0 / max(h, w))
            if s < 1.0:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                gts = [tuple(v * s for v in g) for g in gts]
            hh, ww = img.shape[:2]
            poly = [(2, 2), (ww - 2, 2), (ww - 2, hh - 2), (2, hh - 2)]
            dets = detect_in_zone(det, img, poly, tile_px=320, overlap=0.5,
                                  upscale=args.upscale)
            tot += 1
            ok = False
            for g in gts:
                for d in dets:
                    cx = (d.box[0] + d.box[2]) / 2
                    cy = (d.box[1] + d.box[3]) / 2
                    if g[0] <= cx <= g[2] and g[1] <= cy <= g[3]:
                        ok = True
                        break
                if ok:
                    break
            hit += ok
            if len(panels) < 6 and tot <= 3:
                vis = img.copy()
                for g in gts:
                    cv2.rectangle(vis, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])),
                                  (60, 255, 60), 2)
                for d in dets:
                    x1, y1, x2, y2 = [int(v) for v in d.box]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(vis, (0, 0), (ww, 24), (0, 0, 0), -1)
                cv2.putText(vis, f"{tag}  {len(dets)} phat hien  "
                                 f"{'THAY' if ok else 'KHONG THAY'}",
                            (6, 17), FONT, .5,
                            (60, 255, 60) if ok else (0, 165, 255), 1, cv2.LINE_AA)
                panels.append(cv2.resize(vis, (600, 450)))
        print(f"{tag:16s} thay {hit}/{tot} anh ({100*hit/max(1,tot):.0f}%)")

    if panels:
        while len(panels) % 3:
            panels.append(np.zeros_like(panels[0]))
        cv2.imwrite(args.out, np.vstack([np.hstack(panels[i:i + 3])
                                         for i in range(0, len(panels), 3)]))
        print(f"luu {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
