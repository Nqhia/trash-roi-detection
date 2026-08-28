"""Bộ test RỘNG cho nhiều model — đo ở ĐÚNG chế độ pipeline thật sự chạy.

    python3 tools/bench_suite.py --models trash_yolo11n trash26sp2_v2 --n 120

VÌ SAO CẦN CÁI NÀY
------------------
Bộ test cũ là 6 khung / 21 vật / MỘT camera, và tôi đã dùng chính nó để chỉnh
vùng 320px, conf 0,20, ngưỡng leo thang. Chỉnh chục lần rồi đọc kết quả trên
cùng bộ đó thì con số 86% có phần là KHỚP VÀO BỘ TEST chứ không phải năng lực.
Model mới chưa từng được chỉnh gì trên đó, nên so như vậy lệch hẳn về phía bản
cũ.

Bộ này: ~12k ảnh mà model v2 chưa hề thấy (bốn bộ bị loại khỏi tập train), từ
các bối cảnh khác nhau — chai/lon trên sàn, chai ngoài trời, túi ni lông, rác
tái chế — cộng phần val của hai bộ đã train để đối chiếu.

ĐO Ở CHẾ ĐỘ NÀO
---------------
Pipeline không chạy model trên toàn khung: nó cắt vùng 160-320px quanh chỗ đổi
rồi phóng lên 640px. Chấm trên toàn khung là chấm sai chế độ. Ở đây mô phỏng
đúng: với mỗi vật, cắt một vùng có cạnh = cạnh_vật / tỉ_lệ rồi phóng lên 640,
với tỉ_lệ lấy ngẫu nhiên trong dải mà tầng xác nhận thực sự tạo ra (6-40%).

CÁCH CHẤM
---------
`centre_in`: tâm hộp nằm trong hộp thật -> tính là bắt được. Không dùng IoU vì
pipeline chỉ cần biết "có vật ở ô này không", không cần hộp khớp khít.

So hai model PHẢI ở cùng mức HỘP THỪA, không phải ở cùng ngưỡng conf — hai model
hiệu chỉnh khác nhau, và so ở cùng conf đã một lần cho tôi kết luận ngược.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
D = os.path.join(ROOT, "data")
K = os.path.join(D, "kaggle")
sys.path.insert(0, HERE)


def yolo_items(imgd, labd):
    """-> [(duong_dan_anh, [hop chuan hoa 0-1])]"""
    out = []
    if not os.path.isdir(imgd):
        return out
    for ip in sorted(glob.glob(os.path.join(imgd, "*.jpg")) +
                     glob.glob(os.path.join(imgd, "*.png"))):
        lp = os.path.join(labd, os.path.splitext(os.path.basename(ip))[0] + ".txt")
        if not os.path.exists(lp):
            continue
        bs = []
        for line in open(lp):
            p = line.split()
            if len(p) >= 5:
                cx, cy, w, h = (float(x) for x in p[1:5])
                bs.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        if bs:
            out.append((ip, bs))
    return out


def coco_items(annf, imgd, ren):
    """Toạ độ hộp COCO nằm trong hệ METADATA, ảnh trên đĩa thường nhỏ hơn —
    nên chuẩn hoá theo metadata rồi mới nhân với kích thước ảnh thật."""
    out = []
    if not os.path.exists(annf):
        return out
    a = json.load(open(annf))
    meta = {i["id"]: i for i in a["images"]}
    by = {}
    for an in a["annotations"]:
        by.setdefault(an["image_id"], []).append(an["bbox"])
    for iid, raw in by.items():
        p = os.path.join(imgd, ren(meta[iid]["file_name"]))
        if not os.path.exists(p):
            continue
        mw = float(meta[iid]["width"])
        mh = float(meta[iid]["height"])
        bs = [(b[0] / mw, b[1] / mh, (b[0] + b[2]) / mw, (b[1] + b[3]) / mh)
              for b in raw]
        out.append((p, bs))
    return out


def sources():
    W = os.path.join(D, "wade", "wade-ai", "Trash_Detection", "trash", "dataset")
    PB = os.path.join(K, "Plastic Bottle Image Dataset")
    S = [
        # ---- CHƯA MODEL NÀO THẤY (bị loại khỏi tập train v2) ----
        ("drinking", yolo_items(os.path.join(K, "drinking", "Images_of_Waste", "YOLO_imgs"),
                                os.path.join(K, "drinking", "Images_of_Waste", "YOLO_imgs")), True),
        ("bottles_wild", yolo_items(os.path.join(PB, "valid", "images"),
                                    os.path.join(PB, "valid", "labels"))
         + yolo_items(os.path.join(PB, "test", "images"),
                      os.path.join(PB, "test", "labels")), True),
        ("bagbottle", yolo_items(os.path.join(K, "bagbottle", "dataset", "images", "train"),
                                 os.path.join(K, "bagbottle", "dataset", "labels", "train")), True),
        ("recyc", yolo_items(os.path.join(K, "recyc", "Final Data", "images", "val"),
                             os.path.join(K, "recyc", "Final Data", "labels", "val")), True),
        # ---- ĐÃ TRAIN (phần val), để đối chiếu ----
        ("TACO_val", coco_items(os.path.join(D, "taco", "annotations.json"),
                                os.path.join(D, "taco", "images"),
                                lambda x: x.replace("/", "_")), False),
        ("Wade_val", coco_items(os.path.join(W, "val_wade_ai.json"),
                                os.path.join(W, "val"),
                                lambda x: x.split("/")[-1]), False),
    ]
    return [(n, it, h) for n, it, h in S if it]


def centre_in(b, g):
    cx = (b[0] + b[2]) / 2
    cy = (b[1] + b[3]) / 2
    return g[0] <= cx <= g[2] and g[1] <= cy <= g[3]


def bench_one(model, items, n, side, conf, rng, lo=0.06, hi=0.40):
    """Cắt vùng quanh từng vật ĐÚNG cách tầng xác nhận làm, rồi chấm."""
    hit = tot = extra = 0
    for ip, bs in items[:n]:
        img = cv2.imread(ip)
        if img is None:
            continue
        H, Wd = img.shape[:2]
        px = [(b[0] * Wd, b[1] * H, b[2] * Wd, b[3] * H) for b in bs]
        for g in px:
            gw = g[2] - g[0]
            gh = g[3] - g[1]
            if gw < 3 or gh < 3:
                continue
            frac = float(rng.uniform(lo, hi))
            need = float(np.clip(max(gw, gh) / frac, 24, min(Wd, H)))
            cx = (g[0] + g[2]) / 2
            cy = (g[1] + g[3]) / 2
            x0 = int(np.clip(cx - need / 2, 0, max(0, Wd - need)))
            y0 = int(np.clip(cy - need / 2, 0, max(0, H - need)))
            crop = img[y0:int(y0 + need), x0:int(x0 + need)]
            if crop.size == 0 or min(crop.shape[:2]) < 16:
                continue
            inp = cv2.resize(crop, (side, side), interpolation=cv2.INTER_CUBIC)
            sx = side / crop.shape[1]
            sy = side / crop.shape[0]
            gts = []
            for q in px:
                a = ((q[0] - x0) * sx, (q[1] - y0) * sy,
                     (q[2] - x0) * sx, (q[3] - y0) * sy)
                if a[2] > 0 and a[3] > 0 and a[0] < side and a[1] < side:
                    gts.append(a)
            tgt = ((g[0] - x0) * sx, (g[1] - y0) * sy,
                   (g[2] - x0) * sx, (g[3] - y0) * sy)
            r = model.predict([inp], conf=conf, verbose=False)[0]
            boxes = [tuple(b.xyxy[0].tolist()) for b in r.boxes]
            tot += 1
            if any(centre_in(b, tgt) for b in boxes):
                hit += 1
            extra += sum(1 for b in boxes if not any(centre_in(b, q) for q in gts))
    return hit, tot, extra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--confs", nargs="+", type=float,
                    default=[0.03, 0.05, 0.10, 0.20, 0.30])
    ap.add_argument("--n", type=int, default=120, help="số ảnh mỗi nguồn")
    ap.add_argument("--side", type=int, default=640)
    ap.add_argument("--per-source", action="store_true",
                    help="in thêm bảng theo từng nguồn")
    a = ap.parse_args()

    from ultralytics import YOLO
    srcs = sources()
    print("NGUỒN (held-out = chưa model nào trong so sánh được train trên):")
    for n, it, held in srcs:
        random.Random(0).shuffle(it)
        print(f"  {n:14s} {len(it):6d} ảnh  "
              f"{'HELD-OUT' if held else 'đã train (val)'}")
    print()
    for mname in a.models:
        wt = os.path.abspath(os.path.join(HERE, "..", "models", mname + ".pt"))
        if not os.path.exists(wt):
            print(f"{mname}: không thấy {wt}\n")
            continue
        m = YOLO(wt)
        print(f"### {mname}")
        print(f"  {'conf':>5s} | {'HELD-OUT':>19s} | {'đã train':>19s} | {'hộp thừa':>8s}")
        for c in a.confs:
            rng = np.random.default_rng(0)
            H = T = E = H2 = T2 = 0
            per = []
            for n, it, held in srcs:
                h, t, e = bench_one(m, it, a.n, a.side, c, rng)
                E += e
                per.append((n, h, t, e))
                if held:
                    H += h
                    T += t
                else:
                    H2 += h
                    T2 += t
            print(f"  {c:5.2f} | {H:5d}/{T:<6d} {100*H/max(1,T):5.1f}% | "
                  f"{H2:5d}/{T2:<6d} {100*H2/max(1,T2):5.1f}% | {E:8d}")
            if a.per_source:
                for n, h, t, e in per:
                    print(f"          {n:14s} {h:4d}/{t:<5d} {100*h/max(1,t):5.1f}%  "
                          f"thừa {e}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
