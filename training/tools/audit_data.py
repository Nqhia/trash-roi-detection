"""Soát bộ dữ liệu TRƯỚC khi train. Chạy vài phút, tiết kiệm vài giờ GPU.

    python3 tools/audit_data.py --data data/tiles

Sáu nhóm kiểm, xếp theo mức thiệt hại nếu bỏ sót:

 1. RÒ RỈ TRAIN/VAL. RoLID và UAVVaste là các KHUNG LIÊN TIẾP của video —
    frame1822/1823/1824 gần như trùng nhau nhưng mang image_id khác nhau, nên
    phép chia ngẫu nhiên theo ảnh vẫn ném chúng sang hai bên. Val khi đó đo lại
    chính thứ model vừa học thuộc, và mAP đẹp lên một cách vô nghĩa. Đây là lỗi
    tốn kém nhất vì nó không làm gì hỏng cả — chỉ làm mọi con số nói dối.
 2. Ô RỖNG CÓ THẬT RỖNG KHÔNG. Ô rỗng dùng làm mẫu âm với giả định "không có
    nhãn ở đây nghĩa là không có rác". Giả định đó chỉ đúng nếu bộ gốc gán nhãn
    ĐẦY ĐỦ. Nếu người gán nhãn bỏ sót, ta đang dạy model rằng rác không phải rác.
 3. Toàn vẹn: nhãn mồ côi, hộp suy biến, toạ độ ngoài [0,1].
 4. Phân bố cỡ vật so với mục tiêu CCTV (vỏ chai 19-27px ở khung gốc, tức
    ~40-55px trong ô đã phóng 2x).
 5. Cân bằng nguồn: một nguồn chiếm quá nửa thì model học giọng của nguồn đó.
 6. Ảnh không đọc được.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

import cv2
import numpy as np

ROOT = "/mnt/c/Users/kangh/Documents/TrashDataset"
WADE = f"{ROOT}/data/wade/wade-ai/Trash_Detection/trash/dataset"
SRC = {
    "TACO":     (f"{ROOT}/data/taco/annotations.json", None),
    "RoLID":    (f"{ROOT}/data/rolid/RoLID-11K/validation.json", "rolid"),
    "RoLIDtr":  (f"{ROOT}/data/rolid/RoLID-11K/training.json", "rolid"),
    "UAVVaste": (f"{ROOT}/data/uavvaste/annotations.json", "uav"),
    "WadeTr":   (f"{WADE}/train_wade_ai.json", None),
    "WadeVa":   (f"{WADE}/val_wade_ai.json", None),
}


def clip_of(kind, fn):
    """Tên CLIP để nhóm các khung cùng một video lại với nhau."""
    base = fn.split("/")[-1]
    if kind == "rolid":
        return re.sub(r"_frame\d+\.\w+$", "", base)
    if kind == "uav":
        m = re.match(r"(BATCH_[A-Za-z0-9]+)_", base)
        if m:
            return m.group(1)
    # Ảnh rời: dùng NGUYÊN đường dẫn. Bản đầu cắt lấy tên trần nên gộp nhầm
    # batch_1/000001.jpg với batch_2/000001.jpg của TACO, rồi báo 50 clip rò rỉ
    # trong khi dữ liệu không hề rò rỉ. Phải khớp đúng hàm của make_tiles_data.py
    # — hai định nghĩa "clip" khác nhau thì công cụ kiểm chỉ đang kiểm chính nó.
    return fn


def tile_key(name):
    """RoLIDtr_1234_t7.jpg -> (RoLIDtr, 1234)"""
    m = re.match(r"([A-Za-z]+)_(.+?)_(?:t\d+|n\d+|full)$", os.path.splitext(name)[0])
    return (m.group(1), m.group(2)) if m else (None, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/tiles")
    ap.add_argument("--sample-neg", type=int, default=12)
    args = ap.parse_args()
    D = args.data
    bad = []

    splits = {}
    for sp in ("train", "val"):
        imgs = sorted(os.listdir(os.path.join(D, sp, "images")))
        labs = set(os.listdir(os.path.join(D, sp, "labels")))
        splits[sp] = (imgs, labs)

    print("=" * 70)
    print("3 · TOAN VEN")
    for sp, (imgs, labs) in splits.items():
        stems = {os.path.splitext(f)[0] for f in imgs}
        orphan = {os.path.splitext(l)[0] for l in labs} - stems
        n_pos = sum(1 for f in imgs if os.path.splitext(f)[0] + ".txt" in labs)
        n_box = n_degen = n_oob = 0
        for l in labs:
            for line in open(os.path.join(D, sp, "labels", l)):
                p = line.split()
                if len(p) != 5:
                    n_degen += 1
                    continue
                cx, cy, w, h = [float(v) for v in p[1:]]
                n_box += 1
                if w <= 0 or h <= 0:
                    n_degen += 1
                if (cx - w / 2 < -0.001 or cy - h / 2 < -0.001
                        or cx + w / 2 > 1.001 or cy + h / 2 > 1.001):
                    n_oob += 1
        print(f"  {sp}: {len(imgs)} anh ({n_pos} co nhan, {len(imgs)-n_pos} o rong), "
              f"{n_box} hop")
        print(f"       nhan mo coi {len(orphan)} · hop suy bien {n_degen} · "
              f"toa do ngoai khung {n_oob}")
        if orphan or n_degen or n_oob:
            bad.append(f"{sp}: mo coi={len(orphan)} suy bien={n_degen} ngoai={n_oob}")

    print()
    print("=" * 70)
    print("5 · CAN BANG NGUON (anh co nhan)")
    per = Counter()
    for sp, (imgs, labs) in splits.items():
        for f in imgs:
            if os.path.splitext(f)[0] + ".txt" in labs:
                per[tile_key(f)[0]] += 1
    tot = sum(per.values())
    for k, v in per.most_common():
        flag = "   <-- qua nua bo" if v > tot * 0.5 else ""
        print(f"  {str(k):10s} {v:6d}  {100*v/tot:5.1f}%{flag}")

    print()
    print("=" * 70)
    print("1 · RO RI TRAIN/VAL (khung lien tiep cung mot video)")
    meta = {}
    for pre, (annf, kind) in SRC.items():
        if os.path.exists(annf):
            a = json.load(open(annf))
            meta[pre] = ({str(im["id"]): im["file_name"] for im in a["images"]}, kind)
    clips = defaultdict(set)
    for sp, (imgs, _labs) in splits.items():
        for f in imgs:
            pre, iid = tile_key(f)
            if pre not in meta:
                continue
            fmap, kind = meta[pre]
            fn = fmap.get(iid)
            if fn is not None:
                clips[(pre, clip_of(kind, fn))].add(sp)
    both = [c for c, s in clips.items() if len(s) == 2]
    bysrc = Counter(c[0] for c in both)
    print(f"  tong clip: {len(clips)}   clip nam CA HAI ben: {len(both)}")
    for k, v in bysrc.most_common():
        n_all = sum(1 for c in clips if c[0] == k)
        print(f"    {k:10s} {v}/{n_all} clip bi ro ri  ({100*v/max(1,n_all):.0f}%)")
    if both:
        bad.append(f"ro ri train/val: {len(both)} clip nam ca hai ben")
        print("  vi du:", [c[1][:44] for c in both[:3]])

    print()
    print("=" * 70)
    print("4 · CO VAT trong o da phong to (muc tieu CCTV ~40-55px)")
    px = defaultdict(list)
    for sp, (_imgs, labs) in splits.items():
        for l in labs:
            pre = tile_key(l.replace(".txt", ".jpg"))[0]
            for line in open(os.path.join(D, sp, "labels", l)):
                p = line.split()
                if len(p) == 5:
                    px[pre].append(max(float(p[3]), float(p[4])) * 640)
    for k in sorted(px, key=lambda z: str(z)):
        v = sorted(px[k])
        if not v:
            continue
        small = 100 * sum(1 for s in v if s < 12) / len(v)
        print(f"  {str(k):10s} n={len(v):6d}  p25 {v[len(v)//4]:5.0f}  "
              f"trung vi {v[len(v)//2]:5.0f}  p75 {v[3*len(v)//4]:5.0f}px"
              f"   duoi 12px: {small:.1f}%")

    print()
    print("=" * 70)
    print("2 · O RONG — lay mau de soi bang mat")
    imgs, labs = splits["train"]
    negs = [f for f in imgs if os.path.splitext(f)[0] + ".txt" not in labs]
    random.seed(0)
    random.shuffle(negs)
    pick, seen = [], Counter()
    for f in negs:
        pre = tile_key(f)[0]
        if seen[pre] >= max(1, args.sample_neg // 4):
            continue
        seen[pre] += 1
        pick.append(f)
        if len(pick) >= args.sample_neg:
            break
    panels = []
    for f in pick:
        im = cv2.imread(os.path.join(D, "train", "images", f))
        if im is None:
            bad.append(f"anh hong: {f}")
            continue
        cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(im, f"{tile_key(f)[0]} - phai KHONG co rac", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 165, 255), 1, cv2.LINE_AA)
        panels.append(cv2.resize(im, (420, 420)))
    if panels:
        while len(panels) % 4:
            panels.append(np.zeros_like(panels[0]))
        sheet = np.vstack([np.hstack(panels[i:i + 4])
                           for i in range(0, len(panels), 4)])
        cv2.imwrite("audit_negatives.jpg", sheet)
        print(f"  {len(negs)} o rong · luu {len(panels)} tam -> audit_negatives.jpg")
        print("  PHAI XEM TAM NAY: co rac chua gan nhan trong do thi mau am dang")
        print("  day model rang rac khong phai rac.")

    print()
    print("=" * 70)
    if bad:
        print("CAN SUA TRUOC KHI TRAIN:")
        for b in bad[:10]:
            print("  -", b)
        return 1
    print("Khong thay loi chan. Van phai xem audit_negatives.jpg bang mat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
