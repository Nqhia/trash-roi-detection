"""Kiểm nhãn từng bộ dữ liệu bằng MẮT, mỗi bộ một tờ ảnh.

    python3 tools/check_labels.py --n 6

Kiểm trên NHÃN GỐC (đã áp hệ số quy đổi kích thước), không kiểm trên ô đã cắt —
vì lỗi nếu có thì nằm ở bước đọc annotation, cắt ô chỉ kế thừa lại.

Vì sao phải có bước này, và phải xem nhiều ảnh chứ không phải một:
ảnh TACO và UAVVaste trên đĩa đã bị thu nhỏ so với lúc gán nhãn (TACO 960x1280
vs metadata 1537x2049 ở 119/120 ảnh). Dùng thẳng toạ độ annotation là mọi hộp
lệch ~1,6 lần — mà một tấm mẫu may mắn vẫn có thể trông đúng.

Mỗi tờ in kèm: số hộp, cỡ vật (px), và tỉ lệ vật / cạnh dài khung. Tỉ lệ đó
mới là thứ quyết định bộ nào dạy đúng thang đo cho CCTV.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import cv2
import numpy as np

ROOT = "/mnt/c/Users/kangh/Documents/TrashDataset"
WADE = f"{ROOT}/data/wade/wade-ai/Trash_Detection/trash/dataset"
GINI = (f"{ROOT}/data/gini/spotgarbage-GINI-master/spotgarbage/"
        "non-garbage-queried-images")
# PHẢI trùng đúng danh sách trong make_tiles_data.py. Lần đầu tool này lấy mẫu
# từ CẢ 31 thư mục nên ra toàn "Sound+Waves", "pattern", phông cưới — trong khi
# bộ train chỉ dùng 16 thư mục cảnh đường. Kiểm một tập khác tập đem đi train
# thì không phải là kiểm.
GINI_KEEP = {"Indian+roads", "city+street", "clean+road", "countryside", "crowd",
             "earth+dust", "indian+railway+tracks", "rural+area", "suburb",
             "buildings", "vehicles", "people", "chaos", "chaos+cable",
             "environment", "Places"}
# Hộp lớn hơn ngần này phần cạnh dài = nhãn ở mức CẢ ĐỐNG chứ không phải từng
# vật. Không sai, nhưng dạy model bắn ra hộp to — mà FP quan sát được ở bản
# trước đúng là những hộp to phủ cả tấm vách gỗ.
REGION_FRAC = 0.40

SOURCES = [
    ("TACO", f"{ROOT}/data/taco/annotations.json", f"{ROOT}/data/taco/images",
     lambda fn: fn.replace("/", "_")),
    ("RoLID-val", f"{ROOT}/data/rolid/RoLID-11K/validation.json",
     f"{ROOT}/data/rolid/RoLID-11K/val_images", lambda fn: fn.split("/")[-1]),
    ("RoLID-train", f"{ROOT}/data/rolid/RoLID-11K/training.json",
     f"{ROOT}/data/rolid/RoLID-11K/train_images", lambda fn: fn.split("/")[-1]),
    ("UAVVaste", f"{ROOT}/data/uavvaste/annotations.json",
     f"{ROOT}/data/uavvaste/images", lambda fn: fn.split("/")[-1]),
    ("Wade-train", f"{WADE}/train_wade_ai.json", f"{WADE}/train",
     lambda fn: fn.split("/")[-1]),
    ("Wade-val", f"{WADE}/val_wade_ai.json", f"{WADE}/val",
     lambda fn: fn.split("/")[-1]),
]
FONT = cv2.FONT_HERSHEY_SIMPLEX
CELL = 560


def panel(img, boxes, title, sub):
    """Ảnh + hộp, thu về ô vuông CELL, có nhãn ở trên."""
    h, w = img.shape[:2]
    s = CELL / max(h, w)
    im = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    for x1, y1, x2, y2 in boxes:
        a, b = int(x1 * s), int(y1 * s)
        c, d = int(x2 * s), int(y2 * s)
        cv2.rectangle(im, (a, b), (c, d), (0, 255, 0), 2)
        # Vật rất nhỏ thì thêm vòng tròn cho nhìn thấy được ở cỡ này
        if max(c - a, d - b) < 14:
            cv2.circle(im, ((a + c) // 2, (b + d) // 2), 16, (0, 255, 255), 1)
    out = np.zeros((CELL + 46, CELL, 3), np.uint8)
    y0 = (CELL - im.shape[0]) // 2
    x0 = (CELL - im.shape[1]) // 2
    out[46 + y0:46 + y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    cv2.putText(out, title, (8, 19), FONT, .52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, sub, (8, 38), FONT, .44, (0, 220, 255), 1, cv2.LINE_AA)
    return out


def sheet(panels, cols=3):
    while len(panels) % cols:
        panels.append(np.zeros_like(panels[0]))
    return np.vstack([np.hstack(panels[i:i + cols])
                      for i in range(0, len(panels), cols)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6, help="số ảnh mẫu mỗi bộ")
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for name, annf, imgd, ren in SOURCES:
        if not os.path.exists(annf):
            print(f"{name}: KHÔNG CÓ {annf}")
            continue
        ann = json.load(open(annf))
        meta = {im["id"]: im for im in ann["images"]}
        by: dict = {}
        for a in ann["annotations"]:
            by.setdefault(a["image_id"], []).append(a["bbox"])
        ids = [i for i in by if os.path.exists(os.path.join(imgd, ren(meta[i]["file_name"])))]
        random.seed(args.seed)
        random.shuffle(ids)

        # Thống kê trên TOÀN BỘ nhãn, không chỉ trên mấy tấm lấy mẫu — mẫu 6 tấm
        # không đủ để phát hiện một nhóm nhãn lệch chuẩn chiếm 10% bộ.
        all_fr = []
        for iid in ids:
            m = meta[iid]
            for bx, byy, bw, bh in by[iid]:
                all_fr.append(max(bw, bh) / max(m["width"], m["height"]))
        n_region = sum(1 for f in all_fr if f >= REGION_FRAC)

        panels, sizes, fracs, n_rescaled = [], [], [], 0
        for iid in ids:
            if len(panels) >= args.n:
                break
            img = cv2.imread(os.path.join(imgd, ren(meta[iid]["file_name"])))
            if img is None:
                continue
            h, w = img.shape[:2]
            # HỆ SỐ QUY ĐỔI: ảnh trên đĩa có thể đã bị thu nhỏ so với metadata.
            k = w / float(meta[iid]["width"])
            if abs(k - 1.0) > 0.01:
                n_rescaled += 1
            boxes = [(bx * k, byy * k, (bx + bw) * k, (byy + bh) * k)
                     for bx, byy, bw, bh in by[iid] if bw * k >= 1 and bh * k >= 1]
            if not boxes:
                continue
            ss = [max(x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes]
            sizes += ss
            fracs += [s / max(w, h) for s in ss]
            med = sorted(ss)[len(ss) // 2]
            panels.append(panel(
                img, boxes, f"{name}  {len(boxes)} hop  {w}x{h}",
                f"vat trung vi {med:.0f}px = {100*med/max(w,h):.1f}% canh dai"
                + (f"  (da quy doi x{k:.2f})" if abs(k - 1) > 0.01 else "")))

        if not panels:
            print(f"{name}: không dựng được tấm nào")
            continue
        p = os.path.join(args.out, f"labels_{name}.jpg")
        cv2.imwrite(p, sheet(panels))
        sizes.sort()
        fr = sorted(fracs)
        print(f"{name}: {len(ids)} ảnh có nhãn · mẫu {len(panels)} tấm, "
              f"{n_rescaled} tấm phải quy đổi tỉ lệ")
        print(f"   cỡ vật px: trung vị {sizes[len(sizes)//2]:.0f} "
              f"(p25 {sizes[len(sizes)//4]:.0f}, p75 {sizes[3*len(sizes)//4]:.0f})   "
              f"vật/cạnh dài: trung vị {100*fr[len(fr)//2]:.1f}%")
        print(f"   nhãn mức CẢ ĐỐNG (>={REGION_FRAC:.0%} cạnh dài): "
              f"{n_region}/{len(all_fr)} = {100*n_region/max(1,len(all_fr)):.1f}%")
        print(f"   -> {p}")

    # Mẫu âm GINI: không có nhãn, chỉ cần xem có đúng là cảnh KHÔNG rác không.
    if os.path.isdir(GINI):
        fs = []
        for q in sorted(os.listdir(GINI)):
            qd = os.path.join(GINI, q)
            if q in GINI_KEEP and os.path.isdir(qd):
                fs += [(q, os.path.join(qd, f)) for f in os.listdir(qd)]
        random.seed(args.seed)
        random.shuffle(fs)
        panels = []
        for q, path in fs:
            if len(panels) >= args.n:
                break
            img = cv2.imread(path)
            if img is not None:
                panels.append(panel(img, [], f"GINI mau am  truy van: {q}",
                                    "phai KHONG co rac"))
        if panels:
            p = os.path.join(args.out, "labels_GINI-neg.jpg")
            cv2.imwrite(p, sheet(panels))
            print(f"GINI mẫu âm: {len(fs)} ảnh -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
