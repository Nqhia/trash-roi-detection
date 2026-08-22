"""Bộ dữ liệu đang phủ được NHỮNG LOẠI RÁC NÀO, và ở cỡ nào.

    python3 tools/check_coverage.py

Bài toán không cần phân loại, nhưng vẫn cần dữ liệu ĐỦ ĐA DẠNG: model chỉ nhận
ra được thứ nó từng thấy. Túi rác to, chai bia, thùng carton, hộp xốp — nếu bộ
train chỉ toàn mẩu giấy vụn ven đường thì nó sẽ bỏ sót đúng những thứ đó.

Ba câu hỏi:
 1. Có những lớp nào, mỗi lớp bao nhiêu mẫu (TACO có nhãn lớp chi tiết nhất).
 2. Phân bố cỡ vật ở KHUNG GỐC — vật to (túi rác) có mặt không, hay đã bị lọc
    `--max-box-frac 0.40` cắt hết cùng với nhãn mức "cả đống".
 3. Vật to đến từ ảnh CHỤP CẬN hay ảnh có bối cảnh rộng — chỉ loại thứ hai mới
    dạy đúng thứ CCTV nhìn thấy.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

ROOT = "/mnt/c/Users/kangh/Documents/TrashDataset"
WADE = f"{ROOT}/data/wade/wade-ai/Trash_Detection/trash/dataset"
SOURCES = [
    ("TACO", f"{ROOT}/data/taco/annotations.json"),
    ("RoLID-val", f"{ROOT}/data/rolid/RoLID-11K/validation.json"),
    ("RoLID-train", f"{ROOT}/data/rolid/RoLID-11K/training.json"),
    ("UAVVaste", f"{ROOT}/data/uavvaste/annotations.json"),
    ("Wade-train", f"{WADE}/train_wade_ai.json"),
    ("Wade-val", f"{WADE}/val_wade_ai.json"),
]
# Ngưỡng theo cạnh dài hộp / cạnh dài ảnh. "to" ở đây nghĩa là to SO VỚI KHUNG,
# tức thứ mà ở CCTV sẽ hiện ra như một túi rác chứ không phải một mẩu giấy.
BANDS = [(0.0, 0.02, "rat nho  <2%"), (0.02, 0.05, "nho    2-5%"),
         (0.05, 0.12, "vua    5-12%"), (0.12, 0.40, "to    12-40%"),
         (0.40, 9.9, "ca dong >40%")]


def main() -> int:
    print("=" * 74)
    print("1 · LOP RAC (chi TACO co nhan lop chi tiet)")
    p = f"{ROOT}/data/taco/annotations.json"
    if os.path.exists(p):
        a = json.load(open(p))
        cats = {c["id"]: c for c in a["categories"]}
        sup = Counter()
        fine = Counter()
        for an in a["annotations"]:
            c = cats.get(an["category_id"], {})
            sup[c.get("supercategory", "?")] += 1
            fine[c.get("name", "?")] += 1
        print(f"  {len(cats)} lop chi tiet, {len(sup)} sieu lop, "
              f"{len(a['annotations'])} nhan")
        print("  sieu lop:")
        for k, v in sup.most_common():
            print(f"    {k:28s} {v:5d}")
        print("  10 lop chi tiet nhieu nhat:")
        for k, v in fine.most_common(10):
            print(f"    {k:28s} {v:5d}")
        print("  cac lop VAT TO dang quan tam:")
        for k, v in sorted(fine.items()):
            low = k.lower()
            if any(t in low for t in ("bag", "carton", "box", "bottle", "can",
                                      "cup", "container", "crate")):
                print(f"    {k:28s} {v:5d}")

    print()
    print("=" * 74)
    print("2 · CO VAT SO VOI KHUNG (truoc khi cat o, truoc moi bo loc)")
    print(f"  {'nguon':13s}" + "".join(f"{b[2]:>14s}" for b in BANDS))
    tot_band = Counter()
    for name, annf in SOURCES:
        if not os.path.exists(annf):
            continue
        a = json.load(open(annf))
        meta = {im["id"]: im for im in a["images"]}
        band = Counter()
        for an in a["annotations"]:
            m = meta.get(an["image_id"])
            if not m:
                continue
            f = max(an["bbox"][2], an["bbox"][3]) / max(m["width"], m["height"])
            for lo, hi, lbl in BANDS:
                if lo <= f < hi:
                    band[lbl] += 1
                    tot_band[lbl] += 1
                    break
        n = sum(band.values())
        row = "".join(f"{100*band[b[2]]/max(1,n):13.1f}%" for b in BANDS)
        print(f"  {name:13s}{row}   (n={n})")
    n = sum(tot_band.values())
    row = "".join(f"{100*tot_band[b[2]]/max(1,n):13.1f}%" for b in BANDS)
    print(f"  {'TAT CA':13s}{row}   (n={n})")

    print()
    print("=" * 74)
    print("3 · SAU KHI LOC max-box-frac 0.40 thi con lai gi")
    keep = sum(v for k, v in tot_band.items() if k != "ca dong >40%")
    drop = tot_band["ca dong >40%"]
    print(f"  giu {keep} nhan · bo {drop} nhan mux ca dong "
          f"({100*drop/max(1,keep+drop):.1f}%)")
    big = tot_band["to    12-40%"]
    print(f"  trong so giu lai, vat TO (12-40% khung): {big} = "
          f"{100*big/max(1,keep):.1f}%")
    print()
    print("  Doc: vat 12-40% khung o CCTV 1920px tuong duong 230-770px — to hon")
    print("  ca mot tui rac that. Vat 2-5% (~40-95px) moi la co cua tui rac/")
    print("  thung carton nhin tu camera treo cao. Xem hai cot do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
