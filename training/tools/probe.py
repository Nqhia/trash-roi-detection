"""THÍ NGHIỆM QUYẾT ĐỊNH: zero-shot có thấy được rác ở cỡ px của CCTV không.

    python3 tools/probe.py --backend yoloworld --n 60
    python3 tools/probe.py --backend owlv2 --n 40 --sizes 24,32,48

Cách đo, cố ý làm giống hệt phép đo của hướng 1 để so được:

  1. lấy annotation rác THẬT (TACO / RoLID / UAVVaste)
  2. thu-phóng ảnh gốc sao cho cạnh dài của vật đúng bằng N px
  3. cắt một cửa sổ 640x480 quanh vật -> đây là "khung CCTV" giả lập
  4. chạy cắt ô + phóng to + detector trên cả cửa sổ đó
  5. tính trúng nếu có phát hiện nào IoU >= 0.3 với vật thật

Cộng thêm phép đo âm: chạy trên ảnh CCTV SẠCH (ABODA + khung site) và đếm số
phát hiện. Đó là FP — con số hướng 1 đo được là 15,8% trên cảnh chưa từng thấy.

Mốc phải vượt: hướng 1 bắt được vật **20-24px** với cell 48.
Không đạt cỡ đó thì hướng này chết, và chết rẻ.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import build_detector, detect_in_zone       # noqa: E402
from tools.make_tiles_data import clip_of, split_of            # noqa: E402

ROOT = "/mnt/c/Users/kangh/Documents/TrashDataset"
SOURCES = [
    ("TACO", f"{ROOT}/data/taco/annotations.json", f"{ROOT}/data/taco/images",
     lambda fn: fn.replace("/", "_")),
    ("RoLID", f"{ROOT}/data/rolid/RoLID-11K/validation.json",
     f"{ROOT}/data/rolid/RoLID-11K/val_images", lambda fn: fn.split("/")[-1]),
    ("UAVVaste", f"{ROOT}/data/uavvaste/annotations.json",
     f"{ROOT}/data/uavvaste/images", lambda fn: fn.split("/")[-1]),
]
WIN_W, WIN_H = 640, 480          # cửa sổ "khung CCTV" giả lập
ZONE = [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]]


def coco_items(annf, imgd, ren):
    """[(đường dẫn, [bbox xywh], bề ngang METADATA), ...].

    Phải trả kèm bề ngang metadata: ảnh TACO và UAVVaste trên đĩa **đã bị thu
    nhỏ** so với lúc gán nhãn (TACO 960x1280 vs meta 1537x2049 ở 119/120 ảnh;
    UAVVaste 1600x900 vs 3840x2160). Toạ độ annotation nằm trong hệ metadata,
    dùng thẳng lên ảnh đã thu nhỏ thì mọi hộp lệch ~1,6 lần và recall ra 0%.
    """
    ann = json.load(open(annf))
    by = {}
    for a in ann["annotations"]:
        by.setdefault(a["image_id"], []).append(a["bbox"])
    meta = {im["id"]: im for im in ann["images"]}
    out = []
    for i, boxes in by.items():
        p = os.path.join(imgd, ren(meta[i]["file_name"]))
        if os.path.exists(p):
            out.append((p, boxes, float(meta[i]["width"]), meta[i]["file_name"]))
    return out


def make_scene(img, bbox, target_px):
    """Ảnh gốc + bbox -> cửa sổ 640x480 trong đó vật đúng `target_px`.

    Thu-phóng TRƯỚC khi cắt thì với ảnh 4000px phải resize cả ảnh cho mỗi vật;
    ở đây cắt cửa sổ tương ứng ở ảnh gốc rồi mới resize, chi phí không phụ
    thuộc kích thước ảnh gốc.
    """
    x, y, bw, bh = bbox
    if bw < 2 or bh < 2:
        return None, None
    s = target_px / max(bw, bh)                 # hệ số thu-phóng cần thiết
    win_w, win_h = WIN_W / s, WIN_H / s         # cửa sổ tương ứng ở ảnh gốc
    h, w = img.shape[:2]
    if win_w > w or win_h > h:                  # ảnh gốc quá nhỏ để cắt cửa sổ
        return None, None
    cx, cy = x + bw / 2, y + bh / 2
    x1 = int(round(min(max(0, cx - win_w / 2), w - win_w)))
    y1 = int(round(min(max(0, cy - win_h / 2), h - win_h)))
    crop = img[y1:y1 + int(round(win_h)), x1:x1 + int(round(win_w))]
    if crop.size == 0:
        return None, None
    scene = cv2.resize(crop, (WIN_W, WIN_H),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    gt = ((x - x1) * WIN_W / win_w, (y - y1) * WIN_H / win_h,
          (x - x1 + bw) * WIN_W / win_w, (y - y1 + bh) * WIN_H / win_h)
    if gt[0] < 0 or gt[1] < 0 or gt[2] > WIN_W or gt[3] > WIN_H:
        return None, None
    return scene, gt


def iou(a, b):
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / (ua + 1e-9)


def clean_frames(n: int) -> list:
    """Khung CCTV SẠCH để đo FP: ABODA (ngoài trời) + khung site (trong nhà)."""
    out = []
    for v in ("video1", "video3", "video6"):
        cap = cv2.VideoCapture(f"{ROOT}/data/aboda/{v}.avi")
        for k in range(200):
            ok, f = cap.read()
            if not ok:
                break
            if k % 60 == 0:
                out.append(cv2.resize(f, (WIN_W, WIN_H)))
        cap.release()
    d = f"{ROOT}/data/site/clean2"
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d))[:12]:
            im = cv2.imread(os.path.join(d, fn))
            if im is not None:
                out.append(cv2.resize(im, (WIN_W, WIN_H)))
    random.shuffle(out)
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="yoloworld",
                    choices=["yoloworld", "owlv2", "coco", "trained"])
    ap.add_argument("--weights")
    ap.add_argument("--n", type=int, default=60, help="số vật mỗi cỡ")
    ap.add_argument("--sizes", default="16,24,32,48,64")
    ap.add_argument("--tile-px", type=int, default=320)
    ap.add_argument("--upscale", type=float, default=3.0)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--split", default="val", choices=["all", "train", "val"],
                    help="lấy mẫu từ phía nào của bộ train (mặc định val — "
                         "'all' cho con số HỌC THUỘC, không dùng để báo cáo)")
    ap.add_argument("--clean", type=int, default=40, help="số khung sạch để đo FP")
    ap.add_argument("--no-tile", action="store_true",
                    help="chạy thẳng cả khung, không cắt ô — để thấy tiler đáng giá bao nhiêu")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    random.seed(0)
    items = []
    for name, annf, imgd, ren in SOURCES:
        if not os.path.exists(annf):
            print(f"  (bỏ qua {name}: không có {annf})")
            continue
        got = coco_items(annf, imgd, ren)
        # Lọc theo PHÍA của bộ train. Đo được: nếu không lọc thì 90% ảnh probe
        # nằm trong tập train, và con số recall là học thuộc chứ không phải tổng
        # quát hoá. Dùng đúng hàm chia của make_tiles_data.py, không viết lại —
        # hai định nghĩa khác nhau là quay lại đúng cái bẫy đó.
        if args.split != "all":
            got = [g for g in got
                   if split_of(clip_of(name, g[3]), 0.15) == args.split]
        random.shuffle(got)
        items += [(name, p, b, mw) for p, bs, mw, _fn in got[:args.n * 3]
                  for b in bs[:1]]
    random.shuffle(items)
    print(f"{len(items)} vật thật khả dụng, lấy {args.n} mỗi cỡ")

    kw = {"conf": args.conf}
    if args.weights:
        kw["weights"] = args.weights
    t0 = time.time()
    det = build_detector(args.backend, **kw)
    print(f"nạp {args.backend} mất {time.time()-t0:.1f}s")

    tile_px = 10_000 if args.no_tile else args.tile_px
    upscale = 1.0 if args.no_tile else args.upscale
    print(f"cắt ô {'TẮT (cả khung)' if args.no_tile else f'{args.tile_px}px, phóng {args.upscale}x'}"
          f"   conf={args.conf}\n")

    print("  cỡ vật   trúng      recall   phát hiện/khung   ms/khung")
    print("  " + "-" * 58)
    # Tách theo từng bộ dữ liệu. Con số gộp che mất chuyện quan trọng nhất: xem
    # ảnh thì 6/6 trúng đều là UAVVaste (drone, nhìn từ trên) còn 6/6 bỏ sót đều
    # là RoLID (dashcam). "45% recall" gộp lại là trung bình của ~cao và ~0.
    for tp in sizes:
        hit = tot = ndet = 0
        per: dict = {}
        ms = 0.0
        for name, path, bbox, meta_w in items:
            if tot >= args.n:
                break
            img = cv2.imread(path)
            if img is None:
                continue
            k = img.shape[1] / meta_w        # ảnh trên đĩa đã thu nhỏ bao nhiêu
            scene, gt = make_scene(img, [v * k for v in bbox], tp)
            if scene is None:
                continue
            tot += 1
            t = time.perf_counter()
            dets = detect_in_zone(det, scene, [(int(p[0] * WIN_W), int(p[1] * WIN_H))
                                               for p in ZONE],
                                  tile_px=tile_px, upscale=upscale)
            ms += (time.perf_counter() - t) * 1000
            ndet += len(dets)
            ok = any(iou(d.box, gt) >= 0.3 for d in dets)
            hit += ok
            h, n = per.get(name, (0, 0))
            per[name] = (h + ok, n + 1)
        if tot:
            br = "  ".join(f"{k} {100.0*h/max(1,n):.0f}%({h}/{n})"
                           for k, (h, n) in sorted(per.items()))
            print(f"  {tp:3d}px    {hit:3d}/{tot:<3d}    {100.0*hit/tot:5.1f}%"
                  f"       {ndet/tot:5.2f}          {ms/tot:6.0f}")
            print(f"           theo bo: {br}")

    if args.clean:
        frames = clean_frames(args.clean)
        n_fp = n_img = 0
        for f in frames:
            dets = detect_in_zone(det, f, [(int(p[0] * WIN_W), int(p[1] * WIN_H))
                                           for p in ZONE],
                                  tile_px=tile_px, upscale=upscale)
            n_fp += len(dets)
            n_img += bool(dets)
        print(f"\n  khung SẠCH: {len(frames)} khung -> {n_fp} phát hiện, "
              f"{n_img} khung có ít nhất 1 ({100.0*n_img/max(1,len(frames)):.1f}%)")
        print("  (hướng 1 đo được 15,8% trên 11 cảnh CCTV chưa từng thấy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
