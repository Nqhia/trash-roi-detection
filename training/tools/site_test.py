"""Chạy trên CHÍNH luồng camera EcoVision, chỗ rác được vứt ra thật.

    python3 tools/site_test.py

Đây mới là phép đo đáng tin cho use case: CCTV cố định, góc xiên, trong nhà,
rác thật do người vứt ra chứ không phải ghép. Ba bộ dữ liệu công khai đều là
vật thay thế — UAVVaste là drone nhìn từ trên, RoLID là dashcam.

Nhãn thật lấy bằng cách trừ nền `_ref.npy` (khung sạch chụp trước khi vứt rác),
đúng cách đã tạo ra `_detected.jpg`. Không phải nhãn tay nên có thể sai vài px,
nhưng vị trí và số lượng vật thì chắc chắn đúng.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import build_detector, detect_in_zone   # noqa: E402

SITE = "/mnt/c/Users/kangh/Documents/TrashDataset/data/site/pos_raw"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def gt_boxes(frame, ref, diff_thr=40, min_area=90, x_min_frac=0.10):
    """Hộp vật thật = thành phần liên thông của |khung - nền|, gộp ở mức VẬT.

    Bản đầu ra 97 hộp cho ~5 vật: tờ giấy nhàu bị tách thành 4 mảnh rời và dải
    mép trái (cái ghế đen bị xê dịch) cũng thành "vật". Đếm như thế thì mọi tỉ
    lệ recall đều vô nghĩa — mẫu số toàn là mảnh vụn.

    Nở mạnh trước khi tách thành phần để các mảnh của cùng một vật dính lại,
    và bỏ dải mép trái nơi có ghế chứ không có rác.
    """
    d = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, m = cv2.threshold(d, diff_thr, 255, cv2.THRESH_BINARY)
    m = cv2.dilate(m, np.ones((7, 7), np.uint8), iterations=2)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    x_min = x_min_frac * frame.shape[1]
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area and max(w, h) >= 12 and x >= x_min:
            # trừ lại phần đã nở để hộp bám sát vật
            out.append((float(x + 6), float(y + 6), float(x + w - 6), float(y + h - 6)))
    return out


def iou(a, b):
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)


def hit(gt, dets):
    """Trúng nếu IoU>=0.3 HOẶC tâm phát hiện nằm trong hộp thật.

    Chỉ dùng IoU với vật 19px thì quá khắt khe: lệch 5px đã tụt xuống dưới 0.3
    dù model rõ ràng đã chỉ đúng chỗ. Ta đang hỏi "có thấy không", không phải
    "khoanh có khít không".
    """
    for d in dets:
        if iou(d.box, gt) >= 0.3:
            return d
        cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
        if gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]:
            return d
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--upscale", type=float, default=4.0)
    ap.add_argument("--conf", type=float, default=0.08)
    ap.add_argument("--out", default=".")
    ap.add_argument("--backends", default="owlv2,yoloworld,coco",
                    help="owlv2,yoloworld,coco,trained")
    ap.add_argument("--weights", help="đường dẫn best.pt cho backend 'trained'")
    args = ap.parse_args()

    ref = np.load(f"{SITE}/_ref.npy")
    names = sorted(f for f in os.listdir(SITE) if f.startswith("frame_"))
    # Lấy các khung có NHIỀU vật nhất — đó là lúc đủ cả 2 chai + tờ giấy.
    scored = []
    for fn in names:
        im = cv2.imread(os.path.join(SITE, fn))
        if im is not None and im.shape == ref.shape:
            scored.append((len(gt_boxes(im, ref)), fn, im))
    scored.sort(reverse=True, key=lambda t: t[0])
    picked = scored[:args.frames]
    h, w = ref.shape[:2]
    poly = [(2, 2), (w - 2, 2), (w - 2, h - 2), (2, h - 2)]
    print(f"ảnh {w}x{h}, dùng {len(picked)} khung nhiều vật nhất "
          f"({picked[0][0]} vật ở khung tốt nhất)\n")

    kinds = [k for k in args.backends.split(",") if k]
    panels = {}
    for kind in kinds:
        kw = {"conf": args.conf}
        if kind == "coco":
            kw["weights"] = "yolo11s.pt"
        elif kind == "trained" and args.weights:
            kw["weights"] = args.weights
        det = build_detector(kind, **kw)
        tot = found = nfp = 0
        by_size: dict = {}
        best = None
        for ngt, fn, im in picked:
            gts = gt_boxes(im, ref)
            dets = detect_in_zone(det, im, poly, tile_px=320, overlap=0.5,
                                  upscale=args.upscale)
            matched = set()
            for g in gts:
                tot += 1
                d = hit(g, dets)
                s = int(max(g[2] - g[0], g[3] - g[1]))
                bucket = "<25px" if s < 25 else ("25-45px" if s < 45 else ">=45px")
                a, b = by_size.get(bucket, (0, 0))
                by_size[bucket] = (a + bool(d), b + 1)
                if d is not None:
                    found += 1
                    matched.add(id(d))
            nfp += sum(1 for d in dets if id(d) not in matched)
            if best is None:
                best = (im, gts, dets)
        br = "  ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(by_size.items()))
        print(f"{kind:10s} bắt được {found}/{tot} vật thật ({100.0*found/max(1,tot):.0f}%)"
              f"   báo nhầm {nfp}   [{br}]")

        im, gts, dets = best
        vis = im.copy()
        for g in gts:
            c = (60, 255, 60) if hit(g, dets) else (0, 165, 255)
            cv2.rectangle(vis, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), c, 2)
            cv2.putText(vis, f"{int(max(g[2]-g[0], g[3]-g[1]))}px",
                        (int(g[0]), max(10, int(g[1]) - 3)), FONT, .38, c, 1, cv2.LINE_AA)
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d.box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(vis, f"{d.label} {d.score:.2f}", (x1, min(h - 3, y2 + 11)),
                        FONT, .35, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.rectangle(vis, (0, 0), (w, 20), (0, 0, 0), -1)
        cv2.putText(vis, f"{kind}   XANH=bat duoc  CAM=BO SOT  DO=model doan",
                    (6, 14), FONT, .42, (255, 255, 255), 1, cv2.LINE_AA)
        panels[kind] = vis

    sheet = np.vstack([panels[k] for k in kinds])
    p = os.path.join(args.out, f"viz_site_eco_{'_'.join(kinds)}.jpg")
    cv2.imwrite(p, sheet)
    print(f"\nlưu {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
