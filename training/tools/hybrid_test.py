"""Ghép hướng 1 (cổng đổi nền) với hướng 2 (detector) và đo bằng thước cũ.

    python3 tools/hybrid_test.py

Ba cấu hình chạy trên CÙNG dữ liệu:

  A. chỉ cổng đổi      — hướng 1 nguyên bản, change_only
  B. chỉ detector      — quét cả vùng bằng detector
  C. ghép nối tiếp     — cổng đổi chỉ ra CHỖ đổi, detector phán CÓ PHẢI RÁC

Hai thước, đều nằm ngoài mọi tập train:
  - khung eco có rác thật do người vứt  -> recall
  - khung CCTV sạch (ABODA + site)      -> báo nhầm

Điều cần trả lời: ghép nối tiếp thì recall thành TÍCH của hai tầng. Cổng đổi
bắt gần như mọi vật lạ, detector chỉ 75% trên khung eco. Nếu detector đóng vai
người phủ quyết thì trần recall tụt xuống 75% — thấp hơn chính hướng 1. Đổi lại
được bao nhiêu FP? Con số đó quyết định có nên ghép hay không.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = "/mnt/c/Users/kangh/Documents/TrashDataset"
PATCH = f"{ROOT}/patch_classifier"

# Hai project ĐỀU có package tên `core`. Nạp của hướng 2 trước, giữ lại đúng
# thứ cần, rồi xoá khỏi sys.modules để `core` sau đó trỏ sang hướng 1. Gộp
# đường dẫn ngay từ đầu thì một trong hai bên im lặng che mất bên kia.
sys.path.insert(0, HERE)
from core.detector import build_detector, detect_in_zone   # noqa: E402
from core.tiler import nms                                 # noqa: E402

for _m in [m for m in list(sys.modules) if m == "core" or m.startswith("core.")]:
    del sys.modules[_m]
sys.path.insert(0, PATCH)
from core.pipeline import ZoneTrashDetector as Patch   # noqa: E402
from core.scorers import build_scorer                  # noqa: E402

SITE = f"{ROOT}/data/site/pos_raw"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def gt_boxes(frame, ref, diff_thr=40, min_area=90, x_min_frac=0.10):
    """Nhãn thật trên khung eco = trừ nền, gộp ở mức vật (như site_test.py)."""
    d = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, m = cv2.threshold(d, diff_thr, 255, cv2.THRESH_BINARY)
    m = cv2.dilate(m, np.ones((7, 7), np.uint8), iterations=2)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area and max(w, h) >= 12 and x >= x_min_frac * frame.shape[1]:
            out.append((float(x + 6), float(y + 6),
                        float(x + w - 6), float(y + h - 6)))
    return out


def cluster(cells, gap=1):
    """Gộp các ô đổi kề nhau thành VÙNG. Detector cần bối cảnh 320px, một ô 48px
    đưa thẳng vào là quá nhỏ để nó nhận ra thứ gì."""
    todo = list(cells)
    groups = []
    while todo:
        seed = todo.pop()
        grp = [seed]
        moved = True
        while moved:
            moved = False
            for c in list(todo):
                if any(abs(c.row - g.row) <= gap and abs(c.col - g.col) <= gap
                       for g in grp):
                    grp.append(c)
                    todo.remove(c)
                    moved = True
        groups.append(grp)
    return groups


def region_of(grp, w, h, pad=64, min_side=160):
    x1 = min(c.x1 for c in grp) - pad
    y1 = min(c.y1 for c in grp) - pad
    x2 = max(c.x2 for c in grp) + pad
    y2 = max(c.y2 for c in grp) + pad
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(min_side, x2 - x1, y2 - y1)
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(w, x1 + side)), int(min(h, y1 + side))
    return x1, y1, x2, y2


def run_detector(det, frame, regions, upscale=2.0):
    """Chạy detector trên các vùng do cổng đổi chỉ ra, quy hộp về khung gốc."""
    boxes, scores = [], []
    crops, offs = [], []
    for (x1, y1, x2, y2) in regions:
        c = frame[y1:y2, x1:x2]
        if c.size == 0:
            continue
        s = (320 * upscale) / max(c.shape[:2])
        crops.append(cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC))
        offs.append((x1, y1, s))
    for crop, (ox, oy, s), dets in zip(crops, offs, det.detect(crops)):
        for d in dets:
            boxes.append((ox + d.box[0] / s, oy + d.box[1] / s,
                          ox + d.box[2] / s, oy + d.box[3] / s))
            scores.append(d.score)
    keep = nms(boxes, scores, 0.5)
    return [boxes[i] for i in keep]


def draw(frame, regions, dets, gts=(), tag=""):
    """VANG = vung cong doi chi ra · DO = detector xac nhan · XANH = rac that."""
    v = frame.copy()
    for (x1, y1, x2, y2) in regions:
        cv2.rectangle(v, (x1, y1), (x2, y2), (0, 200, 255), 2)
    for g in gts:
        cv2.rectangle(v, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])),
                      (60, 255, 60), 2)
    for b in dets:
        cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      (0, 0, 255), 2)
    h, w = v.shape[:2]
    cv2.rectangle(v, (0, 0), (w, 24), (0, 0, 0), -1)
    cv2.putText(v, tag, (6, 17), FONT, .48, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.resize(v, (620, 420))


def sheet(panels, cols=3, path="viz.jpg"):
    if not panels:
        return
    while len(panels) % cols:
        panels.append(np.zeros_like(panels[0]))
    cv2.imwrite(path, np.vstack([np.hstack(panels[i:i + cols])
                                 for i in range(0, len(panels), cols)]))
    print(f"luu {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights",
                    default="runs/detect/runs/tiles_audited/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--cell", type=int, default=48)
    args = ap.parse_args()




    det = build_detector("trained", weights=args.weights, conf=args.conf)

    CFG = {
        "grid": {"cell_px": args.cell, "overlap": 0.5, "occlusion_thr": 0.3},
        "change": {"enabled": True, "alpha": 0.05, "thr": 6.0},
        "decide": {"mode": "change_only", "dwell_scans": 2, "min_hot_cells": 1},
        "confirm": {"n": 1, "m": 1},
        "alert": {"mode": "latch", "rearm_scans": 4, "merge_radius_cells": 2},
        "clutter": {"enabled": False},
        "scene_shift": {"thr_px": 12.0},
        "stabilize": {"enabled": False},
    }

    ref = np.load(f"{SITE}/_ref.npy")
    names = sorted(f for f in os.listdir(SITE) if f.startswith("frame_"))
    frames = []
    for fn in names:
        im = cv2.imread(os.path.join(SITE, fn))
        if im is not None and im.shape == ref.shape:
            frames.append(im)
    h, w = ref.shape[:2]
    poly = [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]]
    poly_px = [(2, 2), (w - 2, 2), (w - 2, h - 2), (2, h - 2)]

    # Chỉ lấy các khung nhiều vật nhất, giống site_test.py để so được.
    scored = sorted(((len(gt_boxes(f, ref)), i) for i, f in enumerate(frames)),
                    reverse=True)[:6]

    patch = Patch(CFG, build_scorer({"kind": "constant", "value": 0.0}),
                  camera_id="hy", zone_id="z")
    # Dựng nền từ chính khung sạch, rồi mới đưa khung có rác vào.
    for _ in range(4):
        patch.scan(ref, poly, [], now=0.0)

    tot = a_hit = b_hit = c_hit = 0
    eco_panels, kill_panels = [], []
    a_reg = b_fp = c_fp = 0
    for k, (_n, idx) in enumerate(scored):
        f = frames[idx]
        gts = gt_boxes(f, ref)
        r = patch.scan(f, poly, [], now=float(10 + k))
        cells = r.hot or [c for c in patch.grid.cells if r.scores.get(c.id, 0) > 0]
        regions = [region_of(g, w, h) for g in cluster(cells)]
        a_reg += len(regions)
        # B phải chạy CÓ CẮT Ô như lúc đánh giá riêng, không thì ép cả khung vào
        # 640px là tự bôi xấu nó (đo được: 29% so với 75%).
        det_all = detect_in_zone(det, f, poly_px, tile_px=320, overlap=0.5,
                                 upscale=2.0)
        det_all = [d.box for d in det_all]
        det_hyb = run_detector(det, f, regions) if regions else []

        def covers(bs, g):
            for b in bs:
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if g[0] <= cx <= g[2] and g[1] <= cy <= g[3]:
                    return True
            return False

        if len(eco_panels) < 6:
            eco_panels.append(draw(
                f, regions, det_hyb, gts,
                f"ECO  {len(regions)} vung -> {len(det_hyb)} xac nhan  "
                f"({len(gts)} vat that)"))
        for g in gts:
            tot += 1
            a_hit += any(gg[0] <= (c.x1 + c.x2) / 2 <= gg[2] for gg in [g]
                         for c in cells
                         if g[1] <= (c.y1 + c.y2) / 2 <= g[3])
            b_hit += covers(det_all, g)
            c_hit += covers(det_hyb, g)
        b_fp += max(0, len(det_all) - sum(covers([b], g)
                                          for b in det_all for g in gts))
        c_fp += max(0, len(det_hyb) - len(gts))

    print(f"khung eco: {len(scored)} khung, {tot} vat that")
    print(f"  A · chi cong doi     bat {a_hit}/{tot} ({100*a_hit/max(1,tot):.0f}%)"
          f"   {a_reg} vung dua sang detector")
    print(f"  B · chi detector     bat {b_hit}/{tot} ({100*b_hit/max(1,tot):.0f}%)")
    print(f"  C · ghep noi tiep    bat {c_hit}/{tot} ({100*c_hit/max(1,tot):.0f}%)")

    # ---- bao nham tren khung CCTV sach ----

    # PHẢI chạy trên CHUỖI khung liên tiếp của một video sạch, không phải trên
    # các khung rời. Bản đầu dựng nền từ chính khung sắp quét rồi quét lại đúng
    # khung đó — dĩ nhiên không ô nào đổi, và "C = 0 báo nhầm" chỉ nói rằng
    # không đổi thì không báo. Chuỗi thật mới có nắng đổi, người đi, lá rung.
    nb = nc = nb_img = nc_img = n = na_reg = na_img = 0
    for vid in ("video1", "video3", "video6"):
        cap = cv2.VideoCapture(f"{ROOT}/data/aboda/{vid}.avi")
        seq = []
        k = 0
        while len(seq) < 14:
            ok, fr = cap.read()
            if not ok:
                break
            if k % 25 == 0:          # ~1 khung/giây, giống nhịp quét thưa
                seq.append(fr)
            k += 1
        cap.release()
        if len(seq) < 6:
            continue
        hh, ww = seq[0].shape[:2]
        p2 = [(2, 2), (ww - 2, 2), (ww - 2, hh - 2), (2, hh - 2)]
        pp = Patch(CFG, build_scorer({"kind": "constant", "value": 0.0}),
                   camera_id=vid, zone_id="z")
        for fr in seq[:4]:           # 4 khung đầu dựng nền
            pp.scan(fr, poly, [], now=0.0)
        for j, fr in enumerate(seq[4:]):
            r = pp.scan(fr, poly, [], now=float(10 + j))
            regions = [region_of(g, ww, hh) for g in cluster(r.hot)] if r.hot else []
            db = [d.box for d in detect_in_zone(det, fr, p2, tile_px=320,
                                                overlap=0.5, upscale=2.0)]
            dc = run_detector(det, fr, regions) if regions else []
            n += 1
            na_reg += len(regions)
            na_img += bool(regions)
            nb += len(db)
            nc += len(dc)
            nb_img += bool(db)
            nc_img += bool(dc)
            if regions and not dc and len(kill_panels) < 6:
                kill_panels.append(draw(
                    fr, regions, [], (),
                    f"{vid}: cong doi keu {len(regions)} vung -> "
                    f"detector BAC BO het (khong co rac)"))
    print(f"\nkhung CCTV sach: {n} luot quet tren 3 video (chuoi that, "
          f"co nguoi va anh sang doi)")
    print(f"  B · chi detector     {nb} phat hien, {nb_img}/{n} khung "
          f"({100*nb_img/max(1,n):.0f}%)")
    print(f"  C · ghep noi tiep    {nc} phat hien, {nc_img}/{n} khung "
          f"({100*nc_img/max(1,n):.0f}%)")
    sheet(eco_panels, 3, "viz_hybrid_eco.jpg")
    sheet(kill_panels, 3, "viz_hybrid_killed.jpg")
    print(f"  A · chi cong doi     {na_reg} vung, {na_img}/{n} luot "
          f"({100*na_img/max(1,n):.0f}%)   <- day la FP cua huong 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
