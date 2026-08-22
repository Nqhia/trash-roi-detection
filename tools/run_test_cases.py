"""Chạy lại toàn bộ bộ test và ghi kết quả vào `test_cases/`.

    python3 tools/run_test_cases.py

Bốn cấu hình trên CÙNG dữ liệu, cùng thước đo:

    A · model đứng riêng     detector quét cả vùng, không có cổng đổi
    B · patch đứng riêng     cổng đổi nền, không có detector
    C · patch + model        pipeline thật (ghép nối tiếp)
    D · khung sạch           cả ba chạy trên chuỗi CCTV KHÔNG có rác

Hai tập kiểm, đều nằm ngoài mọi tập train của detector:

  * khung eco   — camera EcoVision, rác thật do người vứt ra. Nhãn lấy bằng
                  trừ nền `_ref.npy` rồi gộp ở mức vật.
  * chuỗi sạch  — 3 video ABODA có người đi và ánh sáng đổi, không có rác.
                  Phải chạy theo CHUỖI chứ không phải khung rời: dựng nền từ
                  chính khung sắp quét rồi quét lại khung đó thì "0 báo nhầm"
                  chỉ nói rằng không đổi thì không báo.

Ghi ra `test_cases/`: `results.md` (bảng số) + ảnh từng ca để soi bằng mắt.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from core.pipeline import ZoneTrashDetector   # noqa: E402
from core.scorers import build_scorer         # noqa: E402
from core.verify import RegionVerifier        # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
SITE = f"{ROOT}/data/site/pos_raw"
ABODA = f"{ROOT}/data/aboda"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FULL_POLY = [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]]


# ---------------------------------------------------------------- nhãn thật

def gt_boxes(frame, ref, diff_thr=40, min_area=90, x_min_frac=0.10):
    """Vật thật = thành phần liên thông của |khung - nền|, gộp ở mức VẬT.

    Nở mạnh trước khi tách: bản đầu ra 97 "vật" cho ~5 món vì tờ giấy nhàu bị
    tách thành 4 mảnh rời. Bỏ dải mép trái nơi có ghế chứ không có rác.
    """
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


def hits(boxes, gt):
    """Trúng nếu TÂM phát hiện nằm trong hộp thật.

    Không dùng IoU: nhãn trừ nền là hộp bao quanh vùng đổi, còn model khoanh
    từng vật — với vật 16px thì lệch 5px đã tụt IoU xuống dưới 0.3 dù model rõ
    ràng chỉ đúng chỗ. Câu hỏi ở đây là "có thấy không", không phải "khít không".
    """
    for b in boxes:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]:
            return True
    return False


# ---------------------------------------------------------------- detector

def model_only(v: RegionVerifier, frame, tile=320, overlap=0.5):
    """Detector quét CẢ vùng bằng cách cắt ô — không có cổng đổi."""
    h, w = frame.shape[:2]
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), step)) or [0]
    ys = list(range(0, max(1, h - tile + 1), step)) or [0]
    if xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    if ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    crops, offs = [], []
    for y in ys:
        for x in xs:
            c = frame[y:y + tile, x:x + tile]
            if c.size:
                crops.append(cv2.resize(c, None, fx=v.upscale, fy=v.upscale,
                                        interpolation=cv2.INTER_CUBIC))
                offs.append((x, y))
    if not crops:
        return []
    out = []
    for (ox, oy), r in zip(offs, v._model().predict(crops, conf=v.conf,
                                                    verbose=False)):
        for b in r.boxes:
            q = b.xyxy[0].tolist()
            out.append((ox + q[0] / v.upscale, oy + q[1] / v.upscale,
                        ox + q[2] / v.upscale, oy + q[3] / v.upscale))
    return out


def draw(frame, gts, boxes, hot, tag, ok_color=(60, 255, 60)):
    vis = frame.copy()
    for c in hot:
        cv2.rectangle(vis, (c.x1, c.y1), (c.x2, c.y2), (60, 60, 255), 2)
    for g in gts:
        c = ok_color if hits(boxes, g) else (0, 165, 255)
        cv2.rectangle(vis, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), c, 2)
    for b in boxes:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      (255, 0, 200), 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(vis, tag, (6, 17), FONT, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def sheet(panels, path, cols=3, cell=(560, 350)):
    if not panels:
        return
    panels = [cv2.resize(p, cell) for p in panels]
    while len(panels) % cols:
        panels.append(np.zeros_like(panels[0]))
    cv2.imwrite(path, np.vstack([np.hstack(panels[i:i + cols])
                                 for i in range(0, len(panels), cols)]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/day_cfg.yaml")
    ap.add_argument("--out", default="test_cases")
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    v = RegionVerifier(cfg.get("verify", {}))
    if not v.enabled:
        return print("verify.enabled = false trong config") or 1

    # cấu hình patch: bỏ mặt nạ nhiễu (cần 5h mới chín, không đo được ở đây)
    base = dict(cfg)
    base["clutter"] = dict(cfg.get("clutter", {}), enabled=False)
    off = dict(base, verify=dict(cfg["verify"], enabled=False))

    lines = ["# Kết quả bộ test", "",
             "Sinh bởi `python3 tools/run_test_cases.py` — chạy lại là ra đúng số này.",
             "", f"Model: `{v.weights}` · conf {v.conf} · ô 320px phóng {v.upscale}×",
             "Mặt nạ nhiễu TẮT trong mọi ca (cần ~5 giờ mới chín, không đo được ở đây).",
             ""]

    # ============ CA 1-3 · khung eco có rác thật ============
    ref = np.load(f"{SITE}/_ref.npy")
    frames = []
    for fn in sorted(f for f in os.listdir(SITE) if f.startswith("frame_")):
        im = cv2.imread(os.path.join(SITE, fn))
        if im is not None and im.shape == ref.shape:
            frames.append((fn, im))
    # Đưa vào TOÀN BỘ khung có rác theo thứ tự, nhưng chỉ CHẤM ĐIỂM từ lượt thứ
    # `warmup` trở đi. Config thật có dwell 5 + confirm 4/6 = phải 8 lượt mới
    # báo; đưa đúng 6 khung rồi chấm là đang đo giai đoạn khởi động, và patch ra
    # 25% trong khi thực tế nó đạt 75%.
    with_trash = [(fn, im) for fn, im in frames if len(gt_boxes(im, ref)) >= 3]
    warmup = int(cfg["decide"]["dwell_scans"]) + int(cfg["confirm"]["m"])
    if len(with_trash) < warmup + args.frames:
        print(f"  chỉ có {len(with_trash)} khung có rác, cần "
              f"{warmup + args.frames} — kết quả sẽ lẫn giai đoạn khởi động")
    seq = with_trash[:warmup]
    picked = [(0, fn, im) for fn, im in with_trash[warmup:warmup + args.frames]]
    print(f"  {len(with_trash)} khung có rác · {warmup} khung đầu để khởi động · "
          f"chấm điểm trên {len(picked)} khung")
    h, w = ref.shape[:2]
    poly_px = [(2, 2), (w - 2, 2), (w - 2, h - 2), (2, h - 2)]

    det_b = ZoneTrashDetector(off, build_scorer({"kind": "constant", "value": 0.0}),
                              camera_id="B", zone_id="z")
    det_c = ZoneTrashDetector(base, build_scorer({"kind": "constant", "value": 0.0}),
                              camera_id="C", zone_id="z")
    for _ in range(4):                       # dựng nền từ khung sạch
        det_b.scan(ref, FULL_POLY, [], now=0.0)
        det_c.scan(ref, FULL_POLY, [], now=0.0)
    for j, (_fn, im) in enumerate(seq):      # cho qua dwell + confirm
        det_b.scan(im, FULL_POLY, [], now=float(1 + j))
        det_c.scan(im, FULL_POLY, [], now=float(1 + j))

    res = {k: [0, 0, 0] for k in "ABC"}   # [trúng, tổng, hộp thừa]
    pa, pb, pc = [], [], []
    for k, (_n, fn, im) in enumerate(picked):
        gts = gt_boxes(im, ref)
        t = float(10 + k)
        a_box = model_only(v, im)
        rb = det_b.scan(im, FULL_POLY, [], now=t)
        rc = det_c.scan(im, FULL_POLY, [], now=t)
        b_box = [(c.x1, c.y1, c.x2, c.y2) for c in rb.hot]
        c_box = list(rc.verify_boxes)
        for name, boxes in (("A", a_box), ("B", b_box), ("C", c_box)):
            for g in gts:
                res[name][1] += 1
                res[name][0] += hits(boxes, g)
            res[name][2] += max(0, len(boxes) - len(gts))
        pa.append(draw(im, gts, a_box, [], f"A model rieng - {len(a_box)} hop"))
        pb.append(draw(im, gts, [], rb.hot, f"B patch rieng - {len(rb.hot)} o nong"))
        pc.append(draw(im, gts, c_box, rc.hot,
                       f"C patch+model - {len(rc.hot)} o nong, {len(c_box)} xac nhan"))
    sheet(pa, f"{args.out}/01_model_only_eco.jpg")
    sheet(pb, f"{args.out}/02_patch_only_eco.jpg")
    sheet(pc, f"{args.out}/03_patch_plus_model_eco.jpg")

    lines += ["## Khung eco — rác thật do người vứt", "",
              f"{len(picked)} khung · {res['A'][1]} vật thật · "
              "ảnh: `01_model_only_eco.jpg`, `02_patch_only_eco.jpg`, "
              "`03_patch_plus_model_eco.jpg`", "",
              "| cấu hình | bắt được | hộp thừa |", "|---|---|---|"]
    for k, nm in (("A", "model đứng riêng"), ("B", "patch đứng riêng"),
                  ("C", "**patch + model**")):
        hit, tot, fp = res[k]
        lines.append(f"| {nm} | {hit}/{tot} · {100*hit/max(1,tot):.0f}% | {fp} |")

    # ============ CA 4 · chuỗi CCTV sạch ============
    lines += ["", "## Chuỗi CCTV sạch — không có rác", ""]
    na = nb = nc = ns = 0
    panels = []
    for vid in ("video1", "video3", "video6"):
        cap = cv2.VideoCapture(f"{ABODA}/{vid}.avi")
        seq, kk = [], 0
        while len(seq) < 16:
            ok, f = cap.read()
            if not ok:
                break
            if kk % 25 == 0:
                seq.append(f)
            kk += 1
        cap.release()
        if len(seq) < 6:
            continue
        hh, ww = seq[0].shape[:2]
        db = ZoneTrashDetector(off, build_scorer({"kind": "constant", "value": 0.0}),
                               camera_id=f"b{vid}", zone_id="z")
        dc = ZoneTrashDetector(base, build_scorer({"kind": "constant", "value": 0.0}),
                               camera_id=f"c{vid}", zone_id="z")
        for f in seq[:4]:
            db.scan(f, FULL_POLY, [], now=0.0)
            dc.scan(f, FULL_POLY, [], now=0.0)
        for j, f in enumerate(seq[4:]):
            t = float(10 + j)
            ab = model_only(v, f)
            rb = db.scan(f, FULL_POLY, [], now=t)
            rc = dc.scan(f, FULL_POLY, [], now=t)
            ns += 1
            na += bool(ab)
            nb += bool(rb.hot)
            nc += bool(rc.verify_boxes)
            if rb.hot and not rc.verify_boxes and len(panels) < 6:
                panels.append(draw(f, [], [], rb.hot,
                                   f"{vid}: patch keu {len(rb.hot)} o -> "
                                   "model BAC BO het"))
    sheet(panels, f"{args.out}/04_clean_rejected.jpg")
    lines += [f"{ns} lượt quét trên 3 video (có người đi, ánh sáng đổi) · "
              "ảnh: `04_clean_rejected.jpg`", "",
              "| cấu hình | lượt có báo nhầm |", "|---|---|",
              f"| model đứng riêng | {na}/{ns} · {100*na/max(1,ns):.0f}% |",
              f"| patch đứng riêng | {nb}/{ns} · {100*nb/max(1,ns):.0f}% |",
              f"| **patch + model** | **{nc}/{ns} · {100*nc/max(1,ns):.0f}%** |", ""]

    open(f"{args.out}/results.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n-> {args.out}/results.md + 4 ảnh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
