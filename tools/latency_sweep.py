"""Quét đánh đổi ĐỘ TRỄ ↔ BÁO NHẦM để chọn dwell / confirm / nhịp.

    python3 tools/latency_sweep.py --clean ../data/aboda/video1.avi,...

Config mặc định báo ở lượt thứ 8 (`dwell 5` + `confirm 4/6`) = 4,0 phút ở nhịp
30s, trong khi mục tiêu là dưới 3 phút. Hạ xuống thì nhanh hơn nhưng có thể
tăng báo nhầm — tool này đo cả hai vế trên cùng một bộ tham số thay vì đoán.

Điểm đáng chú ý: từ khi có tầng xác nhận, `dwell` KHÔNG còn phải gánh một mình
việc dập FP nữa (detector bác bỏ ~3/4 số ô nóng). Nên rất có thể hạ được dwell
mà FP không tăng — đó chính là giả thuyết cần kiểm ở đây.

Độ trễ đo bằng cảnh tổng hợp nên là con số TẤT ĐỊNH (số lượt tới khi bắn).
Báo nhầm đo trên chuỗi CCTV sạch thật.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import cv2
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

from core.pipeline import ZoneTrashDetector   # noqa: E402
from core.scorers import build_scorer         # noqa: E402

COMBOS = [
    (1, 1, 1, "KHONG cho gi ca"),
    (2, 1, 1, ""),
    (5, 1, 1, "chi dwell, bo confirm"),
    (4, 1, 1, ""),
    (3, 1, 1, ""),
    # (dwell, n, m, nhãn)
    (5, 4, 6, "mac dinh"),
    (4, 3, 5, ""),
    (3, 3, 5, ""),
    (3, 2, 4, ""),
    (2, 2, 3, ""),
    (2, 1, 2, "nhanh nhat"),
]


def latency_scans(cfg) -> int:
    """Số LƯỢT từ lúc rác xuất hiện tới lúc bắn. Cảnh tổng hợp -> tất định."""
    from integration_test import ZONE, BlobScorer, frame
    c = copy.deepcopy(cfg)
    c["verify"] = dict(c.get("verify", {}), enabled=False)
    c["clutter"] = dict(c.get("clutter", {}), enabled=False)
    det = ZoneTrashDetector(c, BlobScorer(), camera_id="lat", zone_id="z")
    t = 0.0
    for _ in range(8):
        det.scan(frame(drain=False), ZONE, (), now=t)
        t += 1.0
    for k in range(1, 40):
        if det.scan(frame(drain=False, trash=True), ZONE, (), now=t).alert:
            return k
        t += 1.0
    return -1


def fp_on_clean(cfg, videos, zone, every=25, n_seed=4, n_max=18) -> tuple:
    """Số cảnh báo trên chuỗi CCTV SẠCH (không có rác)."""
    import json
    poly = json.load(open(zone, encoding="utf-8"))["points"]
    n_alert = n_scan = 0
    for vid in videos:
        cap = cv2.VideoCapture(vid)
        seq, k = [], 0
        while len(seq) < n_max:
            ok, f = cap.read()
            if not ok:
                break
            if k % every == 0:
                seq.append(f)
            k += 1
        cap.release()
        if len(seq) <= n_seed:
            continue
        det = ZoneTrashDetector(cfg, build_scorer({"kind": "constant", "value": 0.0}),
                                camera_id=os.path.basename(vid), zone_id="z")
        t = 0.0
        for f in seq[:n_seed]:
            det.scan(f, poly, [], now=t)
            t += 1.0
        for f in seq[n_seed:]:
            n_alert += bool(det.scan(f, poly, [], now=t).alert)
            n_scan += 1
            t += 1.0
    return n_alert, n_scan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/day_cfg.yaml")
    ap.add_argument("--clean", required=True, help="video sạch, cách nhau dấu phẩy")
    ap.add_argument("--zone", required=True)
    ap.add_argument("--intervals", default="30,20,15")
    ap.add_argument("--no-verify", action="store_true",
                    help="tat tang xac nhan de xem no dong gop bao nhieu")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.no_verify:
        base["verify"] = dict(base.get("verify", {}), enabled=False)
    videos = [v for v in args.clean.split(",") if v]
    ivs = [float(x) for x in args.intervals.split(",")]

    print(f"{len(videos)} video sạch · tầng xác nhận: "
          f"{'BẬT' if base.get('verify', {}).get('enabled') else 'tắt'}\n")
    hdr = "  dwell confirm | lượt |" + "".join(f"  {iv:.0f}s   " for iv in ivs) \
          + "| báo nhầm chuỗi sạch"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for dwell, n, m, note in COMBOS:
        cfg = copy.deepcopy(base)
        cfg["decide"] = dict(cfg["decide"], dwell_scans=dwell)
        cfg["confirm"] = {"n": n, "m": m}
        k = latency_scans(cfg)
        cols = ""
        for iv in ivs:
            mins = k * iv / 60.0
            mark = " " if mins <= 3.0 else "!"
            cols += f" {mins:4.1f}{mark}  "
        na, ns = fp_on_clean(cfg, videos, args.zone)
        print(f"  {dwell:^5d} {n}/{m:<5d} | {k:^4d} |{cols}| {na}/{ns} lượt"
              + (f"   <- {note}" if note else ""))
    print("\n  '!' = vượt mục tiêu 3 phút.  Độ trễ tất định (cảnh tổng hợp);")
    print("  báo nhầm đo trên chuỗi CCTV sạch thật, mặt nạ nhiễu theo config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
