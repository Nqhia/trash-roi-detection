"""Xuất ẢNH cho phép đo sự kiện ABODA — để soi bằng mắt, không chỉ đọc số.

    python3 tools/event_report.py

Với mỗi video: chạy pipeline y hệt tools/event_latency.py (cùng config, cùng
nhịp), và lưu ba khoảnh khắc:

    1. lúc VẬT XUẤT HIỆN   (khung sự kiện, hộp vàng = mốc tự dò)
    2. lúc PIPELINE BÁO    (ô nóng đỏ + hộp detector xanh)
    3. nếu KHÔNG BÁO       — khung cuối, để thấy nó đã bỏ lỡ cái gì

Mỗi ảnh có banner ghi lượt sự kiện / lượt báo / độ trễ quy đổi. Kết quả vào
test_cases/25_su_kien_aboda/ — đây là bằng chứng mắt thường cho con số
"recall sự kiện X/6" trong README.
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

from core.pipeline import ZoneTrashDetector      # noqa: E402
from core.scorers import build_scorer            # noqa: E402
from tools.find_events import find_event         # noqa: E402
from tools.event_latency import ABODA, FULL_POLY, USABLE   # noqa: E402


def banner(im, text, color=(255, 255, 255)):
    cv2.rectangle(im, (0, 0), (im.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(im, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1)
    return im


def annotate(frame, res, gt_box=None):
    vis = frame.copy()
    if gt_box is not None:
        x, y, bw, bh = gt_box
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 210, 255), 2)
    for c in res.hot if res is not None else []:
        cv2.rectangle(vis, (c.x1, c.y1), (c.x2, c.y2), (0, 0, 255), 2)
    for b in (res.verify_boxes if res is not None else []):
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      (0, 255, 0), 2)
        if len(b) > 4:
            cv2.putText(vis, f"{b[4]:.2f}", (int(b[0]), max(14, int(b[1]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
    return vis


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/day_cfg.yaml")
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--out", default="test_cases/25_su_kien_aboda")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(HERE, a.config), encoding="utf-8"))
    cfg["clutter"] = dict(cfg.get("clutter", {}), enabled=False)
    out = os.path.join(HERE, a.out)
    os.makedirs(out, exist_ok=True)

    montage = []
    print(f"config {a.config} · quét mỗi {a.step} khung (xem docstring "
          f"event_latency.py về quy đổi lượt -> phút)")
    for v in USABLE:
        p = os.path.join(ABODA, f"video{v}.avi")
        e = find_event(p)
        if e is None:
            print(f"  video{v}: không dò được mốc — bỏ")
            continue
        ev_frame, gt_box, _img, _ref = e
        det = ZoneTrashDetector(cfg, build_scorer(cfg.get("scorer", {})),
                                camera_id=f"rep{v}", zone_id="z")
        cap = cv2.VideoCapture(p)
        i = n = 0
        ev_scan = None
        shot_event = shot_alert = None
        last = None
        alert_scan = None
        # BÁO ĐÚNG CHỖ: ô nóng lúc báo phải TRÙM lên vật. Soi bằng mắt bản đầu
        # phát hiện thước cũ đếm cả cảnh báo bắn ở CHỖ KHÁC (người nán lại, mép
        # bàn ghế) là "bắt được sự kiện" — 6/6 là con số bị thổi phồng.
        strict_scan = None
        gx1, gy1 = gt_box[0], gt_box[1]
        gx2, gy2 = gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % a.step == 0:
                res = det.scan(fr, FULL_POLY, (), now=float(n) * 30.0)
                last = (fr, res)
                if ev_scan is None and i >= ev_frame:
                    ev_scan = n
                    shot_event = annotate(fr, res, gt_box)
                    banner(shot_event,
                           f"video{v} · VAT XUAT HIEN o luot {n} (hop vang = moc tu do)",
                           (0, 210, 255))
                if ev_scan is not None and res.alert:
                    on_gt = any(c.x1 < gx2 and c.x2 > gx1 and c.y1 < gy2 and c.y2 > gy1
                                for c in res.hot)
                    if alert_scan is None:
                        alert_scan = n
                    if on_gt and strict_scan is None:
                        strict_scan = n
                        shot_alert = annotate(fr, res, gt_box)
                        d = strict_scan - ev_scan
                        banner(shot_alert,
                               f"video{v} · BAO DUNG CHO o luot {n} = tre {d} luot"
                               f" (~{d*30/60:.1f} phut quy doi)", (0, 255, 0))
                    elif shot_alert is None:
                        shot_alert = annotate(fr, res, gt_box)
                        banner(shot_alert,
                               f"video{v} · bao o luot {n} nhung O CHO KHAC"
                               f" (khong trum vat vang)", (0, 210, 255))
                n += 1
            i += 1
        cap.release()
        if shot_event is None:
            continue
        if shot_alert is None and last is not None:
            shot_alert = annotate(last[0], last[1], gt_box)
            banner(shot_alert, f"video{v} · KHONG BAO trong {n} luot", (0, 0, 255))
        if strict_scan is None and alert_scan is not None and last is not None:
            # co bao nhung chua bao gio dung cho -> giu banner "O CHO KHAC"
            pass
        pair = np.hstack([cv2.resize(shot_event, (640, 480)),
                          cv2.resize(shot_alert, (640, 480))])
        cv2.imwrite(os.path.join(out, f"video{v}.jpg"), pair)
        if strict_scan is not None:
            d = f"BAO DUNG CHO, tre {strict_scan - ev_scan} luot"
        elif alert_scan is not None:
            d = "co bao nhung SAI CHO (khong trum vat)"
        else:
            d = "KHONG BAO"
        print(f"  video{v}: vat o luot {ev_scan} · {d}  -> {a.out}/video{v}.jpg")
        montage.append(cv2.resize(pair, (960, 360)))
    if montage:
        cv2.imwrite(os.path.join(out, "tong_hop.jpg"), np.vstack(montage))
        print(f"\n-> {a.out}/tong_hop.jpg  (trai=luc vat xuat hien, phai=luc bao)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
