"""Người đi vào rồi ĐỨNG YÊN thì có bị báo không.

    python3 tools/standing_test.py --source ../data/aboda/video6.avi \\
        --zone runs/aboda_v3/zone.json --config <cfg>

Đây là ca duy nhất mà che người bằng YOLO có lý. Người đi qua thì `dwell` lọc
được (đổi 1-2 lượt rồi thôi), nhưng người ĐỨNG YÊN quá `dwell_scans` thì ô của
họ nóng lên y như một vật lạ nằm lại — mà xét theo định nghĩa "vùng trống có
vật lạ thì báo" thì họ ĐÚNG là vật lạ. Chỉ có điều khách không muốn bị báo vì
một người đứng đợi thang máy.

Giả thuyết cần kiểm: tầng xác nhận tự lo được ca này, vì detector chỉ học rác
chứ không học người — nên nó sẽ bác bỏ. Nếu đúng thì không cần YOLO che người,
và ta tránh được vùng mù 24%.

Tool in ra: lượt nào có ô nóng, ô đó có nằm trong box người không, và cuối cùng
có bắn cảnh báo không.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import ZoneTrashDetector   # noqa: E402
from core.scorers import build_scorer         # noqa: E402
from tools.run_video import YoloBoxes, iter_frames   # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--config", required=True)
    # `yolo11n.pt` là tên chuẩn của ultralytics — không có sẵn thì nó tự tải.
    # KHÔNG hardcode đường dẫn tuyệt đối: gói này phải chạy được trên máy khác.
    ap.add_argument("--yolo", default="yolo11n.pt",
                    help="CHỈ để chấm điểm: ô nóng có trùng người không. "
                         "Không đưa vào pipeline.")
    ap.add_argument("--out", default="runs/standing")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    poly = json.load(open(args.zone, encoding="utf-8"))["points"]
    interval = float(cfg["scan"]["interval_s"])
    os.makedirs(args.out, exist_ok=True)

    # YOLO ở đây KHÔNG đưa vào pipeline — chỉ dùng để chấm điểm xem ô nóng có
    # nằm trên người hay không. Đưa vào thì lại che mất chính thứ cần đo.
    boxes_of = YoloBoxes(args.yolo)
    det = ZoneTrashDetector(cfg, build_scorer(cfg.get("scorer", {"kind": "constant"})),
                            camera_id="st", zone_id="z",
                            state_dir=os.path.join(args.out, "state"))

    n = n_alert = n_hot_person = n_hot_other = 0
    n_veto_person = 0
    saved = 0
    for t, frame in iter_frames(args.source, interval):
        people = boxes_of(frame)          # chỉ để chấm điểm
        res = det.scan(frame, poly, [], now=t)   # pipeline KHÔNG nhận box
        n += 1
        if not res.hot:
            continue

        def on_person(c):
            cx, cy = (c.x1 + c.x2) / 2, (c.y1 + c.y2) / 2
            return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in people)

        hp = [c for c in res.hot if on_person(c)]
        ho = [c for c in res.hot if not on_person(c)]
        n_hot_person += len(hp)
        n_hot_other += len(ho)
        if res.alert:
            n_alert += 1
        if hp and not res.verify_boxes:
            n_veto_person += 1

        if hp and saved < 6:
            vis = frame.copy()
            for b in people:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (255, 170, 0), 2)
            for c in hp:
                cv2.rectangle(vis, (c.x1, c.y1), (c.x2, c.y2), (60, 60, 255), 2)
            for b in res.verify_boxes:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (255, 0, 200), 2)
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(vis, f"t={t:.0f}s  {len(hp)} o nong TREN NGUOI  "
                             f"detector khoanh {len(res.verify_boxes)}  "
                             f"{'CANH BAO' if res.alert else 'khong bao'}",
                        (6, 17), FONT, .5,
                        (0, 0, 255) if res.alert else (0, 220, 120), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(args.out, f"standing_{saved}.jpg"), vis)
            saved += 1

    print(f"{n} lượt quét")
    print(f"  ô nóng NẰM TRÊN người : {n_hot_person}")
    print(f"  ô nóng ở chỗ khác     : {n_hot_other}")
    print(f"  lượt có ô nóng trên người mà detector bác bỏ hết: {n_veto_person}")
    print(f"  TỔNG CẢNH BÁO         : {n_alert}")
    print(f"  ảnh lưu ở {args.out}/standing_*.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
