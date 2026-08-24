"""Đào ẢNH ÂM từ chính camera hiện trường — thứ duy nhất diệt được bắt nhầm ghế.

    python3 tools/mine_negatives.py --source rtsp://... --zone config/zone_live.json \
            --out training/data/site_neg --hours 12

Vì sao cần: đo trên 42 giờ chạy thật, 37/37 cảnh báo là GHẾ BỊ DỜI. Chỉnh chốt
cảnh báo không cứu được (quét `rearm_scans` 4->160 và bán kính gộp 2->6 vẫn còn
5,1 cảnh báo/ngày, trong khi ngưỡng là <1-2). Lý do: ghế kéo sang chỗ khác thì
sinh ô CHƯA TỪNG chốt, pipeline không có cách nào biết đó vẫn là cái ghế cũ.
Chỉ detector mới phân biệt được, và nó cần thấy ghế của ĐÚNG cảnh này.

Cách đào: chạy y hệt pipeline thật trên vùng ĐANG KHÔNG CÓ RÁC. Mọi hộp detector
vẽ ra ở đây đều là bắt nhầm, theo định nghĩa. Cắt vùng đó ra làm ảnh âm.

QUAN TRỌNG — ảnh cắt ra là ảnh SẠCH, không có lớp vẽ đè lên. Ảnh cảnh báo mà
`run_video.py` lưu đã bị vẽ ô đỏ và đường bao vùng lên trên, không dùng để train
được; đây là lý do phải có công cụ riêng thay vì tận dụng 37 ảnh sẵn có.

Người chạy phải bảo đảm vùng THẬT SỰ KHÔNG CÓ RÁC trong suốt thời gian đào. Có
rác mà vẫn đào là dạy model rằng rác không phải rác.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from core.pipeline import ZoneTrashDetector      # noqa: E402
from core.scorers import build_scorer            # noqa: E402
from tools.capture import open_source            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--config", default="config/day_cfg.yaml")
    ap.add_argument("--out", default="training/data/site_neg")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--interval", type=float, default=None)
    ap.add_argument("--pad", type=int, default=16)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(HERE, args.config), encoding="utf-8"))
    interval = args.interval or float(cfg.get("scan", {}).get("interval_s", 30))
    poly = json.load(open(os.path.join(HERE, args.zone), encoding="utf-8"))["points"]
    out = os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)

    det = ZoneTrashDetector(cfg, build_scorer(cfg.get("scorer", {})),
                            camera_id="mine", zone_id="z")
    cap = open_source(args.source, tries=5)
    if cap is None:
        return print(f"khong lay duoc khung tu {args.source}") or 1

    t_end = time.time() + args.hours * 3600
    n_scan = n_crop = 0
    print(f"đào ảnh âm · nhịp {interval}s · dừng sau {args.hours}h · -> {args.out}")
    print("VÙNG PHẢI KHÔNG CÓ RÁC trong suốt thời gian này\n")
    try:
        while time.time() < t_end:
            for _ in range(3):            # xả buffer, lấy khung mới nhất
                cap.grab()
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = open_source(args.source, tries=5)
                if cap is None:
                    break
                continue
            res = det.scan(frame, poly, (), now=time.time())
            n_scan += 1
            h, w = frame.shape[:2]
            for b in res.verify_boxes:
                x1 = max(0, int(b[0]) - args.pad)
                y1 = max(0, int(b[1]) - args.pad)
                x2 = min(w, int(b[2]) + args.pad)
                y2 = min(h, int(b[3]) + args.pad)
                crop = frame[y1:y2, x1:x2]        # SẠCH, chưa vẽ gì lên
                if crop.size and min(crop.shape[:2]) >= 24:
                    cf = b[4] if len(b) > 4 else 0.0
                    # Điểm tin cậy nằm trong TÊN FILE: lọc theo ngưỡng lúc train
                    # mà không phải đào lại, và nhìn tên là biết cái nào đáng lo.
                    cv2.imwrite(os.path.join(
                        out, f"neg_{int(time.time())}_{n_crop:05d}_c{cf:.2f}.jpg"), crop)
                    n_crop += 1
            if n_scan % 20 == 0:
                print(f"  {n_scan:5d} lượt · {n_crop:5d} ảnh âm "
                      f"({n_crop/max(1,n_scan):.2f} mỗi lượt)", flush=True)
            time.sleep(max(0.0, interval - res.ms / 1000.0))
    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
    print(f"\nxong: {n_scan} lượt, {n_crop} ảnh âm -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
