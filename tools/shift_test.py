"""Đo NGƯỠNG CHỊU LỆCH của một camera + vùng cụ thể.

    python3 tools/shift_test.py --source rtsp://... --zone z.json --config cfg.yaml

Câu hỏi cần trả lời trước khi treo camera ngoài đường: lệch bao nhiêu px thì
hỏng? Có ba chuyện khác nhau, tool này tách bạch cả ba:

  1. lệch NHỎ hơn `scene_shift.thr_px`  -> cổng dịch khung KHÔNG bắt, ô đổi thật
     -> sau `dwell_scans` lượt là báo nhầm. Đây là vùng nguy hiểm.
  2. lệch LỚN hơn ngưỡng               -> pipeline tự nạp lại nền, không báo
     nhầm, chỉ mất mặt nạ nhiễu.
  3. lệch xảy ra lúc KHÔNG chạy        -> anchor check của live_view.py lo.

Kết quả phụ thuộc kết cấu bề mặt trong vùng: sàn/đường trơn chịu lệch tốt hơn
nhiều so với vùng có vạch kẻ, mép bó vỉa, lá cây. Nên phải đo TẠI CHỖ.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|stimeout;8000000")

import cv2          # noqa: E402
import numpy as np  # noqa: E402
import yaml         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import ZoneTrashDetector   # noqa: E402
from core.scorers import build_scorer         # noqa: E402

SHIFTS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
ROTS = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]


def grab_frames(source: str, n: int, gap: float) -> list:
    if os.path.isdir(source):
        fs = sorted(f for f in os.listdir(source)
                    if os.path.splitext(f)[1].lower() in (".jpg", ".png"))[:n]
        return [cv2.imread(os.path.join(source, f)) for f in fs]
    is_file = os.path.isfile(source)
    cap = cv2.VideoCapture(source) if is_file else cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    out = []
    if is_file:
        # File thì lấy mẫu theo CHỈ SỐ KHUNG. Chờ theo đồng hồ thật sẽ đọc hết
        # video trong vài giây rồi báo "khong du khung".
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, int(fps * gap))
        if total and total < stride * n:
            stride = max(1, total // (n + 1))
        i = 0
        while len(out) < n:
            ok, f = cap.read()
            if not ok:
                break
            if i % stride == 0:
                out.append(f)
            i += 1
    else:
        t = 0.0
        while len(out) < n:
            ok, f = cap.read()
            if not ok:
                break
            now = time.time()
            if now >= t:
                out.append(f)
                t = now + gap
    cap.release()
    return out


def warp(frame, dx: float, dy: float, deg: float = 0.0):
    """Dịch + xoay quanh tâm. BORDER_REFLECT để mép không thành dải đen giả."""
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", type=int, default=10, help="số khung để dựng nền")
    ap.add_argument("--after", type=int, default=10,
                    help="số lượt quét SAU khi lệch (phải > dwell + confirm.m)")
    ap.add_argument("--gap", type=float, default=1.0)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.zone, encoding="utf-8") as f:
        poly = json.load(f)["points"]
    frames = grab_frames(args.source, args.frames + args.after, args.gap)
    if len(frames) < args.frames + 3:
        return print(f"chi lay duoc {len(frames)} khung tu {args.source}") or 1
    seed, after = frames[:args.frames], frames[args.frames:]
    print(f"{len(seed)} khung dung nen + {len(after)} khung sau khi lech, "
          f"khung {seed[0].shape[1]}x{seed[0].shape[0]}")

    thr = float(cfg.get("scene_shift", {}).get("thr_px", 12.0))
    dwell = int(cfg.get("decide", {}).get("dwell_scans", 5))
    min_hot = int(cfg.get("decide", {}).get("min_hot_cells", 2))

    # Giữ NGUYÊN config thật: câu hỏi là hệ thống deploy có báo nhầm không, nên
    # cả guard đổi-sáng-toàn-cục lẫn cổng dịch khung phải bật. Chỉ tắt mặt nạ
    # nhiễu vì nó cần 600 lượt mới chín, không liên quan ở thang thời gian này.
    run_cfg = json.loads(json.dumps(cfg))
    run_cfg["clutter"] = dict(run_cfg.get("clutter", {}), enabled=False)

    def measure(dx, dy, deg):
        """Lệch RỒI GIỮ NGUYÊN trong `len(after)` lượt -> có bắn cảnh báo không?

        Đếm ô đổi ở lượt đầu là chưa đủ: ô đổi phải trụ qua dwell rồi qua
        ConfirmGate mới thành cảnh báo, mà trước đó guard toàn cục còn có thể
        nuốt cả lượt. Chỉ con số cảnh báo cuối cùng mới trả lời được câu hỏi.
        """
        tmp = tempfile.mkdtemp(prefix="shift_test_")   # dùng chung -> nạp nhầm nền cũ
        det = ZoneTrashDetector(run_cfg, build_scorer({"kind": "constant", "value": 0.0}),
                                camera_id="t", zone_id="t", state_dir=tmp)
        for i, f in enumerate(seed):
            det.scan(f, poly, [], now=float(i))
        first, gc, sg, alerts = None, 0, 0, 0
        for k, f in enumerate(after):
            r = det.scan(warp(f, dx, dy, deg), poly, [], now=float(len(seed) + k))
            if first is None:
                first = r
            gc += bool(r.global_change)
            sg += bool(r.scene_shift_px >= thr)
            alerts += bool(r.alert)
        shutil.rmtree(tmp, ignore_errors=True)
        return first, gc, sg, alerts

    base, *_ = measure(0, 0, 0.0)
    n = base.n_cells
    print(f"\nvung {n} o, cell={cfg['grid']['cell_px']}px, dwell={dwell}, "
          f"min_hot={min_hot}, scene_shift.thr_px={thr:.0f}, {len(after)} luot sau khi lech")
    print(f"nen dung: {base.n_changed} o doi khi KHONG lech")
    hdr = "  lech     o doi   %vung   guard toan cuc   cong dich   BAO NHAM"
    for title, rows in (("dich ngang", [(d, 0, 0.0) for d in SHIFTS]),
                        ("xoay quanh tam", [(0, 0, a) for a in ROTS])):
        print(f"\n{title}\n{hdr}\n  " + "-" * 62)
        for dx, dy, deg in rows:
            r, gc, sg, al = measure(dx, dy, deg)
            lbl = f"{dx:3d}px" if deg == 0 else f"{deg:4.1f}d"
            print(f"  {lbl}    {r.n_changed:4d}    {100.0*r.n_changed/max(1,n):5.1f}%"
                  f"   {gc:2d}/{len(after)} luot      {sg:2d}/{len(after)}"
                  f"      {'CO ' + str(al) if al else 'khong'}")
    print("\nDoc: 'BAO NHAM' la cot duy nhat co y nghia. Hai cot guard cho biet")
    print("cai nao da chan - lech vua du de KHONG guard nao bat moi la nguy hiem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
