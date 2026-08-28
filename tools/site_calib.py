"""Hiệu chỉnh pipeline cho MỘT NƠI LẮP MỚI — đo ra hằng số thay vì bê từ nơi cũ.

    python3 tools/site_calib.py --source 'rtsp://...' --zone config/zone.json --minutes 10
    python3 tools/site_calib.py --source sach.mp4    --zone config/zone.json

Vì sao bắt buộc: kiểm kê cho thấy gần như MỌI hằng số trong day_cfg.yaml được đo
trên đúng một camera (văn phòng eco, 1080p):

    cell_px 48        px tuyệt đối — 4K thì ô phủ nửa diện tích tương đối
    change.thr 6.0    ngưỡng trên SÀN NHIỄU của cảm biến đó, nén đó, đêm đó
    stab.min_px 3.0   comment gốc tự nhận "cửa sổ duy nhất trên chính dữ liệu này"

Camera khác -> cảm biến khác, nén khác, độ phân giải khác. Bê nguyên hằng số là
chạy với ngưỡng vô nghĩa mà không có lỗi nào báo. Công cụ này đo lại ba nhóm số
đó từ vài phút footage SẠCH (vùng không có rác, không ai đứng trong vùng) và in
sẵn khối YAML để dán vào config.

KHÔNG đo được ở đây, vẫn phải làm theo cách khác:
    verify.conf       cặp với model — đo bằng tools/bench_sweep.py
    FP/ngày           chỉ có shadow >=24h trả lời được
    dwell/guard       là THỜI GIAN và TỈ LỆ, chuyển được giữa camera
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

from core.grid import build_grid, poly_to_px                     # noqa: E402
from core.reference import (CellReference, estimate_warp, gray_small,  # noqa: E402
                            make_thumb, scene_shift)
from tools.run_video import iter_frames                          # noqa: E402


def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q)) if len(a) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="footage vùng lúc SẠCH")
    ap.add_argument("--zone", required=True)
    ap.add_argument("--minutes", type=float, default=10.0,
                    help="đo bao lâu khi nguồn là stream")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="nhịp lấy mẫu (giây) — dày hơn nhịp thật để đủ mẫu nhanh")
    a = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    poly = json.load(open(os.path.join(here, a.zone) if not os.path.isabs(a.zone)
                          else a.zone, encoding="utf-8"))["points"]

    ref = CellReference(alpha=0.05, change_thr=1e9)   # thr vô hiệu: chỉ đo điểm
    grid = None
    cell_px = None
    anchor_small = None
    prev_thumb = None
    scores: list[float] = []          # điểm đổi của Ô SẠCH — chính là sàn nhiễu
    ecc_spurious: list[float] = []    # |d| ECC trên camera ĐỨNG YÊN — nắn giả
    jitter: list[float] = []          # scene_shift giữa hai lượt liền nhau
    n_scan = 0
    limit_s = a.minutes * 60.0

    for t, frame in iter_frames(a.source, a.interval):
        if t > limit_s:
            break
        h, w = frame.shape[:2]
        if grid is None:
            # Ô đo theo TỈ LỆ khung, không phải px tuyệt đối: giữ đúng "phần
            # cảnh" mà một ô phủ như lúc mọi ngưỡng được đo trên 1080p/48px.
            cell_px = int(np.clip(round(48 * h / 1080 / 8) * 8, 24, 96))
            grid = build_grid(poly_to_px(poly, w, h), w, h,
                              cell_px=cell_px, overlap=0.5)
            print(f"khung {w}x{h} · ô đề xuất {cell_px}px · lưới {len(grid)} ô")

        thumb = make_thumb(frame)
        if prev_thumb is not None:
            jitter.append(scene_shift(prev_thumb, thumb, w, h))
        prev_thumb = thumb

        small = gray_small(frame, 4)
        if anchor_small is None:
            anchor_small = small
        else:
            est = estimate_warp(anchor_small, small)
            if est is not None:
                dx, dy, _deg, _W = est
                ecc_spurious.append(((dx * 4) ** 2 + (dy * 4) ** 2) ** 0.5)

        for c in grid.cells:
            patch = frame[c.y1:c.y2, c.x1:c.x2]
            if patch.size == 0:
                continue
            desc = ref.describe(patch)
            if ref.has(c.id):
                scores.append(ref.change(c.id, desc))
            ref.observe_clean(c.id, desc)   # footage sạch -> mọi ô đều được học
        n_scan += 1
        if n_scan % 20 == 0:
            print(f"  {n_scan} lượt · {len(scores)} mẫu điểm nền", flush=True)

    if n_scan < 10 or not scores:
        print(f"chỉ có {n_scan} lượt — cần >=10 lượt sạch để nói được gì")
        return 1

    thr = max(4.0, min(10.0, pct(scores, 99.9) * 1.5))
    min_px = float(np.clip(pct(ecc_spurious, 99) * 1.5, 2.0, 6.0))
    j99 = pct(jitter, 99)

    print(f"\n== KẾT QUẢ ({n_scan} lượt, {len(scores)} mẫu ô sạch) ==")
    print(f"  sàn nhiễu điểm đổi : p50={pct(scores,50):.2f}  p99={pct(scores,99):.2f}"
          f"  p99.9={pct(scores,99.9):.2f}   -> change.thr = {thr:.1f}")
    print(f"  ECC nắn giả (yên)  : p50={pct(ecc_spurious,50):.2f}px"
          f"  p99={pct(ecc_spurious,99):.2f}px   -> stabilize.min_px = {min_px:.1f}")
    print(f"  scene_shift jitter : p99={j99:.2f}px  (ngưỡng vứt nền 12px"
          f"{' — QUÁ SÁT, xem lại!' if j99 > 6 else ' — dư biên, ổn'})")
    if thr >= 9.5:
        print("  !! sàn nhiễu rất cao — camera nhiễu/nén mạnh; vật mờ nhạt sẽ khó"
              " qua cổng đổi. Cân nhắc giảm nén camera trước khi tin ngưỡng này.")

    print("\n== DÁN VÀO CONFIG ==")
    print(f"grid:\n  cell_px: {cell_px}\n  overlap: 0.5")
    print(f"change:\n  thr: {thr:.1f}")
    print(f"stabilize:\n  min_px: {min_px:.1f}")
    print("\nCÒN LẠI PHẢI TỰ LÀM:")
    print("  - verify.conf: cặp với model — quét bằng tools/bench_sweep.py, ghi tuned_for")
    print("  - mặt nạ nhiễu: học lại từ đầu ở site mới (~5h ô nóng liên tục mới mute)")
    print("  - FP/ngày: shadow >=24h bằng tools/run24h.sh trước khi bật cảnh báo thật")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
