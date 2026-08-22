"""Vẽ ROI bằng chuột trên khung hình thật, lưu ra zone json + ảnh mốc.

    python3 tools/draw_zone.py --source rtsp://... --out zone.json

Chuột trái  thêm đỉnh        chuột phải  xoá đỉnh cuối
ENTER       lưu và thoát     c  xoá hết      r  lấy khung mới      q  thoát

Lưu kèm `<out>.anchor.png` — ảnh mốc 256px của khung lúc vẽ. live_view.py so
khung hiện tại với nó lúc khởi động để biết camera có bị xoay từ lần trước
không (đã dính: camera ngóc lên, cả vùng sàn tụt khỏi khung mà vẫn chạy).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|stimeout;8000000")

import cv2          # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grid import build_grid, poly_to_px      # noqa: E402
from core.reference import make_thumb             # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def grab(source: str):
    if os.path.isfile(source):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    f = None
    for _ in range(15):
        ok, f = cap.read()
        if ok:
            break
        time.sleep(0.3)
    cap.release()
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--cell-px", type=int, default=48, help="chỉ để xem trước số ô")
    args = ap.parse_args()

    frame = grab(args.source)
    if frame is None:
        return print("khong lay duoc khung tu", args.source) or 1
    H, W = frame.shape[:2]
    s = args.width / W
    base = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    pts: list[tuple[int, int]] = []

    def on_mouse(ev, x, y, _flags, _p):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif ev == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    win = "Ve ROI - ENTER de luu"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        d = base.copy()
        if len(pts) >= 3:
            ov = d.copy()
            cv2.fillPoly(ov, [np.array(pts, np.int32)], (0, 140, 0))
            d = cv2.addWeighted(ov, .25, d, .75, 0)
            norm = [[p[0] / d.shape[1], p[1] / d.shape[0]] for p in pts]
            grid = build_grid(poly_to_px(norm, W, H), W, H,
                              cell_px=args.cell_px, overlap=0.5)
            for c in grid.cells:
                cv2.rectangle(d, (int(c.x1 * s), int(c.y1 * s)),
                              (int(c.x2 * s), int(c.y2 * s)), (90, 90, 90), 1)
            info = (f"{len(pts)} dinh  |  {len(grid.cells)} o (cell={args.cell_px}px)"
                    f"  |  vat nho nhat bat duoc ~{args.cell_px // 2}px")
        else:
            info = f"{len(pts)} dinh - can it nhat 3"
        if pts:
            cv2.polylines(d, [np.array(pts, np.int32)], len(pts) >= 3, (0, 220, 255), 2)
            for i, p in enumerate(pts):
                cv2.circle(d, p, 4, (0, 220, 255), -1)
                cv2.putText(d, str(i + 1), (p[0] + 6, p[1] - 6), FONT, .45,
                            (0, 220, 255), 1, cv2.LINE_AA)
        cv2.rectangle(d, (0, 0), (d.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(d, info, (8, 17), FONT, .48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(d, "trai=them  phai=xoa  c=xoa het  r=khung moi  ENTER=luu  q=thoat",
                    (8, d.shape[0] - 10), FONT, .44, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.imshow(win, d)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            cv2.destroyAllWindows()
            return print("bo qua, khong luu") or 0
        if k == ord("c"):
            pts.clear()
        elif k == ord("r"):
            f2 = grab(args.source)
            if f2 is not None:
                base = cv2.resize(f2, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                frame = f2
        elif k in (13, 10):
            if len(pts) < 3:
                print("can it nhat 3 dinh")
                continue
            break
    cv2.destroyAllWindows()

    norm = [[round(p[0] / base.shape[1], 4), round(p[1] / base.shape[0], 4)] for p in pts]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"points": norm}, f)
    anchor = os.path.splitext(args.out)[0] + ".anchor.png"
    cv2.imwrite(anchor, make_thumb(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                .astype(np.uint8))
    grid = build_grid(poly_to_px(norm, W, H), W, H, cell_px=args.cell_px, overlap=0.5)
    print(f"luu {args.out}  ({len(norm)} dinh, {len(grid.cells)} o)")
    print(f"luu {anchor}  (moc chong lech camera)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
