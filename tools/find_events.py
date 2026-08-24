"""Dò mốc vật bị bỏ lại trong ABODA — NHÃN, chưa phải kết quả.

    python3 tools/find_events.py

ABODA không cho số khung của sự kiện, chỉ nói "có vật bị bỏ lại". Muốn đo được
độ trễ thì phải biết vật xuất hiện lúc nào. Chỗ này dò bằng trừ nền rồi đòi
BỀN VỮNG: một đốm chỉ được coi là "vật bị bỏ lại" nếu nó còn ở đó tới cuối.

File này XUẤT ẢNH để soi bằng mắt. Nhãn tự dò mà không nhìn đã giết một hướng
đi trước đây (nhãn TACO lệch 1,6x làm recall đo ra 0% ở mọi cỡ), nên đừng dùng
số của nó trước khi mở ảnh ra xem.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ABODA = os.path.join(ROOT, "data", "aboda")


def frames_of(path: str, step: int):
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i, fr
        i += 1
    cap.release()


def gnorm(bgr: np.ndarray) -> np.ndarray:
    """Xám, TRỪ TRUNG BÌNH. Không có bước này thì một cú đổi sáng toàn cục làm
    cả khung khác nền, và mốc sự kiện dò ra là một cái hộp phủ kín khung —
    video6/7/8 đã ra đúng như vậy (555x478, 667x394, 715x478 px)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g - float(g.mean())


def find_event(path: str, step: int = 15, warm: int = 20, min_area: int = 120,
               max_area_frac: float = 0.15):
    """-> (khung sự kiện, hộp, ảnh lúc đó, ảnh nền) hoặc None.

    `max_area_frac`: vật bị bỏ lại KHÔNG chiếm 15% khung hình. Đốm to hơn thế
    là đổi sáng / camera dịch, phải loại chứ không phải nhận."""
    frs = list(frames_of(path, step))
    if len(frs) < warm + 10:
        return None
    ref = np.median(np.stack([f for _, f in frs[:warm]]), axis=0).astype(np.uint8)
    gref = gnorm(ref)

    masks = []
    for idx, fr in frs:
        d = np.abs(gnorm(fr) - gref)
        m = cv2.morphologyEx((d > 35).astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8))
        masks.append((idx, cv2.dilate(m, np.ones((7, 7), np.uint8))))

    # Vật BỎ LẠI = pixel khác nền ở gần như MỌI khung cuối. Người đi qua thì không.
    tail = np.mean([m for _, m in masks[-max(5, len(masks) // 5):]], axis=0)
    stay = (tail > 0.9).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(stay, 8)
    cap_area = max_area_frac * stay.size
    blobs = [i for i in range(1, n)
             if min_area <= st[i, cv2.CC_STAT_AREA] <= cap_area]
    if not blobs:
        return None
    b = max(blobs, key=lambda i: st[i, cv2.CC_STAT_AREA])
    x, y, w, h = (st[b, cv2.CC_STAT_LEFT], st[b, cv2.CC_STAT_TOP],
                  st[b, cv2.CC_STAT_WIDTH], st[b, cv2.CC_STAT_HEIGHT])

    # Khung sự kiện = khung ĐẦU TIÊN mà từ đó vùng này phủ liên tục tới hết.
    cov = [(idx, float(m[y:y + h, x:x + w].mean())) for idx, m in masks]
    ev = None
    for k in range(len(cov)):
        if all(c >= 0.35 for _, c in cov[k:]):
            ev = cov[k][0]
            break
    if ev is None:
        return None
    img = next(f for i, f in frs if i == ev)
    return ev, (int(x), int(y), int(w), int(h)), img, ref


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="test_cases/13_moc_su_kien_aboda.jpg")
    args = ap.parse_args()

    tiles, rows = [], []
    for i in range(1, 12):
        p = os.path.join(ABODA, f"video{i}.avi")
        if not os.path.exists(p):
            continue
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        r = find_event(p)
        if r is None:
            print(f"  video{i:<2}  KHONG do duoc moc su kien")
            rows.append((i, None, None, total, fps))
            continue
        ev, (x, y, w, h), img, _ref = r
        print(f"  video{i:<2}  su kien o khung {ev:5d}/{total} = {ev/fps:6.1f}s   "
              f"hop {w}x{h}px tai ({x},{y})")
        rows.append((i, ev, (x, y, w, h), total, fps))
        v = img.copy()
        cv2.rectangle(v, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(v, (0, 0), (v.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(v, f"video{i}  khung {ev} ({ev/fps:.0f}s)  {w}x{h}px",
                    (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        tiles.append(cv2.resize(v, (480, 360)))

    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)])
        os.makedirs(os.path.dirname(os.path.join(HERE, args.out)), exist_ok=True)
        cv2.imwrite(os.path.join(HERE, args.out), grid)
        print(f"\n-> {args.out}  ·  MO RA XEM truoc khi dung so nay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
