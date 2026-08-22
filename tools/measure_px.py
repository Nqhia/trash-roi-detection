"""BƯỚC 0 — đo ngân sách pixel. Chạy TRƯỚC khi viết/train bất cứ thứ gì.

Cả dự án bị quyết định bởi một con số: rác nhỏ nhất chiếm bao nhiêu pixel ở
MÉP XA NHẤT của vùng khoanh. Đo sai thì mọi lựa chọn model đều vô nghĩa.

Cách đo: đặt một vật có kích thước biết trước (tờ A4 cạnh dài 29.7cm) ở mép xa
nhất của vùng định khoanh, chụp main stream, rồi click hai đầu vật đó.

    python3 tools/measure_px.py --image far_edge.jpg
    python3 tools/measure_px.py --source rtsp://... --ref-cm 29.7
    python3 tools/measure_px.py --image a.jpg --p1 812,455 --p2 849,461 --ref-cm 29.7

KHÔNG dùng bảng lý thuyết thay cho phép đo này: bảng giả định vật vuông góc
trục quang, còn mặt đất nhìn xiên bị nén dọc nên thực tế luôn xấu hơn.
"""

from __future__ import annotations

import argparse
import sys

import cv2

BOTTLE_CM = 22.0        # vỏ chai nhựa 500ml nằm ngang
BAG_CM = 35.0           # túi nilon cỡ bàn tay
SACK_CM = 50.0          # bao tải / đống rác nhỏ


def grab_frame(args) -> "cv2.Mat":
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"không đọc được ảnh: {args.image}")
        return img
    cap = cv2.VideoCapture(args.source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"không đọc được khung hình từ: {args.source}")
    return frame


def pick_two_points(img) -> tuple[tuple[int, int], tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    view = img.copy()
    win = "Click 2 dau vat tham chieu (r=lam lai, q=thoat)"

    def on_mouse(ev, x, y, _flags, _param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
            cv2.circle(view, (x, y), 4, (0, 0, 255), -1)
            if len(pts) == 2:
                cv2.line(view, pts[0], pts[1], (0, 255, 0), 2)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"):
            cv2.destroyAllWindows()
            sys.exit("huỷ")
        if k == ord("r"):
            pts.clear()
            view = img.copy()
        if len(pts) == 2 and k in (13, 32):
            break
        if len(pts) == 2 and cv2.waitKey(400) & 0xFF in (13, 32):
            break
    cv2.destroyAllWindows()
    return pts[0], pts[1]


def parse_pt(s: str) -> tuple[int, int]:
    x, y = s.split(",")
    return int(x), int(y)


def report(px_per_cm: float, frame_w: int, frame_h: int) -> int:
    px_m = px_per_cm * 100.0
    bottle = px_per_cm * BOTTLE_CM
    bag = px_per_cm * BAG_CM
    sack = px_per_cm * SACK_CM

    print()
    print(f"  khung hình        {frame_w} x {frame_h}")
    print(f"  mật độ            {px_per_cm:.2f} px/cm  ({px_m:.0f} px/m)")
    print()
    print("  kích thước biểu kiến tại vị trí đo:")
    print(f"    vỏ chai 22cm     {bottle:5.1f} px")
    print(f"    túi nilon 35cm   {bag:5.1f} px")
    print(f"    bao tải 50cm     {sack:5.1f} px")
    print()
    # Tham chiếu EN 62676-4: 62.5 px/m = detection, 125 = recognition, 250 = identification
    grade = ("dưới mức 'detection'" if px_m < 62.5 else
             "mức 'detection'" if px_m < 125 else
             "mức 'recognition'" if px_m < 250 else "mức 'identification'")
    print(f"  chuẩn EN 62676-4: {grade}")
    print()

    if bottle >= 40:
        cell = 0
        print("  ==> VỎ CHAI THOẢI MÁI (>=40px).")
        print("      Không cần chia ô. Hỏi VLM/classifier thẳng trên cả vùng.")
        print("      Đặt grid.cell_px lớn (>=160) hoặc bỏ hẳn lưới.")
        rc = 0
    elif bottle >= 25:
        cell = int(round(2.2 * bottle / 16.0)) * 16
        cell = max(32, min(192, cell))
        print(f"  ==> VỎ CHAI BIÊN ({bottle:.0f}px). Bắt buộc chia ô.")
        print(f"      Đặt grid.cell_px: {cell}   (~2.2x chiều dài vật)")
        print("      Bật change.enabled: true — ở cỡ này ngoại hình không đủ")
        print("      phân biệt, phải dựa vào 'chỗ này trước đây không có gì'.")
        rc = 0
    else:
        cell = 0
        print(f"  ==> KHÔNG KHẢ THI với vỏ chai ({bottle:.0f}px < 25px).")
        print("      Không có model nào cứu được. Ba lối ra:")
        print("        (a) thu hẹp vùng khoanh lại gần camera hơn")
        print(f"        (b) đổi ống kính hẹp hơn / lên 4K (cần x{25/max(bottle,1):.1f})")
        print(f"        (c) hạ mức nhỏ nhất: túi nilon {bag:.0f}px, bao tải {sack:.0f}px")
        rc = 2

    if rc == 0:
        print()
        print(f"  Ban đêm/hồng ngoại cần gấp đôi mật độ ({px_m*2:.0f} px/m) —")
        print("  nhớ đo lại bằng ảnh IR trước khi cam kết chạy 24/7.")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="ảnh chụp sẵn (main stream, KHÔNG phải sub-stream)")
    src.add_argument("--source", help="video/RTSP — lấy khung đầu tiên")
    ap.add_argument("--ref-cm", type=float, default=29.7,
                    help="chiều dài thật của vật tham chiếu, cm (mặc định A4 = 29.7)")
    ap.add_argument("--p1", help="điểm đầu 'x,y' (bỏ qua thì mở cửa sổ click)")
    ap.add_argument("--p2", help="điểm cuối 'x,y'")
    args = ap.parse_args()

    img = grab_frame(args)
    h, w = img.shape[:2]
    if w * h <= 704 * 576:
        print(f"  ! CẢNH BÁO: ảnh {w}x{h} — trông như sub-stream. Vỏ chai sẽ chỉ")
        print("    còn ~10px và không phương án nào chạy được. Dùng main stream.")

    if args.p1 and args.p2:
        p1, p2 = parse_pt(args.p1), parse_pt(args.p2)
    else:
        p1, p2 = pick_two_points(img)

    dist_px = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    if dist_px < 2:
        sys.exit("hai điểm trùng nhau")
    if args.ref_cm <= 0:
        sys.exit("--ref-cm phải > 0")

    print(f"\n  vật tham chiếu    {args.ref_cm}cm  =  {dist_px:.1f} px")
    return report(dist_px / args.ref_cm, w, h)


if __name__ == "__main__":
    raise SystemExit(main())
