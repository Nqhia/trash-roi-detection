"""Chấm điểm một lần chạy shadow mode.

Chỉ số nghiệm thu KHÔNG phải mAP hay accuracy, mà là:

    số cảnh báo NHẦM / camera / ngày      <- khách tắt tính năng nếu > 1-2
    recall theo SỰ KIỆN (không theo frame) <- một lần vùng bị bẩn = một sự kiện
    độ trễ phát hiện                       <- mục tiêu < 3 phút

    # clip sạch hoàn toàn: mọi cảnh báo đều là FP
    python3 tools/eval.py --scans runs/shadow/scans.csv

    # clip có rác thật: khai khoảng thời gian có rác
    python3 tools/eval.py --scans runs/shadow/scans.csv --truth truth.json

truth.json: {"intervals": [[120, 900], [3600, 5400]]}   # giây, cùng mốc với scans.csv
"""

from __future__ import annotations

import argparse
import csv
import json


def load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scans", required=True)
    ap.add_argument("--truth", help="JSON khoảng thời gian THẬT SỰ có rác")
    ap.add_argument("--grace-s", type=float, default=180.0,
                    help="cho phép báo muộn bao lâu sau khi rác xuất hiện")
    args = ap.parse_args()

    rows = load(args.scans)
    if not rows:
        print("file rỗng")
        return 1

    ts = [float(r["t_s"]) for r in rows]
    span = max(1e-9, ts[-1] - ts[0])
    days = span / 86400.0
    alerts = [float(r["t_s"]) for r in rows if r["alert"] == "1"]
    # Mốc mặt nạ nhiễu chín. Cảnh báo TRƯỚC mốc này gần như toàn là FP hệ thống
    # chưa kịp mute — gộp chung vào một con số FP/ngày sẽ ra kết quả bi quan giả.
    warm = next((float(r["t_s"]) for r in rows
                 if float(r.get("mask_progress") or 0) >= 1.0), None)
    dirty = sum(1 for r in rows if r["dirty"] == "1")
    dropped = sum(int(r.get("n_dropped") or 0) for r in rows)
    shifted = sum(1 for r in rows if float(r.get("shift_px") or 0) >= 3.0)

    print()
    print(f"  nội dung        {span/3600:.2f} giờ ({days:.3f} ngày), {len(rows)} lượt quét")
    print(f"  lượt 'bẩn'      {dirty}  ({dirty/len(rows)*100:.1f}%)")
    print(f"  cảnh báo        {len(alerts)}")

    intervals = []
    if args.truth:
        with open(args.truth, encoding="utf-8") as f:
            intervals = [tuple(map(float, iv)) for iv in json.load(f)["intervals"]]

    def rate_line(n_fp: int, d: float, what: str) -> None:
        """In FP/ngày, nhưng KHÔNG ngoại suy bừa từ clip ngắn.

        Quy 1 cảnh báo nhầm trên 3 phút thành '480 FP/ngày' là vô nghĩa thống kê
        mà đọc lại rất giật mình — dễ khiến người ta vứt bỏ một cấu hình tốt."""
        hours = d * 24
        if hours < 2.0:
            print(f"  ---> {n_fp} cảnh báo nhầm trên {hours:.2f}h {what}")
            print(f"       (KHÔNG quy ra FP/ngày: {hours:.2f}h quá ngắn để ngoại suy.")
            print("        Cần >= 24h video liên tục của site thật.)")
            return
        r = n_fp / max(1e-9, d)
        verdict = "ĐẠT" if r < 1 else "tạm được" if r < 2 else "KHÔNG ĐẠT"
        print(f"  ---> {r:.2f} FP / camera / ngày   [{verdict}]   ({what})")
        print("       ngưỡng khách chịu được: < 1-2 FP/camera/ngày")

    if not intervals:
        print()
        print("  Không khai truth -> coi toàn bộ clip là SẠCH:")
        rate_line(len(alerts), days, "toàn clip")
    else:
        def inside(t: float) -> bool:
            return any(a <= t <= b for a, b in intervals)

        tp = [t for t in alerts if inside(t)]
        fp = [t for t in alerts if not inside(t)]
        clean_s = span - sum(min(b, ts[-1]) - max(a, ts[0]) for a, b in intervals)
        clean_days = max(1e-9, clean_s / 86400.0)

        hit, lat = 0, []
        for a, b in intervals:
            firsts = [t for t in alerts if a <= t <= b + args.grace_s]
            if firsts:
                hit += 1
                lat.append(min(firsts) - a)

        print()
        print(f"  sự kiện rác     {len(intervals)}")
        print(f"  bắt được        {hit}/{len(intervals)}  "
              f"(recall sự kiện {hit/len(intervals)*100:.0f}%)")
        if lat:
            lat.sort()
            print(f"  độ trễ          trung vị {lat[len(lat)//2]/60:.1f} phút, "
                  f"xấu nhất {lat[-1]/60:.1f} phút   [mục tiêu < 3 phút]")
        print(f"  cảnh báo đúng   {len(tp)}")
        print(f"  cảnh báo nhầm   {len(fp)}")
        rate_line(len(fp), clean_days, "phần vùng sạch")

    if warm is not None and warm > ts[0]:
        after = [t for t in alerts if t >= warm]
        d_after = max(1e-9, (ts[-1] - warm) / 86400.0)
        print()
        print(f"  Mặt nạ nhiễu chín ở giây {warm:.0f} "
              f"({(warm-ts[0])/span*100:.0f}% đầu clip là giai đoạn học).")
        rate_line(len(after), d_after, "sau khi mặt nạ chín")
        print("  Đây mới là con số nên dùng để nghiệm thu; phần trước đó là")
        print("  cold start, và chính nó là lý do phải chạy shadow mode.")

    if dropped:
        print()
        print(f"  ! {dropped} ô bị trần max_score_per_scan cắt bỏ — recall báo cáo")
        print("    ở trên là LẠC QUAN. Tăng trần, hoặc siết change.thr cho ít ô đổi hơn.")
    if shifted:
        print(f"  ! {shifted} lượt phát hiện khung dịch chuyển — camera bị rung/xoay,")
        print("    tham chiếu và mặt nạ đã bị reset ngần ấy lần. Kiểm tra giá đỡ.")

    last = rows[-1]
    mp = float(last.get("mask_progress") or 0)
    if mp < 1.0:
        print()
        print(f"  ! mặt nạ nhiễu mới chín {mp*100:.0f}% — FP HỆ THỐNG (nắp cống,")
        print("    vệt sơn) chưa bị mute hết. Con số FP/ngày ở trên sẽ còn giảm.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
