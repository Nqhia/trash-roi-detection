"""Tóm tắt một lần chạy shadow: cảnh báo xảy ra LÚC NÀO và Ở ĐÂU.

    python3 tools/shadow_report.py --scans runs/shadow/scans.csv

`eval.py` trả lời "bao nhiêu FP/ngày". Tool này trả lời câu khác, cần cho lúc
đang chạy: cảnh báo có **cụm lại theo thời gian** không (thì là hoạt động của
người, không phải FP hệ thống) và có **cụm lại theo vị trí** không (thì là một
vật cố định, mặt nạ nhiễu sẽ lo).

Phải dùng csv reader chứ không tách bằng dấu phẩy: cột `hot_cells` chứa dấu
phẩy bên trong dấu nháy ("4,2 5,3") nên awk -F, cắt sai và im lặng cho ra số vô nghĩa.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # nhan NHIEU file: keepalive doi ten scans.csv moi lan bat lai, doc mot
    # file la bao cao thieu mat cac doan truoc ma khong noi gi.
    ap.add_argument("--scans", required=True, nargs="+")
    ap.add_argument("--bin-min", type=float, default=30.0)
    args = ap.parse_args()

    rows = []
    for f in sorted(args.scans):
        part = list(csv.DictReader(open(f, newline="", encoding="utf-8")))
        if len(args.scans) > 1:
            print(f"  {f}: {len(part)} lượt")
        rows += part
    # Moi lan bat lai, t_s dem lai tu 0 -> phai noi lien mach, khong thi bieu do
    # theo gio nhay lung tung ma khong ai thay.
    off, prev, fixed = 0.0, -1.0, []
    for r in rows:
        t = float(r["t_s"])
        if t < prev:
            off = prev + 30.0
        prev = t
        r["t_s"] = str(t + off)
        fixed.append(r)
    rows = fixed
    if not rows:
        return print("chưa có lượt nào") or 1
    n = len(rows)
    dur = float(rows[-1]["t_s"])
    al = [r for r in rows if r["alert"] == "1"]

    print(f"{n} lượt · {dur/3600:.2f} giờ · "
          f"{sum(float(r['ms']) for r in rows)/n:.0f} ms/lượt")
    print(f"cảnh báo {len(al)} · lượt bẩn {sum(1 for r in rows if r['dirty']=='1')} "
          f"· lượt có ô nóng {sum(1 for r in rows if int(r['n_hot'])>0)}")
    print(f"ô đổi trung bình {sum(int(r['n_changed']) for r in rows)/n:.1f}/"
          f"{rows[0]['n_cells']}")
    vd = sum(int(r["n_verify_dropped"]) for r in rows)
    vb = sum(int(r["n_verify_boxes"]) for r in rows)
    print(f"detector: bác bỏ {vd} ô · xác nhận {vb} hộp "
          f"({100*vd/max(1,vd+vb):.0f}% bị bác bỏ)")
    # KHONG doc dong cuoi: mat na bi reset nhieu lan trong mot lan chay, doc
    # dong cuoi thi tuy may man ma ra 0% hay 100% — da bi con so nay danh lua
    # mot lan (bao "chin 0,0%" trong khi no chin suot 87% lan chay).
    mp = [float(r["mask_progress"]) for r in rows]
    ripe = sum(1 for v in mp if v >= 1.0)
    resets = sum(1 for i in range(1, len(mp)) if mp[i] < mp[i - 1] - 1e-9)
    print(f"mặt nạ nhiễu: chín ở {ripe}/{len(mp)} lượt ({100*ripe/max(1,len(mp)):.0f}%)"
          f" · bị reset {resets} lần · lượt trúng ô đã mute "
          f"{sum(int(r['n_muted_hit']) for r in rows)}")

    print(f"\n--- phân bố theo {args.bin_min:.0f} phút ---")
    b = args.bin_min * 60
    agg = {}
    for r in rows:
        k = int(float(r["t_s"]) // b)
        a = agg.setdefault(k, [0, 0, 0, 0.0])
        a[0] += 1
        a[1] += int(r["n_hot"]) > 0
        a[2] += r["alert"] == "1"
        a[3] += int(r["n_changed"])
    for k in sorted(agg):
        c, h, a, ch = agg[k]
        bar = "#" * a
        print(f"  {k*b/3600:4.1f}-{(k+1)*b/3600:4.1f}h  {c:3d} lượt · "
              f"ô đổi TB {ch/c:5.1f} · {h:2d} lượt có ô nóng · "
              f"{a} cảnh báo {bar}")

    if al:
        print("\n--- vị trí ô nóng lúc cảnh báo (hàng,cột) ---")
        cnt = Counter()
        for r in al:
            cells = [c for c in r["hot_cells"].split() if c]
            cnt.update(cells)
            print(f"  t={float(r['t_s'])/3600:.2f}h  {len(cells):2d} ô  "
                  f"bác bỏ {r['n_verify_dropped']:>3s} · xác nhận {r['n_verify_boxes']}"
                  f"  |  {' '.join(cells[:10])}")
        rep = [c for c, k in cnt.items() if k >= 2]
        print(f"\n  ô xuất hiện ở >=2 cảnh báo: {len(rep)}/{len(cnt)}")
        if len(rep) > 0.4 * len(cnt):
            print("  -> cảnh báo CỤM LẠI cùng một chỗ: nhiều khả năng là MỘT vật")
            print("     cố định / hay bị xê dịch, không phải FP rải rác.")
        else:
            print("  -> cảnh báo RẢI RÁC nhiều chỗ khác nhau.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
