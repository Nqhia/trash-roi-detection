"""Recall MỨC SỰ KIỆN và ĐỘ TRỄ trên video CCTV thật có vật bị bỏ lại.

    python3 tools/event_latency.py

Hai tiêu chí nghiệm thu chưa từng đo được trên chuỗi thật:
  * bắt được bao nhiêu phần trăm SỰ KIỆN (không phải bao nhiêu phần trăm VẬT
    trên ảnh tĩnh — 86% trong test_cases là con số mức vật, trên 6 khung rời)
  * bao nhiêu LƯỢT sau khi vật xuất hiện thì báo

CÁCH ĐỌC — đọc sai là ra kết luận ngược:

  Đo bằng LƯỢT, không bằng giây. Video ABODA chỉ dài 39-225 giây, quét nhịp 30s
  thật thì được 1-7 lượt, không đủ cho dwell 3. Nên ở đây quét dày trong thời
  gian VIDEO. Quy đổi: `lượt` x 30s = độ trễ ở hiện trường.

  Quy đổi này KHÔNG chính xác tuyệt đối. Ở nhịp dày, người đi qua nằm trong
  nhiều lượt liên tiếp nên có thể tự thoả `dwell` — cho báo sớm hơn thực tế; đổi
  lại họ cũng che vật lâu hơn. Con số này là XẤP XỈ có hướng lệch không rõ, dùng
  để bắt lỗi cỡ lớn (báo sau 20 lượt) chứ không phải để chốt SLA.

  Chỉ dùng video đã SOI BẰNG MẮT ở test_cases/13_moc_su_kien_aboda.jpg. Bốn
  video bị loại vì mốc sự kiện dò ra sai: video2/11 (hộp bám vào cạnh kiến
  trúc), video7/8 (cảnh cháy sáng, trừ nền bám vào dãy ghế), video6 (không dò
  được). Loại vì NHÃN sai, không phải vì pipeline làm dở — trộn chúng vào là
  chấm pipeline bằng nhãn rác.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from core.pipeline import ZoneTrashDetector      # noqa: E402
from core.scorers import build_scorer            # noqa: E402
from tools.find_events import find_event         # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
ABODA = os.path.join(ROOT, "data", "aboda")
FULL_POLY = [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]]

# Đã soi bằng mắt — xem docstring vì sao bốn cái kia bị loại.
USABLE = [1, 3, 4, 5, 9, 10]


def run_one(cfg: dict, path: str, step: int, ev_frame: int) -> dict:
    det = ZoneTrashDetector(cfg, build_scorer(cfg.get("scorer", {})),
                            camera_id=os.path.basename(path), zone_id="z")
    cap = cv2.VideoCapture(path)
    i = n_scan = 0
    ev_scan = None
    alert_scan = None
    raw_scan = None            # lượt đầu cổng đổi đã đủ bền vững (dwell xong)
    fp_before = 0
    min_hot = int(cfg.get("decide", {}).get("min_hot_cells", 1))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            res = det.scan(fr, FULL_POLY, (), now=float(n_scan) * 30.0)
            if ev_scan is None and i >= ev_frame:
                ev_scan = n_scan
            if ev_scan is not None and raw_scan is None and len(res.raw_hot) >= min_hot:
                raw_scan = n_scan
            if res.alert:
                if ev_scan is None:
                    fp_before += 1          # báo TRƯỚC khi vật xuất hiện = nhầm
                elif alert_scan is None:
                    alert_scan = n_scan
            n_scan += 1
        i += 1
    cap.release()
    return {"n_scan": n_scan, "ev_scan": ev_scan, "alert_scan": alert_scan,
            "raw_scan": raw_scan, "fp_before": fp_before}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/day_cfg.yaml")
    ap.add_argument("--step", type=int, default=15, help="số khung video mỗi lượt quét")
    ap.add_argument("--conf", type=float, default=None,
                    help="đè ngưỡng tầng xác nhận, để quét đánh đổi trễ/nhầm")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(HERE, args.config), encoding="utf-8"))
    # Mặt nạ nhiễu cần ~600 lượt mới chín — video dài nhất mới 450 lượt, bật lên
    # là đo một thứ chưa hoạt động rồi kết luận về nó.
    cfg["clutter"] = dict(cfg.get("clutter", {}), enabled=False)
    if args.conf is not None:
        cfg["verify"] = dict(cfg.get("verify", {}), conf=args.conf)
    dwell = int(cfg["decide"]["dwell_scans"])
    if args.quiet:
        import builtins
        _p = builtins.print
        builtins.print = lambda *a, **k: None
    print(f"{len(USABLE)} video · quét mỗi {args.step} khung · dwell {dwell} lượt")
    print("độ trễ tính bằng LƯỢT; x30s = độ trễ hiện trường (xấp xỉ, xem docstring)\n")

    hit = lat = fps_ = 0
    d_dwell = d_ver = 0
    for v in USABLE:
        p = os.path.join(ABODA, f"video{v}.avi")
        e = find_event(p)
        if e is None:
            print(f"  video{v:<2}  bỏ: không dò được mốc")
            continue
        r = run_one(cfg, p, args.step, e[0])
        fps_ += r["fp_before"]
        if r["alert_scan"] is None:
            print(f"  video{v:<2}  vật ở lượt {r['ev_scan']:>3}/{r['n_scan']:<3}  "
                  f"KHÔNG BÁO                báo nhầm trước đó: {r['fp_before']}")
        else:
            d = r["alert_scan"] - r["ev_scan"]
            hit += 1
            lat += d
            # Tách độ trễ: chờ cổng đổi đủ bền vững, và chờ detector chịu gật.
            dw = (r["raw_scan"] - r["ev_scan"]) if r["raw_scan"] is not None else None
            ve = (r["alert_scan"] - r["raw_scan"]) if r["raw_scan"] is not None else None
            if dw is not None:
                d_dwell += dw
                d_ver += ve
            print(f"  video{v:<2}  vật ở lượt {r['ev_scan']:>3}/{r['n_scan']:<3}  "
                  f"báo sau {d:>3} lượt (~{d*30/60:4.1f} phút)"
                  f"   [chờ dwell {dw}  ·  chờ detector {ve}]"
                  f"   báo nhầm trước: {r['fp_before']}")

    n = len(USABLE)
    print(f"\n  recall mức sự kiện : {hit}/{n} = {100*hit/n:.0f}%")
    if hit:
        print(f"  độ trễ trung bình  : {lat/hit:.1f} lượt = ~{lat/hit*30/60:.1f} phút"
              f"   (ngân sách thiết kế: {dwell} lượt = {dwell*30/60:.1f} phút)")
        print(f"     trong đó chờ cổng đổi : {d_dwell/hit:5.1f} lượt")
        print(f"     trong đó chờ DETECTOR : {d_ver/hit:5.1f} lượt")
    print(f"  báo nhầm trước khi vật xuất hiện: {fps_}")
    if args.quiet:
        import builtins
        builtins.print = _p
        print(f"  conf {cfg['verify']['conf']:.2f}  recall {hit}/{n}  "
              f"tre TB {lat/max(1,hit):5.1f} luot  bao nham {fps_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
