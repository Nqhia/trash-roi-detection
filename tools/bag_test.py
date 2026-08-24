"""Ca THAT: tui ni long nguoi vut vao vung luc 10:14 ngay 24/08.

    python3 tools/bag_test.py

Ca nay quan trong vi no la lan dau co rac THAT tren camera THAT, va he thong
luc do KHONG BAO trong 4,5 gio. Cong doi thay ngay (o doi 60->76, du dwell luc
10:15) nhung tang xac nhan bac bo sach moi luot.

Khung lay tu chinh lan chay 43 gio, khong dung, khong ghep:
  truoc: alert_147766.01.jpg  (~10:03, san sach)
  sau  : alert_163013.64.jpg  (~14:10, co tui)

Vung thu hep lai quanh cho co tui. Ly do: anh canh bao da bi ve o do len tren,
va o do nam o CHO KHAC trong khung. Neu lay ca vung thi mau do do tu no thanh
"thay doi" va lam ban phep do. Quanh cho co tui thi hai khung deu sach lop ve.
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

FRAMES = os.path.join(HERE, "runs", "shadow24_baseline_khong_leothang", "frames")
# MOT khung moi ben, va hai khung phai GAN NHAU VE THOI GIAN. Ban dau toi luan
# phien 3 khung chup cach nhau hang gio: anh sang khac han nen 68-80/80 o deu
# "doi", guard doi sang ban moi luot, cong doi khong bao gio dat dwell — do ra
# "khong bao" ma chang lien quan gi toi cai tui.
CLEAN = "alert_147766.01.jpg"          # ~10:03, san sach
BAG_RAW = "../../khung_goc.jpg"        # ~10:20, co tui (khung tho, khong ve gi)
# vung quanh cho co tui, toa do CHUAN HOA (khung 1920x1080)
ZONE = [[0.365, 0.815], [0.495, 0.815], [0.495, 0.995], [0.365, 0.995]]


def run(cfg: dict, tag: str, n: int = 30) -> int | None:
    det = ZoneTrashDetector(cfg, build_scorer(cfg.get("scorer", {})),
                            camera_id=tag, zone_id="z")
    t = 0.0
    clean = cv2.imread(os.path.join(FRAMES, CLEAN))
    bag = cv2.imread(os.path.join(FRAMES, BAG_RAW))
    for _ in range(8):                       # dung nen tu khung SACH
        det.scan(clean, ZONE, (), now=t)
        t += 30.0
    for k in range(n):                       # tui da nam trong vung
        res = det.scan(bag, ZONE, (), now=t)
        t += 30.0
        if res.alert:
            return k
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/day_cfg.yaml")
    args = ap.parse_args()
    for f in (CLEAN, BAG_RAW):
        if not os.path.exists(os.path.join(FRAMES, f)):
            print(f"thiếu khung {f} — cần dữ liệu lần chạy 43h")
            return 1

    base = yaml.safe_load(open(os.path.join(HERE, args.config), encoding="utf-8"))
    base["clutter"] = dict(base.get("clutter", {}), enabled=False)
    off = dict(base, verify=dict(base["verify"], escalate_after=0))

    for tag, cfg in (("KHÔNG leo thang", off), ("CÓ leo thang", base)):
        k = run(cfg, tag[:6])
        print(f"  {tag:18s} -> " + (f"BÁO sau {k} lượt (~{k*30/60:.1f} phút)"
                                    if k is not None else "KHÔNG BÁO trong 30 lượt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
