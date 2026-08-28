"""Do co vat model THAT SU nhin thay sau augmentation, khong phai co tho.

    python3 tools/aug_sizes.py --data <data.yaml> --imgsz 640 --n 300

Vi sao can: loai mot bo chi vi co vat tho la sai. Thu nho anh la phep bien doi
THAT (lay mau lai pixel that), khong phai du lieu gia. Mosaic ghep 4 anh lam vat
con 1/4, `scale` co them. Mot cai chai chiem 30% khung hoan toan co the thanh 5%.

Nen thuoc do dung phai la phan bo co vat SAU khi qua duong augmentation cua
ultralytics, chinh cai model an vao.

GIOI HAN phai noi ro: vat 300px thu ve 30px se SAC NET hon vat 30px chup that
bang cam CCTV (nhieu cam bien, nhoe chuyen dong, nen JPEG). Thu nho la thay the
MOT PHAN, khong phai hoan toan.
"""
from __future__ import annotations
import argparse, collections, sys

BANDS = [(0, 12), (12, 20), (20, 32), (32, 48), (48, 80), (80, 10 ** 9)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--mosaic", type=float, default=None)
    a = ap.parse_args()

    from ultralytics.data.dataset import YOLODataset
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    import yaml

    d = yaml.safe_load(open(a.data, encoding="utf-8"))
    root = d.get("path", ".")
    img_path = f"{root}/{d['train']}" if not str(d["train"]).startswith("/") else d["train"]

    hyp = get_cfg(DEFAULT_CFG)
    hyp.imgsz = a.imgsz
    if a.no_aug:
        hyp.mosaic = hyp.mixup = hyp.copy_paste = 0.0
        hyp.scale = hyp.translate = 0.0
        hyp.fliplr = hyp.flipud = 0.0
    else:
        if a.scale is not None:
            hyp.scale = a.scale
        if a.mosaic is not None:
            hyp.mosaic = a.mosaic

    ds = YOLODataset(img_path=img_path, imgsz=a.imgsz, augment=not a.no_aug,
                     hyp=hyp, rect=False, data=d, task="detect")
    cnt = collections.Counter()
    tot = 0
    for i in range(min(a.n, len(ds))):
        s = ds[i]
        bb = s["bboxes"]                    # xywh chuan hoa theo imgsz
        if bb is None or len(bb) == 0:
            continue
        for b in bb.tolist():
            px = max(b[2], b[3]) * a.imgsz
            for lo, hi in BANDS:
                if lo <= px < hi:
                    cnt[(lo, hi)] += 1
                    break
            tot += 1
    lbl = "KHONG augment" if a.no_aug else f"co augment (scale={hyp.scale}, mosaic={hyp.mosaic})"
    print(f"  {lbl}: {tot} hop tu {min(a.n, len(ds))} anh @ {a.imgsz}px")
    for lo, hi in BANDS:
        n = cnt[(lo, hi)]
        nm = f"{lo}-{hi}" if hi < 10 ** 9 else f">{lo}"
        print(f"    {nm:>7s}px {n:6d}  {100*n/max(1,tot):5.1f}%")
    mid = sum(cnt[b] for b in BANDS[2:5])
    print(f"    -> dai CCTV 20-80px: {100*mid/max(1,tot):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
