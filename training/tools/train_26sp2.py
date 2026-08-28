"""Train yolo26s-p2 tren may 5090.  Chay TREN MAY DO, khong chay o may nay.

    python train_26sp2.py --data /root/.../trash.yaml --epochs 120

VI SAO yolo26s-p2 CHU KHONG PHAI yolo11n/26s
--------------------------------------------
Do duoc tren 373 vat: recall theo canh vat la 28% (<12px), 69%, 67%, 75%, 63%,
83% (>80px). Cho yeu la dai 20-48px — dung dai CCTV can. yolo11n va yolo26s deu
co tang phat hien nho nhat o stride 8: o dac trung phu 8px, nen vat 24px chi rong
3 o. Bien the -p2 them tang stride 4 -> 6 o, MA KHONG ton them tham so
(9,77M so voi 10,01M cua yolo26s).

KHONG co trong so pretrained cho -p2 (404 tren kho ultralytics). Nap tu
yolo26s.pt duoc 360/902 lop = 62% tham so (chu yeu backbone); co va dau phai hoc
lai. Do la ly do can nhieu epoch hon binh thuong.

scale=0.9 chu khong phai 0.5: do duoc tren bo anh chup gan, augment mac dinh chi
dua 3,5% -> 4,1% so hop vao dai lam viec, con scale manh dua len 14,9%.
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo26s-p2.yaml")
    ap.add_argument("--weights", default="yolo26s.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--name", default="trash26sp2")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--scale", type=float, default=0.9)
    a = ap.parse_args()

    from ultralytics import YOLO
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)} · torch {torch.__version__}")
    m = YOLO(a.model)
    if a.weights:
        m = m.load(a.weights)          # chuyen phan khop duoc tu 26s
    m.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        workers=a.workers, project=a.project, name=a.name,
        device=0, cache=False, amp=True,
        # anh nen (khong nhan) chiem 21% tap — de ultralytics dung chung
        scale=a.scale,          # co manh: dua vat ve dung dai lam viec
        mosaic=1.0, close_mosaic=15,
        fliplr=0.5, flipud=0.15,        # camera treo cao: vat co the o moi huong
        degrees=10.0, translate=0.15,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
        erasing=0.2,
        patience=30, val=True, plots=True, save_period=20,
    )
    print("TRAIN XONG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
