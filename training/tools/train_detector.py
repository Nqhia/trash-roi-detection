"""Fine-tune YOLO một lớp "trash" trên TACO + RoLID + UAVVaste.

    python3 tools/train_detector.py --epochs 40 --imgsz 960

Có mặt để trả lời sòng phẳng: YOLO-World và YOLO-COCO chưa từng thấy một tấm
ảnh rác nào, so chúng với hướng 1 (đã thu ô âm tại chỗ) là không công bằng.

imgsz để cao vì vật rất nhỏ — RoLID có vật 20-30px trên ảnh 1280. Hạ xuống 640
là vật teo còn 10-15px trước khi vào backbone. VRAM 4GB nên batch phải nhỏ.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/yolo/trash.yaml")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--name", default="trash_det")
    # Mosaic ghép 4 ảnh vào một khung -> mọi vật teo còn một nửa. Khi bộ train
    # đã được dựng để KHỚP THANG với lúc chạy thật (ô 320 phóng 2x), mosaic
    # mạnh sẽ phá đúng cái vừa căn. Giảm chứ không tắt hẳn: nó vẫn là cách rẻ
    # nhất để có nhiều vật/ảnh và nhiều bối cảnh.
    ap.add_argument("--mosaic", type=float, default=0.4)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"chưa có {args.data} — chạy tools/make_yolo_data.py trước")

    from ultralytics import YOLO
    m = YOLO(args.model)
    m.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            project="runs", name=args.name, exist_ok=True,
            patience=8, cache=False, workers=4,
            mosaic=args.mosaic,
            # Tắt hẳn mosaic ở các epoch cuối để model kết thúc trên đúng phân
            # bố mà nó sẽ gặp lúc chạy thật.
            close_mosaic=6,
            # Rác nằm dưới đất, lật dọc là vô nghĩa; lật ngang thì hợp lệ.
            flipud=0.0, fliplr=0.5,
            degrees=5.0, scale=0.4, translate=0.1)
    print(f"\nxong -> runs/{args.name}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
