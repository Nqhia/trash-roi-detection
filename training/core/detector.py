"""Detector open-vocabulary sau chung một interface.

Hai backend, chọn bằng config:

  yoloworld : ultralytics YOLOWorld. Nhanh (vài ms/ô trên GPU), nhận prompt
              bằng danh sách chữ. Kiến trúc YOLO nên vẫn kém ở vật rất nhỏ —
              chính vì thế mới có tiler phóng to.
  owlv2     : transformers Owlv2. Chậm hơn nhiều nhưng thiết kế cho zero-shot
              thật sự, thường tốt hơn hẳn ở vật lạ / không thuộc lớp COCO.

Cả hai đều KHÔNG cần một byte dữ liệu train nào của rác — đó là điểm cả hướng
này nhắm tới. Đổi lại, chất lượng phụ thuộc vào chữ trong prompt, mà prompt
thì phải đo chứ không đoán được.
"""

from __future__ import annotations

import os

import numpy as np

# Nạp ở đây chứ không nạp lười bên trong hàm: hybrid_test.py phải tráo `core`
# giữa hai project, nên import tương đối chạy muộn sẽ không tìm thấy package nữa.
from .tiler import build_tiles, nms
from .zone import box_center_in_zone

# Prompt mặc định. Cố ý mô tả VẬT + BỐI CẢNH ("trên mặt đất") chứ không chỉ
# "rác": model open-vocab bám sát chữ, "trash" đơn lẻ hay khớp vào thùng rác.
# Đo được: câu dài kiểu "a plastic bottle on the ground" làm YOLO-World gần như
# câm (1 phát hiện / 12 ảnh rác to rõ), danh từ ngắn cho 5. Cả hai model đều bám
# chữ rất sát, nên prompt phải đo chứ không đoán được.
DEFAULT_PROMPTS = [
    "litter", "trash", "plastic bottle", "plastic bag",
    "can", "paper cup", "wrapper", "cardboard box",
]


class Detection:
    __slots__ = ("box", "score", "label")

    def __init__(self, box, score, label):
        self.box, self.score, self.label = box, float(score), label

    def size_px(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(x2 - x1, y2 - y1)


class YoloWorldDetector:
    name = "yoloworld"

    def __init__(self, weights: str = "yolov8s-worldv2.pt",
                 prompts: list | None = None, conf: float = 0.05,
                 device: str | None = None):
        from ultralytics import YOLOWorld
        self.prompts = list(prompts or DEFAULT_PROMPTS)
        self.m = YOLOWorld(weights)
        self.m.set_classes(self.prompts)
        self.conf = conf
        self.device = device

    def detect(self, images: list) -> list[list[Detection]]:
        if not images:
            return []
        kw = {"conf": self.conf, "verbose": False}
        if self.device:
            kw["device"] = self.device
        out = []
        for r in self.m.predict(images, **kw):
            dets = []
            for b in r.boxes:
                dets.append(Detection(tuple(b.xyxy[0].tolist()), float(b.conf),
                                      self.prompts[int(b.cls)]))
            out.append(dets)
        return out


class Owlv2Detector:
    name = "owlv2"

    def __init__(self, weights: str = "google/owlv2-base-patch16-ensemble",
                 prompts: list | None = None, conf: float = 0.08,
                 device: str | None = None):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        self.prompts = list(prompts or DEFAULT_PROMPTS)
        self.proc = Owlv2Processor.from_pretrained(weights)
        self.m = Owlv2ForObjectDetection.from_pretrained(weights)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = self.m.to(self.device).eval()
        self.conf = conf

    def detect(self, images: list) -> list[list[Detection]]:
        if not images:
            return []
        torch = self.torch
        # ascontiguousarray bắt buộc: lát cắt ::-1 cho stride âm, torch không nhận.
        rgb = [np.ascontiguousarray(im[:, :, ::-1]) for im in images]
        texts = [self.prompts] * len(rgb)
        inp = self.proc(text=texts, images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.m(**inp)
        sizes = torch.tensor([im.shape[:2] for im in rgb]).to(self.device)
        res = self.proc.post_process_grounded_object_detection(
            outputs=out, target_sizes=sizes, threshold=self.conf)
        dets = []
        for r in res:
            cur = []
            for box, sc, lb in zip(r["boxes"].tolist(), r["scores"].tolist(),
                                   r["labels"].tolist()):
                cur.append(Detection(tuple(box), sc, self.prompts[int(lb)]))
            dets.append(cur)
        return dets


class CocoDetector:
    """YOLO thường (80 lớp COCO), lọc lấy các lớp có thể là rác.

    Có mặt ở đây để TRẢ LỜI BẰNG SỐ câu "sao không dùng YOLO thường". COCO
    không có lớp nào tên là rác; gần nhất chỉ có bottle/cup/bowl. Túi nylon,
    vỏ bánh, giấy vụn, hộp xốp — những thứ chiếm phần lớn rác đường phố —
    không thuộc lớp nào cả, nên trần recall bị chặn bởi từ điển lớp chứ không
    phải bởi độ phân giải.
    """

    name = "coco"
    KEEP = {"bottle", "cup", "bowl", "wine glass", "handbag", "backpack",
            "suitcase", "book", "vase", "sports ball", "frisbee"}

    def __init__(self, weights: str = "yolo11s.pt", conf: float = 0.05,
                 device: str | None = None, prompts=None):
        from ultralytics import YOLO
        self.m = YOLO(weights)
        self.conf = conf
        self.device = device

    def detect(self, images: list) -> list[list[Detection]]:
        if not images:
            return []
        kw = {"conf": self.conf, "verbose": False}
        if self.device:
            kw["device"] = self.device
        out = []
        for r in self.m.predict(images, **kw):
            dets = []
            for b in r.boxes:
                nm = r.names[int(b.cls)]
                if nm in self.KEEP:
                    dets.append(Detection(tuple(b.xyxy[0].tolist()), float(b.conf), nm))
            out.append(dets)
        return out


class TrainedDetector:
    """YOLO đã fine-tune một lớp `trash` trên TACO + RoLID + UAVVaste.

    Có mặt để so cho sòng phẳng: yoloworld và coco chưa từng thấy một tấm ảnh
    rác nào, còn hướng 1 thì đã được thu ô âm ngay tại camera đó.
    """

    name = "trained"

    # QUAN TRỌNG khi gọi: phải suy luận với ĐÚNG `upscale` đã dùng lúc dựng bộ
    # train (ô 320px phóng 2.0x -> ảnh 640px). Dùng upscale khác là quay lại
    # đúng cái lệch thang mà cả vòng này sinh ra để sửa.
    def __init__(self, weights: str = "runs/detect/runs/tiles_audited/weights/best.pt",
                 conf: float = 0.10, device: str | None = None, prompts=None):
        from ultralytics import YOLO
        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"chưa có {weights} — chạy tools/train_detector.py trước")
        self.m = YOLO(weights)
        self.conf = conf
        self.device = device

    def detect(self, images: list) -> list[list[Detection]]:
        if not images:
            return []
        kw = {"conf": self.conf, "verbose": False}
        if self.device:
            kw["device"] = self.device
        out = []
        for r in self.m.predict(images, **kw):
            out.append([Detection(tuple(b.xyxy[0].tolist()), float(b.conf), "trash")
                        for b in r.boxes])
        return out


def build_detector(kind: str, **kw):
    if kind == "yoloworld":
        return YoloWorldDetector(**kw)
    if kind == "owlv2":
        return Owlv2Detector(**kw)
    if kind == "coco":
        return CocoDetector(**kw)
    if kind == "trained":
        return TrainedDetector(**kw)
    raise ValueError(f"backend lạ: {kind}")


def detect_in_zone(det, frame: np.ndarray, poly_px: list, tile_px: int = 320,
                   overlap: float = 0.5, upscale: float = 3.0,
                   batch: int = 8) -> list[Detection]:
    """Cắt ô -> chạy detector -> quy về khung gốc -> lọc theo vùng -> NMS."""

    tiles = build_tiles(frame, poly_px, tile_px, overlap, upscale)
    boxes, scores, labels = [], [], []
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        for tile, dets in zip(chunk, det.detect([t.img for t in chunk])):
            for d in dets:
                b = tile.to_frame(d.box)
                if box_center_in_zone(b, poly_px):
                    boxes.append(b)
                    scores.append(d.score)
                    labels.append(d.label)
    keep = nms(boxes, scores, 0.5)
    return [Detection(boxes[i], scores[i], labels[i]) for i in keep]
