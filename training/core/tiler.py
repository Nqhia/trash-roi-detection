"""Cắt ROI thành ô chồng lấn rồi PHÓNG TO từng ô trước khi đưa vào detector.

Đây là toàn bộ lý do hướng này có cơ hội. Detector nhìn cả khung 1920x1080 thì
vỏ chai 27x8px chiếm 0,01% số pixel và gần như chắc chắn bị bỏ qua — không phải
vì model dốt mà vì sau vài lần downsample trong backbone thì vật còn chưa tới
một ô đặc trưng.

Cắt ROI thành ô 320px rồi phóng 3x lên 960px thì đúng vỏ chai đó thành 81x24px
trong ảnh đưa vào model, tức là cỡ vật mà mọi detector đều bắt tốt.

Trả giá bằng số lần chạy model: ROI 800x400 với ô 320 chồng 0,5 -> 3x2 = 6 ô.
Ở nhịp 30 giây thì 6 lần chạy chẳng là gì; ở 30 khung/giây thì bất khả thi.
Đó là lý do hướng này chỉ hợp lệ vì bài toán quét chậm.
"""

from __future__ import annotations

import cv2
import numpy as np

from .zone import poly_bbox


class Tile:
    __slots__ = ("x1", "y1", "x2", "y2", "scale", "img")

    def __init__(self, x1, y1, x2, y2, scale, img):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.scale = scale
        self.img = img

    def to_frame(self, box: tuple) -> tuple:
        """Box trong toạ độ ô (đã phóng to) -> toạ độ khung gốc."""
        return (self.x1 + box[0] / self.scale, self.y1 + box[1] / self.scale,
                self.x1 + box[2] / self.scale, self.y1 + box[3] / self.scale)


def build_tiles(frame: np.ndarray, poly_px: list, tile_px: int = 320,
                overlap: float = 0.5, upscale: float = 3.0,
                max_side: int = 1280) -> list[Tile]:
    """Cắt bbox của ROI thành ô chồng lấn, mỗi ô phóng `upscale` lần.

    Chồng lấn để vật nằm đúng đường cắt không bị chia đôi thành hai nửa mà
    không nửa nào đủ nhận ra. Với chồng 0,5 thì mọi vật nhỏ hơn nửa ô luôn
    nằm trọn trong ít nhất một ô.
    """
    h, w = frame.shape[:2]
    bx1, by1, bx2, by2 = poly_bbox(poly_px)
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(w, bx2), min(h, by2)
    if bx2 - bx1 < 8 or by2 - by1 < 8:
        return []

    step = max(1, int(tile_px * (1.0 - overlap)))
    xs = list(range(bx1, max(bx1 + 1, bx2 - tile_px + 1), step))
    ys = list(range(by1, max(by1 + 1, by2 - tile_px + 1), step))
    # Luôn phủ tới mép phải/dưới: thiếu thì rác sát mép vùng không bao giờ được nhìn.
    if xs[-1] + tile_px < bx2:
        xs.append(max(bx1, bx2 - tile_px))
    if ys[-1] + tile_px < by2:
        ys.append(max(by1, by2 - tile_px))

    out = []
    for y in ys:
        for x in xs:
            x2, y2 = min(w, x + tile_px), min(h, y + tile_px)
            crop = frame[y:y2, x:x2]
            if crop.size == 0:
                continue
            s = upscale
            if max(crop.shape[:2]) * s > max_side:      # trần để khỏi phình VRAM
                s = max_side / max(crop.shape[:2])
            img = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            out.append(Tile(x, y, x2, y2, s, img))
    return out


def nms(boxes: list, scores: list, iou_thr: float = 0.5) -> list[int]:
    """NMS thường, dùng để gộp phát hiện trùng ở phần chồng lấn giữa các ô.

    Không gộp thì mỗi vật ở vùng chồng lấn sẽ được đếm 2-4 lần và mọi con số
    FP/recall đều sai theo.
    """
    if not boxes:
        return []
    b = np.asarray(boxes, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    order = s.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(b[i, 0], b[rest, 0])
        yy1 = np.maximum(b[i, 1], b[rest, 1])
        xx2 = np.minimum(b[i, 2], b[rest, 2])
        yy2 = np.minimum(b[i, 3], b[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep
