"""Mở nguồn video cho các tool. Tồn tại vì đúng một lỗi, đáng ghi lại.

Trên WSL2, `cv2.VideoCapture('rtsp://...')` mặc định đi UDP. Camera Hikvision ở
LAN trả `isOpened() == True` sau ~32 giây rồi `read()` trả False mãi mãi — gói
UDP không xuyên nổi lớp NAT của WSL. Ép `rtsp_transport;tcp` thì mở tức thì và
đọc được 1920x1080 ngay khung đầu.

Hai điều rút ra, cả hai đã sửa ở đây:

  * `isOpened()` KHÔNG có nghĩa là đọc được khung. Vòng thử lại cũ chỉ xét
    `isOpened()` nên nó không bao giờ kích hoạt đúng lúc cần nhất.
  * Ép TCP phải nằm trong code chứ không phải trong biến môi trường người chạy
    tự nhớ — một lần quên là cả đêm chạy không thu được gì.
"""

from __future__ import annotations

import logging
import os
import time

import cv2

logger = logging.getLogger("capture")

# stimeout tính bằng micro giây: 5s không có gói nào thì bỏ, để backoff bên
# ngoài xử lý thay vì treo vô hạn.
FFMPEG_OPTS = "rtsp_transport;tcp|stimeout;5000000"


def is_stream(source: str) -> bool:
    return isinstance(source, str) and "://" in source


def open_source(source, open_ms: int = 8000, read_ms: int = 8000,
                tries: int = 5, probe: bool = True):
    """-> VideoCapture đã ĐỌC ĐƯỢC ít nhất một khung, hoặc None."""
    if is_stream(source):
        # Không ghi đè nếu người chạy đã tự đặt — họ có thể đang cố ý thử UDP.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", FFMPEG_OPTS)

    for k in range(max(1, tries)):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_ms)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_ms)
        except Exception:      # noqa: BLE001 — build OpenCV cũ không có 2 cờ này
            pass
        if cap.isOpened() and (not probe or cap.read()[0]):
            return cap
        cap.release()
        logger.warning("chưa lấy được khung từ nguồn (lần %d/%d)", k + 1, tries)
        time.sleep(2)
    return None
