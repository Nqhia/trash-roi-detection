"""Hướng P6 — phân loại ô (patch classification) + mặt nạ nhiễu tự học.

Bài toán: khoanh một vùng trên khung hình camera cố định, cứ vài chục giây quét
một lần xem trong vùng có rác không, có thì báo.

Không detect từng vật, không nhận diện người vứt, không đo thời gian tồn đọng.
"""

from .grid import Cell, Grid, build_grid, occluded_ids, poly_to_px, point_in_polygon
from .gates import ConfirmGate, DedupGate
from .clutter import ClutterMask

__all__ = [
    "Cell", "Grid", "build_grid", "occluded_ids", "poly_to_px", "point_in_polygon",
    "ConfirmGate", "DedupGate", "ClutterMask",
]
