"""ROI đa giác. Python thuần, không phụ thuộc hướng 1.

Cố ý chép lại thay vì import từ ../patch_classifier: hai hướng phải so được
với nhau một cách sòng phẳng, mà dùng chung code thì một thay đổi bên kia âm
thầm đổi kết quả bên này. Chỗ này chỉ có 40 dòng.
"""

from __future__ import annotations


def poly_to_px(poly_norm: list, w: int, h: int) -> list[tuple[int, int]]:
    """Đa giác chuẩn hoá 0..1 (định dạng `zones[].points` của backend) -> pixel."""
    return [(int(round(p[0] * w)), int(round(p[1] * h))) for p in poly_norm]


def point_in_polygon(x: float, y: float, poly: list) -> bool:
    """Ray casting. Cùng thuật toán với core/roi.py của worker."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1
            if x < xint:
                inside = not inside
    return inside


def poly_bbox(poly_px: list) -> tuple[int, int, int, int]:
    xs = [p[0] for p in poly_px]
    ys = [p[1] for p in poly_px]
    return min(xs), min(ys), max(xs), max(ys)


def box_center_in_zone(box: tuple, poly_px: list) -> bool:
    """Vật có nằm trong vùng không — xét theo TÂM ĐÁY, không xét theo tâm hộp.

    Vật nằm trên mặt đất thì điểm chạm đất mới là vị trí thật của nó. Xét tâm
    hộp thì một cái túi cao 60px đứng sát mép vùng sẽ bị tính lệch lên 30px.
    """
    x1, y1, x2, y2 = box
    return point_in_polygon((x1 + x2) / 2.0, y2, poly_px)
