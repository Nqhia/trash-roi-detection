"""Lưới ô cố định trên vùng ROI — nền tảng của cả hướng làm.

Thay vì hỏi model "trong vùng này có rác ở đâu" (bài toán detect, rất khó với
vật ~27px), ta chia vùng thành các ô nhỏ CỐ ĐỊNH rồi hỏi từng ô một câu nhị
phân "ô này có rác không". Vỏ chai 27x8px chiếm ~5% diện tích một ô 64px,
so với ~0.07% diện tích cả vùng 800x400 — chênh khoảng 75 lần về tỉ lệ
tín hiệu/nền. Đó là toàn bộ lý do tồn tại của lưới ô.

ID ô phải ỔN ĐỊNH giữa các lượt quét vì mặt nạ nhiễu (clutter.py) đánh dấu
theo ID. ID = (row, col) đếm từ góc trên-trái bbox của đa giác, nên chỉ đổi khi
người dùng vẽ lại vùng hoặc đổi độ phân giải — khi đó `signature` đổi theo và
mặt nạ tự bị vô hiệu (xem ClutterMask.load).

Thuần Python: không import numpy/opencv, để tests/selftest.py chạy được ở
môi trường trần.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Cell:
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def id(self) -> tuple[int, int]:
        return (self.row, self.col)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass(slots=True)
class Grid:
    cells: list[Cell]
    signature: str          # đổi khi vùng/độ phân giải/cỡ ô đổi -> vô hiệu mặt nạ
    cell_px: int
    step: int
    bbox: tuple[int, int, int, int]
    frame_wh: tuple[int, int]
    _by_id: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {c.id: c for c in self.cells}

    def get(self, cell_id: tuple[int, int]) -> Cell | None:
        return self._by_id.get(cell_id)

    def __len__(self) -> int:
        return len(self.cells)


def poly_to_px(poly_norm: list, w: int, h: int) -> list[tuple[int, int]]:
    """Đa giác chuẩn hoá 0..1 (định dạng `zones[].points` của backend) -> pixel."""
    return [(int(round(p[0] * w)), int(round(p[1] * h))) for p in poly_norm]


def point_in_polygon(x: float, y: float, poly: list) -> bool:
    """Ray casting. Cùng thuật toán với core/roi.py của worker, nhưng toạ độ px."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def polygon_bbox(poly: list) -> tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _positions(lo: int, hi: int, cell: int, step: int) -> list[int]:
    """Toạ độ bắt đầu các ô phủ [lo, hi), luôn phủ tới sát mép (không bỏ rìa)."""
    if hi - lo <= cell:
        return [lo]
    out = list(range(lo, hi - cell + 1, step))
    if not out:
        return [lo]
    if out[-1] + cell < hi:
        out.append(hi - cell)
    return out


def build_grid(
    poly_px: list,
    frame_w: int,
    frame_h: int,
    cell_px: int = 64,
    overlap: float = 0.5,
) -> Grid:
    """Phủ lưới ô lên đa giác. Chỉ giữ ô có TÂM nằm trong đa giác.

    `overlap` 0.5 = ô chồng lấn 50% (bước nhảy = nửa ô). Chồng lấn để vật nằm
    vắt qua biên ô vẫn có ít nhất một ô nhìn thấy trọn vẹn — bỏ chồng lấn thì
    vỏ chai bị cắt đôi, mỗi nửa 13px, coi như mất.
    """
    if len(poly_px) < 3:
        raise ValueError("đa giác cần >= 3 điểm")
    cell_px = max(8, int(cell_px))
    overlap = min(0.9, max(0.0, float(overlap)))
    step = max(1, int(round(cell_px * (1.0 - overlap))))

    bx1, by1, bx2, by2 = polygon_bbox(poly_px)
    bx1 = max(0, bx1); by1 = max(0, by1)
    bx2 = min(frame_w, bx2); by2 = min(frame_h, by2)
    if bx2 - bx1 < 2 or by2 - by1 < 2:
        raise ValueError("vùng quá nhỏ so với khung hình")

    xs = _positions(bx1, bx2, cell_px, step)
    ys = _positions(by1, by2, cell_px, step)

    cells: list[Cell] = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            x2 = min(frame_w, x + cell_px)
            y2 = min(frame_h, y + cell_px)
            cx, cy = (x + x2) / 2.0, (y + y2) / 2.0
            if point_in_polygon(cx, cy, poly_px):
                cells.append(Cell(r, c, x, y, x2, y2))

    sig = grid_signature(poly_px, frame_w, frame_h, cell_px, overlap)
    return Grid(cells=cells, signature=sig, cell_px=cell_px, step=step,
                bbox=(bx1, by1, bx2, by2), frame_wh=(frame_w, frame_h))


def grid_signature(poly_px: list, w: int, h: int, cell_px: int, overlap: float) -> str:
    """Vân tay cấu hình lưới. Đổi vùng/khung/cỡ ô -> ID ô không còn nghĩa cũ,
    mặt nạ nhiễu học được trước đó phải bị bỏ.

    Tính được mà KHÔNG cần dựng lưới, nên pipeline gọi mỗi lượt để biết người
    dùng có vẽ lại vùng hay không — rẻ hơn nhiều so với dựng lưới rồi so."""
    parts = [f"{w}x{h}", f"c{cell_px}", f"o{overlap:.2f}"]
    parts += [f"{int(x)},{int(y)}" for x, y in poly_px]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def box_overlap_frac(cell: Cell, box: tuple) -> float:
    """Tỉ lệ diện tích ô bị `box` (x1,y1,x2,y2) che."""
    if cell.area <= 0:
        return 0.0
    ix1 = max(cell.x1, box[0]); iy1 = max(cell.y1, box[1])
    ix2 = min(cell.x2, box[2]); iy2 = min(cell.y2, box[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    return (iw * ih) / float(cell.area)


def occluded_ids(cells, boxes, thr: float = 0.3, pad: float = 0.0) -> set:
    """ID các ô bị người/xe che quá `thr` diện tích (box đã nới ra `pad` px).

    Ô bị che KHÔNG được tính là sạch, cũng không tính là bẩn — chỉ đơn giản là
    lượt này không có dữ liệu. Rác không tự đi mất nên lượt sau nhìn lại vẫn kịp.
    Nguồn `boxes` là person/vehicle lấy từ ctx.prior["object"] của worker.

    `pad` giải quyết nguồn nhiễu LỚN NHẤT đo được trên CCTV thật: bóng đổ và
    rìa người tràn ra ngoài bbox, làm các ô SÁT người cùng đổi theo. Đo trên
    ABODA video3: một người đi bộ làm 13/68 ô đổi, trong đó chỉ 6 ô nằm trong
    bbox — 7 ô còn lại là quầng. Nới nửa ô là dọn gần hết.
    """
    if not boxes:
        return set()
    out = set()
    for c in cells:
        for b in boxes:
            bb = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad) if pad else b
            if box_overlap_frac(c, bb) >= thr:
                out.add(c.id)
                break
    return out
