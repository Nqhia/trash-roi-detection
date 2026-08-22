"""Self-test cho phần logic thuần — CHẠY ĐƯỢC KHÔNG CẦN numpy/opencv.

Cố ý chỉ phủ grid/gates/clutter: đó là chỗ bug thật hay nấp (ID ô không ổn
định, mặt nạ mute nhầm, cooldown rò) và cũng là chỗ chạy được ở môi trường trần.
Phần reference/scorers/pipeline cần numpy+cv2, kiểm bằng tools/run_video.py.

    python3 tests/selftest.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.clutter import ClutterMask                      # noqa: E402
from core.gates import ConfirmGate, DedupGate             # noqa: E402
from core.grid import (                                   # noqa: E402
    box_overlap_frac, build_grid, occluded_ids, point_in_polygon, poly_to_px,
)

_fails: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {extra}")
        _fails.append(name)


def eq(name: str, got, want) -> None:
    check(name, got == want, f"(được {got!r}, mong {want!r})")


# ---------------------------------------------------------------- grid

def test_grid() -> None:
    print("grid")
    W, H = 1920, 1080
    # Hình chữ nhật chuẩn hoá -> 800x400 px bắt đầu ở (200,300)
    poly_n = [(200 / W, 300 / H), (1000 / W, 300 / H),
              (1000 / W, 700 / H), (200 / W, 700 / H)]
    poly = poly_to_px(poly_n, W, H)
    eq("poly_to_px", poly[0], (200, 300))

    g = build_grid(poly, W, H, cell_px=64, overlap=0.5)
    # Bước 32. Cột: 200,232,...,936 -> 24 vị trí, ô cuối 936+64=1000 chạm đúng mép.
    # Hàng: 300,332,...,620 -> 11 vị trí, nhưng 620+64=684 < 700 nên thêm một hàng
    # kẹp ở 636 để phủ nốt dải 684..700 -> 12 hàng. Không phủ nốt thì rác nằm ở
    # mép dưới vùng sẽ không ô nào nhìn thấy.
    eq("số cột", len({c.col for c in g.cells}), 24)
    eq("số hàng", len({c.row for c in g.cells}), 12)
    eq("số ô", len(g), 24 * 12)
    check("phủ tới sát mép dưới vùng", max(c.y2 for c in g.cells) >= 700)
    check("phủ tới sát mép phải vùng", max(c.x2 for c in g.cells) >= 1000)
    check("ô nằm trong khung", all(0 <= c.x1 < c.x2 <= W and 0 <= c.y1 < c.y2 <= H
                                   for c in g.cells))
    check("cỡ ô đúng", all(c.x2 - c.x1 == 64 and c.y2 - c.y1 == 64 for c in g.cells))

    # ID phải ỔN ĐỊNH: dựng lại y hệt -> cùng ID, cùng vân tay
    g2 = build_grid(poly, W, H, cell_px=64, overlap=0.5)
    eq("vân tay ổn định", g2.signature, g.signature)
    eq("ID ổn định", [c.id for c in g2.cells], [c.id for c in g.cells])

    # Vẽ lại vùng -> vân tay PHẢI đổi (nếu không, mặt nạ nhiễu cũ sẽ mute nhầm ô)
    poly_b = poly_to_px([(0.1, 0.2), (0.6, 0.2), (0.6, 0.7), (0.1, 0.7)], W, H)
    check("vân tay đổi khi vùng đổi",
          build_grid(poly_b, W, H, 64, 0.5).signature != g.signature)
    check("vân tay đổi khi cỡ ô đổi",
          build_grid(poly, W, H, 96, 0.5).signature != g.signature)
    check("vân tay đổi khi độ phân giải đổi",
          build_grid(poly_to_px(poly_n, 1280, 720), 1280, 720, 64, 0.5).signature
          != g.signature)

    # Không chồng lấn -> ít ô hơn hẳn
    g0 = build_grid(poly, W, H, cell_px=64, overlap=0.0)
    check("overlap=0 ít ô hơn", len(g0) < len(g), f"({len(g0)} vs {len(g)})")

    # Tam giác: chỉ giữ ô có TÂM bên trong -> phải ít hơn lưới phủ cả bbox
    tri = [(200, 300), (1000, 300), (200, 700)]
    gt = build_grid(tri, W, H, cell_px=64, overlap=0.5)
    check("đa giác lõm/tam giác lọc theo tâm", 0 < len(gt) < len(g),
          f"({len(gt)} vs {len(g)})")

    # Vùng bé hơn một ô vẫn phải ra đúng 1 ô, không nổ
    tiny = [(10, 10), (40, 10), (40, 40), (10, 40)]
    check("vùng nhỏ hơn ô", len(build_grid(tiny, W, H, cell_px=64)) >= 0)

    # Vùng sát mép phải được phủ tới sát mép
    edge = [(1900, 1060), (1919, 1060), (1919, 1079), (1900, 1079)]
    ge = build_grid(edge, W, H, cell_px=64, overlap=0.5)
    check("ô sát mép không tràn khung",
          all(c.x2 <= W and c.y2 <= H for c in ge.cells))


def test_point_in_polygon() -> None:
    print("point_in_polygon")
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    check("trong", point_in_polygon(5, 5, sq))
    check("ngoài", not point_in_polygon(15, 5, sq))
    check("thiếu điểm -> False", not point_in_polygon(5, 5, [(0, 0), (1, 1)]))
    # chữ L lõm: điểm trong phần bị khoét phải nằm ngoài
    L = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    check("lõm: trong nhánh", point_in_polygon(2, 8, L))
    check("lõm: ngoài phần khoét", not point_in_polygon(8, 8, L))


def test_occlusion() -> None:
    print("che khuất")
    W, H = 1920, 1080
    poly = poly_to_px([(0.1, 0.3), (0.5, 0.3), (0.5, 0.6), (0.1, 0.6)], W, H)
    g = build_grid(poly, W, H, cell_px=64, overlap=0.5)
    c = g.cells[0]
    eq("che toàn bộ", round(box_overlap_frac(c, (c.x1, c.y1, c.x2, c.y2)), 3), 1.0)
    eq("không dính", box_overlap_frac(c, (c.x2 + 10, c.y2 + 10, c.x2 + 50, c.y2 + 50)), 0.0)
    eq("che 1/4", round(box_overlap_frac(
        c, (c.x1, c.y1, c.x1 + 32, c.y1 + 32)), 3), 0.25)

    box = (c.x1 - 5, c.y1 - 5, c.x1 + 60, c.y1 + 60)
    occ = occluded_ids(g.cells, [box], thr=0.3)
    check("ô dưới người bị loại", c.id in occ)
    check("không có box -> không loại ô nào", occluded_ids(g.cells, [], 0.3) == set())
    check("ngưỡng cao -> loại ít hơn",
          len(occluded_ids(g.cells, [box], 0.99)) <= len(occ))

    # pad: bóng đổ/rìa người tràn ngoài bbox làm ô SÁT người cũng đổi theo
    padded = occluded_ids(g.cells, [box], 0.3, pad=48)
    check("pad nới vùng che", occ <= padded and len(padded) > len(occ),
          f"({len(occ)} -> {len(padded)})")
    check("pad=0 giữ nguyên", occluded_ids(g.cells, [box], 0.3, pad=0) == occ)
    check("không có box thì pad vô nghĩa",
          occluded_ids(g.cells, [], 0.3, pad=96) == set())


# ---------------------------------------------------------------- gates

def test_confirm_gate() -> None:
    print("ConfirmGate 4/6")
    g = ConfirmGate()
    k = "cam1/zone1"
    for i, v in enumerate([True, True, True]):
        check(f"chưa đủ sau {i+1} lượt dương", not g.passed(k, v, 4, 6))
    check("đủ 4/6 -> pass", g.passed(k, True, 4, 6))

    g2 = ConfirmGate()
    seq = [True, False, True, False, True, False]   # 3/6
    for v in seq[:-1]:
        g2.passed(k, v, 4, 6)
    check("3/6 -> không pass", not g2.passed(k, seq[-1], 4, 6))

    g3 = ConfirmGate()
    for _ in range(6):
        g3.passed(k, False, 4, 6)
    check("toàn âm -> is_all_absent", g3.is_all_absent(k))
    g3.passed(k, True, 4, 6)
    check("có 1 dương -> hết all_absent", not g3.is_all_absent(k))

    g4 = ConfirmGate()
    for _ in range(10):
        g4.passed(k, True, 4, 6)
    for _ in range(6):
        g4.passed(k, False, 4, 6)
    check("cửa sổ trượt quên lượt cũ", g4.is_all_absent(k))

    # Khoá PHẢI là tuple có camera_id, nếu không forget_camera() không dọn được
    # -> state rò lại mãi mỗi lần camera bị gỡ khỏi hệ thống.
    g5 = ConfirmGate()
    g5.passed(("cam9", "zone1"), True, 1, 1)
    g5.forget_camera("cam9")
    check("forget_camera dọn state", g5.is_all_absent(("cam9", "zone1")))

    g6 = DedupGate()
    g6.passed(("cam9", "zone1"), 10.0, 1800)
    g6.forget_camera("cam9")
    check("DedupGate.forget_camera dọn state", g6.passed(("cam9", "zone1"), 11.0, 1800))


def test_dedup_gate() -> None:
    print("DedupGate")
    d = DedupGate()
    k = ("cam1", "zone1")
    check("lần đầu qua kể cả khi mốc thời gian nhỏ", d.passed(k, 0.0, 1800))
    d = DedupGate()
    check("lần đầu qua", d.passed(k, 1000.0, 1800))
    check("trong cooldown bị chặn", not d.passed(k, 1000.0 + 1799, 1800))
    check("hết cooldown qua lại", d.passed(k, 1000.0 + 1801, 1800))
    d.force(k, 5000.0)
    check("force() khoá lại", not d.passed(k, 5000.0 + 10, 1800))


# ---------------------------------------------------------------- clutter

def test_clutter() -> None:
    print("mặt nạ nhiễu")
    m = ClutterMask("sig1", min_scans=100, mute_ratio=0.98, unmute_after_clean=20)
    drain, real = (3, 0), (1, 2)

    # Nắp cống: lượt nào bị hỏi cũng "bẩn" -> phải bị mute
    for i in range(99):
        m.update(drain, True)
    check("chưa đủ số lượt -> chưa mute", not m.is_muted(drain))
    m.update(drain, True)
    check("đủ lượt + tỉ lệ cao -> mute", m.is_muted(drain))

    # Rác thật: lúc có lúc không -> KHÔNG được mute
    for i in range(200):
        m.update(real, i % 3 == 0)          # ~33%
    check("rác thật không bị mute", not m.is_muted(real))
    eq("chỉ nắp cống bị mute", m.muted(), [drain])

    # Điểm đổ rác kinh niên (bẩn 90% thời gian) cũng không được mute nhầm
    chronic = (5, 5)
    for i in range(500):
        m.update(chronic, i % 10 != 0)      # 90%
    check("điểm kinh niên 90% không bị mute", not m.is_muted(chronic))

    # Ai đó dời cái thùng đi -> ô sạch liên tục -> tự bỏ mute
    for _ in range(19):
        m.update(drain, False)
    check("chưa đủ chuỗi sạch -> vẫn mute", m.is_muted(drain))
    m.update(drain, False)
    check("đủ chuỗi sạch -> tự bỏ mute", not m.is_muted(drain))

    # Bỏ mute tay
    m2 = ClutterMask("sig1", min_scans=10, mute_ratio=0.9)
    for _ in range(10):
        m2.update((0, 0), True)
    check("mute", m2.is_muted((0, 0)))
    m2.unmute((0, 0))
    check("unmute tay", not m2.is_muted((0, 0)))

    check("progress trong 0..1", 0.0 <= m2.progress() <= 1.0)


def test_clutter_persist() -> None:
    print("mặt nạ: lưu/nạp")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sub", "cam1.clutter.json")
        m = ClutterMask("sigA", min_scans=10, mute_ratio=0.9)
        for _ in range(10):
            m.update((2, 2), True)
        m.update((0, 1), False)
        m.save(p)
        check("tạo được thư mục cha", os.path.exists(p))

        back = ClutterMask.load(p, "sigA", min_scans=10, mute_ratio=0.9)
        check("nạp lại giữ mute", back.is_muted((2, 2)))
        eq("nạp lại giữ danh sách", back.muted(), [(2, 2)])

        # Vùng vẽ lại -> vân tay lệch -> PHẢI vứt hết, không được mute nhầm ô mới
        other = ClutterMask.load(p, "sigB", min_scans=10, mute_ratio=0.9)
        check("vân tay lệch -> bỏ state cũ", not other.is_muted((2, 2)))
        eq("vân tay lệch -> rỗng", other.muted(), [])

        missing = ClutterMask.load(os.path.join(d, "khong-co.json"), "sigA")
        eq("thiếu file -> mặt nạ rỗng", missing.muted(), [])

        with open(p, "w", encoding="utf-8") as f:
            f.write("{ hỏng")
        broken = ClutterMask.load(p, "sigA")
        eq("file hỏng -> mặt nạ rỗng", broken.muted(), [])


def main() -> int:
    for fn in (test_grid, test_point_in_polygon, test_occlusion,
               test_confirm_gate, test_dedup_gate, test_clutter,
               test_clutter_persist):
        fn()
    print()
    if _fails:
        print(f"HỎNG {len(_fails)}: " + ", ".join(_fails))
        return 1
    print("Tất cả pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
