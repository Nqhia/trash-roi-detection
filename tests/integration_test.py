"""Integration test — chạy pipeline thật trên cảnh tổng hợp. CẦN numpy + opencv.

    /home/nqhia/miniconda3/envs/cv-base/bin/python tests/integration_test.py

selftest.py chỉ phủ phần logic thuần. File này phủ nốt reference.py + pipeline.py:
cổng đổi, mặt nạ nhiễu, che khuất, chống đổi sáng, dò khung dịch chuyển.

Kịch bản dựng đúng bài toán thật: một "nắp cống" NẰM ĐÓ TỪ ĐẦU và trông y hệt
rác với classifier, cộng một túi rác XUẤT HIỆN GIỮA CHỪNG. Pipeline phải học
được rằng cái thứ nhất là một phần của cảnh, còn cái thứ hai thì không.
"""

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import ZoneTrashDetector      # noqa: E402
from core.reference import CellReference, make_thumb, scene_shift  # noqa: E402

W, H = 640, 480
ZONE = [(0.10, 0.30), (0.90, 0.30), (0.90, 0.90), (0.10, 0.90)]
DRAIN = (150, 200)      # nắp cống — có mặt ở MỌI khung hình
TRASH = (420, 330)      # túi rác — chỉ xuất hiện từ giữa chừng
STEP = 30.0             # nhịp quét (giây)

CFG = {
    "grid": {"cell_px": 64, "overlap": 0.5, "occlusion_thr": 0.3},
    "change": {"enabled": True, "alpha": 0.3, "thr": 6.0},
    "decide": {"litter_thr": 0.6, "min_hot_cells": 1, "max_score_per_scan": 0},
    "confirm": {"n": 2, "m": 3},
    "alert": {"cooldown_s": 600, "escalate_ratio": 1.5},
    # rút gọn để test chạy trong vài giây; thực tế là 14400 (~5 ngày)
    "clutter": {"enabled": True, "mute_after_scans": 20, "mute_ratio": 0.9,
                "unmute_after_clean": 5},
    "scene_shift": {"thr_px": 12.0},
}

_fails: list[str] = []
_rng = np.random.default_rng(7)
_BASE = None
_OTHER = None


def check(name, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {extra}")
    if not cond:
        _fails.append(name)


def _textured(lo, hi, tile, seed):
    """Nền có cấu trúc ô gạch NHỎ HƠN Ô LƯỚI, chứ không chỉ nhiễu hạt.

    Nhiễu ngẫu nhiên thuần bị trung bình hoá gần hết khi descriptor thu ảnh về
    8x8, nên hai nền nhiễu khác nhau lại cho descriptor gần như nhau — cảnh
    tổng hợp kiểu đó không kiểm được gì về cổng đổi. Cảnh thật luôn có cạnh và
    mảng lớn sống sót qua bước thu nhỏ.
    """
    r = np.random.default_rng(seed)
    blocks = r.integers(lo, hi, (H // tile + 1, W // tile + 1), dtype=np.uint8)
    g = np.kron(blocks, np.ones((tile, tile), np.uint8))[:H, :W]
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def base_scene():
    """Nền nhựa đường tĩnh — cùng một ảnh ở mọi khung hình."""
    global _BASE
    if _BASE is None:
        _BASE = _textured(80, 120, 16, 11)
    return _BASE.copy()


def other_scene():
    """Nền KHÁC HẲN — mô phỏng camera chuyển IR->màu lúc rạng sáng: cùng một
    chỗ nhưng vân, tương phản, phổ sáng đều đổi. Descriptor trừ mean triệt được
    offset nhưng KHÔNG triệt được cái này."""
    global _OTHER
    if _OTHER is None:
        _OTHER = _textured(35, 200, 16, 23)
    return _OTHER.copy()


def frame(drain=True, trash=False, bright=0, shift=0, noise=True, contrast=1.0,
          newbg=False):
    f = other_scene() if newbg else base_scene()
    if drain:
        cv2.circle(f, DRAIN, 16, (220, 60, 40), -1)
    if trash:
        cv2.circle(f, TRASH, 15, (220, 60, 40), -1)
    if noise:
        f = np.clip(f.astype(np.int16) + _rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8)
    if contrast != 1.0:
        # Đổi TƯƠNG PHẢN (nhân quanh mức xám giữa) — đây mới là thứ mô phỏng
        # rạng sáng / IR->màu. Descriptor trừ mean nên offset thuần bị triệt,
        # còn tương phản thì không.
        f = np.clip((f.astype(np.float32) - 128) * contrast + 128, 0, 255).astype(np.uint8)
    if bright:
        f = np.clip(f.astype(np.int16) + bright, 0, 255).astype(np.uint8)
    if shift:
        f = np.roll(f, shift, axis=1)
    return f


class BlobScorer:
    """Oracle: "có mảng màu marker" = rác. Nắp cống và túi rác cùng màu, nên
    xét riêng từng khung hình thì classifier KHÔNG phân biệt được hai thứ —
    đúng như ở 27px ngoài đời. Chỉ mặt nạ nhiễu mới tách được chúng."""

    name = "blob"

    def score(self, patches):
        out = []
        for p in patches:
            b, g, r = p[:, :, 0], p[:, :, 1], p[:, :, 2]
            n = int(((b > 170) & (g < 120) & (r < 120)).sum())
            out.append(1.0 if n >= 60 else 0.0)
        return out


def run(det, t, **kw):
    return det.scan(frame(**kw), ZONE, kw.pop("boxes", ()), now=t)


def main():
    det = ZoneTrashDetector(CFG, BlobScorer(), camera_id="cam1", zone_id="z1")
    t = 0.0

    # --- Giai đoạn 1: chỉ có nắp cống -----------------------------------
    print("giai đoạn 1 — chỉ nắp cống (cold start)")
    first = run(det, t); t += STEP
    check("lưới dựng đúng", len(det.grid) == 15 * 8, f"({len(det.grid)} ô)")
    check("cold start: mọi ô đều phải hỏi", first.n_scored == first.n_cells,
          f"({first.n_scored}/{first.n_cells})")
    check("cold start BÁO NHẦM nắp cống", len(first.raw_hot) > 0)

    alerts_p1 = int(first.alert)
    for _ in range(29):
        r = run(det, t); t += STEP
        alerts_p1 += int(r.alert)
    check("nắp cống đã bị mute", len(det.clutter.muted()) > 0,
          f"({len(det.clutter.muted())} ô)")
    check("sau khi mute: hết coi là bẩn", not r.dirty)
    check("cold start có báo nhầm (đúng như dự kiến -> cần shadow mode)",
          alerts_p1 >= 1, f"({alerts_p1} lần)")

    # Chốt hạ về compute: cảnh tĩnh -> gần như không ô nào xuống classifier.
    quiet = [run(det, t + i * STEP) for i in range(3)]
    t += 3 * STEP
    check("ổn định: 0 ô phải chấm", all(q.n_scored == 0 for q in quiet),
          f"({[q.n_scored for q in quiet]})")
    check("ổn định: không bẩn", not any(q.dirty for q in quiet))

    # --- Giai đoạn 2: túi rác xuất hiện ----------------------------------
    print("giai đoạn 2 — túi rác xuất hiện")
    r1 = run(det, t, trash=True); t += STEP
    check("bắt được rác mới ngay lượt đầu", r1.dirty, f"(hot={len(r1.hot)})")
    check("chỉ ô quanh rác bị chấm", 0 < r1.n_scored <= 9, f"({r1.n_scored} ô)")
    check("rác KHÔNG bị nhầm là nhiễu cố định",
          all(c.id not in det.clutter.muted() for c in r1.hot))

    r2 = run(det, t, trash=True); t += STEP
    check("ConfirmGate 2/3 -> báo", r2.alert or r1.alert)

    # Rác nằm lâu KHÔNG được nuốt vào nền (bẫy kinh điển của background model)
    for _ in range(10):
        r = run(det, t, trash=True); t += STEP
    check("rác nằm lâu vẫn còn bị coi là bẩn", r.dirty)
    check("không spam: cooldown chặn", not r.alert)

    # --- Giai đoạn 3: dọn rác --------------------------------------------
    print("giai đoạn 3 — dọn rác")
    for _ in range(3):
        r = run(det, t); t += STEP
    check("dọn xong -> hết bẩn", not r.dirty)

    # --- Giai đoạn 4: đổi sáng toàn cục ----------------------------------
    print("giai đoạn 4 — mây tan / đèn bật (sáng +45, không có vật gì mới)")
    lit = [run(det, t + i * STEP, bright=45) for i in range(3)]
    t += 3 * STEP
    check("đổi sáng KHÔNG gây báo nhầm", not any(l.dirty for l in lit),
          f"(hot={[len(l.hot) for l in lit]})")

    # --- Giai đoạn 5: người che ------------------------------------------
    print("giai đoạn 5 — người đứng che chỗ rác")
    box = (TRASH[0] - 60, TRASH[1] - 60, TRASH[0] + 60, TRASH[1] + 60)
    r = det.scan(frame(trash=True), ZONE, [box], now=t); t += STEP
    check("ô bị người che bị loại", r.n_occluded > 0, f"({r.n_occluded} ô)")
    check("bị che -> không kết luận bẩn", not r.dirty)
    r = run(det, t, trash=True); t += STEP
    check("người đi khỏi -> thấy lại rác", r.dirty)

    # --- scene_shift phải đúng ĐƠN VỊ theo từng trục ----------------------
    # Bản cũ nhân cả dy với frame_w/256 nên dịch DỌC bị thổi phồng đúng tỉ lệ
    # khung hình (1,78x với 16:9): hích dọc 7px đọc thành 12,4px -> vứt nền oan.
    print("kiểm scene_shift trục dọc")
    _im = _textured(60, 200, 24, seed=7)[:360, :640]
    _dv = scene_shift(make_thumb(_im), make_thumb(np.roll(_im, 12, axis=0)), 640, 360)
    check("dịch dọc 12px đọc ra ~12px (không phải ~21px)",
          9.0 <= _dv <= 15.0, f"({_dv:.1f}px)")

    # --- Giai đoạn 6: camera bị xoay --------------------------------------
    print("giai đoạn 6 — camera bị va chạm (dịch 20px)")
    n_mask_before = len(det.clutter.muted())
    r = run(det, t, shift=20); t += STEP
    # Phải báo ~20px ảnh GỐC, không phải px ảnh thu nhỏ.
    check("dò được khung dịch chuyển, đúng đơn vị px gốc",
          17.0 <= r.scene_shift_px <= 23.0, f"({r.scene_shift_px:.1f}px)")
    check("reset mặt nạ sau khi dịch",
          len(det.clutter.muted()) == 0 and n_mask_before > 0)
    check("lượt dịch chuyển không báo", not r.alert)

    # --- Vẽ lại vùng ------------------------------------------------------
    print("vẽ lại vùng")
    det2 = ZoneTrashDetector(CFG, BlobScorer(), camera_id="c", zone_id="z")
    det2.scan(frame(), ZONE, (), now=0.0)
    sig1 = det2.grid.signature
    det2.scan(frame(), [(0.2, 0.4), (0.8, 0.4), (0.8, 0.8), (0.2, 0.8)], (), now=STEP)
    check("vùng đổi cùng độ phân giải -> dựng lại lưới",
          det2.grid.signature != sig1)

    # --- Lưu/nạp tham chiếu ----------------------------------------------
    print("tham chiếu: lưu/nạp")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ref.npz")
        a = CellReference()
        a.observe_clean((1, 1), CellReference.describe(frame()[0:64, 0:64]))
        a.save(p, "SIG-A")
        b = CellReference()
        check("nạp lại khi vân tay khớp", b.load(p, "SIG-A") and len(b) == 1)
        c = CellReference()
        check("BỎ khi vân tay lệch", not c.load(p, "SIG-B") and len(c) == 0)

    # --- đổi sáng toàn cục: KHÔNG được thành phát hiện, và phải hồi phục ---
    print("đổi sáng toàn cục (rạng sáng / bật đèn)")
    cfg_co = {**CFG, "decide": {"mode": "change_only", "dwell_scans": 3,
                                "min_hot_cells": 1}}
    d5 = ZoneTrashDetector(cfg_co, None, camera_id="c5", zone_id="z")
    tt = 0.0
    for _ in range(6):                      # dựng nền
        d5.scan(frame(drain=False), ZONE, (), now=tt); tt += STEP
    r = d5.scan(frame(drain=False), ZONE, (), now=tt); tt += STEP
    check("nền ổn định: không nóng", not r.dirty and not r.global_change)

    r = d5.scan(frame(drain=False, contrast=3.2), ZONE, (), now=tt); tt += STEP
    check("tương phản đổi mạnh (rạng sáng) -> toàn cục, KHÔNG báo",
          r.global_change and not r.dirty, f"(nóng={len(r.hot)})")

    # Sau khi nạp lại nền phải HỒI PHỤC — đây là chỗ bản cũ chết vĩnh viễn
    quiet = [d5.scan(frame(drain=False, contrast=3.2), ZONE, (), now=tt + i * STEP)
             for i in range(4)]
    tt += 4 * STEP
    check("hồi phục: hết nóng ở mức tương phản mới",
          all(not q.dirty for q in quiet), f"({[len(q.hot) for q in quiet]})")

    for _ in range(4):
        r = d5.scan(frame(drain=False, trash=True, contrast=3.2), ZONE, (), now=tt)
        tt += STEP
    check("vẫn bắt được rác sau khi đổi tương phản", r.dirty, f"(nóng={len(r.hot)})")

    # --- CHỐT: vật nằm lâu chỉ báo MỘT LẦN --------------------------------
    print("chốt (latch) — xe đỗ / thùng rác đặt trong vùng")
    cfg_l = {**CFG, "decide": {"mode": "change_only", "dwell_scans": 3,
                               "min_hot_cells": 1},
             "alert": {"mode": "latch", "rearm_scans": 3, "escalate_ratio": 1.5},
             "confirm": {"n": 2, "m": 3}}
    d7 = ZoneTrashDetector(cfg_l, None, camera_id="c7", zone_id="z")
    tt = 0.0
    for _ in range(6):
        d7.scan(frame(drain=False), ZONE, (), now=tt); tt += STEP

    # vật xuất hiện và NẰM LẠI RẤT LÂU
    n_alert = 0
    for _ in range(40):
        r = d7.scan(frame(drain=False, trash=True), ZONE, (), now=tt); tt += STEP
        n_alert += int(r.alert)
    check("vật nằm lâu 40 lượt -> chỉ báo ĐÚNG 1 LẦN", n_alert == 1,
          f"({n_alert} lần)")

    # dọn đi -> mở chốt
    for _ in range(6):
        r = d7.scan(frame(drain=False), ZONE, (), now=tt); tt += STEP
    check("dọn xong -> mở chốt", not d7._latched_cells)

    # vật MỚI -> phải báo lại
    n2 = 0
    for _ in range(6):
        r = d7.scan(frame(drain=False, trash=True), ZONE, (), now=tt); tt += STEP
        n2 += int(r.alert)
    check("vật mới sau khi dọn -> báo lại", n2 == 1, f"({n2} lần)")

    # KỊCH BẢN THẬT: thùng rác bị xô lệch (đã chốt) rồi CÓ RÁC Ở CHỖ KHÁC.
    # Chốt theo cả vùng sẽ nuốt mất sự kiện thứ hai; chốt theo ô thì không.
    d9 = ZoneTrashDetector(cfg_l, None, camera_id="c9", zone_id="z")
    tt = 0.0
    for _ in range(6):
        d9.scan(frame(drain=False), ZONE, (), now=tt); tt += STEP
    a1 = 0
    for _ in range(8):                       # "thùng bị xô" = vật ở chỗ A
        r = d9.scan(frame(drain=True), ZONE, (), now=tt); tt += STEP
        a1 += int(r.alert)
    check("vật A báo 1 lần rồi chốt", a1 == 1, f"({a1} lần)")
    a2 = 0
    for _ in range(8):                       # A VẪN Ở ĐÓ, thêm rác ở chỗ B
        r = d9.scan(frame(drain=True, trash=True), ZONE, (), now=tt); tt += STEP
        a2 += int(r.alert)
    check("rác ở chỗ KHÁC vẫn được báo dù A đang chốt", a2 == 1,
          f"({a2} lần, ô mới={r.n_fresh_hot})")

    # chế độ cooldown cũ vẫn báo lặp (cho khách muốn được nhắc)
    cfg_c = {**cfg_l, "alert": {"mode": "cooldown", "cooldown_s": 60,
                                "escalate_ratio": 1.5}}
    d8 = ZoneTrashDetector(cfg_c, None, camera_id="c8", zone_id="z")
    tt = 0.0
    for _ in range(6):
        d8.scan(frame(drain=False), ZONE, (), now=tt); tt += STEP
    n3 = 0
    for _ in range(40):
        r = d8.scan(frame(drain=False, trash=True), ZONE, (), now=tt); tt += STEP
        n3 += int(r.alert)
    check("mode=cooldown vẫn nhắc lại nhiều lần", n3 > 1, f"({n3} lần)")

    # --- vật TO chiếm nửa vùng: KHÔNG được nuốt thành nền -----------------
    print("vật to chiếm nửa vùng (bao rác lớn)")
    d6 = ZoneTrashDetector(cfg_co, None, camera_id="c6", zone_id="z")
    tt = 0.0
    for _ in range(6):
        d6.scan(frame(drain=False), ZONE, (), now=tt); tt += STEP

    def big(**kw):
        """Bao rác lớn: khối GỌN chiếm ~25% diện tích vùng (512x288 px).
        Không kéo hết chiều cao — vật thật không bao giờ trải khắp lưới, đó
        chính là dấu hiệu tách nó khỏi đổi sáng toàn cục."""
        f = frame(drain=False, **kw)
        cv2.rectangle(f, (120, 200), (320, 380), (245, 245, 240), -1)
        return f

    r = d6.scan(big(), ZONE, (), now=tt); tt += STEP
    check("vật to KHÔNG bị coi là đổi sáng toàn cục", not r.global_change,
          f"(đổi={r.n_changed}/{r.n_cells})")
    for _ in range(4):
        r = d6.scan(big(), ZONE, (), now=tt); tt += STEP
    check("vật to vẫn bị BÁO", r.dirty, f"(nóng={len(r.hot)})")

    # --- guard KHÔNG được bắn khi vùng bị che gần hết ---------------------
    print("guard khi vùng bị người che gần hết")
    d10 = ZoneTrashDetector(cfg_co, None, camera_id="c10", zone_id="z")
    tt = 0.0
    for _ in range(6):
        d10.scan(frame(drain=False), ZONE, (), now=tt); tt += STEP
    # người che gần hết vùng, vài ô còn lại đổi hết
    big_box = (int(W*.10), int(H*.30), int(W*.78), int(H*.90))
    r = d10.scan(frame(drain=False, contrast=3.2), ZONE, [big_box], now=tt); tt += STEP
    check("vùng bị che gần hết -> KHÔNG coi là đổi sáng toàn cục",
          not r.global_change, f"(che={r.n_occluded}/{r.n_cells})")
    check("tham chiếu KHÔNG bị xoá", len(d10.ref) > 0, f"({len(d10.ref)} ô còn nền)")

    # --- chốt lại nền theo yêu cầu ---------------------------------------
    print("chốt lại nền (người vận hành dời đồ đạc)")
    d11 = ZoneTrashDetector(cfg_l, None, camera_id="c11", zone_id="z")
    tt = 0.0
    for _ in range(6):
        d11.scan(frame(drain=False), ZONE, (), now=tt); tt += STEP
    for _ in range(6):
        r = d11.scan(frame(drain=True), ZONE, (), now=tt); tt += STEP
    check("đồ đạc dời -> đang bẩn", r.dirty, f"(nóng={len(r.hot)})")
    d11.reset_background()
    for _ in range(4):
        r = d11.scan(frame(drain=True), ZONE, (), now=tt); tt += STEP
    check("sau khi chốt lại nền -> hết bẩn", not r.dirty, f"(nóng={len(r.hot)})")
    for _ in range(5):
        r = d11.scan(frame(drain=True, trash=True), ZONE, (), now=tt); tt += STEP
    check("vẫn bắt được rác mới sau khi chốt lại", r.dirty, f"(nóng={len(r.hot)})")

    # --- cold start "vùng đang sạch" ------------------------------------
    print("cold start assume_clean_start")
    cfg2 = {**CFG, "change": {**CFG["change"], "assume_clean_start": True}}
    d3 = ZoneTrashDetector(cfg2, BlobScorer(), camera_id="c3", zone_id="z")
    r0 = d3.scan(frame(drain=False), ZONE, (), now=0.0)
    check("lượt đầu không hỏi ô nào", r0.n_scored == 0 and not r0.dirty,
          f"(chấm={r0.n_scored})")
    r1 = d3.scan(frame(drain=False, trash=True), ZONE, (), now=STEP)
    check("lượt sau vẫn bắt được rác mới", r1.dirty, f"(hot={len(r1.hot)})")

    d4 = ZoneTrashDetector(CFG, BlobScorer(), camera_id="c4", zone_id="z")
    r0 = d4.scan(frame(drain=False), ZONE, (), now=0.0)
    check("mặc định TẮT -> lượt đầu vẫn hỏi hết", r0.n_scored == r0.n_cells)

    # --- bù rung camera --------------------------------------------------
    # Đo trên ABODA video8: hích 2px + 0,3° giữa chừng thì KHÔNG bù méo làm
    # mất luôn túi rác thật (recall 1/1 -> 0/1), vì mọi ô cùng đổi nên guard
    # toàn cục nạp lại nền liên tục và nuốt cả túi rác vào nền.
    print("bù rung camera")
    # VÙNG CHẾT 3px: dưới mức đó thì CỐ Ý không nắn. Không phải để tiết kiệm —
    # ECC ước lượng từ toàn khung kể cả vật mới xuất hiện, nên vật mới làm lệch
    # chính phép ước lượng. Đo trên khung eco (không camera nào dịch): ECC báo
    # 1,35px, nắn theo đó làm ô đổi tăng 82 -> 158/360. Nắn 1px chỉ sửa 1px
    # nhưng nội suy làm nhoè toàn khung, mà nhoè chính là thứ cổng đổi thấy.
    DEAD = 3.0
    stab = {**CFG, "stabilize": {"enabled": True, "downscale": 2,
                                 "min_px": DEAD, "max_px": 40, "max_deg": 3.0}}
    got = {}
    for tag, cfg_s, sh in (("tat", CFG, 6), ("bat", stab, 6)):
        d = ZoneTrashDetector(cfg_s, BlobScorer(), camera_id=f"s_{tag}", zone_id="z")
        for i in range(6):
            run(d, i * STEP, drain=False)                       # dựng nền
        r = d.scan(frame(drain=False, shift=sh), ZONE, (), now=6 * STEP)
        got[tag] = r.n_changed
        if tag == "bat":
            # Bù méo mà làm nhoè mất rác thì còn tệ hơn không bù.
            r2 = d.scan(frame(drain=False, trash=True, shift=sh), ZONE, (),
                        now=7 * STEP)
            check("có bù -> vẫn bắt được rác trên khung đã lệch",
                  len(r2.hot) > 0, f"(nóng={len(r2.hot)})")
    # Ngưỡng đặt theo TỈ LỆ, không theo con số tuyệt đối: cảnh giả ở đây thô
    # hơn CCTV thật nhiều (2px -> 9% số ô đổi, ngoài đời đo được 86%), nên chốt
    # một mốc tuyệt đối là test sẽ nói dối về mức độ nghiêm trọng.
    check("không bù -> hích 6px vẫn làm hàng loạt ô đổi",
          got["tat"] >= max(5, CFG["decide"].get("min_hot_cells", 1)),
          f"(đổi={got['tat']}/{r.n_cells})")
    check("có bù -> lệch TRÊN vùng chết thì số ô đổi giảm ít nhất 3 lần",
          got["bat"] * 3 <= got["tat"] and got["bat"] <= 0.1 * r.n_cells,
          f"(tắt={got['tat']} -> bật={got['bat']})")

    # Và chiều ngược lại: lệch DƯỚI vùng chết phải được để yên, không nắn.
    d = ZoneTrashDetector(stab, BlobScorer(), camera_id="s_dead", zone_id="z")
    for i in range(6):
        run(d, i * STEP, drain=False)
    r = d.scan(frame(drain=False, shift=1), ZONE, (), now=6 * STEP)
    check("lệch dưới vùng chết -> KHÔNG nắn (tránh nhoè hại hơn lợi)",
          r.stab_px == 0.0 or r.stab_px < DEAD, f"(nắn={r.stab_px:.2f}px)")

    print()
    if _fails:
        print(f"HỎNG {len(_fails)}: " + ", ".join(_fails))
        return 1
    print("Tất cả pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
