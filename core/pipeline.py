"""Ráp tất cả lại — một lượt quét vùng.

    cắt vùng đã khoanh (độ phân giải GỐC)
      -> chia ô 64px chồng lấn 50%
      -> bỏ ô bị người/xe che
      -> cổng đổi: chỉ giữ ô KHÁC tham chiếu
      -> classifier chấm từng ô còn lại
      -> bỏ ô nằm trong mặt nạ nhiễu
      -> ConfirmGate 4/6 -> DedupGate 30' -> BÁO

Ba bộ lọc chồng nhau, mỗi bộ cắt một loại nhiễu khác nhau, không thay thế nhau:
  cổng đổi      -> vệt bẩn cố định GIỐNG NỀN + tiết kiệm compute
  mặt nạ nhiễu  -> FP HỆ THỐNG (nắp cống bị chấm bẩn ở mọi lượt)
  ConfirmGate   -> FP NGẪU NHIÊN (nhiễu nén, bóng thoáng qua)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .clutter import ClutterMask
from .gates import ConfirmGate, DedupGate
from .grid import Cell, Grid, build_grid, grid_signature, occluded_ids, poly_to_px
from .reference import (CellReference, apply_warp, estimate_warp, gray_small,
                        make_thumb, scene_shift)
from .verify import RegionVerifier

logger = logging.getLogger("trash.pipeline")


@dataclass(slots=True)
class ScanResult:
    ts: float
    dirty: bool = False           # kết luận của RIÊNG lượt này (sau mặt nạ)
    alert: bool = False           # sau ConfirmGate + DedupGate — cái đẩy ra MQTT
    escalated: bool = False       # báo lại sớm vì rác tăng thêm
    hot: list = field(default_factory=list)       # ô bẩn sau mặt nạ
    raw_hot: list = field(default_factory=list)   # ô bẩn trước mặt nạ (để debug)
    scores: dict = field(default_factory=dict)    # cell_id -> điểm, chỉ ô được chấm
    notes: dict = field(default_factory=dict)     # cell_id -> ghi chú của VLM
    n_cells: int = 0
    n_occluded: int = 0
    n_changed: int = 0
    n_scored: int = 0
    n_dropped: int = 0            # ô đổi nhưng bị trần cắt — KHÔNG im lặng
    n_muted_hit: int = 0
    scene_shift_px: float = 0.0
    stab_px: float = 0.0           # độ lệch đã BÙ được ở lượt này
    stab_deg: float = 0.0
    mask_progress: float = 0.0
    global_change: bool = False    # đổi sáng toàn cục -> đã nạp lại nền, bỏ lượt
    rearmed: bool = False          # vùng sạch trở lại -> mở chốt, sẵn sàng báo tiếp
    n_fresh_hot: int = 0           # ô nóng CHƯA từng báo — cái thực sự kích hoạt
    n_verify_dropped: int = 0      # ô nóng bị detector bác bỏ
    verify_boxes: list = field(default_factory=list)
    verify_failed: bool = False    # detector lỗi -> giữ nguyên, KHÔNG nuốt im lặng
    ms: float = 0.0

    def summary(self) -> str:
        return (f"ô={self.n_cells} che={self.n_occluded} đổi={self.n_changed} "
                f"chấm={self.n_scored} rớt={self.n_dropped} mute={self.n_muted_hit} "
                f"nóng={len(self.hot)} bẩn={int(self.dirty)} báo={int(self.alert)} "
                f"({self.ms:.0f}ms)")


class ZoneTrashDetector:
    """Một instance cho MỘT cặp (camera, vùng). State không dùng chung."""

    def __init__(self, cfg: dict, scorer, camera_id: str = "cam0",
                 zone_id: str = "zone0", state_dir: str | None = None) -> None:
        self.cfg = cfg
        self.scorer = scorer
        self.camera_id = camera_id
        self.zone_id = zone_id
        # Khoá cổng PHẢI là tuple có camera_id ở đầu — quy ước của core/gates.py
        # trong worker, để forget_camera() dọn được state khi camera bị gỡ.
        self.key = (camera_id, zone_id)
        self.label = f"{camera_id}/{zone_id}"
        self.state_dir = state_dir
        self._slug = f"{camera_id}__{zone_id}".replace("/", "_").replace(":", "_")

        g = cfg.get("grid", {})
        self.cell_px = int(g.get("cell_px", 64))
        self.overlap = float(g.get("overlap", 0.5))
        self.occlusion_thr = float(g.get("occlusion_thr", 0.3))
        # Mặc định 0. Đo trên ABODA video3: nới nửa ô kéo trung vị số ô đổi từ
        # 2 xuống 0, NHƯNG p95 không đổi (14 -> 14) mà lại che tới 24/68 ô (35%
        # vùng) mỗi khi có người trong khung. Đổi 2 lần chạy classifier lấy một
        # phần ba vùng bị mù là lỗ vốn. Chỉ nâng lên nếu classifier hay kêu ở
        # các ô SÁT người.
        self.occlusion_pad = float(g.get("occlusion_pad_px", 0.0))

        c = cfg.get("change", {})
        self.change_on = bool(c.get("enabled", True))
        self.assume_clean_start = bool(c.get("assume_clean_start", False))
        # Bao nhiêu phần vùng cùng đổi thì coi là ĐỔI SÁNG TOÀN CỤC chứ không
        # phải vật xuất hiện. Xem `_global_change` để biết vì sao bắt buộc có.
        self.global_frac = float(c.get("global_change_frac", 0.45))
        # Ô đổi phải trải ra >= ngần này phần số hàng VÀ số cột của lưới thì mới
        # coi là đổi sáng. Chặn trường hợp một vật to chiếm nửa vùng bị nuốt.
        # Đo trên lưới 8x15: vật chiếm 24% diện tích vùng -> trải 0.60; 11% -> 0.47;
        # còn đổi sáng toàn cục -> 1.00. Khoảng cách rất rộng nên 0.8 an toàn.
        self.global_spread = float(c.get("global_change_spread", 0.8))
        # Chỉ xét guard khi còn NHÌN THẤY đủ phần vùng. Đo được trên camera thật:
        # người đứng che 203/227 ô, còn 24 ô cùng đổi -> guard bắn và XOÁ SẠCH nền
        # toàn vùng, có thể xoá luôn dấu vết rác vừa bị vứt.
        self.global_min_visible = float(c.get("global_change_min_visible", 0.6))
        self.ref = CellReference(alpha=float(c.get("alpha", 0.05)),
                                 change_thr=float(c.get("thr", 6.0)))

        d = cfg.get("decide", {})
        # "classifier": hỏi model từng ô — cần dữ liệu train của CHÍNH camera đó.
        # "change_only": bỏ hẳn model. Ô nào KHÁC nền liên tục >= dwell_scans lượt
        #   thì báo. Không cần một byte dữ liệu nào, chạy được ngay ngày lắp.
        #   Dùng khi vùng giám sát là VÙNG TRỐNG: mọi vật lạ nằm lại đều đáng báo,
        #   không cần phân biệt rác với túi xách bỏ quên.
        self.mode = str(d.get("mode", "classifier"))
        self.litter_thr = float(d.get("litter_thr", 0.6))
        self.min_hot = int(d.get("min_hot_cells", 1))
        self.max_score = int(d.get("max_score_per_scan", 24))
        # Người đi qua làm ô đổi 1-2 lượt rồi thôi; vật bỏ lại giữ liên tục tới
        # khi được dọn. Đo trên ABODA: 75% ô nhiễu do người chỉ đổi ĐÚNG 1 lượt,
        # còn ô có vật giữ 10-15 lượt. dwell=5 loại sạch người mà không mất vật.
        self.dwell = int(d.get("dwell_scans", 5))
        self._run: dict = {}

        cf = cfg.get("confirm", {})
        self.n = int(cf.get("n", 4))
        self.m = int(cf.get("m", 6))

        a = cfg.get("alert", {})
        # "latch"    : báo MỘT LẦN ở cạnh sạch->bẩn, im cho tới khi vùng sạch lại.
        #              Vật nằm lâu (xe đỗ, thùng rác đặt trong vùng) chỉ báo 1 lần.
        # "cooldown" : báo lại mỗi cooldown_s khi còn bẩn (hành vi cũ) — chỉ dùng
        #              khi khách muốn được nhắc liên tục tới lúc dọn.
        self.alert_mode = str(a.get("mode", "latch"))
        self.cooldown_s = float(a.get("cooldown_s", 1800))
        self.escalate_ratio = float(a.get("escalate_ratio", 1.5))
        # Bao nhiêu lượt sạch liên tiếp thì mở chốt, sẵn sàng báo sự kiện mới.
        self.rearm_scans = int(a.get("rearm_scans", 4))
        # Bán kính (đơn vị ô lưới) coi là "cùng một vật" với ô đã chốt.
        # Với chồng lấn 50% thì 2 bước lưới ≈ 1 cạnh ô.
        self.merge_radius = int(a.get("merge_radius_cells", 2))
        self._latched_cells: set = set()
        self._clean_run = 0

        self.clutter_on = bool(cfg.get("clutter", {}).get("enabled", True))
        self.shift_thr = float(cfg.get("scene_shift", {}).get("thr_px", 3.0))

        # Bù méo: nắn khung về đúng khung lúc dựng nền trước khi cắt ô.
        # Đo trên CCTV ngoài trời thật: lệch 2px làm 86% số ô "đổi", xoay 0,5°
        # làm 58%. Cổng dịch khung (12px) và guard toàn cục nuốt phần lớn, nhưng
        # còn hở đúng dải 1px / 0,2-0,5° -> lọt một cảnh báo nhầm. Cột điện ngoài
        # đường rung trong gió nằm gọn trong dải đó.
        s = cfg.get("stabilize", {})
        self.stab_on = bool(s.get("enabled", False))
        self.stab_ds = max(1, int(s.get("downscale", 4)))
        self.stab_max_px = float(s.get("max_px", 40.0))
        # VÙNG CHẾT — dưới mức này thì KHÔNG nắn. Không phải chi tiết vặt:
        # warpAffine nội suy lại TỪNG pixel của cả khung, mà nhoè nội suy chính
        # là thứ cổng đổi nhìn thấy. Nắn 1px chỉ sửa được 1px lệch nhưng làm
        # nhoè toàn khung -> lỗ vốn.
        #
        # Tệ hơn: ECC ước lượng từ TOÀN khung, kể cả vật mới xuất hiện, nên vật
        # mới làm lệch chính phép ước lượng. Đo trên khung eco: ECC báo 1,35px
        # (không có camera nào dịch cả), nắn theo con số đó làm ô đổi tăng từ
        # 82 lên 158/360 — bù méo làm HỎNG chứ không làm tốt.
        #
        # Quét vùng chết trên chính dữ liệu này: 3,0px là cửa sổ duy nhất vừa giết
        # được phép nắn giả 1,35px (158 -> 82 ô đổi) vừa cứu được lệch thật 4px
        # (245 -> 0 ô đổi). Dưới 3px thì hại, trên 5px thì mất tác dụng.
        self.stab_min_px = float(s.get("min_px", 3.0))

        self.stab_max_deg = float(s.get("max_deg", 3.0))
        self.stab_iters = int(s.get("iters", 60))
        self._stab_ref: np.ndarray | None = None

        # Tầng xác nhận bằng detector. Xem core/verify.py để biết vì sao nó nằm
        # ở đây chứ không nằm trong scorers.py.
        self.verifier = RegionVerifier(cfg.get("verify", {}))

        self.confirm = ConfirmGate()
        self.dedup = DedupGate()
        self.grid: Grid | None = None
        self.clutter: ClutterMask | None = None
        self._thumb: np.ndarray | None = None
        self._last_alert_hot = 0
        self._scans = 0

    def _decide_alert(self, res: "ScanResult", now: float) -> None:
        """ConfirmGate -> chốt -> cảnh báo. Dùng chung cho cả hai chế độ.

        Chốt theo TỪNG Ô, không theo cả vùng. Chốt cả vùng thì một vật đã báo
        rồi (xe đỗ, thùng rác bị xô lệch) sẽ NUỐT MẤT mọi sự kiện sau: rác vứt
        ở góc khác chỉ thêm vài ô, không đủ vượt ngưỡng leo thang, nên im luôn.
        Theo từng ô thì ô nào CHƯA TỪNG báo mà nóng lên đều là sự kiện mới.
        """
        hot_ids = {c.id for c in res.hot}
        # Ô nóng nằm SÁT ô đã chốt = viền của chính vật đã báo, hấp thụ vào chốt.
        # Các ô của cùng một vật đạt ngưỡng dwell LỆCH NHAU vài lượt, nên nếu
        # không gộp thì một vật duy nhất bắn nhiều cảnh báo khi nó lấp đầy dần.
        if self._latched_cells:
            r = self.merge_radius
            grow = {h for h in hot_ids
                    if any(abs(h[0] - l[0]) <= r and abs(h[1] - l[1]) <= r
                           for l in self._latched_cells)}
            self._latched_cells |= grow
        fresh = hot_ids - self._latched_cells
        res.n_fresh_hot = len(fresh)

        # latch: chỉ ô MỚI mới kích hoạt. cooldown: cứ bẩn là kích hoạt.
        trigger = (len(fresh) >= self.min_hot if self.alert_mode == "latch"
                   else res.dirty)
        if self.confirm.passed(self.key, trigger, self.n, self.m):
            if (self.alert_mode == "latch"
                    or self.dedup.passed(self.key, now, self.cooldown_s)):
                res.alert = True
                res.escalated = bool(self._latched_cells)   # đã có ô chốt từ trước
                self._latched_cells |= hot_ids
                self.dedup.force(self.key, now)
                # XOÁ cửa sổ N/M sau khi bắn. Nó là cửa sổ TRƯỢT nên vẫn giữ các
                # lượt True cũ: bắn ở lượt k rồi lượt k+1 dù không còn ô mới,
                # cửa sổ [True,True,False] vẫn đủ 2/3 và bắn thêm lần nữa.
                self.confirm.clear(self.key)

        if hot_ids:
            self._clean_run = 0
        else:
            self._clean_run += 1
            if self._latched_cells and self._clean_run >= self.rearm_scans:
                # Vùng sạch hẳn đủ lâu -> xoá chốt, sẵn sàng cho sự kiện sau.
                self._latched_cells.clear()
                res.rearmed = True

    # ---- lưới + state ----

    def _paths(self) -> tuple[str, str]:
        base = os.path.join(self.state_dir or ".", self._slug)
        return base + ".clutter.json", base + ".ref.npz"

    def _ensure_grid(self, poly_norm: list, w: int, h: int) -> Grid:
        # So bằng VÂN TAY, không so bằng kích thước khung: người dùng vẽ lại vùng
        # ở cùng độ phân giải là chuyện thường (sửa zone trên UI), mà kiểu so cũ
        # sẽ bỏ qua và tiếp tục dùng lưới sai.
        poly_px = poly_to_px(poly_norm, w, h)
        sig = grid_signature(poly_px, w, h, self.cell_px, self.overlap)
        if self.grid is not None and self.grid.signature == sig:
            return self.grid

        if self.grid is not None:
            # Vùng vẽ lại / đổi độ phân giải: ID ô cũ không còn chỉ đúng chỗ.
            logger.info("[%s] lưới đổi (%s -> %s) -> bỏ tham chiếu + mặt nạ",
                        self.label, self.grid.signature, sig)
            self.ref.reset()
            self.clutter = None
        self.grid = build_grid(poly_px, w, h, cell_px=self.cell_px,
                               overlap=self.overlap)
        grid = self.grid

        if self.clutter is None:
            cc = self.cfg.get("clutter", {})
            kw = dict(min_scans=int(cc.get("mute_after_scans", 14_400)),
                      mute_ratio=float(cc.get("mute_ratio", 0.98)),
                      unmute_after_clean=int(cc.get("unmute_after_clean", 20)))
            cpath, rpath = self._paths()
            if self.state_dir:
                self.clutter = ClutterMask.load(cpath, grid.signature, **kw)
                self.ref.load(rpath, grid.signature)
            else:
                self.clutter = ClutterMask(grid.signature, **kw)
            logger.info("[%s] lưới %d ô (cell=%d, sig=%s), mặt nạ %d ô mute",
                        self.label, len(grid), grid.cell_px, grid.signature,
                        len(self.clutter.muted()))
        return grid

    def _stabilize(self, frame: np.ndarray, res: "ScanResult") -> np.ndarray:
        """Nắn khung hiện tại về khung mốc bằng ECC (dịch + xoay).

        Mốc được lấy lại mỗi lần nền được dựng lại, nên nó luôn cùng thời điểm
        với tham chiếu ô — bù về mốc cũ hơn thì tham chiếu và ảnh lệch nhau.

        Không hội tụ / lệch quá lớn thì TRẢ NGUYÊN khung: đó là camera bị xoay
        thật chứ không phải rung, để cổng dịch khung và guard toàn cục xử lý.
        Nắn đại một phép biến đổi sai còn tệ hơn không nắn.
        """
        ds = self.stab_ds
        small = gray_small(frame, ds)
        if self._stab_ref is None or self._stab_ref.shape != small.shape:
            self._stab_ref = small
            return frame
        est = estimate_warp(self._stab_ref, small, self.stab_iters)
        if est is None:
            return frame
        dx, dy, deg, W = est
        d = ((dx * ds) ** 2 + (dy * ds) ** 2) ** 0.5
        if d > self.stab_max_px or abs(deg) > self.stab_max_deg:
            return frame
        res.stab_px, res.stab_deg = d, deg
        # Dưới ngưỡng này thì nội suy chỉ làm ảnh nhoè thêm mà chẳng bù được gì —
        # mà nhoè lại chính là thứ cổng đổi nhìn thấy.
        # Xét TỔNG độ dịch mà phép nắn gây ra ở chỗ xa tâm nhất, không xét riêng
        # tịnh tiến và góc. Bản đầu đòi `d < min_px AND |deg| < min_deg`: với
        # deg=0,138 > min_deg thì điều kiện luôn sai nên nó LUÔN nắn, `min_px`
        # thành vô nghĩa — vùng chết không hề tồn tại.
        h0, w0 = frame.shape[:2]
        half_diag = (w0 * w0 + h0 * h0) ** 0.5 / 2.0
        max_disp = d + abs(np.sin(np.radians(deg))) * half_diag
        if max_disp < self.stab_min_px:
            return frame
        return apply_warp(frame, W, ds)

    def _run_verify(self, res: "ScanResult", frame: np.ndarray) -> None:
        """Cho detector phán lại các ô nóng. Không bật thì không làm gì."""
        if self.verifier is None or not self.verifier.enabled or not res.hot:
            return
        before = len(res.hot)
        try:
            keep, boxes = self.verifier.verify(frame, res.hot)
        except Exception as e:  # noqa: BLE001
            # Detector hỏng thì KHÔNG được im lặng nuốt cảnh báo — giữ nguyên ô
            # nóng và để hướng 1 quyết định như cũ. Mất precision còn hơn mất
            # cảnh báo mà không ai biết.
            logger.warning("[%s] tầng xác nhận lỗi (%s) -> giữ nguyên %d ô nóng",
                           self.label, e, before)
            res.verify_failed = True
            return
        res.verify_boxes = boxes
        res.n_verify_dropped = before - len(keep)
        res.hot = keep

    def reset_background(self, keep_clutter: bool = True) -> None:
        """Chốt lại nền: coi hiện trạng là chuẩn mới.

        Dùng khi cảnh cố định thay đổi hợp lệ — dời thùng rác, kê lại ghế, chuyển
        chậu cây. Không có cái này thì phải chờ mặt nạ nhiễu mute (~5 giờ), và
        suốt thời gian đó hệ thống báo lặp về một thay đổi mà người vận hành đã
        biết. Mặt nạ giữ nguyên theo mặc định vì nó học nhiễu chứ không học nền.
        """
        self.ref.reset()
        self._run.clear()
        self._latched_cells.clear()
        self._clean_run = 0
        self._stab_ref = None      # mốc bù méo phải cùng thời điểm với nền
        self.confirm.clear(self.key)
        if not keep_clutter and self.clutter is not None:
            self.clutter.reset()
        logger.info("[%s] CHỐT LẠI NỀN theo yêu cầu (giữ mặt nạ=%s)",
                    self.label, keep_clutter)

    def save_state(self) -> None:
        if not self.state_dir or self.clutter is None or self.grid is None:
            return
        cpath, rpath = self._paths()
        self.clutter.save(cpath)
        self.ref.save(rpath, self.grid.signature)

    # ---- quét ----

    def scan(self, frame: np.ndarray, poly_norm: list,
             person_boxes=(), now: float | None = None) -> ScanResult:
        t0 = time.perf_counter()
        now = time.time() if now is None else float(now)
        res = ScanResult(ts=now)

        h, w = frame.shape[:2]
        grid = self._ensure_grid(poly_norm, w, h)
        res.n_cells = len(grid)
        if not grid.cells:
            return res

        # 0) Camera bị xoay/va chạm -> mọi ID ô lệch, state cũ chỉ sai chỗ.
        thumb = make_thumb(frame)
        res.scene_shift_px = scene_shift(self._thumb, thumb, w)
        self._thumb = thumb
        if res.scene_shift_px >= self.shift_thr:
            logger.warning("[%s] khung dịch %.1fpx -> reset tham chiếu + mặt nạ",
                           self.label, res.scene_shift_px)
            self.ref.reset()
            if self.clutter is not None:
                self.clutter.reset()
            self._run.clear()
            self._stab_ref = None
            self.confirm.clear(self.key)
            return res

        # 0b) Rung nhỏ (gió, xe tải chạy qua, giãn nở nhiệt) — dưới ngưỡng trên
        #     nên không ai chặn, mà 2px đã đủ làm 86% số ô "đổi". Nắn về mốc.
        if self.stab_on:
            frame = self._stabilize(frame, res)

        # 1) Ô bị người/xe che: không sạch, không bẩn — lượt này không có dữ liệu.
        occ = occluded_ids(grid.cells, person_boxes, self.occlusion_thr,
                           self.occlusion_pad)
        res.n_occluded = len(occ)
        live = [c for c in grid.cells if c.id not in occ]

        # 2) Cổng đổi.
        # Giữ luôn `desc` để bước cập nhật tham chiếu bên dưới khỏi tính lại.
        cand: list[tuple[float, Cell, np.ndarray, np.ndarray | None]] = []
        for c in live:
            patch = frame[c.y1:c.y2, c.x1:c.x2]
            if patch.size == 0:
                continue
            if not self.change_on:
                cand.append((float("inf"), c, patch, None))
                continue
            desc = self.ref.describe(patch)
            hit, s = self.ref.changed(c.id, desc)
            if hit:
                cand.append((s, c, patch, desc))
        res.n_changed = len(cand)

        # ---- ĐỔI SÁNG TOÀN CỤC: nạp lại nền thay vì coi là phát hiện ----
        #
        # Vì sao bắt buộc: rạng sáng / bật đèn / camera chuyển IR<->màu làm MỌI ô
        # đổi cùng lúc. Tham chiếu chỉ học ở ô KHÔNG đổi, nên khi mọi ô đều đổi
        # thì không ô nào cập nhật được -> tham chiếu đóng băng vĩnh viễn -> cả
        # vùng nóng mãi mãi. Đo được trên camera thật: sau 7,5h chạy, lúc trời
        # sáng dần thì nóng=222/319 và nó KHÔNG BAO GIỜ tự thoát, cứ 30 phút
        # (cooldown) lại bắn một cảnh báo.
        #
        # Vật thật không bao giờ chiếm gần nửa vùng cùng lúc, nên ngưỡng này an
        # toàn: đánh đổi là mất đúng MỘT lượt quét mỗi lần đổi sáng.
        # Chỉ xét ô ĐÃ CÓ tham chiếu: lúc cold start mọi ô đều "đổi" vì chưa có
        # nền, không phải vì đổi sáng — tính cả chúng thì guard bắn nhầm.
        known_cells = [c for c in live if self.ref.has(c.id)]
        chg_cells = [c for _s, c, _p, _d in cand if self.ref.has(c.id)]
        known, chg_known = len(known_cells), len(chg_cells)

        # Ngoài SỐ LƯỢNG còn phải xét ĐỘ TRẢI. Đổi sáng chạm gần như mọi hàng và
        # mọi cột; một vật to (bao rác, thùng bị mang đi) chỉ chiếm một cụm liền.
        # Đo được: mang cái thùng rác đi làm 117/227 ô đổi — vượt ngưỡng số lượng
        # và suýt bị nuốt thành nền. Vùng càng nhỏ càng dễ dính.
        spread = 0.0
        if chg_known:
            rows_all = {c.row for c in known_cells}
            cols_all = {c.col for c in known_cells}
            fr = len({c.row for c in chg_cells}) / max(1, len(rows_all))
            fc = len({c.col for c in chg_cells}) / max(1, len(cols_all))
            spread = min(fr, fc)

        if (self.change_on
                and known >= max(8, self.global_min_visible * len(grid.cells))
                and chg_known >= self.global_frac * known
                and spread >= self.global_spread):
            for c in live:
                p = frame[c.y1:c.y2, c.x1:c.x2]
                if p.size:
                    self.ref.observe_clean(c.id, self.ref.describe(p))
                self._run[c.id] = 0
            # Nền vừa học lại từ khung này -> mốc bù méo phải lấy lại từ đúng
            # khung đó, không thì bù về một thời điểm mà nền không còn khớp.
            self._stab_ref = None
            res.global_change = True
            logger.info("[%s] đổi sáng toàn cục (%d/%d ô, trải %.0f%%) -> nạp lại "
                        "nền, bỏ lượt này", self.label, chg_known, known, spread * 100)
            self.confirm.passed(self.key, False, self.n, self.m)
            res.ms = (time.perf_counter() - t0) * 1000.0
            return res

        # ---- chế độ change_only: không có model, quyết bằng THỜI GIAN ----
        if self.mode == "change_only":
            changed = {c.id for _, c, _p, _d in cand}
            for c in live:
                patch = frame[c.y1:c.y2, c.x1:c.x2]
                if patch.size == 0:
                    continue
                # Ô chưa có tham chiếu: CHỐT NỀN, không tính là đổi.
                # Không có classifier thì không còn gì để dựng tham chiếu, nên
                # lượt đầu phải tự nhận nền — nếu không, mọi ô đều "đổi" và sau
                # `dwell` lượt là cả vùng nóng (đo được: 59/68 ô).
                # Hệ quả: rác CÓ SẴN lúc khởi động sẽ thành nền. Đúng cho bài
                # toán "vùng trống" — bật lúc vùng sạch; muốn reset thì xoá state.
                if self.change_on and not self.ref.has(c.id):
                    self.ref.observe_clean(c.id, self.ref.describe(patch))
                    self._run[c.id] = 0
                    res.scores[c.id] = 0.0
                    continue
                if c.id in changed:
                    self._run[c.id] = self._run.get(c.id, 0) + 1
                else:
                    self._run[c.id] = 0
                    # Ô trùng nền lại -> cho tham chiếu học (người đã đi khỏi).
                    if self.change_on:
                        self.ref.observe_clean(c.id, self.ref.describe(patch))
                raw = self._run.get(c.id, 0) >= self.dwell
                res.scores[c.id] = min(1.0, self._run.get(c.id, 0) / max(1, self.dwell))
                if self.clutter_on:
                    self.clutter.update(c.id, raw)
                if raw:
                    res.raw_hot.append(c)
                muted = self.clutter_on and self.clutter.is_muted(c.id)
                if muted:
                    if raw:
                        res.n_muted_hit += 1
                    # Đã mute -> coi như nền, cho tham chiếu học để thôi tốn lượt.
                    if self.change_on:
                        self.ref.observe_clean(c.id, self.ref.describe(patch))
                        self._run[c.id] = 0
                elif raw:
                    res.hot.append(c)
            res.n_scored = len(live)
            self._run_verify(res, frame)
            res.dirty = len(res.hot) >= self.min_hot
            res.mask_progress = self.clutter.progress() if self.clutter_on else 1.0
            self._decide_alert(res, now)
            self._scans += 1
            res.ms = (time.perf_counter() - t0) * 1000.0
            return res

        # Cold start: chưa có tham chiếu -> MỌI ô đều "đổi" -> phải hỏi hết.
        # Với backend onnx thì kệ (rẻ), nhưng với vlm là 264 call chỉ để khởi
        # động, nhân với số camera thì thành hàng giờ GPU. Nếu lúc lắp đặt vùng
        # đang sạch thì bật cờ này: lượt đầu chỉ chụp lấy nền, không hỏi ai.
        # Mặc định TẮT — vùng đang có rác sẵn mà bật thì rác đó thành "nền".
        if self.change_on and self.assume_clean_start and len(self.ref) == 0 and cand:
            for _, c, _p, desc in cand:
                if desc is not None:
                    self.ref.observe_clean(c.id, desc)
            logger.info("[%s] cold start: lấy %d ô làm nền sạch, bỏ qua lượt này",
                        self.label, len(cand))
            res.ms = (time.perf_counter() - t0) * 1000.0
            return res

        # 3) Trần số ô gửi xuống classifier (bảo vệ backend VLM). Ưu tiên ô đổi
        #    mạnh nhất. Bị cắt thì GHI LOG — cắt im lặng đọc thành "đã quét hết".
        cand.sort(key=lambda t: t[0], reverse=True)
        if self.max_score > 0 and len(cand) > self.max_score:
            res.n_dropped = len(cand) - self.max_score
            logger.warning("[%s] %d ô đổi vượt trần %d -> bỏ qua lượt này",
                           self.label, len(cand), self.max_score)
            cand = cand[: self.max_score]

        # 4) Chấm điểm.
        patches = [p for _, _, p, _ in cand]
        scores = self.scorer.score(patches) if patches else []
        notes = getattr(self.scorer, "last_notes", None)
        res.n_scored = len(scores)

        for i, (_, c, _patch, desc) in enumerate(cand):
            s = float(scores[i])
            res.scores[c.id] = s
            if notes and i < len(notes) and notes[i]:
                res.notes[c.id] = notes[i]
            raw_litter = s >= self.litter_thr

            # Mặt nạ nhiễu chỉ đếm ô THỰC SỰ ĐƯỢC HỎI. Ô bị cổng đổi chặn thì
            # ta có hỏi đâu mà tính. Nhờ vậy tỉ lệ "bẩn/lần được hỏi" của nắp
            # cống là ~100% và nó bị mute, còn rác thật thì tỉ lệ thấp.
            if self.clutter_on:
                self.clutter.update(c.id, raw_litter)

            if raw_litter:
                res.raw_hot.append(c)

            muted = self.clutter_on and self.clutter.is_muted(c.id)
            if muted and raw_litter:
                res.n_muted_hit += 1

            # Tham chiếu CHỈ học ở ô sạch — nếu học vô điều kiện thì rác nằm lâu
            # bị nuốt vào nền rồi biến mất. Ô đã mute coi như sạch, để cổng đổi
            # chặn luôn nó ở các lượt sau (tiết kiệm cả FP lẫn compute).
            if self.change_on and desc is not None and (not raw_litter or muted):
                self.ref.observe_clean(c.id, desc)

            if raw_litter and not muted:
                res.hot.append(c)

        # 5) Kết luận lượt + hai cổng.
        self._run_verify(res, frame)
        res.dirty = len(res.hot) >= self.min_hot
        res.mask_progress = self.clutter.progress() if self.clutter_on else 1.0
        self._decide_alert(res, now)
        self._scans += 1
        res.ms = (time.perf_counter() - t0) * 1000.0
        return res
