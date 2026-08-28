"""TrashConsumer — cắm pipeline này vào worker EcoVision (detect/base.py §3-§4).

Thả file này vào `detect/trash/pipeline.py` của worker và đăng ký ở
`detect/registry.py`. Ruột (`core/`) không đổi một dòng nào.

BA ĐIỀU PHẢI BIẾT TRƯỚC KHI CẮM
-------------------------------

1. `"trash"` chưa có trong `ALLOWED_LABELS` (detect/base.py). Phải thêm, không
   thì engine lọc sạch mọi Detection của module này và nó im lặng không báo gì.
   Đây là thay đổi DUY NHẤT cần làm ở repo worker.

2. KHÔNG khai `requires=("object",)`. Ghi chú cũ trong tools/run_video.py nói
   ngược lại — đó là bản trước khi đo. Che ô theo box người đã bị bỏ: đo được
   nó giảm ĐÚNG 0 cảnh báo trong khi bịt mắt tới 24% diện tích vùng. Tầng xác
   nhận lo phần người đứng lại (ô nóng trên người 30 -> 4). Khai `requires` chỉ
   tổ ép engine chạy thêm detector object cho camera chỉ bật mỗi rác.

3. Module này GIỮ STATE theo vùng (nền EMA, bộ đếm bền vững, mặt nạ nhiễu, chốt
   cảnh báo) — đây là chỗ lệch "luật vàng §2". Không giữ thì không có bài toán:
   toàn bộ cách phân biệt rác với người đi qua nằm ở chỗ so với nền và đếm số
   lượt liên tiếp, mà cell ID là khái niệm riêng của module, engine không biểu
   diễn được. State chỉ theo camera/zone, không đụng MQTT, không lưu snapshot.

   Riêng phần CHỐNG TRÙNG thì có chồng lấn thật với tầng rule của engine. Nếu
   bật rule N/M cho nhãn `trash`, hãy đặt trong config của module:
       confirm: {n: 1, m: 1}
       alert:   {mode: cooldown, cooldown_s: 0}
   để khỏi lọc hai lần (lọc hai lần = trả giá độ trễ hai lần, đo được 4,0 phút
   so với 1,5 phút).
"""

from __future__ import annotations

import logging
import os

import yaml

from core.pipeline import ZoneTrashDetector
from core.scorers import build_scorer

logger = logging.getLogger("trash")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.environ.get("TRASH_CFG", os.path.join(HERE, "config", "day_cfg.yaml"))
STATE = os.environ.get("TRASH_STATE", os.path.join(HERE, "runs", "state"))


class TrashConsumer:
    name = "trash"
    labels = ["trash"]             # PHẢI thêm vào ALLOWED_LABELS ở detect/base.py
    source = "ai"
    roi_mode = "mask"              # cần polygon vùng thật, không crop bbox
    distinct_frames = True
    cost_class = "light"           # 206 ms/lượt ở nhịp 30s = 0,7% thời gian máy
    max_objects = None
    runtime_key = "model_trash_enabled"
    display_label = "Phát hiện rác trong vùng"
    display_hint = ("Vùng trống có vật lạ nằm lại — vẽ vùng rồi báo. Không phân "
                    "loại rác, không cần dữ liệu train của camera đó.")
    engine_managed = True
    tracked = False                # vật nằm yên, không có gì để track
    apply_show_filter = True

    def __init__(self) -> None:
        with open(CFG, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        # Đường dẫn model trong config là TƯƠNG ĐỐI với thư mục gốc gói này.
        v = self.cfg.get("verify", {})
        if v.get("weights") and not os.path.isabs(v["weights"]):
            v["weights"] = os.path.join(HERE, v["weights"])
        self._det: dict = {}        # (camera_id, zone_idx) -> ZoneTrashDetector
        self._nscan: dict = {}      # (camera_id, zone_idx) -> số lượt, để lưu state
        os.makedirs(STATE, exist_ok=True)

    @staticmethod
    def module_on() -> bool:
        from config import settings
        return settings.MODULE_AI

    def interval_s(self) -> float:
        # 30s không phải để tiết kiệm máy (206 ms/lượt) mà vì `dwell_scans` đang
        # đo THỜI GIAN TỒN TẠI: 3 lượt x 30s = 90 giây. Quét dày hơn mà giữ
        # nguyên dwell thì người đứng xem điện thoại 30 giây cũng đủ điều kiện.
        #
        # BẪY: nếu sau này nhịp bị chỉnh từ runtime config/UI thì PHẢI chỉnh
        # `decide.dwell_scans` theo tỉ lệ nghịch, không thì khả năng phân biệt
        # rác với người đi qua đổi theo mà không ai thấy. dwell đếm LƯỢT, chỉ có
        # `interval_s` mới quy nó ra giây.
        return float(self.cfg.get("scan", {}).get("interval_s", 30))

    def camera_on(self, cam: dict) -> bool:
        # Mặc định TẮT (khác weapon mặc định bật): module này vô nghĩa khi chưa
        # ai vẽ vùng, nên phải là bật-có-chủ-ý.
        return bool(cam.get("trash_enabled", False))

    def _for(self, camera_id: str, idx: int) -> ZoneTrashDetector:
        key = (camera_id, idx)
        if key not in self._det:
            self._det[key] = ZoneTrashDetector(
                self.cfg, build_scorer(self.cfg.get("scorer", {})),
                camera_id=camera_id, zone_id=f"z{idx}", state_dir=STATE)
        return self._det[key]

    def detect(self, ctx) -> list:
        # roi.for_detector() lọc zone theo `classes` giao với `labels`, nên zone
        # trên UI phải được gán lớp `trash` — không thì polygons rỗng và module
        # im lặng dù người dùng đã vẽ vùng.
        polys = (ctx.roi.polygons or []) if ctx.roi else []
        if not polys:
            # Không vẽ vùng thì không có bài toán: "vùng này có còn trống không"
            # chỉ có nghĩa khi biết vùng nào. Im lặng, không đoán cả khung.
            return []
        from detect.base import Detection      # chỉ có trong worker
        h, w = ctx.image.shape[:2]
        cam_id = str(ctx.camera.get("id", "cam"))
        out = []
        for i, poly in enumerate(polys):
            norm = [[float(x) / w, float(y) / h] for x, y in poly]
            det = self._for(cam_id, i)
            res = det.scan(ctx.image, norm, (), now=ctx.t)
            # LƯU state định kỳ và ngay khi báo. Bản đầu chỉ NẠP mà quên LƯU:
            # worker restart là cold start — mất nền lẫn mặt nạ đã học 5 giờ,
            # và rác đang nằm trong vùng thành nền mới (đường nuốt rác thứ 5).
            k2 = (cam_id, i)
            self._nscan[k2] = self._nscan.get(k2, 0) + 1
            if res.alert or self._nscan[k2] % 10 == 0:
                det.save_state()
            if not res.alert:
                continue
            for bx in (res.verify_boxes or self._cell_boxes(res)):
                # score=1.0 cố định: tới đây chuỗi cổng (bền vững -> xác nhận
                # -> chốt) ĐÃ quyết rồi, không còn đại lượng liên tục nào để
                # trả. Điểm thô của detector không phải xác suất "có phải rác",
                # đo được là ngưỡng tối ưu của nó không chuyển được sang tầng
                # ghép. Đừng dựng rule lọc theo score cho nhãn này.
                # bx[:4] — hộp từ tầng xác nhận có 5 phần tử (phần tử thứ 5 là
                # điểm tin cậy). Detection.bbox phải ĐÚNG 4, đưa cả 5 vào là
                # engine dịch toạ độ sai mà không báo lỗi gì.
                out.append(Detection(
                    label="trash", score=1.0, bbox=tuple(float(v) for v in bx[:4]),
                    source=self.name,
                    extra={"zone": i, "n_hot": len(res.hot),
                           "det_conf": round(float(bx[4]), 3) if len(bx) > 4 else None,
                           "n_fresh": res.n_fresh_hot,
                           "escalated": bool(res.escalated),
                           "from_sweep": res.n_sweep_hot > 0}))
        return out

    @staticmethod
    def _cell_boxes(res) -> list:
        """Tầng xác nhận tắt -> không có hộp detector, lấy bao của ô nóng."""
        if not res.hot:
            return []
        return [(min(c.x1 for c in res.hot), min(c.y1 for c in res.hot),
                 max(c.x2 for c in res.hot), max(c.y2 for c in res.hot))]

    def on_camera_removed(self, camera_id: str) -> None:
        for key in [k for k in self._det if k[0] == camera_id]:
            self._det[key].save_state()          # giữ nền/mặt nạ cho lần bật lại
            self._det.pop(key, None)
            self._nscan.pop(key, None)
