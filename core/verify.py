"""Tầng XÁC NHẬN: detector phán các ô nóng do cổng đổi chỉ ra có phải rác không.

Vì sao đặt ở đây chứ không nhét vào `scorers.py`: scorer nhận một danh sách ô
rời, không biết ô nào cạnh ô nào. Mà detector cần BỐI CẢNH — một ô 48px đưa
thẳng vào thì nó không nhận ra thứ gì. Tầng này gom ô kề nhau thành vùng rồi
mới hỏi, nên phải nằm ở chỗ còn giữ thông tin không gian.

Đo được trên chuỗi CCTV sạch (30 lượt, có người đi và ánh sáng đổi):

    chỉ cổng đổi     67% số lượt có báo
    chỉ detector     93% số lượt có báo
    ghép hai tầng    37% số lượt có báo      recall khung eco giữ nguyên 75%

Lý do ghép ăn tiền: hai tầng sai ở CHỖ KHÁC NHAU. Cổng đổi bắn vì người đi qua
và ánh sáng đổi; detector bắn vì kết cấu bề mặt trông giống rác. Giao của hai
tập nhỏ hơn hẳn từng tập. Sáu lượt bị bác bỏ khi soi bằng mắt đều là người đang đi.

Và nó rẻ: gộp xong còn ~1 vùng mỗi lượt thay vì 9-12 ô quét toàn vùng.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("trash.verify")


class RegionVerifier:
    """Gom ô nóng -> vùng -> detector -> giữ lại ô được xác nhận."""

    def __init__(self, cfg: dict) -> None:
        v = cfg or {}
        self.enabled = bool(v.get("enabled", False))
        self.weights = str(v.get("weights", ""))
        self.conf = float(v.get("conf", 0.10))
        # Ô kề nhau trong bán kính này thì coi là cùng một vùng.
        self.gap = int(v.get("cluster_gap", 1))
        self.pad = int(v.get("pad_px", 48))
        # Trần cạnh vùng. Vùng càng to thì vật càng nhỏ so với ảnh đưa vào model
        # — đúng vấn đề tỉ lệ tín hiệu/nền mà cả hướng này sinh ra để tránh.
        self.max_side = int(v.get("max_side_px", 256))
        self.min_side = int(v.get("min_side_px", 160))
        self.upscale = float(v.get("upscale", 2.0))
        self.max_regions = int(v.get("max_regions", 8))
        self._m = None

    def _model(self):
        if self._m is None:
            from ultralytics import YOLO      # nạp lười: selftest không cần
            self._m = YOLO(self._resolve(self.weights))
        return self._m

    @staticmethod
    def _resolve(w: str) -> str:
        """Đường dẫn tương đối phải neo vào GÓC GÓI, không vào thư mục đang đứng.

        Bẫy thật, đã dính: chạy tool từ thư mục khác thì `models/trash_yolo11n.pt`
        không giải được, ultralytics coi đó là TÊN MODEL và đi tải từ GitHub. Gặp
        403 rate limit -> ném lỗi -> `_run_verify` giữ nguyên ô nóng (đúng thiết
        kế: detector hỏng thì không được nuốt cảnh báo). Kết quả là pipeline chạy
        KHÔNG CÓ tầng xác nhận mà nhìn vào output vẫn thấy hợp lý — cùng một
        video ra 28 lượt hay 2 lượt tuỳ chỗ đứng lúc gõ lệnh.
        """
        import os
        if not w or os.path.isabs(w) or os.path.exists(w):
            return w
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cand = os.path.join(root, w)
        return cand if os.path.exists(cand) else w

    def clusters(self, cells: list) -> list:
        todo, out = list(cells), []
        while todo:
            grp = [todo.pop()]
            moved = True
            while moved:
                moved = False
                for c in list(todo):
                    if any(abs(c.row - g.row) <= self.gap
                           and abs(c.col - g.col) <= self.gap for g in grp):
                        grp.append(c)
                        todo.remove(c)
                        moved = True
            out.append(grp)
        return out

    def region_of(self, grp: list, w: int, h: int) -> tuple:
        x1 = min(c.x1 for c in grp) - self.pad
        y1 = min(c.y1 for c in grp) - self.pad
        x2 = max(c.x2 for c in grp) + self.pad
        y2 = max(c.y2 for c in grp) + self.pad
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = min(self.max_side, max(self.min_side, x2 - x1, y2 - y1))
        x1 = int(max(0, min(w - side, cx - side / 2)))
        y1 = int(max(0, min(h - side, cy - side / 2)))
        return x1, y1, int(min(w, x1 + side)), int(min(h, y1 + side))

    def verify(self, frame: np.ndarray, cells: list) -> tuple:
        """-> (ô được xác nhận, hộp phát hiện ở toạ độ khung gốc).

        Không xác nhận được thì trả rỗng, tức là KHÔNG báo. Đây là lựa chọn có
        đánh đổi: detector đạt 75% trên khung eco nên nó sẽ vứt đi một phần vật
        mà cổng đổi bắt đúng. Đổi lại báo nhầm giảm gần hai lần. Muốn ưu tiên
        không bỏ sót thì tắt tầng này bằng `verify.enabled: false`.
        """
        import cv2

        if not cells:
            return [], []
        h, w = frame.shape[:2]
        groups = self.clusters(cells)
        if len(groups) > self.max_regions:
            # Quá nhiều cụm = cảnh đang loạn (đổi sáng, gió). Lấy cụm to nhất.
            groups.sort(key=len, reverse=True)
            groups = groups[: self.max_regions]
        regions = [self.region_of(g, w, h) for g in groups]

        crops, offs = [], []
        for (x1, y1, x2, y2) in regions:
            c = frame[y1:y2, x1:x2]
            if c.size == 0:
                continue
            s = (self.max_side * self.upscale) / max(c.shape[:2])
            crops.append(cv2.resize(c, None, fx=s, fy=s,
                                    interpolation=cv2.INTER_CUBIC))
            offs.append((x1, y1, s))
        if not crops:
            return [], []

        boxes = []
        for (ox, oy, s), r in zip(offs, self._model().predict(
                crops, conf=self.conf, verbose=False)):
            for b in r.boxes:
                bx = b.xyxy[0].tolist()
                # Phần tử thứ 5 là ĐIỂM TIN CẬY. Trước đây chỉ trả toạ độ, nên
                # không ai trả lời được "detector tự tin bao nhiêu khi nó gật" —
                # mà chính câu đó quyết định nên chỉnh ngưỡng hay phải train lại.
                boxes.append((ox + bx[0] / s, oy + bx[1] / s,
                              ox + bx[2] / s, oy + bx[3] / s, float(b.conf[0])))

        return self._keep(cells, boxes), boxes

    @staticmethod
    def _keep(cells: list, boxes: list) -> list:
        """Ô được xác nhận = ô có giao với ít nhất một hộp phát hiện."""
        out = []
        for c in cells:
            for bx in boxes:
                if (c.x1 < bx[2] and c.x2 > bx[0]
                        and c.y1 < bx[3] and c.y2 > bx[1]):
                    out.append(c)
                    break
        return out

    def sweep(self, frame: np.ndarray, cells: list, max_tiles: int = 24) -> tuple:
        """Quét detector KHẮP vùng, không cần ô nóng dẫn đường.

        Dùng sau khi nền bị vứt: lúc đó cổng đổi mù (không còn cái gì để so),
        nên detector là thứ duy nhất nhìn thấy vật đã nằm sẵn trong vùng.

        Không dùng lại `verify()` được: nó gom ô kề nhau thành MỘT cụm rồi cắt
        một vùng `max_side`, đưa cả vùng giám sát vào thì nó chỉ soi đúng một
        mảnh 320px ở giữa. Chỗ này lát toàn vùng thành ô `max_side` chồng nhau
        50% — đúng cách dữ liệu train được cắt.
        """
        import cv2

        if not cells:
            return [], []
        h, w = frame.shape[:2]
        x1 = int(max(0, min(c.x1 for c in cells) - self.pad))
        y1 = int(max(0, min(c.y1 for c in cells) - self.pad))
        x2 = int(min(w, max(c.x2 for c in cells) + self.pad))
        y2 = int(min(h, max(c.y2 for c in cells) + self.pad))
        side, step = self.max_side, max(1, self.max_side // 2)
        xs = list(range(x1, max(x1 + 1, x2 - side + 1), step))
        ys = list(range(y1, max(y1 + 1, y2 - side + 1), step))
        if xs[-1] + side < x2:
            xs.append(max(x1, x2 - side))
        if ys[-1] + side < y2:
            ys.append(max(y1, y2 - side))
        tiles = [(x, y) for y in ys for x in xs]
        if len(tiles) > max_tiles:
            # Vùng quá to so với trần: giãn thưa ra thay vì cắt cụt một nửa vùng,
            # thà thưa đều còn hơn mù hẳn phần dưới. NHƯNG phải kêu lên — cắt bớt
            # trong im lặng thì nhìn vào kết quả cứ tưởng đã quét hết vùng.
            k = (len(tiles) + max_tiles - 1) // max_tiles
            logger.warning("quét toàn vùng: cần %d ô lát, trần %d -> lấy thưa "
                           "1/%d, CÓ THỂ SÓT. Thu hẹp vùng hoặc nâng max_tiles.",
                           len(tiles), max_tiles, k)
            tiles = tiles[::k][:max_tiles]

        crops, offs = [], []
        for (tx, ty) in tiles:
            c = frame[ty:min(h, ty + side), tx:min(w, tx + side)]
            if c.size == 0:
                continue
            s = (self.max_side * self.upscale) / max(c.shape[:2])
            crops.append(cv2.resize(c, None, fx=s, fy=s,
                                    interpolation=cv2.INTER_CUBIC))
            offs.append((tx, ty, s))
        if not crops:
            return [], []

        boxes = []
        for (ox, oy, s), r in zip(offs, self._model().predict(
                crops, conf=self.conf, verbose=False)):
            for b in r.boxes:
                bx = b.xyxy[0].tolist()
                boxes.append((ox + bx[0] / s, oy + bx[1] / s,
                              ox + bx[2] / s, oy + bx[3] / s, float(b.conf[0])))
        return self._keep(cells, boxes), boxes
