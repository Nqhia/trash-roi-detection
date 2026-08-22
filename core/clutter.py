"""Mặt nạ nhiễu tự học — thứ duy nhất diệt được FP HỆ THỐNG.

Vấn đề: nắp cống, vệt sơn, ổ gà, gạch vỡ trông y như rác ở 27px. Classifier
chấm chúng "bẩn" ở MỌI lượt quét, nên ConfirmGate 4/6 vô dụng (chúng lặp 6/6).

Nhận xét cứu cánh: rác thật thì LÚC CÓ LÚC KHÔNG (bị vứt rồi được dọn). Thứ mà
5 ngày liền lượt nào cũng "bẩn" gần như chắc chắn là một phần cố định của cảnh.

    ô r3c0:  14.400 lượt,  14.397 lần chấm bẩn  -> 99.98%  ->  MUTE
    ô r1c2:  14.400 lượt,      47 lần chấm bẩn  ->  0.33%  ->  bình thường

Ưu điểm so với cách so ảnh nền theo pixel: nó chạy trên QUYẾT ĐỊNH của
classifier chứ không trên pixel, nên nắng/mưa/sáng tối đổi không làm nó vỡ, và
không cần quy trình "bấm nút xác nhận vùng đang sạch" — thứ thực tế không ai
bấm đúng lúc.

Hai rủi ro đã tính tới:
- Điểm bị đổ rác KINH NIÊN cũng có thể bị mute nhầm. Chặn bằng ngưỡng
  `mute_ratio` rất cao (0.98) + `mute_after_days` dài, và `muted()` trả danh
  sách để UI cho người dùng bỏ mute tay.
- Camera bị xoay/va chạm -> ID ô lệch hết. Xử lý ở pipeline.py (dò dịch chuyển
  khung hình -> gọi reset()).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict


@dataclass(slots=True)
class CellStat:
    total: int = 0          # số lượt ô này thực sự được đánh giá (bỏ lượt bị che)
    litter: int = 0         # số lượt bị chấm "bẩn" (quyết định THÔ, trước mặt nạ)
    clean_streak: int = 0   # số lượt sạch liên tiếp gần nhất -> dùng để bỏ mute
    muted: bool = False


class ClutterMask:
    """Đếm theo ô, bền vững qua restart (lưu JSON theo camera+vùng)."""

    def __init__(
        self,
        signature: str,
        min_scans: int = 14_400,     # ~5 ngày ở nhịp 30s
        mute_ratio: float = 0.98,
        unmute_after_clean: int = 20,
    ) -> None:
        self.signature = signature
        self.min_scans = int(min_scans)
        self.mute_ratio = float(mute_ratio)
        self.unmute_after_clean = int(unmute_after_clean)
        self._stats: dict[tuple[int, int], CellStat] = {}

    # ---- cập nhật ----

    def update(self, cell_id, raw_litter: bool) -> None:
        """Ghi nhận quyết định THÔ của classifier cho ô này.

        Phải gọi với quyết định TRƯỚC khi lọc mặt nạ, nếu không ô đã mute sẽ
        không bao giờ tích luỹ thêm số liệu và không bao giờ được bỏ mute.
        Chỉ gọi cho ô THỰC SỰ được đánh giá lượt này (bỏ ô bị người/xe che).
        """
        st = self._stats.get(cell_id)
        if st is None:
            st = self._stats[cell_id] = CellStat()
        st.total += 1
        if raw_litter:
            st.litter += 1
            st.clean_streak = 0
        else:
            st.clean_streak += 1

        if st.muted:
            # Bỏ mute khi ô đã sạch liên tục đủ lâu (ai đó dời cái thùng đi).
            # Xoá sạch số liệu để nó học lại từ đầu, không mang định kiến cũ.
            if st.clean_streak >= self.unmute_after_clean:
                self._stats[cell_id] = CellStat()
        elif st.total >= self.min_scans and (st.litter / st.total) >= self.mute_ratio:
            st.muted = True

    def is_muted(self, cell_id) -> bool:
        st = self._stats.get(cell_id)
        return bool(st and st.muted)

    def muted(self) -> list[tuple[int, int]]:
        """Danh sách ô đang bị mute — đẩy lên UI cho người dùng bỏ mute tay."""
        return sorted(k for k, v in self._stats.items() if v.muted)

    def unmute(self, cell_id) -> None:
        self._stats.pop(cell_id, None)

    def reset(self) -> None:
        self._stats.clear()

    def progress(self) -> float:
        """0..1 — mức độ "chín" của mặt nạ, để biết còn phải chạy shadow bao lâu."""
        if not self._stats:
            return 0.0
        top = max(s.total for s in self._stats.values())
        return min(1.0, top / float(self.min_scans))

    # ---- lưu/nạp ----

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        blob = {
            "signature": self.signature,
            "min_scans": self.min_scans,
            "mute_ratio": self.mute_ratio,
            "unmute_after_clean": self.unmute_after_clean,
            "stats": {f"{r},{c}": asdict(s) for (r, c), s in self._stats.items()},
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(blob, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, signature: str, **kw) -> "ClutterMask":
        """Nạp mặt nạ. Vân tay lưới lệch (vùng vẽ lại / đổi độ phân giải / đổi
        cỡ ô) thì VỨT hết và học lại — ID ô cũ không còn chỉ đúng chỗ nữa."""
        m = cls(signature, **kw)
        try:
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
        except (OSError, ValueError):
            return m
        if blob.get("signature") != signature:
            return m
        for k, v in (blob.get("stats") or {}).items():
            try:
                r, c = k.split(",")
                m._stats[(int(r), int(c))] = CellStat(**v)
            except (ValueError, TypeError):
                continue
        return m
