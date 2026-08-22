"""ConfirmGate + DedupGate — bản độc lập, API giống core/gates.py của worker.

Giữ nguyên chữ ký hàm để khi bê module này vào ai-worker thì import thẳng
`core.gates` của worker, xoá file này, không phải sửa chỗ gọi.

Vì sao cần cả hai:
- ConfirmGate (N/M): diệt FP NGẪU NHIÊN (nhiễu nén, bóng thoáng qua). Một ô
  nhiễu ngẫu nhiên không lặp lại đúng chính nó 4 trong 6 lượt. Gần như miễn phí.
- DedupGate (cooldown): chống spam. "Thấy rác là báo" nhưng rác nằm 3 tiếng thì
  không được bắn 360 lần.

Cả hai đều BẤT LỰC với FP HỆ THỐNG (nắp cống lượt nào cũng bị chấm bẩn, lặp
6/6) — đó là việc của clutter.py.
"""

from __future__ import annotations


def _key_matches_camera(key, camera_id: str) -> bool:
    if isinstance(key, tuple):
        return bool(key) and camera_id in key
    return key == camera_id


class ConfirmGate:
    """>= n True trong m quan sát gần nhất."""

    def __init__(self) -> None:
        self._hist: dict = {}

    def passed(self, key, present: bool, n: int, m: int) -> bool:
        m = max(1, int(m))
        n = max(1, min(int(n), m))
        hist = self._hist.setdefault(key, [])
        hist.append(bool(present))
        if len(hist) > m:
            del hist[: len(hist) - m]
        return sum(hist) >= n

    def clear(self, key) -> None:
        self._hist.pop(key, None)

    def is_all_absent(self, key) -> bool:
        return not any(self._hist.get(key) or [])

    def forget_camera(self, camera_id: str) -> None:
        for k in [k for k in self._hist if _key_matches_camera(k, camera_id)]:
            del self._hist[k]


class DedupGate:
    """Mỗi khoá chỉ pass một lần trong window_s giây."""

    def __init__(self) -> None:
        self._last: dict = {}

    def passed(self, key, now: float, window_s: float) -> bool:
        # Sentinel None chứ không phải 0.0: bản trong worker dùng
        # `self._last.get(key, 0.0)` nên nếu `now` là số nhỏ (monotonic ngay sau
        # khi máy boot, hoặc mốc thời gian video bắt đầu từ 0) thì lần bắn ĐẦU
        # TIÊN bị nuốt. Sửa ở đây, API giữ nguyên.
        last = self._last.get(key)
        if last is not None and now - last < window_s:
            return False
        self._last[key] = now
        return True

    def peek_last(self, key) -> float:
        return self._last.get(key, 0.0)

    def force(self, key, now: float) -> None:
        """Ép mở cổng (dùng cho leo thang: rác tăng thêm thì báo lại sớm)."""
        self._last[key] = now

    def forget_camera(self, camera_id: str) -> None:
        for k in [k for k in self._last if _key_matches_camera(k, camera_id)]:
            del self._last[k]
