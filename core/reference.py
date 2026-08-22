"""Tham chiếu từng ô + cổng đổi — bộ lọc rẻ chạy TRƯỚC classifier.

Hai việc cùng lúc:
1. Giảm chi phí. Ở trạng thái ổn định gần như không ô nào đổi, nên số ô phải
   đưa xuống classifier ~0. Đây là thứ khiến backend VLM khả thi: không có cổng
   này thì 264 ô x mỗi 30s = 8.8 call VLM/giây/camera, sai bậc độ lớn.
2. Giảm FP. Vệt bẩn cố định giống hệt nền -> không bao giờ đổi -> không bao giờ
   được hỏi.

MẸO QUAN TRỌNG: tham chiếu CHỈ cập nhật ở ô đang được coi là SẠCH. Nếu cập nhật
vô điều kiện (kiểu MOG2) thì rác nằm lâu sẽ bị nuốt vào nền rồi biến mất khỏi
cảnh báo — đúng cái bẫy phải tránh với bài toán này.

Mô tả ô: xám -> thu về 8x8 -> trừ giá trị trung bình của chính nó. Trừ trung
bình để bền với thay đổi độ sáng toàn cục (mây che, đèn đường bật): cả ô sáng
lên đều thì mô tả không đổi, chỉ khi có VẬT xuất hiện mới đổi.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

DESC = 8    # cạnh ảnh mô tả (8x8 = 64 số/ô, đủ để biết "có vật lạ", đủ rẻ)
TOPK = 8    # số ô mô tả sáng nhất được lấy trung bình — xem `change()`


class CellReference:
    def __init__(self, alpha: float = 0.05, change_thr: float = 6.0) -> None:
        self.alpha = float(alpha)          # EMA: 0.05 ~ hằng số thời gian ~20 lượt
        self.change_thr = float(change_thr)  # thang 0..255
        self._ref: dict[tuple[int, int], np.ndarray] = {}

    @staticmethod
    def describe(patch: np.ndarray) -> np.ndarray:
        if patch.ndim == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        d = cv2.resize(patch, (DESC, DESC), interpolation=cv2.INTER_AREA)
        d = d.astype(np.float32)
        return d - float(d.mean())

    def change(self, cell_id, desc: np.ndarray) -> float:
        """Điểm đổi so với tham chiếu. Chưa có tham chiếu -> inf (phải hỏi).

        Lấy trung bình TOPK ô mô tả lệch nhất, KHÔNG lấy trung bình cả 64 ô.
        Lý do: vật cần bắt rất nhỏ so với ô. Vỏ chai 27x8px chiếm ~5% diện tích
        ô 64px, nên trung bình toàn cục bị pha loãng còn

            2 x f x (1-f) x Δ  =  2 x 0.053 x 0.947 x 40  ~  4.0

        (f = tỉ lệ diện tích, Δ = chênh mức xám), tức là lẫn vào nhiễu nén.
        Lấy top-8 thì chính mấy ô mô tả trùng vỏ chai được tính, cho ~15-20 —
        tách bạch hẳn với nhiễu (~0.5 sau khi 8x8 làm trơn). Bản đầu dùng
        trung bình toàn cục và bỏ sót đúng cỡ vật mà cả dự án này nhắm tới.
        """
        ref = self._ref.get(cell_id)
        if ref is None:
            return float("inf")
        d = np.abs(desc - ref).ravel()
        if d.size > TOPK:
            d = np.partition(d, -TOPK)[-TOPK:]
        return float(d.mean())

    def changed(self, cell_id, desc: np.ndarray) -> tuple[bool, float]:
        s = self.change(cell_id, desc)
        return s >= self.change_thr, s

    def observe_clean(self, cell_id, desc: np.ndarray) -> None:
        """Ô này đang sạch -> cho tham chiếu học. CHỈ gọi khi thật sự sạch."""
        ref = self._ref.get(cell_id)
        self._ref[cell_id] = desc.copy() if ref is None else \
            (1.0 - self.alpha) * ref + self.alpha * desc

    def set_clean(self, cell_id, desc: np.ndarray) -> None:
        """Nạp lại nền CỨNG cho ô này — không EMA.

        Dùng khi guard đổi sáng toàn cục bắn. `observe_clean` là EMA alpha=0,05
        nên nó chỉ nhích nền 5% mỗi lượt: sau một lần guard bắn, nền vẫn còn 95%
        cảnh CŨ nên lượt sau vẫn thoả điều kiện guard, và guard bắn lại. Mà mỗi
        lần bắn nó xoá luôn mốc bù méo, nên bù méo không bao giờ lấy được mốc.
        Kết quả đo được: sau một cú hích 30px, guard bắn LIÊN TỤC 8/8 lượt, ô
        đổi giữ nguyên 318-347/360, không gì báo được nữa — vùng mù hoàn toàn
        chứ không chỉ "bỏ lượt này".
        """
        self._ref[cell_id] = desc.copy()

    def has(self, cell_id) -> bool:
        return cell_id in self._ref

    def forget(self, cell_id) -> None:
        self._ref.pop(cell_id, None)

    def reset(self) -> None:
        self._ref.clear()

    def __len__(self) -> int:
        return len(self._ref)

    # ---- lưu/nạp (giữ tham chiếu qua restart, khỏi phải học lại từ đầu) ----

    def save(self, path: str, signature: str) -> None:
        if not self._ref:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        keys = np.array([[r, c] for (r, c) in self._ref], dtype=np.int32)
        vals = np.stack([self._ref[(int(r), int(c))] for r, c in keys])
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, keys=keys, vals=vals,
                            sig=np.array([signature], dtype=object))
        os.replace(tmp, path)

    def load(self, path: str, signature: str) -> bool:
        """Nạp tham chiếu. Vân tay lệch -> BỎ: ID ô cũ chỉ sang chỗ khác, nạp vào
        thì cổng đổi so nhầm ô và cả vùng câm luôn."""
        try:
            z = np.load(path, allow_pickle=True)
            keys, vals, sig = z["keys"], z["vals"], z["sig"]
        except (OSError, ValueError, KeyError):
            return False
        if str(sig[0]) != signature or vals.shape[1:] != (DESC, DESC):
            return False
        self._ref = {(int(r), int(c)): vals[i].astype(np.float32)
                     for i, (r, c) in enumerate(keys)}
        return True


THUMB = 256


def scene_shift(prev_thumb: np.ndarray | None, thumb: np.ndarray,
                frame_w: int) -> float:
    """Độ dịch chuyển khung hình giữa 2 lượt quét, quy về PIXEL ẢNH GỐC.

    Camera bị xoay/va chạm là điểm yếu thật của hướng này: toàn bộ ID ô lệch,
    tham chiếu và mặt nạ nhiễu chỉ sai chỗ. Dò bằng phase correlation rồi reset.

    Phải quy đổi về px ảnh gốc, không trả px ảnh thu nhỏ: cùng một ngưỡng cấu
    hình sẽ có nghĩa khác nhau giữa camera 640 và 1920 nếu không quy đổi.
    """
    if prev_thumb is None or prev_thumb.shape != thumb.shape:
        return 0.0
    (dx, dy), _ = cv2.phaseCorrelate(prev_thumb, thumb)
    d = (dx * dx + dy * dy) ** 0.5
    return float(d * frame_w / max(1, thumb.shape[1]))


def make_thumb(frame: np.ndarray, size: int = THUMB) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


# ------------------------------------------------------------------ bù rung
#
# scene_shift chỉ ĐO ĐỘ LỚN để biết có cần vứt state đi không, và mù hoàn toàn
# với phép xoay (phase correlation chỉ thấy tịnh tiến). Ba hàm dưới ước lượng
# đầy đủ dịch + xoay rồi nắn khung về mốc, dùng cho rung nhỏ mà cổng kia không
# bắt: đo trên CCTV thật, lệch 2px đã làm 86% số ô "đổi".

def gray_small(frame: np.ndarray, ds: int) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(g, None, fx=1.0 / ds, fy=1.0 / ds,
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def estimate_warp(ref_small: np.ndarray, cur_small: np.ndarray,
                  iters: int = 60) -> tuple | None:
    """(dx, dy, độ, W) của `cur` so với `ref`, theo px ẢNH THU NHỎ.

    None nghĩa là ECC không hội tụ — cảnh đổi quá nhiều để nói là cùng một
    khung hình. Gọi bên ngoài phải coi đó là "không bù", đừng nắn liều.
    """
    W = np.eye(2, 3, dtype=np.float32)
    try:
        cv2.findTransformECC(
            ref_small, cur_small, W, cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5), None, 5)
    except cv2.error:
        return None
    return (float(W[0, 2]), float(W[1, 2]),
            float(np.degrees(np.arctan2(W[1, 0], W[0, 0]))), W)


def apply_warp(frame: np.ndarray, W: np.ndarray, ds: int) -> np.ndarray:
    """Nắn `frame` về khung mốc. W đo ở ảnh thu nhỏ nên phần tịnh tiến phải
    nhân lại ds — phần xoay bất biến theo tỉ lệ, nhân cả ma trận là sai."""
    Wf = W.copy()
    Wf[0, 2] *= ds
    Wf[1, 2] *= ds
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, Wf, (w, h),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)
