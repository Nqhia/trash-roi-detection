"""Backend chấm ô: nhận list ảnh ô -> trả list điểm 0..1 ("khả năng có rác").

Ba backend, cùng một giao diện, đổi bằng config:

- ConstantScorer : không mô hình, để thông ống trước khi có model.
- VlmScorer      : không cần dữ liệu, dựng được ngay trong tuần đầu. Đắt
                   (~0.5-1s/call, +3-6GB VRAM), trần ~40 camera. Dùng cho bản
                   demo nhỏ, và để SINH NHÃN offline cho CNN.
- OnnxPatchScorer: bản chính thức. MobileNetV3-Small @96px, ~0.011 GFLOPs/ô,
                   264 ô ~ 2.9 GFLOPs ~ 1/3 một forward YOLOv8n. Ở nhịp 30s là
                   ~0.3% tải của detector object đang chạy. Chạy CPU thoải mái.

Lộ trình mong muốn: VLM chỉ nằm ở OFFLINE (sinh nhãn), runtime chỉ có ONNX.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

import cv2
import numpy as np


class ConstantScorer:
    """Trả cùng một điểm cho mọi ô. Chỉ để kiểm tra đường ống."""

    name = "constant"

    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def score(self, patches: list) -> list[float]:
        return [self.value] * len(patches)


class OnnxPatchScorer:
    """Classifier nhị phân clean/litter chạy theo lô."""

    name = "onnx"

    def __init__(
        self,
        model_path: str,
        input_size: int = 96,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        output: str = "logits2",   # "logits2" | "sigmoid1"
        providers: list | None = None,
    ) -> None:
        import onnxruntime as ort  # import muộn: chỉ backend này mới cần

        self.size = int(input_size)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.output = output
        avail = ort.get_available_providers()
        prov = [p for p in (providers or ["CUDAExecutionProvider",
                                          "CPUExecutionProvider"]) if p in avail]
        self.sess = ort.InferenceSession(model_path, providers=prov or None)
        self.inp = self.sess.get_inputs()[0].name

    def _prep(self, patches: list) -> np.ndarray:
        out = np.empty((len(patches), 3, self.size, self.size), dtype=np.float32)
        for i, p in enumerate(patches):
            # Phóng TO ô lên input size — không bao giờ thu nhỏ, vì vật chỉ ~27px
            # và mọi lần thu nhỏ đều xoá mất nó.
            r = cv2.resize(p, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            out[i] = r.transpose(2, 0, 1)
        return (out - self.mean) / self.std

    def score(self, patches: list) -> list[float]:
        if not patches:
            return []
        y = self.sess.run(None, {self.inp: self._prep(patches)})[0]
        y = np.asarray(y, dtype=np.float32)
        if self.output == "sigmoid1":
            p = 1.0 / (1.0 + np.exp(-y.reshape(-1)))
        else:
            e = np.exp(y - y.max(axis=1, keepdims=True))
            p = (e / e.sum(axis=1, keepdims=True))[:, 1]
        return [float(v) for v in p]


DEFAULT_VLM_PROMPT = (
    "You are inspecting a small crop from a fixed outdoor CCTV camera looking at "
    "ground/pavement. Decide whether this crop contains LITTER that someone "
    "discarded: plastic bottle, plastic bag, food container, paper, cardboard, "
    "can, garbage bag or a pile of refuse.\n"
    "Do NOT count as litter: bare ground, road markings, drain covers, manholes, "
    "puddles, oil stains, cracks, gravel, fallen leaves, grass, vehicles, people, "
    "animals, or goods properly stacked by a shop.\n"
    "If the crop is too blurry or dark to tell, answer false.\n"
    'Reply with STRICT JSON only: {"litter": true|false, "confidence": 0.0-1.0, '
    '"what": "<= 6 words"}'
)


class VlmScorer:
    """Gọi endpoint tương thích OpenAI /chat/completions với ảnh base64.

    Chạy tuần tự, một ô một call. Đây là lý do pipeline PHẢI có cổng đổi chặn
    trước và trần `max_score_per_scan` — không có hai thứ đó thì backend này
    sập ngay ở camera thứ nhất.
    """

    name = "vlm"

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3-VL-2B-Instruct",
        api_key: str = "",
        prompt: str = DEFAULT_VLM_PROMPT,
        zoom: int = 4,
        timeout: float = 20.0,
        jpeg_quality: int = 92,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.prompt = prompt
        self.zoom = max(1, int(zoom))
        self.timeout = float(timeout)
        self.jpeg_quality = int(jpeg_quality)
        self.last_notes: list[str] = []

    def _one(self, patch: np.ndarray) -> tuple[float, str]:
        # Phóng to ô trước khi gửi: VLM tự resize ảnh đầu vào, ô 64px gửi thẳng
        # thì vỏ chai 27px teo mất. Phóng 4x là đủ và không tốn thêm token.
        if self.zoom > 1:
            patch = cv2.resize(patch, None, fx=self.zoom, fy=self.zoom,
                               interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", patch,
                               [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return 0.0, "encode-failed"
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        body = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self.prompt},
                {"type": "image_url", "image_url": {"url": uri}},
            ]}],
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                txt = json.load(r)["choices"][0]["message"]["content"]
        except (urllib.error.URLError, OSError, KeyError, ValueError, TimeoutError) as exc:
            # VLM chết không được giết cả pipeline — coi như ô sạch và ghi log.
            return 0.0, f"vlm-error: {exc}"
        return _parse_vlm(txt)

    def score(self, patches: list) -> list[float]:
        out, notes = [], []
        for p in patches:
            s, note = self._one(p)
            out.append(s)
            notes.append(note)
        self.last_notes = notes
        return out


def _parse_vlm(txt: str) -> tuple[float, str]:
    """Bóc JSON khỏi câu trả lời (model hay bọc trong ```json ... ```)."""
    s = txt.strip()
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(s[i:j + 1])
            hit = bool(d.get("litter"))
            conf = float(d.get("confidence", 1.0 if hit else 0.0))
            conf = min(1.0, max(0.0, conf))
            return (conf if hit else 1.0 - conf), str(d.get("what", ""))[:40]
        except (ValueError, TypeError):
            pass
    # Không ra JSON: cố vớt vát câu trả lời tự do, nhưng mặc định là KHÔNG có rác.
    # Sai sót ở đây phải nghiêng về bỏ sót, không nghiêng về báo nhầm — model
    # lảm nhảm mà bị hiểu thành "có rác" là nguồn FP tệ nhất vì nó không có quy
    # luật gì để mà lọc. `note` giữ nguyên văn để người vận hành thấy prompt hỏng.
    low = s.lower().lstrip("*_# \n\t\"'")
    if low.startswith(("yes", "true", "có rác", "co rac")) or '"litter": true' in low:
        return 1.0, "unparsed-yes"
    return 0.0, f"unparsed: {s[:30]}"


def build_scorer(cfg: dict):
    """Dựng backend từ khối `scorer` trong config.yaml."""
    kind = (cfg.get("kind") or "constant").lower()
    opts = {k: v for k, v in cfg.items() if k != "kind"}
    if kind == "onnx":
        return OnnxPatchScorer(**opts)
    if kind == "vlm":
        return VlmScorer(**opts)
    if kind == "constant":
        return ConstantScorer(**opts)
    raise ValueError(f"scorer.kind không hợp lệ: {kind!r}")
