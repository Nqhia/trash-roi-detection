# Kết quả bộ test

Sinh bởi `python3 tools/run_test_cases.py` — chạy lại là ra đúng số này.

Model: `models/trash_yolo11n.pt` · conf 0.2 · ô 320px phóng 2.0×
Mặt nạ nhiễu TẮT trong mọi ca (cần ~5 giờ mới chín, không đo được ở đây).

## Khung eco — rác thật do người vứt

6 khung · 21 vật thật · ảnh: `01_model_only_eco.jpg`, `02_patch_only_eco.jpg`, `03_patch_plus_model_eco.jpg`

| cấu hình | bắt được | hộp thừa |
|---|---|---|
| model đứng riêng | 18/21 · 86% | 59 |
| patch đứng riêng | 18/21 · 86% | 471 |
| **patch + model** | 18/21 · 86% | 3 |

## Chuỗi CCTV sạch — không có rác

36 lượt quét trên 3 video (có người đi, ánh sáng đổi) · ảnh: `04_clean_rejected.jpg`

| cấu hình | lượt có báo nhầm |
|---|---|
| model đứng riêng | 15/36 · 42% |
| patch đứng riêng | 20/36 · 56% |
| **patch + model** | **0/36 · 0%** |
