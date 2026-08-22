# Kết quả bộ test

Sinh bởi `python3 tools/run_test_cases.py` — chạy lại là ra đúng số này.

Model: `models/trash_yolo11n.pt` · conf 0.1 · ô 320px phóng 2.0×
Mặt nạ nhiễu TẮT trong mọi ca (cần ~5 giờ mới chín, không đo được ở đây).

## Khung eco — rác thật do người vứt

6 khung · 20 vật thật · ảnh: `01_model_only_eco.jpg`, `02_patch_only_eco.jpg`, `03_patch_plus_model_eco.jpg`

| cấu hình | bắt được | hộp thừa |
|---|---|---|
| model đứng riêng | 18/20 · 90% | 82 |
| patch đứng riêng | 18/20 · 90% | 472 |
| **patch + model** | 17/20 · 85% | 0 |

## Chuỗi CCTV sạch — không có rác

36 lượt quét trên 3 video (có người đi, ánh sáng đổi) · ảnh: `04_clean_rejected.jpg`

| cấu hình | lượt có báo nhầm |
|---|---|
| model đứng riêng | 35/36 · 97% |
| patch đứng riêng | 17/36 · 47% |
| **patch + model** | **9/36 · 25%** |
