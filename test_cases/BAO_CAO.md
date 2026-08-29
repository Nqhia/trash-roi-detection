# Báo cáo soi mắt — toàn bộ phép thử, ảnh kèm số

Mỗi mục: **mục đích → dữ liệu → lệnh chạy lại → số → ảnh để soi**. Mọi con số
trong README đều truy về được một mục ở đây. Quy ước màu chung cho ảnh:

    ĐỎ    ô nóng (cổng đổi + dwell)        XANH LÁ  hộp detector xác nhận
    VÀNG  vật thật / mốc sự kiện           TÍM      vùng cắt đưa vào model

---

## 1 · Giải phẫu MỘT lượt quét — pipeline nhìn cái gì

Ảnh: `24_giai_phau_mot_luot.jpg`. Khung thật của camera văn phòng: 360 ô lưới
xám → 37 ô nóng đỏ → gom thành 2 cụm → 2 vùng tím (218px, 240px) → hàng dưới là
ĐÚNG cái ảnh model nhìn thấy (phóng ×2 = 640px) kèm hộp nó trả về (0,53 · 0,31).

Trả lời câu "cho cả vùng hay từng ô": **theo cụm** — không phải từng ô 48px
(quá nhỏ, model mù), không phải cả vùng (vật thành quá nhỏ, đo được 0 hộp).

## 2 · Khung eco — rác thật, camera thật (1 cảnh)

Ảnh: `01_model_only_eco.jpg` · `02_patch_only_eco.jpg` · `03_patch_plus_model_eco.jpg`
Chạy lại: `python3 tools/run_test_cases.py`

| cấu hình | bắt được | hộp thừa |
|---|---|---|
| model đứng riêng | 18/21 · 86% | 59 |
| patch đứng riêng | 18/21 · 86% | 471 |
| **ghép hai tầng** | 18/21 · 86% | **3** |

**Cách đọc cho đúng:** 21 "vật" là 3–4 vật đếm lặp qua 6 khung liên tiếp của
MỘT cảnh — số mẫu độc lập chỉ là 3–4. Bộ này từng được dùng để chỉnh tham số,
nên 86% là trần lạc quan, không phải năng lực tổng quát.

## 3 · Chuỗi CCTV sạch — không có rác (3 camera khác nhau)

Ảnh: `04_clean_rejected.jpg` — các lượt mà cổng đổi kêu nhưng detector bác.
Chạy lại: trong `run_test_cases.py`.

| cấu hình | lượt có báo nhầm |
|---|---|
| model đứng riêng | 15/36 · 42% |
| patch đứng riêng | 20/36 · 56% |
| **ghép** | **0/36 · 0%** |

Đây là bằng chứng chuyển-camera tốt nhất cho precision: 3 cảnh khác nhau,
người đi lại + đổi sáng, không phát nhầm lượt nào.

## 4 · Sự kiện ABODA — vật bị bỏ lại trên 6 cảnh CCTV thật

Ảnh: `25_su_kien_aboda/tong_hop.jpg` (trái = lúc vật xuất hiện, phải = lúc báo)
và `25_su_kien_aboda/videoN.jpg` từng ca. Chạy lại: `python3 tools/event_report.py`

| video | cảnh | kết quả (thước ĐÚNG CHỖ) |
|---|---|---|
| 1 | sảnh trong nhà | có báo nhưng **SAI CHỖ** — ô nóng trên người/bàn, không trùm ba lô |
| 3 | vỉa hè ngoài trời | **đúng chỗ**, trễ 10 lượt |
| 4 | vỉa hè ngoài trời | có báo nhưng **SAI CHỖ** |
| 5 | đêm IR | **đúng chỗ**, trễ 14 lượt |
| 9 | phòng họp | **đúng chỗ**, trễ 17 lượt |
| 10 | phòng họp | có báo nhưng **SAI CHỖ** |

**→ báo đúng chỗ: 3/6 · có báo (bất kể chỗ): 6/6 · im hẳn: 0/6**

**Phát hiện quan trọng của chính lần soi mắt này (29/08):** thước cũ trong
`event_latency.py` đếm "có cảnh báo sau khi vật xuất hiện" mà KHÔNG kiểm vị
trí, nên các cảnh báo bắn vào người nán lại / mép bàn ghế cũng được tính là
"bắt được sự kiện". Con số 5–6/6 báo cáo trước ngày này là **bị thổi phồng**;
số trung thực theo thước đúng-chỗ là 3/6. Ba ca sai chỗ đều là túi/ba lô sẫm
màu nhỏ — đúng điểm yếu túi của model đã biết.

Lưu ý quy đổi: video ngắn nên quét mỗi 15 khung; "lượt" × 30s = phút hiện
trường, là xấp xỉ có hướng lệch không rõ (xem docstring `event_latency.py`).

## 5 · Túi ni lông thật — ca thử duy nhất trên camera đang chạy

Ảnh: `26_tui_that/` — `01_truoc_sau.jpg` (sạch → có túi, khung thật từ lần chạy
43h), `02_khung_co_tui.jpg`, `03_nut_buoc_phong_to.jpg` (hộp 0,49 trên nút
buộc khi cắt sát), `04_quet_day_nhieu_co.jpg` (vùng 320px cho 0 hộp — lý do
trượt ban đầu). Chạy lại: `python3 tools/bag_test.py`

    trước fix : cổng đổi thấy ngay (60→76 ô), detector bác mọi lượt, 4,5 GIỜ không báo
    sau fix   : BÁO sau 3 lượt = 1,5 phút (quét sau reset soi ở cỡ nhỏ)

## 6 · Năm ca nuốt rác — thuộc tính chống mất cảnh báo im lặng

Chạy lại: `python3 tools/absorb_test.py` — phải ra "Không ca nào nuốt rác".
Mỗi ca đã được chạy trên code CŨ trước để chứng minh test thật sự bắt lỗi:

| ca | trước | sau |
|---|---|---|
| A camera bị hích 30px lúc có rác | nuốt | 34 ô nóng |
| A hích 60px (quá trần bù) | nuốt | 56 ô |
| B vận hành chốt lại nền lúc có rác | nuốt | 55 ô |
| C bóng mây trùm 70% khung | nuốt | 36 ô |
| D vận hành sửa vùng trên UI | nuốt | 52 ô |

## 7 · Soi model — bắt gì, trượt gì

- `22_soi/` — hai model trên khung eco, cùng ô nóng: model cũ 21/21, model mới
  15/21 (trượt đúng MỘT vật trắng nhỏ, lặp 6 khung).
- `23_soi_rong/so_sanh_cung_diem.jpg` — 18 cảnh, 6 nguồn, cùng ngưỡng 0,10:
  cũ 20/30 (nhầm 8) vs mới 21/30 (nhầm 12); ở ngưỡng riêng: cũ 15/30 vs mới 23/30.
- `20_review/` — nhãn của từng bộ dữ liệu train (đã soi trước khi train).
- `11/12/13` — kiểm nhãn và mốc sự kiện tự dò (lý do phải soi mắt: 3/11 video
  ABODA mốc tự dò sai, 399/400 ảnh TACO lệch tỉ lệ metadata).

## 8 · Đang đo — FP/ngày bản code hiện tại

`runs/shadow_A/` — vùng sàn giữa góc mới (173 ô), code sau kiểm toán 28/08,
bật 29/08. Đọc: `python3 tools/shadow_report.py --scans runs/shadow_A/scans*.csv`
Số FP/ngày duy nhất đang có (7,9/ngày, vùng 41 ô, 30h) đo bằng code TRƯỚC kiểm
toán — chỉ dùng tham khảo, không dùng nghiệm thu.
