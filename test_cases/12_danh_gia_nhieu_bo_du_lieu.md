# Có cần train thêm không — đo trên nhiều bộ rác

Chạy lại được bằng:

```bash
cd training
python3 tools/eval_datasets.py  --weights ../models/trash_yolo11n.pt --n 40 --conf 0.20
python3 tools/recall_by_size.py --weights ../models/trash_yolo11n.pt --n 40 --conf 0.20
```

Chấm theo **tâm hộp** (`centre_in`) chứ không IoU: pipeline chỉ cần biết "có vật ở
ô này không", không cần hộp khớp khít.

## Tách ba loại dữ liệu, vì chúng nói ba điều khác nhau

| | conf 0,10 | **conf 0,20** (điểm làm việc thật) |
|---|---|---|
| **ĐÃ TRAIN** — học thuộc, không kết luận gì | | |
| TACO | 66,7% | 56,5% |
| RoLID (val json) | 83,8% | 70,3% |
| RoLID (train json) | 82,3% | 67,1% |
| UAVVaste | 77,5% | 70,2% |
| Wade | 69,4% | 59,5% |
| **VAL chia theo CLIP** — tổng quát hoá TRONG MIỀN | | |
| TACO | 76,0% | **67,4%** |
| RoLID | 85,1% | **80,6%** |
| UAVVaste | 74,6% | 64,4% (chỉ 16 ảnh) |
| Wade | 74,7% | **65,3%** |
| **CHƯA HỀ THẤY** — đây mới là câu trả lời | | |
| GINI (907 ảnh bãi rác, chưa bao giờ đưa vào train) | 87,5% | 65,0% |
| Wade bãi rác (bị `--max-box-frac 0.40` loại khỏi train) | 87,5% | 67,5% |

**Val cao hơn train trên gần hết các bộ** (TACO 67,4 vs 56,5 · RoLID 80,6 vs 70,3 ·
Wade 65,3 vs 59,5). Đây là dấu hiệu ngược của overfit: model không hề học thuộc.
Bộ **chưa hề thấy** rơi đúng vào dải của val (65–67,5% vs 64–80%) — chuyển miền
không phải chỗ hỏng.

## Recall theo cỡ vật — chỗ hỏng thật nằm ở đây

conf 0,20, val chia theo clip, 373 vật:

| cạnh vật (px) | recall | số vật |
|---|---|---|
| 0–12 | **28,1%** | 32 |
| 12–20 | 69,2% | 65 |
| 20–32 | 66,7% | 72 |
| 32–48 | 75,0% | 64 |
| 48–80 | 63,4% | 41 |
| > 80 | 83,3% | 150 |

Vực dốc nằm **dưới 12px**, rồi phẳng 63–75% suốt 12–80px. Đây là **thiếu pixel,
không phải thiếu dữ liệu** — gán thêm nhãn cho vật 8px không dạy được model thứ
không có trong ảnh. Cách sửa là đặt camera, không phải train.

Quy ra khoảng cách (chai nhựa cao 25cm, 1080p, **ước lượng** theo ống kính):

| ống kính | ≥ 20px (recall ~69%) | ≥ 32px (recall ~75%) |
|---|---|---|
| 2,8mm (VFOV ~55°) | ≤ 13 m | ≤ 8 m |
| 4mm (VFOV ~40°) | ≤ 18 m | ≤ 11 m |

## "Hộp thừa" ở bảng trên KHÔNG phải số báo nhầm của pipeline

Hai lý do, cả hai đều kiểm bằng mắt trong `11_hop_thua_that_hay_gia.jpg`:

1. **Nhãn thiếu.** GINI gán *một* hộp phủ một phần đống rác. Hộp "thừa" nằm trên
   rác thật ở ngoài khung nhãn vẫn bị tính là thừa.
2. **Chạy sai cách.** Bảng trên chạy model trên **toàn ảnh**. Pipeline không bao
   giờ chạy như vậy — nó chỉ đưa vùng 320px mà cổng đổi đã chỉ ra. Trên chuỗi
   CCTV sạch, số lượt báo nhầm của pipeline ghép là **0%** (`day_cfg.yaml`).

Nhưng ảnh cũng cho thấy phần báo nhầm **thật**: trên TACO, hộp đỏ nằm trên sỏi,
đá, vải sofa, chân đèn — đúng kiểu "kết cấu bề mặt" đã ghi trong README, và đúng
thứ đã bắt nhầm chân ghế ở luồng thật.

## Kết luận

**Không cần train thêm trên dữ liệu công khai.** Ba chỗ đo đều chỉ cùng một
hướng: model không overfit (val ≥ train), không kẹt chuyển miền (chưa-hề-thấy
ngang val), và chỗ trượt dồn vào vật < 12px là vấn đề pixel.

Hai đòn bẩy còn lại, theo thứ tự đáng làm:

1. **Ảnh âm từ chính hiện trường.** Chân ghế, bóng nắng, sỏi — model bắn vì kết
   cấu bề mặt, và chỉ ảnh âm của đúng cảnh đó mới dạy được. Chưa thu được.
2. **Ràng buộc lắp camera**: chai phải ≥ 20px trong khung. Dưới ngưỡng đó recall
   sập còn 28% và không có cách train nào cứu.
