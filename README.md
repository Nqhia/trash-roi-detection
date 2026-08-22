# Phát hiện rác trong vùng ROI

Vẽ một ROI trên camera cố định. Có vật lạ nằm lại trong vùng thì báo.
Không cần biết ai vứt, không cần biết nằm bao lâu, không cần phân loại rác.
Vật nhỏ nhất phải bắt được: **vỏ chai**.

---

## 1 · Pipeline có những gì

Bốn tầng nối tiếp. Mỗi tầng chặn một loại lỗi mà tầng khác không chặn được.

```
        ┌─ tầng 1 ─────────┐  ┌─ tầng 2 ──┐  ┌─ tầng 3 ────┐  ┌─ tầng 4 ────────┐
khung → │ cổng đổi nền     │→ │ gộp ô     │→ │ detector    │→ │ dwell/confirm/  │→ cảnh báo
  ảnh   │ lọc ~97% số ô    │  │ thành vùng│  │ phán là rác │  │ chốt/mặt nạ     │
        └──────────────────┘  └───────────┘  └─────────────┘  └─────────────────┘
         biết cảnh NÀY        ~1 vùng/lượt    biết CÁI GÌ      chặn thoáng qua
         bình thường ra sao   thay vì 9-12 ô  là rác           và báo lặp
```

**Vì sao phải có cả tầng 1 lẫn tầng 3.** Cổng đổi biết *chỗ nào* khác thường
nhưng không biết *cái gì*; detector biết cái gì nhưng không biết cảnh này bình
thường ra sao. Hai bên sai ở chỗ khác nhau — cổng đổi bắn vì người đi qua và
ánh sáng đổi, detector bắn vì kết cấu bề mặt — nên giao của chúng nhỏ hơn hẳn
từng bên. Đo được: báo nhầm trên chuỗi CCTV sạch **42% (model) và 56% (patch)
tụt về 0% khi ghép**, recall giữ nguyên 86%.

### Tầng 1 · cổng đổi nền — `core/reference.py`

Chia ROI thành lưới ô chồng lấn 50%. Mỗi ô có mô tả 8×8 xám **trừ mean** (triệt
offset độ sáng, không triệt tương phản), so với tham chiếu EMA (α = 0,05).

Điểm đổi = **trung bình 8 hiệu tuyệt đối lớn nhất**, không phải trung bình cả 64
số. Lấy trung bình toàn cục thì vỏ chai bị pha loãng còn `2 × 0,053 × 0,947 ×
40 ≈ 4,0` — lẫn vào nhiễu nén. Top-8 thì đúng mấy ô mô tả trùng vỏ chai được
tính: **15–20**, tách bạch hẳn với nhiễu (~0,5).

Tham chiếu **chỉ học ở ô sạch**. Học vô điều kiện thì rác nằm lâu bị nuốt vào nền.

### Tầng 2 · gộp ô thành vùng — `core/verify.py`

Ô 48px quá nhỏ để detector nhận ra gì. Gộp ô kề nhau, nới biên, chặn trần cạnh
vùng 256px → **~1 vùng mỗi lượt** thay vì 9–12 ô nếu quét toàn ROI.

### Tầng 3 · detector xác nhận — `models/trash_yolo11n.pt`

Chi tiết ở §3. Chạy trên vùng đã phóng **2×** — phải trùng đúng `upscale` lúc
dựng bộ train (ô 320px → ảnh 640px).

Detector lỗi (mất file, hết VRAM) thì pipeline **giữ nguyên ô nóng** và cảnh báo
như cũ, kèm cờ `verify_failed`. Mất precision còn hơn nuốt cảnh báo mà không ai biết.

### Tầng 4 · các cổng chặn báo lặp — `core/gates.py`, `core/clutter.py`

- **`dwell_scans`** — ô phải đổi liên tục N lượt mới "nóng". Đo trên ABODA: 75%
  ô nhiễu do người chỉ đổi **đúng 1 lượt**, ô có vật giữ 10–15 lượt.
- **ConfirmGate N/M** — cửa sổ trượt, diệt FP *ngẫu nhiên*. Phải xoá sau khi
  bắn, không thì lượt sau cửa sổ vẫn còn các `True` cũ và bắn thêm lần nữa.
  **Đang đặt `1/1`, tức tắt.** Nó sinh ra cho mode `classifier` nơi model chấm
  lại từng ô mỗi lượt; ở `change_only` thì `dwell` đã ép tính bền vững rồi, để
  cả hai là lọc trùng và trả giá bằng độ trễ (4,0 → 1,5 phút khi bỏ). Bật lại
  nếu chuyển sang `classifier`.
- **Chốt theo TỪNG Ô** — chốt cả vùng thì một vật đã báo (xe đỗ, thùng rác) nuốt
  mất mọi sự kiện sau. `merge_radius_cells` gộp ô sát ô đã chốt vì các ô của
  cùng một vật đạt ngưỡng dwell lệch nhau vài lượt.
- **Mặt nạ nhiễu** — mute ô bẩn ~100% số lần *được hỏi*. Cần ~5 giờ mới chín.

### Cơ chế an toàn

**Guard đổi sáng toàn cục** — phải đủ **cả ba**: ≥45% ô cùng đổi **và** trải
≥80% số hàng lẫn cột **và** còn thấy ≥60% lưới. Mỗi điều kiện từ một lần hỏng thật:
thiếu `frac` → rạng sáng làm nền đóng băng, nóng 222/319 ô suốt 2,5 giờ;
thiếu `spread` → bao rác chiếm 117/227 ô bị nuốt luôn;
thiếu `min_visible` → người che 203/227 ô, 24 ô còn lại cùng đổi → **xoá sạch nền**.

**Bù rung ECC + vùng chết 3px.** Nắn khung về mốc trước khi cắt ô. Lệch 2px làm
86% số ô "đổi", nên cần bù. **Nhưng ECC ước lượng từ toàn khung kể cả vật mới
xuất hiện**, nên vật mới làm lệch chính phép ước lượng: đo trên khung eco (không
camera nào dịch) ECC báo 1,35px, nắn theo đó làm ô đổi tăng **82 → 158/360**.
Nắn 1px chỉ sửa 1px nhưng nội suy làm nhoè *toàn khung*, mà nhoè chính là thứ
cổng đổi thấy. Quét vùng chết:

| vùng chết | khung eco (không lệch thật) | lệch thật 4px |
|---|---|---|
| 0,15px | 158/360 ✗ | 0/360 ✓ |
| 2,0px | 158/360 ✗ | 0/360 ✓ |
| **3,0px** | **82/360 ✓** | **0/360 ✓** |
| 5,0px | 82/360 ✓ | 245/360 ✗ |

3,0px là cửa sổ duy nhất đúng cả hai chiều.

**Ảnh mốc chống lệch** — `scene_shift` chỉ so hai lượt **liền nhau**, nên camera
bị xoay lúc chương trình không chạy thì không ai biết. `live_view.py` lưu
`<zone>.anchor.png` và so lúc khởi động. Đã dính thật: camera ngóc lên 289px,
cả vùng sàn tụt khỏi khung mà pipeline vẫn chạy ngon lành.

---

## 2 · Hoạt động ra sao

```bash
pip install ultralytics opencv-python pyyaml numpy
```

**Mọi lệnh chạy từ thư mục gốc** — config trỏ tới `models/` bằng đường dẫn tương đối.

### Vẽ vùng

```bash
python3 tools/draw_zone.py --source 'rtsp://...' --out config/zone.json
```

Chuột trái thêm đỉnh, phải xoá, `ENTER` lưu. Hiện luôn số ô và cỡ vật nhỏ nhất
bắt được, lưu kèm ảnh mốc chống lệch.

**Vẽ ở đâu.** Yêu cầu thật không phải "vùng trống" mà là **"vùng ổn định"** —
đo được: vùng bao cả bàn và 3 ghế chạy 1,5 giờ đầu ra **0 cảnh báo** khi không
ai động vào. Tránh: **cây cảnh, rèm, biển đung đưa** (ô đó nóng mãi, chốt không
mở, vùng đó mù với mọi sự kiện sau) và **đồ đạc hay bị xê dịch** (mỗi lần dời là
một cảnh báo).

### Xem trực tiếp

```bash
python3 tools/live_view.py --source 'rtsp://...' --zone config/zone.json \
  --config config/day_cfg.yaml --out runs/live --interval 8
```

Ba luồng tách hẳn nhau nên hình không giật: luồng đọc giữ khung mới nhất, luồng
quét chạy pipeline theo nhịp, luồng chính chỉ vẽ overlay của lượt gần nhất.

| màu | là gì | dẫn tới báo? |
|---|---|---|
| 🟡 vàng | viền ROI | — |
| 🔵 xanh dương | người/xe từ YOLO (nếu bật `--yolo`) | **không — nó LOẠI ô đó đi** |
| 🟠 cam | ô vừa đổi, chưa đủ `dwell` | chưa |
| 🔴 đỏ | ô nóng | có thể |
| 🟣 tím | **detector xác nhận** | **đây mới là thứ quyết định** |

Đỏ mà không tím = cổng đổi kêu, detector bác bỏ, hệ thống im.
Phím: `q` thoát · `r` **chốt lại nền** · `g` ẩn lưới · `b` ẩn box · `s` lưu ảnh.

### Chạy shadow để lấy số

```bash
python3 tools/run_video.py --source 'rtsp://...' --zone config/zone.json \
  --config config/day_cfg.yaml --out runs/shadow
python3 tools/eval.py --scans runs/shadow/scans.csv
python3 tools/shadow_report.py --scans runs/shadow/scans.csv
```

**Chạy dài (≥24h) thì dùng `keepalive.sh`**, đừng gọi `run_video.py` trực tiếp:

```bash
bash tools/keepalive.sh 'rtsp://...' config/zone.json config/day_cfg.yaml runs/shadow
```

Hai lần chạy dài trước đều chết giữa chừng — một lần máy crash, một lần máy ngủ —
và **không lần nào có dòng "mất kết nối hẳn" trong log**, tức tiến trình bị giết
từ ngoài chứ không phải lỗi RTSP. `run_video.py` có backoff cho RTSP nhưng không
tự sống lại được. State (nền + mặt nạ nhiễu) nằm trong `<out>/state` và được nạp
lại khi khởi động, nên bật lại **không mất 5 giờ học mặt nạ**.

`scans.csv` được `flush()` sau **mỗi lượt**. Bản đầu chỉ `close()` ở cuối nên
dòng nằm trong buffer, và một lần chết đột ngột đã **mất 6 phút cuối** của lần
chạy 5,9 giờ.

`eval.py` **từ chối ngoại suy** FP/ngày từ clip dưới 2 giờ.
`shadow_report.py` trả lời câu khác: cảnh báo có **cụm lại theo thời gian** không
(→ hoạt động của người) và có **cụm theo vị trí** không (→ một vật cố định).

### Vận hành

**Dời đồ đạc hợp lệ** → bấm `r`, hoặc `touch runs/shadow/state/reset.flag`.
**Camera bị xoay** → `live_view` cảnh báo lúc khởi động; vẽ lại ROI.
**Đo ngưỡng chịu lệch tại chỗ** → `tools/shift_test.py`.
**Chọn `cell_px`** → quy tắc **cạnh ô ≈ 2× cạnh vật**: 48px bắt vật 20–24px,
64px bắt 28px, 96px bắt 48px. Đo cỡ vật bằng `tools/measure_px.py`.

---

## 3 · Model — thông số và cách train

### Thông số

| | |
|---|---|
| kiến trúc | yolo11n · **2,59M tham số** · 5,5 MB |
| lớp | **1 lớp duy nhất `trash`** (bài toán không cần phân loại) |
| đầu vào | 640×640 (ô 320px × phóng 2) |
| **precision** | **0,821** |
| **recall** | **0,646** |
| **mAP50** | **0,747** |
| mAP50-95 | 0,376 |
| tốc độ | **11,5 ms** GPU · **61,7 ms** CPU |

Val chia theo **clip**, không rò rỉ.

### Dữ liệu train

26069 ảnh: **19047 ô có rác + 7022 ô rỗng** làm mẫu âm.

| nguồn | ảnh | loại |
|---|---|---|
| RoLID train | 11591 | đường phố, dashcam |
| RoLID val | 7193 | đường phố, dashcam |
| UAVVaste | 2548 | drone nhìn từ trên |
| Wade-AI train | 2286 | ảnh điện thoại + Google Street View |
| TACO | 2064 | ảnh chụp cận |
| Wade-AI val | 387 | |

Cỡ vật trong ô 640px: p10 = 32px · **trung vị 57px** · p90 = 168px.

### Ba quyết định trong bộ dữ liệu, mỗi cái đều đo được

**1 · Cắt ô rồi phóng to** để khớp đúng thứ detector gặp lúc chạy. Train trên
ảnh nguyên thu về 960px thì cùng một vỏ chai chiếm ~15px lúc train và ~50px lúc
chạy — đo được **0/12 vật dưới 25px** trên khung eco dù val mAP50 tới 0,595.

**2 · Bỏ nhãn mức "cả đống"** (`--max-box-frac 0.40`). Wade có 14,9% và TACO
8,9% số nhãn là hộp phủ 40–77% khung — nhãn cho nguyên một bãi rác. Dạy model
bắn ra hộp khổng lồ, đúng khớp với FP quan sát được: hộp to phủ cả tấm vách gỗ.

**3 · Ô rỗng cắt từ chính ảnh có rác** làm mẫu âm — cùng cảnh, cùng ánh sáng,
cùng nền. Ba bộ gốc chỉ có ảnh chứa rác, không một khung sạch nào, nên model
chưa bao giờ học "cái gì KHÔNG phải rác": đo được 9,9 hộp thừa mỗi khung sạch.
Lọc bỏ ô rỗng là trời/mây/kính mờ — model có bao giờ bắn lên trời đâu.

### Train lại

```bash
cd training
python3 tools/make_tiles_data.py --out data/tiles --tile 320 --upscale 2.0 \
    --neg-per-img 2 --max-per-src 1800 --max-box-frac 0.40
python3 tools/check_labels.py --n 6      # XEM BANG MAT truoc khi train
python3 tools/audit_data.py --data data/tiles
python3 tools/train_detector.py --data data/tiles/trash.yaml --model yolo11n.pt \
    --epochs 12 --imgsz 640 --batch 16
```

`audit_data.py` là bắt buộc — nó bắt được rò rỉ train/val, nhãn mồ côi, hộp suy
biến, và ô rỗng có phải thật rỗng không.

---

## 4 · Kết quả

### Bộ test — `test_cases/`

Sinh lại bằng `python3 tools/run_test_cases.py`. Hai tập kiểm đều **nằm ngoài
mọi tập train** của detector.

**Khung eco** — camera EcoVision, rác thật do người vứt, 6 khung / 21 vật.
Nhãn lấy bằng trừ nền rồi gộp ở mức vật.

| cấu hình | bắt được | hộp thừa |
|---|---|---|
| model đứng riêng | 18/21 · 86% | 59 |
| patch đứng riêng | 18/21 · 86% | 471 |
| **patch + model** | **18/21 · 86%** | **3** |

**Chuỗi CCTV sạch** — 3 video ABODA có người đi và ánh sáng đổi, không có rác.
Phải chạy theo *chuỗi*: dựng nền từ chính khung sắp quét rồi quét lại khung đó
thì "0 báo nhầm" chỉ nói rằng không đổi thì không báo.

| cấu hình | lượt có báo nhầm |
|---|---|
| model đứng riêng | 15/36 · 42% |
| patch đứng riêng | 20/36 · 56% |
| **patch + model** | **0/36 · 0%** |

Ghép **không mất recall nào** (cả ba đều 86%) mà đưa hộp thừa từ 59 xuống **3** và cắt
báo nhầm từ 42%/56% xuống **0%**. Ảnh trong `test_cases/`: `01_model_only_eco.jpg`,
`02_patch_only_eco.jpg`, `03_patch_plus_model_eco.jpg`, `04_clean_rejected.jpg`.

`test_cases/model_only/` là detector chạy **một mình** trên bốn miền khác nhau,
để thấy nó mạnh yếu ở đâu: conf tụt theo mức quen thuộc — drone 0,88 → ven đường
0,81 → sàn gạch trong nhà 0,58 — và trên khung CCTV *không có rác* nó vẫn bắn
vào ghế, nắp cống, thùng điện, tấm biển.

### Chạy thật trên camera EcoVision

5,88 giờ · 697 lượt · nhịp 30s · vùng 225 ô bao cả bàn ghế:

```
0.0-1.0h  17.6 o doi TB ·  3 luot co o nong · 0 canh bao
1.0-2.0h  76.0 o doi TB · 28 luot co o nong · 3 canh bao
2.0-3.0h 101.1 o doi TB · 59 luot co o nong · 3 canh bao
3.0-4.0h  59.4 o doi TB · 57 luot co o nong · 3 canh bao
4.0-5.0h  69.8 o doi TB · 52 luot co o nong · 2 canh bao
5.0-6.0h  81.9 o doi TB · 39 luot co o nong · 1 canh bao
```

- **1,5 giờ đầu không ai động vào vùng → 0 cảnh báo.** Nền tĩnh không sinh FP.
- 12 cảnh báo, **10/12 dính đúng ba ô `4,21–4,23`** — một chỗ duy nhất, là đồ đạc
  bị xê dịch trong giờ có người dùng bàn ghế.
- Detector **bác bỏ 98%** ô nóng (19463 ô bị loại, 419 hộp được xác nhận).
- Chuỗi cổng cắt **238 lượt có ô nóng → 12 cảnh báo**.
- Guard đổi sáng bắn đúng lúc cần (`131/225 ô, trải 88%`), nạp lại nền, **không**
  bắn cảnh báo.
- 206 ms/lượt (11ms cổng đổi + ~195ms detector) = 0,7% thời gian máy ở nhịp 30s.

### Chỉ số nghiệm thu

| chỉ số | ngưỡng | pipeline này |
|---|---|---|
| cảnh báo nhầm / camera / ngày | < 1–2 | **chưa đo đủ 24h** |
| recall theo sự kiện | càng cao càng tốt | 86% (21 vật, 1 camera trong nhà) |
| độ trễ phát hiện | < 3 phút | **1,5 phút ✓** |

**FP/ngày.** Lần chạy 5,9 giờ cho 49 FP/ngày nếu tính trên toàn clip, nhưng mặt
nạ nhiễu mới chín ở giờ thứ 5,83 nên 99% thời gian là giai đoạn học. Sau khi
chín chỉ có 0,05h dữ liệu — quá ngắn để kết luận gì. **Cần ≥24h liên tục.**

**Độ trễ: 4,0 → 1,5 phút.** Bản đầu dùng `dwell 5` + `confirm 4/6` = phải 8 lượt
mới báo. Đó là **lỗi thiết kế, không phải đánh đổi cần thiết**: hai cổng lọc
trùng nhau. ConfirmGate sinh ra cho mode `classifier`, nơi model chấm lại từng ô
mỗi lượt nên điểm nhảy lung tung và cần lọc nhiễu *ngẫu nhiên*. Ở `change_only`
thì `dwell` đã ép tính bền vững bằng bộ đếm `_run` — ô phải đổi **liên tiếp** N
lượt. Để nguyên 4/6 chồng lên là bắt bền vững hai lần, trả giá hai lần bằng độ
trễ. Quét bằng `tools/latency_sweep.py`:

| dwell | confirm | lượt | @30s | báo nhầm chuỗi sạch |
|---|---|---|---|---|
| 5 | 4/6 | 8 | 4,0 ❗ | 1/42 |
| **3** | **1/1** | **3** | **1,5** | **1/42** |
| 1 | 1/1 | 1 | 0,5 | 2/42 |

Chờ gấp 8 lần chỉ đổi được 2/42 xuống 1/42 — mà chênh 1 với 2 trên 42 mẫu nằm
trong nhiễu. Đã chốt **`dwell 3` + `confirm 1/1`**.

Vẫn giữ `dwell 3` chứ không hạ xuống 1, vì `dwell` là lớp bảo hiểm cho những gì
**chưa đo**: bóng nắng quét ngang, lá bay, mưa. Bỏ hẳn ngay trước khi ra ngoài
trời là bỏ luôn lớp bảo hiểm đó. Sàn cứng còn lại là chính nhịp quét — vứt rác
ngay sau một lượt thì phải đợi lượt sau, tức 0–30 giây nữa.

---

## 5 · Còn hở — đọc trước khi deploy

**Chưa đo gì cả**: nắng và bóng cây quét ngang vùng · mưa, mặt đường ướt · đèn
pha xe ban đêm. Bài toán là rác đường phố mà **chưa có một phép đo nào trên
camera cố định ngoài trời**.

**Không có ảnh CCTV cố định nào trong 26k ảnh train.** Conf tụt theo độ quen:
drone 0,88 → ven đường 0,81 → sàn gạch trong nhà 0,58. Đây là lý do chính nên
thu vài trăm ảnh rác **trên chính camera sẽ deploy** rồi finetune — tiền lệ đo
được ở nhánh classifier: 300 ô âm tại chỗ đưa FP 7,12 → **0,00**/lượt, còn ô âm
cảnh khác chỉ 2,88 → 2,00.

**Đã so với 4 model rác công khai** (`tools/bench_models.py`), không cái nào
thay được. Đọc phải xem **cả hai cột**, chỉ nhìn recall là bị lừa:

| model | bắt được | hộp thừa | hộp/khung sạch |
|---|---|---|---|
| **của mình** | 18/20 · 90% | **4** | **1,0** |
| sharktide/waste-detection | 20/20 · 100% | 98 | 3,8 |
| turhancan97/yolov8-segment | 18/20 · 90% | 47 | 6,0 |
| esapzoi/litter-detection | 3/20 · 15% | 73 | 9,2 |

`sharktide` đạt 100% bằng cách rải 118 hộp lên 6 khung chứa 20 vật. Ba bộ kia
đều không có mẫu âm trong lúc train — nhiều khả năng đó là chỗ khác biệt.

**Lỗ recall tự động.** `scene_shift ≥ thr_px` → xoá nền → lượt sau mọi ô chưa có
nền **tự chốt hiện trạng làm nền**. Rác đang nằm trong vùng lúc đó thành bình
thường, **im vĩnh viễn**, không log gì báo:

```
A. camera bi hich 30px trong luc co rac -> sau 15 luot: nong=0  *** THANH NEN ***
B. van hanh bam CHOT LAI NEN luc co rac -> sau 15 luot: nong=0  *** THANH NEN ***
C. rang dong 60 phut, rac vut tu truoc  -> sau rang dong: nong=7  van bao duoc
```

A là tự động. Hướng sửa: **dời** nền theo độ lệch ECC đã đo thay vì vứt đi.

**Cảnh đông người không dùng được.** video11 có người ở 99% số lượt → 21/78 ô
nóng, `dwell` vô dụng. Ngưỡng tự kiểm: >70% số lượt có người → không dùng được.

**Đồ đạc bị dời gây cảnh báo.** Bấm `r` sau khi dời, loại đồ đạc khỏi ROI, hoặc
thêm trần cỡ hộp trong tầng xác nhận (chưa làm — đánh đổi thật với túi rác to).

**Chưa cắm được vào worker** — `TrashConsumer` vẫn là comment.

**Không general, và không thể general.** "Vùng này có còn trống không" định nghĩa
tương đối với từng camera. Cái đạt được là **lắp không cần train**.

---

## 6 · Hiện test như nào

```bash
bash tools/selfcheck.sh              # kiem toan goi — chay cai nay truoc
python3 tests/selftest.py            # ~60 case, stdlib, vài giây
python3 tests/integration_test.py    # cần cv2
python3 tools/run_test_cases.py      # sinh lại test_cases/
```

`selfcheck.sh` bắt các lỗi chỉ lộ ra khi ai đó **chạy thật trên máy khác**: cú
pháp, tool có khởi động nổi không, **đường dẫn tuyệt đối** ngoài `training/`,
config có trỏ đúng file model không, file lạ ở thư mục gốc, và cả hai test suite.
Nó đã bắt được hai lỗi thật lúc đóng gói.

`integration_test.py` phủ: cold start · guard đổi sáng (vật to, người che) ·
chốt một lần và mở chốt · chốt lại nền · **bù rung hai chiều** (lệch trên vùng
chết phải nắn, lệch dưới vùng chết phải để yên).

`run_test_cases.py` sinh lại toàn bộ `test_cases/` — chạy lại là ra đúng số trên.

### Ba lỗi mà chính bộ test bắt được

Cả ba đều nằm ở chỗ đã tự tin là xong, và không cái nào lộ ra khi đọc lại code.

- **Bù méo làm hỏng chính thứ nó sinh ra để sửa** (§Cơ chế an toàn). ECC ước
  lượng từ toàn khung kể cả vật mới, nên vật mới làm lệch phép ước lượng. Cộng
  thêm vùng chết viết sai logic (`AND` giữa hai điều kiện) nên nó **luôn nắn**.
- **`scans.csv` không flush** — mất 6 phút cuối của lần chạy 5,9 giờ khi máy ngủ.
- **`dwell` + `ConfirmGate` lọc trùng** — 4,0 phút độ trễ, một nửa là thừa.

### Chín cái bẫy đo đạc đã dính

Tất cả đều nằm ở **thước đo**, không nằm ở code. Ghi lại để không dính lại.

1. **Ảnh TACO/UAVVaste trên đĩa đã thu nhỏ so với metadata** — TACO 960×1280 vs
   1537×2049 ở 119/120 ảnh → mọi hộp lệch ~1,6 lần → probe ra **0% recall**.
2. **Rò rỉ train/val 94% clip RoLID** — dashcam quay liên tục, khung liền nhau
   gần trùng nhau nhưng khác `image_id`. Phải chia theo **clip**.
3. **Probe lấy 90% mẫu từ chính tập train** — recall 38–67% ban đầu là học thuộc.
4. **Biến `sp` dùng lại của ảnh trước** — 723 ảnh bị chia train/val tuỳ tiện.
5. **Đo FP trên khung tĩnh so với chính nó** — "0 báo nhầm" chỉ nói rằng không
   đổi thì không báo.
6. **So detector không cắt ô với có cắt ô** — 29% vs 75% trên cùng dữ liệu.
7. **Chấm điểm khi chưa qua warm-up** — config có `dwell 5 + confirm 4/6` = phải
   8 lượt mới báo; đưa 6 khung rồi chấm thì patch ra 25% trong khi thực tế 90%.
8. **Vùng chết bù méo đòi `d < min_px AND |deg| < min_deg`** — `deg` vượt ngưỡng
   nên điều kiện luôn sai, `min_px` thành vô nghĩa, vùng chết không hề tồn tại.
9. **Công cụ audit sai và `pgrep -f` tự khớp chính nó** — audit gộp nhầm
   `batch_1/000001.jpg` với `batch_2/000001.jpg`; vòng chờ khớp luôn câu lệnh
   bash đang chạy vòng chờ nên treo vĩnh viễn.

---

## Bản đồ file

```
core/pipeline.py     điều phối, ZoneTrashDetector.scan()
core/reference.py    cổng đổi (EMA 8x8 top-8), scene_shift, bù méo ECC
core/verify.py       tầng xác nhận: gộp ô -> vùng -> detector
core/grid.py         dựng lưới, ROI, che khuất — Python thuần
core/clutter.py      mặt nạ nhiễu, lưu JSON có kiểm vân tay
core/gates.py        ConfirmGate (N/M trượt), DedupGate
core/scorers.py      constant / vlm / onnx cho mode classifier

tools/draw_zone.py       vẽ ROI bằng chuột, lưu kèm ảnh mốc
tools/live_view.py       cửa sổ xem trực tiếp, 3 luồng tách rời
tools/live_monitor.py    xem qua trình duyệt (MJPEG, có trễ)
tools/run_video.py       chạy shadow mode, ghi scans.csv
tools/eval.py            chấm FP/ngày, recall sự kiện, độ trễ
tools/shadow_report.py   cảnh báo cụm theo thời gian / vị trí không
tools/run_test_cases.py  sinh lại test_cases/
tools/try_model.py       chạy detector lên ảnh bất kỳ, xem đầu ra THÔ
tools/latency_sweep.py   quét đánh đổi độ trễ <-> báo nhầm
tools/shift_test.py      đo ngưỡng chịu lệch của camera + vùng
tools/standing_test.py   người đứng yên có bị báo không
tools/measure_px.py      đo cỡ vật theo px để chọn cell_px
tools/calibrate.py       dò ngưỡng litter_thr cho mode classifier
tools/bench_models.py    so nhiều model rác bằng cùng một thước đo
tools/bench_sweep.py     quét ngưỡng, so ở điểm làm việc tương đương
tools/keepalive.sh       chạy shadow liên tục, tự bật lại nếu chết
tools/selfcheck.sh       kiểm gói trước khi bàn giao (cú pháp, đường dẫn, test)

tests/selftest.py         ~60 case, stdlib, không cần cv2
tests/integration_test.py cần cv2

models/trash_yolo11n.pt   detector 1 lớp, mAP50 0,747
config/day_cfg.yaml       cấu hình đang chạy thật
test_cases/               kết quả bộ test + ảnh
training/                 dựng bộ dữ liệu + train lại + soát dữ liệu
```

### Hiệu chỉnh tầng xác nhận — `tools/bench_sweep.py`

Ngưỡng tối ưu đo trên model **đứng riêng không chuyển được** sang tầng xác nhận,
vì hai bên đưa vào model hai loại ảnh khác nhau: đứng riêng thì cắt ô 320px
phóng 2× (640px), còn tầng xác nhận cắt vùng quanh cụm ô nóng. Đo được:

| verify.conf (vùng 256) | model đứng riêng | **ghép** |
|---|---|---|
| 0,10 | 86% | **86%** |
| 0,20 | 86% | **29%** |
| 0,30 | 86% | **0%** |

Ở conf 0,30 model đứng riêng vẫn 86% mà ghép sập về 0. **Phải hiệu chỉnh tầng
xác nhận bằng chính bộ test của pipeline, không suy từ số của model.**

Quét cả hai tham số cùng lúc:

| vùng | conf | eco | hộp thừa | chuỗi sạch |
|---|---|---|---|---|
| 256 | 0,10 | 86% | 0 | 28% |
| 320 | 0,10 | 86% | 3 | 6% |
| **320** | **0,20** | **86%** | **3** | **0%** |
| 192 | 0,10 | **48%** ✗ | 0 | 25% |
| 320 | 0,30 | **38%** ✗ | 0 | 0% |

`max_side_px: 320` × `upscale 2` = **640px, đúng bằng ảnh model được train**.
Bản đầu đặt 256 với lý luận "vùng càng to thì vật càng nhỏ so với ảnh" — nghe
hợp lý nhưng phá mất sự khớp thang mà cả vòng làm dữ liệu sinh ra để đạt được.

Hai vách phải tránh: **vùng 192** crop quá chặt, mất bối cảnh, recall sập 48%;
**conf 0,30** sập 38%. Điểm 320/0,20 có biên cả hai phía.
