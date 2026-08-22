"""Dựng bộ train BẰNG ĐÚNG THỨ detector nhìn thấy lúc chạy thật: ô đã phóng to.

    python3 tools/make_tiles_data.py --out data/tiles --tile 320 --upscale 2.0

Lý do phải có bước này. Lúc suy luận, `detect_in_zone` cắt ROI thành ô 320px rồi
PHÓNG TO trước khi đưa vào model — đó là toàn bộ cách hướng này bắt được vật nhỏ.
Nhưng bản train đầu tiên lại học trên ảnh nguyên, thu về 960px. Hai thang khác
hẳn nhau: cùng một vỏ chai, lúc train chiếm ~15px, lúc chạy chiếm ~50px. Model
học phân bố nào thì giỏi phân bố đó, nên đo được 0/12 vật dưới 25px trên khung
eco dù val mAP50 lên tới 0,595.

Cắt ô lúc train còn được một thứ MIỄN PHÍ và quan trọng không kém: **ô rỗng**.
Cả ba bộ dữ liệu chỉ có ảnh chứa rác, không một khung sạch nào, nên model chưa
bao giờ học "cái gì KHÔNG phải rác" — đo được 9,9 hộp thừa trên mỗi khung CCTV
sạch. Ô rỗng cắt từ chính ảnh có rác là mẫu âm cùng cảnh, cùng ánh sáng, cùng
nền: loại mẫu âm khó nhất và đúng nhất.

Hộp nằm vắt qua mép ô: giữ nếu >=60% diện tích nằm trong ô (rồi cắt cho khít),
còn 10-60% thì BỎ CẢ Ô. Không thể để một nửa cái chai nằm trong ô mà không có
nhãn — đó là dạy model rằng nửa cái chai là nền.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re

import cv2

ROOT = "/mnt/c/Users/kangh/Documents/TrashDataset"
WADE = f"{ROOT}/data/wade/wade-ai/Trash_Detection/trash/dataset"
SOURCES = [
    ("TACO", f"{ROOT}/data/taco/annotations.json", f"{ROOT}/data/taco/images",
     lambda fn: fn.replace("/", "_")),
    ("RoLID", f"{ROOT}/data/rolid/RoLID-11K/validation.json",
     f"{ROOT}/data/rolid/RoLID-11K/val_images", lambda fn: fn.split("/")[-1]),
    # 7990 ảnh / 14644 nhãn nằm trong 5 file zip, trước đây bỏ không. Đây là bộ
    # ĐƯỜNG PHỐ — miền yếu nhất (recall 21-46%) và cũng là miền gần CCTV nhất.
    ("RoLIDtr", f"{ROOT}/data/rolid/RoLID-11K/training.json",
     f"{ROOT}/data/rolid/RoLID-11K/train_images", lambda fn: fn.split("/")[-1]),
    ("UAVVaste", f"{ROOT}/data/uavvaste/annotations.json",
     f"{ROOT}/data/uavvaste/images", lambda fn: fn.split("/")[-1]),
    # Wade-AI: ảnh điện thoại chụp rác ngoài đường, do cộng đồng đóng góp ở
    # nhiều nước. Khác hẳn ba bộ kia về người chụp, góc và bối cảnh -> đúng thứ
    # cần để bớt phụ thuộc vào một miền.
    ("WadeTr", f"{WADE}/train_wade_ai.json", f"{WADE}/train",
     lambda fn: fn.split("/")[-1]),
    ("WadeVa", f"{WADE}/val_wade_ai.json", f"{WADE}/val",
     lambda fn: fn.split("/")[-1]),
]
# GINI có 1605 ảnh trả về từ truy vấn kiểu "city garbage" nhưng người gán nhãn
# xác nhận là KHÔNG có rác. Đó là mẫu âm KHÓ đúng nghĩa: cảnh đường phố, thùng,
# đồ lộn xộn, ẩm ướt — mọi thứ trông giống rác trừ chính rác. Nhãn vùng của GINI
# quá lỏng để làm nhãn detection, nhưng phần "không rác" thì dùng được ngay.
GINI_NEG = (f"{ROOT}/data/gini/spotgarbage-GINI-master/spotgarbage/"
            "non-garbage-queried-images")
# Ảnh nằm trong 31 thư mục con đặt theo TRUY VẤN, không nằm phẳng — lần đầu quét
# phẳng nên ra đúng 0 ô. Và không phải truy vấn nào cũng dùng được: "Night+Sky",
# "fruits+vegetables", "face" thì làm mẫu âm chẳng dạy được gì. Chỉ giữ các
# truy vấn cảnh đường/ngoài trời — mẫu âm chỉ có giá trị khi nó GIỐNG mặt tích
# cực: mặt đường trống, có thể có rác nhưng không có.
GINI_KEEP = {"Indian+roads", "city+street", "clean+road", "countryside", "crowd",
             "earth+dust", "indian+railway+tracks", "rural+area", "suburb",
             "buildings", "vehicles", "people", "chaos", "chaos+cable",
             "environment", "Places"}
KEEP_IN = 0.60      # >= phần này nằm trong ô -> giữ hộp
DROP_IN = 0.10      # trong khoảng (DROP_IN, KEEP_IN) -> bỏ cả ô


def clip_of(name: str, fn: str) -> str:
    """Khoá CLIP: gom các khung cùng một video vào cùng một nhóm.

    RoLID là dashcam quay liên tục — frame1822/1823/1824 gần như trùng nhau
    nhưng mang image_id khác nhau. Chia train/val ngẫu nhiên theo ẢNH thì chúng
    rơi về hai phía, và val hoá ra đo lại đúng thứ model vừa học thuộc. Đo được:
    94% clip của RoLID-val và 77% của RoLID-train bị rò rỉ kiểu này.
    """
    base = fn.split("/")[-1]
    if name.startswith("RoLID"):
        return "R:" + re.sub(r"_frame\d+\.\w+$", "", base)
    if name == "UAVVaste":
        m = re.match(r"(BATCH_[A-Za-z0-9]+)_", base)
        if m:
            return "U:" + m.group(1)
    # Ảnh rời: dùng NGUYÊN đường dẫn, không dùng tên trần. TACO có
    # batch_1/000001.jpg và batch_2/000001.jpg — cắt lấy tên trần là gộp nhầm
    # hai ảnh khác nhau làm một.
    #
    # KHÔNG gắn tên nguồn vào khoá: Wade có 12 tên file nằm ở CẢ train lẫn val
    # của bộ gốc. Gắn tên nguồn thì WadeTr:482.jpg và WadeVa:482.jpg thành hai
    # clip khác nhau, và cùng một tấm ảnh rơi về hai phía. Khoá phải nhận diện
    # TẤM ẢNH, không nhận diện nguồn.
    return fn


def split_of(clip: str, val_frac: float) -> str:
    """Chia theo CLIP và ổn định giữa các lần chạy.

    hash() của Python có seed ngẫu nhiên mỗi tiến trình nên không dùng được:
    dựng lại bộ là ảnh nhảy sang phía khác, mọi so sánh giữa hai lần train mất
    ý nghĩa.
    """
    d = hashlib.md5(clip.encode("utf-8")).hexdigest()
    return "val" if (int(d[:8], 16) % 10000) < val_frac * 10000 else "train"


def usable_negative(crop, drone: bool, min_std: float = 16.0) -> bool:
    """Ô rỗng có đáng làm mẫu âm không.

    Soi 12 ô rỗng đầu tiên thì ~60% là trời, mây, kính chắn gió mờ, ngọn cây.
    Model có bao giờ bắn lên trời đâu mà cần dạy. Mẫu âm chỉ đáng giá khi nó là
    MẶT ĐẤT — thứ dễ nhầm với rác.
    """
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if float(g.std()) < min_std:      # phẳng lì: trời, tường, ảnh cháy sáng
        return False
    if not drone and float(g.mean()) > 200:   # mảng sáng trắng = trời
        return False
    return True


def tiles_of(w, h, tile, overlap):
    step = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), step))
    ys = list(range(0, max(1, h - tile + 1), step))
    if xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    if ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    return [(x, y) for y in ys for x in xs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/tiles")
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--upscale", type=float, default=2.0)
    ap.add_argument("--neg-per-img", type=int, default=2,
                    help="số ô RỖNG giữ lại mỗi ảnh (mẫu âm cùng cảnh)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    # Mặc định TẮT. Kiểm bằng mắt 6 tấm lấy từ đúng 16 thư mục đã lọc thì ra:
    # một ảnh vệ tinh Đại Tây Dương, hai ảnh stock có watermark, và tệ nhất là
    # một tấm "clean+road" chụp nhóm người đi nhặt rác đang XÁCH TÚI RÁC TRẮNG
    # — gắn nhãn "không có rác". Mẫu âm đó dạy model rằng túi rác không phải
    # rác. Ô rỗng cắt từ chính ảnh có rác an toàn hơn hẳn: chúng rỗng vì KHÔNG
    # CÓ nhãn nào ở đó, không phải vì ai đó bảo thế.
    ap.add_argument("--gini-neg", type=int, default=0,
                    help="số ô mẫu âm từ ảnh 'không rác' GINI (0 = tắt, xem lý do trong code)")
    ap.add_argument("--only", default="",
                    help="chỉ dùng các nguồn này, cách nhau bằng dấu phẩy "
                         "(TACO,RoLID,RoLIDtr,UAVVaste,WadeTr,WadeVa)")
    ap.add_argument("--max-box-frac", type=float, default=0.40,
                    help="bỏ nhãn lớn hơn ngần này phần cạnh dài (nhãn mức cả đống)")
    ap.add_argument("--ctx-frac", type=float, default=0.12,
                    help="vật lớn hơn ngần này phần cạnh dài -> coi là ảnh chụp "
                         "cận, giữ nguyên khung thay vì cắt ô")
    ap.add_argument("--max-per-src", type=int, default=0,
                    help="trần số ảnh mỗi nguồn (0 = không giới hạn)")
    args = ap.parse_args()

    random.seed(0)
    for sp in ("train", "val"):
        for k in ("images", "labels"):
            os.makedirs(os.path.join(args.out, sp, k), exist_ok=True)

    tot_pos = tot_neg = tot_box = 0
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    for name, annf, imgd, ren in SOURCES:
        if only and name not in only:
            print(f"  bỏ qua {name} (không có trong --only)")
            continue
        if not os.path.exists(annf):
            print(f"  bỏ qua {name}")
            continue
        ann = json.load(open(annf))
        meta = {im["id"]: im for im in ann["images"]}
        by = {}
        for a in ann["annotations"]:
            by.setdefault(a["image_id"], []).append(a["bbox"])
        n_pos = n_neg = n_box = n_whole = n_region = n_drop_img = n_sky = 0
        ids = list(by)
        # RoLID là các KHUNG LIÊN TIẾP của video (frame1822, 1823, 1824...), gần
        # như trùng nhau. Lấy hết 7990 ảnh vừa nhân đôi thời gian train vừa làm
        # bộ dữ liệu nghiêng hẳn về một miền. Xáo rồi cắt trần là đủ đa dạng.
        if args.max_per_src and len(ids) > args.max_per_src:
            random.shuffle(ids)
            ids = ids[:args.max_per_src]
        for iid in ids:
            raw = by[iid]
            img = cv2.imread(os.path.join(imgd, ren(meta[iid]["file_name"])))
            if img is None:
                continue
            h, w = img.shape[:2]
            # Ảnh trên đĩa đã thu nhỏ so với lúc gán nhãn -> phải quy đổi.
            k = w / float(meta[iid]["width"])
            # Kẹp vào trong khung: có annotation tràn ra ngoài mép ảnh, để
            # nguyên thì sinh nhãn YOLO có toạ độ ngoài [0,1] (audit đếm được 1).
            boxes = [(max(0.0, bx * k), max(0.0, byy * k),
                      min(float(w), (bx + bw) * k), min(float(h), (byy + bh) * k))
                     for bx, byy, bw, bh in raw if bw * k >= 2 and bh * k >= 2]
            boxes = [b for b in boxes if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
            # Bỏ nhãn mức CẢ ĐỐNG. Wade có 14,9% và TACO 8,9% số nhãn là hộp phủ
            # 40-77% khung — nhãn cho nguyên một bãi rác chứ không phải một vật.
            # Không sai với người, nhưng dạy model bắn ra hộp khổng lồ, và FP đo
            # được ở bản trước đúng là hộp to phủ cả tấm vách gỗ, cả cái quầy.
            # Ảnh mất hết nhãn thì BỎ CẢ ẢNH: giữ lại mà không có nhãn là biến
            # một bãi rác thành mẫu âm.
            if args.max_box_frac < 1.0:
                lim = args.max_box_frac * max(w, h)
                kept = [b for b in boxes if max(b[2] - b[0], b[3] - b[1]) <= lim]
                if len(kept) != len(boxes):
                    n_region += len(boxes) - len(kept)
                    if not kept:
                        n_drop_img += 1
                        continue
                boxes = kept
            if not boxes:
                continue
            # Ảnh nhỏ hơn ô thì không cắt được -> bỏ, khỏi phải đệm giả.
            if w < args.tile or h < args.tile:
                continue

            # Cắt ô chỉ có nghĩa khi ảnh CÓ bối cảnh rộng để cắt. TACO và Wade
            # là ảnh chụp cận: vật chiếm 20-40% khung, cắt ô 320px chỉ được một
            # mảng vân bề mặt phóng to — dạy model sai hoàn toàn thang kích
            # thước so với CCTV. Mà thu nhỏ cũng không cứu được: tỉ lệ vật/khung
            # là bất biến theo phép thu phóng, chỉ cắt rộng ra mới đổi được, và
            # ảnh macro thì không có phần rộng nào để cắt.
            #
            # Nên quyết định theo TỪNG ẢNH: vật nhỏ so với khung -> cắt ô (học
            # đúng thang). Vật to so với khung -> giữ nguyên cả ảnh, thu về đúng
            # cỡ ảnh ra (học hình dạng rác, để augment lo phần thang).
            # Chốt phía train/val NGAY ĐÂY, trước mọi nhánh ghi file. Bản trước
            # đặt dòng này ở dưới nhánh "ảnh chụp cận", nên 723 ảnh chụp cận ghi
            # bằng `sp` còn sót lại của ẢNH TRƯỚC ĐÓ — chia ngẫu nhiên một cách
            # âm thầm, và audit bắt được nó dưới dạng vài clip nằm cả hai bên.
            sp = split_of(clip_of(name, meta[iid]["file_name"]), args.val_frac)
            drone = name == "UAVVaste"

            med = sorted(max(x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes)
            med = med[len(med) // 2]
            out_px = int(args.tile * args.upscale)
            if med > args.ctx_frac * max(w, h):
                s = out_px / max(w, h)
                whole = cv2.resize(img, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_AREA if s < 1
                                   else cv2.INTER_CUBIC)
                oh, ow = whole.shape[:2]
                lines = [f"0 {(x1+x2)/2*s/ow:.6f} {(y1+y2)/2*s/oh:.6f} "
                         f"{(x2-x1)*s/ow:.6f} {(y2-y1)*s/oh:.6f}"
                         for x1, y1, x2, y2 in boxes]
                stem = f"{name}_{iid}_full"
                cv2.imwrite(os.path.join(args.out, sp, "images", stem + ".jpg"),
                            whole, [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(os.path.join(args.out, sp, "labels", stem + ".txt"), "w") as f:
                    f.write("\n".join(lines) + "\n")
                n_pos += 1
                n_box += len(lines)
                n_whole += 1
                continue
            empties = []
            for ti, (tx, ty) in enumerate(tiles_of(w, h, args.tile, args.overlap)):
                keep, bad = [], False
                for x1, y1, x2, y2 in boxes:
                    ix1, iy1 = max(x1, tx), max(y1, ty)
                    ix2, iy2 = min(x2, tx + args.tile), min(y2, ty + args.tile)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    frac = inter / max(1e-9, (x2 - x1) * (y2 - y1))
                    if frac >= KEEP_IN:
                        keep.append((ix1 - tx, iy1 - ty, ix2 - tx, iy2 - ty))
                    elif frac > DROP_IN:
                        bad = True      # vật bị cắt đôi -> ô này không dùng được
                        break
                if bad:
                    continue
                if not keep:
                    empties.append((tx, ty))
                    continue
                crop = img[ty:ty + args.tile, tx:tx + args.tile]
                s = args.upscale
                out = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
                oh, ow = out.shape[:2]
                lines = [f"0 {((a+c)/2*s)/ow:.6f} {((b+d)/2*s)/oh:.6f} "
                         f"{((c-a)*s)/ow:.6f} {((d-b)*s)/oh:.6f}"
                         for a, b, c, d in keep]
                stem = f"{name}_{iid}_t{ti}"
                cv2.imwrite(os.path.join(args.out, sp, "images", stem + ".jpg"), out,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                with open(os.path.join(args.out, sp, "labels", stem + ".txt"), "w") as f:
                    f.write("\n".join(lines) + "\n")
                n_pos += 1
                n_box += len(keep)
            # Ô rỗng: mẫu âm CÙNG CẢNH. Ultralytics coi ảnh không có file nhãn
            # là ảnh nền, nên chỉ cần ghi ảnh, không ghi .txt.
            random.shuffle(empties)
            n_kept = 0
            for j, (tx, ty) in enumerate(empties):
                if n_kept >= args.neg_per_img:
                    break
                crop = img[ty:ty + args.tile, tx:tx + args.tile]
                if not usable_negative(crop, drone):
                    n_sky += 1
                    continue
                n_kept += 1
                s = args.upscale
                out = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(args.out, sp, "images",
                                         f"{name}_{iid}_n{j}.jpg"), out,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                n_neg += 1
        print(f"  {name}: {n_pos} ảnh có rác ({n_box} hộp), {n_neg} ô rỗng"
              f"  [{n_whole} chụp cận giữ khung, {n_pos - n_whole} cắt ô]"
              f"  bỏ {n_region} nhãn cả đống / {n_drop_img} ảnh"
              f", {n_sky} ô rỗng là trời/mờ")
        tot_pos += n_pos
        tot_neg += n_neg
        tot_box += n_box

    # --- mẫu âm khó từ GINI ------------------------------------------------
    if args.gini_neg and os.path.isdir(GINI_NEG):
        fs = []
        for q in sorted(os.listdir(GINI_NEG)):
            if q not in GINI_KEEP:
                continue
            qd = os.path.join(GINI_NEG, q)
            if not os.path.isdir(qd):
                continue
            fs += [os.path.join(qd, f) for f in sorted(os.listdir(qd))
                   if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")]
        random.shuffle(fs)
        n = 0
        for path in fs:
            if n >= args.gini_neg:
                break
            fn = os.path.basename(path)
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            if w < args.tile or h < args.tile:
                continue
            cand = tiles_of(w, h, args.tile, args.overlap)
            random.shuffle(cand)
            sp = "val" if random.random() < args.val_frac else "train"
            for j, (tx, ty) in enumerate(cand[:args.neg_per_img]):
                crop = img[ty:ty + args.tile, tx:tx + args.tile]
                out = cv2.resize(crop, None, fx=args.upscale, fy=args.upscale,
                                 interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(args.out, sp, "images",
                                         f"GINIneg_{os.path.splitext(fn)[0]}_{j}.jpg"),
                            out, [cv2.IMWRITE_JPEG_QUALITY, 92])
                n += 1
        print(f"  GINI không-rác: {n} ô rỗng (mẫu âm khó, cảnh đường phố)")
        tot_neg += n

    yml = os.path.join(args.out, "trash.yaml")
    with open(yml, "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n"
                "train: train/images\nval: val/images\nnc: 1\nnames: [trash]\n")
    tr = len(os.listdir(os.path.join(args.out, "train", "images")))
    va = len(os.listdir(os.path.join(args.out, "val", "images")))
    print(f"\n{tot_pos} ô có rác + {tot_neg} ô rỗng = {tr} train / {va} val")
    print(f"ô {args.tile}px phóng {args.upscale}x -> ảnh "
          f"{int(args.tile*args.upscale)}px  ->  {yml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
