"""Chạy shadow mode trên video / RTSP / thư mục ảnh.

Mục tiêu KHÔNG phải là "xem nó bắt được rác không" — mà là đếm **số lần báo
nhầm mỗi camera mỗi ngày**. Đó là chỉ số duy nhất quyết định khách có bật tính
năng hay tắt nó sau ba hôm. mAP không liên quan gì ở đây.

    # thông ống, chưa cần model
    python3 tools/run_video.py --source clip.mp4 --zone zone.json --scorer constant

    # tuần đầu: VLM chấm ô, không cần dữ liệu
    python3 tools/run_video.py --source clip.mp4 --zone zone.json --out runs/d1

    # có người/xe che: dùng YOLO rời (worker thật lấy từ ctx.prior["object"])
    python3 tools/run_video.py --source clip.mp4 --zone zone.json --yolo yolo11n.pt

zone.json: {"points": [[0.12,0.53],[0.61,0.50],[0.66,0.88],[0.09,0.91]]}
(toạ độ CHUẨN HOÁ 0..1 — cùng định dạng `zones[].points` backend trả về)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time

# PHẢI đặt trước `import cv2`: OpenCV đọc biến này lúc nạp backend FFmpeg.
# Không có nó thì RTSP mặc định chạy UDP -> mất gói, "Stream timeout triggered"
# và tool im lặng quét được 0 lượt. Camera IP thực tế gần như luôn cần TCP.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    os.environ.get("RTSP_TRANSPORT_OPTS", "rtsp_transport;tcp|stimeout;8000000"),
)

import cv2          # noqa: E402
import numpy as np  # noqa: E402
import yaml         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import ZoneTrashDetector      # noqa: E402
from core.scorers import build_scorer            # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Số lần thử nối lại trước khi bỏ cuộc. Với backoff luỹ tiến (2,4,8,16,32,60,60...)
# thì 30 lần ≈ 20 phút — đủ để camera reboot xong hoặc nhả session.
MAX_RECONNECT = 30


# ------------------------------------------------------------------ nguồn

def iter_frames(source: str, interval_s: float):
    """Sinh (timestamp_giây, frame) đã lấy mẫu theo `interval_s`.

    - thư mục ảnh: mỗi ảnh một mốc, cách nhau interval_s (giả lập)
    - video file : dùng mốc thời gian TRONG video -> 1h video chạy xong trong vài phút
    - rtsp/webcam: dùng đồng hồ thật
    """
    if os.path.isdir(source):
        files = sorted(f for f in os.listdir(source)
                       if os.path.splitext(f)[1].lower() in IMG_EXT)
        for i, f in enumerate(files):
            img = cv2.imread(os.path.join(source, f))
            if img is not None:
                yield i * interval_s, img
        return

    is_file = os.path.isfile(source)
    live = not is_file

    def _open():
        c = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        try:
            c.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
            c.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
        except Exception:  # noqa: BLE001 — build OpenCV cũ không có 2 cờ này
            pass
        return c

    # Camera IP hay từ chối kết nối đầu (giới hạn số session, vừa reboot...).
    # Thử lại vài lần thay vì chết ngay — đã gặp thật trên camera Hikvision.
    cap = None
    for k in range(5 if live else 1):
        cap = _open()
        if cap.isOpened():
            break
        cap.release()
        logging.getLogger("run_video").warning(
            "chưa mở được nguồn, thử lại (%d)...", k + 1)
        time.sleep(2)
    if cap is None or not cap.isOpened():
        sys.exit(f"không mở được nguồn: {source}")
    next_t, t0, fails = 0.0, time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Nguồn live rớt là chuyện thường (mạng, camera reboot, hết
                # session). Camera IP cần HÀNG PHÚT mới nhả session — thử 10 lần
                # cách nhau 2s (=20s) là bỏ cuộc quá sớm: đã giết một phiên chạy
                # ban ngày sau 1,66h vì camera trả 500 liên tục.
                # Backoff luỹ tiến, kiên trì tới ~20 phút.
                if not live or fails >= MAX_RECONNECT:
                    logging.getLogger("run_video").error(
                        "mất kết nối hẳn sau %d lần thử — dừng", fails)
                    break
                fails += 1
                wait = min(60, 2 ** min(fails, 5))
                logging.getLogger("run_video").warning(
                    "mất kết nối, nối lại lần %d/%d sau %ds...",
                    fails, MAX_RECONNECT, wait)
                cap.release()
                time.sleep(wait)
                cap = _open()
                continue
            if fails:
                logging.getLogger("run_video").info("đã nối lại sau %d lần thử", fails)
            fails = 0
            t = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) if is_file else (time.time() - t0)
            if t + 1e-6 >= next_t:
                next_t = t + interval_s
                yield t, frame
    finally:
        cap.release()


class YoloBoxes:
    """Nguồn box person/vehicle rời, chỉ dùng cho tool này.

    Trong worker thật thì KHÔNG cần, mà cũng KHÔNG dùng: che ô theo box người
    đã bị bỏ vì đo được nó giảm đúng 0 cảnh báo trong khi bịt mắt tới 24% diện
    tích vùng. `integration/trash_consumer.py` gọi `scan(..., person_boxes=())`.
    Giữ cờ `--yolo` ở tool này chỉ để so lại được số cũ.
    """

    KEEP = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

    def __init__(self, weights: str, conf: float = 0.35) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("--yolo cần `pip install ultralytics`, hoặc bỏ cờ này đi")
        self.m = YOLO(weights)
        self.conf = conf

    def __call__(self, frame) -> list:
        out = []
        for r in self.m.predict(frame, conf=self.conf, verbose=False):
            names = r.names
            for b in r.boxes:
                if names[int(b.cls)] in self.KEEP:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    out.append((x1, y1, x2, y2))
        return out


# ------------------------------------------------------------------ vẽ

def draw(frame, poly_px, det, res):
    vis = frame.copy()
    cv2.polylines(vis, [np.array(poly_px, np.int32)], True, (0, 220, 255), 2)

    muted = set(det.clutter.muted()) if det.clutter else set()
    for c in (det.grid.cells if det.grid else []):
        if c.id in muted:
            cv2.rectangle(vis, (c.x1, c.y1), (c.x2, c.y2), (140, 140, 140), 1)
    for c in res.hot:
        cv2.rectangle(vis, (c.x1, c.y1), (c.x2, c.y2), (0, 0, 255), 2)
        s = res.scores.get(c.id)
        if s is not None:
            cv2.putText(vis, f"{s:.2f}", (c.x1 + 2, c.y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    tag = "ALERT" if res.alert else ("dirty" if res.dirty else "clean")
    col = (0, 0, 255) if res.alert else ((0, 165, 255) if res.dirty else (0, 200, 0))
    # ASCII thuần: font Hershey của OpenCV không có dấu tiếng Việt, dùng
    # res.summary() ở đây thì chữ ra thành "??=136 ?????i=19".
    line = (f"{tag}  cells={res.n_cells} occl={res.n_occluded} chg={res.n_changed} "
            f"scored={res.n_scored} drop={res.n_dropped} muted={res.n_muted_hit} "
            f"hot={len(res.hot)} ({res.ms:.0f}ms)")
    cv2.putText(vis, line, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    if muted:
        cv2.putText(vis, f"clutter mask: {len(muted)} cells "
                         f"({res.mask_progress*100:.0f}% mature)",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
    return vis


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video | rtsp:// | thư mục ảnh")
    ap.add_argument("--zone", required=True, help="JSON có 'points' chuẩn hoá 0..1")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="runs/shadow")
    ap.add_argument("--camera-id", default="cam0")
    ap.add_argument("--zone-id", default="zone0")
    ap.add_argument("--scorer", help="ghi đè scorer.kind: constant|vlm|onnx")
    ap.add_argument("--yolo", help="weights YOLO để lấy box người/xe (tuỳ chọn)")
    ap.add_argument("--save-all", action="store_true", help="lưu ảnh MỌI lượt, không chỉ lượt báo")
    ap.add_argument("--live", action="store_true", help="hiện cửa sổ xem trực tiếp")
    ap.add_argument("--limit", type=int, default=0, help="dừng sau N lượt quét")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.scorer:
        cfg.setdefault("scorer", {})["kind"] = args.scorer
    with open(args.zone, encoding="utf-8") as f:
        poly_norm = json.load(f)["points"]
    if len(poly_norm) < 3:
        sys.exit("vùng cần >= 3 điểm")

    interval = float(cfg.get("scan", {}).get("interval_s", 30))
    state_dir = os.path.join(args.out, "state")
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    det = ZoneTrashDetector(cfg, build_scorer(cfg.get("scorer", {})),
                            camera_id=args.camera_id, zone_id=args.zone_id,
                            state_dir=state_dir)
    boxes_of = YoloBoxes(args.yolo) if args.yolo else (lambda _f: [])

    csv_path = os.path.join(args.out, "scans.csv")
    fcsv = open(csv_path, "w", newline="", encoding="utf-8")
    wr = csv.writer(fcsv)
    wr.writerow(["t_s", "dirty", "alert", "escalated", "n_hot", "n_raw_hot",
                 "n_cells", "n_occluded", "n_changed", "n_scored", "n_dropped",
                 "n_muted_hit", "shift_px", "stab_px", "stab_deg", "n_verify_dropped",
                 "n_verify_boxes", "mask_progress",
                 "ms", "hot_cells", "notes"])

    # Nói rõ đang chạy tiếp state cũ hay bắt đầu từ trắng — nếu không, một lần
    # chạy warm-start trông y hệt cold start và số liệu bị hiểu sai hoàn toàn.
    old = [f for f in os.listdir(state_dir)] if os.path.isdir(state_dir) else []
    print(f"nguồn   {args.source}")
    print(f"state   {state_dir}  "
          + (f"({len(old)} file có sẵn -> CHẠY TIẾP, không phải cold start)"
             if old else "(trống -> COLD START)"))
    print(f"scorer  {cfg.get('scorer', {}).get('kind')}   nhịp {interval}s\n")

    # Cờ file để người vận hành CHỐT LẠI NỀN trên tiến trình đang chạy:
    #     touch <out>/state/reset.flag
    # Cần vì dời đồ đạc là chuyện thường; không có nó thì phải chờ mặt nạ nhiễu
    # mute (~5 giờ) và suốt thời gian đó hệ thống báo lặp. Trong worker thật thì
    # đây sẽ là một endpoint HTTP.
    reset_flag = os.path.join(state_dir, "reset.flag")

    n_scan = n_alert = n_dirty = 0
    t_first = t_last = None
    ms_sum = 0.0
    try:
        for t, frame in iter_frames(args.source, interval):
            if os.path.exists(reset_flag):
                det.reset_background()
                try:
                    os.remove(reset_flag)
                except OSError:
                    pass
                print(f"[{t:8.1f}s] đã CHỐT LẠI NỀN theo yêu cầu", flush=True)
            res = det.scan(frame, poly_norm, boxes_of(frame), now=t)
            n_scan += 1
            n_dirty += int(res.dirty)
            n_alert += int(res.alert)
            ms_sum += res.ms
            t_first = t if t_first is None else t_first
            t_last = t

            wr.writerow([f"{t:.2f}", int(res.dirty), int(res.alert), int(res.escalated),
                         len(res.hot), len(res.raw_hot), res.n_cells, res.n_occluded,
                         res.n_changed, res.n_scored, res.n_dropped, res.n_muted_hit,
                         f"{res.scene_shift_px:.2f}", f"{res.stab_px:.2f}",
                         f"{res.stab_deg:.3f}", res.n_verify_dropped, len(res.verify_boxes),
                         f"{res.mask_progress:.3f}",
                         f"{res.ms:.1f}",
                         " ".join(f"{c.row},{c.col}" for c in res.hot),
                         " | ".join(res.notes.values())])
            # FLUSH mỗi lượt. Mở file rồi chỉ close() ở cuối thì dòng nằm trong
            # buffer, và tiến trình chết đột ngột là mất sạch phần chưa ghi. Đã
            # mất thật 6 phút cuối của một lần chạy 5,9 giờ khi máy ngủ giữa
            # chừng. Ở nhịp 30s thì flush mỗi lượt không tốn gì.
            fcsv.flush()

            if res.alert or args.save_all or args.live:
                vis = draw(frame, [(int(p[0] * frame.shape[1]), int(p[1] * frame.shape[0]))
                                   for p in poly_norm], det, res)
                if res.alert or args.save_all:
                    kind = "alert" if res.alert else "scan"
                    cv2.imwrite(os.path.join(args.out, "frames",
                                             f"{kind}_{t:09.2f}.jpg"), vis)
                if args.live:
                    cv2.imshow("trash", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if n_scan % 20 == 0:
                print(f"[{t:8.1f}s] {res.summary()}")
            if args.limit and n_scan >= args.limit:
                break
    except KeyboardInterrupt:
        print("\nngắt bởi người dùng")
    finally:
        fcsv.close()
        det.save_state()
        cv2.destroyAllWindows()

    span_s = max(1e-6, (t_last or 0) - (t_first or 0))
    days = span_s / 86400.0
    print("\n" + "=" * 62)
    print(f"  lượt quét        {n_scan}   ({span_s/3600:.2f} giờ nội dung)")
    print(f"  lượt 'bẩn'       {n_dirty}")
    print(f"  CẢNH BÁO         {n_alert}")
    if days > 0:
        print(f"  ---> {n_alert/days:.2f} cảnh báo / camera / ngày")
        print("       (nếu clip này KHÔNG có rác thật thì đây là FP/ngày.")
        print("        Ngưỡng khách chịu được: < 1-2)")
    print(f"  thời gian quét   {ms_sum/max(1,n_scan):.0f} ms/lượt trung bình")
    if det.clutter:
        mp = det.clutter.progress()
        print(f"  mặt nạ nhiễu     {len(det.clutter.muted())} ô mute, "
              f"chín {mp*100:.1f}%")
        if mp < 1.0:
            print("       (chưa chín — FP hệ thống vẫn còn, chạy tiếp shadow mode)")
    print(f"  log              {csv_path}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
