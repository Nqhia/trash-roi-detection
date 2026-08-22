"""Cửa sổ xem TRỰC TIẾP, mượt, không qua web.

    python3 tools/live_view.py --source rtsp://... --zone zone.json \\
        --config cfg.yaml --yolo yolo11n.pt

Ba luồng tách hẳn nhau, đó là lý do hình không giật:
  - luồng ĐỌC   : cap.read() liên tục, luôn giữ khung mới nhất (RTSP đệm sẵn,
                  đọc chậm là ảnh trễ dần rồi nhảy cóc)
  - luồng QUÉT  : cứ interval_s mới chạy YOLO + det.scan() một lần.
                  YOLO trên CPU mất 30-80ms; để chung luồng hiển thị là rớt hình.
  - luồng CHÍNH : chỉ vẽ lại overlay của lượt quét gần nhất lên khung mới nhất
                  rồi imshow. ~1ms/khung nên chạy đúng FPS của camera.

Phím:
  q / ESC  thoát          r  chốt lại nền (reset_background)
  g        ẩn/hiện lưới   b  ẩn/hiện box người-xe
  s        lưu ảnh        space  tạm dừng hình (quét vẫn chạy)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|stimeout;8000000")

import cv2          # noqa: E402
import numpy as np  # noqa: E402
import yaml         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.capture import open_source   # noqa: E402

from core.grid import poly_to_px                          # noqa: E402
from core.pipeline import ZoneTrashDetector               # noqa: E402
from core.reference import make_thumb, scene_shift        # noqa: E402
from core.scorers import build_scorer                     # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
C_HOT, C_WARM, C_MUTE, C_GRID = (60, 60, 255), (0, 165, 255), (130, 130, 130), (80, 80, 80)
C_ROI, C_BOX, C_OK = (0, 220, 255), (255, 170, 0), (90, 220, 90)


class Grabber(threading.Thread):
    """Đọc khung liên tục, chỉ giữ khung MỚI NHẤT."""

    daemon = True

    def __init__(self, source: str):
        super().__init__()
        self.source, self.frame, self.lock = source, None, threading.Lock()
        self.stop_ev, self.fails, self.n = threading.Event(), 0, 0

    def _open(self):
        cap = open_source(self.source, tries=5)
        if cap is None:
            raise SystemExit(f'khong lay duoc khung tu {self.source}')
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # càng ít đệm càng ít trễ
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
        except Exception:  # noqa: BLE001 — build cũ không có mấy cờ này
            pass
        return cap

    def run(self):
        cap = self._open()
        while not self.stop_ev.is_set():
            ok, f = cap.read()
            if not ok:
                self.fails += 1
                cap.release()
                time.sleep(min(10, 2 ** min(self.fails, 4)))
                cap = self._open()
                continue
            with self.lock:
                self.frame, self.n = f, self.n + 1
        cap.release()

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


class Scanner(threading.Thread):
    """Chạy pipeline theo nhịp, đẩy kết quả ra cho luồng vẽ."""

    daemon = True

    def __init__(self, det, grab: Grabber, poly, interval: float, boxes_of, out: str):
        super().__init__()
        self.det, self.grab, self.poly = det, grab, poly
        self.interval, self.boxes_of, self.out = interval, boxes_of, out
        self.stop_ev, self.reset_ev = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.rects: list = []      # (x1,y1,x2,y2,màu,dày) theo px khung gốc
        self.boxes: list = []
        self.hud, self.n_alert, self.last_alert = "dang khoi dong...", 0, 0.0
        self.vboxes: list = []
        self.log: deque = deque(maxlen=8)
        self.alerting = False

    def run(self):
        nxt = 0.0
        while not self.stop_ev.is_set():
            if self.reset_ev.is_set():
                self.det.reset_background()
                self.reset_ev.clear()
                self.log.appendleft((time.strftime("%H:%M:%S"), "da chot lai nen"))
            now = time.time()
            if now < nxt:
                time.sleep(min(0.2, nxt - now))
                continue
            frame = self.grab.latest()
            if frame is None:
                time.sleep(0.2)
                continue
            nxt = now + self.interval
            boxes = self.boxes_of(frame)
            res = self.det.scan(frame, self.poly, boxes, now=now)
            self.publish(frame, res, boxes, now)

    def publish(self, frame, res, boxes, now):
        det = self.det
        muted = set(det.clutter.muted()) if det.clutter else set()
        hot = {c.id for c in res.hot}
        rects = []
        for c in det.grid.cells:
            v = res.scores.get(c.id, 0.0)
            if c.id in hot:
                rects.append((c.x1, c.y1, c.x2, c.y2, C_HOT, 2))
            elif c.id in muted:
                rects.append((c.x1, c.y1, c.x2, c.y2, C_MUTE, 1))
            elif v > 0:
                rects.append((c.x1, c.y1, c.x2, c.y2, C_WARM, 1))
            else:
                rects.append((c.x1, c.y1, c.x2, c.y2, C_GRID, 1))
        tag = ("ALERT" if res.alert else
               "GLOBAL-RESEED" if res.global_change else
               "dirty" if res.dirty else "clean")
        stab = (f" bu={res.stab_px:.1f}px/{res.stab_deg:+.2f}d"
                if res.stab_px or res.stab_deg else "")
        ver = ""
        if res.verify_failed:
            ver = " VERIFY-LOI"
        elif res.n_verify_dropped or res.verify_boxes:
            ver = f" bacbo={res.n_verify_dropped} det={len(res.verify_boxes)}"
        hud = (f"{tag}  o={res.n_cells} che={res.n_occluded} doi={res.n_changed} "
               f"nong={len(res.hot)} moi={res.n_fresh_hot} mute={len(muted)} "
               f"mask={res.mask_progress*100:.0f}%{stab}{ver}  {res.ms:.0f}ms")
        if res.alert:
            self.n_alert += 1
            self.last_alert = now
            self.log.appendleft((time.strftime("%H:%M:%S"),
                                 f"CANH BAO #{self.n_alert} - {len(res.hot)} o"
                                 f" ({res.n_fresh_hot} moi)"))
            # Lưu khung CÓ OVERLAY. Bản đầu lưu khung thô nên xem lại cảnh báo
            # là đoán mò — không biết ô nào nóng, detector khoanh cái gì. Vừa
            # dính đúng lúc cần soi một cảnh báo nhầm vào chân ghế.
            snap = frame.copy()
            for c in res.hot:
                cv2.rectangle(snap, (c.x1, c.y1), (c.x2, c.y2), C_HOT, 2)
            for b in res.verify_boxes:
                cv2.rectangle(snap, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (255, 0, 200), 2)
            cv2.rectangle(snap, (0, 0), (snap.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(snap, hud, (8, 18), FONT, .5, C_HOT, 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(self.out, f"alert_{self.n_alert:03d}.jpg"), snap)
        elif res.rearmed:
            self.log.appendleft((time.strftime("%H:%M:%S"), "vung sach lai - mo chot"))
        with self.lock:
            self.rects, self.boxes, self.hud = rects, boxes, hud
            self.vboxes = list(res.verify_boxes)
            self.alerting = bool(res.alert)

    def snapshot(self):
        with self.lock:
            return self.rects, self.boxes, self.hud, list(self.log), self.vboxes


def check_anchor(zone_path: str, frame, thr_px: float, save: bool) -> str | None:
    """So khung hiện tại với ảnh mốc lưu lúc vẽ vùng.

    scene_shift trong pipeline chỉ so hai lượt LIỀN NHAU, nên camera bị xoay
    lúc chương trình không chạy thì không ai biết — vùng ROI lệch đi mà vẫn
    chạy ngon lành. Đã dính thật: camera ngóc lên, cả vùng sàn tụt khỏi khung.
    """
    path = os.path.splitext(zone_path)[0] + ".anchor.png"
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if save or not os.path.exists(path):
        # PNG chỉ chứa được 8 bit; make_thumb trả float32 nên phải ép kiểu,
        # không thì imwrite tự hạ cấp và lần đọc sau phaseCorrelate ném lỗi.
        cv2.imwrite(path, make_thumb(g).astype(np.uint8))
        return None
    old = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    d = scene_shift(old, make_thumb(g), frame.shape[1])
    if d > thr_px:
        return (f"CAMERA DA LECH ~{d:.0f}px so voi luc ve vung (nguong {thr_px:.0f}). "
                f"Ve lai ROI hoac chinh camera ve cho cu, roi chay lai voi --save-anchor")
    return None


def draw(disp, sc: Scanner, s: float, show_grid: bool, show_box: bool,
         fps: float, warn: str | None, banner_until: float):
    rects, boxes, hud, log, vboxes = sc.snapshot()
    h, w = disp.shape[:2]
    if show_grid:
        for x1, y1, x2, y2, col, th in rects:
            if col is C_GRID and th == 1:
                cv2.rectangle(disp, (int(x1*s), int(y1*s)), (int(x2*s), int(y2*s)), col, 1)
    for x1, y1, x2, y2, col, th in rects:     # ô có trạng thái vẽ đè lên lưới
        if col is not C_GRID:
            cv2.rectangle(disp, (int(x1*s), int(y1*s)), (int(x2*s), int(y2*s)), col, th)
    if show_box:
        for b in boxes:
            cv2.rectangle(disp, (int(b[0]*s), int(b[1]*s)),
                          (int(b[2]*s), int(b[3]*s)), C_BOX, 2)
    # Hộp detector xác nhận: đây mới là thứ quyết định có báo hay không.
    for b in vboxes:
        cv2.rectangle(disp, (int(b[0]*s), int(b[1]*s)),
                      (int(b[2]*s), int(b[3]*s)), (255, 0, 200), 2)
    cv2.rectangle(disp, (0, 0), (w, 24), (0, 0, 0), -1)
    col = C_HOT if hud.startswith("ALERT") else (C_WARM if "dirty" in hud else C_OK)
    cv2.putText(disp, hud, (8, 17), FONT, .48, col, 1, cv2.LINE_AA)
    cv2.putText(disp, f"{fps:4.1f} fps", (w - 78, 17), FONT, .48, (200, 200, 200), 1,
                cv2.LINE_AA)
    y = h - 8 - 15 * len(log)
    for t, msg in reversed(log):
        cv2.putText(disp, f"{t} {msg}", (8, y), FONT, .42,
                    C_HOT if "CANH BAO" in msg else (210, 210, 210), 1, cv2.LINE_AA)
        y += 15
    if sc.alerting and int(time.time() * 2) % 2 == 0:
        cv2.rectangle(disp, (2, 26), (w - 3, h - 3), C_HOT, 3)
    if warn and time.time() < banner_until:
        cv2.rectangle(disp, (0, 26), (w, 54), (0, 0, 140), -1)
        cv2.putText(disp, warn[:105], (8, 46), FONT, .48, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="runs/live")
    ap.add_argument("--yolo")
    ap.add_argument("--interval", type=float, help="ghi đè scan.interval_s")
    ap.add_argument("--width", type=int, default=1280, help="bề ngang cửa sổ")
    ap.add_argument("--save-anchor", action="store_true",
                    help="lưu lại ảnh mốc sau khi vẽ lại vùng")
    ap.add_argument("--camera-id", default="cam0")
    ap.add_argument("--zone-id", default="zone0")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.zone, encoding="utf-8") as f:
        poly = json.load(f)["points"]
    interval = args.interval or float(cfg.get("scan", {}).get("interval_s", 30))
    os.makedirs(args.out, exist_ok=True)

    grab = Grabber(args.source)
    grab.start()
    t0 = time.time()
    while grab.latest() is None:
        if time.time() - t0 > 30:
            return print("khong mo duoc nguon:", args.source) or 1
        time.sleep(0.2)
    frame = grab.latest()
    print(f"khung {frame.shape[1]}x{frame.shape[0]}  nhip {interval}s  "
          f"che do {cfg.get('decide', {}).get('mode', 'classifier')}")

    warn = check_anchor(args.zone, frame,
                        float(cfg.get("scene_shift", {}).get("thr_px", 12.0)),
                        args.save_anchor)
    if warn:
        print("!! " + warn)

    scorer = build_scorer(cfg.get("scorer", {"kind": "constant", "value": 0.0}))
    boxes_of = (lambda _f: [])
    if args.yolo:
        from tools.run_video import YoloBoxes
        boxes_of = YoloBoxes(args.yolo)
    det = ZoneTrashDetector(cfg, scorer, camera_id=args.camera_id,
                            zone_id=args.zone_id,
                            state_dir=os.path.join(args.out, "state"))
    sc = Scanner(det, grab, poly, interval, boxes_of, args.out)
    sc.start()

    s = args.width / frame.shape[1]
    win = "Giam sat rac - truc tiep"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    poly_px = None
    show_grid = show_box = True
    paused = False
    banner_until = time.time() + 20
    times: deque = deque(maxlen=30)
    last = None
    try:
        while True:
            f = last if paused else grab.latest()
            if f is None:
                time.sleep(0.03)
                continue
            last = f
            disp = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if poly_px is None:
                poly_px = np.array(poly_to_px(poly, disp.shape[1], disp.shape[0]), np.int32)
            cv2.polylines(disp, [poly_px], True, C_ROI, 2)
            times.append(time.time())
            fps = (len(times) - 1) / max(1e-6, times[-1] - times[0]) if len(times) > 1 else 0
            draw(disp, sc, s, show_grid, show_box, fps, warn, banner_until)
            cv2.imshow(win, disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("r"):
                sc.reset_ev.set()
            elif k == ord("g"):
                show_grid = not show_grid
            elif k == ord("b"):
                show_box = not show_box
            elif k == ord(" "):
                paused = not paused
            elif k == ord("s"):
                p = os.path.join(args.out, f"snap_{time.strftime('%H%M%S')}.jpg")
                cv2.imwrite(p, disp)
                print("luu", p)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sc.stop_ev.set()
        grab.stop_ev.set()
        cv2.destroyAllWindows()
    print(f"dung. {sc.n_alert} canh bao, anh luu o {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
