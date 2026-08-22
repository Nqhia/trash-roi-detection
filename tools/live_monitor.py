"""Chạy pipeline trên luồng RTSP và phục vụ giao diện xem TRỰC TIẾP trên trình duyệt.

    python3 tools/live_monitor.py --source rtsp://... --zone zone.json \\
        --config config.yaml --yolo yolo11n.pt --port 8091

Rồi mở http://localhost:8091 (hoặc http://<ip-máy>:8091 từ máy khác).

Giao diện có:
  - ảnh trực tiếp kèm overlay: lưới ô, ô nóng (đỏ), ô vừa đổi (cam), ô bị che (xanh)
  - số liệu cập nhật liên tục: ô đổi/nóng/che, cảnh báo, tiến độ mặt nạ nhiễu
  - danh sách cảnh báo gần nhất, bấm vào xem ảnh
  - nút CHỐT LẠI NỀN — dùng khi dời đồ đạc hợp lệ, khỏi chờ mặt nạ nhiễu ~5 giờ

Dùng MJPEG + http.server của stdlib: không thêm phụ thuộc, không cần màn hình
(chạy trong WSL/không có X vẫn xem được từ Windows).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|stimeout;8000000")

import cv2          # noqa: E402
import numpy as np  # noqa: E402
import yaml         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.grid import poly_to_px                   # noqa: E402
from core.pipeline import ZoneTrashDetector        # noqa: E402
from core.scorers import build_scorer              # noqa: E402
from tools.run_video import YoloBoxes, iter_frames  # noqa: E402

PAGE = """<!doctype html><meta charset="utf-8"><title>Giám sát rác — trực tiếp</title>
<style>
 :root{--bg:#101418;--fg:#e6edf3;--mut:#8b98a5;--line:#222a31;--hot:#ff4d4f;--ok:#3fb950}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
 header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;
   gap:16px;align-items:center;flex-wrap:wrap}
 h1{font-size:15px;margin:0;font-weight:600}
 .wrap{display:flex;gap:16px;padding:16px;flex-wrap:wrap}
 .view{flex:1 1 640px;min-width:320px}
 .view img{width:100%;border:1px solid var(--line);border-radius:6px;display:block}
 .side{flex:0 1 300px;min-width:260px}
 table{width:100%;border-collapse:collapse} td{padding:3px 0}
 td.k{color:var(--mut)} td.v{text-align:right;font-variant-numeric:tabular-nums}
 .big{font-size:22px;font-weight:600}
 .alert{color:var(--hot)} .clean{color:var(--ok)}
 button{background:#1f6feb;color:#fff;border:0;padding:8px 14px;border-radius:6px;
   cursor:pointer;font:inherit} button:hover{background:#388bfd}
 button.warn{background:#8b3a3a} button.warn:hover{background:#a94442}
 ul{list-style:none;margin:8px 0 0;padding:0;max-height:280px;overflow:auto}
 li{border-top:1px solid var(--line);padding:5px 0}
 li a{color:var(--fg);text-decoration:none} li a:hover{color:#58a6ff}
 .legend span{display:inline-block;margin-right:12px;color:var(--mut)}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
</style>
<header>
  <h1>Giám sát rác — trực tiếp</h1>
  <span id="state" class="big clean">—</span>
  <span class="legend">
    <span><i class="sw" style="background:#ff4d4f"></i>ô nóng</span>
    <span><i class="sw" style="background:#ffa500"></i>vừa đổi</span>
    <span><i class="sw" style="background:#00a0ff"></i>bị che</span>
    <span><i class="sw" style="background:#888"></i>đã mute</span>
  </span>
</header>
<div class="wrap">
  <div class="view"><img src="/stream.mjpg" alt="live"></div>
  <div class="side">
    <table id="stats"></table>
    <p style="margin:14px 0 4px">
      <button class="warn" onclick="reseed()">CHỐT LẠI NỀN</button>
    </p>
    <p style="color:var(--mut);margin:0">Dùng khi vừa dời đồ đạc hợp lệ
      (thùng rác, ghế…). Không chờ mặt nạ nhiễu.</p>
    <h3 style="font-size:13px;margin:18px 0 0">Cảnh báo gần nhất</h3>
    <ul id="alerts"></ul>
  </div>
</div>
<script>
const F=[["t","Thời gian chạy"],["n_cells","Tổng ô"],["n_occluded","Ô bị che"],
 ["n_changed","Ô vừa đổi"],["n_hot","Ô nóng"],["n_fresh","Ô nóng MỚI"],
 ["n_muted","Ô đã mute"],["mask","Mặt nạ chín"],["scans","Lượt quét"],
 ["alerts","Cảnh báo"],["ms","ms/lượt"],["fails","Lần mất kết nối"]];
async function tick(){
 try{
  const s=await (await fetch("/stats")).json();
  document.getElementById("stats").innerHTML=F.map(([k,l])=>
    `<tr><td class="k">${l}</td><td class="v">${s[k]??"—"}</td></tr>`).join("");
  const e=document.getElementById("state");
  e.textContent=s.alerting?"CẢNH BÁO":(s.dirty?"đang bẩn":"sạch");
  e.className="big "+(s.dirty?"alert":"clean");
  document.getElementById("alerts").innerHTML=(s.recent||[]).map(a=>
    `<li><a href="/alert/${a.id}.jpg" target="_blank">${a.time} — ${a.hot} ô`
    +`${a.fresh?` (${a.fresh} mới)`:""}</a></li>`).join("")||"<li>chưa có</li>";
 }catch(err){}
}
async function reseed(){
 await fetch("/reset",{method:"POST"});
 document.getElementById("state").textContent="đã chốt lại nền";
}
setInterval(tick,1000); tick();
</script>
"""


class Monitor:
    def __init__(self, args, cfg, poly):
        self.args, self.cfg, self.poly = args, cfg, poly
        self.lock = threading.Lock()
        self.jpg: bytes | None = None
        self.stats: dict = {"scans": 0, "alerts": 0, "fails": 0}
        self.alerts: deque = deque(maxlen=12)
        self.alert_imgs: dict = {}
        self.reset_req = threading.Event()
        self.det: ZoneTrashDetector | None = None
        self.t0 = time.time()

    # ---- vòng quét chạy nền ----
    def run(self):
        interval = float(self.cfg.get("scan", {}).get("interval_s", 30))
        scorer = build_scorer(self.cfg.get("scorer", {"kind": "constant", "value": 0.0}))
        boxes_of = YoloBoxes(self.args.yolo) if self.args.yolo else (lambda _f: [])
        self.det = ZoneTrashDetector(self.cfg, scorer,
                                     camera_id=self.args.camera_id,
                                     zone_id=self.args.zone_id,
                                     state_dir=os.path.join(self.args.out, "state"))
        os.makedirs(os.path.join(self.args.out, "state"), exist_ok=True)
        n_alert = 0
        for t, frame in iter_frames(self.args.source, interval):
            if self.reset_req.is_set():
                self.det.reset_background()
                self.reset_req.clear()
            boxes = boxes_of(frame)
            res = self.det.scan(frame, self.poly, boxes, now=t)
            vis = self.annotate(frame, res, boxes)
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if res.alert:
                n_alert += 1
                aid = str(n_alert)
                ok2, ab = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok2:
                    self.alert_imgs[aid] = ab.tobytes()
                self.alerts.appendleft({"id": aid, "hot": len(res.hot),
                                        "fresh": res.n_fresh_hot,
                                        "time": time.strftime("%H:%M:%S")})
            with self.lock:
                if ok:
                    self.jpg = buf.tobytes()
                self.stats = {
                    "t": f"{(time.time()-self.t0)/3600:.2f} h",
                    "n_cells": res.n_cells, "n_occluded": res.n_occluded,
                    "n_changed": res.n_changed, "n_hot": len(res.hot),
                    "n_fresh": res.n_fresh_hot, "n_muted": res.n_muted_hit,
                    "mask": f"{res.mask_progress*100:.0f}%",
                    "scans": self.stats.get("scans", 0) + 1,
                    "alerts": n_alert, "ms": f"{res.ms:.0f}",
                    "fails": self.stats.get("fails", 0),
                    "dirty": bool(res.dirty), "alerting": bool(res.alert),
                    "recent": list(self.alerts),
                }
        with self.lock:
            self.stats["stopped"] = True

    def annotate(self, frame, res, boxes):
        vis = frame.copy()
        h, w = frame.shape[:2]
        cv2.polylines(vis, [np.array(poly_to_px(self.poly, w, h), np.int32)],
                      True, (0, 220, 255), 2)
        det = self.det
        muted = set(det.clutter.muted()) if (det and det.clutter) else set()
        hot = {c.id for c in res.hot}
        for q in (det.grid.cells if det and det.grid else []):
            v = res.scores.get(q.id, 0.0)
            if q.id in hot:
                col, th = (0, 0, 255), 2
            elif q.id in muted:
                col, th = (136, 136, 136), 1
            elif v > 0:
                col, th = (0, 165, 255), 1
            else:
                col, th = (70, 70, 70), 1
            cv2.rectangle(vis, (q.x1, q.y1), (q.x2, q.y2), col, th)
        for b in boxes:
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (255, 160, 0), 2)
        tag = ("ALERT" if res.alert else
               ("GLOBAL-RESEED" if res.global_change else
                ("dirty" if res.dirty else "clean")))
        col = (0, 0, 255) if res.alert else ((0, 165, 255) if res.dirty else (0, 200, 0))
        cv2.rectangle(vis, (0, 0), (w, 26), (0, 0, 0), -1)
        cv2.putText(vis, f"{tag}  cells={res.n_cells} occl={res.n_occluded} "
                         f"chg={res.n_changed} hot={len(res.hot)} "
                         f"fresh={res.n_fresh_hot} ({res.ms:.0f}ms)",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, col, 1)
        return vis


def make_handler(mon: Monitor):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path == "/reset":
                mon.reset_req.set()
                self._send(200, "application/json", b'{"ok":true}')
            else:
                self._send(404, "text/plain", b"no")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif self.path == "/stats":
                with mon.lock:
                    body = json.dumps(mon.stats).encode()
                self._send(200, "application/json", body)
            elif self.path.startswith("/alert/"):
                aid = self.path.split("/")[-1].replace(".jpg", "")
                img = mon.alert_imgs.get(aid)
                self._send(200 if img else 404, "image/jpeg", img or b"")
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                try:
                    last = None
                    while True:
                        with mon.lock:
                            j = mon.jpg
                        if j is not None and j is not last:
                            last = j
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                             b"Content-Length: "
                                             + str(len(j)).encode() + b"\r\n\r\n"
                                             + j + b"\r\n")
                        time.sleep(0.25)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(404, "text/plain", b"no")
    return H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="runs/live")
    ap.add_argument("--yolo")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--camera-id", default="cam0")
    ap.add_argument("--zone-id", default="zone0")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.zone, encoding="utf-8") as f:
        poly = json.load(f)["points"]
    os.makedirs(args.out, exist_ok=True)

    mon = Monitor(args, cfg, poly)
    threading.Thread(target=mon.run, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(mon))
    print(f"Mở trình duyệt:  http://localhost:{args.port}")
    print(f"  nguồn  {args.source}")
    print(f"  nhịp   {cfg.get('scan', {}).get('interval_s', 30)}s   "
          f"chế độ {cfg.get('decide', {}).get('mode', 'classifier')}")
    print("  Ctrl+C để dừng")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\ndừng")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
