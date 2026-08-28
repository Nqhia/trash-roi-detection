"""Tai not anh TACO/UAVVaste con thieu (annotations co san URL Flickr).

    python3 tools/fetch_taco.py --which taco

Vi sao can: dem duoc 861/1500 anh TACO va 546/772 anh UAVVaste KHONG co tren dia.
Model dang train tren 639 anh TACO trong khi bo co 1500 — va TACO chinh la bo co
co vat sat dai CCTV nhat (canh trung vi 79px @1280, chi 27% vat duoi 32px), nguoc
han RoLID dang ap dao tap train (90% vat duoi 32px).
"""
from __future__ import annotations
import argparse, json, os, sys, threading, queue, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SETS = {
    "taco":     (f"{ROOT}/data/taco/annotations.json",     f"{ROOT}/data/taco/images",     lambda fn: fn.replace("/", "_")),
    "uavvaste": (f"{ROOT}/data/uavvaste/annotations.json", f"{ROOT}/data/uavvaste/images", lambda fn: fn.split("/")[-1]),
}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="taco", choices=list(SETS))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    annf, imgd, ren = SETS[args.which]
    os.makedirs(imgd, exist_ok=True)
    a = json.load(open(annf))
    todo = []
    for im in a["images"]:
        p = os.path.join(imgd, ren(im["file_name"]))
        if os.path.exists(p):
            continue
        url = im.get("flickr_640_url") or im.get("flickr_url") or im.get("coco_url")
        if url:
            todo.append((url, p))
    print(f"{args.which}: {len(a['images'])} anh, thieu {len(todo)} co URL", flush=True)
    q = queue.Queue()
    for t in todo:
        q.put(t)
    ok = [0]; bad = [0]
    lock = threading.Lock()

    def work():
        while True:
            try:
                url, p = q.get_nowait()
            except queue.Empty:
                return
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=45) as r, open(p + ".part", "wb") as f:
                    f.write(r.read())
                os.replace(p + ".part", p)
                with lock:
                    ok[0] += 1
                    if ok[0] % 50 == 0:
                        print(f"  tai {ok[0]}/{len(todo)} · loi {bad[0]}", flush=True)
            except Exception:
                try: os.remove(p + ".part")
                except OSError: pass
                with lock:
                    bad[0] += 1
            finally:
                q.task_done()

    ts = [threading.Thread(target=work, daemon=True) for _ in range(args.workers)]
    for t in ts: t.start()
    for t in ts: t.join()
    print(f"xong: tai duoc {ok[0]}, loi {bad[0]}")
    n = sum(1 for im in a["images"] if os.path.exists(os.path.join(imgd, ren(im["file_name"]))))
    print(f"gio co {n}/{len(a['images'])} anh")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
