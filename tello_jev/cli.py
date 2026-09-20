import argparse
import json
import os
import sys

from . import log as logmod
from . import power, wifi
from .mission import Mission
from .tello import Tello

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_secrets(path=os.path.join(BASE, "secrets.json")):
    """secrets.json (gitignored; see secrets.example.json) holds API keys/tokens kept out
    of config.json so the rest of the project is safe to commit/share as-is."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for k, v in json.load(f).items():
            os.environ.setdefault(k, str(v))


load_secrets()


def load_cfg(path):
    with open(path) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tello-jev", description="Tello person-search mission using the jev vision model")
    ap.add_argument("-c", "--config", default=os.path.join(BASE, "config.json"))
    def frontend_flags(parser, suppress=False):
        d = {"default": argparse.SUPPRESS} if suppress else {}
        fe = parser.add_mutually_exclusive_group()
        fe.add_argument("--opencv", action="store_true", help="frontend: OpenCV HOG people detector feeds jev", **d)
        fe.add_argument("--vision", action="store_true", help="frontend: fast vision model (jev.vision_model) feeds jev", **d)
        fe.add_argument("--frontend", choices=["features", "opencv", "vision"], help="explicit frontend choice", **d)
        parser.add_argument("--vision-model", help="override jev.vision_model (default openai/gpt-4o-mini)", **d)

    frontend_flags(ap)
    common = argparse.ArgumentParser(add_help=False)
    frontend_flags(common, suppress=True)  # same flags accepted after the subcommand
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", parents=[common], help="full mission: power, wifi, launch, search, photo, flip, return, land")
    r.add_argument("--dry-run", action="store_true", help="simulate drone/video/detector, touch no hardware")
    r.add_argument("--skip-power", action="store_true", help="drone already powered on")
    r.add_argument("--skip-wifi", action="store_true", help="host already on the TELLO network")
    r.add_argument("--no-flip", action="store_true")
    r.add_argument("--search-only", action="store_true", help="find person, but no photo/flip")

    sub.add_parser("power-on", parents=[common])
    sub.add_parser("power-off", parents=[common])
    sub.add_parser("wifi", parents=[common], help="scan + connect to the Tello AP")
    p = sub.add_parser("probe", parents=[common], help="SDK readiness probe + battery/telemetry")
    p.add_argument("--probes", type=int, default=6)
    sub.add_parser("land", parents=[common], help="send land (use if a mission left the drone airborne)")
    sub.add_parser("detector-check", parents=[common], help="verify jev is reachable through the gateway")
    cal = sub.add_parser("calibrate", parents=[common], help="run jev on folders of images to tune jev.min_confidence")
    cal.add_argument("--positives", help="folder of frames WITH a person")
    cal.add_argument("--negatives", help="folder of frames WITHOUT a person")

    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    if a.opencv:
        cfg["jev"]["frontend"] = "opencv"
    elif a.vision:
        cfg["jev"]["frontend"] = "vision"
    elif a.frontend:
        cfg["jev"]["frontend"] = a.frontend
    if a.vision_model:
        cfg["jev"]["vision_model"] = a.vision_model
    log = logmod.setup(os.path.join(BASE, cfg["mission"]["logs_dir"]))

    if a.cmd == "run":
        if a.no_flip:
            cfg["mission"]["do_flip"] = False
        rep = Mission(cfg, dry_run=a.dry_run, base_dir=BASE).run(
            skip_power=a.skip_power, skip_wifi=a.skip_wifi, search_only=a.search_only)
        print("\n=== MISSION REPORT ===")
        print("outcome : %s" % rep["outcome"])
        print("detector: %s" % rep.get("detector"))
        print("photo   : %s" % rep.get("photo"))
        print("person  : %s" % json.dumps(rep.get("person")))
        print("flight  : %ss, final pose %s" % (rep["flight_s"], rep["final_pose"]))
        return 0 if rep["outcome"].startswith(("person", "no person")) else 1

    if a.cmd == "power-on":
        return 0 if power.power_on(cfg["power"]) else 1
    if a.cmd == "power-off":
        return 0 if power.power_off(cfg["power"]) else 1
    if a.cmd == "wifi":
        ssid = wifi.connect_tello(cfg["drone"], cfg["wifi"])
        print("connected:", ssid)
        return 0 if ssid else 2
    if a.cmd in ("probe", "land"):
        d = cfg["drone"]
        t = Tello(d["ip"], d["cmd_port"], d["state_port"], d["local_cmd_port"])
        try:
            ok = t.wait_ready(getattr(a, "probes", 4), 2.0)
            if not ok:
                print("SDK not ready")
                return 3
            if a.cmd == "land":
                t.send("land", timeout=10)
            else:
                import time; time.sleep(1.5)
                print("battery: %s%%  telemetry: %s" % (t.battery(), t.telemetry()))
        finally:
            t.close()
        return 0
    if a.cmd == "calibrate":
        import glob
        from .detector import make_detector
        det = make_detector(cfg["jev"])
        for label, folder in (("POS", a.positives), ("NEG", a.negatives)):
            if not folder:
                continue
            probs = []
            for f in sorted(glob.glob(os.path.join(folder, "*.jp*g"))):
                det.frames.prev_gray = None
                try:
                    r = det.detect(open(f, "rb").read())
                except Exception as exc:
                    print("%s %-40s ERR %s" % (label, os.path.basename(f), exc))
                    continue
                probs.append(r.confidence)
                print("%s %-40s p=%.2f  %s" % (label, os.path.basename(f), r.confidence, r.description))
            if probs:
                print("%s mean=%.2f min=%.2f max=%.2f n=%d" % (label, sum(probs) / len(probs), min(probs), max(probs), len(probs)))
        print("current jev.min_confidence = %s" % cfg["jev"]["min_confidence"])
        return 0
    if a.cmd == "detector-check":
        from .detector import make_detector
        try:
            make_detector(cfg["jev"])
            print("jev OK: %s via %s, frontend=%s" % (cfg["jev"]["model"], cfg["jev"]["base_url"], cfg["jev"].get("frontend", "features")))
            return 0
        except RuntimeError as exc:
            print("jev NOT available:", exc)
            return 4


if __name__ == "__main__":
    sys.exit(main())
