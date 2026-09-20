"""Mission state machine: power -> wifi -> SDK gate -> launch -> patrol/search ->
photo -> flip -> return-to-home -> land.

Position is tracked by dead reckoning from the commands we send (Tello has no
indoor positioning), so return-to-home is approximate; keep legs short.
"""
import json
import logging
import math
import os
import signal
import time
from typing import Dict, Optional

from . import power, wifi
from . import log as logmod
from .detector import Detection, make_detector, OVERLAY_BOTTOM_NAME
from .tello import FakeTello, Tello, TelloError
from .video import FakeVideoStream, VideoStream

log = logging.getLogger("mission")


class Pose:
    """x forward / y left of the launch heading, in cm; yaw in degrees, cw positive."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

    def move(self, direction: str, cm: int):
        # heading in math convention: cw yaw => negative angle
        a = math.radians(-self.yaw)
        if direction == "forward":
            dx, dy = cm, 0
        elif direction == "back":
            dx, dy = -cm, 0
        elif direction == "left":
            dx, dy = 0, cm
        elif direction == "right":
            dx, dy = 0, -cm
        else:
            return
        self.x += dx * math.cos(a) - dy * math.sin(a)
        self.y += dx * math.sin(a) + dy * math.cos(a)

    def rotate(self, direction: str, deg: int):
        self.yaw = (self.yaw + (deg if direction == "cw" else -deg)) % 360

    def home_vector(self):
        """(distance_cm, cw_turn_deg) needed to face and reach the launch point."""
        dist = math.hypot(self.x, self.y)
        if dist < 1:
            return 0.0, 0.0
        bearing = math.degrees(math.atan2(-self.y, -self.x))  # math (ccw) angle to home
        target_yaw = (-bearing) % 360                          # into cw convention
        turn = (target_yaw - self.yaw + 540) % 360 - 180        # shortest signed cw turn
        return dist, turn

    def as_dict(self):
        return {"x_cm": round(self.x), "y_cm": round(self.y), "yaw_deg": round(self.yaw)}


class Mission:
    def __init__(self, cfg: Dict, dry_run: bool = False, base_dir: str = "."):
        self.cfg = cfg
        self.dry = dry_run
        self.base = base_dir
        m = cfg["mission"]
        self.photos_dir = os.path.join(base_dir, m["photos_dir"])
        os.makedirs(self.photos_dir, exist_ok=True)
        logs_dir = os.path.join(base_dir, m["logs_dir"])
        # fixed (not timestamped) paths: log.py's stdout tee keeps overlay_top overwritten
        # with the latest console line, JevDetector keeps overlay_bottom overwritten with
        # its latest decision -- the recording burns in whatever each currently holds.
        self.overlay_top = os.path.join(logs_dir, logmod.OVERLAY_TOP_NAME)
        self.overlay_bottom = os.path.join(logs_dir, OVERLAY_BOTTOM_NAME)
        self.pose = Pose()
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.report: Dict = {"dry_run": dry_run, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "events": [], "person": None, "photo": None, "outcome": "not started"}
        self.t_takeoff: Optional[float] = None
        self.target_height_cm: Optional[int] = None
        self.drone: Optional[Tello] = None
        self.video: Optional[VideoStream] = None
        self.prev_conn: Optional[str] = None
        self.airborne = False

    # ------------------------------------------------------------------ utils
    def ev(self, msg: str, **kw):
        log.info(msg)
        self.report["events"].append({"t": round(time.time() - (self.t_takeoff or time.time()), 1),
                                      "msg": msg, **kw})

    def check_limits(self):
        d = self.cfg["sdk"]
        if self.t_takeoff and time.time() - self.t_takeoff > self.cfg["mission"]["max_flight_s"]:
            raise TelloError("max flight time exceeded")
        bat = self.drone.battery()
        if 0 <= bat < d["min_battery_takeoff"] - 5:
            raise TelloError("battery critical (%d%%)" % bat)

    def fly(self, cmd: str, settle: float = 0.8, retries: int = 2) -> bool:
        """Best-effort movement: retries a command that errors, then skips it (mission
        keeps going) rather than aborting the whole flight over one bad command. Pose is
        only updated on success, so dead reckoning doesn't drift from a move that never
        happened. check_limits() (battery-critical / max-flight-time) is NOT retried or
        swallowed here -- those still abort immediately, same as before."""
        self.check_limits()
        for attempt in range(retries + 1):
            try:
                self.drone.flight(cmd, settle=settle)
                break
            except TelloError as exc:
                if attempt < retries:
                    self.ev("retry '%s' after error: %s" % (cmd, exc))
                    time.sleep(0.5)
                    continue
                self.ev("SKIP '%s' after %d failed attempts: %s" % (cmd, retries + 1, exc))
                return False
        parts = cmd.split()
        if len(parts) == 2 and parts[0] in ("forward", "back", "left", "right"):
            self.pose.move(parts[0], int(parts[1]))
        elif len(parts) == 2 and parts[0] in ("cw", "ccw"):
            self.pose.rotate(parts[0], int(parts[1]))
        return True

    # ---------------------------------------------------------------- phases
    def phase_power(self, skip: bool):
        if skip:
            self.ev("power: skipped (assumed on)")
            return
        if self.dry:
            self.ev("power: on (dry-run)")
            return
        if not power.power_on(self.cfg["power"]):
            raise RuntimeError("could not power on Tello plug")
        self.ev("power: on, boot wait done")

    def phase_wifi(self, skip: bool):
        if skip or self.dry:
            self.ev("wifi: skipped (dry-run or --skip-wifi)")
            return
        w = self.cfg["wifi"]
        if w.get("restore_previous_connection"):
            self.prev_conn = wifi.current_connection(w["iface"])
        ssid = wifi.connect_tello(self.cfg["drone"], w)
        if not ssid:
            raise RuntimeError("no TELLO SSID visible / connect failed -> power reset needed")
        self.ev("wifi: connected to %s" % ssid, ssid=ssid)

    def phase_sdk(self):
        d = self.cfg["drone"]
        cls = FakeTello if self.dry else Tello
        self.drone = cls(d["ip"], d["cmd_port"], d["state_port"], d["local_cmd_port"])
        s = self.cfg["sdk"]
        if not self.drone.wait_ready(s["probe_count"], s["probe_interval_s"]):
            raise RuntimeError("SDK not ready (command never acked) -> power reset needed")
        time.sleep(1.0)
        bat = self.drone.battery()
        self.ev("sdk: ready, battery %d%%" % bat, battery=bat)
        if 0 <= bat < s["min_battery_takeoff"]:
            raise RuntimeError("battery %d%% below takeoff minimum %d%%" % (bat, s["min_battery_takeoff"]))

    def _streamon(self, attempts: int = 3) -> bool:
        for i in range(attempts):
            if self.drone.send("streamon") == "ok":
                return True
            time.sleep(0.5)
        return False

    def phase_launch(self):
        d, m = self.cfg["drone"], self.cfg["mission"]
        cls = FakeVideoStream if self.dry else VideoStream
        record_path = None
        if not self.dry:
            rec_dir = os.path.join(self.base, m.get("recordings_dir", "recordings"))
            record_path = os.path.join(rec_dir, "flight_%s.mp4" % self.run_id)
            self.report["recording"] = record_path
        self.video = cls(d["video_port"], os.path.join(self.base, "logs", "stream"), record_path=record_path,
                         overlay_top=self.overlay_top, overlay_bottom=self.overlay_bottom,
                         overlay_font=m.get("overlay_font"))
        self.video.start()
        # the listener must be bound before streamon: the Tello emits its H.264 SPS/PPS
        # parameter sets once at stream start, and a listener that isn't ready yet misses
        # them permanently (every later frame then decodes as corrupt).
        if not self._streamon():
            raise RuntimeError("drone never acked streamon -> power reset needed")
        if not self.video.wait_first_frame(8):
            # a redundant streamon while already streaming does not re-emit SPS/PPS;
            # an off/on toggle does, and recovers a missed first attempt.
            self.drone.send("streamoff")
            time.sleep(0.5)
            if not self._streamon():
                raise RuntimeError("drone never acked streamon -> power reset needed")
            if not self.video.wait_first_frame(8):
                self.drone.send("streamoff")
                raise RuntimeError("video stream produced no frames; aborting before takeoff")
        self.ev("launch: takeoff")
        self.drone.flight("takeoff", settle=2.5)
        self.airborne = True
        self.t_takeoff = time.time()
        self.fly("stop")
        if m["takeoff_climb_cm"] >= 20:
            self.fly("up %d" % m["takeoff_climb_cm"], settle=1.2)
        h = self.drone.height_cm()
        self.target_height_cm = h if h > 0 else None
        self.ev("launch: hovering, telemetry %s" % self.drone.telemetry())

    def maintain_height(self):
        """Correct drift from the hover height set right after takeoff. The Tello holds
        altitude on its own via its barometer, but that can still drift over a long search;
        SDK up/down need >=20cm to be a valid command, so smaller drift is left alone."""
        if self.target_height_cm is None:
            return
        h = self.drone.height_cm()
        if h <= 0:
            return  # no reliable telemetry yet
        drift = self.target_height_cm - h
        if drift >= 20:
            self.fly("up %d" % min(drift, 500), settle=0.6)
        elif drift <= -20:
            self.fly("down %d" % min(-drift, 500), settle=0.6)

    def look(self, detector) -> Optional[Detection]:
        m, j = self.cfg["mission"], self.cfg["jev"]
        self.maintain_height()
        time.sleep(m["frame_settle_s"])
        dets = []
        for _ in range(m["frames_per_heading"]):
            frame = self.video.latest_frame()
            if not frame:
                time.sleep(0.5)
                continue
            try:
                det = detector.detect(frame)
            except Exception as exc:
                log.warning("detector error: %s", exc)
                continue
            dets.append((det, frame))
            time.sleep(m.get("frame_gap_s", 0.4))
        if not dets:
            return None
        # a single lucky/unlucky frame can otherwise swing a mean past threshold (a door
        # once hit 0.84/0.81/0.03 -- mean 0.56, comfortably "found" -- despite jev itself
        # scoring the exact same frame 0.03 in isolation): every frame in the burst must
        # individually clear min_confidence, not just the average.
        mean_p = sum(d.confidence for d, _ in dets) / len(dets)
        min_p = min(d.confidence for d, _ in dets)
        best, frame = max(dets, key=lambda df: df[0].confidence)
        log.info("jev burst: %s -> mean %.2f min %.2f", " ".join("%.2f" % d.confidence for d, _ in dets), mean_p, min_p)
        if len(dets) >= m["frames_per_heading"] and min_p >= j["min_confidence"] and best.bbox is not None:
            self._last_frame = frame
            best.confidence = round(mean_p, 3)
            return best
        return None

    def phase_search(self, detector) -> Optional[Detection]:
        """jev decides where and when to navigate, one stop at a time -- no fixed route.
        At each stop: look for a person; if none, ask jev whether to rotate to a new
        heading here, advance forward into open space, or treat this direction as a dead
        end (-> rotate away from it). The only hard limits are a distance-from-launch
        ceiling (dead reckoning drift makes return-to-home less reliable the further it
        roams) and a search time budget that leaves room to still fly home and land."""
        m = self.cfg["mission"]
        max_dist = m.get("max_distance_from_launch_cm", 500)
        step_cm = m.get("nav_forward_step_cm", 100)
        scan_deg = m["scan_step_deg"]
        search_deadline = (self.t_takeoff + m["max_flight_s"] - m.get("return_buffer_s", 60)) if self.t_takeoff else None
        while True:
            if search_deadline and time.time() > search_deadline:
                self.ev("SEARCH: giving up (search time budget exhausted)")
                return None
            det = self.look(detector)
            if det:
                self.ev("SEARCH: person found, heading %d° (conf %.2f, %s)" % (
                    self.pose.yaw, det.confidence, det.backend),
                    pose=self.pose.as_dict(), detection=det.to_dict())
                return det
            dist_home, _ = self.pose.home_vector()
            action = detector.decide_navigation(dist_home, max_dist)
            self.ev("SEARCH: jev says '%s' (%.0fcm from launch)" % (action, dist_home), pose=self.pose.as_dict())
            if action == "advance" and dist_home + step_cm <= max_dist:
                self.fly("forward %d" % step_cm, settle=1.0)
            else:
                self.fly("cw %d" % scan_deg, settle=0.6)

    def phase_photo_and_flip(self, det: Detection):
        m = self.cfg["mission"]
        if m["center_on_person"] and det.center_x() is not None:
            off = det.center_x() - 0.5
            if abs(off) > 0.15:
                turn = int(min(45, max(10, abs(off) * 80)))
                self.fly(("cw %d" if off > 0 else "ccw %d") % turn, settle=1.0)
                self.ev("centered on person (%s %d°)" % ("cw" if off > 0 else "ccw", turn))
        time.sleep(0.8)
        frame = self.video.latest_frame() or getattr(self, "_last_frame", None)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        photo = os.path.join(self.photos_dir, "person_%s.jpg" % stamp)
        if frame:
            with open(photo, "wb") as f:
                f.write(frame)
            with open(photo[:-4] + ".json", "w") as f:
                json.dump({"detection": det.to_dict(), "pose": self.pose.as_dict(),
                           "telemetry": self.drone.telemetry(), "time": stamp}, f, indent=2)
            self.report["photo"] = photo
            self.ev("PHOTO saved %s" % photo)
        else:
            self.ev("PHOTO failed: no frame available")
        self.report["person"] = {"detection": det.to_dict(), "pose": self.pose.as_dict()}

        if m["do_flip"]:
            bat = self.drone.battery()
            if bat >= 0 and bat < self.cfg["sdk"]["min_battery_flip"]:
                self.ev("FLIP skipped: battery %d%% < %d%%" % (bat, self.cfg["sdk"]["min_battery_flip"]))
            else:
                self.fly("up 30", settle=1.0)  # headroom, then celebrate
                if self.fly("flip %s" % m["flip_direction"], settle=2.5):
                    self.ev("FLIP done (%s)" % m["flip_direction"])
                else:
                    self.ev("FLIP refused by drone")
                self.fly("down 30", settle=1.0)

    def phase_return(self):
        dist, turn = self.pose.home_vector()
        self.ev("RTH: %.0fcm away, turning %+.0f°" % (dist, turn), pose=self.pose.as_dict())
        if abs(turn) >= 5:
            self.fly(("cw %d" if turn > 0 else "ccw %d") % abs(round(turn)), settle=1.0)
        remaining = int(round(dist))
        while remaining >= 20:
            step = min(remaining, 300)
            self.fly("forward %d" % step, settle=1.0)
            remaining -= step
        # face original heading
        back = (0 - self.pose.yaw + 540) % 360 - 180
        if abs(back) >= 5:
            self.fly(("cw %d" if back > 0 else "ccw %d") % abs(round(back)), settle=1.0)
        self.ev("RTH: at home (dead-reckoned) %s" % self.pose.as_dict())

    def phase_land(self):
        if self.airborne:
            self.ev("LAND")
            self.drone.send("land", timeout=10.0)
            self.airborne = False
            time.sleep(2)
        self.drone.send("streamoff")

    # ------------------------------------------------------------------ run
    def run(self, skip_power=False, skip_wifi=False, search_only=False) -> Dict:
        detector = None
        # SIGTERM (e.g. a killed/stopped process) is not caught by Python by default and
        # would skip straight past the except/finally below -- leaving the drone airborne
        # with no landing attempt at all. Route it through the same safe shutdown as Ctrl-C.
        def _on_sigterm(signum, frame):
            raise KeyboardInterrupt("SIGTERM")
        old_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
        try:
            self.phase_power(skip_power)
            self.phase_wifi(skip_wifi)
            self.phase_sdk()
            decisions_path = os.path.join(self.base, self.cfg["mission"]["logs_dir"],
                                          "jev_decisions_%s.jsonl" % self.run_id)
            detector = make_detector(self.cfg["jev"], dry_run=self.dry, decisions_path=decisions_path,
                                     overlay_path=self.overlay_bottom)
            self.report["detector"] = detector.name
            if not self.dry:
                self.report["jev_decisions_log"] = decisions_path
            self.phase_launch()
            det = self.phase_search(detector)
            if det and not search_only:
                self.phase_photo_and_flip(det)
            elif not det:
                self.ev("SEARCH: no person found on the 1st floor")
            self.phase_return()
            self.phase_land()
            self.report["outcome"] = "person found" if det else "no person found"
        except (TelloError, RuntimeError) as exc:
            self.ev("ABORT: %s" % exc)
            self.report["outcome"] = "aborted: %s" % exc
            if self.drone and self.airborne:
                self.drone.emergency_land()
        except KeyboardInterrupt as exc:
            self.ev("ABORT: %s" % ("terminated" if str(exc) == "SIGTERM" else "operator interrupt"))
            self.report["outcome"] = "aborted by operator"
            if self.drone and self.airborne:
                self.drone.emergency_land()
        finally:
            signal.signal(signal.SIGTERM, old_sigterm)
            if self.video:
                self.video.stop()
            if self.drone:
                self.drone.close()
            if self.prev_conn and not self.dry:
                wifi.restore(self.cfg["wifi"]["iface"], self.prev_conn)
            if self.cfg["power"].get("power_off_after_mission") and not self.dry:
                power.power_off(self.cfg["power"])
        self.report["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.report["final_pose"] = self.pose.as_dict()
        self.report["flight_s"] = round(time.time() - self.t_takeoff, 1) if self.t_takeoff else 0
        path = os.path.join(self.base, self.cfg["mission"]["logs_dir"], "report_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
        with open(path, "w") as f:
            json.dump(self.report, f, indent=2)
        log.info("report -> %s", path)
        return self.report
