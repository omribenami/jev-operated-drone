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


class MissionStop(Exception):
    """A deliberate end of flight, as opposed to an error.

    Landing policy: the only things that ever put this drone on the ground on purpose are
    (1) the person has been found, photographed and flown home, (2) the battery has reached
    the land floor, and (3) the operator interrupting. Command errors explicitly do NOT --
    see fly() and Mission.run() -- because on this airframe a failed command is far more
    often a transient IMU/link hiccup than a reason to end an otherwise healthy flight."""


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
        self.launch_height_cm: Optional[int] = None
        self.drone: Optional[Tello] = None
        self.video: Optional[VideoStream] = None
        self.prev_conn: Optional[str] = None
        self.airborne = False
        self.consecutive_command_failures = 0
        self.powered_on_by_mission = False
        self.low_battery_reads = 0
        self.visited = [(0.0, 0.0)]  # launch point; used to prefer headings into new space

    # ------------------------------------------------------------------ utils
    def ev(self, msg: str, **kw):
        log.info(msg)
        self.report["events"].append({"t": round(time.time() - (self.t_takeoff or time.time()), 1),
                                      "msg": msg, **kw})

    def check_limits(self):
        """The only check that ends a flight by itself. Battery is the single stop
        condition; flight time is a backstop used *only* when the battery can't be read at
        all, because otherwise nothing would ever bring the drone down.

        battery% on this pack is a noisy estimate that sags hard under load (5+ points on
        takeoff current alone at a full charge), so one dip below the floor is not enough --
        two consecutive low reads are required before landing, and any healthy read in
        between resets the count."""
        floor = self.cfg["sdk"].get("min_battery_land", self.cfg["sdk"]["min_battery_takeoff"])
        bat = self.drone.battery()
        if bat < 0:
            if self.t_takeoff and time.time() - self.t_takeoff > self.cfg["mission"]["max_flight_s"]:
                raise MissionStop("battery telemetry unreadable and max flight time exceeded")
            return
        if bat < floor:
            self.low_battery_reads += 1
            self.ev("battery %d%% below land floor %d%% (read %d/2)" % (bat, floor, self.low_battery_reads))
            if self.low_battery_reads >= 2:
                raise MissionStop("battery %d%% at/below land floor %d%%" % (bat, floor))
        else:
            self.low_battery_reads = 0

    def fly(self, cmd: str, settle: float = 0.8, retries: int = 2) -> bool:
        """Best-effort movement: retries a command that errors, then skips it and returns
        False. Pose is only updated on success, so dead reckoning doesn't drift from a move
        that never happened.

        A command error NEVER ends the flight. 'No valid imu', 'Motor stop' and bare
        no-response acks are routine on this airframe and usually clear on their own within
        a few seconds; the drone is hovering perfectly well while they happen. The only
        thing that can end a flight from in here is check_limits() raising MissionStop on
        the battery floor. (There used to be a circuit breaker that aborted after N
        consecutive failures; it is gone -- if the link really is dead, the abort couldn't
        have sent 'land' either, so it bought nothing and cost real flights. The count is
        still tracked and logged so a dead link is obvious in the report.)"""
        self.check_limits()
        for attempt in range(retries + 1):
            try:
                self.drone.flight(cmd, settle=settle)
                break
            except TelloError as exc:
                if attempt < retries:
                    self.ev("retry '%s' after error: %s" % (cmd, exc))
                    time.sleep(0.4)
                    continue
                self.consecutive_command_failures += 1
                self.ev("SKIP '%s' after %d failed attempts (%d in a row now, still flying): %s"
                        % (cmd, retries + 1, self.consecutive_command_failures, exc))
                return False
        self.consecutive_command_failures = 0
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
        self.powered_on_by_mission = True
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
        # NOT "stop": this firmware doesn't recognize it ("unknown command: stop") on every
        # single flight tonight -- the Tello already hovers in place after takeoff on its own.
        if m["takeoff_climb_cm"] >= 20:
            self.fly("up %d" % m["takeoff_climb_cm"], settle=1.2)
        h = self.drone.height_cm()
        self.target_height_cm = h if h > 0 else None
        self.launch_height_cm = self.target_height_cm
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

    # ------------------------------------------------------------- perception
    def grab_frame(self, tries: int = 3) -> Optional[bytes]:
        for _ in range(tries):
            f = self.video.latest_frame()
            if f:
                return f
            time.sleep(0.25)
        return None

    def scan_circle(self):
        """Rotate a full circle grabbing ONE frame per heading, and nothing else.

        This is deliberately pure mechanics with no network in it. Detection costs two
        sequential HTTP round trips per frame (vision model, then jev), and doing that
        inline -- per frame, per heading -- is what made a 360° scan take ~70 s of a ~150 s
        battery: the drone spent nearly the whole flight hovering on a socket read. Capture
        is ~2 s per heading, so the whole circle is grabbed in the time detection used to
        take for two headings, and detect_frames() then judges the circle concurrently.

        Returns [(yaw_at_capture, jpeg)]. The yaw is the *recorded pose* at capture rather
        than an assumed i*scan_step, so a rotate the drone skipped (routine: 'No valid imu')
        can no longer silently mislabel every heading after it -- the turn back to a chosen
        heading is computed from live pose at decision time."""
        m = self.cfg["mission"]
        scan_deg = m["scan_step_deg"]
        n_steps = max(1, 360 // scan_deg)
        self.maintain_height()
        shots = []
        for i in range(n_steps):
            time.sleep(m["frame_settle_s"])
            frame = self.grab_frame()
            if frame:
                shots.append((self.pose.yaw, frame))
            if i < n_steps - 1:
                self.fly("cw %d" % scan_deg, settle=m.get("scan_settle_s", 0.35))
        return shots

    def turn_to(self, yaw: float, settle: float = 0.6) -> bool:
        """Turn to an absolute (dead-reckoned) yaw by the shortest way round."""
        turn = round((yaw - self.pose.yaw + 540) % 360 - 180)
        if abs(turn) < 5:
            return True
        return self.fly(("cw %d" if turn > 0 else "ccw %d") % abs(turn), settle=settle)

    def confirm_person(self, detector, yaw: float) -> Optional[Detection]:
        """Re-check a heading that looked like a person before committing the mission to it.

        The scan itself is one frame per heading, which on its own would be exactly the
        single-lucky-frame failure the burst rule was written for (a door once scored 0.84
        in one frame and 0.03 in the next). So a hit is cheap to *notice* and expensive to
        *believe*, in three separate ways: turn back to it and take `confirm_frames` FRESH
        frames; ask jev the skeptical STRICT_Q instead of the sensitive scan question; and
        require every one of those frames to individually clear confirm_min_confidence (a
        dropped frame fails the check outright rather than being judged on a partial burst).

        The frames are judged concurrently, so a 3-frame confirm costs about the same
        wall-clock as one frame did before."""
        m, j = self.cfg["mission"], self.cfg["jev"]
        need = m.get("confirm_frames", 3)
        bar = j.get("confirm_min_confidence", j["min_confidence"])
        self.turn_to(yaw)
        time.sleep(m["frame_settle_s"])
        frames = []
        for _ in range(need):
            f = self.grab_frame()
            if f:
                frames.append(f)
            time.sleep(m.get("frame_gap_s", 0.25))
        results = detector.detect_frames(frames, workers=m.get("detect_workers", 6), strict=True)
        dets = [d for d, _, _ in results if d is not None]
        if not dets:
            return None
        confs = [d.confidence for d in dets]
        best = max(dets, key=lambda d: d.confidence)
        self.ev("CONFIRM burst at %d° (strict): %s (need %d frames all >= %.2f)"
                % (round(yaw), " ".join("%.2f" % c for c in confs), need, bar))
        if len(dets) >= need and min(confs) >= bar and best.bbox is not None:
            best.confidence = round(sum(confs) / len(confs), 3)
            return best
        return None

    def _mark_visited(self):
        self.visited.append((self.pose.x, self.pose.y))

    def _is_unsearched(self, yaw: float, step_cm: float) -> bool:
        """Would one step along this heading land somewhere the drone hasn't been?"""
        a = math.radians(-yaw)
        nx, ny = self.pose.x + step_cm * math.cos(a), self.pose.y + step_cm * math.sin(a)
        return all(math.hypot(nx - vx, ny - vy) > step_cm for vx, vy in self.visited)

    # ---------------------------------------------------------------- search
    def phase_search(self, detector) -> Optional[Detection]:
        """jev decides where to go, after seeing the whole circle from each spot.

        One cycle is: capture a 360° circle of frames (mechanical, no network), judge them
        all concurrently in one batch, and -- if nothing looked like a person -- make a
        single jev call comparing every heading and pick the one to fly into. A heading
        that does look like a person is confirmed with a fresh burst before the mission
        commits to it.

        Returns the confirmed Detection, or None if there is no viable direction left from
        this spot (the caller repositions and keeps searching -- running out of ideas is
        not a reason to land). The hard limits on where jev may send it are unchanged:
        max_distance_from_launch_cm and max_height_deviation_cm, because there is no
        obstacle sensing and dead-reckoning drift compounds with distance. There is no
        longer a search *time* budget: only the battery floor ends a flight."""
        m, j = self.cfg["mission"], self.cfg["jev"]
        max_dist = m.get("max_distance_from_launch_cm", 500)
        step_cm = m.get("nav_forward_step_cm", 100)
        vert_step_cm = m.get("nav_vertical_step_cm", 30)
        max_height_dev = m.get("max_height_deviation_cm", 100)
        while True:
            shots = self.scan_circle()
            if not shots:
                self.ev("SEARCH: no video frames for a whole circle -- retrying")
                time.sleep(1.0)
                continue
            t0 = time.time()
            results = detector.detect_frames([f for _, f in shots], workers=m.get("detect_workers", 6))
            self.ev("SEARCH: %d headings judged in %.1fs [%s]" % (
                len(results), time.time() - t0,
                " ".join("%.2f" % d.confidence if d else "--" for d, _, _ in results)))

            # strongest person-looking heading in the circle, if any cleared the threshold
            hits = [(d.confidence, shots[i][0]) for i, (d, _, _) in enumerate(results)
                    if d is not None and d.confidence >= j["min_confidence"] and d.bbox is not None]
            if hits:
                conf, yaw = max(hits)
                self.ev("SEARCH: candidate person at %d° (scan conf %.2f) -- confirming" % (round(yaw), conf))
                det = self.confirm_person(detector, yaw)
                if det:
                    self.ev("SEARCH: person found, heading %d° (conf %.2f, %s)" % (
                        round(self.pose.yaw), det.confidence, det.backend),
                        pose=self.pose.as_dict(), detection=det.to_dict())
                    return det
                self.ev("SEARCH: candidate did not survive the confirm burst -- continuing")

            dist_home, _ = self.pose.home_vector()
            height_dev = (self.target_height_cm - self.launch_height_cm) if (
                self.target_height_cm is not None and self.launch_height_cm is not None) else 0
            headings = []
            for i, (yaw, _) in enumerate(shots):
                _, state, blocked = results[i]
                state = state or {}
                # turn is measured from live pose, so it stays correct even if the drone
                # skipped a rotate during the scan or during the confirm detour above
                turn = round((yaw - self.pose.yaw + 540) % 360 - 180)
                headings.append({
                    "turn_from_here_deg": turn,
                    "blocked": bool((blocked or {}).get("blocked")),
                    "scene": state.get("scene"),
                    "openness": state.get("openness"),
                    "path": state.get("path"),
                    "exits": state.get("exits"),
                    "unsearched": self._is_unsearched(yaw, step_cm),
                    "bright_frac": (blocked or {}).get("bright_frac"),
                    "edge_density": (blocked or {}).get("edge_density"),
                })
            # subtract step_cm up front so a heading jev picks is never one the mission then
            # has to second-guess and discard for being juuust over the ceiling
            choice = detector.decide_navigation(headings, dist_home, max_dist - step_cm, height_dev, max_height_dev)
            self.ev("SEARCH: jev picks %r from %d headings (%.0fcm from launch, h%+.0fcm)"
                    % (choice, len(headings), dist_home, height_dev), pose=self.pose.as_dict())
            if choice == "ascend" and height_dev + vert_step_cm <= max_height_dev:
                if self.fly("up %d" % vert_step_cm, settle=0.8) and self.target_height_cm is not None:
                    self.target_height_cm += vert_step_cm
            elif choice == "descend" and height_dev - vert_step_cm >= -max_height_dev:
                if self.fly("down %d" % vert_step_cm, settle=0.8) and self.target_height_cm is not None:
                    self.target_height_cm -= vert_step_cm
            elif isinstance(choice, int) and 0 <= choice < len(headings):
                self.turn_to(shots[choice][0])
                if self.fly("forward %d" % step_cm, settle=0.8):
                    self._mark_visited()
            else:
                self.ev("SEARCH: no viable direction from this spot")
                return None

    def reposition(self):
        """Nothing viable from here, but running out of ideas is not a reason to land.
        Nudge somewhere else -- back toward the launch point if the drone has wandered,
        otherwise a quarter turn and a step -- and let the next circle see a new room."""
        m = self.cfg["mission"]
        step_cm = m.get("nav_forward_step_cm", 100)
        dist, turn = self.pose.home_vector()
        if dist > step_cm:
            self.ev("REPOSITION: backtracking %.0fcm toward launch" % min(dist, step_cm))
            if abs(turn) >= 5:
                self.fly(("cw %d" if turn > 0 else "ccw %d") % abs(round(turn)), settle=0.6)
        else:
            self.ev("REPOSITION: quarter turn and a step")
            self.fly("cw 90", settle=0.6)
        if self.fly("forward %d" % step_cm, settle=0.8):
            self._mark_visited()

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

        # The flip is the point of finding the person, so the software gate on it is now the
        # same battery floor as everything else rather than a separate high threshold: this
        # pack sags to the high teens under search load, and a 55% gate meant the flip was
        # skipped on every real flight that actually found someone. The drone's own firmware
        # still refuses a flip it considers unsafe, and that refusal is just logged -- it is
        # a failed command like any other and does not end the flight.
        if m["do_flip"]:
            bat = self.drone.battery()
            floor = self.cfg["sdk"].get("min_battery_flip", 0)
            if 0 <= bat < floor:
                self.ev("FLIP skipped: battery %d%% < %d%%" % (bat, floor))
            else:
                self.fly("up 30", settle=1.0)  # headroom, then celebrate
                if self.fly("flip %s" % m["flip_direction"], settle=2.5):
                    self.ev("FLIP done (%s)" % m["flip_direction"])
                else:
                    self.ev("FLIP refused by drone (battery %d%%)" % bat)
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
        # Stop our local ffmpeg capture BEFORE telling the drone to streamoff, not after:
        # once the drone stops transmitting, ffmpeg's UDP input has no more packets to read
        # at all (UDP has no EOF), so its blocking read never returns and it never gets to
        # notice the SIGTERM stop() sends -- it just hangs until the grace timeout kills it,
        # losing the mp4 trailer (moov atom) and the whole recording. Stopping it while the
        # drone is still streaming lets it exit cleanly between reads, same as any abort path
        # (which never sends drone-side streamoff and so never hits this).
        if self.video:
            self.video.stop()
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
            # Airborne from here on, and the landing policy takes over: only a found person,
            # the battery floor (MissionStop) or the operator puts this drone down. A command
            # that errors is skipped inside fly(); an error that escapes a whole phase is
            # logged and the search simply resumes, rather than ending the flight.
            det = None
            while det is None:
                try:
                    det = self.phase_search(detector)
                    if det is None:
                        self.reposition()
                except MissionStop:
                    raise
                except (TelloError, RuntimeError) as exc:
                    self.ev("ERROR during search (not landing, continuing): %s" % exc)
                    time.sleep(1.0)
            if not search_only:
                try:
                    self.phase_photo_and_flip(det)
                except (TelloError, RuntimeError) as exc:
                    self.ev("ERROR during photo/flip (continuing): %s" % exc)
            try:
                self.phase_return()
            except (TelloError, RuntimeError) as exc:
                self.ev("ERROR during return (landing anyway): %s" % exc)
            self.phase_land()
            self.report["outcome"] = "person found"
        except MissionStop as exc:
            # Land here rather than flying home: this is the battery floor, and the charge
            # left is worth more as a controlled descent than as a return leg.
            self.ev("STOP: %s -- landing here" % exc)
            self.report["outcome"] = "landed: %s" % exc
            try:
                self.phase_land()
            except Exception as land_exc:  # nothing left to try but the emergency path
                self.ev("LAND failed (%s) -- emergency land" % land_exc)
                if self.drone and self.airborne:
                    self.drone.emergency_land()
        except (TelloError, RuntimeError) as exc:
            # Only the pre-takeoff phases (power / wifi / SDK gate / detector probe / launch)
            # can reach this now -- everything after takeoff absorbs its own errors above.
            # If it somehow happens airborne, land: an unknown failure outside the flight
            # loop is not a state to keep flying in.
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
            # covers every ending: a normal landing after the person was found, the
            # battery-floor landing, a pre-takeoff abort and an operator interrupt -- this
            # always runs after cleanup, once the drone is either landed or has had
            # emergency_land() sent. Only for a plug this mission itself turned on: a
            # --skip-power run means the operator is managing power by hand, and this
            # shouldn't switch it off out from under them.
            if self.powered_on_by_mission and self.cfg["power"].get("power_off_after_mission") and not self.dry:
                power.power_off(self.cfg["power"])
                self.ev("power: off")
        self.report["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.report["final_pose"] = self.pose.as_dict()
        self.report["flight_s"] = round(time.time() - self.t_takeoff, 1) if self.t_takeoff else 0
        path = os.path.join(self.base, self.cfg["mission"]["logs_dir"], "report_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
        with open(path, "w") as f:
            json.dump(self.report, f, indent=2)
        log.info("report -> %s", path)
        return self.report
