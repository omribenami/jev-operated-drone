"""Person detection: jev is the judge; a pluggable *frontend* turns the frame into text state.

Frontends (config jev.frontend / CLI --opencv, --vision, --frontend):
  features : classical OpenCV statistics only (no learned model, offline).
  opencv   : OpenCV's built-in HOG people detector adds "hog" candidates with SVM weights.
  vision   : fast vision model on the Vercel AI Gateway (default openai/gpt-4o-mini; ~1.7 s/frame,
             gpt-4.1-nano is faster but hallucinated people in tests)
             describes the frame as JSON (people, bboxes, real-vs-screen) via the native
             /v4/ai/language-model endpoint.
jev (text-only) then evaluates the state with typed questions and returns the
probability that a real person is visible plus which candidate is the target.
"""
import base64
import json
import re
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import requests

log = logging.getLogger("detector")

# fixed filename (not timestamped) under logs_dir: mission.py points the video recording's
# bottom text overlay at this same path, always overwritten with jev's latest decision.
OVERLAY_BOTTOM_NAME = "overlay_bottom.txt"


@dataclass
class Detection:
    person: bool
    confidence: float
    bbox: Optional[list]  # normalized [x1,y1,x2,y2]
    description: str
    backend: str

    def center_x(self) -> Optional[float]:
        if not self.bbox:
            return None
        return (self.bbox[0] + self.bbox[2]) / 2.0

    def to_dict(self) -> Dict:
        return asdict(self)


class FrameState:
    """Frontend 'features': classical (non-ML) frame -> text state. Requires opencv + numpy."""
    name = "features"

    def __init__(self):
        import cv2
        import numpy as np
        self.cv2, self.np = cv2, np
        self.prev_gray = None
        self.frame_count = 0
        self.bg = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=32, detectShadows=False)

    def build(self, jpeg: bytes) -> Dict:
        cv2, np = self.cv2, self.np
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"decodable": False}
        img = cv2.resize(img, (480, 360))  # fixed size so motion/background models always match
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # skin-tone mask (YCrCb ranges), motion mask, edges
        skin = cv2.inRange(ycc, (0, 135, 85), (255, 180, 135))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = self.bg.apply(img)
        self.frame_count += 1
        if self.frame_count <= 5:  # background model not settled yet: everything looks like motion
            fg = np.zeros_like(fg)
        motion = 0.0
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            motion = float(np.mean(cv2.absdiff(gray, self.prev_gray)) / 255.0)
        self.prev_gray = gray
        edges = cv2.Canny(gray, 60, 160)

        objs: List[Dict] = []

        def describe(x, y, bw, bh, seed):
            x, y = max(0, x), max(0, y)
            bw, bh = min(w - x, bw), min(h - y, bh)
            if bw < 4 or bh < 4:
                return None
            roi = slice(y, y + bh), slice(x, x + bw)
            top = slice(y, y + max(1, bh // 3)), slice(x, x + bw)
            area = float(bw * bh)
            return {
                "seed": seed,
                "bbox": [round(x / w, 3), round(y / h, 3), round((x + bw) / w, 3), round((y + bh) / h, 3)],
                "height_frac": round(bh / h, 2), "width_frac": round(bw / w, 2),
                "aspect_h_over_w": round(bh / float(bw), 2),
                "skin_ratio": round(float(np.mean(skin[roi] > 0)), 3),
                "skin_in_top_third": round(float(np.mean(skin[top] > 0)), 3),
                "moving_ratio": round(float(np.mean(fg[roi] > 0)), 3),
                "edge_density": round(float(np.mean(edges[roi] > 0)), 3),
                "brightness": round(float(np.mean(gray[roi])) / 255.0, 2),
                "touches_floor": bool(y + bh > 0.85 * h),
            }

        # (a) skin-seeded candidates: each skin blob (face/hands) implies a body box below it
        n, labels, stats, cents = cv2.connectedComponentsWithStats(skin)
        for i in range(1, n):
            sx, sy, sw, sh, sarea = stats[i]
            if sarea < 0.0006 * w * h or sh > 0.35 * h or not (0.5 <= sw / float(max(sh, 1)) <= 2.0):
                continue  # too small, too tall, or not head/hand shaped
            head = max(sw, sh)
            o = describe(int(sx + sw / 2 - head * 1.5), int(sy - head * 0.3), int(head * 3), int(head * 7.5), "skin_blob")
            if o:
                objs.append(o)

        # (b) motion/edge silhouettes (once the background model has history)
        sil = cv2.bitwise_or(fg, cv2.dilate(edges, None, iterations=1))
        sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(sil)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 0.01 * w * h or bh < 0.2 * h or bw > 0.9 * w:
                continue
            o = describe(int(x), int(y), int(bw), int(bh), "silhouette")
            if o:
                objs.append(o)
        objs.extend(self.extra_candidates(img, describe))
        objs.sort(key=lambda o: (o.get("detector_weight", 0) * 3 + o["skin_in_top_third"] * 2 + o["moving_ratio"]) * o["height_frac"], reverse=True)

        gh, gw = 3, 4
        grid = [[round(float(np.mean(gray[r * h // gh:(r + 1) * h // gh, c * w // gw:(c + 1) * w // gw])) / 255.0, 2)
                 for c in range(gw)] for r in range(gh)]
        return {
            "decodable": True,
            "frame": {"mean_brightness": round(float(np.mean(gray)) / 255.0, 3),
                      "blur_score": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
                      "edge_density": round(float(np.mean(edges > 0)), 4),
                      "skin_coverage": round(float(np.mean(skin > 0)), 4),
                      "motion_vs_prev_frame": round(motion, 4),
                      "saturation_mean": round(float(np.mean(hsv[:, :, 1])) / 255.0, 3),
                      "brightness_grid_3x4": grid},
            "candidate_objects": objs[:6],
            "notes": self.notes() + " features from classical image processing on a forward-facing drone camera ~1m above floor; "
                     "skin_blob candidates are body boxes inferred from a skin-colored blob (face/hands); "
                     "a standing person typically: height_frac>0.35, aspect 1.5-4, skin_in_top_third>0.05, touches_floor",
        }


    def extra_candidates(self, img, describe) -> List[Dict]:
        return []

    def notes(self) -> str:
        return ""


class HogFrontend(FrameState):
    """Frontend 'opencv': OpenCV HOG+SVM people detector candidates on top of the features."""
    name = "opencv"

    def __init__(self):
        super().__init__()
        self.hog = self.cv2.HOGDescriptor()
        self.hog.setSVMDetector(self.cv2.HOGDescriptor_getDefaultPeopleDetector())

    def extra_candidates(self, img, describe) -> List[Dict]:
        # run HOG on a 2x upscale so people that are small in the frame still fill the 64x128 window
        big = self.cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
        rects, weights = self.hog.detectMultiScale(big, winStride=(8, 8), padding=(8, 8), scale=1.05)
        out = []
        for (x, y, bw, bh), wgt in zip(rects, weights):
            o = describe(int(x // 2), int(y // 2), int(bw // 2), int(bh // 2), "hog")
            if o:
                o["detector_weight"] = round(float(wgt), 2)
                out.append(o)
        return out

    def notes(self) -> str:
        return ("hog candidates come from OpenCV's HOG pedestrian detector; detector_weight > 0.5 is a "
                "fairly confident pedestrian-shaped window, > 1.0 strong.")


class VisionFrontend:
    """Frontend 'vision': a fast vision model describes the frame as JSON; jev judges the JSON."""
    name = "vision"
    PROMPT = (
        "You are the camera of an indoor search drone about 1 m above the floor. Describe this frame as compact "
        "JSON only: {\"people\": [{\"bbox\": [x1,y1,x2,y2] normalized 0-1, \"pose\": \"standing|sitting|lying|unknown\", "
        "\"is_real_person\": true|false, \"notes\": \"short\"}], \"scene\": \"one sentence\", "
        "\"image_quality\": \"good|blurry|dark\"}. People on screens, posters, photos or mirrors get is_real_person=false."
    )

    def __init__(self, cfg: Dict, headers: Dict):
        self.url = cfg["base_url"].rstrip("/") + "/v4/ai/language-model"
        self.model = cfg.get("vision_model", "openai/gpt-4o-mini")
        self.timeout = cfg.get("vision_timeout_s", 20)
        self.headers = dict(headers)
        self.headers.update({"ai-language-model-specification-version": "3", "ai-model-id": self.model})
        self.headers.pop("ai-evaluation-model-specification-version", None)
        self.prev_gray = None  # calibrate resets this; harmless here

    def build(self, jpeg: bytes) -> Dict:
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        body = {"prompt": [{"role": "user", "content": [
                    {"type": "text", "text": self.PROMPT},
                    {"type": "file", "mediaType": "image/jpeg", "data": {"type": "url", "url": data_url}}]}],
                "maxOutputTokens": 300, "temperature": 0}
        t0 = time.time()
        r = requests.post(self.url, headers=self.headers, json=body, timeout=self.timeout)
        if not r.ok:
            raise RuntimeError("vision model HTTP %s: %s" % (r.status_code, r.text[:200]))
        text = "".join(c.get("text", "") for c in r.json().get("content", []) if c.get("type") == "text")
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("vision model returned non-JSON: %r" % text[:120])
        state = json.loads(m.group(0))
        people = [p for p in state.get("people", []) if isinstance(p, dict) and isinstance(p.get("bbox"), list) and len(p["bbox"]) == 4]
        for p in people:
            p["seed"] = "vision"
        state["candidate_objects"] = people
        state["decodable"] = True
        state["vision_model"] = self.model
        state["notes"] = ("candidate_objects were produced by a vision-language model looking at the frame; "
                          "is_real_person=false means it saw a depiction (screen/poster), not a body.")
        log.debug("vision %s %.1fs: %d people", self.model, time.time() - t0, len(people))
        return state


class JevDetector:
    """typesafe-ai/jev through the Vercel AI Gateway evaluation endpoint.

    POST {base}/v4/ai/evaluation-model with {state, questions}; jev answers typed
    questions in one round trip: a boolean (probability a real person is visible)
    and a choice (which candidate silhouette is the person).
    """
    name = "jev"

    def __init__(self, cfg: Dict, decisions_path: Optional[str] = None, overlay_path: Optional[str] = None):
        self.url = cfg["base_url"].rstrip("/") + "/v4/ai/evaluation-model"
        self.model = cfg["model"]
        self.timeout = cfg.get("timeout_s", 20)
        self.decisions_path = decisions_path
        # fixed file (not append-only), always overwritten with just the latest decision's
        # one-line description -- mission.py points the recording's bottom text overlay at it.
        self.overlay_path = overlay_path
        self.last_state: Optional[Dict] = None
        key = os.environ.get(cfg.get("api_key_env", "JEV_API_KEY"), "")
        if not key:
            raise RuntimeError("no API key in env %s (put it in secrets.json)" % cfg.get("api_key_env", "JEV_API_KEY"))
        self.headers = {
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "ai-gateway-protocol-version": "0.0.1",
            "ai-evaluation-model-specification-version": "4",
            "ai-model-id": self.model,
        }
        if decisions_path:
            os.makedirs(os.path.dirname(decisions_path), exist_ok=True)
        frontend = cfg.get("frontend", "features")
        if frontend == "vision":
            self.frames = VisionFrontend(cfg, self.headers)
        elif frontend == "opencv":
            self.frames = HogFrontend()
        elif frontend == "features":
            self.frames = FrameState()
        else:
            raise RuntimeError("unknown jev.frontend %r (features|opencv|vision)" % frontend)
        self.name = "jev+" + frontend

    def evaluate(self, state, questions: Dict, attempts: int = 2) -> Dict:
        last = None
        for i in range(attempts):
            try:
                r = requests.post(self.url, headers=self.headers, json={"state": state, "questions": questions},
                                  timeout=self.timeout)
            except requests.RequestException as exc:
                last = exc
                log.warning("jev request failed (%d/%d): %s", i + 1, attempts, exc)
                continue
            if r.status_code == 403 and "customer_verification" in r.text:
                raise RuntimeError("Vercel AI Gateway refuses requests: add a credit card to the Vercel team")
            if r.status_code in (429, 529, 502, 503):
                last = RuntimeError("jev HTTP %s" % r.status_code)
                time.sleep(1.0 + i)
                continue
            if not r.ok:
                raise RuntimeError("jev HTTP %s: %s" % (r.status_code, r.text[:200]))
            return r.json()["answers"]
        raise RuntimeError("jev unreachable: %s" % last)

    def healthy(self) -> bool:
        try:
            a = self.evaluate("The build failed with exit code 1.",
                              {"passed": {"type": "boolean", "instructions": "Did the build succeed?"}})
            p = float(a["passed"]["probability"])
            log.info("jev probe ok (sanity probability %.2f)", p)
            return p < 0.5
        except Exception as exc:
            log.error("jev probe failed: %s", exc)
            return False

    def detect(self, jpeg: bytes) -> Detection:
        state = self.frames.build(jpeg)
        self.last_state = state  # reused by decide_navigation, so it needn't re-describe the frame
        if not state.get("decodable"):
            return Detection(False, 0.0, None, "undecodable frame", self.name)
        objs = state["candidate_objects"]
        questions = {
            "person_visible": {
                "type": "boolean",
                "instructions": "This is analysis output for one frame from an indoor search drone's forward camera "
                                "about 1 m above the floor. Is at least one real, physically present human visible? "
                                "Use candidate_objects and notes. Numeric candidates: person-like when tall "
                                "(height_frac > 0.35), aspect_h_over_w 1.5-4, skin_in_top_third > 0.05, touching the "
                                "floor, moving, or with detector_weight > 0.5. Vision candidates: trust "
                                "is_real_person. Furniture and walls are static, low skin, often wide. Be conservative.",
                "criteria": {"true": "a real human body is present in the frame",
                             "false": "only furniture, walls, floor, objects or noise"},
            },
        }
        if objs:
            questions["target"] = {
                "type": "choice",
                "instructions": "Which candidate_objects index is most likely the human body?",
                "criteria": {str(i): "candidate_objects[%d] (%s): %s" % (
                    i, o.get("seed"), json.dumps({k: v for k, v in o.items() if k not in ("seed", "bbox")})[:160])
                    for i, o in enumerate(objs)},
            }
        answers = self.evaluate(state, questions)
        prob = float(answers["person_visible"]["probability"])
        bbox = None
        if objs and "target" in answers:
            try:
                bbox = objs[int(answers["target"]["choice"])]["bbox"]
            except (KeyError, ValueError, IndexError):
                bbox = objs[0]["bbox"]
        person = prob >= 0.5
        det = Detection(person, prob, bbox if person else None,
                        "jev p=%.2f over %d candidates (%s)" % (prob, len(objs), self.frames.name), self.name)
        self._log_decision(state, questions, answers, det)
        return det

    def _log_decision(self, state: Dict, questions: Dict, answers: Dict, det: Detection):
        """Append the full input/output of one jev call, so a mission can be reviewed
        after the fact to see exactly what candidates jev saw and how it judged them."""
        if self.decisions_path:
            rec = {
                "t": time.strftime("%Y-%m-%d %H:%M:%S"),
                "frontend": self.frames.name,
                "candidate_objects": state.get("candidate_objects", []),
                "questions": {k: v.get("instructions") for k, v in questions.items()},
                "answers": answers,
                "result": det.to_dict(),
            }
            try:
                with open(self.decisions_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError as exc:
                log.warning("could not write jev decision log: %s", exc)
        if self.overlay_path:
            try:
                with open(self.overlay_path, "w") as f:
                    f.write(det.description)
            except OSError:
                pass

    def decide_navigation(self, distance_from_launch_cm: float, max_distance_cm: float) -> str:
        """Ask jev where to search next, reusing the last detect() call's scene/candidate
        state so this needs no fresh frame or frontend pass. Returns 'advance', 'rotate', or
        'stop'; falls back to 'rotate' on any failure since standing pat and looking
        elsewhere is always safe."""
        state = self.last_state or {}
        can_advance = distance_from_launch_cm < max_distance_cm
        criteria = {
            "rotate": "turn in place to look in a different direction from here; nothing here suggests advancing is useful",
            "stop": "this direction looks like a dead end (wall, closed door, furniture blocking the path) -- do not go further this way",
        }
        if can_advance:
            criteria["advance"] = "the scene shows a continuing open path (hallway, doorway, open room) in the current heading -- move forward"
        questions = {
            "next_action": {
                "type": "choice",
                "instructions": (
                    "This is an indoor search drone deciding where to search next for a missing person, using the "
                    "same frame analysis a person-check just ran on. Should it rotate to look elsewhere from this "
                    "spot, advance forward into open space, or treat this heading as a dead end?"
                    + ("" if can_advance else " It has already travelled far from its launch point (dead-reckoned, "
                       "no GPS), so advancing is not offered right now -- pick rotate or stop.")
                ),
                "criteria": criteria,
            },
        }
        action, answers = "rotate", {}
        try:
            answers = self.evaluate(state, questions)
            choice = answers["next_action"]["choice"]
            if choice in criteria:
                action = choice
        except Exception as exc:
            log.warning("jev navigation decision failed, defaulting to rotate: %s", exc)
        if self.decisions_path:
            rec = {
                "t": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "navigation",
                "distance_from_launch_cm": round(distance_from_launch_cm), "action": action, "answers": answers,
            }
            try:
                with open(self.decisions_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError as exc:
                log.warning("could not write jev decision log: %s", exc)
        if self.overlay_path:
            try:
                with open(self.overlay_path, "w") as f:
                    f.write("jev nav: %s (dist %.0fcm)" % (action, distance_from_launch_cm))
            except OSError:
                pass
        return action


class FakeDetector:
    """Dry-run: reports a person after N calls so the whole mission path executes."""
    name = "fake"

    def __init__(self, hit_after: int = 6):
        self.calls = 0
        self.hit_after = hit_after

    def healthy(self):
        return True

    def detect(self, jpeg: bytes) -> Detection:
        self.calls += 1
        if self.calls >= self.hit_after:
            return Detection(True, 0.93, [0.55, 0.2, 0.8, 0.9], "simulated person", self.name)
        return Detection(False, 0.05, None, "nothing", self.name)

    def decide_navigation(self, distance_from_launch_cm: float, max_distance_cm: float) -> str:
        # alternate so a dry run exercises both the advance and rotate code paths
        return "advance" if distance_from_launch_cm < max_distance_cm and self.calls % 2 == 0 else "rotate"


def make_detector(cfg: Dict, dry_run: bool = False, decisions_path: Optional[str] = None,
                   overlay_path: Optional[str] = None):
    if dry_run:
        return FakeDetector()
    try:
        jev = JevDetector(cfg, decisions_path=decisions_path, overlay_path=overlay_path)
    except ImportError as exc:
        raise RuntimeError("frontend '%s' needs opencv-python-headless + numpy: %s" % (cfg.get("frontend"), exc))
    if not jev.healthy():
        raise RuntimeError("jev (%s) is not answering through %s; refusing to fly" % (cfg["model"], cfg["base_url"]))
    log.info("jev ready: %s via %s, frontend=%s", cfg["model"], cfg["base_url"], jev.frames.name)
    return jev
