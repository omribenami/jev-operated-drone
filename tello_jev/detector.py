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
import threading
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
    # The camera is fixed and points slightly down, so at search height a person often enters
    # frame as legs/feet/an arm long before a whole body does -- the prompt says so explicitly,
    # because an earlier version described a real person half-behind a door as
    # is_real_person=false ("partially visible"), and jev then scored that frame 0.18.
    # The navigation fields (openness/path/exits) cost nothing extra -- this call is already
    # being made for detection -- and they are the only real depth/layout signal jev gets for
    # decide_navigation() on this machine, where the opencv blocked-check can't run at all.
    PROMPT = (
        "You are the forward camera of an indoor search drone flying about 1.5 m above the floor; the lens points "
        "slightly downward, so people often appear only partly in frame. Reply with compact JSON only: "
        "{\"people\": [{\"bbox\": [x1,y1,x2,y2] normalized 0-1, \"pose\": \"standing|sitting|lying|unknown\", "
        "\"is_real_person\": true|false, \"notes\": \"short\"}], \"scene\": \"one sentence\", "
        "\"openness\": \"open|partial|blocked\", \"path\": \"one short phrase for what lies straight ahead\", "
        "\"exits\": [\"doorway|hallway|open floor|stairs, each with rough left/centre/right position\"], "
        "\"image_quality\": \"good|blurry|dark\"}. "
        "A REAL HUMAN COUNTS EVEN IF ONLY PARTLY VISIBLE: bare legs, feet, an arm, a hand, a head, or a body behind "
        "or beside a door or furniture are all is_real_person=true -- say which part you can see in notes. "
        "Only a depiction of a person (on a screen, poster, photo or mirror) gets is_real_person=false. "
        "openness describes the path directly ahead: \"open\" = floor you could fly several metres across, "
        "\"partial\" = a gap or clutter to thread, \"blocked\" = a wall, closed door or furniture filling the view."
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
        self.last_blocked: Dict = {"blocked": False}
        # detect_frames() runs detect_one() on several threads at once; they all append to
        # the same decisions log and overwrite the same overlay file.
        self._log_lock = threading.Lock()
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

    def _check_blocked(self, jpeg: bytes) -> Dict:
        """Cheap, deterministic 'mostly flat and bright' check -- a wall or closed door
        filling the frame -- independent of whatever frontend/LLM is in use, so 'advance'
        can be hard-gated on it rather than trusting jev to always infer it from a scene
        description alone. Degrades to 'not blocked' if opencv isn't installed (it's an
        optional dependency for the vision-only path)."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return {"blocked": False}
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"blocked": False}
        img = cv2.resize(img, (160, 120))
        bright_frac = float(np.mean(img > 200))
        edge_density = float(np.mean(cv2.Canny(img, 60, 160) > 0))
        # a wall/door fills the frame with bright, near-featureless surface; a bright but
        # navigable scene (open doorway, lit room) still has real edge detail in it.
        blocked = bright_frac > 0.6 and edge_density < 0.03
        return {"blocked": blocked, "bright_frac": round(bright_frac, 3), "edge_density": round(edge_density, 4)}

    def detect(self, jpeg: bytes) -> Detection:
        """Single-frame detection (calibrate / detector-check / any sequential caller).
        Keeps last_state/last_blocked up to date for callers that read them afterwards --
        detect_frames() deliberately does not, since concurrent calls would race on them."""
        det, state, blocked = self.detect_one(jpeg)
        self.last_state, self.last_blocked = state, blocked
        return det

    def detect_frames(self, frames: List[bytes], workers: int = 6, strict: bool = False) -> List[tuple]:
        """Detect over a batch of frames concurrently, returning [(Detection, state, blocked)]
        in the same order as `frames` (a failed frame's entry is (None, {}, {})).

        `strict` swaps jev's question for the skeptical one -- see detect_one().

        The mission captures a whole 360° circle of frames mechanically first and then judges
        them all at once through here: the per-frame cost is two sequential network round trips
        (vision model, then jev), which sequentially dominated the search loop -- a full circle
        took ~70 s of which the drone spent almost all of it hovering, waiting on HTTP. Run
        across headings they overlap into roughly one frame's latency for the whole circle.

        Per-call state lives in the returned tuple rather than on self, because self.last_state
        / self.last_blocked cannot be shared by concurrent calls."""
        from concurrent.futures import ThreadPoolExecutor
        if not frames:
            return []

        def one(jpeg):
            try:
                return self.detect_one(jpeg, strict=strict)
            except Exception as exc:
                log.warning("detector error on frame: %s", exc)
                return (None, {}, {})

        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(frames)))) as pool:
            return list(pool.map(one, frames))

    # The scan and the confirm deliberately ask jev *different* questions, because they want
    # different things. Sweeping a room wants recall: the frontend is told to report a bare
    # leg or an arm behind a door as a person, since missing someone entirely is the one
    # unrecoverable failure and a false alarm only costs a few seconds. Committing the
    # mission to a heading wants precision: the same sensitivity that stops it walking past
    # a person also makes it call a tan cushion in a doorway "part of a body" (seen exactly
    # that: p=0.81 on a rounded flesh-toned object, with a bbox that didn't even land on it).
    # So the confirm re-asks in the skeptical form, against a higher bar, on fresh frames.
    SCAN_Q = ("This is analysis output for one frame from an indoor search drone's forward camera "
              "about 1.5 m above the floor, lens angled slightly down. Could at least one real, "
              "physically present human be visible? Use candidate_objects and notes. A person may be "
              "only partly in frame -- legs, feet, an arm, a head, or a body behind a door or furniture "
              "all count. Numeric candidates: person-like when tall (height_frac > 0.35), aspect_h_over_w "
              "1.5-4, skin_in_top_third > 0.05, touching the floor, moving, or with detector_weight > 0.5. "
              "Vision candidates: trust is_real_person. This is a first sweep, so lean towards yes when "
              "it is genuinely ambiguous -- a second, stricter check follows.")
    # Measured on frames from the 2026-09-21 flight (see README): with this wording a real
    # person seen only as bare legs at the frame edge scores 0.73-0.78, the tan blob in a
    # doorway that the sensitive scan rates 0.82-0.86 drops to 0.52-0.71, and an empty room
    # is 0.02-0.05. The bar sits between them at jev.confirm_min_confidence (0.60) and every
    # frame in the burst has to clear it, which is what actually separates the two -- the
    # blob cleared 0.60 on one frame out of four. An earlier, harsher wording ("is a human
    # DEFINITELY visible") rejected the real person too (0.39-0.45): the thing to be strict
    # about is human *form*, not how much of the body happens to be in frame.
    STRICT_Q = ("This is analysis output for one frame from an indoor search drone's forward camera about "
                "1.5 m above the floor, lens angled slightly down. A first sweep flagged this direction and "
                "the drone is about to commit its remaining battery to it, so judge it carefully: is a real "
                "human body part clearly recognisable here? YES for anything with unmistakable human form and "
                "proportion even if only part of it is in frame or it is at the edge -- bare legs, feet, a "
                "hand, an arm, a torso, a head. NO for a shape that is only person-*coloured* or "
                "person-*sized* without recognisable human form: cushions, plush toys, laundry, bags, wood, "
                "cardboard, a vague blob in a doorway or behind furniture, or a candidate whose bbox does not "
                "match what the notes describe. NO for people on screens, posters, photos and mirrors.")

    def detect_one(self, jpeg: bytes, strict: bool = False) -> tuple:
        """Thread-safe core of detect(): returns (Detection, state, blocked) and touches no
        instance state except the append-only decision log.

        strict=True asks the skeptical confirm question instead of the sensitive scan one."""
        state = self.frames.build(jpeg)
        blocked = self._check_blocked(jpeg)
        if not state.get("decodable"):
            return Detection(False, 0.0, None, "undecodable frame", self.name), state, blocked
        objs = state["candidate_objects"]
        questions = {
            "person_visible": {
                "type": "boolean",
                "instructions": self.STRICT_Q if strict else self.SCAN_Q,
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
                        "jev p=%.2f over %d candidates (%s%s)" % (
                            prob, len(objs), self.frames.name, ", strict" if strict else ""), self.name)
        self._log_decision(state, questions, answers, det, strict)
        return det, state, blocked

    def _log_decision(self, state: Dict, questions: Dict, answers: Dict, det: Detection, strict: bool = False):
        """Append the full input/output of one jev call, so a mission can be reviewed
        after the fact to see exactly what candidates jev saw and how it judged them."""
        rec = {
            "t": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frontend": self.frames.name,
            "candidate_objects": state.get("candidate_objects", []),
            "scene": state.get("scene"), "openness": state.get("openness"), "path": state.get("path"),
            "strict": strict,
            "questions": {k: v.get("instructions") for k, v in questions.items()},
            "answers": answers,
            "result": det.to_dict(),
        }
        with self._log_lock:
            if self.decisions_path:
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

    def decide_navigation(self, headings: List[Dict], distance_from_launch_cm: float, max_distance_cm: float,
                          height_dev_cm: float = 0.0, max_height_dev_cm: float = 0.0) -> Optional[str]:
        """Pick where to go only after seeing the full circle from this spot -- a single
        jev call comparing every heading scanned (one look() per heading already ran and
        filled each entry's 'blocked'/'scene'), rather than reacting to whichever way the
        drone happens to be facing. Straight-line moves into whichever open heading looks
        most promising, instead of shuffling forward one step at a time.

        Returns the chosen heading's index (int, as a plain int though jev answers with its
        string key), 'ascend', 'descend', or None if there is nothing viable at all (every
        heading blocked and no height room left) -- in which case the mission should give
        up on this spot and head home. Falls back to the first open heading (or None) on
        any jev failure, since picking *a* heading is safer than picking none arbitrarily.

        A heading only makes it into the choice set if it isn't a dead end by one of two
        gates, rather than being left purely to jev's reading of the scene text:
          - _check_blocked() (deterministic, opencv) -- a flat bright wall/door filling the
            frame. This is the strong gate, but it needs opencv+numpy importable and does
            nothing at all without them (which is the case on the current host), so:
          - the vision frontend's own openness=="blocked" -- the model that actually looked
            at the pixels calling it a wall/closed door/furniture. Weaker than the opencv
            check (it is still a model's judgement) but, unlike jev reading a one-line scene
            string, it is made while looking at the image."""
        open_idx = [i for i, h in enumerate(headings)
                    if not h.get("blocked") and h.get("openness") != "blocked"]
        can_advance = distance_from_launch_cm < max_distance_cm and bool(open_idx)
        can_ascend = max_height_dev_cm > 0 and height_dev_cm < max_height_dev_cm
        can_descend = max_height_dev_cm > 0 and height_dev_cm > -max_height_dev_cm
        criteria = {}
        if can_advance:
            for i in open_idx:
                h = headings[i]
                # everything the frontend saw down this heading, not just the one-line scene:
                # jev is text-only and this string is literally all it knows about the
                # direction it is being asked to fly the drone into.
                bits = [h.get("scene") or "(no scene description available)"]
                if h.get("path"):
                    bits.append("straight ahead: %s" % h["path"])
                if h.get("exits"):
                    bits.append("exits visible: %s" % ", ".join(str(e) for e in h["exits"][:4]))
                if h.get("openness"):
                    bits.append("openness: %s" % h["openness"])
                if h.get("unsearched"):
                    bits.append("this heading leads away from where the drone has already searched")
                criteria[str(i)] = "turn %+d° from current facing -- %s" % (
                    h.get("turn_from_here_deg", 0), "; ".join(bits))
        if can_ascend:
            criteria["ascend"] = "none of the directions looked as promising as gaining height here would -- e.g. the view is blocked by something low with open space visible above it"
        if can_descend:
            criteria["descend"] = "none of the directions looked as promising as losing height here would -- e.g. something suggests a person could be lower than these frames show (seated, lying, under furniture)"
        if not criteria:
            return None  # nothing offered at all -- caller gives up on this spot
        why_not = []
        if not can_advance:
            why_not.append("every direction is either a flat bright wall/door filling the frame, or advancing "
                           "would take it too far from its launch point (dead-reckoned, no GPS) -- do not invent a heading choice")
        if not can_ascend and max_height_dev_cm > 0:
            why_not.append("it is already at its ceiling for this flight -- do not pick ascend")
        if not can_descend and max_height_dev_cm > 0:
            why_not.append("it is already at its floor for this flight -- do not pick descend")
        questions = {
            "best_direction": {
                "type": "choice",
                "instructions": (
                    "This indoor search drone just rotated a full circle at this spot, looking for a missing "
                    "person at each heading (none found) and noting the scene in each direction. Pick whichever "
                    "single option offers the best next move to continue searching. Prefer, in order: a heading "
                    "that opens into space the drone has not searched yet (a doorway, hallway or open room leading "
                    "somewhere new) over more of the same room; an open path over a tight or cluttered one; and a "
                    "direction where a person could plausibly be (seating, desks, beds, other rooms) over bare "
                    "floor or a corner. The drone has limited battery, so favour whichever heading is most likely "
                    "to put a person in frame soonest."
                    + ("" if not why_not else " " + " ".join(why_not))
                ),
                "criteria": criteria,
            },
        }
        action, answers = (str(open_idx[0]) if open_idx else None), {}
        try:
            answers = self.evaluate({"headings": headings}, questions)
            choice = answers["best_direction"]["choice"]
            if choice in criteria:
                action = choice
        except Exception as exc:
            log.warning("jev navigation decision failed, defaulting to %r: %s", action, exc)
        if self.decisions_path:
            rec = {
                "t": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "navigation", "headings": headings,
                "distance_from_launch_cm": round(distance_from_launch_cm), "height_dev_cm": round(height_dev_cm),
                "action": action, "answers": answers,
            }
            try:
                with open(self.decisions_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError as exc:
                log.warning("could not write jev decision log: %s", exc)
        if self.overlay_path:
            try:
                with open(self.overlay_path, "w") as f:
                    f.write("jev nav: %s (dist %.0fcm, h%+.0fcm)" % (action, distance_from_launch_cm, height_dev_cm))
            except OSError:
                pass
        return int(action) if action is not None and action.lstrip("-").isdigit() else action


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

    def detect_frames(self, frames: List[bytes], workers: int = 6, strict: bool = False) -> List[tuple]:
        out = []
        for i, f in enumerate(frames):
            det = self.detect(f)
            scene = "simulated heading %d" % i
            out.append((det, {"scene": scene, "openness": "open", "path": "simulated open floor",
                              "decodable": True, "candidate_objects": []}, {"blocked": False}))
        return out

    def decide_navigation(self, headings: List[Dict], distance_from_launch_cm: float, max_distance_cm: float,
                          height_dev_cm: float = 0.0, max_height_dev_cm: float = 0.0) -> Optional[str]:
        # cycle through action kinds so a dry run exercises every code path
        open_idx = [i for i, h in enumerate(headings) if not h.get("blocked")]
        kinds = ["advance", "ascend", "descend"]
        kind = kinds[self.calls % len(kinds)]
        if kind == "advance" and distance_from_launch_cm < max_distance_cm and open_idx:
            return open_idx[self.calls % len(open_idx)]
        if kind == "ascend" and max_height_dev_cm > 0 and height_dev_cm < max_height_dev_cm:
            return "ascend"
        if kind == "descend" and max_height_dev_cm > 0 and height_dev_cm > -max_height_dev_cm:
            return "descend"
        return open_idx[0] if open_idx else None


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
