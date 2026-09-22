# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CLI tool that flies a DJI Tello drone through an autonomous person-search
mission: power on its plug, join the Tello's Wi-Fi, gate on SDK readiness,
take off, and search the space for a person — **jev itself decides where and
when to navigate** at each stop (rotate / advance / back out of a dead end),
not a fixed scripted route. On finding someone: photograph them, do a flip,
dead-reckon back to the launch point, and land. This is real hardware
control code, not a simulation — treat changes to flight logic, safety
gates, and abort paths with the caution that implies. This hardware has
proven fragile in practice (IMU faults, thermal shutdowns, a degrading
battery sagging hard under load) — a flight-command failure is as likely to
be a real hardware issue as a bug, don't assume it's the latter.

## Commands

```bash
./run.sh run --dry-run              # full mission simulated end-to-end, touches no hardware/network
./run.sh run                        # real mission, vision frontend (default)
./run.sh run --opencv               # real mission, offline OpenCV frontend
./run.sh run --skip-power --skip-wifi --no-flip --search-only
./run.sh power-on / power-off       # Tasmota plug via Node-RED (HA fallback)
./run.sh wifi                       # scan + connect to the TELLO AP
./run.sh probe --probes 6           # SDK gate + battery/telemetry, no flight
./run.sh land                       # send land (recovery if a mission left the drone airborne)
./run.sh detector-check             # verify jev is reachable through the gateway (no drone needed)
./run.sh calibrate --vision --positives dir1 --negatives dir2   # tune jev.min_confidence
```

There is no test suite, linter, or build step — `run.sh` invokes
`python3 -m tello_jev`. `--dry-run` (module-level `FakeTello` /
`FakeVideoStream` / `FakeDetector`) is the closest thing to a test harness;
use it to validate changes to `Mission` control flow without hardware.
`detector-check` and `calibrate` validate the jev pipeline without a drone.

Frontend/model flags (`--opencv`, `--vision`, `--frontend`, `--vision-model`)
are accepted both before and after the subcommand (see the `frontend_flags`
helper in `cli.py`).

## Architecture

**Everything is orchestrated by `Mission` (`mission.py`), a phase-sequenced
state machine**: `phase_power -> phase_wifi -> phase_sdk -> phase_launch ->
phase_search -> phase_photo_and_flip -> phase_return -> phase_land`, wrapped
in try/except/finally that guarantees cleanup (video stop, socket close,
Wi-Fi restore, power-off if configured) and an emergency landing on any
`TelloError`/`RuntimeError`/Ctrl-C. Every phase appends timestamped events to
`self.report`, which is dumped to `logs/report_<ts>.json` at the end of
`run()` regardless of outcome. Read this file first when tracing mission
behavior — the other modules are single-purpose services it calls into.

**Real vs. fake implementations share an interface, selected by `dry_run`**:
`Tello`/`FakeTello` (`tello.py`), `VideoStream`/`FakeVideoStream`
(`video.py`), `JevDetector`/`FakeDetector` (`detector.py`). When changing a
real class's public method signature, update its fake counterpart too —
`Mission` calls them interchangeably and has no other branching on `dry_run`
inside the phase methods themselves.

**Flight safety gate (from the `tello-flight-recovery` policy)**: no flight
command may be sent before `command` acks `ok`. This is enforced in
`Tello.flight()`, which raises `TelloError` if `self.ready` is false — the
readiness flag is only set by `wait_ready()`'s successful probe loop.
`Mission.fly()` wraps every movement to also track dead-reckoned pose and
call `check_limits()` before each command; pose is only updated on a command
that actually succeeded.

**Landing policy — the drone comes down for exactly three reasons**: the
person was found (photo → flip → return → land), the battery reached
`sdk.min_battery_land` (`MissionStop` → land *in place*, since the remaining
charge is worth more as a controlled descent than a return leg), or the
operator interrupted (Ctrl-C/`SIGTERM`). **Errors explicitly do not land
it.** A command that errors is retried a couple of times and then skipped
(`fly()` returns `False`); an error that escapes a whole phase is logged and
the search loop in `run()` simply resumes. Running out of viable directions
isn't a landing either — `phase_search` returns `None` and `reposition()`
nudges the drone somewhere new. There is no search *time* budget and no
consecutive-command-failure circuit breaker (it used to abort→emergency-land
after N failures; if the link is truly dead that abort couldn't send `land`
either, so it only cost otherwise-healthy flights — the count is still
tracked and logged). `check_limits()` reads the battery from the passive
state listener (no round trip) and requires **two consecutive** reads below
the floor, because this pack sags hard under load; `max_flight_s` is now
only a backstop for when battery telemetry is unreadable, since otherwise
nothing would ever bring the drone down.
Killing the process (any `SIGTERM`, not just Ctrl-C/`SIGINT`) is caught and
routed through the abort→emergency-land→cleanup path — `Mission.run()` installs a handler that turns `SIGTERM` into a
`KeyboardInterrupt` for exactly this reason; without it a killed process
would skip straight past `except`/`finally` and could leave the drone
airborne with no landing attempt.

**Position tracking is dead reckoning, not real telemetry**: `Pose` in
`mission.py` integrates forward/back/left/right/rotate commands into
`(x, y, yaw)`; the Tello has no indoor positioning. `phase_return` uses
`Pose.home_vector()` to compute a single turn + a chunked-forward path (≤300
cm per `forward` command) back to the launch point. `Mission.maintain_height()`
similarly corrects barometer drift against the post-takeoff target height
(SDK `up`/`down` need ≥20cm to be valid, so smaller drift is left alone).

**Navigation has no fixed route, and the whole circle is captured before
anything is judged** (`phase_search` in `mission.py`). One search cycle is
three separable steps:

1. `scan_circle()` — rotate a full circle grabbing **one** frame per heading
   (`scan_step_deg`, 60° → 6 headings), and nothing else. Pure mechanics, no
   network. It returns `[(yaw_at_capture, jpeg)]`, recording the *live pose*
   at each capture rather than an assumed `i*scan_step`, so a rotate the
   drone skipped (routine — "No valid imu") can no longer silently mislabel
   every heading after it. Turns back to a heading are computed from live
   pose at decision time.
2. `JevDetector.detect_frames()` — judge the whole circle **concurrently**
   (`detect_workers` threads). Each frame costs two sequential HTTP round
   trips (vision model, then jev), and doing that inline per heading is what
   made a 360° scan take ~70 s of a ~150 s battery; run across headings they
   overlap into roughly one frame's latency for the circle (measured 4.1×
   on a 6-frame circle). Per-call state comes back in the returned tuple
   `(Detection, state, blocked)` rather than on `self`, because
   `last_state`/`last_blocked` can't be shared by concurrent calls (`detect()`
   still sets them for the sequential callers: `calibrate`, `detector-check`).
3. One jev call — `decide_navigation(headings, ...)` — compares every heading
   and returns either the index of the best one to advance into, or
   `'ascend'`/`'descend'`, or `None` (nothing viable here). `phase_search`
   turns to face the chosen heading and moves `mission.nav_forward_step_cm`
   forward; the next iteration starts a fresh circle from the new spot.
   `ascend`/`descend` move `mission.nav_vertical_step_cm` and update
   `Mission.target_height_cm` on success — `maintain_height()` corrects drift
   against whatever this current value is, not the original post-takeoff
   height, so it doesn't fight a deliberate height change back down.

**Detection is two-stage by design: a sensitive scan, then a skeptical
confirm.** The two stages ask jev *different* questions
(`JevDetector.SCAN_Q` / `STRICT_Q`, selected by `detect_one(strict=)`)
because they want different things. Sweeping a room wants recall — the
frontend is told to report a bare leg or an arm behind a door as a person,
since missing someone is the one unrecoverable failure and a false alarm
costs seconds. Committing the mission to a heading wants precision, so
`Mission.confirm_person()` turns back to the flagged heading, takes
`confirm_frames` (3) **fresh** frames, asks the skeptical question, and
requires every frame to individually clear `jev.confirm_min_confidence`
(0.60, higher than the scan's `min_confidence`). The frames are judged
concurrently, so the 3-frame confirm costs about what one frame used to.

Measured on frames from the 2026-09-21 flight: a real person seen only as
bare legs at the frame edge scores 0.89 on the scan and 0.73–0.78 strict; a
tan blob in a doorway scores 0.82–0.86 on the scan and drops to 0.52–0.71
strict; an empty room is 0.02–0.08 throughout. The "every frame must clear"
rule is what actually separates the last two — the blob cleared 0.60 on one
frame out of four. Don't make `STRICT_Q` harsher without re-measuring: an
earlier wording ("is a human DEFINITELY visible") rejected the real person
too (0.39–0.45). The thing to be strict about is human *form*, not how much
of the body happens to be in frame.

A heading only makes it into jev's choice set if it isn't a deterministic
dead end — not left purely to jev's judgment of the scene text. Two gates:
`JevDetector._check_blocked()` computes a cheap "mostly flat and bright"
signal (>60% near-white pixels *and* low Canny edge density) directly from
the frame, and the vision frontend's own `openness == "blocked"`. The first
is the strong gate but **needs `opencv`+`numpy` importable and silently does
nothing without them — which is the case on the current host** (no cv2, no
numpy, no pip; `bright_frac`/`edge_density` come out `null` in the decisions
log, and the `features`/`opencv` frontends can't run at all). That's why the
vision frontend's `openness` exists: weaker, since it's still a model's
judgment, but made while *looking at the image* rather than by jev reading a
one-line string. Installing `opencv-python-headless` + `numpy` re-arms the
strong gate with no code change.

The only hard limits regardless of what jev picks:
`mission.max_distance_from_launch_cm` and `mission.max_height_deviation_cm`
(no obstacle sensing — forward/vertical movement is blind, and
dead-reckoning drift compounds the further/higher it roams, so a heading
within one step of the distance ceiling is never offered as an `advance`
candidate, and `ascend`/`descend` stop being offered past their own
ceiling). Nothing else constrains it: there is no search time budget — the
search keeps going until a person is found or the battery floor is reached.

**`Tello.flight("stop")` is not sent** — this firmware returns `unknown
command: stop` for it on every single flight tested; it was a no-op call
right after takeoff (the Tello already hovers on its own) and is simply
removed rather than worked around.

**Detection is a two-stage pipeline: a pluggable *frontend* + jev as the
fixed judge** (`detector.py`). jev (`typesafe-ai/jev`, called through the
Vercel AI Gateway's `/v4/ai/evaluation-model` endpoint) is text-only, so a
frontend turns each frame into a JSON "state" that jev answers typed
questions about (a boolean `person_visible` probability, a `choice` of which
`candidate_objects` index is the target, and — for navigation — a `choice`
of `next_action`):
  - `features` (`FrameState`): classical OpenCV stats only — skin-tone blobs,
    motion vs. background subtractor, edge density, brightness grid. Offline.
  - `opencv` (`HogFrontend(FrameState)`): adds HOG+SVM pedestrian-detector
    candidates on top of the features. Offline.
  - `vision` (`VisionFrontend`, default): calls a fast vision-language model
    (default `openai/gpt-4o-mini`) via the gateway's `/v4/ai/language-model`
    endpoint to describe people directly as JSON, plus the layout fields
    `scene`/`openness`/`path`/`exits` — the only real signal
    `decide_navigation()` has to reason about open paths vs. dead ends, and
    free, since the call is being made for detection anyway. The prompt
    explicitly counts a partly-visible body (bare legs, feet, an arm, a
    head, someone behind a door) as a real person: an earlier version
    described the actual search target as `is_real_person=false`
    ("partially visible"), and jev then scored that frame 0.18. The offline
    frontends give jev almost nothing to go on for navigation, only for
    person detection.

  The offline frontends need `opencv`+`numpy`, which are **not installed on
  the current host** — `vision` is the only frontend that actually runs here.
  The scan/confirm split and the every-frame-must-clear rule are described
  under Navigation above. `make_detector()` also runs
  `jev.healthy()` as a startup sanity probe; if jev doesn't answer, the
  mission refuses to take off (there is deliberately no fallback detector for
  the live mission — only for `--dry-run`). Every jev call (detection and
  navigation) is appended to `logs/jev_decisions_<run_id>.jsonl`
  (`JevDetector._log_decision`) — full candidate/question/answer trail for
  reviewing *why* jev did what it did, and the single latest one-line
  decision also lands in `logs/overlay_bottom.txt` (see Recording below).

**Video capture has real hardware quirks, not just plumbing** (`video.py`).
`VideoStream` runs one ffmpeg process with two outputs sharing the same
decoded input: `latest.jpg` (refreshed at `fps`, read by the detector) and,
optionally, an mp4 recording. Three non-obvious fixes that took real
debugging to find, don't casually revert them:
  - `-f h264` is required — without it ffmpeg has to auto-probe the raw
    H.264-over-UDP stream, which can take longer than the startup window.
  - The video listener must be `start()`-ed **before** `streamon` is sent
    (`phase_launch` does this in that order deliberately) — the Tello emits
    its SPS/PPS parameter sets once at stream start, and a listener that
    isn't bound yet misses them permanently (every later frame then decodes
    as "corrupt"/"non-existing PPS referenced"). A redundant `streamon`
    while already streaming does *not* re-emit them; only an off→on toggle
    does, which is why `phase_launch` retries via `streamoff`+`streamon` (via
    `Mission._streamon()`, which also verifies the ack) rather than just
    resending `streamon`.
  - No `-fflags nobuffer -flags low_delay` — those disable ffmpeg's
    reordering buffer, and slightly out-of-order UDP packets (routine over
    Wi-Fi) then decode as corrupt frames. Only 3fps snapshots are needed, not
    true minimum latency.
  - ffmpeg's stderr goes to a file (`logs/stream/ffmpeg.log`), never a plain
    `subprocess.PIPE` — an undrained pipe can fill and deadlock ffmpeg's own
    writes if it logs enough (exactly what happens during a PPS-reference
    warning storm), which silently manifested as "no video frame" timeouts
    with zero diagnostic output.
  - The Tello encodes non-full-range (studio) YUV; the mjpeg encoder used for
    `latest.jpg` refuses that outright without `-strict unofficial`.

**Recording + on-video text overlays**: when not `--dry-run`, the recording
output additionally burns in two live text overlays via ffmpeg `drawtext`
(`reload=1`, so it re-reads the file every frame) — this switches that
output from a lossless stream-copy to a `libx264` re-encode. `log.py`'s
stdout/stderr tee keeps `logs/overlay_top.txt` overwritten with the latest
console line (so `run.sh`'s own output ends up burned into the video, top of
frame); `JevDetector` keeps `logs/overlay_bottom.txt` overwritten with its
latest decision (bottom of frame). Both are *fixed* filenames (not per-run
timestamped) that get overwritten in place — only ever the single most
recent line is shown, not the whole growing log.

**External side-effect boundary**: `power.py` (HA REST call to `switch.tello`
first, Node-RED inject as fallback) must run *before* the host's Wi-Fi
switches to the Tello AP — LAN becomes unreachable after that, which is why
`phase_power` always precedes `phase_wifi`. HA's service-call endpoint
returns 200 with an empty change list if `entity_id` matches nothing at all
— a wrong/misspelled entity otherwise silently "succeeds" — so `_ha_switch`
checks the entity exists first rather than trusting the HTTP status. `wifi.py`
shells out to `nmcli`; it remembers the host's prior connection
(`current_connection`) so `phase_wifi`/`Mission.run`'s `finally` can restore
it after the mission, disables `autoconnect` on any Tello profile it
connects to (NetworkManager's own autoconnect otherwise races explicit
connect attempts — and each other — for the Tello's tiny DHCP pool), and
explicitly `nmcli device disconnect`s before/after each attempt so a
stuck/still-activating attempt from a previous run can't hold a lease
hostage. `wifi._run()` never lets a hung/timed-out `nmcli` call raise past
it (catches `subprocess.TimeoutExpired`) — this class of hang is common
enough with this drone's Wi-Fi that letting it propagate has previously
crashed mid-cleanup and skipped writing the mission report entirely.

## Configuration

`config.json` (loaded once in `cli.py`, mutated in place by CLI flags for
frontend/model overrides) drives everything: drone network params, SDK probe
counts/battery minimums, jev model/frontend/confidence thresholds, and the
navigation limits (`max_distance_from_launch_cm`, `nav_forward_step_cm`,
`nav_vertical_step_cm`, `max_height_deviation_cm`) described above.

The knobs that set the search's pace and its two confidence bars:
`scan_step_deg` (60 → 6 headings/circle), `confirm_frames` (3),
`detect_workers` (6), `frame_settle_s`/`scan_settle_s`/`frame_gap_s` (dead
time per frame — these are per-heading, so they multiply), and
`jev.min_confidence` (0.45, the sensitive scan bar) vs
`jev.confirm_min_confidence` (0.60, the skeptical confirm bar). Search
altitude comes from `takeoff_climb_cm` (70): the camera is fixed and angled
down, so at the old ~90 cm hover the frame was mostly carpet and a person
only ever entered it as feet — flying at ~1.6 m is what puts torsos in
frame, and it is the single highest-leverage setting for finding people at
all. `sdk.min_battery_land` (15) is the only battery number that ends a
flight; `min_battery_flip` is now the same 15 rather than 55, because this
pack sags into the high teens under search load and a 55 gate meant the flip
was skipped on every real flight that found someone (the firmware still
refuses a flip it considers unsafe, and that refusal is logged, not fatal). `power.ha_entity` must be the real HA
entity ID (verify via `GET /api/states/<id>`, not just a friendly
name/guess) — a wrong one won't error, it'll just silently do nothing.

`secrets.json` (gitignored; `secrets.example.json` shows the shape) holds
`JEV_API_KEY` (Vercel AI Gateway key, MyApi vault entry "Vercel api gateway")
and `HA_TOKEN` (Home Assistant long-lived access token, MyApi vault entry
"home assistant" — a JWT tied to the HA user, works against the local
`power.ha_url` regardless of it also being a Nabu Casa remote token);
`cli.py`'s `load_secrets()` reads it into `os.environ` and only
`setdefault`s — real env vars win.
