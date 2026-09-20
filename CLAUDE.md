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
check flight-time/battery limits (`check_limits()`) before each command.
`check_limits()` failures (battery-critical, max-flight-time) always abort
the mission immediately — but a command that errors is retried a couple of
times and then just *skipped* (mission keeps going) rather than aborting the
whole flight; pose is only updated on a command that actually succeeded.
Killing the process (any `SIGTERM`, not just Ctrl-C/`SIGINT`) is caught and
routed through the same abort→emergency-land→cleanup path as everything
else — `Mission.run()` installs a handler that turns `SIGTERM` into a
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

**Navigation has no fixed route — jev decides where and when** (`phase_search`
in `mission.py`). At each stop: burst-check for a person (`look()`, unchanged
from before); if none, ask jev a second typed question, `next_action`
(choice: `rotate`/`advance`/`stop`), reusing the same frame state the person
check just built (`JevDetector.last_state` — no extra frontend pass, one
extra jev round trip: `JevDetector.decide_navigation()`). `stop` (jev calling
this direction a dead end) is treated the same as `rotate`. The only hard
limits regardless of what jev picks: `mission.max_distance_from_launch_cm`
(no obstacle sensing — forward movement is blind, and dead-reckoning drift
compounds the further it roams, so `advance` stops being offered past this)
and `mission.return_buffer_s` (the search loop gives up once
`max_flight_s - return_buffer_s` has elapsed, so there's still time to fly
home and land instead of hard-aborting mid-search).

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
    endpoint to describe people directly as JSON, including a one-line
    `scene` description — the main signal `decide_navigation()` has to
    reason about open paths vs. dead ends. The offline frontends give jev
    almost nothing to go on for navigation, only for person detection.

  `Mission.look()` bursts `frames_per_heading` (4) detections per heading and
  requires **every** frame in the burst to individually clear
  `jev.min_confidence` before accepting a detection — a mean-based check let
  a door through once (0.84/0.81/0.03, mean 0.56, above threshold) that jev
  itself scored 0.03 in isolation; one flaky frame in an otherwise-empty
  burst no longer drags the average past threshold. A dropped frame (burst
  shorter than `frames_per_heading`) also fails the check outright rather
  than being judged on whatever partial subset succeeded. `make_detector()` also runs
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
counts/battery minimums, jev model/frontend/confidence threshold, and the
navigation limits (`max_distance_from_launch_cm`, `nav_forward_step_cm`,
`return_buffer_s`) described above. `power.ha_entity` must be the real HA
entity ID (verify via `GET /api/states/<id>`, not just a friendly
name/guess) — a wrong one won't error, it'll just silently do nothing.

`secrets.json` (gitignored; `secrets.example.json` shows the shape) holds
`JEV_API_KEY` (Vercel AI Gateway key, MyApi vault entry "Vercel api gateway")
and `HA_TOKEN` (Home Assistant long-lived access token, MyApi vault entry
"home assistant" — a JWT tied to the HA user, works against the local
`power.ha_url` regardless of it also being a Nabu Casa remote token);
`cli.py`'s `load_secrets()` reads it into `os.environ` and only
`setdefault`s — real env vars win.
