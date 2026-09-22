# tello-jev-mission

Autonomous Tello mission: power the drone on, join its Wi-Fi, gate on SDK
readiness, take off, and search the 1st floor for a person -- **jev itself
decides where and when to navigate** (rotate to look elsewhere, advance into
open space, or back out of a dead end), not a fixed scripted route.
Photograph the person, do a flip, dead-reckon back to the launch point and
land. Built on the `tello-flight-recovery` skill (recovery-first, never send
flight commands before `command -> ok`).

## Requirements

- A host **with Wi-Fi** (the laptop, interface `wlp3s0`). The home server
  has no wireless adapter and cannot control the drone.
- Python 3.8+, `ffmpeg`, `nmcli`.
- `pip install -r requirements.txt` (OpenCV/numpy only needed for the
  offline fallback detector).
- Power-on must happen while the host still has LAN access; the tool does
  this before switching to the TELLO network and restores your Wi-Fi after.
- `cp secrets.example.json secrets.json` and fill in real values (gitignored,
  never committed) — see below for what each key is.
- The drone's own Wi-Fi (`drone.ssid_prefix`/`ssid_preferred`), the plug's HA
  entity (`power.ha_entity`, `power.ha_url`), and Node-RED inject URLs
  (`power.on_url`/`off_url`) in `config.json` are all specific to one
  physical setup — treat them as placeholders to replace, not defaults that
  will work as-is.

## The "jev" model and its frontends

`typesafe-ai/jev` is the decision model, called through the Vercel AI Gateway's
native evaluation endpoint (`POST https://ai-gateway.vercel.sh/v4/ai/evaluation-model`).
The key lives in `secrets.json` as `JEV_API_KEY` (MyApi vault entry "Vercel api gateway").

Per TypeSafe's docs jev accepts **text only** ("Images, audio, and video are not
supported (yet)"), so a *frontend* turns each frame into text state that jev
judges with two typed questions: a **boolean** (probability a real person is
visible) and a **choice** (which candidate is the target). Pick the frontend
with a flag (before or after the subcommand) or `jev.frontend` in config:

| flag | frontend | what feeds jev | cost / speed |
|---|---|---|---|
| `--vision` (default) | `vision` | `jev.vision_model` (default `openai/gpt-4o-mini`) describes the frame as JSON via the gateway's `/v4/ai/language-model` endpoint | ~1.7 s/frame, gateway tokens |
| `--opencv` | `opencv` | OpenCV HOG+SVM pedestrian detector windows plus the classical features below | ~0.3 s/frame, offline |
| `--frontend features` | `features` | classical statistics only: brightness grid, edges, skin-tone blobs, motion, geometry of candidate boxes | ~0.2 s/frame, offline |

Calibration on Frigate driveway snapshots (4 with a person, 3 without):

| frontend | person frames (jev p) | empty frames (jev p) |
|---|---|---|
| vision (gpt-4o-mini) | 0.73 – 0.81 | 0.03 – 0.04 |
| opencv | 0.35 – 0.70 | 0.07 – 0.70 (cars fooled it) |
| features | 0.35 – 0.74 | 0.07 – 0.77 (cars fooled it) |

`gpt-4.1-nano` is faster but invented people in an empty room; `gemini` and
`claude` models reject inline images on this endpoint. There is **no fallback**:
if jev does not answer a probe before launch the mission refuses to take off.

Re-tune on your own drone frames: `./run.sh calibrate --vision --positives dir1 --negatives dir2`,
then set `jev.min_confidence` (0.45 now, the sensitive scan bar) between the
two means; `jev.confirm_min_confidence` (0.60) is the stricter bar the
confirm burst has to clear.
`calibration/negatives/` holds a real false positive from this rig (a closed
door jev's vision model called a person) — add your own tricky negatives
there as you find them.

Detection runs in two stages that ask jev **different questions**, because
they want different things:

- **Scan** (`SCAN_Q`, one frame per heading, bar `jev.min_confidence` 0.45)
  wants recall. The vision prompt explicitly counts a partly visible body —
  bare legs, feet, an arm, someone behind a door — as a real person, because
  missing someone is the one unrecoverable failure and a false alarm costs
  seconds. An earlier prompt called the actual search target
  `is_real_person=false` ("partially visible") and jev scored that frame
  0.18; with the current one the same frame scores 0.83.
- **Confirm** (`STRICT_Q`, `confirm_frames` = 3 **fresh** frames, bar
  `jev.confirm_min_confidence` 0.60) wants precision, and every frame must
  individually clear the bar — not the average. A mean-based check once let
  a closed door through (0.84/0.81/0.03, mean 0.56) that jev scored 0.03 in
  isolation.

Measured on frames from a real flight: a person seen only as bare legs at
the frame edge scores 0.89 scan / 0.73–0.78 strict; a tan blob in a doorway
scores 0.82–0.86 scan but drops to 0.52–0.71 strict; an empty room stays
0.02–0.08. The every-frame rule is what separates the last two — the blob
cleared 0.60 on one frame out of four. Don't harden `STRICT_Q` without
re-measuring: a harsher wording ("is a human DEFINITELY visible") rejected
the real person too, at 0.39–0.45.

## Navigation: jev decides where and when

There is no fixed patrol route, and the drone doesn't react one step at a
time to whatever it happens to be facing -- it looks at its **whole
surroundings before committing to a direction**. At each stop:

1. `scan_circle()` rotates a full 360°, grabbing **one** frame per heading
   (`scan_step_deg`, 60° → 6 headings) and doing nothing else -- no network
   at all, ~2 s per heading.
2. `detect_frames()` judges the whole circle **concurrently**. Each frame
   costs two sequential HTTP round trips (vision model, then jev), so doing
   it inline per heading is what used to make a scan take ~70 s of a ~150 s
   battery; overlapped, the circle costs about one frame's latency (measured
   4.1× faster on a 6-frame circle). A heading that looks like a person is
   re-checked by `confirm_person()` with a strict fresh burst before the
   mission commits to it.
3. If nobody was found, **one** jev call compares every heading scanned --
   `next_action`'s reasoning is now "which of these directions is the most
   promising", not "should I move from here" -- and jev picks a single
   heading to commit to, or `ascend`/`descend` if none of the headings beat
   gaining/losing height for a better vantage.
4. The drone turns to face the chosen heading and moves `nav_forward_step_cm`
   forward in a straight line, then the next stop starts a fresh 360° scan.

`ascend`/`descend` move `nav_vertical_step_cm` up/down (e.g. seeing over a
low obstacle, or checking whether someone is seated/lying below the current
view) -- the drone's `target_height_cm` (what `maintain_height()` holds
against barometer drift) moves with a deliberate ascend/descend, so
drift-correction doesn't fight jev's own choice back down to the original
hover height. If every heading is blocked and height is already maxed in
both directions, the drone gives up on that spot, repositions and keeps
searching -- running out of ideas is not a reason to land.

A heading is only offered to jev as a candidate to advance into if it's not
a clear, deterministic dead end -- not left purely to jev's reading of a
scene description: a frame that's mostly a flat, bright surface (a wall or
closed door filling the view) is detected directly
(`JevDetector._check_blocked()`: >60% of pixels near-white *and* very low
edge density, cheap to compute, independent of frontend) and that heading is
simply excluded from the choice, regardless of what jev might have inferred
from the frontend's text. (Needs `opencv-python-headless` + `numpy`
installed -- optional for the vision frontend otherwise -- to actually run;
degrades to "never blocks" without them, silently. **They are not installed
on this host**, so the second gate below is currently the only one doing
anything.)

Second gate: the vision frontend reports `openness` (`open`/`partial`/
`blocked`) per frame alongside `scene`, `path` and `exits`, and a heading it
calls `blocked` is excluded too. Weaker than the opencv check -- it is still
a model's judgement -- but unlike jev reading a one-line scene string, it is
made while looking at the pixels. Those same fields are what jev is given
for each heading, instead of a single vague sentence.

Three things stay hard limits no matter what jev picks, since the Tello has
no obstacle sensing (forward/vertical movement is blind) and no GPS (dead
reckoning drift gets worse the further/higher it roams):

- `max_distance_from_launch_cm`: once the dead-reckoned distance from the
  launch point would come within one step of this, no heading is offered as
  an `advance` candidate -- only `ascend`/`descend` remain, if height allows.
- `max_height_deviation_cm`: how far `target_height_cm` may drift from the
  height set right after takeoff, in either direction; `ascend`/`descend`
  aren't offered past that ceiling/floor.
There is deliberately no *time* limit on the search: only the battery floor
ends a flight (see Landing policy below).

## Usage

```bash
./run.sh run --dry-run          # full simulation, no hardware touched
./run.sh power-on               # Tasmota plug via Node-RED (HA fallback with HA_TOKEN)
./run.sh wifi                   # scan + connect TELLO-C4A2A1(-EXT)
./run.sh probe                  # SDK gate + battery/telemetry
./run.sh run                    # the real mission (vision frontend)
./run.sh run --opencv           # offline OpenCV HOG frontend instead
./run.sh run --skip-power --skip-wifi --no-flip
./run.sh calibrate --positives ~/frames/person --negatives ~/frames/empty
./run.sh land                   # if something left the drone airborne
./run.sh power-off
```

## Mission sequence

1. power on plug (HA `switch.<entity>` first, Node-RED inject fallback),
   wait `boot_wait_s` (45 s)
2. scan/connect TELLO SSID, remember previous connection
3. probe `command` up to 15x; abort if never `ok`; abort if battery < 25 %
4. start the video listener **before** `streamon` (order matters — see
   Recording below), require a first frame before takeoff
5. `takeoff` -> `up 70`; remember this as the target hover height (~1.6 m,
   so the downward-angled camera frames torsos rather than carpet)
6. search loop, per cycle: correct height drift, capture a 360 deg circle of
   frames (one per heading, no network), judge the whole circle concurrently
   in one batch, confirm any hit with a strict 3-frame burst, else make one
   jev call to pick the heading to fly into (see Navigation above). Repeats
   until a person is found or the battery floor is reached
7. on person: center on bbox, save `photos/person_<ts>.jpg` + `.json`
   (detection, pose, telemetry); `up 30`, `flip b`, `down 30`
8. return to launch by dead reckoning (turn to face home, forward in <=300 cm
   chunks, restore heading), `land`, `streamoff`
9. restore Wi-Fi, write `logs/report_<ts>.json`

**Landing policy: the drone lands for exactly three reasons** — the person
was found (photo -> flip -> return -> land), the battery reached
`sdk.min_battery_land` (15 %, which lands it *in place* rather than spending
the remaining charge on a return leg), or the operator interrupted. **Errors
do not land it.** A movement command that errors is retried a couple of
times and then **skipped** (pose is only updated on a command that actually
succeeded); an error escaping a whole phase is logged and the search
resumes; running out of viable directions just repositions and keeps
looking. `check_limits()` needs *two consecutive* battery reads below the
floor, since this pack sags hard under load, and `max_flight_s` (300) is now
only a backstop for when battery telemetry is unreadable. Killing the
process (any signal, not just Ctrl-C) is
caught and routed through the same land/cleanup path — see Development log.

## Recording

Every real (non-`--dry-run`) flight also saves the full flight video to
`recordings/flight_<run_id>.mp4` — separate from `logs/stream/latest.jpg`,
which stays the low-latency feed the detector actually reads. Two live text
overlays get burned into the recording: the console/log's latest line at the
top, jev's latest decision (detection or navigation) at the bottom — both
just the single most recent line, not the whole scrolling log. Every jev
call, in full, also lands in `logs/jev_decisions_<run_id>.jsonl` for
after-the-fact review of exactly what it saw and why it decided what it did.

## Safety notes

- Dead reckoning drifts; keep `max_distance_from_launch_cm` conservative for
  your space and keep it clear of obstacles -- forward movement is blind.
- The flip needs ~1 m of clearance above and around the drone.
- If the SSID never appears or `command` never acks: power-cycle the plug,
  wait 30 s, retry (`./run.sh power-off && sleep 8 && ./run.sh power-on`).
- This specific rig's hardware proved fragile in testing (IMU faults,
  thermal shutdowns, a battery that sagged hard under load) — a flight
  command coming back with an error is as likely to be real hardware trouble
  as a bug. See Development log for the exact symptoms encountered.

## Development log

A working session's worth of real hardware debugging, roughly in the order
it happened — kept here because most of it isn't derivable from the code
alone, and the failure modes are worth recognizing if they recur.

**Power-on was a silent no-op.** `config.json` pointed `power.ha_entity` at
`switch.tasmota`, which doesn't exist on this Home Assistant instance — the
real entity is `switch.tello`. Home Assistant's service-call endpoint
returns HTTP 200 with an *empty* change list when `entity_id` matches
nothing at all, so the wrong entity looked like success. `power.py` now
verifies the entity actually exists (`GET /api/states/<id>`) before trusting
a 200. Once pointed at the right entity, it turned out to be a **SwitchBot
Bot** — a small robot that physically presses a button, not a mains relay
(the Tello runs on an internal battery with a physical power button, so this
makes sense in hindsight) — and its physical alignment against the button
needed to be checked/adjusted by hand; nothing about that is fixable from
software.

**Wi-Fi connects were flaky and one crash skipped the whole mission
report.** NetworkManager's own `autoconnect` on the Tello's saved profile
was racing this tool's own explicit connect attempts (and, worse, racing
itself across separate runs) for the Tello's tiny DHCP address pool —
`wifi.py` now disables `autoconnect` on any Tello profile it connects to and
explicitly disconnects before/after each attempt. Separately, a hung
`nmcli` call once raised an uncaught `subprocess.TimeoutExpired` out of the
mission's own `finally` block (during Wi-Fi restore), which is bad enough
that this Python process died with a traceback and never wrote a report —
`wifi._run()` now catches that itself, logs a warning, and moves on.

**The video stream had three independent, real bugs**, each individually
enough to cause "no video frame" timeouts with no diagnostic output:

- Missing `-f h264` on the ffmpeg input made it auto-probe the raw
  H.264-over-UDP stream, which can take longer than the startup window.
- `streamon` was being sent *before* the video listener was even bound. The
  Tello emits its H.264 SPS/PPS parameter sets once at stream start; a
  listener that isn't ready yet misses them permanently, and every later
  frame decodes as "corrupt"/"non-existing PPS referenced". Fixed by
  starting the listener first, and — since even that is a race — the
  mission also retries once via a genuine `streamoff`→`streamon` toggle
  (confirmed: a *redundant* `streamon` while already streaming does not
  re-emit the parameter sets; an actual off→on transition does).
- `-fflags nobuffer -flags low_delay` were disabling ffmpeg's packet
  reordering buffer, so ordinary out-of-order UDP packets (routine over
  Wi-Fi) decoded as corrupt frames. Removed — 3fps snapshots don't need true
  minimum latency.

A fourth issue was purely diagnostic: ffmpeg's stderr went to a plain
`subprocess.PIPE` that nothing ever drained, which can fill up and deadlock
ffmpeg's own writes if it logs enough — exactly what happens during a
PPS-reference warning storm. It now goes to `logs/stream/ffmpeg.log`
instead, which is what made the above bugs possible to actually diagnose.

**A firewall was silently dropping the drone's own video/telemetry.**
`ufw`'s default-deny-incoming policy was dropping the Tello's *unsolicited*
UDP pushes on the video (11111) and state (8890) ports — the command channel
worked fine since it's a request/reply the host initiates, but nothing had
ever told the host to accept unprompted incoming packets on those ports.
Fixed with two scoped `ufw allow ... on wlp3s0` rules.

**Recording needed both a real recording and, once added, two rounds of
bugfix.** The mission previously had no video artifact at all beyond the
detector's own low-res snapshots — added a proper
`recordings/flight_<run_id>.mp4` (see Recording above). Burning in the live
text overlays required switching that output from a lossless stream-copy to
a `libx264` re-encode, which needs real time after `SIGTERM` to flush its
encoder and write the mp4 trailer (moov atom) — first fix bumped the
shutdown grace period from 3s to 10s, which worked for short clips but still
left longer/heavier flights (a 159s flight with a flip and several
`advance`s) with a missing moov atom and an unplayable file. The real cause
was `-movflags +faststart`: it makes ffmpeg rewrite the *entire* file at
shutdown to relocate the moov atom to the front, and that rewrite scales
with file size — no fixed grace period was ever going to cover every flight
length. Removed entirely; these are local-only files, never streamed, so
faststart bought nothing but risk.

**A flight-command error used to abort and land immediately, no matter
what.** Real flights hit both a `Motor stop` error (once right after
takeoff — the drone had visibly tipped; once well into a stable hover after
a full minute of normal flight, telemetry showing internal temps around
86-88°C, plausibly a thermal protection trip) and, once, `No valid imu`,
which failed **every single** rotation for an entire flight. By request,
`Mission.fly()` now retries a failed command a couple of times and then just
skips that one action (mission keeps going) rather than aborting the whole
flight — `check_limits()` (battery-critical, max-flight-time) is
deliberately *not* covered by this and still aborts immediately, since those
represent genuinely dangerous states rather than one bad command.

**Killing the process used to leave the drone flying with no landing
attempt.** Stopping the mission process externally (as opposed to Ctrl-C)
sends `SIGTERM`, which Python does not catch by default — it skips straight
past the `except`/`finally` cleanup that calls `emergency_land()`. Confirmed
live: the rotors were still spinning after a kill. `Mission.run()` now
installs a `SIGTERM` handler that raises `KeyboardInterrupt`, routing any
kill through the exact same abort→emergency-land→cleanup path as Ctrl-C.

**Navigation was rebuilt from a fixed route to jev-driven, by request** --
first to a one-step-at-a-time model (`next_action`: `rotate`/`advance`/`stop`
reacting to whatever heading the drone happened to already be facing), then,
on further request, to the full-360°-scan-then-commit model described in
Navigation above, once it became clear "look at one heading, maybe move,
repeat" wasn't actually using the fact that the drone can see everything
around it before choosing a direction. Height control (`ascend`/`descend`)
and the deterministic wall/door block-check were added the same round, for
the same reason as the detection burst fix below: a couple of things were
still being left to jev's own judgment of a scene description that a cheap,
explicit check could decide correctly and consistently instead.

**Other real hardware trouble encountered along the way, for context**: an
overheat that auto-shut the drone off entirely, a battery that dropped from
27% to 15% in under 30 seconds of ordinary hover-and-rotate (a strong signal
of a degrading battery sagging under load rather than normal drain), and the
IMU/Motor-stop faults above. None of these are software bugs; they're why
the safety-net changes above (retry-then-skip, SIGTERM handling) mattered in
the first place.

The first fully clean end-to-end run after all of the above: power on ->
Wi-Fi -> SDK gate -> takeoff -> jev chose `advance` twice -> found a person
(jev p=0.84 on the best frame) -> photographed -> flew home by dead
reckoning -> landed. Exit code 0.
