# Use case: one autonomous search flight, start to finish

A real flight of the mission in `../tello_jev/`, on 2026-09-21, with every
artifact it produced. No simulation, no staging beyond a person standing in a
room: the drone powers on its own plug, joins the Tello's Wi-Fi, takes off,
decides for itself where to look, finds the person, photographs them, flips,
and lands.

| | |
|---|---|
| `flight.mp4` | the flight as the drone saw it, with live overlays burned in (7 MB, re-encoded from the 35 MB original) |
| `person.jpg` / `person.json` | the photo it took on finding the person, plus detection, pose and telemetry |
| `report.json` | the mission report: every phase event, timestamped from takeoff |
| `jev_decisions.jsonl` | all 16 jev calls — the candidates, the questions asked, the probabilities returned |
| `before/` | the same mission before the changes described below, for comparison |

The two text overlays in the video are live: the top line is the console log,
the bottom line is jev's latest decision. `jev p=0.78 over 1 candidates
(vision, strict)` at the bottom of the frame is the confirm stage running.

## What happened

```
  t=0.0s   takeoff, climb to ~1.3 m
  t=21.5s  circle 1 captured and judged: [0.06 0.07 0.08 0.04 0.03 0.07] in 2.9s
  t=21.9s  jev picks heading 2 of the 3 that weren't dead ends -> fly 1 m
  t=46.5s  circle 2: [0.10 0.06 0.05 0.89 0.92 0.04] in 2.7s
  t=46.5s  candidate person at 0deg -- turning back to confirm
  t=53.1s  strict confirm burst: 0.78 0.77 0.71, all >= 0.60 -> person found
  t=56.9s  centered, photo saved
  t=66.2s  FLIP done
  t=67-70s "Motor stop" on four commands in a row -- skipped, kept flying
  t=70.1s  land
```

Person found in **53 s**, flip completed, total flight **73 s**.

## How the drone decides where to go

There is no patrol route. At each stop it captures a full 360° circle — one
frame per heading, six headings — judges the whole circle at once, and makes a
single jev call comparing every direction. From `jev_decisions.jsonl`, circle 1:

| heading | turn | openness | scene |
|---|---|---|---|
| 0 | +60° | **blocked** | a narrow hallway with two closed doors |
| 1 | +120° | partial | a cluttered desk with a computer |
| 2 | −180° | partial | a gaming setup with a chair and desk |
| 3 | −120° | **blocked** | two windows with blinds partially closed |
| 4 | −60° | **blocked** | a blank wall with a whiteboard |
| 5 | +0° | partial | a cluttered room, "towards the doorway" |

Three headings were excluded as dead ends before jev ever saw them — a wall,
closed doors, a window. jev chose between the remaining three (p = 0.31 / 0.35
/ 0.32) and picked heading 2. One step later, the person was in frame.

## Detection is two questions, not one

The same frame is judged twice, and the two stages deliberately ask jev
*different* things:

- **Scan** — one frame per heading, sensitive. The vision prompt counts a
  partly visible body (bare legs, an arm, someone behind a door) as a person,
  because missing someone is the only unrecoverable failure and a false alarm
  costs seconds. Circle 2 scored 0.89 and 0.92 on two adjacent headings.
- **Confirm** — three *fresh* frames, skeptical, and **every** frame must clear
  the higher bar (0.60) on its own. Here: 0.78 / 0.77 / 0.71.

That split is doing real work. On frames from the earlier flight in `before/`:

| | scan | strict | outcome |
|---|---|---|---|
| real person, only bare legs at the frame edge | 0.89 | 0.73–0.78 | confirmed |
| tan blob in a doorway | 0.82–0.86 | 0.52–0.71 | **rejected** |
| empty room | 0.03–0.08 | 0.02–0.05 | rejected |

The blob cleared 0.60 on one frame out of four — the every-frame rule is what
rejected it, not the threshold alone.

## Errors are not a reason to land

Immediately after the flip, the drone returned `Motor stop` to four consecutive
commands (`down 30`, `ccw 36`, `forward 100`, `cw 24`). Each was retried twice,
skipped, and the flight continued; dead-reckoned pose was left untouched for
the moves that never happened, so the report ends honestly at
`{x: -50, y: -87}` rather than claiming it made it home.

This drone lands for exactly three reasons: the person was found, the battery
hit 15 %, or the operator stopped it. An earlier circuit breaker aborted the
flight after N consecutive failures — pointless, since a link dead enough to
justify aborting can't deliver a `land` command either.

## What changed, and why it mattered

`before/` is the same mission on the same hardware ~50 minutes earlier.

| | before | after |
|---|---|---|
| time to find the person | 126.8 s | **53.1 s** |
| one 360° scan | ~70 s | **~20 s** (2.7–2.9 s of it judging) |
| flip | skipped — battery 19 % < 55 % gate | **done** |
| total flight | 140.3 s | 72.7 s |

Four things were wrong, all visible in `before/`:

1. **The camera was pointed at the carpet.** Hovering at ~90 cm with a fixed,
   down-angled lens, most frames were floor and baseboard; the person was only
   ever detected as bare feet at the frame edge. Search height went to ~1.3–1.6 m.
2. **The scan was ~70 s of a ~150 s battery, almost all of it waiting on HTTP.**
   Each frame costs two sequential round trips (vision model, then jev), run
   inline per heading. Capture is now mechanical and network-free, and the whole
   circle is judged concurrently — 4.1× faster on a 6-frame circle.
3. **jev was navigating nearly blind.** The deterministic dead-end check needs
   opencv+numpy, which aren't installed on this host, so it silently never fired
   (`bright_frac: null` throughout `before/jev_decisions.jsonl`). jev was
   choosing from six vague one-liners. It now gets `openness`, `path` and
   `exits` per heading, free — that call was already being made.
4. **The old prompt rejected the actual target.** At 22:31:17 in
   `before/jev_decisions.jsonl`, the vision model described the search target as
   `is_real_person: false` — "partially visible behind door" — and jev scored it
   0.18. The same frame scores 0.83 now.

## Caveats

- Position is dead reckoning from the commands sent; the Tello has no indoor
  positioning, so "home" is an estimate that degrades with every skipped command.
- The 0.60 confirm bar is calibrated on frames from these two flights. Re-tune
  with `./run.sh calibrate` on your own drone, room and lighting.
- The head is still out of frame in `person.jpg` — the lens angle means even at
  1.3 m the drone frames a standing adult from the waist down. More height would
  help, bounded by `max_height_deviation_cm`.
- This is real hardware control. The drone has no obstacle sensing: forward and
  vertical movement are blind.
