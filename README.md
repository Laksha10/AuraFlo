# AuraFlo — Real-Time Yoga Pose Feedback

A camera-based yoga coaching pipeline. It watches your pose, gives live joint-level corrections, and once you hold the pose steadily, asks how it feels. It then uses your spoken answer to adapt the routine.

## How it works

There are two loops, connected by an event.

**Visual loop (every frame)**
1. MediaPipe Pose Landmarker (BlazePose) detects 33 body landmarks.
2. Six joint angles (knees, elbows, shoulders) are computed with vector math (atan2).
3. The angles are compared with the active pose's JSON profile:
   - Each joint within ±20° of its target counts as correct.
   - If the **average** deviation is above 45°, the user is transitioning, and no feedback is shown.
   - Otherwise, corrections are given one joint at a time, largest deviation first.

**Bridge**

A hold-timer state machine fires a `pose_held` event after an **unbroken 10-second hold**. Breaking the pose resets the timer to zero.

**Audio loop (once per pose, on the event)**
1. Visual feedback pauses and the user is asked "How does this feel?"
2. 5 seconds of audio are recorded and checked for silence.
3. Whisper (tiny) transcribes the audio, and VADER scores the sentiment.
4. A deterministic lookup table decides what happens next:

| Answer | Action |
|---|---|
| Pain words ("hurts", "ache"…) | Swap the next pose for the rest pose (Child's Pose) |
| Negative | Swap the next pose for the easier variant |
| Neutral / no answer | Continue the plan |
| Positive (too easy) | Swap the next pose for the harder variant, or hold 10 s longer if there is none |

Every session and every check-in is logged to SQLite in two tables: `sessions` and `feedback_events`.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Entry point: camera loop and session state machine |
| `config.py` | All tunable values: tolerances, timers, thresholds, routine |
| `pose_profiles.json` | 9 front-facing pose profiles plus a rest pose, with target angles and easier/harder/rest pointers |
| `pose_matcher.py` | Joint-angle calculation and pose matching |
| `hold_timer.py` | 10-second hold state machine |
| `events.py` | Minimal event bus that decouples the pose side from the audio side |
| `audio_checkin.py` | Speech check-in: record → Whisper, with a busy flag and silence guards |
| `sentiment.py` | VADER sentiment with a pain-word safety override |
| `routine.py` | Deterministic routine state machine |
| `logger.py` | SQLite logging |

## Setup

1. Use Python 3.11 or 3.12.
2. Install the dependencies:
```
   pip install -r requirements.txt
```
3. Download the BlazePose model into the project folder and keep its name, `pose_landmarker_lite.task`:
   https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
4. Run the app:
```
   python main.py
```

Stand facing the camera with your whole body in view.

Keys:
- `q` quit
- `n` skip to the next pose
- `p` print your live joint angles (useful for calibrating the targets in `pose_profiles.json`)

## Notes

- Target angles are for a front-facing camera. Calibrate them for your own camera setup using `p`.
- One-sided poses (Tree, Warrior II, Triangle, and the others) are defined for the left side.
