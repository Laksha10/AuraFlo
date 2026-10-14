# main.py
# Runs one guided session: the visual loop (every frame) + the event bridge + the routine.
#
# Session states:
#   GUIDING    -> live corrections + 10 s hold timer for the current pose
#   CHECK_IN   -> visual feedback paused, mic open (audio runs in the background)
#   EXTENDING  -> user said "too easy" and there's no harder pose: hold 10 s more
#   RESTING    -> timed rest pose (Child's Pose), no matching
#   FINISHED   -> routine done
#
# Keys: q = quit, n = skip to next pose, p = print live angles (use this to calibrate the JSON)

import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

import config
import events
import logger
from pose_matcher import load_profiles, get_all_angles, smooth_angles, match_pose
from hold_timer import HoldTimer
from audio_checkin import AudioCheckin
from routine import Routine

FONT = cv2.FONT_HERSHEY_SIMPLEX

# OpenCV colours are (Blue, Green, Red)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)

# Which landmarks to join with lines when drawing the skeleton (body only, no face)
SKELETON_LINES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),      # shoulders + arms
    (11, 23), (12, 24), (23, 24),                          # torso
    (23, 25), (25, 27), (24, 26), (26, 28),                # legs
]


def put_text(frame, text, y, color=WHITE, scale=0.7):
    cv2.putText(frame, text, (10, y), FONT, scale, color, 2)


def draw_skeleton(frame, landmarks, frame_w, frame_h):
    for start_idx, end_idx in SKELETON_LINES:
        p1 = landmarks[start_idx]
        p2 = landmarks[end_idx]
        x1, y1 = int(p1.x * frame_w), int(p1.y * frame_h)
        x2, y2 = int(p2.x * frame_w), int(p2.y * frame_h)
        cv2.line(frame, (x1, y1), (x2, y2), (245, 66, 230), 2)
    for idx in range(11, 29):
        lm = landmarks[idx]
        cv2.circle(frame, (int(lm.x * frame_w), int(lm.y * frame_h)), 4, (245, 117, 66), -1)


def begin_current_pose(routine, profiles, hold_timer, now):
    """Sets up whatever pose is now at the front of the queue. Returns (state, rest_end_time)."""
    if routine.is_finished():
        return "FINISHED", None

    profile = profiles[routine.current_pose()]
    print(f"➡️  Now: {profile['display_name']}")

    if "rest_seconds" in profile:
        return "RESTING", now + profile["rest_seconds"]

    hold_timer.reset(config.HOLD_SECONDS)
    hold_timer.checked_in = False            # the check-in happens once per pose
    return "GUIDING", None


def main():
    profiles = load_profiles(config.PROFILES_PATH)
    routine = Routine(config.ROUTINE, profiles)
    hold_timer = HoldTimer(config.HOLD_SECONDS)
    checkin = AudioCheckin()

    # Wire up the event: the audio module listens for "pose_held".
    events.subscribe("pose_held", checkin.on_pose_held)

    logger.init_db()
    session_id = logger.start_session(config.ROUTINE)

    # MediaPipe Pose Landmarker (BlazePose) in VIDEO mode
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=config.MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return

    smoothed = {}
    focus_joint = None
    focus_pose = None
    start_clock = time.time()
    last_timestamp_ms = -1
    last_frame_time = time.time()
    session_state, rest_end_time = begin_current_pose(routine, profiles, hold_timer, time.time())

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        print("📷 Camera started. q = quit, n = next pose, p = print angles")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Empty frame received. Exiting.")
                break

            now = time.time()
            frame_h, frame_w = frame.shape[:2]

            # ---------- MediaPipe: frame -> 33 landmarks ----------
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int((now - start_clock) * 1000)
            if timestamp_ms <= last_timestamp_ms:         # VIDEO mode needs ever-increasing timestamps
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            body_found = len(result.pose_landmarks) > 0
            if body_found:
                landmarks = result.pose_landmarks[0]      # first (and only) person
                draw_skeleton(frame, landmarks, frame_w, frame_h)
                live_angles = get_all_angles(landmarks, frame_w, frame_h)
                smoothed = smooth_angles(live_angles, smoothed)
            else:
                smoothed = {}

            pose_name = routine.current_pose()
            if pose_name != focus_pose:              # new pose -> start corrections fresh
                focus_joint = None
                focus_pose = pose_name

            # ---------- Session state machine ----------
            if session_state == "GUIDING" or session_state == "EXTENDING":
                profile = profiles[pose_name]
                put_text(frame, f"Pose {routine.index + 1}/{len(routine.queue)}: {profile['display_name']}", 80)

                if body_found:
                    match = match_pose(smoothed, profile, focus_joint)
                else:
                    match = {"status": "NOT_VISIBLE", "message": "Step into the camera view", "focus_joint": None}
                focus_joint = match["focus_joint"]   # remember which joint we're correcting

                completed = hold_timer.update(match["status"] == "IN_POSE", now)

                if match["status"] == "IN_POSE":
                    held = hold_timer.seconds_held(now)
                    put_text(frame, f"Hold it! {held:.0f} / {hold_timer.required_seconds} s", 120, GREEN)
                elif match["status"] == "CORRECTING":
                    put_text(frame, match["message"], 120, ORANGE)
                elif match["status"] == "NOT_VISIBLE":
                    put_text(frame, match["message"], 120, RED)
                # TRANSITIONING: show nothing on purpose

                if completed:
                    if session_state == "GUIDING" and not hold_timer.checked_in:
                        hold_timer.checked_in = True
                        session_state = "CHECK_IN"
                        events.emit("pose_held", {"pose": pose_name, "timestamp": now})
                    else:
                        # extra hold finished -> move on, no second check-in for this pose
                        routine.advance()
                        session_state, rest_end_time = begin_current_pose(routine, profiles, hold_timer, now)

            elif session_state == "CHECK_IN":
                put_text(frame, config.CHECKIN_QUESTION, 80, YELLOW, 0.9)
                put_text(frame, "Listening...", 120, YELLOW)

                if not checkin.results.empty():
                    checkin_result = checkin.results.get()
                    action = routine.apply_sentiment(checkin_result["label"])
                    logger.log_feedback(
                        session_id,
                        checkin_result["pose"],
                        checkin_result["text"],
                        checkin_result["compound"],
                        checkin_result["label"],
                        action,
                        routine.next_pose(),
                    )
                    if action == "EXTEND":
                        hold_timer.reset(config.EXTEND_SECONDS)
                        session_state = "EXTENDING"
                        print(f"⏱️  Too easy - holding {config.EXTEND_SECONDS} s longer")
                    else:
                        routine.advance()
                        session_state, rest_end_time = begin_current_pose(routine, profiles, hold_timer, now)

            elif session_state == "RESTING":
                remaining = rest_end_time - now
                put_text(frame, profiles[pose_name]["display_name"], 80, GREEN)
                put_text(frame, f"Rest and breathe: {remaining:.0f} s", 120, GREEN)
                if remaining <= 0:
                    routine.advance()
                    session_state, rest_end_time = begin_current_pose(routine, profiles, hold_timer, now)

            elif session_state == "FINISHED":
                put_text(frame, "Session complete! Press q to quit.", 80, GREEN, 0.9)

            # ---------- FPS + window ----------
            fps = 1 / max(time.time() - last_frame_time, 1e-6)
            last_frame_time = time.time()
            put_text(frame, f"FPS: {fps:.1f}", 40)
            cv2.imshow("AuraFlo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("👋 Exiting...")
                break
            elif key == ord("n") and session_state != "CHECK_IN" and session_state != "FINISHED":
                routine.advance()
                session_state, rest_end_time = begin_current_pose(routine, profiles, hold_timer, now)
            elif key == ord("p"):
                print("📐 Live angles:")
                for joint_name in smoothed:
                    value = smoothed[joint_name]
                    if value is None:
                        print(f"   {joint_name}: hidden")
                    else:
                        print(f"   {joint_name}: {value:.0f}")

    cap.release()
    cv2.destroyAllWindows()
    logger.end_session(session_id)


if __name__ == "__main__":
    main()