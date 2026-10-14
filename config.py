# config.py
# Every number you might want to tune lives here, in one place.

# ---------------- Camera + pose model ----------------
CAMERA_INDEX = 0                          # 0 = default webcam
MODEL_PATH = "pose_landmarker_lite.task"  # BlazePose model file (download link in setup steps)
VISIBILITY_MIN = 0.5                      # ignore landmarks MediaPipe is less than 50% sure about
EMA_ALPHA = 0.2                           # angle smoothing: closer to 0 = smoother but slower to react

# ---------------- Pose matching ----------------
# Within a joint's "tolerance" (20 deg, set per joint in pose_profiles.json) -> that joint is correct.
# AVERAGE deviation across all of the pose's joints above TRANSITION_LIMIT  -> "transitioning", show nothing.
# Otherwise -> corrections, one joint at a time, largest deviation first.
TRANSITION_LIMIT = 45

# ---------------- Hold timer ----------------
HOLD_SECONDS = 10      # unbroken hold needed before the check-in fires
EXTEND_SECONDS = 10    # extra hold when the user says it's too easy and there is no harder variant

# ---------------- Audio check-in ----------------
CHECKIN_QUESTION = "How does this feel?"
USE_SPOKEN_QUESTION = True   # set False if the spoken question misbehaves on your laptop
SAMPLE_RATE = 16000          # Whisper expects 16,000 samples per second
RECORD_SECONDS = 5           # length of the audio burst handed to Whisper
SILENCE_RMS = 0.01           # recordings quieter than this are treated as silence (skip Whisper)
NO_SPEECH_PROB_LIMIT = 0.6   # Whisper segments it thinks are "probably not speech" get dropped
WHISPER_MODEL = "tiny"       # small + fast; "base" is a bit more accurate but slower

# ---------------- Sentiment -> decision ----------------
# VADER gives a "compound" score from -1 (very negative) to +1 (very positive).
NEGATIVE_THRESHOLD = -0.05   # at or below this -> struggling -> easier variant
POSITIVE_THRESHOLD = 0.4     # at or above this -> too easy -> harder variant (or extend hold)
# Any of these words -> treated as pain -> rest pose, whatever VADER says (safety first).
PAIN_WORDS = ["pain", "hurt", "ache", "aching", "sharp", "injur", "dizzy", "cramp", "strain", "sore"]

# ---------------- Files ----------------
PROFILES_PATH = "pose_profiles.json"
DB_PATH = "auraflo.db"

# ---------------- The guided routine (order of poses) ----------------
ROUTINE = [
    "mountain",
    "upward_salute",
    "star",
    "warrior_2_left",
    "warrior_2_right",
    "goddess",
    "tree_left",
    "tree_right",
    "triangle_left",
    "triangle_right",
]