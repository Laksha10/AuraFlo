# pose_matcher.py
# The custom pose-matching layer on top of MediaPipe's 33 landmarks:
#   landmarks -> joint angles (vector math) -> compare with the active pose's JSON profile
#   -> status: IN_POSE / CORRECTING / TRANSITIONING / NOT_VISIBLE  (+ one correction message)
# Corrections go one joint at a time, largest deviation first, then the next largest, and so on.

import json
import numpy as np
import config

# Each joint = 3 MediaPipe landmark numbers: (first point, VERTEX, last point).
# The angle is measured at the middle (vertex) point.
JOINT_LANDMARKS = {
    "left_elbow":     (11, 13, 15),   # left shoulder  -> left elbow     -> left wrist
    "right_elbow":    (12, 14, 16),   # right shoulder -> right elbow    -> right wrist
    "left_shoulder":  (23, 11, 13),   # left hip       -> left shoulder  -> left elbow
    "right_shoulder": (24, 12, 14),   # right hip      -> right shoulder -> right elbow
    "left_knee":      (23, 25, 27),   # left hip       -> left knee      -> left ankle
    "right_knee":     (24, 26, 28),   # right hip      -> right knee     -> right ankle
}

# What to say when a joint's live angle is too big or too small.
# Knees/elbows: bigger angle = straighter limb.   Shoulders: bigger angle = arm raised higher.
CORRECTION_TEXT = {
    "left_elbow":     {"too_big": "Bend your left elbow more",     "too_small": "Straighten your left arm"},
    "right_elbow":    {"too_big": "Bend your right elbow more",    "too_small": "Straighten your right arm"},
    "left_shoulder":  {"too_big": "Lower your left arm a little",  "too_small": "Raise your left arm higher"},
    "right_shoulder": {"too_big": "Lower your right arm a little", "too_small": "Raise your right arm higher"},
    "left_knee":      {"too_big": "Bend your left knee more",      "too_small": "Straighten your left knee"},
    "right_knee":     {"too_big": "Bend your right knee more",     "too_small": "Straighten your right knee"},
}


def load_profiles(path):
    """Reads the JSON pose library into a Python dictionary."""
    with open(path) as f:
        profiles = json.load(f)
    return profiles


def get_point(landmarks, idx, frame_w, frame_h):
    """Returns one landmark as a pixel (x, y) point, or None if MediaPipe isn't confident about it."""
    lm = landmarks[idx]
    if lm.visibility is not None and lm.visibility < config.VISIBILITY_MIN:
        return None
    # MediaPipe gives x and y as fractions (0 to 1) of width and height.
    # Converting to pixels keeps angles true, because the frame's width and height differ.
    return np.array([lm.x * frame_w, lm.y * frame_h])


def calculate_angle(a, b, c):
    """Angle in degrees at point b, between the 'bones' b->a and b->c. Always 0 to 180."""
    ba = a - b                                  # arrow from the joint to the first point
    bc = c - b                                  # arrow from the joint to the last point
    dot = ba[0] * bc[0] + ba[1] * bc[1]         # how much the arrows point the same way
    cross = ba[0] * bc[1] - ba[1] * bc[0]       # how much they point sideways from each other
    angle = np.degrees(np.arctan2(abs(cross), dot))
    return angle


def get_all_angles(landmarks, frame_w, frame_h):
    """Calculates every joint angle. A joint with any hidden point gets None."""
    angles = {}
    for joint_name in JOINT_LANDMARKS:
        first_idx, vertex_idx, last_idx = JOINT_LANDMARKS[joint_name]
        a = get_point(landmarks, first_idx, frame_w, frame_h)
        b = get_point(landmarks, vertex_idx, frame_w, frame_h)
        c = get_point(landmarks, last_idx, frame_w, frame_h)
        if a is None or b is None or c is None:
            angles[joint_name] = None
        else:
            angles[joint_name] = calculate_angle(a, b, c)
    return angles


def smooth_angles(new_angles, smoothed_angles):
    """Exponential moving average per joint, to calm frame-to-frame jitter."""
    alpha = config.EMA_ALPHA
    for joint_name in new_angles:
        new_value = new_angles[joint_name]
        if new_value is None:
            smoothed_angles[joint_name] = None                 # joint hidden: forget old value
        elif joint_name not in smoothed_angles or smoothed_angles[joint_name] is None:
            smoothed_angles[joint_name] = new_value            # first reading: nothing to smooth with
        else:
            smoothed_angles[joint_name] = alpha * new_value + (1 - alpha) * smoothed_angles[joint_name]
    return smoothed_angles


def match_pose(angles, profile, focus_joint):
    """
    Compares live angles with the ACTIVE pose's profile only.
    focus_joint = the joint currently being corrected (None if none yet).
    Returns a dictionary: {"status": ..., "message": ..., "focus_joint": ...}
    """
    # Step 1: work out the delta (live - target) for every joint this pose cares about
    joint_results = []
    total_deviation = 0
    for joint_name in profile["joints"]:
        live = angles.get(joint_name)
        if live is None:
            return {"status": "NOT_VISIBLE", "message": "Step back so your whole body is in view",
                    "focus_joint": None}

        target = profile["joints"][joint_name]["target"]
        tolerance = profile["joints"][joint_name]["tolerance"]
        delta = live - target                    # positive = angle too big, negative = too small
        joint_results.append({"joint": joint_name, "delta": delta, "tolerance": tolerance})
        total_deviation = total_deviation + abs(delta)

    # Step 2: AVERAGE deviation over all joints above 45 deg -> moving between poses. Say nothing.
    average_deviation = total_deviation / len(joint_results)
    if average_deviation > config.TRANSITION_LIMIT:
        return {"status": "TRANSITIONING", "message": "", "focus_joint": None}

    # Step 3: collect every joint that is outside its tolerance
    wrong_joints = []
    for result in joint_results:
        if abs(result["delta"]) > result["tolerance"]:
            wrong_joints.append(result)

    # Step 4: nothing outside tolerance -> in the pose
    if len(wrong_joints) == 0:
        return {"status": "IN_POSE", "message": "Hold it!", "focus_joint": None}

    # Step 5: one joint at a time. Keep correcting the focus joint until it's fixed;
    # once it is (or there is none yet), move to the joint with the LARGEST remaining deviation.
    chosen = None
    for result in wrong_joints:
        if result["joint"] == focus_joint:
            chosen = result
    if chosen is None:
        for result in wrong_joints:
            if chosen is None or abs(result["delta"]) > abs(chosen["delta"]):
                chosen = result

    if chosen["delta"] > 0:
        text = CORRECTION_TEXT[chosen["joint"]]["too_big"]
    else:
        text = CORRECTION_TEXT[chosen["joint"]]["too_small"]
    return {"status": "CORRECTING",
            "message": f"{text} ({abs(chosen['delta']):.0f} deg off)",
            "focus_joint": chosen["joint"]}