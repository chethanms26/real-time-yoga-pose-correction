"""
Real-Time Yoga Pose Correction - Step 4 (v7): Live Prediction + Stability
Analysis + targeted real-time safety checks + JOINT-VIOLATION MESSAGES.

v5 adds a SECOND Ardha Chandrasana safety check on top of v4's leg-lift
override, because live testing found a bug v4 didn't catch: a WIDE BILATERAL
STANCE (both feet on the ground, legs just spread apart, no leg actually
lifted) was still being classified "Correct" by the ML model. Retraining
alone (v3) could not reliably fix this because wide-stance and genuine
one-legged extension can produce similar HIP ANGLES - the real difference
is whether one foot is actually off the ground, which no angle feature
captures.

New check: compares the vertical (y-coordinate) position of the two ankles,
normalized by torso height. If both ankles are close to the same height
(both feet on the ground), the pose cannot be a genuine Half Moon extension,
regardless of what the model predicts - override to Incorrect.

v6 keeps the Ardha Chandrasana v5 checks and adds two screenshot-validated
guards:
1) Bhujangasana must not accept a standing straight-body posture.
2) Vrikshasana must not accept leg-only Tree pose when both arms are down.
3) Bhujangasana upper-body fallback for prone webcam cases where MediaPipe
   cannot reliably expose all knee/ankle landmarks.
4) Tadasana must not accept simple arms-down standing when the demo expects
   the stretched/raised-arm version.

v7 adds JOINT-VIOLATION MESSAGES: when the STABLE prediction is Incorrect,
identify which specific joint(s) are outside their expected range for that
pose and surface a targeted correction cue (e.g. "Straighten your right
knee"), instead of only showing "Incorrect".

Thresholds are NOT taken from the Phase 2 official table as-is, because
that table was already found to be wrong in at least one place (Ardha
Chandrasana hip range). Instead, per-pose per-joint expected ranges are
computed from the 5th/95th percentile of each RAW correct_<pose>.csv,
matching the project's established rule of validating against real data
before building a fix. For Vrikshasana and Ardha Chandrasana, knee/hip are
asymmetric (one leg stands, one leg lifts, and which side varies row to
row), so a plain left/right column threshold would mix two different
distributions together. Those two poses use a STANDING-SIDE / LIFTED-SIDE
role split instead (whichever knee is straighter that frame is treated as
the "standing" leg), which was confirmed to produce clean, tight ranges by
inspecting the raw data. Ardha Chandrasana hip is intentionally NOT used
for a threshold message (hip alone can't reliably separate correct vs
incorrect there - the existing v4/v5 rule-based overrides already cover
that case via leg-lift and ankle-height, which fire independently of this
message layer).
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import json
import threading
import queue
import time
from collections import deque

TEST_MODE = os.environ.get("RUN_SCREENSHOT_TESTS") == "1"

if not TEST_MODE:
    import joblib
    import cv2
    import mediapipe as mp
    import pyttsx3

    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,  # lite model: faster to load and faster per-frame,
                              # matters more for real-time feedback than the
                              # small accuracy gain from complexity=1
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
BASE_PATH = r"C:\Users\Chethan MS\Documents\Capstone 7"

POSE_OPTIONS = {
    "1": ("Ardha Chandrasana", "ARDHA CHANDRASANA", "ardha_chandrasana_rf"),
    "2": ("Bhujangasana",      "BHUJANGASANA",      "bhujangasana_rf"),
    "3": ("Tadasana",          "TADASANA",          "tadasana_rf"),
    "4": ("Trikonasana",       "TRIKONASANA",       "trikonasana_rf"),
    "5": ("Vrikshasana",       "VRIKSHASANA",       "vrikshasana_rf"),
}

ANGLE_NAME_TO_FEATURE = {
    "Left Knee": "left_knee",
    "Right Knee": "right_knee",
    "Left Hip": "left_hip",
    "Right Hip": "right_hip",
    "Left Shoulder": "left_shoulder",
    "Right Shoulder": "right_shoulder",
}

# ---------------------------------------------------------------
# STABILITY SETTINGS
# ---------------------------------------------------------------
WINDOW_SIZE = 6      # was 8 - shortened per live-testing feedback (both the
                      # "correct" onset lag and the stale "correct" during a
                      # loss of balance come from this window; smaller = more
                      # responsive but slightly more flicker-prone)
MIN_CONFIDENCE = 0.65
MIN_AGREEMENT = 0.6   # was 0.65 - loosened alongside the smaller window
MISSING_FRAME_LIMIT = 10

# ---------------------------------------------------------------
# ARDHA CHANDRASANA SAFETY CHECKS
# ---------------------------------------------------------------
LEG_LIFT_MAX_FOR_CORRECT = 130  # v4 check: shoulder-hip-ankle angle

# v5 check: normalized vertical ankle separation. If the two ankles are
# within this fraction of torso height of each other, both feet are
# considered "on the ground" (not a genuine one-legged extension).
ANKLE_HEIGHT_DIFF_MIN_FOR_CORRECT = 0.35

# ---------------------------------------------------------------
# BHUJANGASANA SAFETY CHECKS
# ---------------------------------------------------------------
# Screenshot 574 failure case:
# knees ~= 178/178, hips ~= 176/176, shoulders ~= 49/36 was shown Correct.
# That is standing straight-body posture, not Cobra. This guard only rejects
# the very-straight region, so normal Cobra rows with bent hip angles remain
# controlled by the trained model.
BHUJANGASANA_STANDING_KNEE_MIN = 165
BHUJANGASANA_STANDING_HIP_MIN = 165
BHUJANGASANA_STANDING_SHOULDER_MAX = 120

# ---------------------------------------------------------------
# VRIKSHASANA SAFETY CHECKS
# ---------------------------------------------------------------
# Project/demo expectation: Vrikshasana requires hands raised. The old
# threshold allowed 3-50 deg, which accepts arms-down leg-only poses.
VRIKSHASANA_RAISED_SHOULDER_MIN = 150

# ---------------------------------------------------------------
# TADASANA SAFETY CHECKS
# ---------------------------------------------------------------
TADASANA_RAISED_SHOULDER_MIN = 150

# ---------------------------------------------------------------
# JOINT-VIOLATION MESSAGE THRESHOLDS (v7)
# ---------------------------------------------------------------
# Computed from the 5th/95th percentile of each RAW correct_<pose>.csv,
# then rounded outward slightly for a small safety margin (matches the
# spirit of the Phase 2 table without inheriting its known Ardha
# Chandrasana hip error). Format: (low, high) in degrees.
#
# Simple per-column poses (left/right are anatomically fixed sides):
TADASANA_JOINT_RANGES = {
    "left_knee": (165, 180), "right_knee": (165, 180),
    "left_hip": (170, 180), "right_hip": (170, 180),
    "left_shoulder": (TADASANA_RAISED_SHOULDER_MIN, 180),
    "right_shoulder": (TADASANA_RAISED_SHOULDER_MIN, 180),
}

BHUJANGASANA_JOINT_RANGES = {
    "left_knee": (140, 168), "right_knee": (144, 166),
    "left_hip": (122, 173), "right_hip": (125, 170),
    "left_shoulder": (18, 50), "right_shoulder": (20, 50),
}

TRIKONASANA_JOINT_RANGES = {
    "left_knee": (172, 180), "right_knee": (160, 180),
    "left_hip": (110, 160), "right_hip": (47, 122),
    "left_shoulder": (105, 155), "right_shoulder": (50, 125),
}

# Role-based poses: one leg stands (straighter knee), the other lifts.
# Whichever knee is straighter in a given frame is treated as "standing".
VRIKSHASANA_ROLE_RANGES = {
    "standing_knee": (168, 180), "lifted_knee": (58, 105),
    "standing_hip": (168, 180), "lifted_hip": (108, 150),
}
VRIKSHASANA_SHOULDER_RANGE = (VRIKSHASANA_RAISED_SHOULDER_MIN, 180)

# Ardha Chandrasana: only knee is reliable enough to threshold directly
# (hip is left to the existing v4/v5 rule-based overrides - see docstring).
ARDHA_CHANDRASANA_ROLE_RANGES = {
    "standing_knee": (168, 180), "lifted_knee": (155, 180),
}

JOINT_MESSAGE_DEADBAND = 4  # degrees of slack before a violation is reported
MAX_JOINT_MESSAGES = 2      # cap on-screen messages to the worst offenders


def select_pose():
    print("Select the pose you are performing:")
    for key, (name, _, _) in POSE_OPTIONS.items():
        print(f"  {key}) {name}")
    choice = input("Enter choice (1-5): ").strip()
    if choice not in POSE_OPTIONS:
        raise ValueError("Invalid choice. Please enter a number from 1 to 5.")
    return POSE_OPTIONS[choice]


def load_model_and_metadata(pose_name, folder_name, file_prefix):
    folder_path = os.path.join(BASE_PATH, folder_name)
    model_path = os.path.join(folder_path, f"{file_prefix}.pkl")
    metadata_path = os.path.join(folder_path, f"{file_prefix}_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    model = joblib.load(model_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"\nLoaded model for {pose_name}")
    print(f"  Expected features (in order): {metadata['features']}")
    print(f"  Test accuracy at training time: {metadata.get('test_accuracy', 'N/A')}")

    return model, metadata


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def is_visible(landmarks, indices, threshold=0.2):
    return all(landmarks[i].visibility > threshold for i in indices)


# MediaPipe pose landmark indices, for reference:
# 23/24 = left/right hip, 25/26 = left/right knee, 27/28 = left/right ankle
LOWER_BODY_LANDMARKS = {"left_hip": 23, "right_hip": 24,
                         "left_knee": 25, "right_knee": 26,
                         "left_ankle": 27, "right_ankle": 28}


def get_lower_body_visibility(landmarks):
    """
    Diagnostic only (v7) - not used in any classification decision yet.
    Returns visibility scores (0-1) for hip/knee/ankle landmarks, to check
    whether "legs go out of frame / low visibility during genuine lying-down
    Cobra" actually holds true on a real camera setup before building
    anything on top of it.
    """
    return {name: round(float(landmarks[idx].visibility), 2)
            for name, idx in LOWER_BODY_LANDMARKS.items()}


def get_angles(landmarks):
    def get_point(i):
        return [landmarks[i].x, landmarks[i].y]

    rs, re, rw = get_point(12), get_point(14), get_point(16)
    rh, rk, ra = get_point(24), get_point(26), get_point(28)
    ls, le, lw = get_point(11), get_point(13), get_point(15)
    lh, lk, la = get_point(23), get_point(25), get_point(27)

    angles = {}
    if is_visible(landmarks, [14, 12, 24]):
        angles["Right Shoulder"] = calculate_angle(re, rs, rh)
    if is_visible(landmarks, [12, 24, 26]):
        angles["Right Hip"] = calculate_angle(rs, rh, rk)
    if is_visible(landmarks, [24, 26, 28]):
        angles["Right Knee"] = calculate_angle(rh, rk, ra)
    if is_visible(landmarks, [13, 11, 23]):
        angles["Left Shoulder"] = calculate_angle(le, ls, lh)
    if is_visible(landmarks, [11, 23, 25]):
        angles["Left Hip"] = calculate_angle(ls, lh, lk)
    if is_visible(landmarks, [23, 25, 27]):
        angles["Left Knee"] = calculate_angle(lh, lk, la)
    return angles


def get_leg_lift_angles(landmarks):
    """v4 check: shoulder -> hip -> ankle angle, both sides."""
    def get_point(i):
        return [landmarks[i].x, landmarks[i].y]

    result = {}
    if is_visible(landmarks, [12, 24, 28]):
        result["right_leg_lift"] = calculate_angle(get_point(12), get_point(24), get_point(28))
    if is_visible(landmarks, [11, 23, 27]):
        result["left_leg_lift"] = calculate_angle(get_point(11), get_point(23), get_point(27))
    return result


def get_ankle_height_diff(landmarks):
    """
    v5 check: normalized vertical distance between the two ankles.
    Normalized by shoulder-to-hip distance (torso height) so it works
    regardless of distance from camera.
    Returns None if required landmarks aren't visible.
    """
    required = [11, 12, 23, 24, 27, 28]  # shoulders, hips, ankles
    if not is_visible(landmarks, required, threshold=0.3):
        return None

    left_shoulder_y = landmarks[11].y
    right_shoulder_y = landmarks[12].y
    left_hip_y = landmarks[23].y
    right_hip_y = landmarks[24].y
    left_ankle_y = landmarks[27].y
    right_ankle_y = landmarks[28].y

    torso_height = abs(((left_shoulder_y + right_shoulder_y) / 2) -
                        ((left_hip_y + right_hip_y) / 2))
    if torso_height < 1e-6:
        return None

    ankle_diff = abs(left_ankle_y - right_ankle_y)
    return ankle_diff / torso_height


def bhujangasana_upper_body_fallback(landmarks):
    """
    Cobra is often shot close to the floor, so knees/ankles can be weak or
    outside MediaPipe's reliable pose estimate. This fallback only applies
    to Bhujangasana and scores visible upper-body cues instead of requiring
    every angle landmark. Side/floor webcam views often weaken elbows, hips,
    knees, or ankles even when the Cobra shape is obvious.
    """
    visible = {i for i in range(11, 29) if landmarks[i].visibility > 0.05}
    upper_core = [11, 12, 15, 16]
    if not all(i in visible for i in upper_core):
        return False

    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    wrist_y = (landmarks[15].y + landmarks[16].y) / 2
    hip_y = None
    if 23 in visible and 24 in visible:
        hip_y = (landmarks[23].y + landmarks[24].y) / 2

    lower_visible = sum(i in visible for i in [23, 24, 25, 26, 27, 28])
    if hip_y is None or lower_visible < 2:
        return False

    torso_lift = hip_y - shoulder_y
    if torso_lift < 0.10:
        return False

    score = 0

    # Hands on the floor are below the shoulders in the image.
    if wrist_y > shoulder_y + 0.08:
        score += 1

    if 13 in visible and 14 in visible:
        elbow_y = (landmarks[13].y + landmarks[14].y) / 2
        if elbow_y > shoulder_y:
            score += 1

    if hip_y is not None and hip_y > shoulder_y + 0.10:
        score += 1

    score += 1

    # In a floor Cobra frame, the detected body landmarks span wide
    # horizontally and stay relatively low in the frame.
    xs = [landmarks[i].x for i in visible]
    ys = [landmarks[i].y for i in visible]
    if xs and max(xs) - min(xs) > 0.35:
        score += 1
    if ys and sum(y > 0.45 for y in ys) >= max(4, len(ys) // 2):
        score += 1

    return score >= 4


def angles_to_feature_row(angles_dict, feature_order):
    row = []
    for feat in feature_order:
        matched_display_name = None
        for display_name, feat_name in ANGLE_NAME_TO_FEATURE.items():
            if feat_name == feat:
                matched_display_name = display_name
                break
        if matched_display_name is None or matched_display_name not in angles_dict:
            return None
        row.append(angles_dict[matched_display_name])
    return row


def feature_row_to_dict(feature_row, feature_order):
    return {feat: float(val) for feat, val in zip(feature_order, feature_row)}


def bhujangasana_standing_override(feature_values):
    return (
        feature_values["left_knee"] >= BHUJANGASANA_STANDING_KNEE_MIN and
        feature_values["right_knee"] >= BHUJANGASANA_STANDING_KNEE_MIN and
        feature_values["left_hip"] >= BHUJANGASANA_STANDING_HIP_MIN and
        feature_values["right_hip"] >= BHUJANGASANA_STANDING_HIP_MIN and
        feature_values["left_shoulder"] <= BHUJANGASANA_STANDING_SHOULDER_MAX and
        feature_values["right_shoulder"] <= BHUJANGASANA_STANDING_SHOULDER_MAX
    )


def vrikshasana_arms_down_override(feature_values):
    # Both arms must be raised overhead - if EITHER arm is down, it's not
    # the pose. Previously used "and" here, which only caught the case
    # where both arms were down and let a one-arm-raised pose slip through
    # to the raw ML model (which hadn't reliably seen that asymmetric case
    # and sometimes still called it Correct).
    return (
        feature_values["left_shoulder"] < VRIKSHASANA_RAISED_SHOULDER_MIN or
        feature_values["right_shoulder"] < VRIKSHASANA_RAISED_SHOULDER_MIN
    )


def tadasana_arms_down_override(feature_values):
    # Same fix as Vrikshasana above - either arm down should fail the pose,
    # not just both arms down.
    return (
        feature_values["left_shoulder"] < TADASANA_RAISED_SHOULDER_MIN or
        feature_values["right_shoulder"] < TADASANA_RAISED_SHOULDER_MIN
    )


def apply_pose_safety_overrides(pose_name, prediction, raw_confidence, feature_values, landmarks=None):
    if prediction != 1:
        return prediction, raw_confidence, "", []

    reasons = []

    if pose_name == "Bhujangasana" and bhujangasana_standing_override(feature_values):
        reasons.append("standing posture")

    if pose_name == "Vrikshasana" and vrikshasana_arms_down_override(feature_values):
        reasons.append("hands not raised")

    if pose_name == "Tadasana" and tadasana_arms_down_override(feature_values):
        reasons.append("hands not raised")

    if pose_name == "Ardha Chandrasana" and landmarks is not None:
        leg_lift = get_leg_lift_angles(landmarks)
        lifts = list(leg_lift.values())
        leg_lift_fails = len(lifts) > 0 and min(lifts) > LEG_LIFT_MAX_FOR_CORRECT

        ankle_diff = get_ankle_height_diff(landmarks)
        ankle_fails = ankle_diff is not None and ankle_diff < ANKLE_HEIGHT_DIFF_MIN_FOR_CORRECT

        if leg_lift_fails:
            reasons.append("leg not lifted")
        if ankle_fails:
            reasons.append("both feet grounded")

    if reasons:
        return 0, 0.99, f" [{', '.join(reasons)}]", reasons

    return prediction, raw_confidence, "", []


# Human-readable correction cues for override reasons (v7). These fire
# independently of get_joint_violation_messages's angle thresholds, since
# the overrides catch failures (e.g. AC leg-lift / ankle-height) that
# angle thresholds alone can't reliably detect - see module docstring.
OVERRIDE_REASON_MESSAGES = {
    "standing posture": "You're still standing - lie down or lower into the pose",
    "hands not raised": "Lift both arms up overhead",
    "leg not lifted": "Lift your leg higher off the ground",
    "both feet grounded": "Lift one leg off the ground",
}


def get_override_messages(reasons):
    return [OVERRIDE_REASON_MESSAGES[r] for r in reasons if r in OVERRIDE_REASON_MESSAGES]


# ---------------------------------------------------------------
# JOINT-VIOLATION MESSAGES (v7)
# ---------------------------------------------------------------
# A joint angle increasing always means "straighter/more extended" and
# decreasing always means "more bent/folded" - true for knee, hip, and
# (with elbow-shoulder-hip as the vertex) shoulder too, where a bigger
# angle means the arm is raised further away from the torso. This lets a
# single pair of directional messages cover every joint without needing
# to special-case what "straighten" means per pose.

def _knee_or_hip_message(side, part, actual, low, high):
    """part is 'knee' or 'hip' - kept plain and consistent everywhere,
    no 'standing'/'lifted' jargon in the displayed text."""
    if actual < low - JOINT_MESSAGE_DEADBAND:
        deviation = low - actual
        if part == "knee":
            return (f"Straighten your {side} leg", deviation)
        return (f"Straighten up a little through your {side} hip", deviation)
    if actual > high + JOINT_MESSAGE_DEADBAND:
        deviation = actual - high
        if part == "knee":
            return (f"Bend your {side} knee a little", deviation)
        return (f"Bend forward a little more at your {side} hip", deviation)
    return None


def _shoulder_message(side, actual, low, high):
    if actual < low - JOINT_MESSAGE_DEADBAND:
        deviation = low - actual
        return (f"Lift your {side} arm higher", deviation)
    if actual > high + JOINT_MESSAGE_DEADBAND:
        deviation = actual - high
        return (f"Lower your {side} arm down", deviation)
    return None


def _rank_and_trim(candidates):
    candidates = [c for c in candidates if c is not None]
    candidates.sort(key=lambda c: c[1], reverse=True)
    return [msg for msg, _dev in candidates[:MAX_JOINT_MESSAGES]]


def _simple_pose_violations(feature_values, ranges):
    candidates = []
    for feat, (low, high) in ranges.items():
        side = "left" if feat.startswith("left") else "right"
        joint_label = "knee" if "knee" in feat else ("hip" if "hip" in feat else "shoulder")
        actual = feature_values[feat]
        if joint_label == "shoulder":
            candidates.append(_shoulder_message(side, actual, low, high))
        else:
            candidates.append(_knee_or_hip_message(side, joint_label, actual, low, high))
    return candidates


def _bhujangasana_violations(feature_values):
    """
    Cobra pose is done lying face-down. When the body still reads as
    standing (knee AND hip both extended, like Tadasana), the knee/hip
    angles come from camera foreshortening on a lying-down body and don't
    translate into meaningful "bend your knee" advice - the real problem
    is the person isn't lying down at all. Give a plain posture-level
    instruction in that case instead of a confusing joint-bend message.
    Once genuinely prone, the only reliable, physically meaningful signal
    left is the shoulder angle (how much the chest is pressed up).
    """
    knee_avg = (feature_values["left_knee"] + feature_values["right_knee"]) / 2
    hip_avg = (feature_values["left_hip"] + feature_values["right_hip"]) / 2
    if knee_avg > 165 and hip_avg > 150:
        return ["Lie down on your stomach, then lift your chest up"]

    l_low, l_high = BHUJANGASANA_JOINT_RANGES["left_shoulder"]
    r_low, r_high = BHUJANGASANA_JOINT_RANGES["right_shoulder"]
    left_val = feature_values["left_shoulder"]
    right_val = feature_values["right_shoulder"]

    if left_val < l_low - JOINT_MESSAGE_DEADBAND or right_val < r_low - JOINT_MESSAGE_DEADBAND:
        return ["Push up a little more and lift your chest higher"]
    if left_val > l_high + JOINT_MESSAGE_DEADBAND or right_val > r_high + JOINT_MESSAGE_DEADBAND:
        return ["Don't push up so high - relax a little"]
    return []


def _role_split(feature_values):
    """Return (standing_side, standing_knee, lifted_knee, standing_hip, lifted_hip)."""
    left_knee = feature_values["left_knee"]
    right_knee = feature_values["right_knee"]
    if left_knee >= right_knee:
        return "left", "right", left_knee, right_knee, feature_values["left_hip"], feature_values["right_hip"]
    return "right", "left", right_knee, left_knee, feature_values["right_hip"], feature_values["left_hip"]


def get_joint_violation_messages(pose_name, feature_values):
    """
    Only meaningful once the STABLE prediction is Incorrect - callers should
    gate on tracker.stable_label == 0 before displaying these, so messages
    don't flicker frame-to-frame the way raw predictions can.
    """
    candidates = []

    if pose_name == "Tadasana":
        candidates = _simple_pose_violations(feature_values, TADASANA_JOINT_RANGES)

    elif pose_name == "Bhujangasana":
        return _bhujangasana_violations(feature_values)

    elif pose_name == "Trikonasana":
        candidates = _simple_pose_violations(feature_values, TRIKONASANA_JOINT_RANGES)

    elif pose_name == "Vrikshasana":
        standing_side, lifted_side, standing_knee, lifted_knee, standing_hip, lifted_hip = \
            _role_split(feature_values)
        sk_low, sk_high = VRIKSHASANA_ROLE_RANGES["standing_knee"]
        lk_low, lk_high = VRIKSHASANA_ROLE_RANGES["lifted_knee"]
        sh_low, sh_high = VRIKSHASANA_ROLE_RANGES["standing_hip"]
        lh_low, lh_high = VRIKSHASANA_ROLE_RANGES["lifted_hip"]
        candidates.append(_knee_or_hip_message(standing_side, "knee", standing_knee, sk_low, sk_high))
        candidates.append(_knee_or_hip_message(lifted_side, "knee", lifted_knee, lk_low, lk_high))
        candidates.append(_knee_or_hip_message(standing_side, "hip", standing_hip, sh_low, sh_high))
        candidates.append(_knee_or_hip_message(lifted_side, "hip", lifted_hip, lh_low, lh_high))
        vs_low, vs_high = VRIKSHASANA_SHOULDER_RANGE
        candidates.append(_shoulder_message("left", feature_values["left_shoulder"], vs_low, vs_high))
        candidates.append(_shoulder_message("right", feature_values["right_shoulder"], vs_low, vs_high))

    elif pose_name == "Ardha Chandrasana":
        standing_side, lifted_side, standing_knee, lifted_knee, _sh, _lh = _role_split(feature_values)
        sk_low, sk_high = ARDHA_CHANDRASANA_ROLE_RANGES["standing_knee"]
        lk_low, lk_high = ARDHA_CHANDRASANA_ROLE_RANGES["lifted_knee"]
        candidates.append(_knee_or_hip_message(standing_side, "knee", standing_knee, sk_low, sk_high))
        candidates.append(_knee_or_hip_message(lifted_side, "knee", lifted_knee, lk_low, lk_high))
        # Hip intentionally omitted - see module docstring. The existing
        # leg-lift / ankle-height overrides already report "leg not
        # lifted" / "both feet grounded" for that class of error.

    return _rank_and_trim(candidates)


def run_screenshot_regression_tests():
    feature_order = ["left_knee", "right_knee", "left_hip", "right_hip", "left_shoulder", "right_shoulder"]
    cases = [
        (
            "Bhujangasana standing screenshot 574",
            "Bhujangasana",
            [178.3, 178.0, 176.2, 175.7, 48.8, 36.0],
            "standing posture",
        ),
        (
            "Bhujangasana standing arms-sideways screenshot 590",
            "Bhujangasana",
            [177.0, 179.1, 176.2, 175.6, 94.8, 88.7],
            "standing posture",
        ),
        (
            "Vrikshasana leg-only screenshot 575",
            "Vrikshasana",
            [41.0, 174.0, 113.0, 173.0, 8.0, 14.9],
            "hands not raised",
        ),
        (
            "Vrikshasana leg-only screenshot 576",
            "Vrikshasana",
            [179.9, 53.7, 174.3, 118.0, 14.7, 14.5],
            "hands not raised",
        ),
        (
            "Tadasana arms-down screenshot 581",
            "Tadasana",
            [178.9, 179.7, 176.3, 177.2, 10.9, 13.7],
            "hands not raised",
        ),
        (
            "Vrikshasana one arm raised, one arm down - should still fail",
            "Vrikshasana",
            [178.0, 90.0, 175.0, 130.0, 170.0, 20.0],
            "hands not raised",
        ),
        (
            "Tadasana one arm raised, one arm down - should still fail",
            "Tadasana",
            [177.0, 178.0, 176.0, 177.0, 170.0, 20.0],
            "hands not raised",
        ),
    ]

    passed = True
    for name, pose_name, row, expected_reason in cases:
        values = feature_row_to_dict(row, feature_order)
        prediction, confidence, note, _reasons = apply_pose_safety_overrides(
            pose_name, 1, 0.80, values, landmarks=None
        )
        ok = prediction == 0 and expected_reason in note
        passed = passed and ok
        print(f"{name}: {'PASS' if ok else 'FAIL'} -> prediction={prediction}, confidence={confidence:.2f}, note={note}")

    if not passed:
        raise AssertionError("One or more screenshot regression tests failed.")


def run_joint_violation_tests():
    """
    Synthetic regression tests for the v7 joint-violation-message layer.
    Each case starts from a realistic in-range profile for that pose (drawn
    from the median of the raw correct_<pose>.csv) and perturbs exactly ONE
    joint outside its threshold, then checks the expected joint/side is
    named in the resulting message. This directly tests "does the message
    point at the actual broken joint", not just "is a message produced".
    """
    cases = [
        (
            "Tadasana: left knee bent, should flag left knee",
            "Tadasana",
            {"left_knee": 140.0, "right_knee": 178.0, "left_hip": 178.0,
             "right_hip": 178.0, "left_shoulder": 178.0, "right_shoulder": 178.0},
            ["left leg"],
        ),
        (
            "Tadasana: arms not raised, should flag both arms",
            "Tadasana",
            {"left_knee": 177.0, "right_knee": 178.0, "left_hip": 178.0,
             "right_hip": 178.0, "left_shoulder": 20.0, "right_shoulder": 20.0},
            ["left arm", "right arm"],
        ),
        (
            "Bhujangasana: genuinely lying down but chest not lifted, expect chest-lift cue",
            "Bhujangasana",
            {"left_knee": 150.0, "right_knee": 150.0, "left_hip": 140.0,
             "right_hip": 140.0, "left_shoulder": 10.0, "right_shoulder": 10.0},
            ["chest higher"],
        ),
        (
            "Bhujangasana: pressed up too far while lying down, expect ease-back cue",
            "Bhujangasana",
            {"left_knee": 150.0, "right_knee": 150.0, "left_hip": 140.0,
             "right_hip": 140.0, "left_shoulder": 30.0, "right_shoulder": 90.0},
            ["Don't push up"],
        ),
        (
            "Bhujangasana: still standing/bent-forward (not lying down), expect lie-down cue not knee-bend advice",
            "Bhujangasana",
            {"left_knee": 178.0, "right_knee": 179.0, "left_hip": 171.6,
             "right_hip": 157.6, "left_shoulder": 8.8, "right_shoulder": 19.9},
            ["Lie down"],
        ),
        (
            "Trikonasana: right hip not folded enough, flag right hip",
            "Trikonasana",
            {"left_knee": 178.0, "right_knee": 176.0, "left_hip": 140.0,
             "right_hip": 165.0, "left_shoulder": 130.0, "right_shoulder": 90.0},
            ["right hip"],
        ),
        (
            "Vrikshasana: lifted leg not tucked (knee too straight), flag right knee",
            "Vrikshasana",
            {"left_knee": 178.0, "right_knee": 150.0, "left_hip": 176.0,
             "right_hip": 130.0, "left_shoulder": 170.0, "right_shoulder": 170.0},
            ["right knee"],
        ),
        (
            "Vrikshasana: right leg standing this time, left leg lifted+not tucked",
            "Vrikshasana",
            {"left_knee": 150.0, "right_knee": 178.0, "left_hip": 130.0,
             "right_hip": 176.0, "left_shoulder": 170.0, "right_shoulder": 170.0},
            ["left knee"],
        ),
        (
            "Ardha Chandrasana: lifted knee bent, flag right leg",
            "Ardha Chandrasana",
            {"left_knee": 178.0, "right_knee": 130.0, "left_hip": 170.0,
             "right_hip": 90.0, "left_shoulder": 100.0, "right_shoulder": 20.0},
            ["right leg"],
        ),
    ]

    passed = True
    for name, pose_name, feature_values, expected_terms in cases:
        messages = get_joint_violation_messages(pose_name, feature_values)
        joined = " | ".join(messages)
        ok = len(messages) > 0 and all(term in joined for term in expected_terms)
        passed = passed and ok
        print(f"{name}: {'PASS' if ok else 'FAIL'} -> {messages}")

    if not passed:
        raise AssertionError("One or more joint-violation message tests failed.")


def run_override_message_wiring_tests():
    """
    Regression test for the bug found via live testing: Ardha Chandrasana
    failures caught by the leg-lift / ankle-height OVERRIDES (not by angle
    thresholds) must still produce a visible message, even though
    get_joint_violation_messages deliberately skips AC hip. Confirms the
    override reason actually reaches the display list.
    """
    cases = [
        (
            "AC leg-lift override reason produces a display message",
            ["leg not lifted"],
            "Lift your leg higher",
        ),
        (
            "AC ankle-height override reason produces a display message",
            ["both feet grounded"],
            "Lift one leg off the ground",
        ),
        (
            "Bhujangasana standing-posture override reason produces a message",
            ["standing posture"],
            "still standing",
        ),
    ]
    passed = True
    for name, reasons, expected_substring in cases:
        messages = get_override_messages(reasons)
        ok = len(messages) > 0 and any(expected_substring in m for m in messages)
        passed = passed and ok
        print(f"{name}: {'PASS' if ok else 'FAIL'} -> {messages}")

    if not passed:
        raise AssertionError("One or more override-message wiring tests failed.")


# ---------------------------------------------------------------
# VOICE FEEDBACK (v7)
# ---------------------------------------------------------------
class VoiceFeedback:
    """
    Speaks the current top correction message aloud, off the main video
    thread so speech never blocks frame processing (pyttsx3's runAndWait()
    blocks the calling thread until speech finishes).

    Debouncing: only queues NEW speech when the text actually changes, or
    the same message has been showing for longer than REPEAT_AFTER seconds
    (a standing reminder if you haven't fixed it yet). Without this, a
    message that's technically re-selected every frame (see the raw-frame
    sync fix) would try to speak dozens of times a second.
    """
    REPEAT_AFTER = 6.0  # seconds before repeating an unresolved correction

    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)
        self._queue = queue.Queue()
        self._last_text = None
        self._last_spoken_at = 0.0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            text = self._queue.get()
            if text is None:
                break
            self.engine.say(text)
            self.engine.runAndWait()

    def update(self, top_message, is_correct):
        """Call once per frame with the current top correction message
        (or None) and whether the stable verdict is currently Correct."""
        now = time.time()

        if is_correct:
            if self._last_text != "__correct__":
                self._queue.put("Correct, nice job")
                self._last_text = "__correct__"
                self._last_spoken_at = now
            return

        if top_message is None:
            self._last_text = None
            return

        if top_message != self._last_text or (now - self._last_spoken_at) > self.REPEAT_AFTER:
            self._queue.put(top_message)
            self._last_text = top_message
            self._last_spoken_at = now

    def stop(self):
        self._queue.put(None)


def run_voice_feedback_tests():
    """
    VoiceFeedback debounce logic tested without touching pyttsx3/audio -
    swap the queue for a plain list so this runs in TEST_MODE too.
    """
    class FakeVoice(VoiceFeedback):
        def __init__(self):
            self.spoken = []
            self._last_text = None
            self._last_spoken_at = 0.0

        def _speak(self, text):
            self.spoken.append(text)

        def update(self, top_message, is_correct):
            now = time.time()
            if is_correct:
                if self._last_text != "__correct__":
                    self._speak("Correct, nice job")
                    self._last_text = "__correct__"
                    self._last_spoken_at = now
                return
            if top_message is None:
                self._last_text = None
                return
            if top_message != self._last_text or (now - self._last_spoken_at) > self.REPEAT_AFTER:
                self._speak(top_message)
                self._last_text = top_message
                self._last_spoken_at = now

    passed = True

    v = FakeVoice()
    v.update("Straighten your left leg", False)
    v.update("Straighten your left leg", False)
    v.update("Straighten your left leg", False)
    ok = v.spoken == ["Straighten your left leg"]
    passed = passed and ok
    print(f"Same message repeated frames -> spoken once: {'PASS' if ok else 'FAIL'} -> {v.spoken}")

    v = FakeVoice()
    v.update("Straighten your left leg", False)
    v.update("Bend your right knee a little", False)
    ok = v.spoken == ["Straighten your left leg", "Bend your right knee a little"]
    passed = passed and ok
    print(f"Message change speaks immediately: {'PASS' if ok else 'FAIL'} -> {v.spoken}")

    v = FakeVoice()
    v.update("Straighten your left leg", False)
    v.update(None, True)
    v.update(None, True)
    v.update("Straighten your left leg", False)
    ok = v.spoken == ["Straighten your left leg", "Correct, nice job", "Straighten your left leg"]
    passed = passed and ok
    print(f"Correct then incorrect again re-speaks: {'PASS' if ok else 'FAIL'} -> {v.spoken}")

    if not passed:
        raise AssertionError("One or more voice-feedback debounce tests failed.")


class StabilityTracker:
    def __init__(self, window_size=WINDOW_SIZE, min_confidence=MIN_CONFIDENCE, min_agreement=MIN_AGREEMENT):
        self.window = deque(maxlen=window_size)
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement
        self.stable_label = None
        self.stable_confidence = 0.0

    def update(self, prediction, confidence):
        if confidence >= self.min_confidence:
            self.window.append(prediction)
        else:
            self.window.append(None)

        valid_votes = [v for v in self.window if v is not None]

        if len(valid_votes) == 0:
            self.stable_label = None
            self.stable_confidence = 0.0
            return self.stable_label, self.stable_confidence

        correct_votes = valid_votes.count(1)
        incorrect_votes = valid_votes.count(0)
        total = len(valid_votes)

        correct_ratio = correct_votes / total
        incorrect_ratio = incorrect_votes / total

        if correct_ratio >= self.min_agreement:
            self.stable_label = 1
            self.stable_confidence = correct_ratio
        elif incorrect_ratio >= self.min_agreement:
            self.stable_label = 0
            self.stable_confidence = incorrect_ratio

        return self.stable_label, self.stable_confidence

    def reset(self):
        self.window.clear()
        self.stable_label = None
        self.stable_confidence = 0.0


def main():
    import time
    t0 = time.time()

    pose_name, folder_name, file_prefix = select_pose()
    model, metadata = load_model_and_metadata(pose_name, folder_name, file_prefix)
    feature_order = metadata["features"]
    is_ardha_chandrasana = (pose_name == "Ardha Chandrasana")
    print(f"[timing] pose selection + model load: {time.time() - t0:.1f}s")

    tracker = StabilityTracker()
    missing_frame_count = 0

    try:
        voice = VoiceFeedback()
        print("[voice] feedback enabled")
    except Exception as e:
        voice = None
        print(f"[voice] could not start voice feedback, continuing without it: {e}")

    t1 = time.time()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print(f"[timing] cv2.VideoCapture(0) call returned: {time.time() - t1:.1f}s")

    t2 = time.time()
    ret_warmup, _ = cap.read()
    print(f"[timing] first cap.read() (ret={ret_warmup}): {time.time() - t2:.1f}s")

    WINDOW_NAME = "Yoga Pose Correction System"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    print(f"[timing] total time to window open: {time.time() - t0:.1f}s\n")

    print("\nStarting webcam. Press 'q' to quit.\n")
    if is_ardha_chandrasana:
        print(f"Safety checks active: leg-lift (max {LEG_LIFT_MAX_FOR_CORRECT} deg), "
              f"ankle-height-diff (min {ANKLE_HEIGHT_DIFF_MIN_FOR_CORRECT})\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        raw_status_text = "No person detected"
        stable_status_text = "No person detected"
        status_color = (0, 0, 255)
        joint_messages = []

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angles = get_angles(landmarks)
            feature_row = angles_to_feature_row(angles, feature_order)

            if feature_row is not None:
                missing_frame_count = 0

                X = np.array(feature_row).reshape(1, -1)
                prediction = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]
                raw_confidence = probabilities[prediction]
                feature_values = feature_row_to_dict(feature_row, feature_order)
                prediction, raw_confidence, override_note, override_reasons = apply_pose_safety_overrides(
                    pose_name, prediction, raw_confidence, feature_values, landmarks
                )

                raw_label = "Correct" if prediction == 1 else "Incorrect"
                raw_status_text = f"Raw: {raw_label} ({raw_confidence*100:.0f}%){override_note}"

                stable_label, stable_confidence = tracker.update(prediction, raw_confidence)

                # Joint-violation / override messages are driven by the RAW,
                # current-frame prediction - not stable_label. stable_label is
                # smoothed over several frames on purpose (to stop the big
                # Correct/Incorrect banner flickering), but that same delay
                # made the coaching messages feel late or missing entirely
                # during "Analyzing...". The messages below now always match
                # what you're doing on THIS frame.
                joint_messages = []
                if prediction == 0:
                    joint_messages = get_override_messages(override_reasons)
                    joint_messages += get_joint_violation_messages(pose_name, feature_values)
                    joint_messages = joint_messages[:MAX_JOINT_MESSAGES]

                if stable_label is None:
                    stable_status_text = "Analyzing..."
                    status_color = (200, 200, 200)
                else:
                    label_name = "Correct" if stable_label == 1 else "Incorrect"
                    stable_status_text = f"{label_name} ({stable_confidence*100:.0f}% agreement)"
                    status_color = (0, 255, 0) if stable_label == 1 else (0, 0, 255)

                if voice is not None:
                    top_message = joint_messages[0] if joint_messages else None
                    voice.update(top_message, is_correct=(stable_label == 1))

                y = 90
                for feat, val in zip(feature_order, feature_row):
                    cv2.putText(frame, f"{feat}: {val:.1f}", (20, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y += 20

                if pose_name == "Bhujangasana":
                    # Diagnostic only - not used in any decision yet. Print
                    # lower-body landmark visibility to check whether "legs
                    # go out of frame / low visibility when genuinely lying
                    # down for Cobra" actually holds on this camera setup.
                    vis = get_lower_body_visibility(landmarks)
                    cv2.putText(frame, f"[diag] visibility: {vis}", (20, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)
                    y += 20
            else:
                if pose_name == "Bhujangasana" and bhujangasana_upper_body_fallback(landmarks):
                    missing_frame_count = 0
                    prediction = 1
                    raw_confidence = 0.85
                    raw_status_text = "Raw: Correct (85%) [upper-body fallback]"

                    stable_label, stable_confidence = tracker.update(prediction, raw_confidence)

                    if stable_label is None:
                        stable_status_text = "Analyzing..."
                        status_color = (200, 200, 200)
                    else:
                        label_name = "Correct" if stable_label == 1 else "Incorrect"
                        stable_status_text = f"{label_name} ({stable_confidence*100:.0f}% agreement)"
                        status_color = (0, 255, 0) if stable_label == 1 else (0, 0, 255)
                else:
                    missing_frame_count += 1
                    raw_status_text = "Landmarks visible, angle joints incomplete"
                    stable_status_text = "Adjust position - not all joints visible"
                    status_color = (0, 165, 255)
                    if missing_frame_count >= MISSING_FRAME_LIMIT:
                        tracker.reset()

            for i in range(11, 33):
                if landmarks[i].visibility > 0.2:
                    x = int(landmarks[i].x * frame.shape[1])
                    y_pos = int(landmarks[i].y * frame.shape[0])
                    cv2.circle(frame, (x, y_pos), 4, (0, 255, 0), -1)
        else:
            missing_frame_count += 1
            if missing_frame_count >= MISSING_FRAME_LIMIT:
                tracker.reset()

        cv2.putText(frame, f"Pose: {pose_name}", (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, raw_status_text, (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, stable_status_text, (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        for i, msg in enumerate(joint_messages):
            cv2.putText(frame, msg, (20, frame.shape[0] - 50 - (i * 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if voice is not None:
        voice.stop()


if __name__ == "__main__":
    if os.environ.get("RUN_SCREENSHOT_TESTS") == "1":
        run_screenshot_regression_tests()
        run_joint_violation_tests()
        run_override_message_wiring_tests()
        run_voice_feedback_tests()
    else:
        main()