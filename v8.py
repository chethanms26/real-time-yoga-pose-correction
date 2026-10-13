"""Real-time yoga pose correction: webcam -> MediaPipe landmarks -> per-pose
RandomForest classifier -> rule-based safety overrides -> on-screen text and
voice feedback."""

import os
from pathlib import Path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import numpy as np
import json
import warnings
import threading
import queue
import time
from collections import deque

warnings.filterwarnings(
    "ignore",
    message=r"Trying to unpickle estimator .* from version .*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names.*",
    category=UserWarning,
)

TEST_MODE = os.environ.get("RUN_SCREENSHOT_TESTS") == "1"

if not TEST_MODE:
    import joblib
    import cv2
    import mediapipe as mp
    import pyttsx3

    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,  # lite model - faster load and per-frame inference
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
BASE_PATH = str(Path(__file__).resolve().parent)

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
# STABILITY SETTINGS (used by StabilityTracker, kept for future logging use)
# ---------------------------------------------------------------
WINDOW_SIZE = 6
MIN_CONFIDENCE = 0.65
MIN_AGREEMENT = 0.6
MISSING_FRAME_LIMIT = 10

# ---------------------------------------------------------------
# ARDHA CHANDRASANA SAFETY CHECKS - catch what angle thresholds can't:
# wide bilateral stance vs. genuine one-legged lift can produce near-
# identical hip angles, so these check raw landmark position instead.
# ---------------------------------------------------------------
LEG_LIFT_MAX_FOR_CORRECT = 130  # shoulder-hip-ankle angle
ANKLE_HEIGHT_DIFF_MIN_FOR_CORRECT = 0.35  # normalized vertical ankle gap

# ---------------------------------------------------------------
# BHUJANGASANA SAFETY CHECK - rejects standing straight-body posture
# being misclassified as Cobra
# ---------------------------------------------------------------
BHUJANGASANA_STANDING_KNEE_MIN = 165
BHUJANGASANA_STANDING_HIP_MIN = 165
BHUJANGASANA_STANDING_SHOULDER_MAX = 120

# ---------------------------------------------------------------
# VRIKSHASANA / TADASANA - both arms must be raised overhead
# ---------------------------------------------------------------
VRIKSHASANA_RAISED_SHOULDER_MIN = 150
TADASANA_RAISED_SHOULDER_MIN = 150

# ---------------------------------------------------------------
# JOINT-VIOLATION MESSAGE THRESHOLDS
# Computed from 5th/95th percentile of each raw correct_<pose>.csv,
# widened slightly for margin. Format: (low, high) in degrees.
# ---------------------------------------------------------------
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

# Role-based: one leg stands (straighter knee), the other lifts - whichever
# knee is straighter that frame is treated as "standing".
VRIKSHASANA_ROLE_RANGES = {
    "standing_knee": (168, 180), "lifted_knee": (58, 105),
    "standing_hip": (168, 180), "lifted_hip": (108, 150),
}
VRIKSHASANA_SHOULDER_RANGE = (VRIKSHASANA_RAISED_SHOULDER_MIN, 180)

# AC hip is deliberately not thresholded here - unreliable; covered by the
# leg-lift/ankle-height safety checks instead.
ARDHA_CHANDRASANA_ROLE_RANGES = {
    "standing_knee": (168, 180), "lifted_knee": (155, 180),
}

JOINT_MESSAGE_DEADBAND = 4  # degrees of slack before a violation is reported
FALLBACK_INCORRECT_MESSAGE = "Adjust your posture to match the pose"
MAX_JOINT_MESSAGES = 3      # keep arm + both leg corrections when needed


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
    """Angle at vertex b, between rays b->a and b->c, in degrees."""
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


# MediaPipe pose landmark indices: 23/24 = hip, 25/26 = knee, 27/28 = ankle
LOWER_BODY_LANDMARKS = {"left_hip": 23, "right_hip": 24,
                         "left_knee": 25, "right_knee": 26,
                         "left_ankle": 27, "right_ankle": 28}


def get_lower_body_visibility(landmarks):
    """Diagnostic only - visibility scores (0-1) for hip/knee/ankle."""
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
    """Shoulder -> hip -> ankle angle, both sides (Ardha Chandrasana safety check)."""
    def get_point(i):
        return [landmarks[i].x, landmarks[i].y]

    result = {}
    if is_visible(landmarks, [12, 24, 28]):
        result["right_leg_lift"] = calculate_angle(get_point(12), get_point(24), get_point(28))
    if is_visible(landmarks, [11, 23, 27]):
        result["left_leg_lift"] = calculate_angle(get_point(11), get_point(23), get_point(27))
    return result


def get_ankle_height_diff(landmarks):
    """Vertical ankle gap normalized by torso height. None if landmarks unreliable."""
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
    """Scores visible upper-body cues when knee/ankle landmarks are too weak
    to trust (common for a close-to-floor Cobra shot)."""
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

    if wrist_y > shoulder_y + 0.08:  # hands on the floor, below shoulders
        score += 1

    if 13 in visible and 14 in visible:
        elbow_y = (landmarks[13].y + landmarks[14].y) / 2
        if elbow_y > shoulder_y:
            score += 1

    if hip_y is not None and hip_y > shoulder_y + 0.10:
        score += 1

    score += 1

    xs = [landmarks[i].x for i in visible]
    ys = [landmarks[i].y for i in visible]
    if xs and max(xs) - min(xs) > 0.35:  # wide horizontal spread, floor pose
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
    # Either arm down fails the pose (not "and" - that missed one-arm-raised cases)
    return (
        feature_values["left_shoulder"] < VRIKSHASANA_RAISED_SHOULDER_MIN or
        feature_values["right_shoulder"] < VRIKSHASANA_RAISED_SHOULDER_MIN
    )


def tadasana_arms_down_override(feature_values):
    return (
        feature_values["left_shoulder"] < TADASANA_RAISED_SHOULDER_MIN or
        feature_values["right_shoulder"] < TADASANA_RAISED_SHOULDER_MIN
    )


def apply_pose_safety_overrides(pose_name, prediction, raw_confidence, feature_values, landmarks=None):
    """Rule-based backstop on top of the ML prediction - see file docstring."""
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


OVERRIDE_REASON_MESSAGES = {
    "standing posture": "You're still standing - lie down or lower into the pose",
    "leg not lifted": "Lift your leg higher off the ground",
    "both feet grounded": "Lift one leg off the ground",
}


def get_arms_message(feature_values, min_threshold):
    """Names the specific arm still down, instead of always saying "lift both"."""
    left_down = feature_values["left_shoulder"] < min_threshold
    right_down = feature_values["right_shoulder"] < min_threshold
    if left_down and right_down:
        return "Lift both arms up overhead"
    if left_down:
        return "Your right arm's up - lift your left arm too"
    if right_down:
        return "Your left arm's up - lift your right arm too"
    return "Lift both arms up overhead"  # fallback, shouldn't normally hit


def get_override_messages(reasons, feature_values=None, arm_threshold=None):
    messages = []
    for r in reasons:
        if r == "hands not raised" and feature_values is not None and arm_threshold is not None:
            messages.append(get_arms_message(feature_values, arm_threshold))
        elif r in OVERRIDE_REASON_MESSAGES:
            messages.append(OVERRIDE_REASON_MESSAGES[r])
    return messages


# A bigger angle always means straighter/more extended; smaller always means
# more bent/folded - true for knee, hip, and shoulder alike, so one pair of
# directional messages covers every joint.

def _knee_or_hip_message(side, part, actual, low, high):
    if actual < low - JOINT_MESSAGE_DEADBAND:
        deviation = low - actual
        if part == "knee":
            return (f"Straighten your {side} leg", deviation)
        return (f"Straighten up a little through your {side} hip", deviation)
    if actual > high + JOINT_MESSAGE_DEADBAND:
        deviation = actual - high
        if part == "knee":
            return (f"Fold your {side} knee a little", deviation)
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
    candidates.sort(key=lambda c: c[1], reverse=True)  # worst deviation first
    return [msg for msg, _dev in candidates[:MAX_JOINT_MESSAGES]]


def _simple_pose_violations(feature_values, ranges):
    """Per-column threshold check for poses where left/right are fixed sides."""
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
    """Cobra is lying face-down. If the body still reads as standing (knee
    AND hip both extended), the real issue is "not lying down" - a plain
    posture instruction, not a knee-bend message. Once genuinely prone,
    the only reliable signal left is the shoulder/chest-lift angle."""
    knee_avg = (feature_values["left_knee"] + feature_values["right_knee"]) / 2
    hip_avg = (feature_values["left_hip"] + feature_values["right_hip"]) / 2
    if knee_avg > 165 and hip_avg > 150:
        return ["Lie face-down on the floor for Cobra pose"]

    l_low, l_high = BHUJANGASANA_JOINT_RANGES["left_shoulder"]
    r_low, r_high = BHUJANGASANA_JOINT_RANGES["right_shoulder"]
    left_val = feature_values["left_shoulder"]
    right_val = feature_values["right_shoulder"]

    if left_val < l_low - JOINT_MESSAGE_DEADBAND or right_val < r_low - JOINT_MESSAGE_DEADBAND:
        return ["You're lying down - now press up and lift your chest a little higher"]
    if left_val > l_high + JOINT_MESSAGE_DEADBAND or right_val > r_high + JOINT_MESSAGE_DEADBAND:
        return ["You're lying down and lifted well - just ease back a little, don't push up so far"]
    return []


def _role_split(feature_values):
    """Whichever knee is straighter this frame = standing leg, other = lifted leg."""
    left_knee = feature_values["left_knee"]
    right_knee = feature_values["right_knee"]
    if left_knee >= right_knee:
        return "left", "right", left_knee, right_knee, feature_values["left_hip"], feature_values["right_hip"]
    return "right", "left", right_knee, left_knee, feature_values["right_hip"], feature_values["left_hip"]


def get_joint_violation_messages(pose_name, feature_values):
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

    return _rank_and_trim(candidates)


def run_screenshot_regression_tests():
    """Real bug cases confirmed from live screenshots - see PROJECT_CONTEXT."""
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
    """Perturb one joint from a realistic in-range profile, check the message names it."""
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
            ["lift your chest"],
        ),
        (
            "Bhujangasana: pressed up too far while lying down, expect ease-back cue",
            "Bhujangasana",
            {"left_knee": 150.0, "right_knee": 150.0, "left_hip": 140.0,
             "right_hip": 140.0, "left_shoulder": 30.0, "right_shoulder": 90.0},
            ["ease back"],
        ),
        (
            "Bhujangasana: still standing/bent-forward (not lying down), expect lie-down cue not knee-bend advice",
            "Bhujangasana",
            {"left_knee": 178.0, "right_knee": 179.0, "left_hip": 171.6,
             "right_hip": 157.6, "left_shoulder": 8.8, "right_shoulder": 19.9},
            ["Lie face-down"],
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
    """Confirms override reasons (leg-lift, ankle-height, standing posture)
    actually reach the display list, not just the tiny debug status line."""
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


def run_arms_message_tests():
    """Confirms the specific arm still down is named, not a generic "lift both"."""
    cases = [
        ("Vrikshasana: both arms down", "Vrikshasana", VRIKSHASANA_RAISED_SHOULDER_MIN,
         {"left_shoulder": 20.0, "right_shoulder": 20.0}, "Lift both arms up overhead"),
        ("Vrikshasana: only left arm down", "Vrikshasana", VRIKSHASANA_RAISED_SHOULDER_MIN,
         {"left_shoulder": 20.0, "right_shoulder": 170.0}, "left arm too"),
        ("Vrikshasana: only right arm down", "Vrikshasana", VRIKSHASANA_RAISED_SHOULDER_MIN,
         {"left_shoulder": 170.0, "right_shoulder": 20.0}, "right arm too"),
        ("Tadasana: both arms down", "Tadasana", TADASANA_RAISED_SHOULDER_MIN,
         {"left_shoulder": 15.0, "right_shoulder": 15.0}, "Lift both arms up overhead"),
        ("Tadasana: only left arm down", "Tadasana", TADASANA_RAISED_SHOULDER_MIN,
         {"left_shoulder": 15.0, "right_shoulder": 175.0}, "left arm too"),
        ("Tadasana: only right arm down", "Tadasana", TADASANA_RAISED_SHOULDER_MIN,
         {"left_shoulder": 175.0, "right_shoulder": 15.0}, "right arm too"),
    ]

    passed = True
    for name, pose_name, threshold, feature_values, expected_substring in cases:
        messages = get_override_messages(["hands not raised"], feature_values, threshold)
        ok = len(messages) == 1 and expected_substring in messages[0]
        passed = passed and ok
        print(f"{name}: {'PASS' if ok else 'FAIL'} -> {messages}")

    if not passed:
        raise AssertionError("One or more arms-message tests failed.")


class VoiceFeedback:
    """Speaks the top unique correction messages, off the main video thread so
    speech never blocks frame processing. The screen may rank several issues,
    but the voice layer keeps the newest unique subset rather than only the
    single worst item. Debounced (won't repeat unless the message set changes
    or REPEAT_AFTER seconds pass) and drains any stale queued message before
    speaking a new one, so it never falls behind with old corrections."""
    REPEAT_AFTER = 6.0
    MAX_SPOKEN_MESSAGES = 2

    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)
        self._queue = queue.Queue()
        self._last_target = None
        self._last_spoken_at = 0.0
        self._speech_lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                with self._speech_lock:
                    self.engine.say(text)
                    self.engine.runAndWait()
            except Exception as e:
                print(f"[voice] speech error, skipping this line: {e}")

    def _drain_queue(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _build_payload(self, messages):
        payload = []
        seen = set()
        for msg in messages:
            if not msg or msg in seen:
                continue
            seen.add(msg)
            payload.append(msg)
            if len(payload) >= self.MAX_SPOKEN_MESSAGES:
                break
        return payload

    def update(self, messages, is_correct):
        """messages = current ranked correction list, worst first. Speak the
        top unique messages, capped to MAX_SPOKEN_MESSAGES, as one serialized
        utterance so pyttsx3 does not overlap separate speech calls."""
        now = time.time()

        if is_correct:
            target = ("__correct__",)
            payload = ["Correct, nice job"]
        else:
            if not messages:
                self._drain_queue()
                self._last_target = None
                return
            payload = self._build_payload(messages)
            if not payload:
                self._last_target = None
                return
            target = tuple(payload)

        if target != self._last_target or (now - self._last_spoken_at) > self.REPEAT_AFTER:
            self._drain_queue()
            utterance = ". ".join(payload)
            self._queue.put(utterance)
            self._last_target = target
            self._last_spoken_at = now

    def stop(self):
        self._drain_queue()
        self._queue.put(None)


def update_voice_for_verdict(voice, verdict_label, joint_messages):
    if voice is None:
        return
    if verdict_label == 1:
        voice.update([], is_correct=True)
    elif verdict_label == 0:
        voice.update(joint_messages, is_correct=False)
    else:
        voice.update([], is_correct=False)


def run_instant_verdict_tests():
    """Confirms InstantVerdict needs 3 agreeing frames (fast), not 6 (the old lag)."""
    passed = True

    v = InstantVerdict()
    v.update(1, 0.9)
    label, conf = v.update(1, 0.9)
    ok = label is None
    passed = passed and ok
    print(f"Two frames -> no verdict yet (needs 3 agreeing): {'PASS' if ok else 'FAIL'} -> {label}")

    label, conf = v.update(1, 0.9)
    ok = label == 1
    passed = passed and ok
    print(f"Three agreeing frames -> Correct confirmed: {'PASS' if ok else 'FAIL'} -> {label}")

    label, conf = v.update(0, 0.8)
    ok = label is None
    passed = passed and ok
    print(f"Single disagreeing frame after Correct -> back to no verdict (not stuck stale): {'PASS' if ok else 'FAIL'} -> {label}")
        
    v2 = InstantVerdict()
    v2.update(0, 0.8)
    v2.update(0, 0.8)
    label, conf = v2.update(0, 0.8)
    ok = label == 0
    passed = passed and ok
    print(f"Three agreeing Incorrect frames -> Incorrect confirmed: {'PASS' if ok else 'FAIL'} -> {label}")

    if not passed:
        raise AssertionError("One or more InstantVerdict tests failed.")


def run_voice_feedback_tests():
    """Uses the real VoiceFeedback.update()/_drain_queue() logic (queue faked,
    no real audio) to exercise one-at-a-time speaking + backlog prevention."""
    class FakeQueue:
        def __init__(self):
            self.items = []
            self.history = []

        def put(self, item):
            self.items.append(item)
            self.history.append(item)

        def empty(self):
            return len(self.items) == 0

        def get_nowait(self):
            if not self.items:
                raise queue.Empty
            return self.items.pop(0)

    class FakeVoice(VoiceFeedback):
        def __init__(self):
            self._queue = FakeQueue()
            self._last_target = None
            self._last_spoken_at = 0.0

    passed = True

    v = FakeVoice()
    v.update(["Straighten your left leg"], False)
    v.update(["Straighten your left leg"], False)
    v.update(["Straighten your left leg"], False)
    ok = v._queue.history == ["Straighten your left leg"]
    passed = passed and ok
    print(f"Same message repeated frames -> spoken once: {'PASS' if ok else 'FAIL'} -> {v._queue.history}")

    v = FakeVoice()
    v.update(["Straighten your left leg", "Lift your right arm higher"], False)
    ok = v._queue.history == ["Straighten your left leg. Lift your right arm higher"]
    passed = passed and ok
    print(f"Two unique corrections at once -> both top suggestions are spoken in one serialized utterance: "
          f"{'PASS' if ok else 'FAIL'} -> {v._queue.history}")

    v = FakeVoice()
    v.update(["Straighten your left leg", "Lift your right arm higher"], False)
    v.update(["Lift your right arm higher"], False)
    ok = v._queue.history == ["Straighten your left leg. Lift your right arm higher", "Lift your right arm higher"]
    passed = passed and ok
    print(f"When the rank changes, the new top unique suggestion is spoken: "
          f"{'PASS' if ok else 'FAIL'} -> {v._queue.history}")

    v = FakeVoice()
    v.update(["Straighten your left leg"], False)
    v.update(["Bend your right knee a little"], False)
    ok = v._queue.history == ["Straighten your left leg", "Bend your right knee a little"]
    passed = passed and ok
    print(f"Message change speaks immediately: {'PASS' if ok else 'FAIL'} -> {v._queue.history}")

    v = FakeVoice()
    v.update(["Straighten your left leg"], False)
    v.update(None, True)
    v.update(None, True)
    v.update(["Straighten your left leg"], False)
    ok = v._queue.history == ["Straighten your left leg", "Correct, nice job", "Straighten your left leg"]
    passed = passed and ok
    print(f"Correct then incorrect again re-speaks: {'PASS' if ok else 'FAIL'} -> {v._queue.history}")

    v = FakeVoice()
    v.update(["Straighten your left leg"], False)
    v.update(["Bend your right knee a little"], False)
    v.update(["Lift your left arm higher"], False)
    ok = v._queue.items == ["Lift your left arm higher"]
    passed = passed and ok
    print(f"Rapid different corrections -> no backlog, only the latest is still queued: "
          f"{'PASS' if ok else 'FAIL'} -> queued={v._queue.items}")

    v = FakeVoice()
    v.update(["Straighten your left leg"], False)
    v.update(["Bend your right knee a little"], False)
    v.update(None, True)
    ok = v._queue.items == ["Correct, nice job"]
    passed = passed and ok
    print(f"Fixing the pose clears pending corrections immediately: "
          f"{'PASS' if ok else 'FAIL'} -> queued={v._queue.items}")

    if not passed:
        raise AssertionError("One or more voice-feedback debounce tests failed.")


def run_fallback_message_tests():
    """Confirms an Incorrect verdict never reaches update_voice_for_verdict
    with an empty message list."""
    class FakeVoice:
        def __init__(self):
            self.calls = []
        def update(self, messages, is_correct):
            self.calls.append((messages, is_correct))

    passed = True
    voice = FakeVoice()
    update_voice_for_verdict(voice, 0, [])
    ok = voice.calls == [([], False)]
    passed = passed and ok
    print(f"Empty joint_messages still reaches voice as empty (main() must pre-fill it): "
          f"{'PASS' if ok else 'FAIL'} -> {voice.calls}")

    voice = FakeVoice()
    update_voice_for_verdict(voice, 0, [FALLBACK_INCORRECT_MESSAGE])
    ok = voice.calls == [([FALLBACK_INCORRECT_MESSAGE], False)]
    passed = passed and ok
    print(f"Fallback message reaches the voice layer: {'PASS' if ok else 'FAIL'} -> {voice.calls}")

    if not passed:
        raise AssertionError("One or more fallback-message tests failed.")


def run_all_pose_voice_tests():
    """Confirms every selectable asana uses the confirmed-correct voice path."""
    class FakeVoice:
        def __init__(self):
            self.calls = []

        def update(self, messages, is_correct):
            self.calls.append((messages, is_correct))

    passed = True
    for pose_name, _folder_name, _file_prefix in POSE_OPTIONS.values():
        voice = FakeVoice()
        update_voice_for_verdict(voice, 1, [])
        ok = voice.calls == [([], True)]
        passed = passed and ok
        print(f"{pose_name}: confirmed correct -> 'Correct, nice job' path: "
              f"{'PASS' if ok else 'FAIL'}")

    if not passed:
        raise AssertionError("One or more asana voice-path tests failed.")


class StabilityTracker:
    """Majority-vote smoothing over a window of frames. No longer drives
    on-screen display (see InstantVerdict) - kept for future session-stats use."""
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


class InstantVerdict:
    """Drives the big Correct/Incorrect banner - needs CONFIRM_FRAMES
    agreeing frames (fast) instead of StabilityTracker's window (was
    laggy). If the model itself is genuinely near 50/50 on a borderline
    pose, some flicker is a model-calibration issue, not fixable by
    tuning this further."""
    CONFIRM_FRAMES = 3

    def __init__(self):
        self.window = deque(maxlen=self.CONFIRM_FRAMES)
        self.label = None
        self.confidence = 0.0

    def update(self, prediction, confidence):
        self.window.append((prediction, confidence))
        if len(self.window) == self.CONFIRM_FRAMES and all(p == prediction for p, _ in self.window):
            self.label = prediction
            self.confidence = sum(c for _, c in self.window) / len(self.window)
        else:
            self.label = None
            self.confidence = 0.0
        return self.label, self.confidence

    def reset(self):
        self.window.clear()
        self.label = None
        self.confidence = 0.0


def main():
    import time
    t0 = time.time()

    pose_name, folder_name, file_prefix = select_pose()
    model, metadata = load_model_and_metadata(pose_name, folder_name, file_prefix)
    feature_order = metadata["features"]
    is_ardha_chandrasana = (pose_name == "Ardha Chandrasana")
    print(f"[timing] pose selection + model load: {time.time() - t0:.1f}s")

    tracker = StabilityTracker()
    verdict = InstantVerdict()
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

    print("\nStarting webcam. Press 'q' to quit, 'd' to toggle the debug overlay.\n")
    if is_ardha_chandrasana:
        print(f"Safety checks active: leg-lift (max {LEG_LIFT_MAX_FOR_CORRECT} deg), "
              f"ankle-height-diff (min {ANKLE_HEIGHT_DIFF_MIN_FOR_CORRECT})\n")

    fps_frame_times = deque(maxlen=30)
    current_fps = 0.0
    show_debug = False  # press 'd' to reveal angle values + diagnostics

    while True:
        frame_start = time.time()
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
        verdict_label = None

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
                verdict_label, verdict_confidence = verdict.update(prediction, raw_confidence)

                # Messages driven by the raw, current-frame prediction - always
                # matches what you're doing right now, unlike the smoothed banner.
                joint_messages = []
                if prediction == 0:
                    arm_threshold = (VRIKSHASANA_RAISED_SHOULDER_MIN if pose_name == "Vrikshasana"
                                      else TADASANA_RAISED_SHOULDER_MIN if pose_name == "Tadasana"
                                      else None)
                    joint_messages = get_override_messages(override_reasons, feature_values, arm_threshold)
                    joint_violations = get_joint_violation_messages(pose_name, feature_values)
                    if "hands not raised" in override_reasons:
                        joint_violations = [
                            message for message in joint_violations
                            if "arm" not in message.lower()
                        ]
                    joint_messages += joint_violations
                    joint_messages = joint_messages[:MAX_JOINT_MESSAGES]
                    if not joint_messages:
                        joint_messages = [FALLBACK_INCORRECT_MESSAGE]

                if verdict_label is None:
                    stable_status_text = "..."
                    status_color = (200, 200, 200)
                else:
                    label_name = "Correct" if verdict_label == 1 else "Incorrect"
                    stable_status_text = f"{label_name} ({verdict_confidence*100:.0f}%)"
                    status_color = (0, 255, 0) if verdict_label == 1 else (0, 0, 255)

                y = 90
                if show_debug:
                    for feat, val in zip(feature_order, feature_row):
                        cv2.putText(frame, f"{feat}: {val:.1f}", (20, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y += 20

                    if pose_name == "Bhujangasana":
                        vis = get_lower_body_visibility(landmarks)
                        cv2.putText(frame, f"[diag] visibility: {vis}", (20, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)
                        y += 20

                    if pose_name == "Ardha Chandrasana":
                        leg_lift_diag = get_leg_lift_angles(landmarks)
                        ankle_diag = get_ankle_height_diff(landmarks)
                        leg_lift_text = ", ".join(
                            f"{name.replace('_leg_lift', '')}: {float(val):.1f}"
                            for name, val in leg_lift_diag.items()
                        ) or "n/a"
                        cv2.putText(frame, f"[diag] leg-lift angles: {leg_lift_text}", (20, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)
                        y += 20
                        ankle_text = "n/a (landmarks not confident)" if ankle_diag is None else f"{ankle_diag:.2f}"
                        cv2.putText(frame, f"[diag] ankle-height-diff: {ankle_text}", (20, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)
                        y += 20
            else:
                if pose_name == "Bhujangasana" and bhujangasana_upper_body_fallback(landmarks):
                    missing_frame_count = 0
                    prediction = 1
                    raw_confidence = 0.85
                    raw_status_text = "Raw: Correct (85%) [upper-body fallback]"

                    stable_label, stable_confidence = tracker.update(prediction, raw_confidence)
                    verdict_label, verdict_confidence = verdict.update(prediction, raw_confidence)

                    if verdict_label is None:
                        stable_status_text = "..."
                        status_color = (200, 200, 200)
                    else:
                        label_name = "Correct" if verdict_label == 1 else "Incorrect"
                        stable_status_text = f"{label_name} ({verdict_confidence*100:.0f}%)"
                        status_color = (0, 255, 0) if verdict_label == 1 else (0, 0, 255)
                else:
                    missing_frame_count += 1
                    raw_status_text = "Landmarks visible, angle joints incomplete"
                    stable_status_text = "Adjust position - not all joints visible"
                    status_color = (0, 165, 255)
                    if missing_frame_count >= MISSING_FRAME_LIMIT:
                        tracker.reset()
                        verdict.reset()

            for i in range(11, 33):
                if landmarks[i].visibility > 0.2:
                    x = int(landmarks[i].x * frame.shape[1])
                    y_pos = int(landmarks[i].y * frame.shape[0])
                    cv2.circle(frame, (x, y_pos), 4, (0, 255, 0), -1)
        else:
            missing_frame_count += 1
            if missing_frame_count >= MISSING_FRAME_LIMIT:
                tracker.reset()
                verdict.reset()

        update_voice_for_verdict(voice, verdict_label, joint_messages)

        cv2.putText(frame, f"Pose: {pose_name}", (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, raw_status_text, (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        if not show_debug:
            cv2.putText(frame, "Press 'd' for debug info", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
        cv2.putText(frame, stable_status_text, (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        for i, msg in enumerate(joint_messages):
            cv2.putText(frame, msg, (20, frame.shape[0] - 50 - (i * 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

        # Live FPS + derived message/banner sync time, in ms
        fps_frame_times.append(time.time() - frame_start)
        if fps_frame_times:
            avg_frame_time = sum(fps_frame_times) / len(fps_frame_times)
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        message_delay_ms = (1000.0 / current_fps) if current_fps > 0 else 0.0
        banner_delay_ms = message_delay_ms * InstantVerdict.CONFIRM_FRAMES
        fps_text = (f"FPS: {current_fps:.0f}  |  messages: ~{message_delay_ms:.0f}ms  |  "
                    f"banner: ~{banner_delay_ms:.0f}ms")
        cv2.putText(frame, fps_text, (frame.shape[1] - 480, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d'):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()
    if voice is not None:
        voice.stop()


if __name__ == "__main__":
    if os.environ.get("RUN_SCREENSHOT_TESTS") == "1":
        run_screenshot_regression_tests()
        run_joint_violation_tests()
        run_override_message_wiring_tests()
        run_arms_message_tests()
        run_instant_verdict_tests()
        run_voice_feedback_tests()
        run_fallback_message_tests()
        run_all_pose_voice_tests()
    else:
        main()