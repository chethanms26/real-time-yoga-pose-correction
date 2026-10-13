"""
Real-Time Yoga Pose Correction - Step 3 (v6): Live Prediction + Stability
Analysis + targeted real-time safety checks.

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

v7 adds:
3) Bhujangasana upper-body fallback for prone webcam cases where MediaPipe
   cannot reliably expose all knee/ankle landmarks.
4) Tadasana must not accept simple arms-down standing when the demo expects
   the stretched/raised-arm version.
"""

import os
from pathlib import Path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import json
from collections import deque

TEST_MODE = os.environ.get("RUN_SCREENSHOT_TESTS") == "1"

if not TEST_MODE:
    import joblib
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
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
# STABILITY SETTINGS
# ---------------------------------------------------------------
WINDOW_SIZE = 8
MIN_CONFIDENCE = 0.65
MIN_AGREEMENT = 0.65
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
    return (
        feature_values["left_shoulder"] < VRIKSHASANA_RAISED_SHOULDER_MIN and
        feature_values["right_shoulder"] < VRIKSHASANA_RAISED_SHOULDER_MIN
    )


def tadasana_arms_down_override(feature_values):
    return (
        feature_values["left_shoulder"] < TADASANA_RAISED_SHOULDER_MIN and
        feature_values["right_shoulder"] < TADASANA_RAISED_SHOULDER_MIN
    )


def apply_pose_safety_overrides(pose_name, prediction, raw_confidence, feature_values, landmarks=None):
    if prediction != 1:
        return prediction, raw_confidence, ""

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
        return 0, 0.99, f" [{', '.join(reasons)}]"

    return prediction, raw_confidence, ""


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
    ]

    passed = True
    for name, pose_name, row, expected_reason in cases:
        values = feature_row_to_dict(row, feature_order)
        prediction, confidence, note = apply_pose_safety_overrides(
            pose_name, 1, 0.80, values, landmarks=None
        )
        ok = prediction == 0 and expected_reason in note
        passed = passed and ok
        print(f"{name}: {'PASS' if ok else 'FAIL'} -> prediction={prediction}, confidence={confidence:.2f}, note={note}")

    if not passed:
        raise AssertionError("One or more screenshot regression tests failed.")


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
    pose_name, folder_name, file_prefix = select_pose()
    model, metadata = load_model_and_metadata(pose_name, folder_name, file_prefix)
    feature_order = metadata["features"]
    is_ardha_chandrasana = (pose_name == "Ardha Chandrasana")

    tracker = StabilityTracker()
    missing_frame_count = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    WINDOW_NAME = "Yoga Pose Correction System"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

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
                prediction, raw_confidence, override_note = apply_pose_safety_overrides(
                    pose_name, prediction, raw_confidence, feature_values, landmarks
                )

                raw_label = "Correct" if prediction == 1 else "Incorrect"
                raw_status_text = f"Raw: {raw_label} ({raw_confidence*100:.0f}%){override_note}"

                stable_label, stable_confidence = tracker.update(prediction, raw_confidence)

                if stable_label is None:
                    stable_status_text = "Analyzing..."
                    status_color = (200, 200, 200)
                else:
                    label_name = "Correct" if stable_label == 1 else "Incorrect"
                    stable_status_text = f"{label_name} ({stable_confidence*100:.0f}% agreement)"
                    status_color = (0, 255, 0) if stable_label == 1 else (0, 0, 255)

                y = 90
                for feat, val in zip(feature_order, feature_row):
                    cv2.putText(frame, f"{feat}: {val:.1f}", (20, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if os.environ.get("RUN_SCREENSHOT_TESTS") == "1":
        run_screenshot_regression_tests()
    else:
        main()
