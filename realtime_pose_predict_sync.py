"""
Synchronized real-time yoga pose runner.

This runner reuses the project's existing detection/model/correction code from
v8.py, but fixes the synchronization boundary:

    raw frame -> 3-frame confirmed state -> display + corrections + TTS

The same confirmed state is the single source of truth for all user-visible
feedback. This prevents the banner, correction text and audio from describing
different frames.

Keep this file in the repository root, beside v8.py and
the five pose/model folders.
"""

from pathlib import Path
import time
import os
from collections import deque

# Keep third-party startup logs from obscuring the login prompt.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

def _load_runtime_modules_quietly():
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 2)
            import cv2
            import numpy as np
            import v8 as app
            return cv2, np, app
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)


cv2, np, app = _load_runtime_modules_quietly()
from session_login import finish, login

if os.environ.get("YOGA_VERBOSE") != "1":
    quiet_stderr_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(quiet_stderr_fd, 2)
    os.close(quiet_stderr_fd)


# Make the model paths portable instead of using the original author's
# machine-specific absolute Windows path.
app.BASE_PATH = str(Path(__file__).resolve().parent)


class SyncedVoice:
    """
    Audio controller driven only by confirmed display states.

    The queue is "latest state wins": stale messages waiting in the queue are
    removed before the newest confirmed state is queued. Audio is never driven
    directly from raw frame predictions.
    """

    REPEAT_AFTER = 8.0

    def __init__(self):
        import pyttsx3
        import queue

        self._queue = queue.Queue()
        self._queue_module = queue
        self._stop_sentinel = object()
        self._last_state = None
        self._last_queued_at = 0.0
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", 175)
        self._engine.setProperty("volume", 1.0)

    def _drain_pending(self):
        # Remove only items that have not started speaking yet.
        while True:
            try:
                self._queue.get_nowait()
            except self._queue_module.Empty:
                break

    def set_state(self, label, messages):
        """
        Queue speech only after the synchronized display state is confirmed.

        `label` is 1 for Correct, 0 for Incorrect.
        `messages` are exactly the same correction messages rendered on screen.
        """
        now = time.time()
        target = ("correct",) if label == 1 else ("incorrect", *messages)

        # Ignore identical states until the periodic reminder interval.
        if target == self._last_state and (now - self._last_queued_at) < self.REPEAT_AFTER:
            return

        self._drain_pending()

        if label == 1:
            text = "Correct, nice job."
        elif messages:
            # Speak each current correction in one serialized utterance so no
            # valid feedback is silently dropped.
            text = ". ".join(dict.fromkeys(messages))
        else:
            text = "Adjust your position."

        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:
            print(f"[voice] speech error: {exc}")
        self._last_state = target
        self._last_queued_at = now

    def clear(self):
        self._drain_pending()
        self._last_state = None

    def stop(self):
        self._drain_pending()


class ConfirmedState:
    """
    A small state synchronizer.

    Three consecutive identical raw predictions are required before the
    state becomes visible/audible. The correction messages are taken from the
    newest frame in that same confirmed run, so display and audio share one
    state transition.
    """

    CONFIRM_FRAMES = 3

    def __init__(self):
        self.samples = deque(maxlen=self.CONFIRM_FRAMES)
        self.label = None
        self.confidence = 0.0
        self.messages = []
        self._generation = 0

    @property
    def generation(self):
        return self._generation

    def update(self, raw_label, raw_confidence, messages):
        sample = {
            "label": int(raw_label),
            "confidence": float(raw_confidence),
            "messages": list(messages),
        }
        self.samples.append(sample)

        if (
            len(self.samples) == self.CONFIRM_FRAMES
            and all(s["label"] == sample["label"] for s in self.samples)
        ):
            new_label = sample["label"]
            new_confidence = sum(s["confidence"] for s in self.samples) / len(self.samples)
            new_messages = list(sample["messages"])

            changed = (
                self.label != new_label
                or self.messages != new_messages
            )

            self.label = new_label
            self.confidence = new_confidence
            self.messages = new_messages

            if changed:
                self._generation += 1

            return True

        return False

    def reset(self):
        self.samples.clear()
        self.label = None
        self.confidence = 0.0
        self.messages = []
        self._generation += 1


def choose_pose_model():
    pose_name, folder_name, file_prefix = app.select_pose()
    model, metadata = app.load_model_and_metadata(
        pose_name, folder_name, file_prefix
    )
    return pose_name, model, metadata


def main():
    t0 = time.time()

    session = login()
    pose_name = None
    try:
        pose_name, model, metadata = choose_pose_model()
    except Exception:
        finish(session, pose_name)
        raise
    feature_order = metadata["features"]
    is_ardha_chandrasana = pose_name == "Ardha Chandrasana"

    print(f"[timing] startup/model load: {time.time() - t0:.1f}s")
    print(f"[paths] models loaded from: {app.BASE_PATH}")

    confirmed = ConfirmedState()
    missing_frame_count = 0

    voice = None
    try:
        voice = SyncedVoice()
        print("[voice] synchronized feedback enabled")
    except Exception as exc:
        print(f"[voice] could not start voice feedback: {exc}")

    # Keep the original Windows DirectShow behavior when available, while
    # allowing the code to fall back to the default backend.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[camera] DirectShow open failed; trying default camera backend...")
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        finish(session, pose_name)
        raise RuntimeError(
            "Could not open webcam. Check Windows Camera permissions and close "
            "other apps that may be using the camera."
        )

    window_name = "Yoga Pose Correction System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\nStarting webcam. Press 'q' to quit, 'd' to toggle debug overlay.\n")

    if is_ardha_chandrasana:
        print(
            "[safety] Ardha Chandrasana checks: "
            f"leg-lift <= {app.LEG_LIFT_MAX_FOR_CORRECT} deg, "
            f"ankle-height-diff >= {app.ANKLE_HEIGHT_DIFF_MIN_FOR_CORRECT}\n"
        )

    fps_times = deque(maxlen=30)
    show_debug = False
    last_audio_generation = -1

    try:
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[camera] Failed to read frame; stopping.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = app.pose.process(rgb)
            rgb.flags.writeable = True

            raw_status_text = "No person detected"
            display_status_text = "Analyzing..."
            status_color = (200, 200, 200)
            display_messages = confirmed.messages

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                angles = app.get_angles(landmarks)
                feature_row = app.angles_to_feature_row(angles, feature_order)

                raw_label = None
                raw_confidence = 0.0
                raw_messages = []

                if feature_row is not None:
                    missing_frame_count = 0

                    X = np.array(feature_row).reshape(1, -1)
                    prediction = int(model.predict(X)[0])
                    probabilities = model.predict_proba(X)[0]
                    raw_confidence = float(probabilities[prediction])

                    feature_values = app.feature_row_to_dict(
                        feature_row, feature_order
                    )

                    (
                        prediction,
                        raw_confidence,
                        override_note,
                        override_reasons,
                    ) = app.apply_pose_safety_overrides(
                        pose_name,
                        prediction,
                        raw_confidence,
                        feature_values,
                        landmarks,
                    )

                    prediction = int(prediction)
                    raw_label = prediction

                    raw_status_text = (
                        f"Raw: {'Correct' if prediction == 1 else 'Incorrect'} "
                        f"({raw_confidence * 100:.0f}%){override_note}"
                    )

                    if prediction == 0:
                        arm_threshold = (
                            app.VRIKSHASANA_RAISED_SHOULDER_MIN
                            if pose_name == "Vrikshasana"
                            else app.TADASANA_RAISED_SHOULDER_MIN
                            if pose_name == "Tadasana"
                            else None
                        )

                        raw_messages = app.get_override_messages(
                            override_reasons,
                            feature_values,
                            arm_threshold,
                        )

                        joint_violations = app.get_joint_violation_messages(
                            pose_name,
                            feature_values,
                        )
                        if "hands not raised" in override_reasons:
                            joint_violations = [
                                message for message in joint_violations
                                if "arm" not in message.lower()
                            ]

                        raw_messages += joint_violations
                        raw_messages = raw_messages[: app.MAX_JOINT_MESSAGES]

                elif (
                    pose_name == "Bhujangasana"
                    and app.bhujangasana_upper_body_fallback(landmarks)
                ):
                    # Preserve the project's original Cobra fallback.
                    missing_frame_count = 0
                    raw_label = 1
                    raw_confidence = 0.85
                    raw_status_text = "Raw: Correct (85%) [upper-body fallback]"
                    raw_messages = []

                else:
                    missing_frame_count += 1
                    raw_status_text = "Landmarks visible, angle joints incomplete"

                # The critical synchronization boundary:
                # ONLY confirmed states become display/audible states.
                if raw_label is not None:
                    confirmed_now = confirmed.update(
                        raw_label,
                        raw_confidence,
                        raw_messages,
                    )

                    if confirmed.label is None:
                        display_status_text = "Analyzing..."
                        status_color = (200, 200, 200)
                        display_messages = []
                    else:
                        label_name = "Correct" if confirmed.label == 1 else "Incorrect"
                        display_status_text = (
                            f"{label_name} ({confirmed.confidence * 100:.0f}%)"
                        )
                        status_color = (
                            (0, 255, 0)
                            if confirmed.label == 1
                            else (0, 0, 255)
                        )
                        display_messages = confirmed.messages

                        # Speak the same confirmed state that is rendered. The
                        # voice controller debounces identical states and
                        # repeats them only after REPEAT_AFTER seconds.
                        if (
                            voice is not None
                            and confirmed_now
                        ):
                            voice.set_state(
                                confirmed.label,
                                confirmed.messages,
                            )
                            last_audio_generation = confirmed.generation

                if missing_frame_count >= app.MISSING_FRAME_LIMIT:
                    confirmed.reset()
                    if voice is not None:
                        voice.clear()
                    last_audio_generation = confirmed.generation

                # Draw landmarks.
                for i in range(11, 33):
                    if landmarks[i].visibility > 0.2:
                        x = int(landmarks[i].x * frame.shape[1])
                        y_pos = int(landmarks[i].y * frame.shape[0])
                        cv2.circle(
                            frame,
                            (x, y_pos),
                            4,
                            (0, 255, 0),
                            -1,
                        )

                # Optional debug information.
                if show_debug and feature_row is not None:
                    y = 90
                    for feat, val in zip(feature_order, feature_row):
                        cv2.putText(
                            frame,
                            f"{feat}: {val:.1f}",
                            (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )
                        y += 20

                    if pose_name == "Bhujangasana":
                        vis = app.get_lower_body_visibility(landmarks)
                        cv2.putText(
                            frame,
                            f"[diag] visibility: {vis}",
                            (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 200, 200),
                            1,
                        )
                        y += 20

                    if pose_name == "Ardha Chandrasana":
                        leg_lift = app.get_leg_lift_angles(landmarks)
                        ankle_diff = app.get_ankle_height_diff(landmarks)

                        leg_text = ", ".join(
                            f"{name.replace('_leg_lift', '')}: {float(value):.1f}"
                            for name, value in leg_lift.items()
                        ) or "n/a"

                        cv2.putText(
                            frame,
                            f"[diag] leg-lift: {leg_text}",
                            (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 200, 200),
                            1,
                        )
                        y += 20

                        ankle_text = (
                            "n/a"
                            if ankle_diff is None
                            else f"{float(ankle_diff):.2f}"
                        )

                        cv2.putText(
                            frame,
                            f"[diag] ankle-height-diff: {ankle_text}",
                            (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 200, 200),
                            1,
                        )

            else:
                missing_frame_count += 1
                if missing_frame_count >= app.MISSING_FRAME_LIMIT:
                    confirmed.reset()
                    if voice is not None:
                        voice.clear()
                    last_audio_generation = confirmed.generation

            # Render only the synchronized state.
            cv2.putText(
                frame,
                f"Pose: {pose_name}",
                (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                raw_status_text,
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )

            if not show_debug:
                cv2.putText(
                    frame,
                    "Press 'd' for debug info",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (140, 140, 140),
                    1,
                )

            cv2.putText(
                frame,
                display_status_text,
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )

            for i, msg in enumerate(display_messages):
                cv2.putText(
                    frame,
                    msg,
                    (20, frame.shape[0] - 50 - (i * 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    1,
                )

            fps_times.append(time.time() - frame_start)
            fps = (
                1.0 / (sum(fps_times) / len(fps_times))
                if fps_times
                else 0.0
            )

            sync_ms = (1000.0 / fps) if fps > 0 else 0.0
            confirm_ms = sync_ms * ConfirmedState.CONFIRM_FRAMES

            cv2.putText(
                frame,
                f"FPS: {fps:.0f} | confirm: ~{confirm_ms:.0f}ms",
                (frame.shape[1] - 290, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_debug = not show_debug

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if voice is not None:
            voice.stop()
        finish(session, pose_name)


if __name__ == "__main__":
    main()
