import argparse
import os
from pathlib import Path
from collections import deque
import time

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

WIN_NAME = "Squat vs Deadlift Classifier + Rep Counter"
DISPLAY_W = 1024
DISPLAY_H = 768


def calculate_angle(a, b, c) -> float:
    """Angle ABC in degrees, points are (x,y) pixels. Range [0,180]."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def draw_text_top_right(img, text, y=40, scale=1.0, thickness=2):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = w - tw - 18
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def resize_with_letterbox(img, target_w=DISPLAY_W, target_h=DISPLAY_H):
    """Resize to (target_w,target_h) while preserving aspect ratio + black padding."""
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class RobustPoseGate:
    """
    Robust "user in frame" gate:
      - requires CORE groups (shoulder, hip, knee) with either-side visibility
      - checks normalized coords are inside frame with a margin
      - hysteresis enter/exit to prevent flicker
      - grace frames: brief tracking losses won't stop counting
    """

    def __init__(
        self,
        min_vis: float = 0.5,
        frame_margin: float = 0.08,
        enter_frames: int = 4,
        exit_frames: int = 10,
        quality_enter: float = 0.45,
        quality_exit: float = 0.25,
        quality_smooth_n: int = 8,
        max_lost_frames: int = 8,
    ):
        self.min_vis = min_vis
        self.frame_margin = frame_margin
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.quality_enter = quality_enter
        self.quality_exit = quality_exit
        self.max_lost_frames = max_lost_frames

        self.in_frame = False
        self.good_streak = 0
        self.bad_streak = 0
        self.lost_frames = 0
        self.last_good_landmarks = None
        self.quality_hist = deque(maxlen=quality_smooth_n)

        # Core landmark pairs (either left OR right side is acceptable)
        self.core_pairs = [
            (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
            (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
        ]

    def _inside_frame_norm(self, lm, idx) -> bool:
        x, y = lm[idx].x, lm[idx].y
        m = self.frame_margin
        return (-m <= x <= 1.0 + m) and (-m <= y <= 1.0 + m)

    def _any_visible(self, lm, pair) -> bool:
        a, b = pair
        return (lm[a].visibility >= self.min_vis) or (lm[b].visibility >= self.min_vis)

    def _best_vis(self, lm, pair) -> float:
        a, b = pair
        return float(max(lm[a].visibility, lm[b].visibility))

    def update(self, pose_landmarks):
        """
        Returns:
          usable_landmarks: landmarks to use for counting (may be last_good during grace)
          can_count: bool
          status_msg: str
          quality: float (smoothed)
          in_frame: bool (gate)
        """
        if pose_landmarks is None:
            if self.last_good_landmarks is not None and self.lost_frames < self.max_lost_frames:
                self.lost_frames += 1
                msg = f"Tracking lost (grace {self.lost_frames}/{self.max_lost_frames})"
                q_s = float(np.mean(self.quality_hist)) if self.quality_hist else 0.0
                return self.last_good_landmarks, True, msg, q_s, self.in_frame
            return None, False, "No person detected. Step into frame.", 0.0, False

        lm = pose_landmarks

        core_ok = all(self._any_visible(lm, p) for p in self.core_pairs)

        core_inside = True
        for a, b in self.core_pairs:
            if not (self._inside_frame_norm(lm, a) or self._inside_frame_norm(lm, b)):
                core_inside = False
                break

        q = float(np.mean([self._best_vis(lm, p) for p in self.core_pairs])) if core_ok else 0.0
        self.quality_hist.append(clamp01(q))
        q_s = float(np.mean(self.quality_hist)) if self.quality_hist else q

        if not self.in_frame:
            if core_ok and core_inside and q_s >= self.quality_enter:
                self.good_streak += 1
            else:
                self.good_streak = 0
            if self.good_streak >= self.enter_frames:
                self.in_frame = True
                self.bad_streak = 0
        else:
            if (not core_ok) or (not core_inside) or (q_s <= self.quality_exit):
                self.bad_streak += 1
            else:
                self.bad_streak = 0
            if self.bad_streak >= self.exit_frames:
                self.in_frame = False
                self.good_streak = 0

        if self.in_frame:
            self.last_good_landmarks = lm
            self.lost_frames = 0
            return lm, True, "", q_s, True

        if self.last_good_landmarks is not None and self.lost_frames < self.max_lost_frames:
            self.lost_frames += 1
            msg = f"Tracking unstable (grace {self.lost_frames}/{self.max_lost_frames})"
            return self.last_good_landmarks, True, msg, q_s, self.in_frame

        return None, False, "Move backwards / keep full body in frame", q_s, False


class SquatDeadliftCounter:
    def __init__(
        self,
        squat_down_angle=100.0,
        squat_up_angle=150.0,
        deadlift_extension=165.0,
        deadlift_margin_norm=0.02,  # normalized y margin (resolution independent)
        smooth_angles_n=5,
        smooth_y_n=5,
    ):
        self.squat_down_angle = float(squat_down_angle)
        self.squat_up_angle = float(squat_up_angle)
        self.deadlift_extension = float(deadlift_extension)
        self.deadlift_margin_norm = float(deadlift_margin_norm)

        self.rep_count = 0
        self.stage = "up"

        self.hip_hist = deque(maxlen=smooth_angles_n)
        self.knee_hist = deque(maxlen=smooth_angles_n)
        self.hand_y_hist = deque(maxlen=smooth_y_n)
        self.knee_y_hist = deque(maxlen=smooth_y_n)

        self.exercise_locked = False
        self.exercise = "auto"      # "auto" | "squat" | "deadlift"
        self._label = "unknown"

        self.squat_ema = 0.0
        self.deadlift_ema = 0.0
        self.squat_votes = 0
        self.deadlift_votes = 0
        self.VOTE_TO_LOCK = 15

        self._down_hold = 0
        self._up_hold = 0
        self.HOLD_FRAMES = 2

    def reset(self):
        self.rep_count = 0
        self.stage = "up"
        self.hip_hist.clear()
        self.knee_hist.clear()
        self.hand_y_hist.clear()
        self.knee_y_hist.clear()
        self._down_hold = 0
        self._up_hold = 0

    def unlock(self):
        self.exercise_locked = False
        self.exercise = "auto"
        self._label = "unknown"
        self.squat_ema = 0.0
        self.deadlift_ema = 0.0
        self.squat_votes = 0
        self.deadlift_votes = 0

    @staticmethod
    def _pt_if_vis(lm, idx, w, h, min_vis):
        if lm[idx].visibility >= min_vis:
            return (lm[idx].x * w, lm[idx].y * h)
        return None

    @staticmethod
    def _angle_if_possible(a, b, c):
        if a is None or b is None or c is None:
            return None
        return calculate_angle(a, b, c)

    @staticmethod
    def _avg_available(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    @staticmethod
    def _avg_norm_y(lm, left_idx, right_idx, min_vis):
        y_vals = []
        if lm[left_idx].visibility >= min_vis:
            y_vals.append(float(lm[left_idx].y))
        if lm[right_idx].visibility >= min_vis:
            y_vals.append(float(lm[right_idx].y))
        return float(np.mean(y_vals)) if y_vals else None

    def _compute_angles_and_ys(self, lm, w, h, min_vis):
        hip_L = self._angle_if_possible(
            self._pt_if_vis(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h, min_vis),
            self._pt_if_vis(lm, mp_pose.PoseLandmark.LEFT_HIP.value, w, h, min_vis),
            self._pt_if_vis(lm, mp_pose.PoseLandmark.LEFT_KNEE.value, w, h, min_vis),
        )
        hip_R = self._angle_if_possible(
            self._pt_if_vis(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h, min_vis),
            self._pt_if_vis(lm, mp_pose.PoseLandmark.RIGHT_HIP.value, w, h, min_vis),
            self._pt_if_vis(lm, mp_pose.PoseLandmark.RIGHT_KNEE.value, w, h, min_vis),
        )
        hip = self._avg_available([hip_L, hip_R])

        knee_L = self._angle_if_possible(
            self._pt_if_vis(lm, mp_pose.PoseLandmark.LEFT_HIP.value, w, h, min_vis),
            self._pt_if_vis(lm, mp_pose.PoseLandmark.LEFT_KNEE.value, w, h, min_vis),
            self._pt_if_vis(lm, mp_pose.PoseLandmark.LEFT_ANKLE.value, w, h, min_vis),
        )
        knee_R = self._angle_if_possible(
            self._pt_if_vis(lm, mp_pose.PoseLandmark.RIGHT_HIP.value, w, h, min_vis),
            self._pt_if_vis(lm, mp_pose.PoseLandmark.RIGHT_KNEE.value, w, h, min_vis),
            self._pt_if_vis(lm, mp_pose.PoseLandmark.RIGHT_ANKLE.value, w, h, min_vis),
        )
        knee = self._avg_available([knee_L, knee_R])

        hand_y = self._avg_norm_y(lm, mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value, min_vis)
        knee_y = self._avg_norm_y(lm, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, min_vis)

        return hip, knee, hand_y, knee_y

    def _smooth(self, hip, knee, hand_y, knee_y):
        if hip is not None:
            self.hip_hist.append(float(hip))
        if knee is not None:
            self.knee_hist.append(float(knee))
        if hand_y is not None:
            self.hand_y_hist.append(float(hand_y))
        if knee_y is not None:
            self.knee_y_hist.append(float(knee_y))

        hip_s = float(np.mean(self.hip_hist)) if self.hip_hist else None
        knee_s = float(np.mean(self.knee_hist)) if self.knee_hist else None
        hand_y_s = float(np.mean(self.hand_y_hist)) if self.hand_y_hist else None
        knee_y_s = float(np.mean(self.knee_y_hist)) if self.knee_y_hist else None
        return hip_s, knee_s, hand_y_s, knee_y_s

    def _update_classification(self, hip_s, knee_s, hand_y_s, knee_y_s):
        if self.exercise_locked or self.exercise != "auto":
            self._label = "squat" if self.exercise == "squat" else "deadlift" if self.exercise == "deadlift" else self._label
            return

        if hip_s is None or knee_s is None or hand_y_s is None or knee_y_s is None:
            return

        hands_below_knees = hand_y_s > (knee_y_s + self.deadlift_margin_norm)
        deep_squat_like = (hip_s < 120.0 and knee_s < 120.0)

        squat_frame = 1.0 if (deep_squat_like and not hands_below_knees) else 0.0
        deadlift_frame = 1.0 if hands_below_knees else 0.0

        self.squat_ema = 0.90 * self.squat_ema + 0.10 * squat_frame
        self.deadlift_ema = 0.90 * self.deadlift_ema + 0.10 * deadlift_frame

        if self.squat_ema - self.deadlift_ema > 0.15:
            self.squat_votes += 1
            self.deadlift_votes = max(0, self.deadlift_votes - 1)
        elif self.deadlift_ema - self.squat_ema > 0.15:
            self.deadlift_votes += 1
            self.squat_votes = max(0, self.squat_votes - 1)
        else:
            self.squat_votes = max(0, self.squat_votes - 1)
            self.deadlift_votes = max(0, self.deadlift_votes - 1)

        if self._label == "unknown":
            if self.squat_votes >= self.VOTE_TO_LOCK:
                self._label = "squat"
                self.exercise_locked = True
            elif self.deadlift_votes >= self.VOTE_TO_LOCK:
                self._label = "deadlift"
                self.exercise_locked = True

    def update(self, lm, w, h, min_vis):
        hip, knee, hand_y, knee_y = self._compute_angles_and_ys(lm, w, h, min_vis)
        hip_s, knee_s, hand_y_s, knee_y_s = self._smooth(hip, knee, hand_y, knee_y)

        self._update_classification(hip_s, knee_s, hand_y_s, knee_y_s)

        label = self._label

        debug = {
            "hip": hip_s,
            "knee": knee_s,
            "hand_y": hand_y_s,
            "knee_y": knee_y_s,
            "squat_votes": self.squat_votes,
            "deadlift_votes": self.deadlift_votes,
        }

        if label == "unknown":
            return label, self.rep_count, self.stage, debug

        if label == "squat":
            if hip_s is None or knee_s is None:
                return label, self.rep_count, self.stage, debug

            if hip_s < self.squat_down_angle and knee_s < self.squat_down_angle and self.stage == "up":
                self._down_hold += 1
                if self._down_hold >= self.HOLD_FRAMES:
                    self.stage = "down"
                    self._down_hold = 0
            else:
                self._down_hold = 0

            if hip_s > self.squat_up_angle and knee_s > self.squat_up_angle and self.stage == "down":
                self._up_hold += 1
                if self._up_hold >= self.HOLD_FRAMES:
                    self.rep_count += 1
                    self.stage = "up"
                    self._up_hold = 0
            else:
                self._up_hold = 0

        elif label == "deadlift":
            if hip_s is None or hand_y_s is None or knee_y_s is None:
                return label, self.rep_count, self.stage, debug

            hands_below_knees = hand_y_s > (knee_y_s + self.deadlift_margin_norm)
            hands_above_knees = hand_y_s < (knee_y_s - self.deadlift_margin_norm)

            if hands_below_knees and self.stage == "up":
                self.stage = "down"

            if hands_above_knees and hip_s >= self.deadlift_extension and self.stage == "down":
                self.rep_count += 1
                self.stage = "up"

        return label, self.rep_count, self.stage, debug


def resolve_video_path(video_arg: str) -> str:
    """Resolve relative path robustly against CWD and script directory."""
    if not video_arg:
        return ""

    p = Path(video_arg)
    if p.is_file():
        return str(p)

    script_dir = Path(__file__).resolve().parent
    p2 = (script_dir / video_arg).resolve()
    if p2.is_file():
        return str(p2)

    raise FileNotFoundError(
        f"Video not found: '{video_arg}'. Tried CWD='{os.getcwd()}' and script_dir='{script_dir}'."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="Video path (e.g., video/squat.mp4). If omitted, webcam.")
    parser.add_argument("--exercise", type=str, default="auto", choices=["auto", "squat", "deadlift"], help="Force exercise label or auto.")
    parser.add_argument("--min_vis", type=float, default=0.50, help="Min landmark visibility (0-1). Lower = more tolerant.")
    parser.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2], help="MediaPipe Pose model complexity.")
    parser.add_argument("--min_det", type=float, default=0.35, help="Min detection confidence.")
    parser.add_argument("--min_track", type=float, default=0.50, help="Min tracking confidence.")

    parser.add_argument("--squat_down", type=float, default=100.0)
    parser.add_argument("--squat_up", type=float, default=150.0)
    parser.add_argument("--deadlift_ext", type=float, default=165.0)
    parser.add_argument("--deadlift_margin", type=float, default=0.02, help="Normalized y margin; ~2% of frame height.")
    parser.add_argument("--enter_frames", type=int, default=4)
    parser.add_argument("--exit_frames", type=int, default=10)
    parser.add_argument("--max_lost", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.video:
        video_path = resolve_video_path(args.video)
        cap = cv2.VideoCapture(video_path)
        src_label = f"VIDEO: {video_path}"
    else:
        cap = cv2.VideoCapture(0)
        src_label = "WEBCAM"

    if not cap.isOpened():
        raise RuntimeError("Could not open video source. Bad path, missing file, or unsupported codec.")

    # Create a standardized window size (hardcoded)
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, DISPLAY_W, DISPLAY_H)

    gate = RobustPoseGate(
        min_vis=args.min_vis,
        enter_frames=args.enter_frames,
        exit_frames=args.exit_frames,
        max_lost_frames=args.max_lost,
    )

    counter = SquatDeadliftCounter(
        squat_down_angle=args.squat_down,
        squat_up_angle=args.squat_up,
        deadlift_extension=args.deadlift_ext,
        deadlift_margin_norm=args.deadlift_margin,
    )
    counter.exercise = args.exercise
    counter.exercise_locked = (args.exercise != "auto")
    counter._label = "unknown" if args.exercise == "auto" else args.exercise

    prev_t = time.time()
    fps = 0.0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_track,
    ) as pose:

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            h, w = frame.shape[:2]

            # FPS estimate
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose.process(rgb)
            rgb.flags.writeable = True

            pose_lm = res.pose_landmarks.landmark if res.pose_landmarks else None
            usable_lm, can_count, status_msg, q_s, in_frame = gate.update(pose_lm)

            label = counter._label
            debug = {}

            if can_count and usable_lm is not None:
                label, reps, stage, debug = counter.update(usable_lm, w, h, args.min_vis)
            else:
                reps, stage = counter.rep_count, counter.stage

            # Draw skeleton on the ORIGINAL frame (better pose accuracy), then resize for display
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2),
                )

            # Standardize the displayed image size (fixed 1280x720)
            display = resize_with_letterbox(frame, DISPLAY_W, DISPLAY_H)
            dh, dw = display.shape[:2]

            # TOP-LEFT: label + stage
            cv2.putText(display, f"{label.upper()}  |  {stage.upper()}",
                        (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

            # TOP-RIGHT: rep count
            draw_text_top_right(display, f"Reps: {counter.rep_count}", y=40, scale=1.0, thickness=2)

            # Bottom status bar
            bar_h = 62
            cv2.rectangle(display, (0, dh - bar_h), (dw, dh), (0, 0, 0), -1)

            gate_line = f"In-frame: {in_frame} | GateQ: {q_s:.2f} | {src_label}"
            cv2.putText(display, gate_line, (18, dh - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

            if status_msg:
                cv2.putText(display, status_msg, (18, dh - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(display, "Press 'r' to reset reps. Press 'u' to unlock AUTO classification.",
                            (18, dh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            if args.debug and debug:
                d1 = f"Hip:{(debug.get('hip') or 0):.0f}  Knee:{(debug.get('knee') or 0):.0f}  FPS:{fps:.1f}"
                d2 = f"Votes S:{debug.get('squat_votes',0)} D:{debug.get('deadlift_votes',0)}"
                cv2.putText(display, d1, (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display, d2, (18, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WIN_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                counter.reset()
            elif key == ord("u"):
                counter.unlock()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
