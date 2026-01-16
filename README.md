# Squat vs Deadlift Classifier + Rep Counter (MediaPipe + OpenCV)

A Python script that:
- Loads a **video file** (or webcam feed) and shows it in a **standardized 1280×720 window**
- Runs **MediaPipe Pose** to track landmarks
- **Classifies** the movement as **Squat** or **Deadlift**
- **Counts repetitions** using deterministic, rule-based logic
- Displays **Rep count on the top-right** of the window

| squat | deadlift | 
|---|---|---|
| ![](sample/squat.jpg) | ![](sample/deadlift.jpg) |

---

## How it Works

### Phase 1 — Extract joint coordinates
The system uses **MediaPipe Pose** to estimate body landmarks and extracts the key joints needed for classification and rep counting:

Core joints (used for in-frame gating + angles):
- Left/Right **Shoulder**
- Left/Right **Hip**
- Left/Right **Knee**

Additional joints (used for angles + deadlift logic):
- Left/Right **Ankle** (for knee angle computation)
- Left/Right **Wrist** (used as “hands” position)

> Even if one side is temporarily occluded, the system can compute angles using the **other visible side**.

---

### Phase 2 — Robust “User In Frame” detection (Fix for inconsistent landmark loss)
Pose tracking can briefly fail due to occlusion, blur, or fast motion. Instead of requiring *all* landmarks every frame, this script uses a robust gating strategy:

**A) Core landmark requirement**
The user is considered “in frame” if **at least one side** (left OR right) of each core group is visible:
- Shoulder pair (L or R)
- Hip pair (L or R)
- Knee pair (L or R)

**B) Frame boundary check**
Ensures those core landmarks are within the visible frame (with a small margin to avoid edge flicker).

**C) Hysteresis (enter/exit stability)**
- Requires several consecutive good frames before entering “in frame”
- Requires several consecutive bad frames before leaving “in frame”

This prevents the system from flipping rapidly between “in frame” and “out of frame.”

**D) Grace frames (short tracking dropouts won’t break counting)**
If tracking dips temporarily, the system uses the **last good landmark set** for a short window (default: 8 frames).  
This prevents missed reps when MediaPipe loses landmarks for a moment.

---

### Phase 3 — Calculate joint angles
The script computes:
- **Hip angle** (shoulder–hip–knee)
- **Knee angle** (hip–knee–ankle)

Angles are computed for both sides when possible and then averaged. If only one side is reliable, it uses that side.

To reduce noise, angles are smoothed using a short moving average window.

---

### Phase 4 — Classification + Rep Counting

## Exercise Classification (Auto)
Classification is deterministic and uses movement cues consistent with your constraints:

**Deadlift evidence**
- Hands (wrists) go **below the knees** (based on normalized Y position)
  - Uses a small margin to avoid flicker around the knee line

**Squat evidence**
- Hip and knee angles both become **deep** (both typically < ~120°)
- AND hands are **not** below the knees

**Voting + EMA smoothing**
Instead of deciding based on a single frame, classification uses:
- **EMA smoothing** on squat/deadlift “evidence”
- A **vote counter** requiring multiple consistent frames before locking the label

Once enough evidence is accumulated, the exercise is **locked** to either:
- `SQUAT`
- `DEADLIFT`

> You can also force classification with `--exercise squat` or `--exercise deadlift`.

---

## Rep Counting Logic

### Squat Rep Counter
Uses hip and knee angle thresholds:

**Rep start (descent)**
- Triggered when BOTH:
  - hip angle < **100°**
  - knee angle < **100°**
- Requires a small multi-frame hold (debounce) to reduce false triggers

**Rep completion (standing up)**
- Triggered when BOTH:
  - hip angle > **150°**
  - knee angle > **150°**
- Also debounced to prevent jitter-based increments

---

### Deadlift Rep Counter
Uses hand position relative to knees + hip extension requirement:

**Rep start**
- Triggered when hands (wrists) descend **below knee level**

**Rep completion**
- Triggered when hands rise **above knee level**
- AND hip angle indicates full extension:
  - hip angle ≥ **165°**

---

## Display & UI
- Video is always displayed in a **fixed 1280×720 window**
- Video is resized with **letterboxing** (preserves aspect ratio; no stretching)
- Overlay includes:
  - **Exercise label** + stage (top-left)
  - **Reps count** (top-right)
  - “In-frame” status + gate quality (bottom bar)

---

## Requirements
Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

## How to run:
- Terminal: python SD-counter.py --video video/**video_file.mp4**
