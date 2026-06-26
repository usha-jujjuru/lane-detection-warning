# Lane Detection & Departure Warning

A real-time lane detection and departure warning system using classical computer vision — no neural network required.

Detects left and right lane markings using Canny edge detection and Hough transform, then monitors the ego vehicle's lateral position to trigger safety warnings.

![Lane Detection Preview](preview.jpg)

---

## How It Works

```
Frame
  │
  ├─ ROI trapezoid — bottom 22% of frame (excludes sky, gantry signs, clutter)
  ├─ Grayscale + Gaussian blur
  ├─ Canny edge detection
  ├─ Probabilistic Hough transform → raw line segments
  ├─ Slope filter — rejects near-horizontal noise and near-vertical vehicle edges
  ├─ Bottom-endpoint x-split — left segments stay left, right stay right
  ├─ Polyfit averaging → one stable line per side
  ├─ Position plausibility check — rejects lines on wrong side of frame center
  ├─ Minimum lane width check — rejects if both lines too close (center-line pickup)
  ├─ Temporal smoothing — 6-frame rolling average reduces flicker
  ├─ Line extrapolated to 58% frame height for display (longer visual ahead)
  ├─ Departure check — ego center vs lane boundaries
  └─ Annotated output frame
```

### Departure States

| State | Condition | Overlay |
|-------|-----------|---------|
| `SAFE` | Ego center well within lane | Green polygon |
| `WARNING` | Ego within 22% of lane width from a boundary | Yellow polygon + banner |
| `DEPARTURE` | Ego outside lane boundary | Red polygon + alert |
| `NO_LANES` | No markings detected — smoother holds last known position | No overlay |

---

## Quickstart

```bash
git clone https://github.com/usha-jujjuru/lane-detection-warning.git
cd lane-detection-warning
pip install -r requirements.txt

# Run on sample Autobahn video
python detect_lanes.py --source sample_video/autobahn.mp4

# Run on webcam (live)
python detect_lanes.py --source 0 --show

# Run on IP camera / RTSP stream
python detect_lanes.py --source rtsp://your-camera-ip/stream --show
```

---

## Usage

```bash
python detect_lanes.py \
  --source         sample_video/autobahn.mp4 \
  --output         output/lane_output.mp4 \
  --smooth-frames  6 \
  --warning-margin 0.22 \
  --show
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | required | Video path, `0` (webcam), or RTSP URL |
| `--output` | output/lane_output.mp4 | Output video path |
| `--smooth-frames` | 6 | Frame buffer size for temporal smoothing |
| `--warning-margin` | 0.22 | Fraction of lane width that triggers WARNING |
| `--show` | off | Show live display window while processing |
| `--no-stats` | off | Omit stats panel from output video |

### Live display controls (--show mode)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Toggle turn signal — suppresses departure warning during intentional lane changes |

---

## Project Structure

```
lane-detection-warning/
├── lane_detector.py      # Canny + Hough + slope/position filters + smoother
├── departure_checker.py  # Departure logic with turn signal suppression
├── visualizer.py         # Lane polygon, alert banner, stats panel, line clipping
├── detect_lanes.py       # Entry point — video / webcam / RTSP, all CLI options
├── sample_video/         # Sample Autobahn dashcam clip
├── output/               # Generated output (git-ignored)
├── preview.jpg
└── requirements.txt
```

---

## Limitations

These are inherent to the classical Canny + Hough approach and represent the engineering boundary that motivates deep learning in production ADAS:

| Scenario | Behaviour | Root Cause |
|----------|-----------|------------|
| **White trucks alongside** | Lane line may track truck bottom edge | White truck paint and white lane markings are identical in grayscale — no semantic distinction without a trained model |
| **Road shadows** | Shadow boundary may be detected as a spurious edge | Canny picks up any strong brightness gradient — shadow transitions create similar gradients to lane markings |
| **Overhead gantry signs** | Excluded by raising ROI top to 78% of frame height | White arrows/text on signs generate Hough lines at lane-like slopes |
| **Dense traffic** | Lines frozen at last known position via smoother | Vehicles fully blocking markings — no markings visible = no new detection |
| **Gentle curves** | Polygon drifts as curve accumulates | Hough fits straight lines only — polynomial fitting required for curves |
| **Worn or faded markings** | Reduced detection confidence | Low-contrast markings may fall below Canny gradient threshold |
| **Night without headlights** | No detection | Markings not illuminated, no gradient to detect |

### Turn signal integration

The `s` key simulates a turn indicator. In a real vehicle the departure warning is suppressed automatically when the indicator is active, preventing false alerts during intentional lane changes. This is how production LDW systems (Volvo, Mercedes, BMW LCA) handle intentional lane changes.

---

## Author

**Usha Rani Jujjuru**
M.Sc. Automotive Software Engineering — TU Chemnitz
Perception Engineer | Computer Vision | ADAS | Autonomous Driving
[LinkedIn](https://linkedin.com/in/usha-rani-jujjuru) · [GitHub](https://github.com/usha-jujjuru)
