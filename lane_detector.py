"""Lane detection using Canny edge detection + probabilistic Hough transform."""

import cv2
import numpy as np
from collections import deque


# ── Type alias ────────────────────────────────────────────────────────────────
Line = tuple[int, int, int, int]

# ── Tuning constants ──────────────────────────────────────────────────────────
_SLOPE_MIN           = 0.3    # reject near-horizontal lines below this absolute slope
_SLOPE_MAX           = 2.5    # reject near-vertical vehicle edges above this
_CANNY_LOW           = 50
_CANNY_HIGH          = 150
_HOUGH_THRESHOLD     = 25
_HOUGH_MIN_LENGTH    = 25
_HOUGH_MAX_GAP       = 80
_Y_DETECT_FRAC       = 0.78   # top of detection ROI (fraction of frame height)
_Y_DRAW_FRAC         = 0.58   # top of drawn lane polygon
_ROI_X_FAR_LEFT      = 0.08   # ROI trapezoid corner x-fractions
_ROI_X_INNER_LEFT    = 0.44
_ROI_X_INNER_RIGHT   = 0.56
_ROI_X_FAR_RIGHT     = 0.92
_SIDE_SPLIT_LEFT     = 0.55   # left segments: bottom endpoint must be left of this
_SIDE_SPLIT_RIGHT    = 0.45   # right segments: bottom endpoint must be right of this
_PLAUSIBILITY_CENTER = 0.50   # fitted line bottom-x must be on the correct side
_MIN_LANE_WIDTH_FRAC = 0.25   # discard both lines if lane is narrower (centre-line artefact)


def _make_roi_mask(frame: np.ndarray, roi_vertices: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [roi_vertices], 255)
    return cv2.bitwise_and(frame, mask)


def _slope_intercept(line):
    x1, y1, x2, y2 = line
    if x2 == x1:
        return None, None
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def _fit_lane(segments: list, y_bottom: int, y_top: int) -> Line | None:
    """Average a set of raw Hough segments into one (x1,y1,x2,y2) line."""
    if not segments:
        return None
    slopes, intercepts = [], []
    for seg in segments:
        s, i = _slope_intercept(seg)
        # Reject near-horizontal (noise) and near-vertical (truck/vehicle sides)
        if s is None or abs(s) < _SLOPE_MIN or abs(s) > _SLOPE_MAX:
            continue
        slopes.append(s)
        intercepts.append(i)
    if not slopes:
        return None
    slope = np.mean(slopes)
    intercept = np.mean(intercepts)
    x_bottom = int((y_bottom - intercept) / slope)
    x_top    = int((y_top    - intercept) / slope)
    return (x_bottom, y_bottom, x_top, y_top)


class LaneDetector:
    def __init__(self, smooth_frames: int = 6):
        self._left_buf  = deque(maxlen=smooth_frames)
        self._right_buf = deque(maxlen=smooth_frames)

    def detect(self, frame: np.ndarray) -> tuple[Line | None, Line | None]:
        """
        Returns (left_line, right_line).

        Each line is (x1, y1, x2, y2) in pixel coords, or None if not found.
        """
        h, w = frame.shape[:2]

        # Two separate y boundaries:
        #   y_detect_top — top of the ROI used for Hough detection.
        #                  Set high (78%) to exclude gantry overhead signs whose
        #                  bottom edges sit at ~72-75% of frame height.
        #   y_draw_top   — how far up the fitted line is drawn on screen.
        #                  Set lower (58%) so the lane polygon looks long and
        #                  extends well ahead of the vehicle.
        # Detection uses only clean close-road pixels; drawing extrapolates
        # the same fitted line slope upward into the gantry-free zone.
        y_detect_top = int(h * _Y_DETECT_FRAC)
        y_draw_top   = int(h * _Y_DRAW_FRAC)
        y_top        = y_detect_top   # used for ROI and _fit_lane computation
        y_bottom = h
        roi_pts  = np.array([
            [int(w * _ROI_X_FAR_LEFT),    y_bottom],
            [int(w * _ROI_X_INNER_LEFT),  y_top],
            [int(w * _ROI_X_INNER_RIGHT), y_top],
            [int(w * _ROI_X_FAR_RIGHT),   y_bottom],
        ], dtype=np.int32)

        # --------------------- Step 2: Canny on grayscale ---------------------
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, threshold1=_CANNY_LOW, threshold2=_CANNY_HIGH)
        roi   = _make_roi_mask(edges, roi_pts)

        # --------------------------- Step 3: Hough ----------------------------
        segments = cv2.HoughLinesP(
            roi,
            rho=1, theta=np.pi / 180,
            threshold=_HOUGH_THRESHOLD,
            minLineLength=_HOUGH_MIN_LENGTH,
            maxLineGap=_HOUGH_MAX_GAP,
        )

        # --------- Step 4+5: Slope filter + bottom-endpoint side split --------
        # Use the BOTTOM endpoint (larger y = closest to car) for x-side check.
        # Bottom endpoints are stable in perspective; top endpoints converge
        # toward the vanishing point and cross the frame center — unusable for
        # side-of-frame filtering.
        left_segs, right_segs = [], []
        if segments is not None:
            for seg in segments:
                x1, y1, x2, y2 = seg[0]
                slope, _ = _slope_intercept((x1, y1, x2, y2))
                if slope is None or abs(slope) < _SLOPE_MIN or abs(slope) > _SLOPE_MAX:
                    continue
                bx = x1 if y1 >= y2 else x2   # bottom endpoint x
                if slope < 0:
                    if bx < w * _SIDE_SPLIT_LEFT:          # left: bottom point in left 55%
                        left_segs.append((x1, y1, x2, y2))
                else:
                    if bx > w * _SIDE_SPLIT_RIGHT:         # right: bottom point in right 55%
                        right_segs.append((x1, y1, x2, y2))

        # Fit using detection ROI, then re-extrapolate top point to y_draw_top
        # so the drawn line extends further ahead of the vehicle.
        left_raw  = _fit_lane(left_segs,  y_bottom, y_draw_top)
        right_raw = _fit_lane(right_segs, y_bottom, y_draw_top)

        # ------------------- Step 6: Position plausibility --------------------
        # Fitted line bottom x must be on its correct side of frame center.
        if left_raw is not None and left_raw[0] > w * _PLAUSIBILITY_CENTER:
            left_raw = None
        if right_raw is not None and right_raw[0] < w * _PLAUSIBILITY_CENTER:
            right_raw = None

        # --------------------- Step 7: Minimum lane width ---------------------
        # If both lines are detected but less than 25% of frame width apart
        # at the bottom, one is likely tracking a center dashed line.
        # Discard both and let the smoother hold the last good values.
        if left_raw is not None and right_raw is not None:
            if right_raw[0] - left_raw[0] < w * _MIN_LANE_WIDTH_FRAC:
                left_raw  = None
                right_raw = None

        # --------------------- Step 8: Temporal smoothing ---------------------
        left_line  = self._smooth(left_raw,  self._left_buf)
        right_line = self._smooth(right_raw, self._right_buf)

        return left_line, right_line

    @staticmethod
    def _smooth(line, buf: deque):
        if line is not None:
            buf.append(line)
        if not buf:
            return None
        arr = np.array(buf, dtype=np.float32)
        return tuple(np.mean(arr, axis=0).astype(int).tolist())
