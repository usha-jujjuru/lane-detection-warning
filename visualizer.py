"""
Overlay functions for lane lines, lane polygon, and departure alert.
"""

import cv2
import numpy as np
from departure_checker import DepartureResult, DepartureState


# Colour palette (BGR)
_COLOR = {
    DepartureState.SAFE:      (0,   200,  0),     # green
    DepartureState.WARNING:   (0,   200, 255),    # yellow-orange
    DepartureState.DEPARTURE: (0,    0,  220),    # red
    DepartureState.NO_LANES:  (180, 180, 180),    # grey
}

_ALERT_TEXT = {
    DepartureState.SAFE:      "",
    DepartureState.WARNING:   "! WARNING: Approaching Lane",
    DepartureState.DEPARTURE: "!! LANE DEPARTURE !!",
    DepartureState.NO_LANES:  "",
}


_POLYGON_ALPHA = 0.25   # lane polygon fill transparency


def _clip_line(line, w, h):
    """Clip a line's x endpoints to [0, w] so it stays within the frame."""
    x1, y1, x2, y2 = line
    x1 = int(np.clip(x1, 0, w))
    x2 = int(np.clip(x2, 0, w))
    y1 = int(np.clip(y1, 0, h))
    y2 = int(np.clip(y2, 0, h))
    return x1, y1, x2, y2


def draw_lanes(frame: np.ndarray, left_line, right_line,
               result: DepartureResult) -> np.ndarray:
    overlay = frame.copy()
    color   = _COLOR[result.state]
    h, w    = frame.shape[:2]

    # Clip lines to frame bounds before drawing.
    # Lines extrapolated from a shallow-slope detection can have x coordinates
    # hundreds of pixels outside the frame — clipping keeps polygon and lines
    # visually contained without changing the lane position logic.
    left_c  = _clip_line(left_line,  w, h) if left_line  is not None else None
    right_c = _clip_line(right_line, w, h) if right_line is not None else None

    if left_c is not None and right_c is not None:
        lx1, ly1, lx2, ly2 = left_c
        rx1, ry1, rx2, ry2 = right_c
        pts = np.array([[lx1, ly1], [lx2, ly2],
                        [rx2, ry2], [rx1, ry1]], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        frame = cv2.addWeighted(overlay, _POLYGON_ALPHA, frame, 1 - _POLYGON_ALPHA, 0)

    if left_c is not None:
        x1, y1, x2, y2 = left_c
        cv2.line(frame, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)

    if right_c is not None:
        x1, y1, x2, y2 = right_c
        cv2.line(frame, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)

    cv2.circle(frame, (result.ego_x, h - 10), 6, (255, 255, 255), -1)

    return frame


def draw_alert(frame: np.ndarray, result: DepartureResult) -> np.ndarray:
    text = _ALERT_TEXT[result.state]
    if not text:
        return frame

    h, w = frame.shape[:2]
    color = _COLOR[result.state]

    # Alert banner at top of frame
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    thickness  = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = (w - tw) // 2
    ty = 50

    # Dark background strip
    cv2.rectangle(frame, (0, ty - th - 12), (w, ty + 12), (0, 0, 0), -1)
    cv2.putText(frame, text, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)

    return frame


def draw_stats(frame: np.ndarray, frame_num: int, fps: float,
               state: DepartureState) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_h = 90
    panel   = np.zeros((panel_h, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    color = (220, 220, 220)
    small = 0.5

    lines = [
        f"Frame: {frame_num}",
        f"FPS:   {fps:.1f}",
        f"State: {state.value}",
    ]

    for i, line in enumerate(lines):
        cv2.putText(panel, line, (10, 20 + i * 22), font, small, color, 1, cv2.LINE_AA)

    return np.vstack([panel, frame])
