"""
Lane departure logic.

Classifies each frame as SAFE / WARNING / DEPARTURE based on the
ego-vehicle's lateral position relative to the detected lane boundaries.

Assumptions:
  - Camera is center-mounted → ego center x = frame_width / 2
  - Lines are (x1, y1, x2, y2) where y1 = bottom of frame (larger y value)
"""

from dataclasses import dataclass
from enum import Enum


class DepartureState(Enum):
    SAFE       = "SAFE"
    WARNING    = "WARNING"
    DEPARTURE  = "DEPARTURE"
    NO_LANES   = "NO_LANES"


@dataclass
class DepartureResult:
    state:       DepartureState
    ego_x:       int
    left_x:      int | None
    right_x:     int | None
    lane_width:  int | None


class DepartureChecker:
    def __init__(self, warning_margin: float = 0.22, sustain_frames: int = 15):
        """
        warning_margin:  fraction of lane width that triggers WARNING.
                         0.22 fires when ego is within 22% of lane width from a boundary
                         — gives earlier warning than the default 15%.
        sustain_frames:  hold a WARNING/DEPARTURE state for at least this many frames
                         so brief events are visible in the output video.
        """
        self.warning_margin  = warning_margin
        self.sustain_frames  = sustain_frames
        self._sustain_state  = DepartureState.SAFE
        self._sustain_counter = 0

    def check(self, frame_width: int, frame_height: int,
              left_line, right_line,
              signal_active: bool = False) -> DepartureResult:

        ego_x = frame_width // 2

        left_x  = self._x_at_bottom(left_line,  frame_height)
        right_x = self._x_at_bottom(right_line, frame_height)

        if left_x is None and right_x is None:
            return DepartureResult(DepartureState.NO_LANES, ego_x, None, None, None)

        # If only one lane visible, use frame edge as the other boundary
        if left_x is None:
            left_x = 0
        if right_x is None:
            right_x = frame_width

        lane_width = right_x - left_x
        if lane_width <= 0:
            return DepartureResult(DepartureState.NO_LANES, ego_x, left_x, right_x, 0)

        margin_px = int(lane_width * self.warning_margin)

        if ego_x <= left_x or ego_x >= right_x:
            raw_state = DepartureState.DEPARTURE
        elif ego_x <= left_x + margin_px or ego_x >= right_x - margin_px:
            raw_state = DepartureState.WARNING
        else:
            raw_state = DepartureState.SAFE

        # Turn signal active → intentional lane change, suppress warning
        if signal_active and raw_state in (DepartureState.WARNING, DepartureState.DEPARTURE):
            raw_state = DepartureState.SAFE

        # Sustain non-SAFE states so brief events are visible
        if raw_state in (DepartureState.WARNING, DepartureState.DEPARTURE):
            self._sustain_state   = raw_state
            self._sustain_counter = self.sustain_frames
        elif self._sustain_counter > 0:
            self._sustain_counter -= 1
            raw_state = self._sustain_state
        else:
            self._sustain_state = DepartureState.SAFE

        return DepartureResult(raw_state, ego_x, left_x, right_x, lane_width)

    @staticmethod
    def _x_at_bottom(line, frame_height: int):
        """Extrapolate line to y = frame_height and return x."""
        if line is None:
            return None
        x1, y1, x2, y2 = line
        if y1 == y2:
            return None
        slope = (x2 - x1) / (y2 - y1)
        return int(x1 + slope * (frame_height - y1))
