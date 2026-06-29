"""
Lane Detection & Departure Warning — entry point.

Supports video files, webcam (--source 0), and RTSP streams.
"""

import cv2
import time
import argparse
from collections import deque
from pathlib import Path

from lane_detector import LaneDetector
from departure_checker import DepartureChecker, DepartureState
from visualizer import draw_lanes, draw_alert, draw_stats


def parse_source(source: str):
    try:
        return int(source)
    except ValueError:
        return source


def run(source, output_path, smooth_frames, warning_margin, show, no_stats):
    src = parse_source(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_live      = isinstance(src, int) or str(source).lower().startswith("rtsp")

    detector = LaneDetector(smooth_frames=smooth_frames)
    checker  = DepartureChecker(warning_margin=warning_margin)

    stats_h  = 0 if no_stats else 90
    out_h    = height + stats_h
    Path("output").mkdir(exist_ok=True)
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          src_fps, (width, out_h))

    frame_num      = 0
    frame_times: deque = deque(maxlen=30)
    departure_count = 0
    warning_count   = 0
    signal_active   = False   # turn signal state — toggled with 's' key

    mode = "LIVE" if is_live else f"{total_frames} frames"
    print(f"Source  : {source}  ({mode})")
    print(f"Size    : {width}x{height} @ {src_fps:.1f} fps")
    print(f"Smoother: {smooth_frames}-frame buffer | Warning margin: {warning_margin*100:.0f}%")
    if show:
        print("Controls: [q] quit   [s] toggle turn signal (suppresses departure warning)")
    print("-" * 55)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # ----------------------------- Detection ------------------------------
        left_line, right_line = detector.detect(frame)

        # -------------------------- Departure check ---------------------------
        result = checker.check(width, height, left_line, right_line,
                               signal_active=signal_active)

        # -------------------------------- FPS ---------------------------------
        frame_times.append(time.perf_counter() - t0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))

        # ------------------------------ Counters ------------------------------
        if result.state == DepartureState.DEPARTURE:
            departure_count += 1
        elif result.state == DepartureState.WARNING:
            warning_count += 1

        # ----------------------------- Visualise ------------------------------
        frame = draw_lanes(frame, left_line, right_line, result)
        frame = draw_alert(frame, result)
        if not no_stats:
            frame = draw_stats(frame, frame_num, fps, result.state)

        out.write(frame)

        if show:
            # Show turn signal indicator on frame
            if signal_active:
                cv2.putText(frame, ">> SIGNAL ON <<", (width - 180, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2, cv2.LINE_AA)
            cv2.imshow("Lane Detection & Departure Warning  [q = quit | s = signal]", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nStopped by user.")
                break
            elif key == ord("s"):
                signal_active = not signal_active
                print(f"  Turn signal: {'ON  (warnings suppressed)' if signal_active else 'OFF'}")

        frame_num += 1
        if frame_num % 30 == 0:
            label = str(frame_num) if is_live else f"{frame_num}/{total_frames}"
            print(f"  Frame {label:>9s} | FPS {fps:5.1f} | State: {result.state.value}")

    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\nOutput      : {output_path}")
    print(f"Total frames: {frame_num}")
    print(f"Warnings    : {warning_count} frames")
    print(f"Departures  : {departure_count} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lane Detection & Departure Warning — video / webcam / RTSP"
    )
    parser.add_argument("--source",         required=True,
                        help="Video path | webcam index (0) | RTSP URL")
    parser.add_argument("--output",         default="output/lane_output.mp4",
                        help="Output video path")
    parser.add_argument("--smooth-frames",  type=int,   default=6,
                        help="Frame buffer size for lane line smoothing (default: 6)")
    parser.add_argument("--warning-margin", type=float, default=0.22,
                        help="Fraction of lane width that triggers WARNING (default: 0.22)")
    parser.add_argument("--show",           action="store_true",
                        help="Show live display window while processing")
    parser.add_argument("--no-stats",       action="store_true",
                        help="Omit stats panel from output video")
    args = parser.parse_args()

    run(args.source, args.output, args.smooth_frames,
        args.warning_margin, args.show, args.no_stats)
