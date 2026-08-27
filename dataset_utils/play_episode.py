"""Play back camera streams from a raw chemdata episode h5 recording.

Usage:
    python dataset_utils/play_episode.py <episode.h5> [options]

Keys during playback:
    [space] pause/resume    [.] step forward    [,] step back    [q]/[esc] quit

The overlay shows frame index, stream time and the gap to the previous
stamp, so dropped-frame bursts are directly visible while playing. When a
stream ends, the final frame is held with a banner showing how much longer
the arm kept recording (the 'frozen tail' artifact).

If the installed opencv build has no gui support (headless wheel), playback
automatically falls back to ffplay, or to writing an mp4 clip opened with
the system player. Interactive frame stepping then requires a gui-enabled
opencv (pip install opencv-python).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np

CAM_SOURCES = {
    "exo": "zed__zed_node__rgb__color__rect__image",
    "wrist": "wrist_cam__image_raw",
}
ARMS = "soarms__data"
DEFAULT_FPS = 15.0
QUIT_KEYS = (ord("q"), 27)
ARROW_RIGHT = 83
ARROW_LEFT = 81


class EpisodePlayer:
    """Displays one or two camera streams of a raw episode near real time."""

    def __init__(self, h5: h5py.File, cams: list[str], scale: int, show_overlay: bool) -> None:
        self.h5 = h5
        self.cams = cams
        self.scale = scale
        self.show_overlay = show_overlay
        self.streams: dict[str, tuple[h5py.Dataset, np.ndarray, np.ndarray]] = {}
        for cam in cams:
            src = CAM_SOURCES[cam]
            if f"observations/{src}" not in h5:
                avail = list(h5["observations"].keys())
                sys.exit(f"[ERROR]: stream '{cam}' ({src}) not in file. Available: {avail}")
            ds = h5[f"observations/{src}"]
            stamps = h5[f"timestamps/{src}"][:]
            gaps_ms = np.r_[0.0, np.diff(stamps) * 1000.0]
            self.streams[cam] = (ds, stamps, gaps_ms)
        self.arm_end_s = float(h5[f"timestamps/{ARMS}"][:].max())

    @property
    def n(self) -> int:
        return min(ds.shape[0] for ds, _, _ in self.streams.values())

    def render(self, i: int, banner: list[str] | None = None) -> np.ndarray:
        """Compose the display image for stream frame index i."""
        tiles = []
        for cam in self.cams:
            ds, stamps, gaps = self.streams[cam]
            frame = cv2.cvtColor(ds[i], cv2.COLOR_RGB2BGR)
            if self.scale != 1:
                frame = cv2.resize(
                    frame,
                    (frame.shape[1] * self.scale, frame.shape[0] * self.scale),
                    interpolation=cv2.INTER_NEAREST,
                )
            if self.show_overlay:
                lines = [
                    cam,
                    f"frame {i}/{self.n - 1}",
                    f"t={stamps[i] - stamps[0]:.2f}s  dt={gaps[i]:+.0f}ms",
                ]
                frame = _overlay(frame, lines)
            tiles.append(frame)
        height = min(t.shape[0] for t in tiles)
        tiles = [t if t.shape[0] == height else
                 cv2.resize(t, (t.shape[1] * height // t.shape[0], height)) for t in tiles]
        image = np.hstack(tiles)
        if banner:
            image = _overlay(image, banner, bottom=True)
        return image

    def run(self, start: int, end: int, fps: float) -> None:
        """Play [start, end) at fps; falls back gracefully on headless opencv."""
        try:
            self._run_cv2(start, end, fps)
        except cv2.error as err:
            # ! the venv opencv build has no gui support; fall back to external players
            print(f"[WARN]: opencv gui unavailable ({str(err).splitlines()[-1].strip()})")
            if shutil.which("ffplay"):
                self._play_ffplay(start, end, fps)
            else:
                self._save_and_open(start, end, fps)

    def _end_banner(self, end: int) -> list[str]:
        """Lines shown on the held final frame (frozen tail info)."""
        stamps = self.streams[self.cams[0]][1]
        return [
            f"END OF STREAM at t={stamps[end - 1] - stamps[0]:.2f}s "
            f"(arm records until t={self.arm_end_s - stamps[0]:.2f}s)",
        ]

    def _run_cv2(self, start: int, end: int, fps: float) -> None:
        """Interactive window loop: pause/step with the keyboard."""
        window = "episode player  [space] pause  [.] fwd  [,] back  [q] quit"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        delay_ms = max(1, round(1000.0 / fps))
        i = start
        paused = False
        while True:
            if i >= end:
                # ! hold the final frame with the arm-vs-stream end offset (frozen tail)
                banner = self._end_banner(end) + ["press [q] to quit"]
                cv2.imshow(window, self.render(end - 1, banner=banner))
                while (cv2.waitKey(0) & 0xFF) not in QUIT_KEYS:
                    pass
                break

            cv2.imshow(window, self.render(i))
            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
            if key in QUIT_KEYS:
                break
            if key == ord(" "):
                paused = not paused
            elif key in (ord("."), ord("n"), ARROW_RIGHT):
                i = min(i + 1, end - 1)
                paused = True
            elif key in (ord(","), ord("p"), ARROW_LEFT):
                i = max(start, i - 1)
                paused = True
            if not paused:
                i += 1
        cv2.destroyAllWindows()

    def _play_ffplay(self, start: int, end: int, fps: float) -> None:
        """Stream frames into ffplay (pause/seek/quit handled by ffplay keys)."""
        print("[INFO]: playing via ffplay  [space] pause  [<-]/[->] seek  [q] quit")
        first = self.render(start)
        h, w = first.shape[:2]
        cmd = [
            "ffplay", "-hide_banner", "-loglevel", "error", "-autoexit",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{w}x{h}", "-framerate", f"{fps:.3f}", "-i", "-",
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in range(start, end):
                proc.stdin.write(self.render(i).tobytes())
            proc.stdin.write(self.render(end - 1, banner=self._end_banner(end) + ["end of clip"]).tobytes())
        except BrokenPipeError:
            pass  # user closed ffplay early
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            proc.wait()

    def _save_and_open(self, start: int, end: int, fps: float) -> None:
        """Last resort: write an mp4 clip and open it with the system player."""
        stem = Path(self.h5.filename).stem
        out = Path(tempfile.gettempdir()) / f"play_{stem}_{'-'.join(self.cams)}_{start}-{end}.mp4"
        first = self.render(start)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (first.shape[1], first.shape[0]))
        for i in range(start, end):
            writer.write(self.render(i))
        writer.write(self.render(end - 1, banner=self._end_banner(end) + ["end of clip"]))
        writer.release()
        print(f"[INFO]: wrote clip to {out}")
        if shutil.which("xdg-open"):
            subprocess.run(["xdg-open", str(out)], check=False)
        else:
            print("[INFO]: open the clip manually with your video player")


def _overlay(image: np.ndarray, lines: list[str], bottom: bool = False) -> np.ndarray:
    """Draw a readable text block (black bar) on the image."""
    line_h = 24
    pad = 8
    block_h = line_h * len(lines) + pad
    top = image.shape[0] - block_h if bottom else 0
    cv2.rectangle(image, (0, top), (image.shape[1], top + block_h), (0, 0, 0), -1)
    for k, line in enumerate(lines):
        cv2.putText(image, line, (8, top + pad + 16 + k * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def auto_fps(stamps: np.ndarray) -> float:
    """Derive playback fps from the median stamp gap of the primary stream."""
    if len(stamps) < 3:
        return DEFAULT_FPS
    med_gap_ms = float(np.median(np.diff(stamps)) * 1000.0)
    if med_gap_ms <= 0:
        return DEFAULT_FPS
    return float(np.clip(1000.0 / med_gap_ms, 1.0, 120.0))


def list_streams(h5: h5py.File, path: Path) -> None:
    """Print every source in the episode with shape, span and rate."""
    print(f"[INFO]: {path}")
    for src in h5["observations"]:
        ds = h5[f"observations/{src}"]
        st = h5[f"timestamps/{src}"][:]
        span = st.max() - st.min()
        rate = (len(st) / span) if span > 0 else 0.0
        cam = next((name for name, s in CAM_SOURCES.items() if s == src), "-")
        print(f"  cam={cam:6s} {src:55s} shape={ds.shape} span={span:.1f}s rate={rate:.1f}hz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("episode", type=Path, help="path to a raw episode h5 file")
    parser.add_argument("--cam", choices=[*CAM_SOURCES, "both"], default="both",
                        help="which stream to play (default: both, side by side)")
    parser.add_argument("--start", type=int, default=0,
                        help="first frame index (negative counts from the end)")
    parser.add_argument("--end", type=int, default=None,
                        help="play until before this frame (negative counts from the end)")
    parser.add_argument("--fps", default="auto",
                        help="'auto' derives from stamps, or a number (default: auto)")
    parser.add_argument("--scale", default="auto",
                        help="'auto' picks 2x for small frames, or an integer factor")
    parser.add_argument("--no-overlay", action="store_true", help="hide the info overlay")
    parser.add_argument("--list", action="store_true", help="list the episode sources and exit")
    args = parser.parse_args()

    if not args.episode.exists():
        sys.exit(f"[ERROR]: no such file: {args.episode}")

    cams = list(CAM_SOURCES) if args.cam == "both" else [args.cam]

    with h5py.File(args.episode, "r") as h5:
        if args.list:
            list_streams(h5, args.episode)
            return

        if args.scale == "auto":
            sample_h = h5[f"observations/{CAM_SOURCES[cams[0]]}"].shape[1]
            scale = 2 if sample_h < 400 else 1
        else:
            scale = int(args.scale)

        player = EpisodePlayer(h5, cams, scale=scale, show_overlay=not args.no_overlay)
        fps = auto_fps(player.streams[cams[0]][1]) if args.fps == "auto" else float(args.fps)

        n = player.n
        start = max(0, n + args.start) if args.start < 0 else min(args.start, n - 1)
        end = n if args.end is None else (max(0, n + args.end) if args.end < 0 else min(args.end, n))
        if end <= start:
            sys.exit(f"[ERROR]: empty range start={start} end={end} (n={n})")

        print(f"[INFO]: playing {args.episode.name} cams={cams} frames [{start},{end}) "
              f"at {fps:.1f}fps (scale x{scale})")
        player.run(start, end, fps)


if __name__ == "__main__":
    main()
