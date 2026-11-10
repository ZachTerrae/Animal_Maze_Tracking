"""Runs DeepLabCut on a batch of videos, in a process of its own.

The main window starts this file as a separate program::

    python -m posetally.worker --job job.json

and reads its progress as lines of JSON. Keeping DeepLabCut out of the main
window's process means the window stays responsive, a crash in the network code
cannot take the window down, "Cancel" is simply ending this process, and the
graphics card's memory is released when it ends.

Every line this program prints for the main window starts with ``PREFIX``.
Anything else on the output is DeepLabCut's own chatter and is shown in the log.
"""

from __future__ import annotations

import argparse
import fnmatch
import inspect
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

PREFIX = "@@POSETALLY@@ "

_out = sys.stdout
_lock = threading.Lock()
_mid_line = False  # True while DeepLabCut has printed part of a line


def emit(event: str, **data) -> None:
    """Send one message to the main window. Always starts on a line of its own."""
    global _mid_line
    line = PREFIX + json.dumps({"event": event, **data}, default=str)
    with _lock:
        try:
            _out.write(("\n" if _mid_line else "") + line + "\n")
            _out.flush()
            _mid_line = False
        except Exception:
            pass


def parse_line(line: str) -> tuple[str, dict | None]:
    """Split a line of worker output into (ordinary text, message or None)."""
    at = line.find(PREFIX)
    if at < 0:
        return line, None
    try:
        message = json.loads(line[at + len(PREFIX) :])
    except ValueError:
        return line, None
    return line[:at], message if isinstance(message, dict) else None


class _Watch:
    """Passes text through to the real output while letting us read it line by line."""

    def __init__(self, stream, callback):
        self._stream = stream
        self._callback = callback
        self._partial = ""

    def write(self, text):
        global _mid_line
        text = str(text)
        with _lock:
            written = self._stream.write(text)
            if text:
                _mid_line = not text.endswith("\n")
        try:
            # print("a", b) arrives as several writes; put the line back together first
            self._partial += text
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                self._callback(line)
            if len(self._partial) > 20000:
                self._partial = self._partial[-2000:]
        except Exception:
            pass
        return written

    def flush(self):
        with _lock:
            return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def environment_report() -> dict:
    """What is installed and which devices can run the network."""
    report: dict = {"python": sys.version.split()[0], "executable": sys.executable}
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda"] = bool(torch.cuda.is_available())
        if report["cuda"]:
            try:
                report["cuda_name"] = torch.cuda.get_device_name(0)
            except Exception:
                report["cuda_name"] = "NVIDIA GPU"
        mps = getattr(torch.backends, "mps", None)
        report["mps"] = bool(mps is not None and mps.is_available())
    except Exception as exc:
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import deeplabcut

        report["deeplabcut"] = getattr(deeplabcut, "__version__", "unknown")
    except Exception as exc:
        report["deeplabcut_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import tensorflow  # noqa: F401

        report["tensorflow"] = tensorflow.__version__
    except Exception:
        report["tensorflow"] = None
    return report


# DeepLabCut's messages for situations the user can do something about, in plain words.
_PLAIN_WORDS = (
    (
        "tracklets are empty",
        "DeepLabCut did not find any animal in this video, so there was nothing to link into tracks.",
    ),
    (
        "could not reconstruct",
        "DeepLabCut could not link what it found into continuous tracks. If these videos hold fewer animals "
        "than the project was set up for, set “Animals per video” in the Track window.",
    ),
    (
        "out of memory",
        "The computer or its graphics card ran out of memory. In the Track window, set “Frames at a time” "
        "to a smaller number, or choose “Processor only” under “Run on”.",
    ),
)


def in_plain_words(message: str) -> str:
    """Put an explanation in front of the DeepLabCut messages we know how to explain."""
    lowered = message.lower()
    for key, text in _PLAIN_WORDS:
        if key in lowered:
            return f"{text} (DeepLabCut's message: {message})"
    return message


def _video_opens(path: str) -> str | None:
    """Return a message if OpenCV cannot read the first frame of a video."""
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                return "The video file could not be opened."
            ok, frame = cap.read()
            if not ok or frame is None:
                return "The video file opened but no frame could be read from it."
        finally:
            cap.release()
    except Exception as exc:
        return f"The video file could not be checked: {exc}"
    return None


class Job:
    def __init__(self, spec: dict):
        self.config = spec["config"]
        self.videos: list[str] = [os.path.abspath(v) for v in spec["videos"]]
        self.shuffle = int(spec.get("shuffle", 1))
        self.trainingsetindex = int(spec.get("trainingsetindex", 0))
        self.engine = str(spec.get("engine", "pytorch"))
        self.destfolder = spec.get("destfolder") or None
        self.save_csv = bool(spec.get("save_csv", False))
        self.device = str(spec.get("device", "auto") or "auto")
        self.n_tracks = int(spec.get("n_animals", 0) or 0)
        self.batch_size = int(spec.get("batch_size", 0) or 0)
        self.multi_animal = bool(spec.get("multi_animal", False))
        self.tag = str(spec.get("tag", ""))
        self.started = time.time()
        self.done: set[str] = set()
        self.failed: set[str] = set()
        self.current: str | None = None
        self.setup_ok = False
        self._index = None

    # -- finding results ----------------------------------------------------------

    def output_for(self, video: str, index=None) -> str | None:
        """The finished tracking table for a video made by this model, if it exists."""
        from .tracks import TrackIndex

        index = index or TrackIndex()
        stem = os.path.splitext(os.path.basename(video))[0]
        folders = [self.destfolder] if self.destfolder else [os.path.dirname(video)]
        best = None
        for cand in index.candidates(stem, folders, self.tag):
            if not cand.is_h5 or cand.filtered:
                continue
            if self.tag and not cand.matches_model:
                continue
            if self.multi_animal and not cand.tracker:
                continue
            if best is None or cand.mtime > best.mtime:
                best = cand
        return best.path if best else None

    def stale_detections(self, video: str) -> bool:
        """True if an earlier, interrupted run left detections but no final table."""
        stem = os.path.splitext(os.path.basename(video))[0]
        folder = self.destfolder or os.path.dirname(video)
        try:
            names = os.listdir(folder)
        except OSError:
            return False
        for name in names:
            if name.startswith(stem + "DLC") and name.endswith("_full.pickle") and (not self.tag or self.tag in name):
                try:
                    if os.path.getmtime(os.path.join(folder, name)) < self.started:
                        return True
                except OSError:
                    pass
        return False

    def mark_done(self, video: str, index=None) -> bool:
        if video in self.done:
            return True
        output = self.output_for(video, index)
        if output:
            self.done.add(video)
            emit("video_done", video=video, output=output)
            return True
        return False

    # -- watching DeepLabCut's messages ------------------------------------------

    def saw_text(self, text: str, pending: list[str]) -> None:
        lowered = text.lower()
        if any(k in lowered for k in ("using scorer", "using snapshot", "starting to analyze", "analyzing videos with")):
            self.setup_ok = True
        if "starting to analyze" in lowered or "loading " in lowered:
            for video in pending:
                if video in self.done or video in self.failed:
                    continue
                if video in text or os.path.basename(video) in text:
                    if self.current and self.current != video:
                        self.mark_done(self.current)
                    if self.current != video:
                        self.current = video
                        emit("video_start", video=video)
                    break

    # -- calling DeepLabCut --------------------------------------------------------

    def build_kwargs(self, analyze, overwrite: bool = False) -> dict:
        params = inspect.signature(analyze).parameters
        accepts_extra = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs: dict = {"shuffle": self.shuffle, "trainingsetindex": self.trainingsetindex, "save_as_csv": self.save_csv}
        if self.destfolder:
            kwargs["destfolder"] = self.destfolder
        if "in_random_order" in params:
            kwargs["in_random_order"] = False
        if self.n_tracks > 0 and "n_tracks" in params:
            kwargs["n_tracks"] = self.n_tracks
        if self.batch_size > 0:
            if "batch_size" in params:
                kwargs["batch_size"] = self.batch_size
            elif "batchsize" in params:
                kwargs["batchsize"] = self.batch_size
        if self.engine == "pytorch" and self.device != "auto" and ("device" in params or accepts_extra):
            kwargs["device"] = self.device
        if overwrite and "overwrite" in params:
            kwargs["overwrite"] = True
        return kwargs

    def run(self) -> int:
        emit("status", message="Starting DeepLabCut (this can take a minute the first time)…")
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("DLClight", "True")
        # On Apple chips: steps the graphics chip cannot do run on the processor instead of failing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            import deeplabcut
        except Exception as exc:
            emit(
                "fatal",
                kind="no_deeplabcut",
                error=f"DeepLabCut could not be started: {type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            return 2
        emit("ready", **environment_report())

        from .tracks import TrackIndex

        pending: list[str] = []
        listing = TrackIndex()
        for video in self.videos:
            if not os.path.isfile(video):
                self.failed.add(video)
                emit("video_error", video=video, error="The video file was not found.")
                continue
            if self.mark_done(video, listing):
                continue
            name = os.path.basename(video)
            if fnmatch.fnmatch(name, "*_labeled.*") or fnmatch.fnmatch(name, "*_full.*"):
                self.failed.add(video)
                emit(
                    "video_error",
                    video=video,
                    error=(
                        "DeepLabCut skips videos whose names end in “_labeled” or “_full”, because it gives "
                        "those names to the videos it creates itself. Rename this video and add it again."
                    ),
                )
                continue
            problem = _video_opens(video)
            if problem:
                self.failed.add(video)
                emit("video_error", video=video, error=problem)
                continue
            pending.append(video)

        stop_polling = threading.Event()

        def poll() -> None:
            # Safety net for progress: notice finished videos even if DeepLabCut's
            # messages change in a future version. One folder listing per round.
            while not stop_polling.wait(3.0):
                fresh = TrackIndex()
                for video in list(pending):
                    if video not in self.done and video not in self.failed and video != self.current:
                        self.mark_done(video, fresh)

        watcher = threading.Thread(target=poll, daemon=True)
        watcher.start()

        same_error_streak = 0
        last_error = ""
        code = 0
        sys.stdout = _Watch(sys.stdout, lambda text: self.saw_text(text, pending))
        try:
            # DeepLabCut works through the videos in sorted order, whatever order it is given
            remaining = sorted(pending, key=lambda v: Path(v).absolute())
            while remaining:
                self.current = None
                self.setup_ok = False
                kwargs = self.build_kwargs(deeplabcut.analyze_videos)
                try:
                    deeplabcut.analyze_videos(self.config, list(remaining), **kwargs)
                except KeyboardInterrupt:
                    raise
                except BaseException as exc:  # DeepLabCut failed on some video or on set-up
                    message = f"{type(exc).__name__}: {exc}"
                    trace = traceback.format_exc()
                    for video in remaining:
                        self.mark_done(video)
                    left = [v for v in remaining if v not in self.done]
                    if not left:
                        break
                    if not self.setup_ok:
                        emit("fatal", kind="setup", error=message, traceback=trace)
                        code = 2
                        break
                    culprit = self.current if self.current in left else left[0]
                    self.failed.add(culprit)
                    emit("video_error", video=culprit, error=in_plain_words(message), traceback=trace)
                    same_error_streak = same_error_streak + 1 if message == last_error else 1
                    last_error = message
                    if same_error_streak >= 5 and not self.done:
                        emit(
                            "fatal",
                            kind="repeated",
                            error="Every video fails in the same way, so the run was stopped. " + in_plain_words(message),
                            traceback=trace,
                        )
                        code = 2
                        break
                    remaining = [v for v in left if v != culprit]
                    continue
                for video in remaining:
                    self.mark_done(video)
                remaining = []

            # An earlier interrupted run can leave detections without a final table.
            # DeepLabCut then says "already analyzed" and skips the video; redo those.
            if code == 0:
                redo = [v for v in pending if v not in self.done and v not in self.failed and self.stale_detections(v)]
                kwargs = self.build_kwargs(deeplabcut.analyze_videos, overwrite=True)
                if redo and kwargs.get("overwrite"):
                    emit("status", message=f"Finishing {len(redo)} video(s) left half-done by an earlier run…")
                    for video in redo:
                        self.current = None
                        try:
                            deeplabcut.analyze_videos(self.config, [video], **kwargs)
                        except KeyboardInterrupt:
                            raise
                        except BaseException as exc:
                            self.failed.add(video)
                            emit(
                                "video_error",
                                video=video,
                                error=in_plain_words(f"{type(exc).__name__}: {exc}"),
                                traceback=traceback.format_exc(),
                            )
                            continue
                        self.mark_done(video)

            if code == 0:
                for video in pending:
                    if video in self.done or video in self.failed:
                        continue
                    if not self.mark_done(video):
                        self.failed.add(video)
                        emit(
                            "video_error",
                            video=video,
                            error=(
                                "DeepLabCut finished without producing a tracking table for this video. "
                                "This usually means no animal was detected in it."
                            ),
                        )
        finally:
            stop_polling.set()
            sys.stdout = _out
        emit("finished", done=len(self.done), failed=len(self.failed), total=len(self.videos))
        return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PoseTally's DeepLabCut runner")
    parser.add_argument("--job", help="JSON file describing the videos to analyze")
    parser.add_argument("--check", action="store_true", help="print what is installed and exit")
    args = parser.parse_args(argv)
    if args.check:
        emit("environment", **environment_report())
        return 0
    if not args.job:
        parser.error("--job or --check is required")
    try:
        with open(args.job, encoding="utf-8") as handle:
            spec = json.load(handle)
        return Job(spec).run()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        emit("fatal", kind="internal", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
