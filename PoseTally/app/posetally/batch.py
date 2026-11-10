"""Scoring a batch of videos and laying the results out as a table."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from .formulas import evaluate_formulas
from .idcolumn import make_id
from .model import Layout
from .scoring import AnimalResult, score_tracks
from .tracks import TrackFormatError, TrackSet, load_tracks
from .videoio import VideoInfo, probe_video

NAN = float("nan")


@dataclass
class VideoItem:
    """One entry in the batch: a video, its tracking file, or both."""

    key: str  # unique: the video path, or the tracking path when there is no video
    name: str  # file name without extension
    video_path: str | None = None
    track_path: str | None = None
    track_matches_model: bool = False
    manual_track: bool = False  # the tracking file was picked by hand
    info: VideoInfo | None = None

    @property
    def tracked(self) -> bool:
        return bool(self.track_path)

    @property
    def source(self) -> str:
        return self.video_path or self.track_path or self.name


@dataclass
class VideoResult:
    key: str
    name: str
    source: str
    track_path: str | None = None
    fps: float = 0.0
    n_frames: int = 0
    frame_size: tuple[int, int] | None = None
    multi_animal: bool = False
    animals: list[AnimalResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


def frame_size_from_tracks(tracks: TrackSet) -> tuple[int, int] | None:
    """Best guess at the frame size when the video file itself is not available."""
    size = tracks.meta.get("frame_size")
    if size and size[0] > 0 and size[1] > 0:
        return (int(size[0] + tracks.crop_offset[0]), int(size[1] + tracks.crop_offset[1]))
    best_x = best_y = 0.0
    for arr in tracks.data.values():
        with np.errstate(invalid="ignore"):
            if np.isfinite(arr[:, 0]).any():
                best_x = max(best_x, float(np.nanmax(arr[:, 0])))
                best_y = max(best_y, float(np.nanmax(arr[:, 1])))
    if best_x <= 0 or best_y <= 0:
        return None
    return (int(math.ceil(best_x * 1.05)), int(math.ceil(best_y * 1.05)))


def describe_item(item: VideoItem, tracks: TrackSet | None, default_fps: float) -> tuple[float, tuple[int, int] | None, list[str]]:
    """Frame rate and frame size to use for an item, with notes about any guesswork."""
    notes: list[str] = []
    if item.info is None:
        item.info = probe_video(item.video_path) if item.video_path else VideoInfo()
    info = item.info
    fps = info.fps
    if not fps > 0 and tracks is not None:
        fps = float(tracks.meta.get("fps") or 0.0)
        if fps > 0 and item.video_path:
            notes.append("Frame rate taken from DeepLabCut's record because the video did not report one.")
    if not fps > 0:
        fps = default_fps
        notes.append(f"Frame rate unknown; {default_fps:g} frames per second assumed.")
    size = info.size
    if size is None and tracks is not None:
        size = frame_size_from_tracks(tracks)
    return fps, size, notes


def score_item(item: VideoItem, layout: Layout, tracks: TrackSet | None = None) -> VideoResult:
    """Score one video. Never raises: problems are reported in the result."""
    result = VideoResult(key=item.key, name=item.name, source=item.source, track_path=item.track_path)
    if not item.track_path:
        result.error = "Not tracked yet"
        return result
    try:
        if tracks is None:
            tracks = load_tracks(item.track_path)
        fps, size, notes = describe_item(item, tracks, layout.settings.default_fps)
        result.fps = fps
        result.frame_size = size
        result.n_frames = tracks.n_frames
        result.multi_animal = tracks.multi_animal
        result.notes.extend(notes)
        if item.info and item.info.n_frames and abs(item.info.n_frames - tracks.n_frames) > 2:
            result.notes.append(
                f"The video reports {item.info.n_frames} frames but the tracking has {tracks.n_frames}."
            )
        if tracks.crop_offset != (0.0, 0.0):
            result.notes.append("Tracking was run on a cropped area; positions were moved back onto the full frame.")
        shapes = layout.resolve(item.name, size)
        if shapes.scaled and (shapes.lines or shapes.rois):
            result.notes.append("Frame size differs from the one the shapes were drawn on; shapes were stretched to fit.")
        if any(s.custom for s in shapes.lines + shapes.rois):
            result.notes.append("Uses its own shape positions.")
        result.animals = score_tracks(tracks, shapes, layout.settings, fps)
    except TrackFormatError as exc:
        result.error = str(exc)
    except Exception as exc:  # one bad file must not stop a batch of a thousand
        result.error = f"Could not score: {exc}"
    return result


# --------------------------------------------------------------------------- table


@dataclass
class TableRow:
    video_key: str
    video_id: str
    animal: str
    values: dict[str, float]
    tracked_percent: float | None = None
    gap_events: int = 0
    notes: str = ""
    error: str | None = None


@dataclass
class ResultsTable:
    columns: list[str] = field(default_factory=list)
    rows: list[TableRow] = field(default_factory=list)
    show_animal: bool = False
    formula_errors: dict[str, str] = field(default_factory=dict)
    id_header: str = "File"


def animal_label(name: str) -> str:
    return name if name else ""


def build_table(results: list[VideoResult], layout: Layout) -> ResultsTable:
    """One row per video and animal, one column per score or formula."""
    table = ResultsTable(columns=layout.columns(), id_header=layout.id_column.header or "File")
    score_columns = layout.score_columns()
    formulas = layout.active_formulas()
    table.show_animal = any(r.multi_animal for r in results)
    for result in results:
        video_id = make_id(layout.id_column, result.source)
        if result.error or not result.animals:
            blank, errors = evaluate_formulas({c: NAN for c in score_columns}, formulas)
            table.formula_errors.update(errors)
            table.rows.append(
                TableRow(
                    video_key=result.key,
                    video_id=video_id,
                    animal="",
                    values={c: NAN for c in blank},
                    notes="; ".join(result.notes),
                    error=result.error or "No animals in the tracking file",
                )
            )
            continue
        for animal in result.animals:
            base = {c: animal.scores.get(c, NAN) for c in score_columns}
            values, errors = evaluate_formulas(base, formulas)
            table.formula_errors.update(errors)
            notes = list(result.notes) + list(animal.notes)
            table.rows.append(
                TableRow(
                    video_key=result.key,
                    video_id=video_id,
                    animal=animal_label(animal.animal),
                    values=values,
                    tracked_percent=animal.tracked_percent(),
                    gap_events=animal.gap_events,
                    notes="; ".join(notes),
                )
            )
    return table


def natural_key(text: str):
    """Sort key that puts "trial2" before "trial10"."""
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def default_export_name(project_name: str | None) -> str:
    base = (project_name or "scores").strip() or "scores"
    safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in base)
    return f"{safe} scores.xlsx"


def relative_or_absolute(path: str, base_folder: str) -> str:
    """Store a path relative to a folder when it lives inside it."""
    try:
        rel = os.path.relpath(path, base_folder)
    except ValueError:  # different drive on Windows
        return path
    return path if rel.startswith("..") else rel
