"""Turning tracking plus a scoring setup into numbers.

For each animal in a video the steps are:

1. Clean the path of every body part that a rule uses: drop points DeepLabCut
   was unsure about, bridge short gaps, and remove single-frame jumps.
2. Find the events: crossings of each line (with their direction) and entries
   into and exits from each region.
3. Apply the rules attached to each shape and add up the columns.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .geometry import (
    DIRECTION_LABELS,
    crossing_vector,
    find_line_crossings,
    movement_matches,
    trace_region,
)
from .model import ResolvedShapes, Rule, ScoringSettings
from .tracks import TrackSet

NAN = float("nan")


@dataclass
class ScoredEvent:
    """One line of the event log."""

    frame: int
    time_s: float
    animal: str
    shape_id: str
    shape_name: str
    shape_kind: str  # "line" or "roi"
    what: str  # e.g. "crossed left to right", "entered", "left"
    bodypart: str
    column: str
    points: float | None  # None: listed for information, adds no points itself
    gap_frames: int = 0  # untracked frames just before the event
    rule_id: str = ""


@dataclass
class AnimalResult:
    animal: str
    scores: dict[str, float] = field(default_factory=dict)  # score columns, NaN = empty
    events: list[ScoredEvent] = field(default_factory=list)
    tracked: dict[str, float] = field(default_factory=dict)  # body part -> share of frames tracked
    gap_events: int = 0
    notes: list[str] = field(default_factory=list)

    def tracked_percent(self) -> float | None:
        if not self.tracked:
            return None
        return 100.0 * min(self.tracked.values())


@dataclass
class Trajectory:
    xy: np.ndarray  # (N, 2), NaN where the point is not usable
    tracked: float  # share of frames at or above the likelihood cut-off


# ------------------------------------------------------------------- path clean-up


def fill_short_gaps(xy: np.ndarray, max_gap: int) -> np.ndarray:
    """Join the dots across gaps of up to ``max_gap`` frames (straight line)."""
    n = len(xy)
    bad = np.isnan(xy[:, 0]) | np.isnan(xy[:, 1])
    if max_gap <= 0 or not bad.any() or bad.all():
        return xy
    out = xy.copy()
    idx = np.arange(n)
    good = ~bad
    for c in (0, 1):
        out[bad, c] = np.interp(idx[bad], idx[good], xy[good, c])
    edges = np.diff(np.concatenate(([0], bad.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for start, end in zip(starts, ends):
        if start == 0 or end == n or (end - start) > max_gap:
            out[start:end] = np.nan
    return out


def median_smooth(xy: np.ndarray, window: int) -> np.ndarray:
    """Running median over ``window`` frames. Missing frames stay missing."""
    window = int(window)
    if window < 3 or len(xy) < 3:
        return xy
    if window % 2 == 0:
        window += 1
    half = window // 2
    missing = np.isnan(xy[:, 0]) | np.isnan(xy[:, 1])
    out = xy.copy()
    pad = np.full(half, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for c in (0, 1):
            padded = np.concatenate([pad, xy[:, c], pad])
            view = np.lib.stride_tricks.sliding_window_view(padded, window)
            out[:, c] = np.nanmedian(view, axis=1)
    out[missing] = np.nan
    return out


def prepare_trajectory(arr: np.ndarray, settings: ScoringSettings, fps: float) -> Trajectory:
    """Clean one body part's (N, 3) table of x, y, likelihood."""
    xy = np.array(arr[:, :2], dtype=float)
    lik = arr[:, 2]
    with np.errstate(invalid="ignore"):
        good = np.isfinite(xy).all(axis=1) & (lik >= settings.pcutoff)
    tracked = float(good.mean()) if len(good) else 0.0
    xy[~good] = np.nan
    max_gap = int(round(settings.max_gap_s * fps)) if fps > 0 else 0
    if max_gap > 0:
        xy = fill_short_gaps(xy, max_gap)
    if settings.smooth_frames >= 3:
        xy = median_smooth(xy, settings.smooth_frames)
    return Trajectory(xy=xy, tracked=tracked)


# ------------------------------------------------------------------------- scoring


@dataclass
class _Hit:
    frame: int
    what: str
    gap: int = 0


def _describe_movement(direction: str, mx: float, my: float) -> str:
    if direction != "any":
        return "crossed " + DIRECTION_LABELS[direction]
    if abs(mx) >= abs(my):
        return "crossed " + DIRECTION_LABELS["left_to_right" if mx > 0 else "right_to_left"]
    return "crossed " + DIRECTION_LABELS["top_to_bottom" if my > 0 else "bottom_to_top"]


class _AnimalScorer:
    def __init__(self, animal: str, tracks: TrackSet, shapes: ResolvedShapes, settings: ScoringSettings, fps: float):
        self.animal = animal
        self.tracks = tracks
        self.shapes = shapes
        self.settings = settings
        self.fps = fps if fps and fps > 0 else settings.default_fps
        self._trajectories: dict[str, Trajectory | None] = {}
        self._crossings: dict[tuple[str, str], list] = {}
        self._traces: dict[tuple[str, str], object] = {}
        self.result = AnimalResult(animal=animal)

    # -- inputs -----------------------------------------------------------------

    def trajectory(self, bodypart: str) -> Trajectory | None:
        if bodypart not in self._trajectories:
            arr = self.tracks.get(self.animal, bodypart)
            traj = prepare_trajectory(arr, self.settings, self.fps) if arr is not None else None
            self._trajectories[bodypart] = traj
        return self._trajectories[bodypart]

    def heading(self, frame: int) -> tuple[float, float] | None:
        """Vector from tail to head on a frame, if both are tracked."""
        s = self.settings
        if not (s.use_heading and s.head_part and s.tail_part):
            return None
        head = self.trajectory(s.head_part)
        tail = self.trajectory(s.tail_part)
        if head is None or tail is None or frame >= len(head.xy):
            return None
        hx, hy = head.xy[frame]
        tx, ty = tail.xy[frame]
        if not (np.isfinite(hx) and np.isfinite(hy) and np.isfinite(tx) and np.isfinite(ty)):
            return None
        return (float(hx - tx), float(hy - ty))

    # -- events -----------------------------------------------------------------

    def line_hits(self, resolved, rule: Rule, traj: Trajectory) -> list[_Hit]:
        key = (resolved.shape.id, rule.bodypart)
        if key not in self._crossings:
            try:
                self._crossings[key] = find_line_crossings(traj.xy, resolved.p1, resolved.p2, self.settings.margin_px)
            except ValueError:  # a line with no length
                self._crossings[key] = []
        hits: list[_Hit] = []
        for crossing in self._crossings[key]:
            mx, my = crossing_vector(resolved.p1, resolved.p2, crossing.toward)
            if not movement_matches(rule.direction, mx, my):
                continue
            if crossing.gap > 0 and not self.settings.count_gap_crossings:
                continue
            facing = self.heading(crossing.frame)
            if facing is not None and facing[0] * mx + facing[1] * my <= 0:
                continue  # moving backwards across the line
            hits.append(_Hit(crossing.frame, _describe_movement(rule.direction, mx, my), crossing.gap))
        return hits

    def region_trace(self, resolved, rule: Rule, traj: Trajectory):
        key = (resolved.shape.id, rule.bodypart)
        if key not in self._traces:
            self._traces[key] = trace_region(traj.xy, resolved.outline, self.settings.margin_px)
        return self._traces[key]

    # -- everything -------------------------------------------------------------

    def run(self) -> AnimalResult:
        res = self.result
        fps = self.fps
        hits_by_rule: dict[str, list[_Hit]] = {}  # event rules and first_time rules
        info_by_rule: dict[str, list[_Hit]] = {}  # entries/exits listed behind a time total
        seconds_by_rule: dict[str, float] = {}
        owners: dict[str, tuple] = {}  # rule id -> (resolved shape, rule)
        skipped: set[str] = set()
        missing_parts: list[str] = []

        def note_missing(part: str) -> None:
            if part and part not in missing_parts:
                missing_parts.append(part)

        for resolved in self.shapes.lines:
            for rule in resolved.shape.rules:
                owners[rule.id] = (resolved, rule)
                traj = self.trajectory(rule.bodypart) if rule.bodypart else None
                if traj is None or not rule.column:
                    skipped.add(rule.id)
                    note_missing(rule.bodypart if traj is None else "")
                    continue
                res.tracked[rule.bodypart] = traj.tracked
                hits_by_rule[rule.id] = self.line_hits(resolved, rule, traj)

        for resolved in self.shapes.rois:
            for rule in resolved.shape.rules:
                owners[rule.id] = (resolved, rule)
                traj = self.trajectory(rule.bodypart) if rule.bodypart else None
                if traj is None or not rule.column:
                    skipped.add(rule.id)
                    note_missing(rule.bodypart if traj is None else "")
                    continue
                res.tracked[rule.bodypart] = traj.tracked
                trace = self.region_trace(resolved, rule, traj)
                entries = [_Hit(int(f), "entered", int(g)) for f, g in zip(trace.entries, trace.entry_gaps)]
                exits = [_Hit(int(f), "left", int(g)) for f, g in zip(trace.exits, trace.exit_gaps)]
                if rule.measure == "time":
                    inside = trace.state == 1
                    if self.settings.roi_gap_policy == "exclude":
                        inside = inside & trace.valid
                    seconds_by_rule[rule.id] = float(inside.sum()) / fps
                    info = entries + exits
                    if trace.first_known >= 0 and trace.state[trace.first_known] == 1:
                        info.append(_Hit(trace.first_known, "already inside when first tracked"))
                    info_by_rule[rule.id] = sorted(info, key=lambda h: h.frame)
                elif rule.measure == "entries":
                    hits_by_rule[rule.id] = entries
                elif rule.measure == "exits":
                    hits_by_rule[rule.id] = exits
                else:  # first_time
                    first = list(entries)
                    if trace.first_known >= 0 and trace.state[trace.first_known] == 1:
                        first.append(_Hit(trace.first_known, "already inside when first tracked"))
                    hits_by_rule[rule.id] = sorted(first, key=lambda h: h.frame)

        # "first choice" groups: only the rule whose event happened first counts
        earliest: dict[str, int] = {}
        for rule_id, hits in hits_by_rule.items():
            rule = owners[rule_id][1]
            if rule.is_event_rule and rule.count_mode == "first_in_group" and rule.group and hits:
                first_frame = hits[0].frame
                if rule.group not in earliest or first_frame < earliest[rule.group]:
                    earliest[rule.group] = first_frame

        contributions: dict[str, list[float | None]] = {}
        for rule_id, (resolved, rule) in owners.items():
            if not rule.column:
                continue
            bucket = contributions.setdefault(rule.column, [])
            if rule_id in skipped:
                bucket.append(None)
                continue
            shape = resolved.shape

            def log(hit: _Hit, points: float | None) -> None:
                res.events.append(
                    ScoredEvent(
                        frame=hit.frame,
                        time_s=hit.frame / fps,
                        animal=self.animal,
                        shape_id=shape.id,
                        shape_name=shape.name,
                        shape_kind=shape.kind,
                        what=hit.what,
                        bodypart=rule.bodypart,
                        column=rule.column,
                        points=points,
                        gap_frames=hit.gap,
                        rule_id=rule.id,
                    )
                )
                if points is not None and hit.gap > 0:
                    res.gap_events += 1

            if rule.measure == "time":
                bucket.append(seconds_by_rule.get(rule_id, 0.0) * rule.points)
                for hit in info_by_rule.get(rule_id, []):
                    log(hit, None)
                continue

            hits = hits_by_rule.get(rule_id, [])
            if rule.measure == "first_time":
                if hits:
                    bucket.append(hits[0].frame / fps)
                    log(hits[0], None)
                else:
                    bucket.append(None)
                continue

            if rule.count_mode == "first":
                hits = hits[:1]
            elif rule.count_mode == "first_in_group":
                wins = bool(hits) and bool(rule.group) and earliest.get(rule.group) == hits[0].frame
                hits = hits[:1] if wins else []
            bucket.append(rule.points * len(hits))
            for hit in hits:
                log(hit, rule.points)

        for column, values in contributions.items():
            present = [v for v in values if v is not None]
            res.scores[column] = float(sum(present)) if present else NAN

        res.events.sort(key=lambda e: (e.frame, e.shape_name, e.column))
        for part in missing_parts:
            res.notes.append(f"Body part “{part}” is not in the tracking file.")
        return res


def score_tracks(tracks: TrackSet, shapes: ResolvedShapes, settings: ScoringSettings, fps: float) -> list[AnimalResult]:
    """Score every animal in one video. Returns one result per animal."""
    return [_AnimalScorer(animal, tracks, shapes, settings, fps).run() for animal in tracks.animals()]
