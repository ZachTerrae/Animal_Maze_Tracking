"""A small made-up project for trying the program without a trained network.

It draws simple videos of a blob moving through a T-maze (one animal) and
around an open arena (two animals) and writes tracking files in exactly the
layout DeepLabCut uses, so the rest of the program treats them like the real
thing. The same data is used by the automated tests, where the right answers
are known in advance.
"""

from __future__ import annotations

import json
import math
import os
import pickle

import numpy as np

from .idcolumn import IdColumn
from .model import FormulaColumn, Layout, LineShape, RoiShape, Rule

FRAME_W, FRAME_H = 640, 480
FPS = 20.0
BODYPARTS = ["nose", "head", "body", "tailbase"]
PART_OFFSETS = {"nose": 0.0, "head": 8.0, "body": 22.0, "tailbase": 38.0}  # pixels behind the nose
TASK, DATE, SCORER_NAME = "MazeDemo", "Nov10", "demo"
DLC_SCORER = f"DLC_Resnet50_{TASK}{DATE}shuffle1_snapshot_200"

START = (320.0, 430.0)
JUNCTION = (320.0, 110.0)
LEFT_END = (100.0, 110.0)
RIGHT_END = (540.0, 110.0)

# name -> list of (x, y, frames to wait there)
MAZE_TRIALS = {
    "trial_01_groupA": [(*START, 30), (*JUNCTION, 5), (*LEFT_END, 60)],
    "trial_02_groupA": [(*START, 20), (*JUNCTION, 5), (*RIGHT_END, 25), (*JUNCTION, 5), (*LEFT_END, 40)],
    "trial_03_groupB": [(*START, 45), (*JUNCTION, 10), (280.0, 110.0, 10), (*JUNCTION, 5), (*RIGHT_END, 30), (*JUNCTION, 5), (*START, 20)],
}


def walk(waypoints, speed: float = 4.0) -> np.ndarray:
    """Positions frame by frame for a point that visits the waypoints in turn."""
    points = [(float(waypoints[0][0]), float(waypoints[0][1]))]
    for _ in range(int(waypoints[0][2])):
        points.append(points[-1])
    for (x0, y0, _), (x1, y1, wait) in zip(waypoints[:-1], waypoints[1:]):
        distance = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(round(distance / speed)))
        for i in range(1, steps + 1):
            t = i / steps
            points.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
        for _ in range(int(wait)):
            points.append((x1, y1))
    return np.asarray(points, dtype=float)


def trail(nose: np.ndarray, behind: float) -> np.ndarray:
    """A point that follows the same path, ``behind`` pixels back along it."""
    if behind <= 0:
        return nose.copy()
    steps = np.hypot(*np.diff(nose, axis=0).T)
    travelled = np.concatenate([[0.0], np.cumsum(steps)])
    # Before moving the body lies straight back from the first direction of travel.
    moving = np.flatnonzero(steps > 1e-9)
    if len(moving):
        d = nose[moving[0] + 1] - nose[moving[0]]
        d = d / np.hypot(*d)
    else:
        d = np.array([0.0, -1.0])
    out = np.empty_like(nose)
    for i, s in enumerate(travelled):
        target = s - behind
        if target <= 0:
            out[i] = nose[0] + d * target  # target is negative: behind the start
        else:
            j = int(np.searchsorted(travelled, target))
            j = min(max(j, 1), len(travelled) - 1)
            span = travelled[j] - travelled[j - 1]
            t = 0.0 if span < 1e-12 else (target - travelled[j - 1]) / span
            out[i] = nose[j - 1] + (nose[j] - nose[j - 1]) * t
    return out


def body_parts_for(nose: np.ndarray) -> dict[str, np.ndarray]:
    return {part: trail(nose, PART_OFFSETS[part]) for part in BODYPARTS}


def _noisy(rng, xy: np.ndarray, sigma: float = 0.6) -> np.ndarray:
    return xy + rng.normal(0.0, sigma, size=xy.shape)


def _write_tracking(path_h5: str, parts_by_animal: dict[str, dict[str, np.ndarray]], multi: bool, rng, dropouts=()):
    """Save tracking the way DeepLabCut does (pandas table under 'df_with_missing')."""
    import pandas as pd

    columns, arrays = [], []
    n = len(next(iter(next(iter(parts_by_animal.values())).values())))
    for animal, parts in parts_by_animal.items():
        for part in BODYPARTS:
            xy = _noisy(rng, parts[part])
            likelihood = np.clip(rng.normal(0.97, 0.02, size=n), 0.0, 1.0)
            for start, stop in dropouts:
                likelihood[start:stop] = rng.uniform(0.01, 0.2, size=max(0, stop - start))
                xy[start:stop] += rng.normal(0.0, 40.0, size=xy[start:stop].shape)
            for coord, values in (("x", xy[:, 0]), ("y", xy[:, 1]), ("likelihood", likelihood)):
                columns.append((DLC_SCORER, animal, part, coord) if multi else (DLC_SCORER, part, coord))
                arrays.append(values)
    names = ["scorer", "individuals", "bodyparts", "coords"] if multi else ["scorer", "bodyparts", "coords"]
    df = pd.DataFrame(np.column_stack(arrays), columns=pd.MultiIndex.from_tuples(columns, names=names), index=range(n))
    df.to_hdf(path_h5, key="df_with_missing", format="table", mode="w")
    meta = {
        "data": {
            "Scorer": DLC_SCORER,
            "fps": FPS,
            "nframes": n,
            "frame_dimensions": (FRAME_W, FRAME_H),
            "cropping": False,
            "cropping_parameters": [0, FRAME_W, 0, FRAME_H],
            "pytorch-config": {},
        }
    }
    stem = path_h5[:-3]
    for suffix in ("_el",):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    with open(stem + "_meta.pickle", "wb") as handle:
        pickle.dump(meta, handle)


def _maze_background():
    from .videoio import get_cv2

    cv2 = get_cv2()
    img = np.full((FRAME_H, FRAME_W, 3), 52, dtype=np.uint8)
    floor = (176, 182, 178)
    cv2.rectangle(img, (60, 80), (580, 140), floor, -1)  # top corridor
    cv2.rectangle(img, (290, 140), (350, 380), floor, -1)  # stem
    cv2.rectangle(img, (270, 380), (370, 465), floor, -1)  # start box
    cv2.rectangle(img, (50, 60), (150, 160), floor, -1)  # left end
    cv2.rectangle(img, (490, 60), (590, 160), floor, -1)  # right end
    return img


def _arena_background():
    from .videoio import get_cv2

    cv2 = get_cv2()
    img = np.full((FRAME_H, FRAME_W, 3), 52, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (600, 440), (176, 182, 178), -1)
    return img


def _draw_animal(img, parts: dict[str, np.ndarray], frame: int, color) -> None:
    from .videoio import get_cv2

    cv2 = get_cv2()
    pts = np.array([parts[p][frame] for p in BODYPARTS])
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), color, 9, cv2.LINE_AA)
    cv2.circle(img, (int(round(pts[0][0])), int(round(pts[0][1]))), 6, color, -1, cv2.LINE_AA)


def _write_video(path_no_ext: str, background, animals: list[tuple[dict, tuple]], n_frames: int) -> str:
    from .videoio import get_cv2

    cv2 = get_cv2()
    for ext, fourcc in ((".mp4", "mp4v"), (".avi", "MJPG")):
        path = path_no_ext + ext
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), FPS, (FRAME_W, FRAME_H))
        if not writer.isOpened():
            writer.release()
            continue
        for frame in range(n_frames):
            img = background.copy()
            for parts, color in animals:
                _draw_animal(img, parts, frame, color)
            writer.write(img)
        writer.release()
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    raise RuntimeError("Could not write the demo video.")


def _write_config(folder: str, multi: bool) -> str:
    lines = [
        "# Project definitions (do not edit)",
        f"Task: {TASK}",
        f"scorer: {SCORER_NAME}",
        f"date: {DATE}",
        f"multianimalproject: {'true' if multi else 'false'}",
        "identity: false",
        "",
        "# Project path (change when moving around)",
        f"project_path: {json.dumps(os.path.abspath(folder))}",
        "",
        "video_sets: {}",
    ]
    if multi:
        lines += ["individuals:", "- animal1", "- animal2", "uniquebodyparts: []", "multianimalbodyparts:"]
        lines += [f"- {p}" for p in BODYPARTS]
        lines += ["bodyparts: MULTI!"]
    else:
        lines += ["bodyparts:"] + [f"- {p}" for p in BODYPARTS]
    lines += [
        "",
        "start: 0",
        "stop: 1",
        "numframes2pick: 20",
        "skeleton:",
        "- - nose",
        "  - head",
        "- - head",
        "  - body",
        "- - body",
        "  - tailbase",
        "skeleton_color: black",
        "pcutoff: 0.6",
        "dotsize: 8",
        "alphavalue: 0.7",
        "colormap: rainbow",
        "TrainingFraction:",
        "- 0.95",
        "iteration: 0",
        "default_net_type: resnet_50",
        "snapshotindex: -1",
        "batch_size: 8",
        "cropping: false",
        "engine: pytorch",
        "",
    ]
    path = os.path.join(folder, "config.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def maze_layout() -> Layout:
    """A ready-made example setup for the T-maze demo."""
    layout = Layout(ref_size=(FRAME_W, FRAME_H))
    left = LineShape(name="Left arm", p1=(240.0, 70.0), p2=(240.0, 150.0), color="#00e5ff")
    left.rules = [
        Rule(bodypart="head", measure="count", direction="right_to_left", points=1, column="Left entries"),
        Rule(bodypart="head", measure="count", direction="right_to_left", points=1, column="Chose left",
             count_mode="first_in_group", group="arm"),
    ]  # fmt: skip
    right = LineShape(name="Right arm", p1=(400.0, 70.0), p2=(400.0, 150.0), color="#ff5ad1")
    right.rules = [
        Rule(bodypart="head", measure="count", direction="left_to_right", points=1, column="Right entries"),
        Rule(bodypart="head", measure="count", direction="left_to_right", points=1, column="Chose right",
             count_mode="first_in_group", group="arm"),
    ]  # fmt: skip
    leave = LineShape(name="Leave start", p1=(370.0, 372.0), p2=(270.0, 372.0), color="#ffd400")
    leave.rules = [
        Rule(bodypart="head", measure="first_time", direction="bottom_to_top", column="Time to leave start (s)"),
    ]
    box = RoiShape(
        name="Start box",
        form="polygon",
        points=[(270.0, 380.0), (370.0, 380.0), (370.0, 465.0), (270.0, 465.0)],
        color="#7dff5a",
    )
    box.rules = [Rule(bodypart="head", measure="time", column="Time in start box (s)")]
    goal = RoiShape(name="Left goal", form="ellipse", points=[(50.0, 60.0), (150.0, 160.0)], color="#ff8a3d")
    goal.rules = [Rule(bodypart="nose", measure="entries", points=1, column="Left goal visits")]
    layout.lines = [left, right, leave]
    layout.rois = [box, goal]
    layout.formulas = [FormulaColumn("Arm entries (total)", "[Left entries] + [Right entries]")]
    layout.id_column = IdColumn(header="Trial", mode="pieces", delimiters="_", pieces="1-2")
    layout.settings.head_part = "nose"
    layout.settings.tail_part = "tailbase"
    layout.sync_columns()
    return layout


def arena_paths(n_frames: int = 360) -> dict[str, np.ndarray]:
    """Two animals circling an open arena in opposite directions."""
    t = np.arange(n_frames) / n_frames
    one = np.column_stack([320 + 200 * np.cos(2 * np.pi * 2 * t), 240 + 130 * np.sin(2 * np.pi * 2 * t)])
    two = np.column_stack([320 + 90 * np.cos(-2 * np.pi * 3 * t + 1.0), 240 + 150 * np.sin(-2 * np.pi * 3 * t + 1.0)])
    return {"animal1": one, "animal2": two}


def arena_layout() -> Layout:
    layout = Layout(ref_size=(FRAME_W, FRAME_H))
    middle = LineShape(name="Midline", p1=(320.0, 40.0), p2=(320.0, 440.0), color="#ffd400")
    middle.rules = [
        Rule(bodypart="head", measure="count", direction="left_to_right", points=1, column="Left to right"),
        Rule(bodypart="head", measure="count", direction="right_to_left", points=1, column="Right to left"),
    ]
    centre = RoiShape(name="Centre", form="ellipse", points=[(220.0, 150.0), (420.0, 330.0)], color="#00e5ff")
    centre.rules = [
        Rule(bodypart="body", measure="time", column="Time in centre (s)"),
        Rule(bodypart="body", measure="entries", points=1, column="Centre entries"),
    ]
    layout.lines = [middle]
    layout.rois = [centre]
    layout.formulas = [FormulaColumn("Midline crossings", "[Left to right] + [Right to left]")]
    layout.sync_columns()
    return layout


def make_demo(parent_folder: str, with_videos: bool = True) -> dict[str, str]:
    """Create the demo projects. Returns the path of each config.yaml."""
    rng = np.random.default_rng(7)
    out: dict[str, str] = {}

    # ---- one animal in a T-maze
    single = os.path.join(parent_folder, "T-maze (one animal)")
    videos = os.path.join(single, "videos")
    os.makedirs(videos, exist_ok=True)
    out["single"] = _write_config(single, multi=False)
    background = _maze_background() if with_videos else None
    entries: list[tuple[str, bool]] = []
    for index, (name, waypoints) in enumerate(MAZE_TRIALS.items()):
        nose = walk(waypoints)
        parts = body_parts_for(nose)
        dropouts = [(60, 63)] if index == 1 else ()
        track_file = os.path.join(videos, name + DLC_SCORER + ".h5")
        _write_tracking(track_file, {"": parts}, False, rng, dropouts)
        if with_videos:
            video = _write_video(os.path.join(videos, name), background, [(parts, (40, 60, 200))], len(nose))
            entries.append((video, False))
        else:
            entries.append((track_file, True))
    _save_session(single, maze_layout(), entries)

    # ---- two animals in an open arena
    multi = os.path.join(parent_folder, "Open arena (two animals)")
    videos = os.path.join(multi, "videos")
    os.makedirs(videos, exist_ok=True)
    out["multi"] = _write_config(multi, multi=True)
    paths = arena_paths()
    parts_by_animal = {animal: body_parts_for(nose) for animal, nose in paths.items()}
    track_file = os.path.join(videos, "arena_01" + DLC_SCORER + "_el.h5")
    _write_tracking(track_file, parts_by_animal, True, rng)
    if with_videos:
        colors = [(40, 60, 200), (200, 110, 30)]
        animals = [(parts_by_animal[a], colors[i]) for i, a in enumerate(parts_by_animal)]
        video = _write_video(os.path.join(videos, "arena_01"), _arena_background(), animals, len(paths["animal1"]))
        entries = [(video, False)]
    else:
        entries = [(track_file, True)]
    _save_session(multi, arena_layout(), entries)
    return out


def _save_session(project_folder: str, layout: Layout, entries: list[tuple[str, bool]]) -> None:
    """Store the example setup and the list of demo videos where the program looks for them."""
    from .session import SESSION_FILE, SESSION_FOLDER

    folder = os.path.join(project_folder, SESSION_FOLDER)
    os.makedirs(folder, exist_ok=True)
    videos = [
        {"path": os.path.relpath(path, project_folder).replace("\\", "/"), "tracks_only": tracks_only}
        for path, tracks_only in entries
    ]
    data = {"version": 1, "videos": videos, "setup": layout.to_dict()}
    with open(os.path.join(folder, SESSION_FILE), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Create the PoseTally demo projects.")
    parser.add_argument("folder", help="where to put them")
    args = parser.parse_args(argv)
    for kind, path in make_demo(args.folder).items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
