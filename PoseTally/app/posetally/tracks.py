"""Reading DeepLabCut's results.

For every video DeepLabCut writes a table with one row per frame and, for each
body part (and each animal in multi-animal projects), an x, a y and a
likelihood. The table is stored as ``<video name>DLC_<model details>.h5`` and
optionally a ``.csv`` with the same content. This module finds those files and
turns them into plain arrays.
"""

from __future__ import annotations

import io
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".webm",
    ".flv",
    ".mts",
    ".m2ts",
    ".qt",
)

TRACKER_SUFFIXES = ("_el", "_sk", "_bx", "_ctd", "_tr")
SINGLE_ANIMAL = ""  # name used for "the animal" in single-animal projects
UNIQUE_PARTS_OWNER = "single"  # DeepLabCut's name for parts that belong to no animal


class TrackFormatError(Exception):
    """The file is not a DeepLabCut results table we can read."""


@dataclass
class TrackSet:
    """Tracking for one video."""

    path: str
    scorer: str = ""
    individuals: list[str] = field(default_factory=list)
    bodyparts: dict[str, list[str]] = field(default_factory=dict)  # per individual
    n_frames: int = 0
    multi_animal: bool = False
    data: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)  # (N, 3): x, y, likelihood
    crop_offset: tuple[float, float] = (0.0, 0.0)
    meta: dict = field(default_factory=dict)

    def get(self, individual: str, bodypart: str) -> np.ndarray | None:
        return self.data.get((individual, bodypart))

    def all_bodyparts(self) -> list[str]:
        out: list[str] = []
        for individual in self.individuals:
            for part in self.bodyparts.get(individual, []):
                if part not in out:
                    out.append(part)
        return out

    def animals(self) -> list[str]:
        """Individuals that are animals (not the owner of "unique" parts)."""
        animals = [i for i in self.individuals if i != UNIQUE_PARTS_OWNER]
        return animals or list(self.individuals)


# ------------------------------------------------------------------ finding files


def is_video_file(name: str) -> bool:
    return name.lower().endswith(VIDEO_EXTENSIONS)


def looks_like_dlc_output_video(stem: str) -> bool:
    """Videos DeepLabCut itself creates (labeled copies) are not input videos."""
    if "DLC_" not in stem and "DeepCut_" not in stem:
        return False
    return stem.endswith("labeled") or stem.endswith("_full") or stem.endswith("_id")


def _track_remainder(file_stem: str, video_stem: str) -> str | None:
    """What follows the video name in a results file name, or None if no match."""
    if not file_stem.startswith(video_stem):
        return None
    rest = file_stem[len(video_stem) :]
    if rest.startswith(("DLC", "DeepCut", "_superanimal")):
        return rest
    return None


@dataclass
class TrackCandidate:
    path: str
    filtered: bool
    tracker: str  # "_el", "_sk", ... or ""
    matches_model: bool
    is_h5: bool
    mtime: float

    def sort_key(self, prefer_filtered: bool) -> tuple:
        tracker_rank = {"_el": 5, "_ctd": 4, "_tr": 3, "_sk": 2, "_bx": 1}.get(self.tracker, 0)
        return (
            1 if self.matches_model else 0,
            1 if self.filtered == prefer_filtered else 0,
            tracker_rank,
            1 if self.is_h5 else 0,
            self.mtime,
        )


def describe_track_file(path: str, model_tag: str = "") -> TrackCandidate:
    stem = Path(path).stem
    filtered = "_filtered" in stem
    plain = stem.replace("_filtered", "")
    tracker = next((s for s in TRACKER_SUFFIXES if plain.endswith(s)), "")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return TrackCandidate(
        path=str(path),
        filtered=filtered,
        tracker=tracker,
        matches_model=bool(model_tag) and model_tag in stem,
        is_h5=str(path).lower().endswith(".h5"),
        mtime=mtime,
    )


_NAME_MARKERS = ("DLC", "DeepCut", "_superanimal")


def _possible_video_stems(file_stem: str) -> list[str]:
    """Every video name a results file could belong to (the text before each marker)."""
    out: list[str] = []
    for marker in _NAME_MARKERS:
        at = file_stem.find(marker)
        while at > 0:
            head = file_stem[:at]
            if head not in out:
                out.append(head)
            at = file_stem.find(marker, at + 1)
    return out


class TrackIndex:
    """Looks up result files for videos without listing folders again and again."""

    def __init__(self) -> None:
        # folder -> {video name -> [(file name without extension, path)]}
        self._folders: dict[str, dict[str, list[tuple[str, str]]]] = {}

    def forget(self, folder: str | None = None) -> None:
        if folder is None:
            self._folders.clear()
        else:
            self._folders.pop(os.path.abspath(folder), None)

    def _listing(self, folder: str) -> dict[str, list[tuple[str, str]]]:
        folder = os.path.abspath(folder)
        if folder not in self._folders:
            by_video: dict[str, list[tuple[str, str]]] = {}
            try:
                with os.scandir(folder) as it:
                    for entry in it:
                        lower = entry.name.lower()
                        if not (lower.endswith(".h5") or lower.endswith(".csv")):
                            continue
                        stem = entry.name[: entry.name.rfind(".")]
                        if stem.endswith("_skeleton"):
                            continue
                        for video_stem in _possible_video_stems(stem):
                            by_video.setdefault(video_stem, []).append((stem, entry.path))
            except OSError:
                pass
            self._folders[folder] = by_video
        return self._folders[folder]

    def candidates(self, video_stem: str, folders: list[str], model_tag: str = "") -> list[TrackCandidate]:
        found: dict[str, TrackCandidate] = {}
        for folder in folders:
            if not folder:
                continue
            for stem, path in self._listing(folder).get(video_stem, []):
                if _track_remainder(stem, video_stem) is None:
                    continue
                found[path] = describe_track_file(path, model_tag)
        return list(found.values())

    def orphan_tracks(self, folder: str, video_stems: set[str]) -> dict[str, list[str]]:
        """Result files in a folder whose video is not there, grouped by video name."""
        out: dict[str, list[str]] = {}
        listing = self._listing(folder)
        claimed = {path for stem in video_stems for _, path in listing.get(stem, [])}
        for video_stem, entries in listing.items():
            for file_stem, path in entries:
                if path in claimed:
                    continue
                # use the shortest reading of the name: the text before the first marker
                if _possible_video_stems(file_stem)[0] != video_stem:
                    continue
                if path not in out.setdefault(video_stem, []):
                    out[video_stem].append(path)
        return {k: v for k, v in out.items() if v}


def pick_best(candidates: list[TrackCandidate], prefer_filtered: bool = False) -> TrackCandidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.sort_key(prefer_filtered))


def model_tag(task: str, date: str, shuffle: int) -> str:
    """The piece of a results file name that identifies project and shuffle."""
    return f"{task}{date}shuffle{shuffle}_"


# ----------------------------------------------------------------- reading tables


def _read_table(path: str):
    import pandas as pd

    lower = path.lower()
    if lower.endswith(".h5"):
        try:
            return pd.read_hdf(path)
        except (ValueError, KeyError):
            for key in ("df_with_missing", "tracks", "df"):
                try:
                    return pd.read_hdf(path, key=key)
                except (KeyError, ValueError):
                    continue
            raise
    if lower.endswith(".csv"):
        with open(path, encoding="utf-8-sig", errors="replace") as handle:
            first = [handle.readline() for _ in range(4)]
        labels = [line.split(",")[0].strip().lower() for line in first]
        if labels[:1] != ["scorer"]:
            raise TrackFormatError("This CSV file is not a DeepLabCut results table.")
        header = [0, 1, 2, 3] if labels[1] == "individuals" else [0, 1, 2]
        return pd.read_csv(path, header=header, index_col=0)
    raise TrackFormatError("Tracking files must end in .h5 or .csv.")


def load_tracks(path: str) -> TrackSet:
    """Read a DeepLabCut results file (.h5 or .csv)."""
    import pandas as pd

    path = str(path)
    try:
        df = _read_table(path)
    except TrackFormatError:
        raise
    except ImportError as exc:
        raise TrackFormatError(
            "Reading .h5 files needs the 'tables' package, which is missing. Run the installer again."
        ) from exc
    except Exception as exc:  # corrupt or unrelated file
        raise TrackFormatError(f"Could not read {os.path.basename(path)}: {exc}") from exc

    if not isinstance(df, pd.DataFrame) or not isinstance(df.columns, pd.MultiIndex):
        raise TrackFormatError(f"{os.path.basename(path)} is not a DeepLabCut results table.")
    names = [str(n) if n is not None else "" for n in df.columns.names]
    nlevels = df.columns.nlevels
    if "bodyparts" in names and "coords" in names:
        i_body, i_coord = names.index("bodyparts"), names.index("coords")
        i_ind = names.index("individuals") if "individuals" in names else None
        i_scorer = names.index("scorer") if "scorer" in names else None
    elif nlevels == 3:
        i_scorer, i_ind, i_body, i_coord = 0, None, 1, 2
    elif nlevels == 4:
        i_scorer, i_ind, i_body, i_coord = 0, 1, 2, 3
    else:
        raise TrackFormatError(f"{os.path.basename(path)} is not a DeepLabCut results table.")

    index = df.index
    if isinstance(index, pd.MultiIndex) or not pd.api.types.is_numeric_dtype(index):
        raise TrackFormatError(
            f"{os.path.basename(path)} holds labeled frames, not the tracking of a whole video."
        )
    frames = np.asarray(index, dtype=np.int64)
    values = df.to_numpy(dtype=float, na_value=np.nan)
    values[~np.isfinite(values)] = np.nan
    if len(frames) and (frames[0] != 0 or np.any(np.diff(frames) != 1)):
        frames = np.maximum(frames, 0)
        full = np.full((int(frames.max()) + 1, values.shape[1]), np.nan)
        full[frames] = values
        values = full
    n_frames = values.shape[0]

    tracks = TrackSet(path=path, n_frames=n_frames, multi_animal=i_ind is not None)
    columns: dict[tuple[str, str], dict[str, int]] = {}
    for position, key in enumerate(df.columns):
        individual = str(key[i_ind]) if i_ind is not None else SINGLE_ANIMAL
        bodypart = str(key[i_body])
        coord = str(key[i_coord]).lower()
        columns.setdefault((individual, bodypart), {})[coord] = position
        if i_scorer is not None and not tracks.scorer:
            tracks.scorer = str(key[i_scorer])
    for (individual, bodypart), coords in columns.items():
        if "x" not in coords or "y" not in coords:
            continue
        arr = np.empty((n_frames, 3))
        arr[:, 0] = values[:, coords["x"]]
        arr[:, 1] = values[:, coords["y"]]
        arr[:, 2] = values[:, coords["likelihood"]] if "likelihood" in coords else 1.0
        if individual not in tracks.individuals:
            tracks.individuals.append(individual)
            tracks.bodyparts[individual] = []
        tracks.bodyparts[individual].append(bodypart)
        tracks.data[(individual, bodypart)] = arr
    if not tracks.data:
        raise TrackFormatError(f"{os.path.basename(path)} has no x/y columns.")

    tracks.meta = load_meta_for(path)
    offset = crop_offset_from_meta(tracks.meta)
    if offset != (0.0, 0.0):
        # DeepLabCut stores positions relative to the crop; move them back onto the full frame.
        for arr in tracks.data.values():
            arr[:, 0] += offset[0]
            arr[:, 1] += offset[1]
        tracks.crop_offset = offset
    return tracks


def tracks_to_dlc_csv(tracks: TrackSet, out_path: str) -> None:
    """Write tracking in DeepLabCut's CSV layout (used for hand-off to other tools)."""
    import pandas as pd

    scorer = tracks.scorer or "DLC"
    tuples, arrays = [], []
    for individual in tracks.individuals:
        for bodypart in tracks.bodyparts[individual]:
            arr = tracks.data[(individual, bodypart)].copy()
            arr[:, 0] -= tracks.crop_offset[0]
            arr[:, 1] -= tracks.crop_offset[1]
            for column, coord in enumerate(("x", "y", "likelihood")):
                if tracks.multi_animal:
                    tuples.append((scorer, individual, bodypart, coord))
                else:
                    tuples.append((scorer, bodypart, coord))
                arrays.append(arr[:, column])
    names = ["scorer", "individuals", "bodyparts", "coords"] if tracks.multi_animal else ["scorer", "bodyparts", "coords"]
    df = pd.DataFrame(np.column_stack(arrays), columns=pd.MultiIndex.from_tuples(tuples, names=names))
    df.to_csv(out_path)


# ----------------------------------------------------------------- run information


class _Placeholder(dict):
    """Stands in for any object in a pickle that we do not need to understand."""

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __setstate__(self, state):
        pass

    def __hash__(self):
        return id(self)

    def __call__(self, *args, **kwargs):
        return _Placeholder()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Placeholder()


_SAFE_BUILTINS = {
    "dict", "list", "tuple", "set", "frozenset", "str", "bytes", "bytearray",
    "int", "float", "bool", "complex", "slice", "range", "NoneType",
}  # fmt: skip


class _TolerantUnpickler(pickle.Unpickler):
    """Reads plain data from a pickle and never runs code from it."""

    def find_class(self, module, name):
        if module == "builtins" and name in _SAFE_BUILTINS:
            return super().find_class(module, name)
        if module == "collections" and name == "OrderedDict":
            return super().find_class(module, name)
        root = module.split(".")[0]
        if root == "numpy" and name in ("ndarray", "dtype", "_reconstruct", "scalar", "_frombuffer"):
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                return _Placeholder
        return _Placeholder


def read_pickle_safely(path: str) -> dict:
    try:
        with open(path, "rb") as handle:
            data = _TolerantUnpickler(io.BytesIO(handle.read())).load()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def meta_path_for(track_path: str) -> str | None:
    """The ``..._meta.pickle`` DeepLabCut wrote next to a results file."""
    folder, name = os.path.split(track_path)
    stem = name[: name.rfind(".")] if "." in name else name
    tried = set()
    for _ in range(4):
        if stem in tried:
            break
        tried.add(stem)
        candidate = os.path.join(folder, stem + "_meta.pickle")
        if os.path.isfile(candidate):
            return candidate
        shorter = re.sub(r"_filtered$", "", stem)
        if shorter == stem:
            shorter = re.sub("(" + "|".join(TRACKER_SUFFIXES) + ")$", "", stem)
        stem = shorter
    return None


def load_meta_for(track_path: str) -> dict:
    """Frame rate, frame count and crop that DeepLabCut recorded, when available."""
    path = meta_path_for(track_path)
    if not path:
        return {}
    raw = read_pickle_safely(path).get("data")
    if not isinstance(raw, dict):
        return {}
    out: dict = {}

    def number(key):
        try:
            value = float(raw.get(key))
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    fps = number("fps")
    if fps and fps > 0:
        out["fps"] = fps
    nframes = number("nframes")
    if nframes and nframes > 0:
        out["nframes"] = int(nframes)
    out["cropping"] = bool(raw.get("cropping")) if isinstance(raw.get("cropping"), (bool, int, np.bool_)) else False
    params = raw.get("cropping_parameters")
    try:
        params = [float(v) for v in params]
        if len(params) == 4:
            out["cropping_parameters"] = params  # x1, x2, y1, y2
    except (TypeError, ValueError):
        pass
    dims = raw.get("frame_dimensions")
    try:
        d0, d1 = float(dims[0]), float(dims[1])
        # The PyTorch engine stores (width, height); the TensorFlow engine (height, width).
        if "pytorch-config" in raw:
            out["frame_size"] = (int(d0), int(d1))
        else:
            out["frame_size"] = (int(d1), int(d0))
    except (TypeError, ValueError, IndexError):
        pass
    scorer = raw.get("Scorer")
    if isinstance(scorer, str):
        out["scorer"] = scorer
    out["engine"] = "pytorch" if "pytorch-config" in raw else "tensorflow"
    return out


def crop_offset_from_meta(meta: dict) -> tuple[float, float]:
    if not meta or not meta.get("cropping"):
        return (0.0, 0.0)
    params = meta.get("cropping_parameters")
    if not params or len(params) != 4:
        return (0.0, 0.0)
    return (float(params[0]), float(params[2]))
