"""The working state: which project is open, which videos are in the batch,
the scoring setup, and the latest results. No user-interface code lives here,
so everything can be driven from a script or a test.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import OrderedDict
from collections.abc import Callable

from .batch import ResultsTable, VideoItem, VideoResult, build_table, natural_key, relative_or_absolute, score_item
from .dlcproject import DLCProject, TrainedModel
from .model import Layout
from .tracks import (
    TrackIndex,
    TrackSet,
    describe_track_file,
    is_video_file,
    load_tracks,
    looks_like_dlc_output_video,
    pick_best,
)

SESSION_FOLDER = "PoseTally"
SESSION_FILE = "session.json"
SESSION_VERSION = 1

DEFAULT_TRACK_OPTIONS = {
    "device": "auto",  # auto | cuda | mps | cpu
    "save_csv": False,
    "n_animals": 0,  # 0 = as in the project
    "batch_size": 0,  # 0 = as in the project
    "retrack_other_models": False,
}


def user_data_folder() -> str:
    """A per-user folder for things that cannot be stored next to the project."""
    home = os.path.expanduser("~")
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        return os.path.join(base, "PoseTally")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Application Support", "PoseTally")
    return os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share"), "PoseTally")


class Session:
    def __init__(self) -> None:
        self.project: DLCProject | None = None
        self.items: list[VideoItem] = []
        self.layout: Layout = Layout()
        self.model_key: str | None = None
        self.results_folder: str | None = None  # extra folder where tracking files are kept
        self.track_options: dict = dict(DEFAULT_TRACK_OPTIONS)
        self.results: dict[str, VideoResult] = {}
        self.results_signature: str | None = None
        self.notices: list[str] = []
        self._index = TrackIndex()
        self._track_cache: OrderedDict[tuple[str, float], TrackSet] = OrderedDict()
        self._known_bodyparts: list[str] = []

    # ------------------------------------------------------------------ project

    def open_project(self, config_path: str) -> None:
        """Open a DeepLabCut project and restore the setup saved for it, if any."""
        project = DLCProject.load(config_path)
        self.project = project
        self.items = []
        self.layout = Layout()
        self.layout.settings.pcutoff = project.pcutoff
        self.model_key = None
        self.results_folder = None
        self.track_options = dict(DEFAULT_TRACK_OPTIONS)
        self.results = {}
        self.results_signature = None
        self.notices = []
        self._index.forget()
        self._track_cache.clear()
        self._known_bodyparts = []
        self._load_saved()
        if self.model() is None:
            default = project.default_model()
            self.model_key = default.key if default else None

    def model(self) -> TrainedModel | None:
        if not self.project:
            return None
        return self.project.find_model(self.model_key)

    def model_tag(self) -> str:
        model = self.model()
        if not self.project or model is None:
            return ""
        return self.project.results_tag(model.shuffle)

    def bodyparts(self) -> list[str]:
        """Body parts that rules can use: the project's, plus any seen in tracking files."""
        out = list(self.project.all_bodyparts()) if self.project else []
        out += [p for p in self._known_bodyparts if p not in out]
        return out

    # -------------------------------------------------------------------- batch

    def _search_folders(self, item: VideoItem) -> list[str]:
        folders = []
        if item.video_path:
            folders.append(os.path.dirname(item.video_path))
        elif item.track_path:
            folders.append(os.path.dirname(item.track_path))
        if self.results_folder:
            folders.append(self.results_folder)
        return folders

    def _resolve_track(self, item: VideoItem) -> None:
        if item.manual_track and item.track_path and os.path.isfile(item.track_path):
            return
        item.manual_track = False
        candidates = self._index.candidates(item.name, self._search_folders(item), self.model_tag())
        best = pick_best(candidates, self.layout.settings.prefer_filtered)
        item.track_path = best.path if best else None
        item.track_matches_model = bool(best and best.matches_model)

    def refresh_tracking(self) -> None:
        """Look again for tracking files (after tracking ran or files changed)."""
        self._index.forget()
        self._track_cache.clear()
        for item in self.items:
            self._resolve_track(item)

    def add_videos(self, paths: list[str]) -> int:
        existing = {item.key for item in self.items}
        added = 0
        for path in paths:
            path = os.path.abspath(path)
            name = os.path.splitext(os.path.basename(path))[0]
            if path in existing or not os.path.isfile(path):
                continue
            if not is_video_file(path) or looks_like_dlc_output_video(name):
                continue
            item = VideoItem(key=path, name=name, video_path=path)
            self._resolve_track(item)
            self.items.append(item)
            existing.add(path)
            added += 1
        self._sort_items()
        return added

    def add_folder(self, folder: str, recursive: bool = True) -> tuple[int, int]:
        """Add every video in a folder. Returns (videos added, tracking-only entries added)."""
        folder = os.path.abspath(folder)
        videos: list[str] = []
        folders_seen: list[str] = []
        walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder) if os.path.isdir(folder) else [])]
        for root, dirs, files in walker:
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in (SESSION_FOLDER, "labeled-data", "training-datasets", "dlc-models", "dlc-models-pytorch", "evaluation-results", "evaluation-results-pytorch")]
            folders_seen.append(root)
            for name in files:
                if name.startswith("."):
                    continue
                if is_video_file(name) and not looks_like_dlc_output_video(os.path.splitext(name)[0]):
                    videos.append(os.path.join(root, name))
        added = self.add_videos(videos)

        # tracking files whose video is not in the folder (videos archived elsewhere)
        existing = {item.key for item in self.items}
        orphans = 0
        for root in folders_seen:
            stems = {os.path.splitext(f)[0] for f in os.listdir(root) if is_video_file(f)} if os.path.isdir(root) else set()
            for stem, paths in self._index.orphan_tracks(root, stems).items():
                key = "tracks:" + os.path.join(root, stem)
                if key in existing:
                    continue
                best = pick_best([describe_track_file(p, self.model_tag()) for p in paths], self.layout.settings.prefer_filtered)
                if not best:
                    continue
                item = VideoItem(key=key, name=stem, video_path=None, track_path=best.path, track_matches_model=best.matches_model)
                self.items.append(item)
                existing.add(key)
                orphans += 1
        self._sort_items()
        return added, orphans

    def remove_items(self, keys: list[str]) -> None:
        drop = set(keys)
        self.items = [item for item in self.items if item.key not in drop]
        for key in drop:
            self.results.pop(key, None)

    def _sort_items(self) -> None:
        self.items.sort(key=lambda item: (natural_key(item.name), item.key))

    def find_item(self, key: str | None) -> VideoItem | None:
        for item in self.items:
            if item.key == key:
                return item
        return None

    def set_manual_track(self, item: VideoItem, track_path: str | None) -> None:
        if track_path:
            item.track_path = os.path.abspath(track_path)
            item.manual_track = True
            item.track_matches_model = bool(self.model_tag()) and self.model_tag() in os.path.basename(track_path)
        else:
            item.manual_track = False
            self._resolve_track(item)
        self.results.pop(item.key, None)

    def items_to_track(self) -> list[VideoItem]:
        """Videos that still need DeepLabCut to run on them."""
        redo_others = bool(self.track_options.get("retrack_other_models"))
        out = []
        for item in self.items:
            if not item.video_path:
                continue
            if not item.track_path or (redo_others and not item.track_matches_model and not item.manual_track):
                out.append(item)
        return out

    # ------------------------------------------------------------------ scoring

    def tracks_for(self, item: VideoItem) -> TrackSet | None:
        """Tracking for an item, kept in memory for a few recently used videos."""
        if not item.track_path:
            return None
        try:
            stamp = os.path.getmtime(item.track_path)
        except OSError:
            return None
        key = (item.track_path, stamp)
        if key in self._track_cache:
            self._track_cache.move_to_end(key)
            return self._track_cache[key]
        tracks = load_tracks(item.track_path)
        self._remember_bodyparts(tracks)
        self._track_cache[key] = tracks
        while len(self._track_cache) > 12:
            self._track_cache.popitem(last=False)
        return tracks

    def _remember_bodyparts(self, tracks: TrackSet) -> None:
        for part in tracks.all_bodyparts():
            if part not in self._known_bodyparts:
                self._known_bodyparts.append(part)

    def layout_signature(self) -> str:
        payload = {
            "setup": self.layout.to_dict(),
            "tracks": [(item.key, item.track_path) for item in self.items],
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def results_are_current(self) -> bool:
        return bool(self.results) and self.results_signature == self.layout_signature()

    def score_all(
        self,
        progress: Callable[[int, int, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        """Score every video in the batch. Returns False if it was cancelled."""
        layout = self.layout.copy()
        signature = self.layout_signature()
        results: dict[str, VideoResult] = {}
        total = len(self.items)
        for number, item in enumerate(self.items, start=1):
            if cancelled and cancelled():
                return False
            if progress:
                progress(number, total, item.name)
            result = score_item(item, layout)
            results[item.key] = result
        self.results = results
        self.results_signature = signature
        return True

    def ordered_results(self) -> list[VideoResult]:
        return [self.results[item.key] for item in self.items if item.key in self.results]

    def table(self) -> ResultsTable:
        return build_table(self.ordered_results(), self.layout)

    # ------------------------------------------------------------------- saving

    def _session_paths(self) -> list[str]:
        if not self.project:
            return []
        digest = hashlib.sha1(self.project.config_path.encode("utf-8")).hexdigest()[:16]
        return [
            os.path.join(self.project.folder, SESSION_FOLDER, SESSION_FILE),
            os.path.join(user_data_folder(), "sessions", digest + ".json"),
        ]

    def to_dict(self) -> dict:
        base = self.project.folder if self.project else ""
        videos = []
        manual = {}
        for item in self.items:
            source = item.video_path or item.track_path
            if not source:
                continue
            entry = relative_or_absolute(source, base).replace("\\", "/") if base else source
            videos.append({"path": entry, "tracks_only": item.video_path is None})
            if item.manual_track and item.track_path:
                manual[entry] = relative_or_absolute(item.track_path, base).replace("\\", "/") if base else item.track_path
        results_folder = self.results_folder
        if results_folder and base:
            results_folder = relative_or_absolute(results_folder, base).replace("\\", "/")
        return {
            "version": SESSION_VERSION,
            "videos": videos,
            "manual_tracking_files": manual,
            "results_folder": results_folder,
            "model": self.model_key,
            "track_options": dict(self.track_options),
            "setup": self.layout.to_dict(),
        }

    def save(self) -> str | None:
        """Save next to the project, or in the user's data folder if that is not writable."""
        data = json.dumps(self.to_dict(), indent=1)
        for path in self._session_paths():
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                temp = path + ".tmp"
                with open(temp, "w", encoding="utf-8") as handle:
                    handle.write(data)
                os.replace(temp, path)
                return path
            except OSError:
                continue
        return None

    def _load_saved(self) -> None:
        candidates = [p for p in self._session_paths() if os.path.isfile(p)]
        if not candidates:
            return
        path = max(candidates, key=os.path.getmtime)
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            self.notices.append("The saved setup for this project could not be read and was ignored.")
            return
        if not isinstance(data, dict):
            return
        base = self.project.folder if self.project else ""

        def absolute(entry: str) -> str:
            entry = str(entry)
            return entry if os.path.isabs(entry) else os.path.normpath(os.path.join(base, entry))

        self.layout = Layout.from_dict(data.get("setup"))
        options = data.get("track_options")
        if isinstance(options, dict):
            for key, default in DEFAULT_TRACK_OPTIONS.items():
                if key in options and isinstance(options[key], type(default)):
                    self.track_options[key] = options[key]
        folder = data.get("results_folder")
        if folder:
            folder = absolute(folder)
            self.results_folder = folder if os.path.isdir(folder) else None
        model_key = data.get("model")
        if isinstance(model_key, str) and self.project and self.project.find_model(model_key):
            self.model_key = model_key

        manual = data.get("manual_tracking_files") or {}
        missing = 0
        videos, track_only = [], []
        for entry in data.get("videos", []):
            if isinstance(entry, str):
                entry = {"path": entry, "tracks_only": False}
            if not isinstance(entry, dict) or "path" not in entry:
                continue
            full = absolute(entry["path"])
            if not os.path.isfile(full):
                missing += 1
                continue
            (track_only if entry.get("tracks_only") else videos).append((entry["path"], full))
        self.add_videos([full for _, full in videos])
        for _, full in track_only:
            stem = os.path.basename(full)
            cut = max(stem.find("DLC_"), stem.find("DeepCut_"))
            if cut <= 0:
                continue
            key = "tracks:" + os.path.join(os.path.dirname(full), stem[:cut])
            if self.find_item(key) is None:
                self.items.append(VideoItem(key=key, name=stem[:cut], track_path=full))
        for original, full in videos:
            chosen = manual.get(original) if isinstance(manual, dict) else None
            if chosen and os.path.isfile(absolute(chosen)):
                item = self.find_item(os.path.abspath(full))
                if item:
                    self.set_manual_track(item, absolute(chosen))
        self._sort_items()
        if missing:
            self.notices.append(
                f"{missing} video{'s' if missing != 1 else ''} from the saved list could not be found "
                "and were left out. Add the folder again if they have moved."
            )

    # -------------------------------------------------------- sharing the setup

    def export_setup(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"posetally_setup": self.layout.to_dict()}, handle, indent=1)

    def import_setup(self, path: str, keep_positions_per_video: bool = False) -> None:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "posetally_setup" in data:
            data = data["posetally_setup"]
        elif isinstance(data, dict) and "setup" in data:
            data = data["setup"]
        if not isinstance(data, dict) or ("lines" not in data and "regions" not in data):
            raise ValueError("This file does not contain a PoseTally scoring setup.")
        layout = Layout.from_dict(data)
        if not keep_positions_per_video:
            layout.overrides = {}
        self.layout = layout
        self.results = {}
        self.results_signature = None
