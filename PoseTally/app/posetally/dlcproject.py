"""Reading a DeepLabCut project.

Only the project's ``config.yaml`` and its folder layout are read here.
DeepLabCut itself is not imported, so this is fast and works even when
DeepLabCut is not installed (scoring videos that are already tracked).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


class ProjectError(Exception):
    """The chosen file is not a usable DeepLabCut project configuration."""


@dataclass
class TrainedModel:
    engine: str  # "pytorch" or "tensorflow"
    iteration: int
    shuffle: int
    trainset_percent: int
    trainingsetindex: int | None
    folder: str
    snapshots: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.engine}:{self.iteration}:{self.trainset_percent}:{self.shuffle}"

    def label(self) -> str:
        engine = "PyTorch" if self.engine == "pytorch" else "TensorFlow"
        count = len(self.snapshots)
        return (
            f"Shuffle {self.shuffle} · {engine} · training set {self.trainset_percent}% · "
            f"iteration {self.iteration} · {count} snapshot{'s' if count != 1 else ''}"
        )


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "on")
    return bool(value)


def _as_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v) != ""]
    return []


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_yaml(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        text = handle.read()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        # Unusual tags: fall back to reading every value as plain text.
        data = yaml.load(text, Loader=yaml.BaseLoader)  # noqa: S506 (BaseLoader builds only str/list/dict)
    if not isinstance(data, dict):
        raise ProjectError("This file is not a DeepLabCut config.yaml.")
    return data


@dataclass
class DLCProject:
    config_path: str
    folder: str
    task: str = ""
    scorer: str = ""
    date: str = ""
    multi_animal: bool = False
    bodyparts: list[str] = field(default_factory=list)
    unique_bodyparts: list[str] = field(default_factory=list)
    individuals: list[str] = field(default_factory=list)
    skeleton: list[tuple[str, str]] = field(default_factory=list)
    pcutoff: float = 0.6
    iteration: int = 0
    training_fractions: list[float] = field(default_factory=list)
    engine: str | None = None
    models: list[TrainedModel] = field(default_factory=list)

    @staticmethod
    def load(config_path: str) -> DLCProject:
        config_path = os.path.abspath(str(config_path))
        if not os.path.isfile(config_path):
            raise ProjectError(f"File not found: {config_path}")
        try:
            cfg = read_yaml(config_path)
        except ProjectError:
            raise
        except Exception as exc:
            raise ProjectError(f"Could not read {os.path.basename(config_path)}: {exc}") from exc
        if "Task" not in cfg or ("bodyparts" not in cfg and "multianimalbodyparts" not in cfg):
            raise ProjectError(
                "This does not look like a DeepLabCut project file. "
                "Choose the config.yaml that sits at the top of the project folder."
            )

        project = DLCProject(config_path=config_path, folder=os.path.dirname(config_path))
        project.task = str(cfg.get("Task", ""))
        project.scorer = str(cfg.get("scorer", ""))
        project.date = str(cfg.get("date", ""))
        project.multi_animal = _as_bool(cfg.get("multianimalproject", False))
        if project.multi_animal:
            project.bodyparts = _as_list(cfg.get("multianimalbodyparts"))
            project.unique_bodyparts = _as_list(cfg.get("uniquebodyparts"))
            project.individuals = _as_list(cfg.get("individuals"))
        else:
            project.bodyparts = _as_list(cfg.get("bodyparts"))
        skeleton = cfg.get("skeleton")
        if isinstance(skeleton, (list, tuple)):
            for pair in skeleton:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    project.skeleton.append((str(pair[0]), str(pair[1])))
        pcutoff = cfg.get("pcutoff", 0.6)
        if isinstance(pcutoff, (list, tuple, dict)):  # per-body-part cut-offs: take the strictest common one
            values = list(pcutoff.values()) if isinstance(pcutoff, dict) else list(pcutoff)
            numbers = [_as_float(v, 0.6) for v in values] or [0.6]
            project.pcutoff = max(numbers)
        else:
            project.pcutoff = _as_float(pcutoff, 0.6)
        project.pcutoff = min(1.0, max(0.0, project.pcutoff))
        project.iteration = _as_int(cfg.get("iteration", 0), 0)
        fractions = cfg.get("TrainingFraction")
        if isinstance(fractions, (list, tuple)):
            project.training_fractions = [_as_float(f, 0.95) for f in fractions]
        engine = cfg.get("engine")
        if isinstance(engine, str) and engine.strip():
            lowered = engine.strip().lower()
            project.engine = "tensorflow" if lowered in ("tensorflow", "tf") else "pytorch"
        project.models = _find_models(project)
        return project

    # -- body parts -------------------------------------------------------------

    def all_bodyparts(self) -> list[str]:
        out = list(self.bodyparts)
        out += [p for p in self.unique_bodyparts if p not in out]
        return out

    # -- trained models ---------------------------------------------------------

    def default_model(self) -> TrainedModel | None:
        if not self.models:
            return None

        def rank(model: TrainedModel) -> tuple:
            return (
                1 if model.iteration == self.iteration else 0,
                1 if (self.engine or "pytorch") == model.engine else 0,
                model.iteration,
                -model.shuffle,
                1 if model.trainingsetindex == 0 else 0,
            )

        return max(self.models, key=rank)

    def find_model(self, key: str | None) -> TrainedModel | None:
        for model in self.models:
            if model.key == key:
                return model
        return None

    def results_tag(self, shuffle: int) -> str:
        """Text that appears in the name of every results file made by this project and shuffle."""
        return f"{self.task}{self.date}shuffle{shuffle}_"

    def videos_folder(self) -> str | None:
        candidate = os.path.join(self.folder, "videos")
        return candidate if os.path.isdir(candidate) else None

    def describe(self) -> str:
        kind = "multi-animal" if self.multi_animal else "single-animal"
        parts = len(self.all_bodyparts())
        return f"{self.task or 'project'} · {kind} · {parts} body part{'s' if parts != 1 else ''}"


_POSE_SNAPSHOT_PT = re.compile(r"^snapshot(-best)?-\d+\.pt$")


def _find_models(project: DLCProject) -> list[TrainedModel]:
    """Look through the project folder for networks that have been trained."""
    root = Path(project.folder)
    found: list[TrainedModel] = []
    for engine, sub in (("pytorch", "dlc-models-pytorch"), ("tensorflow", "dlc-models")):
        base = root / sub
        if not base.is_dir():
            continue
        try:
            iteration_dirs = sorted(p for p in base.iterdir() if p.is_dir())
        except OSError:
            continue
        for it_dir in iteration_dirs:
            match = re.match(r"^iteration-(\d+)$", it_dir.name)
            if not match:
                continue
            iteration = int(match.group(1))
            try:
                shuffle_dirs = sorted(p for p in it_dir.iterdir() if p.is_dir())
            except OSError:
                continue
            for shuffle_dir in shuffle_dirs:
                match2 = re.search(r"-trainset(\d+)shuffle(\d+)$", shuffle_dir.name)
                train = shuffle_dir / "train"
                if not match2 or not train.is_dir():
                    continue
                try:
                    names = [p.name for p in train.iterdir() if p.is_file()]
                except OSError:
                    continue
                if engine == "pytorch":
                    snapshots = sorted(n for n in names if _POSE_SNAPSHOT_PT.match(n))
                else:
                    snapshots = sorted({n[: -len(".index")] for n in names if n.startswith("snapshot-") and n.endswith(".index")})
                if not snapshots:
                    continue
                percent = int(match2.group(1))
                index = None
                for i, fraction in enumerate(project.training_fractions):
                    if int(fraction * 100) == percent:
                        index = i
                        break
                found.append(
                    TrainedModel(
                        engine=engine,
                        iteration=iteration,
                        shuffle=int(match2.group(2)),
                        trainset_percent=percent,
                        trainingsetindex=index,
                        folder=str(shuffle_dir),
                        snapshots=snapshots,
                    )
                )
    return found
