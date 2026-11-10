import os
import pickle

import numpy as np
import pandas as pd
import pytest

from posetally.dlcproject import DLCProject, ProjectError
from posetally.tracks import (
    TrackFormatError,
    TrackIndex,
    load_tracks,
    looks_like_dlc_output_video,
    meta_path_for,
    pick_best,
    read_pickle_safely,
    tracks_to_dlc_csv,
)

SCORER = "DLC_Resnet50_TaskMay1shuffle1_snapshot_200"


def write_dlc_table(path, animals, parts, n=20, fmt="h5"):
    """Write a table exactly as DeepLabCut does. animals=None makes a single-animal file."""
    rng = np.random.default_rng(0)
    cols, arrays = [], []
    for animal in animals or [None]:
        for part in parts:
            for coord in ("x", "y", "likelihood"):
                cols.append((SCORER, part, coord) if animal is None else (SCORER, animal, part, coord))
                arrays.append(rng.uniform(0, 1, n) if coord == "likelihood" else rng.uniform(0, 500, n))
    names = ["scorer", "bodyparts", "coords"] if animals is None else ["scorer", "individuals", "bodyparts", "coords"]
    df = pd.DataFrame(np.column_stack(arrays), columns=pd.MultiIndex.from_tuples(cols, names=names), index=range(n))
    if fmt == "h5":
        df.to_hdf(path, key="df_with_missing", format="table", mode="w")
    else:
        df.to_csv(path)
    return df


def test_read_single_animal_h5_and_csv_give_the_same_numbers(tmp_path):
    df = write_dlc_table(tmp_path / f"vid{SCORER}.h5", None, ["snout", "tail base"])
    write_dlc_table(tmp_path / f"vid{SCORER}.csv", None, ["snout", "tail base"], fmt="csv")
    h5 = load_tracks(str(tmp_path / f"vid{SCORER}.h5"))
    csv = load_tracks(str(tmp_path / f"vid{SCORER}.csv"))
    for tracks in (h5, csv):
        assert tracks.multi_animal is False and tracks.individuals == [""]
        assert tracks.all_bodyparts() == ["snout", "tail base"]
        assert tracks.n_frames == 20 and tracks.scorer == SCORER
        np.testing.assert_allclose(tracks.get("", "snout")[:, 0], df[SCORER]["snout"]["x"].to_numpy())
        np.testing.assert_allclose(tracks.get("", "tail base")[:, 2], df[SCORER]["tail base"]["likelihood"].to_numpy())


def test_read_multi_animal_with_unique_parts(tmp_path):
    path = tmp_path / f"vid{SCORER}_el.h5"
    df = write_dlc_table(path, ["fish1", "fish2"], ["snout", "tail"])
    extra = pd.DataFrame(
        np.ones((20, 3)),
        columns=pd.MultiIndex.from_product([[SCORER], ["single"], ["corner"], ["x", "y", "likelihood"]], names=df.columns.names),
    )
    df.join(extra).to_hdf(path, key="df_with_missing", format="table", mode="w")
    for target in (path, tmp_path / "copy.csv"):
        if str(target).endswith(".csv"):
            df.join(extra).to_csv(target)
        tracks = load_tracks(str(target))
        assert tracks.multi_animal is True
        assert tracks.individuals == ["fish1", "fish2", "single"]
        assert tracks.animals() == ["fish1", "fish2"]
        assert tracks.bodyparts["single"] == ["corner"]
        np.testing.assert_allclose(tracks.get("fish2", "tail")[:, 1], df[SCORER]["fish2"]["tail"]["y"].to_numpy())


def test_cropped_analysis_is_moved_back_onto_the_full_frame(tmp_path):
    path = tmp_path / f"vid{SCORER}.h5"
    df = write_dlc_table(path, None, ["snout"])
    meta = {"data": {"fps": 25.0, "nframes": 20, "cropping": True, "cropping_parameters": [100, 400, 50, 300],
                     "frame_dimensions": (300, 250), "pytorch-config": {}, "Scorer": SCORER}}  # fmt: skip
    with open(tmp_path / f"vid{SCORER}_meta.pickle", "wb") as handle:
        pickle.dump(meta, handle)
    tracks = load_tracks(str(path))
    assert tracks.crop_offset == (100.0, 50.0)
    np.testing.assert_allclose(tracks.get("", "snout")[:, 0], df[SCORER]["snout"]["x"].to_numpy() + 100)
    np.testing.assert_allclose(tracks.get("", "snout")[:, 1], df[SCORER]["snout"]["y"].to_numpy() + 50)
    assert tracks.meta["fps"] == 25.0 and tracks.meta["frame_size"] == (300, 250)
    # the hand-off CSV goes back to DeepLabCut's own (cropped) numbers
    out = tmp_path / "handoff.csv"
    tracks_to_dlc_csv(tracks, str(out))
    again = pd.read_csv(out, header=[0, 1, 2], index_col=0)
    np.testing.assert_allclose(again[SCORER]["snout"]["x"].to_numpy(), df[SCORER]["snout"]["x"].to_numpy())


def test_meta_file_is_found_for_tracker_and_filtered_names(tmp_path):
    meta = tmp_path / f"vid{SCORER}_meta.pickle"
    meta.write_bytes(pickle.dumps({"data": {"fps": 30}}))
    for name in (f"vid{SCORER}.h5", f"vid{SCORER}_el.h5", f"vid{SCORER}_el_filtered.h5", f"vid{SCORER}_filtered.csv"):
        assert meta_path_for(str(tmp_path / name)) == str(meta)
    assert meta_path_for(str(tmp_path / "other.h5")) is None


class _Boom:
    def __reduce__(self):
        return (os.system, ("echo should-never-run > " + _Boom.marker,))


def test_reading_a_pickle_never_runs_code_from_it(tmp_path):
    _Boom.marker = str(tmp_path / "ran.txt")
    path = tmp_path / "evil_meta.pickle"
    path.write_bytes(pickle.dumps({"data": {"fps": 30.0, "payload": _Boom()}}))
    data = read_pickle_safely(str(path))
    assert data["data"]["fps"] == 30.0
    assert not os.path.exists(_Boom.marker)


def test_numpy_values_in_meta_are_understood(tmp_path):
    path = tmp_path / f"vid{SCORER}.h5"
    write_dlc_table(path, None, ["snout"])
    meta = {"data": {"fps": np.float64(59.94), "nframes": np.int64(20), "cropping": False,
                     "cropping_parameters": [0, 640, 0, 480], "frame_dimensions": (480, 640),
                     "DLC-model-config file": {"anything": np.arange(3)}}}  # fmt: skip
    with open(tmp_path / f"vid{SCORER}_meta.pickle", "wb") as handle:
        pickle.dump(meta, handle)
    tracks = load_tracks(str(path))
    assert tracks.meta["fps"] == pytest.approx(59.94)
    assert tracks.meta["frame_size"] == (640, 480)  # TensorFlow-era files store (height, width)
    assert tracks.meta["engine"] == "tensorflow"


def test_wrong_files_are_refused_politely(tmp_path):
    plain = tmp_path / "notes.csv"
    plain.write_text("a,b\n1,2\n")
    with pytest.raises(TrackFormatError):
        load_tracks(str(plain))
    labeled = pd.DataFrame(
        np.ones((2, 3)),
        index=["labeled-data/v/img0.png", "labeled-data/v/img1.png"],
        columns=pd.MultiIndex.from_product([["me"], ["snout"], ["x", "y", "likelihood"]], names=["scorer", "bodyparts", "coords"]),
    )
    labeled.to_hdf(tmp_path / "CollectedData_me.h5", key="df_with_missing", mode="w")
    with pytest.raises(TrackFormatError):
        load_tracks(str(tmp_path / "CollectedData_me.h5"))
    broken = tmp_path / "broken.h5"
    broken.write_bytes(b"not an hdf5 file")
    with pytest.raises(TrackFormatError):
        load_tracks(str(broken))


def test_finding_the_right_file_for_each_video(tmp_path):
    names = [
        f"trial1{SCORER}.h5",
        f"trial1{SCORER}.csv",
        f"trial1{SCORER}_filtered.h5",
        f"trial10{SCORER}.h5",
        f"trial1{SCORER}_skeleton.h5",
        "trial1DLC_resnet50_OtherJan2shuffle3_50000.h5",
        "unrelated.h5",
        f"archived{SCORER}_el.h5",
        f"archived{SCORER}_sk.h5",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"")
    index = TrackIndex()
    tag = "TaskMay1shuffle1_"
    found = index.candidates("trial1", [str(tmp_path)], tag)
    assert sorted(os.path.basename(c.path) for c in found) == sorted(
        [f"trial1{SCORER}.h5", f"trial1{SCORER}.csv", f"trial1{SCORER}_filtered.h5", "trial1DLC_resnet50_OtherJan2shuffle3_50000.h5"]
    )
    assert os.path.basename(pick_best(found).path) == f"trial1{SCORER}.h5"
    assert os.path.basename(pick_best(found, prefer_filtered=True).path) == f"trial1{SCORER}_filtered.h5"
    # with another model selected, that model's file wins
    other = index.candidates("trial1", [str(tmp_path)], "OtherJan2shuffle3_")
    assert os.path.basename(pick_best(other).path) == "trial1DLC_resnet50_OtherJan2shuffle3_50000.h5"
    assert [os.path.basename(c.path) for c in index.candidates("trial10", [str(tmp_path)], tag)] == [f"trial10{SCORER}.h5"]
    assert index.candidates("trial", [str(tmp_path)], tag) == []
    # ellipse-tracked file is preferred for multi-animal results
    assert os.path.basename(pick_best(index.candidates("archived", [str(tmp_path)], tag)).path) == f"archived{SCORER}_el.h5"
    orphans = index.orphan_tracks(str(tmp_path), {"trial1", "trial10"})
    assert list(orphans) == ["archived"]


def test_labeled_videos_made_by_deeplabcut_are_not_inputs():
    assert looks_like_dlc_output_video(f"trial1{SCORER}_labeled")
    assert looks_like_dlc_output_video(f"trial1{SCORER}_filtered_labeled")
    assert looks_like_dlc_output_video(f"trial1{SCORER}_p60_labeled")
    assert not looks_like_dlc_output_video("trial1")
    assert not looks_like_dlc_output_video("my_labeled_fish")


CONFIG_SINGLE = """
Task: Maze
scorer: lab
date: May1
multianimalproject: false
project_path: /somewhere/else/that/no/longer/exists
video_sets:
  ? C:\\Users\\lab\\Videos\\a very long path that makes the yaml writer use a question mark key\\video one.mp4
  : crop: 0, 640, 0, 480
bodyparts:
- snout
- left eye
- right eye
- head-body junction
- tail base
skeleton:
- - snout
  - head-body junction
pcutoff: 0.7
TrainingFraction:
- 0.95
iteration: 0
snapshotindex: -1
"""

CONFIG_MULTI = """
Task: Shoal
scorer: lab
date: Jun2
multianimalproject: true
engine: pytorch
individuals: [fish1, fish2, fish3]
uniquebodyparts: [feeder]
multianimalbodyparts: [snout, tail]
bodyparts: MULTI!
TrainingFraction: [0.8]
iteration: 2
"""


def test_project_config_single(tmp_path):
    (tmp_path / "config.yaml").write_text(CONFIG_SINGLE)
    project = DLCProject.load(str(tmp_path / "config.yaml"))
    assert project.task == "Maze" and project.date == "May1" and not project.multi_animal
    assert project.bodyparts == ["snout", "left eye", "right eye", "head-body junction", "tail base"]
    assert project.pcutoff == 0.7 and project.skeleton == [("snout", "head-body junction")]
    assert project.folder == str(tmp_path)  # the folder of the file, not the stale project_path
    assert project.models == [] and project.default_model() is None
    assert project.results_tag(1) == "MazeMay1shuffle1_"


def test_project_config_multi_and_models(tmp_path):
    (tmp_path / "config.yaml").write_text(CONFIG_MULTI)
    torch_train = tmp_path / "dlc-models-pytorch" / "iteration-2" / "ShoalJun2-trainset80shuffle3" / "train"
    torch_train.mkdir(parents=True)
    for name in ("snapshot-100.pt", "snapshot-best-150.pt", "snapshot-detector-050.pt", "pytorch_config.yaml"):
        (torch_train / name).write_bytes(b"")
    old_train = tmp_path / "dlc-models" / "iteration-0" / "ShoalJun2-trainset80shuffle1" / "train"
    old_train.mkdir(parents=True)
    for name in ("snapshot-50000.index", "snapshot-50000.meta", "snapshot-50000.data-00000-of-00001", "pose_cfg.yaml"):
        (old_train / name).write_bytes(b"")
    empty = tmp_path / "dlc-models-pytorch" / "iteration-2" / "ShoalJun2-trainset80shuffle4" / "train"
    empty.mkdir(parents=True)  # created but never trained

    project = DLCProject.load(str(tmp_path / "config.yaml"))
    assert project.multi_animal and project.individuals == ["fish1", "fish2", "fish3"]
    assert project.all_bodyparts() == ["snout", "tail", "feeder"]
    assert [(m.engine, m.iteration, m.shuffle, m.trainset_percent, m.trainingsetindex) for m in project.models] == [
        ("pytorch", 2, 3, 80, 0),
        ("tensorflow", 0, 1, 80, 0),
    ]
    assert project.models[0].snapshots == ["snapshot-100.pt", "snapshot-best-150.pt"]
    assert project.models[1].snapshots == ["snapshot-50000"]
    assert project.default_model().engine == "pytorch"
    assert project.find_model(project.models[1].key).shuffle == 1


def test_not_a_project_file(tmp_path):
    (tmp_path / "other.yaml").write_text("just: something\n")
    with pytest.raises(ProjectError):
        DLCProject.load(str(tmp_path / "other.yaml"))
    with pytest.raises(ProjectError):
        DLCProject.load(str(tmp_path / "missing.yaml"))
    (tmp_path / "list.yaml").write_text("- a\n- b\n")
    with pytest.raises(ProjectError):
        DLCProject.load(str(tmp_path / "list.yaml"))
