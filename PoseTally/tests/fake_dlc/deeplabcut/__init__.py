"""A stand-in for DeepLabCut used only by the tests.

It has the same ``analyze_videos`` signature as DeepLabCut 3.0.2, prints the
same messages, and writes files with the same names and layout, but instead of
running a network it writes made-up positions. How it behaves for a video is
steered by words in the video's file name:

* ``crash``    - raise an error when that video is reached
* ``nodetect`` - finish without writing a table (as when no animal is found)
* ``vanish``   - end the whole process on the spot, without a word (as when the
  system stops a program that has used up the memory)
* ``noanimal`` - fail the way DeepLabCut 3.0.2 does when a multi-animal network
  detects nothing in a video
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "3.0.2-fake"

SCORER = "DLC_Resnet50_MazeDemoNov10shuffle{shuffle}_snapshot_200"


def analyze_videos(
    config,
    videos,
    video_extensions=None,
    shuffle=1,
    trainingsetindex=0,
    gputouse=None,
    save_as_csv=False,
    in_random_order=True,
    destfolder=None,
    batch_size=None,
    cropping=None,
    TFGPUinference=True,
    dynamic=(False, 0.5, 10),
    modelprefix="",
    robust_nframes=False,
    allow_growth=False,
    use_shelve=False,
    auto_track=True,
    n_tracks=None,
    animal_names=None,
    calibrate=False,
    identity_only=False,
    use_openvino=None,
    engine=None,
    overwrite=False,
    **torch_kwargs,
):
    record = os.environ.get("FAKE_DLC_RECORD")
    if record:
        with open(record, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "videos": [str(v) for v in videos],
                        "shuffle": shuffle,
                        "in_random_order": in_random_order,
                        "destfolder": None if destfolder is None else str(destfolder),
                        "save_as_csv": save_as_csv,
                        "batch_size": batch_size,
                        "n_tracks": n_tracks,
                        "overwrite": overwrite,
                        "torch_kwargs": torch_kwargs,
                    }
                )
                + "\n"
            )
    if os.environ.get("FAKE_DLC_SETUP_FAIL"):
        raise FileNotFoundError("No snapshots were found for this shuffle (pretend).")
    if destfolder:  # DeepLabCut creates the output folder when it does not exist
        Path(destfolder).mkdir(parents=True, exist_ok=True)
    multi = bool(os.environ.get("FAKE_DLC_MULTI"))
    scorer = SCORER.format(shuffle=shuffle)
    print(f"Analyzing videos with /pretend/snapshot-200.pt")
    print(f"Using scorer: {scorer}")
    # like DeepLabCut 3.0.2: sorted order, and names ending in _labeled or _full are skipped
    todo = sorted(Path(v).absolute() for v in videos)
    todo = [v for v in todo if not (v.match("*_labeled.*") or v.match("*_full.*"))]
    for video in todo:
        out_dir = Path(destfolder) if destfolder else video.parent
        prefix = video.stem + scorer
        full_pickle = out_dir / f"{prefix}_full.pickle"
        if not overwrite and full_pickle.exists():
            print(f"Video {video} already analyzed at {full_pickle}!")
            continue
        print(f"Starting to analyze {video}")
        if "crash" in video.stem:
            raise RuntimeError("pretend decoding error")
        if "noanimal" in video.stem:
            raise OSError("Tracklets are empty.")
        if "vanish" in video.stem:
            sys.stdout.flush()
            os._exit(137)
        for step in range(0, 101, 25):
            sys.stderr.write(f"\r{step:3d}%|{'#' * (step // 10):10s}| {step}/100 [00:00<00:00, 99.0it/s]")
            sys.stderr.flush()
            time.sleep(0.01)
        sys.stderr.write("\n")
        with open(out_dir / f"{prefix}_meta.pickle", "wb") as handle:
            pickle.dump({"data": {"fps": 20.0, "nframes": 50, "cropping": False, "pytorch-config": {}}}, handle)
        with open(full_pickle, "wb") as handle:
            pickle.dump({"metadata": {}}, handle)
        if "nodetect" in video.stem:
            continue
        n = 50
        parts = ["nose", "head"]
        animals = ["animal1", "animal2"] if multi else [None]
        cols, arrays = [], []
        for animal in animals:
            for part in parts:
                for coord in ("x", "y", "likelihood"):
                    cols.append((scorer, part, coord) if animal is None else (scorer, animal, part, coord))
                    arrays.append(np.linspace(10, 300, n) if coord != "likelihood" else np.ones(n))
        names = ["scorer", "bodyparts", "coords"] if not multi else ["scorer", "individuals", "bodyparts", "coords"]
        df = pd.DataFrame(np.column_stack(arrays), columns=pd.MultiIndex.from_tuples(cols, names=names), index=range(n))
        target = out_dir / (prefix + ("_el.h5" if multi else ".h5"))
        df.to_hdf(target, key="df_with_missing", format="table", mode="w")
        if save_as_csv:
            df.to_csv(target.with_suffix(".csv"))
    print("The videos are analyzed. Now your research can truly start!")
    return scorer
