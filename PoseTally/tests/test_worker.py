"""The tracking worker, driven against a stand-in for DeepLabCut (tests/fake_dlc)."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from posetally.demo import make_demo
from posetally.worker import PREFIX, parse_line

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(HERE), "app")
FAKE = os.path.join(HERE, "fake_dlc")


@pytest.fixture()
def batch(tmp_path_factory):
    """A project folder plus a folder of four untracked videos."""
    root = tmp_path_factory.mktemp("worker")
    configs = make_demo(str(root))
    source = os.path.join(os.path.dirname(configs["single"]), "videos", "trial_01_groupA.mp4")
    videos = root / "new videos"
    videos.mkdir()
    paths = []
    for name in ("fish_a.mp4", "fish_b.mp4", "fish_c.mp4", "fish_d.mp4"):
        shutil.copy(source, videos / name)
        paths.append(str(videos / name))
    return {"config": configs["single"], "videos": paths, "folder": str(videos), "root": str(root)}


def run_worker(tmp_path, spec, env_extra=None, with_dlc=True):
    job = tmp_path / "job.json"
    job.write_text(json.dumps(spec))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([APP] + ([FAKE] if with_dlc else []))
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "posetally.worker", "--job", str(job)], capture_output=True, text=True, env=env, timeout=120
    )
    events = [m for m in (parse_line(line)[1] for line in proc.stdout.splitlines()) if m]
    return proc, events


def spec_for(batch, **changes):
    spec = {
        "config": batch["config"],
        "videos": batch["videos"],
        "shuffle": 1,
        "trainingsetindex": 0,
        "engine": "pytorch",
        "tag": "MazeDemoNov10shuffle1_",
        "multi_animal": False,
    }
    spec.update(changes)
    return spec


def names(events, kind):
    return [os.path.basename(e["video"]) for e in events if e["event"] == kind]


def test_a_batch_runs_and_reports_each_video(batch, tmp_path):
    record = tmp_path / "calls.jsonl"
    proc, events = run_worker(tmp_path, spec_for(batch, device="cuda", batch_size=4), {"FAKE_DLC_RECORD": str(record)})
    assert proc.returncode == 0, proc.stderr
    assert events[0]["event"] == "status" and events[1]["event"] == "ready"
    assert events[1]["deeplabcut"] == "3.0.2-fake"
    assert names(events, "video_start") == ["fish_a.mp4", "fish_b.mp4", "fish_c.mp4", "fish_d.mp4"]
    assert sorted(names(events, "video_done")) == ["fish_a.mp4", "fish_b.mp4", "fish_c.mp4", "fish_d.mp4"]
    assert events[-1] == {"event": "finished", "done": 4, "failed": 0, "total": 4}
    assert all(os.path.isfile(e["output"]) and e["output"].endswith(".h5") for e in events if e["event"] == "video_done")
    assert "Starting to analyze" in proc.stdout and "%|" in proc.stderr  # DeepLabCut's own output passes through
    (call,) = [json.loads(line) for line in record.read_text().splitlines()]
    assert call["in_random_order"] is False and call["batch_size"] == 4
    assert call["torch_kwargs"] == {"device": "cuda"} and call["destfolder"] is None

    # running again does nothing: every video already has its table
    proc, events = run_worker(tmp_path, spec_for(batch), {"FAKE_DLC_RECORD": str(record)})
    assert events[-1] == {"event": "finished", "done": 4, "failed": 0, "total": 4}
    assert len(record.read_text().splitlines()) == 1


def test_results_can_go_to_another_folder(batch, tmp_path):
    out = tmp_path / "tracking results"
    proc, events = run_worker(tmp_path, spec_for(batch, destfolder=str(out), save_csv=True))
    assert proc.returncode == 0, proc.stderr
    outputs = [e["output"] for e in events if e["event"] == "video_done"]
    assert len(outputs) == 4 and all(os.path.dirname(o) == str(out) for o in outputs)
    assert len([n for n in os.listdir(out) if n.endswith(".csv")]) == 4
    assert not [n for n in os.listdir(batch["folder"]) if n.endswith(".h5")]


def test_one_bad_video_does_not_stop_the_rest(batch, tmp_path):
    folder = batch["folder"]
    os.rename(os.path.join(folder, "fish_b.mp4"), os.path.join(folder, "fish_crash.mp4"))
    with open(os.path.join(folder, "broken.mp4"), "wb") as handle:
        handle.write(b"this is not a video")
    os.rename(os.path.join(folder, "fish_c.mp4"), os.path.join(folder, "fish_nodetect.mp4"))
    videos = [os.path.join(folder, n) for n in ("broken.mp4", "fish_a.mp4", "fish_crash.mp4", "fish_d.mp4", "fish_nodetect.mp4", "gone.mp4")]
    proc, events = run_worker(tmp_path, spec_for(batch, videos=videos))
    assert proc.returncode == 0, proc.stderr
    errors = {os.path.basename(e["video"]): e["error"] for e in events if e["event"] == "video_error"}
    assert set(errors) == {"broken.mp4", "fish_crash.mp4", "fish_nodetect.mp4", "gone.mp4"}
    assert "pretend decoding error" in errors["fish_crash.mp4"]
    assert "no animal was detected" in errors["fish_nodetect.mp4"]
    assert "not found" in errors["gone.mp4"]
    assert sorted(names(events, "video_done")) == ["fish_a.mp4", "fish_d.mp4"]
    assert events[-1] == {"event": "finished", "done": 2, "failed": 4, "total": 6}


def test_names_deeplabcut_would_silently_skip_are_explained(batch, tmp_path):
    folder = batch["folder"]
    os.rename(os.path.join(folder, "fish_b.mp4"), os.path.join(folder, "arena_full.mp4"))
    videos = [os.path.join(folder, n) for n in ("fish_d.mp4", "arena_full.mp4", "fish_a.mp4")]
    proc, events = run_worker(tmp_path, spec_for(batch, videos=videos))
    assert proc.returncode == 0, proc.stderr
    errors = {os.path.basename(e["video"]): e["error"] for e in events if e["event"] == "video_error"}
    assert list(errors) == ["arena_full.mp4"] and "Rename this video" in errors["arena_full.mp4"]
    # DeepLabCut goes through videos in sorted order, not the order they were listed in
    assert names(events, "video_start") == ["fish_a.mp4", "fish_d.mp4"]
    assert events[-1] == {"event": "finished", "done": 2, "failed": 1, "total": 3}


def test_a_problem_with_the_model_stops_the_run_with_one_clear_message(batch, tmp_path):
    proc, events = run_worker(tmp_path, spec_for(batch), {"FAKE_DLC_SETUP_FAIL": "1"})
    assert proc.returncode == 2
    fatal = [e for e in events if e["event"] == "fatal"]
    assert len(fatal) == 1 and fatal[0]["kind"] == "setup" and "No snapshots" in fatal[0]["error"]
    assert names(events, "video_error") == []


def test_missing_deeplabcut_is_reported(batch, tmp_path):
    proc, events = run_worker(tmp_path, spec_for(batch), with_dlc=False)
    try:
        import deeplabcut  # noqa: F401
    except ImportError:
        assert proc.returncode == 2
        assert events[-1]["event"] == "fatal" and events[-1]["kind"] == "no_deeplabcut"


def test_multi_animal_waits_for_the_tracked_table_and_finishes_interrupted_runs(batch, tmp_path):
    folder = batch["folder"]
    stale = os.path.join(folder, "fish_a" + "DLC_Resnet50_MazeDemoNov10shuffle1_snapshot_200_full.pickle")
    with open(stale, "wb") as handle:  # detections left behind by a run that was interrupted
        handle.write(b"old")
    os.utime(stale, (1, 1))
    record = tmp_path / "calls.jsonl"
    env = {"FAKE_DLC_MULTI": "1", "FAKE_DLC_RECORD": str(record)}
    proc, events = run_worker(tmp_path, spec_for(batch, multi_animal=True, n_animals=2), env)
    assert proc.returncode == 0, proc.stderr
    done = {os.path.basename(e["video"]): os.path.basename(e["output"]) for e in events if e["event"] == "video_done"}
    assert sorted(done) == ["fish_a.mp4", "fish_b.mp4", "fish_c.mp4", "fish_d.mp4"]
    assert all(name.endswith("_el.h5") for name in done.values())
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    assert calls[0]["n_tracks"] == 2 and calls[0]["overwrite"] is False
    assert calls[-1]["overwrite"] is True and [os.path.basename(v) for v in calls[-1]["videos"]] == ["fish_a.mp4"]


def test_check_mode_lists_what_is_installed(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([APP, FAKE])
    proc = subprocess.run([sys.executable, "-m", "posetally.worker", "--check"], capture_output=True, text=True, env=env, timeout=120)
    (line,) = [line for line in proc.stdout.splitlines() if line.startswith(PREFIX)]
    report = json.loads(line[len(PREFIX):])
    assert report["event"] == "environment" and report["deeplabcut"] == "3.0.2-fake" and "python" in report


def test_known_deeplabcut_failures_are_explained_in_plain_words(batch, tmp_path):
    from posetally.worker import in_plain_words

    assert in_plain_words("ValueError: something else") == "ValueError: something else"
    assert "smaller number" in in_plain_words("OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert "Animals per video" in in_plain_words("ValueError: Could not reconstruct 3 tracks from the tracklets given.")
    # the way DeepLabCut 3.0.2 really fails when a multi-animal network finds nothing in a video
    folder = batch["folder"]
    os.rename(os.path.join(folder, "fish_b.mp4"), os.path.join(folder, "fish_noanimal.mp4"))
    videos = [os.path.join(folder, n) for n in ("fish_a.mp4", "fish_noanimal.mp4", "fish_c.mp4")]
    proc, events = run_worker(tmp_path, spec_for(batch, videos=videos))
    assert proc.returncode == 0, proc.stderr
    errors = {os.path.basename(e["video"]): e["error"] for e in events if e["event"] == "video_error"}
    assert list(errors) == ["fish_noanimal.mp4"]
    assert errors["fish_noanimal.mp4"].startswith("DeepLabCut did not find any animal in this video")
    assert "Tracklets are empty." in errors["fish_noanimal.mp4"]  # the original message is kept
    assert sorted(names(events, "video_done")) == ["fish_a.mp4", "fish_c.mp4"]
