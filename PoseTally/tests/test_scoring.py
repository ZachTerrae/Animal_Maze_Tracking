import math

import numpy as np
import pytest

from posetally.model import Layout, LineShape, RoiShape, Rule, ScoringSettings
from posetally.scoring import fill_short_gaps, median_smooth, prepare_trajectory, score_tracks
from posetally.tracks import TrackSet

FPS = 10.0


def make_tracks(paths: dict, likelihood: dict | None = None, multi: bool = False) -> TrackSet:
    """paths: {(animal, bodypart): (N, 2) array}"""
    n = len(next(iter(paths.values())))
    tracks = TrackSet(path="memory", n_frames=n, multi_animal=multi)
    for (animal, part), xy in paths.items():
        arr = np.empty((n, 3))
        arr[:, :2] = xy
        arr[:, 2] = (likelihood or {}).get((animal, part), np.ones(n))
        if animal not in tracks.individuals:
            tracks.individuals.append(animal)
            tracks.bodyparts[animal] = []
        tracks.bodyparts[animal].append(part)
        tracks.data[(animal, part)] = arr
    return tracks


def horizontal_run(xs, y=100.0):
    return np.column_stack([np.asarray(xs, dtype=float), np.full(len(xs), float(y))])


def plain_settings(**changes) -> ScoringSettings:
    settings = ScoringSettings(pcutoff=0.6, max_gap_s=0.0, smooth_frames=1, margin_px=0.0)
    for key, value in changes.items():
        setattr(settings, key, value)
    return settings


def run(layout: Layout, tracks: TrackSet, frame_size=(400, 300), video="video"):
    return score_tracks(tracks, layout.resolve(video, frame_size), layout.settings, FPS)


def back_and_forth():
    # crosses x = 100 to the right, back to the left, and to the right again
    xs = list(range(50, 151, 10)) + list(range(140, 49, -10)) + list(range(60, 151, 10))
    return horizontal_run(xs)


def line_at_100(rules) -> LineShape:
    return LineShape(name="Gate", p1=(100.0, 0.0), p2=(100.0, 200.0), rules=rules)


def test_direction_rules_and_points():
    layout = Layout(settings=plain_settings())
    layout.lines = [
        line_at_100(
            [
                Rule(bodypart="head", direction="left_to_right", points=1, column="L>R"),
                Rule(bodypart="head", direction="right_to_left", points=1, column="R>L"),
                Rule(bodypart="head", direction="any", points=0.5, column="Any x0.5"),
                Rule(bodypart="head", direction="top_to_bottom", points=1, column="Down"),
            ]
        )
    ]
    (animal,) = run(layout, make_tracks({("", "head"): back_and_forth()}))
    assert animal.scores == {"L>R": 2.0, "R>L": 1.0, "Any x0.5": 1.5, "Down": 0.0}
    counted = [e for e in animal.events if e.column == "L>R"]
    assert [e.what for e in counted] == ["crossed left to right"] * 2
    assert counted[0].time_s == pytest.approx(counted[0].frame / FPS)


def test_two_rules_can_feed_one_column_and_one_line_many_columns():
    layout = Layout(settings=plain_settings())
    layout.lines = [
        line_at_100(
            [
                Rule(bodypart="head", direction="left_to_right", points=2, column="Score"),
                Rule(bodypart="head", direction="right_to_left", points=-1, column="Score"),
                Rule(bodypart="head", direction="any", points=1, column="Crossings"),
            ]
        )
    ]
    (animal,) = run(layout, make_tracks({("", "head"): back_and_forth()}))
    assert animal.scores == {"Score": 2 * 2 - 1, "Crossings": 3.0}


def test_first_time_only_and_time_of_first_crossing():
    layout = Layout(settings=plain_settings())
    layout.lines = [
        line_at_100(
            [
                Rule(bodypart="head", direction="left_to_right", points=1, column="Ever", count_mode="first"),
                Rule(bodypart="head", measure="first_time", direction="right_to_left", column="First back (s)"),
                Rule(bodypart="head", measure="first_time", direction="top_to_bottom", column="Never (s)"),
            ]
        )
    ]
    (animal,) = run(layout, make_tracks({("", "head"): back_and_forth()}))
    assert animal.scores["Ever"] == 1.0
    # frames: 0..10 go right (x=50..150), 11.. go left from 140; x=90 is frame 16
    assert animal.scores["First back (s)"] == pytest.approx(16 / FPS)
    assert math.isnan(animal.scores["Never (s)"])


def test_first_choice_group_only_the_first_line_scores():
    # the animal crosses the right line first, then comes back over the left one
    xs = list(range(100, 201, 10)) + list(range(190, -1, -10))
    layout = Layout(settings=plain_settings())
    left = LineShape(name="Left", p1=(50.0, 0.0), p2=(50.0, 200.0))
    left.rules = [
        Rule(bodypart="head", direction="right_to_left", column="Chose left", count_mode="first_in_group", group="arm"),
        Rule(bodypart="head", direction="right_to_left", column="Left visits"),
    ]
    right = LineShape(name="Right", p1=(150.0, 0.0), p2=(150.0, 200.0))
    right.rules = [
        Rule(bodypart="head", direction="left_to_right", column="Chose right", count_mode="first_in_group", group="arm"),
        Rule(bodypart="head", direction="left_to_right", column="Right visits"),
    ]
    layout.lines = [left, right]
    (animal,) = run(layout, make_tracks({("", "head"): horizontal_run(xs)}))
    assert animal.scores == {"Chose left": 0.0, "Left visits": 1.0, "Chose right": 1.0, "Right visits": 1.0}


def test_region_time_entries_exits_and_first_entry():
    xs = np.concatenate([np.linspace(55, 155, 11), np.full(5, 155.0), np.linspace(155, 255, 11), np.linspace(245, 155, 10)])
    layout = Layout(settings=plain_settings())
    box = RoiShape(name="Box", points=[(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)])
    box.rules = [
        Rule(bodypart="body", measure="time", column="Time (s)"),
        Rule(bodypart="body", measure="entries", points=1, column="Entries"),
        Rule(bodypart="body", measure="exits", points=1, column="Exits"),
        Rule(bodypart="body", measure="first_time", column="First entry (s)"),
        Rule(bodypart="body", measure="time", points=1 / 60, column="Time (min)"),
    ]
    layout.rois = [box]
    tracks = make_tracks({("", "body"): horizontal_run(xs, y=150.0)})
    (animal,) = run(layout, tracks)
    inside = int(((xs > 100) & (xs < 200)).sum())
    assert animal.scores["Time (s)"] == pytest.approx(inside / FPS)
    assert animal.scores["Time (min)"] == pytest.approx(inside / FPS / 60)
    assert animal.scores["Entries"] == 2.0 and animal.scores["Exits"] == 1.0
    assert animal.scores["First entry (s)"] == pytest.approx(5 / FPS)


def test_ellipse_region():
    layout = Layout(settings=plain_settings())
    oval = RoiShape(name="Oval", form="ellipse", points=[(100.0, 50.0), (200.0, 150.0)])
    oval.rules = [Rule(bodypart="body", measure="entries", column="In")]
    layout.rois = [oval]
    tracks = make_tracks({("", "body"): horizontal_run(np.linspace(20, 280, 53))})
    (animal,) = run(layout, tracks)
    assert animal.scores["In"] == 1.0


def test_each_animal_gets_its_own_tally():
    layout = Layout(settings=plain_settings())
    layout.lines = [line_at_100([Rule(bodypart="head", direction="any", column="Crossings")])]
    stay = horizontal_run(np.full(31, 60.0))
    tracks = make_tracks({("fish1", "head"): back_and_forth(), ("fish2", "head"): stay}, multi=True)
    results = run(layout, tracks)
    assert [(r.animal, r.scores["Crossings"]) for r in results] == [("fish1", 3.0), ("fish2", 0.0)]


def test_unique_parts_owner_is_not_an_animal():
    layout = Layout(settings=plain_settings())
    layout.lines = [line_at_100([Rule(bodypart="head", direction="any", column="Crossings")])]
    tracks = make_tracks(
        {("fish1", "head"): back_and_forth(), ("single", "corner"): horizontal_run(np.full(31, 5.0))}, multi=True
    )
    assert [r.animal for r in run(layout, tracks)] == ["fish1"]


def test_missing_body_part_leaves_the_column_empty_and_says_why():
    layout = Layout(settings=plain_settings())
    layout.lines = [line_at_100([Rule(bodypart="fin", direction="any", column="Fin crossings")])]
    (animal,) = run(layout, make_tracks({("", "head"): back_and_forth()}))
    assert math.isnan(animal.scores["Fin crossings"])
    assert any("fin" in note for note in animal.notes)


def test_low_likelihood_points_are_ignored():
    xy = horizontal_run(np.full(20, 60.0))
    xy[10] = (140.0, 100.0)  # a single wild guess on the other side of the line
    lik = np.ones(20)
    lik[10] = 0.1
    layout = Layout(settings=plain_settings())
    layout.lines = [line_at_100([Rule(bodypart="head", direction="any", column="Crossings")])]
    tracks = make_tracks({("", "head"): xy}, {("", "head"): lik})
    assert run(layout, tracks)[0].scores["Crossings"] == 0.0
    layout.settings.pcutoff = 0.0  # trusting everything counts the jump there and back
    assert run(layout, tracks)[0].scores["Crossings"] == 2.0


def test_smoothing_removes_a_confident_single_frame_jump():
    xy = horizontal_run(np.full(20, 60.0))
    xy[10] = (140.0, 100.0)
    layout = Layout(settings=plain_settings())
    layout.lines = [line_at_100([Rule(bodypart="head", direction="any", column="Crossings")])]
    tracks = make_tracks({("", "head"): xy})
    assert run(layout, tracks)[0].scores["Crossings"] == 2.0
    layout.settings.smooth_frames = 5
    assert run(layout, tracks)[0].scores["Crossings"] == 0.0


def test_crossing_inside_a_gap_can_be_counted_or_not():
    xs = np.array([60, 70, 80, 90, np.nan, np.nan, np.nan, 120, 130], dtype=float)
    layout = Layout(settings=plain_settings())
    layout.lines = [line_at_100([Rule(bodypart="head", direction="left_to_right", column="Crossings")])]
    tracks = make_tracks({("", "head"): horizontal_run(xs)})
    (animal,) = run(layout, tracks)
    assert animal.scores["Crossings"] == 1.0 and animal.gap_events == 1
    layout.settings.count_gap_crossings = False
    assert run(layout, tracks)[0].scores["Crossings"] == 0.0


def test_body_orientation_can_be_required():
    # the head point moves left to right over the line while the animal faces left (backing up)
    head = horizontal_run(np.arange(50, 151, 10))
    tail = head + np.array([20.0, 0.0])  # tail is to the right of the head: facing left
    layout = Layout(settings=plain_settings())
    layout.lines = [line_at_100([Rule(bodypart="head", direction="left_to_right", column="Forward crossings")])]
    tracks = make_tracks({("", "head"): head, ("", "tail"): tail})
    assert run(layout, tracks)[0].scores["Forward crossings"] == 1.0
    layout.settings.use_heading = True
    layout.settings.head_part, layout.settings.tail_part = "head", "tail"
    assert run(layout, tracks)[0].scores["Forward crossings"] == 0.0
    tracks = make_tracks({("", "head"): head, ("", "tail"): head - np.array([20.0, 0.0])})
    assert run(layout, tracks)[0].scores["Forward crossings"] == 1.0


def test_time_in_region_while_untracked():
    xs = np.array([50, 80, 120, 150, np.nan, np.nan, np.nan, 150, 250], dtype=float)
    layout = Layout(settings=plain_settings())
    box = RoiShape(name="Box", points=[(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)])
    box.rules = [Rule(bodypart="body", measure="time", column="Time (s)")]
    layout.rois = [box]
    tracks = make_tracks({("", "body"): horizontal_run(xs, y=150.0)})
    assert run(layout, tracks)[0].scores["Time (s)"] == pytest.approx(6 / FPS)
    layout.settings.roi_gap_policy = "exclude"
    assert run(layout, tracks)[0].scores["Time (s)"] == pytest.approx(3 / FPS)


def test_shapes_stretch_to_other_frame_sizes_and_videos_can_have_their_own():
    layout = Layout(settings=plain_settings(), ref_size=(400, 300))
    layout.lines = [line_at_100([Rule(bodypart="head", direction="any", column="Crossings")])]
    # same scene filmed at twice the resolution: the line should sit at x = 200
    double = make_tracks({("", "head"): horizontal_run(np.arange(150, 251, 10), y=200.0)})
    assert run(layout, double, frame_size=(800, 600))[0].scores["Crossings"] == 1.0
    assert run(layout, double, frame_size=(400, 300))[0].scores["Crossings"] == 0.0
    # one video gets its own line position
    line_id = layout.lines[0].id
    layout.store_geometry(line_id, {"p1": [300.0, 0.0], "p2": [300.0, 300.0]}, "odd_video", (400, 300), only_this_video=True)
    assert run(layout, double, frame_size=(400, 300), video="odd_video")[0].scores["Crossings"] == 0.0
    far = make_tracks({("", "head"): horizontal_run(np.arange(250, 351, 10), y=200.0)})
    assert run(layout, far, frame_size=(400, 300), video="odd_video")[0].scores["Crossings"] == 1.0
    assert run(layout, far, frame_size=(400, 300), video="other")[0].scores["Crossings"] == 0.0


def test_gap_filling_and_smoothing_helpers():
    xy = horizontal_run([0, 10, np.nan, np.nan, 40, np.nan, np.nan, np.nan, np.nan, 90])
    filled = fill_short_gaps(xy, 2)
    assert filled[2, 0] == pytest.approx(20) and filled[3, 0] == pytest.approx(30)
    assert np.isnan(filled[5:9, 0]).all()  # a 4-frame gap is too long to bridge
    smooth = median_smooth(horizontal_run([0, 0, 50, 0, 0, np.nan, 7, 7]), 3)
    assert smooth[2, 0] == 0 and np.isnan(smooth[5, 0])
    arr = np.column_stack([horizontal_run([1, 2, 3, 4]), [0.9, 0.2, 0.9, 0.9]])
    traj = prepare_trajectory(arr, plain_settings(max_gap_s=0.5), FPS)
    assert traj.tracked == pytest.approx(0.75) and traj.xy[1, 0] == pytest.approx(2.0)
