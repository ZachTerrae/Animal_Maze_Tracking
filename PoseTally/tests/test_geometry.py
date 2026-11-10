import numpy as np
import pytest

from posetally.geometry import (
    crossing_vector,
    distance_to_polygon_edge,
    ellipse_polygon,
    find_line_crossings,
    inradius_estimate,
    meaningful_directions,
    movement_matches,
    points_in_polygon,
    polygon_centroid,
    simplify_path,
    trace_region,
)

VERTICAL = ((100.0, 0.0), (100.0, 200.0))  # a vertical line at x = 100
HORIZONTAL = ((0.0, 100.0), (200.0, 100.0))


def path(xs, y=100.0):
    return np.column_stack([np.asarray(xs, dtype=float), np.full(len(xs), float(y))])


def directions_of(xy, line, margin=0.0):
    out = []
    for c in find_line_crossings(xy, *line, margin):
        mx, my = crossing_vector(*line, c.toward)
        for name in ("left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"):
            if movement_matches(name, mx, my):
                out.append((c.frame, name))
    return out


def test_single_crossing_left_to_right():
    xy = path(np.arange(50, 151, 10))  # 50, 60, ... 150
    assert directions_of(xy, VERTICAL) == [(6, "left_to_right")]


def test_direction_does_not_depend_on_how_the_line_was_drawn():
    xy = path(np.arange(50, 151, 10))
    flipped = (VERTICAL[1], VERTICAL[0])
    assert directions_of(xy, flipped) == [(6, "left_to_right")]


def test_right_to_left_and_back():
    xs = list(range(150, 49, -10)) + list(range(60, 151, 10))
    found = directions_of(path(xs), VERTICAL)
    assert [name for _, name in found] == ["right_to_left", "left_to_right"]


def test_vertical_movement_over_horizontal_line():
    ys = np.arange(50, 151, 10, dtype=float)
    xy = np.column_stack([np.full(len(ys), 100.0), ys])
    assert [n for _, n in directions_of(xy, HORIZONTAL)] == ["top_to_bottom"]
    assert [n for _, n in directions_of(xy[::-1], HORIZONTAL)] == ["bottom_to_top"]


def test_jitter_on_the_line_counts_once_with_margin():
    rng = np.random.default_rng(1)
    approach = np.linspace(60, 98, 20)
    jitter = 100 + rng.uniform(-2.5, 2.5, size=60)  # dithering right on top of the line
    leave = np.linspace(103, 140, 20)
    xy = path(np.concatenate([approach, jitter, leave]))
    assert len(find_line_crossings(xy, *VERTICAL, margin=0.0)) > 1  # naive counting is fooled
    assert [n for _, n in directions_of(xy, VERTICAL, margin=4.0)] == ["left_to_right"]


def test_approach_and_retreat_is_not_a_crossing():
    xs = np.concatenate([np.linspace(60, 99, 20), np.linspace(101, 103, 5), np.linspace(99, 60, 20)])
    assert find_line_crossings(path(xs), *VERTICAL, margin=4.0) == []


def test_going_around_the_end_of_the_line_is_not_counted():
    short = ((100.0, 0.0), (100.0, 50.0))
    xy = path(np.arange(50, 151, 10), y=120.0)  # passes below the end of the line
    assert find_line_crossings(xy, *short) == []
    # coming back through the line itself is counted, in the right direction
    back = path(np.arange(150, 49, -10), y=25.0)
    both = np.vstack([xy, back])
    assert [n for _, n in directions_of(both, short)] == ["right_to_left"]


def test_crossing_during_a_tracking_gap_reports_the_gap():
    xs = [60, 70, 80, 90, np.nan, np.nan, np.nan, 120, 130]
    crossings = find_line_crossings(path(xs), *VERTICAL)
    assert len(crossings) == 1
    assert crossings[0].frame == 7 and crossings[0].prev_frame == 3 and crossings[0].gap == 3


def test_untracked_path_gives_nothing():
    assert find_line_crossings(path([np.nan] * 10), *VERTICAL) == []
    assert find_line_crossings(np.zeros((0, 2)), *VERTICAL) == []


def test_zero_length_line_is_rejected():
    with pytest.raises(ValueError):
        find_line_crossings(path([1, 2, 3]), (5, 5), (5, 5))


def test_meaningful_directions():
    assert meaningful_directions(*VERTICAL) == ["any", "left_to_right", "right_to_left"]
    assert meaningful_directions(*HORIZONTAL) == ["any", "top_to_bottom", "bottom_to_top"]
    assert len(meaningful_directions((0, 0), (100, 100))) == 5


SQUARE = np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]])


def test_points_in_polygon_and_distance():
    pts = np.array([[150.0, 150.0], [50.0, 150.0], [150.0, 250.0], [np.nan, 1.0]])
    assert points_in_polygon(pts, SQUARE).tolist() == [True, False, False, False]
    dist = distance_to_polygon_edge(pts, SQUARE)
    assert dist[0] == pytest.approx(50.0) and dist[1] == pytest.approx(50.0) and np.isnan(dist[3])


def test_concave_polygon():
    u_shape = np.array([[0, 0], [300, 0], [300, 300], [200, 300], [200, 100], [100, 100], [100, 300], [0, 300]], dtype=float)
    pts = np.array([[50.0, 200.0], [150.0, 200.0], [250.0, 200.0], [150.0, 50.0]])
    assert points_in_polygon(pts, u_shape).tolist() == [True, False, True, True]


def test_region_entries_exits_and_states():
    xs = np.concatenate([np.linspace(55, 155, 11), np.full(5, 155.0), np.linspace(155, 255, 11)])
    xy = path(xs, y=150.0)
    trace = trace_region(xy, SQUARE)
    assert trace.entries.tolist() == [5]  # x = 105 is the first point inside
    assert trace.exits.tolist() == [21]  # x = 205 is the first point outside again
    assert int((trace.state == 1).sum()) == 16


def test_region_jitter_on_the_edge_counts_once_with_margin():
    rng = np.random.default_rng(3)
    xs = np.concatenate([np.linspace(60, 98, 15), 100 + rng.uniform(-2.5, 2.5, 50), np.linspace(104, 150, 15)])
    xy = path(xs, y=150.0)
    assert len(trace_region(xy, SQUARE, margin=0.0).entries) > 1
    assert len(trace_region(xy, SQUARE, margin=4.0).entries) == 1


def test_region_keeps_last_state_while_untracked():
    xs = [50, 80, 120, 150, np.nan, np.nan, np.nan, 150, 250]
    trace = trace_region(path(xs, y=150.0), SQUARE)
    assert trace.state.tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 0]
    assert trace.valid.tolist() == [True, True, True, True, False, False, False, True, True]


def test_starting_inside_is_not_an_entry():
    trace = trace_region(path([150, 150, 150, 250], y=150.0), SQUARE)
    assert trace.entries.tolist() == [] and trace.exits.tolist() == [3]
    assert trace.first_known == 0 and trace.state[0] == 1


def test_narrow_region_can_still_be_entered_with_a_large_margin():
    strip = np.array([[100.0, 0.0], [106.0, 0.0], [106.0, 300.0], [100.0, 300.0]])  # 6 px wide
    xy = path([80, 90, 103, 103, 103, 120, 130], y=150.0)
    trace = trace_region(xy, strip, margin=8.0)
    assert trace.entries.tolist() == [2]
    assert inradius_estimate(strip) == pytest.approx(3.0, abs=0.6)


def test_ellipse_polygon_and_centroid():
    poly = ellipse_polygon((100, 100), (300, 200))
    assert points_in_polygon(np.array([[200.0, 150.0], [105.0, 105.0]]), poly).tolist() == [True, False]
    cx, cy = polygon_centroid(poly)
    assert cx == pytest.approx(200.0, abs=0.5) and cy == pytest.approx(150.0, abs=0.5)


def test_simplify_path_keeps_corners():
    pts = [(x, 0.0) for x in range(0, 101, 5)] + [(100.0, y) for y in range(5, 101, 5)]
    simple = simplify_path(pts, 1.0)
    assert simple == [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]
