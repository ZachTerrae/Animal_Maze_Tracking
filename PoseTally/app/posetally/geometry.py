"""Geometry used for scoring.

Everything here works in image pixel coordinates: x grows to the right and y
grows downward, exactly as in the video frame and in DeepLabCut's output.

Three questions are answered:

* which side of a line a point is on, and when a path crosses a line segment
  (including which way it was travelling);
* whether a point is inside a polygon, and when a path enters or leaves it;
* small helpers for shapes (ellipse outline, simplifying a freehand outline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DIRECTIONS = ("any", "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top")

DIRECTION_LABELS = {
    "any": "in either direction",
    "left_to_right": "left to right",
    "right_to_left": "right to left",
    "top_to_bottom": "top to bottom",
    "bottom_to_top": "bottom to top",
}


# --------------------------------------------------------------------------- lines


def line_frame(a, b):
    """Return ``(length, u, n)`` for the segment a -> b.

    ``u`` is the unit vector along the line and ``n`` the unit normal
    (``u`` rotated by a quarter turn). Side +1 of the line is the side that
    ``n`` points to.
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-9:
        raise ValueError("A line needs two different end points.")
    ux, uy = dx / length, dy / length
    return length, (ux, uy), (-uy, ux)


def signed_distance(xy: np.ndarray, a, b) -> np.ndarray:
    """Signed distance (pixels) of each point from the infinite line a-b."""
    _, _, n = line_frame(a, b)
    return (xy[:, 0] - float(a[0])) * n[0] + (xy[:, 1] - float(a[1])) * n[1]


def crossing_vector(a, b, toward: int) -> tuple[float, float]:
    """Direction of travel for a crossing that ends on side ``toward`` (+1/-1)."""
    _, _, n = line_frame(a, b)
    return (toward * n[0], toward * n[1])


def movement_matches(direction: str, mx: float, my: float, eps: float = 1e-9) -> bool:
    """Does a movement vector count as the named screen direction?"""
    if direction == "any":
        return True
    if direction == "left_to_right":
        return mx > eps
    if direction == "right_to_left":
        return mx < -eps
    if direction == "top_to_bottom":
        return my > eps
    if direction == "bottom_to_top":
        return my < -eps
    raise ValueError(f"Unknown direction: {direction!r}")


def meaningful_directions(a, b, min_component: float = 0.2) -> list[str]:
    """Screen directions that make sense for this line.

    A vertical line can be crossed left to right or right to left, but "top to
    bottom" is meaningless for it. A diagonal line supports all four names.
    """
    try:
        _, _, n = line_frame(a, b)
    except ValueError:
        return ["any"]
    out = ["any"]
    if abs(n[0]) >= min_component:
        out += ["left_to_right", "right_to_left"]
    if abs(n[1]) >= min_component:
        out += ["top_to_bottom", "bottom_to_top"]
    return out


@dataclass
class Crossing:
    """One crossing of a line segment by a tracked point."""

    frame: int  # first frame on the new side
    prev_frame: int  # last tracked frame on the old side
    toward: int  # +1 if it ended on the side the normal points to, else -1
    where: float  # position along the segment, 0 = first end point, 1 = second
    gap: int  # untracked frames between prev_frame and frame


def find_line_crossings(xy: np.ndarray, a, b, margin: float = 0.0) -> list[Crossing]:
    """Find every time a path crosses the segment a-b.

    ``xy`` is an (N, 2) array of positions per frame with NaN where the point is
    not tracked.

    A crossing is only registered once the point is more than ``margin`` pixels
    past the line on the new side, having earlier been more than ``margin``
    pixels on the other side. This stops a point that jitters on top of the line
    from being counted many times. The reported frame is the first frame on the
    new side during that final pass over the line.

    The path has to pass through the drawn segment itself. Going around the end
    of the line changes side without being counted.
    """
    xy = np.asarray(xy, dtype=float)
    n_frames = len(xy)
    if n_frames < 2:
        return []
    length, u, n = line_frame(a, b)
    ax, ay = float(a[0]), float(a[1])
    margin = max(0.0, float(margin))

    valid = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    if valid.sum() < 2:
        return []
    d = np.full(n_frames, np.nan)
    d[valid] = (xy[valid, 0] - ax) * n[0] + (xy[valid, 1] - ay) * n[1]

    firm = np.zeros(n_frames, dtype=np.int8)
    with np.errstate(invalid="ignore"):
        firm[d > margin] = 1
        firm[d < -margin] = -1
    firm_idx = np.flatnonzero(firm)
    if len(firm_idx) < 2:
        return []
    sides = firm[firm_idx]
    change = np.flatnonzero(sides[1:] != sides[:-1]) + 1

    out: list[Crossing] = []
    for t in change:
        i_prev = int(firm_idx[t - 1])
        i_cur = int(firm_idx[t])
        new_side = int(sides[t])
        window = np.arange(i_prev, i_cur + 1)
        window = window[valid[i_prev : i_cur + 1]]
        dv = d[window] * new_side
        positive = dv > 0
        # last place where the path steps from "not yet over" to "over"
        steps = np.flatnonzero(positive[1:] & ~positive[:-1]) + 1
        if len(steps) == 0:  # cannot happen, but never crash a batch over it
            continue
        q = int(steps[-1])
        j, k = int(window[q - 1]), int(window[q])
        d_before, d_after = float(dv[q - 1]), float(dv[q])
        alpha = d_before / (d_before - d_after) if d_after != d_before else 0.0
        px = xy[j, 0] + alpha * (xy[k, 0] - xy[j, 0])
        py = xy[j, 1] + alpha * (xy[k, 1] - xy[j, 1])
        where = ((px - ax) * u[0] + (py - ay) * u[1]) / length
        if -1e-9 <= where <= 1 + 1e-9:
            out.append(Crossing(frame=k, prev_frame=j, toward=new_side, where=float(where), gap=k - j - 1))
    return out


# ------------------------------------------------------------------------ polygons


def points_in_polygon(xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Boolean array: is each point inside the polygon? NaN points are False."""
    xy = np.asarray(xy, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    if len(poly) < 3:
        return inside
    xj, yj = poly[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        for xi, yi in poly:
            if yi != yj:
                crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
                inside ^= crosses
            xj, yj = xi, yi
    inside[~(np.isfinite(x) & np.isfinite(y))] = False
    return inside


def distance_to_polygon_edge(xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Distance (pixels) from each point to the nearest edge of the polygon."""
    xy = np.asarray(xy, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    best = np.full(len(xy), np.inf)
    if len(poly) < 2:
        return best
    x, y = xy[:, 0], xy[:, 1]
    x1, y1 = poly[-1]
    with np.errstate(invalid="ignore"):
        for x2, y2 in poly:
            ex, ey = x2 - x1, y2 - y1
            len2 = ex * ex + ey * ey
            if len2 < 1e-18:
                dist = np.hypot(x - x1, y - y1)
            else:
                t = np.clip(((x - x1) * ex + (y - y1) * ey) / len2, 0.0, 1.0)
                dist = np.hypot(x - (x1 + t * ex), y - (y1 + t * ey))
            best = np.fmin(best, dist)
            x1, y1 = x2, y2
    best[~(np.isfinite(x) & np.isfinite(y))] = np.nan
    return best


def inradius_estimate(polygon, grid: int = 40) -> float:
    """Roughly how far the deepest interior point is from the edge (pixels)."""
    poly = np.asarray(polygon, dtype=float)
    if len(poly) < 3:
        return 0.0
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    if x1 - x0 < 1e-9 or y1 - y0 < 1e-9:
        return 0.0
    gx, gy = np.meshgrid(np.linspace(x0, x1, grid), np.linspace(y0, y1, grid))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inside = points_in_polygon(pts, poly)
    if not inside.any():
        return 0.0
    return float(np.nanmax(distance_to_polygon_edge(pts[inside], poly)))


@dataclass
class RegionTrace:
    """Frame-by-frame record of one tracked point relative to one region."""

    state: np.ndarray  # int8 per frame: 1 inside, 0 outside, -1 not known yet
    valid: np.ndarray  # bool per frame: the point was tracked on this frame
    entries: np.ndarray  # frames at which the point entered
    exits: np.ndarray  # frames at which the point left
    entry_gaps: np.ndarray  # untracked frames just before each entry
    exit_gaps: np.ndarray  # untracked frames just before each exit
    first_known: int  # first tracked frame, or -1 if never tracked


def trace_region(xy: np.ndarray, polygon: np.ndarray, margin: float = 0.0) -> RegionTrace:
    """Work out, frame by frame, whether a point is inside a region.

    The same ``margin`` idea as for lines applies: once inside, the point has
    to move more than ``margin`` pixels outside the edge before it counts as
    having left, and the other way round. While the point is not tracked the
    last known answer is kept.
    """
    xy = np.asarray(xy, dtype=float)
    n_frames = len(xy)
    valid = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1]) if n_frames else np.zeros(0, dtype=bool)
    empty = np.zeros(0, dtype=np.int64)
    state = np.full(n_frames, -1, dtype=np.int8)
    if n_frames == 0 or not valid.any() or len(polygon) < 3:
        return RegionTrace(state, valid, empty, empty, empty, empty, -1)

    inside = points_in_polygon(xy, polygon)
    margin = max(0.0, float(margin))
    code = np.full(n_frames, -1, dtype=np.int8)
    if margin > 0:
        # A small or narrow region may be thinner than the margin. Shrink the
        # inner margin so that it is always possible to be "clearly inside".
        inner = min(margin, 0.5 * inradius_estimate(polygon))
        dist = distance_to_polygon_edge(xy, polygon)
        with np.errstate(invalid="ignore"):
            code[valid & inside & (dist > inner)] = 1
            code[valid & ~inside & (dist > margin)] = 0
    else:
        code[valid & inside] = 1
        code[valid & ~inside] = 0

    first = int(np.flatnonzero(valid)[0])
    if code[first] < 0:  # starts inside the margin band: take the plain answer
        code[first] = 1 if inside[first] else 0

    known = code >= 0
    last_known = np.where(known, np.arange(n_frames), 0)
    np.maximum.accumulate(last_known, out=last_known)
    state = code[last_known]
    state[:first] = -1

    prev = state[:-1]
    cur = state[1:]
    entries = np.flatnonzero((prev == 0) & (cur == 1)) + 1
    exits = np.flatnonzero((prev == 1) & (cur == 0)) + 1

    valid_idx = np.where(valid, np.arange(n_frames), -1)
    np.maximum.accumulate(valid_idx, out=valid_idx)

    def gaps_before(frames: np.ndarray) -> np.ndarray:
        if len(frames) == 0:
            return empty
        before = valid_idx[frames - 1]
        return np.where(before >= 0, frames - before - 1, 0).astype(np.int64)

    return RegionTrace(state, valid, entries, exits, gaps_before(entries), gaps_before(exits), first)


# -------------------------------------------------------------------------- shapes


def ellipse_polygon(p0, p1, segments: int = 72) -> np.ndarray:
    """Outline of the ellipse that fits the box with opposite corners p0, p1."""
    cx, cy = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
    rx, ry = abs(p1[0] - p0[0]) / 2.0, abs(p1[1] - p0[1]) / 2.0
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    return np.column_stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)])


def polygon_centroid(polygon) -> tuple[float, float]:
    poly = np.asarray(polygon, dtype=float)
    if len(poly) == 0:
        return (0.0, 0.0)
    if len(poly) < 3:
        return (float(poly[:, 0].mean()), float(poly[:, 1].mean()))
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return (float(x.mean()), float(y.mean()))
    return (float(((x + xn) * cross).sum() / (6.0 * area)), float(((y + yn) * cross).sum() / (6.0 * area)))


def polygon_area(polygon) -> float:
    poly = np.asarray(polygon, dtype=float)
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) / 2.0)


def simplify_path(points, tolerance: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker: drop points that add less than ``tolerance``."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < 3 or tolerance <= 0:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        x1, y1 = pts[first]
        x2, y2 = pts[last]
        ex, ey = x2 - x1, y2 - y1
        len2 = ex * ex + ey * ey
        worst, worst_i = -1.0, -1
        for i in range(first + 1, last):
            px, py = pts[i]
            if len2 < 1e-18:
                dist = math.hypot(px - x1, py - y1)
            else:
                t = max(0.0, min(1.0, ((px - x1) * ex + (py - y1) * ey) / len2))
                dist = math.hypot(px - (x1 + t * ex), py - (y1 + t * ey))
            if dist > worst:
                worst, worst_i = dist, i
        if worst > tolerance:
            keep[worst_i] = True
            stack.append((first, worst_i))
            stack.append((worst_i, last))
    return [p for p, k in zip(pts, keep) if k]


def distance_point_to_segment(p, a, b) -> float:
    px, py = float(p[0]), float(p[1])
    x1, y1 = float(a[0]), float(a[1])
    x2, y2 = float(b[0]), float(b[1])
    ex, ey = x2 - x1, y2 - y1
    len2 = ex * ex + ey * ey
    if len2 < 1e-18:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * ex + (py - y1) * ey) / len2))
    return math.hypot(px - (x1 + t * ex), py - (y1 + t * ey))
