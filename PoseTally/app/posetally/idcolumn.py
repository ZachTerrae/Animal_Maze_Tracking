"""The first column of the results: a label for each video built from its file name."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass

ID_MODES = ("name", "name_ext", "folder_name", "pieces", "between", "regex")

ID_MODE_LABELS = {
    "name": "File name",
    "name_ext": "File name with extension",
    "folder_name": "Folder and file name",
    "pieces": "Part of the file name (split it into pieces)",
    "between": "Text between two markers in the file name",
    "regex": "Regular expression (advanced)",
}


@dataclass
class IdColumn:
    header: str = "File"
    mode: str = "name"
    delimiters: str = "_"  # characters that separate the pieces
    pieces: str = "1"  # which pieces to keep: "1", "2-3", "1,4", "last"
    after: str = ""  # keep the text after this marker ...
    before: str = ""  # ... and before this one
    pattern: str = ""  # regular expression; group 1 is used if there is one

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> IdColumn:
        out = IdColumn()
        for key, value in (d or {}).items():
            if hasattr(out, key) and isinstance(value, str):
                setattr(out, key, value)
        if out.mode not in ID_MODES:
            out.mode = "name"
        return out


def _parse_piece_spec(spec: str, count: int) -> list[int]:
    """Turn "1,3-4,last" into zero-based indices that exist."""
    picked: list[int] = []

    def resolve(token: str) -> int | None:
        token = token.strip().lower()
        if token == "last":
            return count - 1
        if re.fullmatch(r"-?\d+", token):
            number = int(token)
            index = number - 1 if number > 0 else count + number
            return index
        return None

    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(last|\d+)\s*-\s*(last|\d+)", part.lower())
        if match:
            start, end = resolve(match.group(1)), resolve(match.group(2))
            if start is None or end is None:
                continue
            step = 1 if end >= start else -1
            picked.extend(range(start, end + step, step))
        else:
            index = resolve(part)
            if index is not None:
                picked.append(index)
    return [i for i in picked if 0 <= i < count]


def make_id(cfg: IdColumn, path_or_name: str) -> str:
    """Build the label for one video. Never returns an empty string."""
    text = str(path_or_name).replace("\\", "/")
    base = os.path.basename(text)
    stem = os.path.splitext(base)[0]
    if not stem:  # e.g. ".hidden"
        stem = base
    mode = cfg.mode

    if mode == "name_ext":
        return base
    if mode == "folder_name":
        parent = os.path.basename(os.path.dirname(text))
        return f"{parent}/{stem}" if parent else stem
    if mode == "pieces":
        delims = cfg.delimiters or "_"
        parts = [p for p in re.split("[" + re.escape(delims) + "]", stem) if p != ""]
        chosen = _parse_piece_spec(cfg.pieces or "1", len(parts))
        if chosen:
            return delims[0].join(parts[i] for i in chosen)
        return stem
    if mode == "between":
        result = stem
        if cfg.after:
            pos = result.find(cfg.after)
            if pos < 0:
                return stem
            result = result[pos + len(cfg.after) :]
        if cfg.before:
            pos = result.find(cfg.before)
            if pos < 0:
                return stem
            result = result[:pos]
        return result or stem
    if mode == "regex" and cfg.pattern:
        try:
            match = re.search(cfg.pattern, stem)
        except re.error:
            return stem
        if match:
            value = match.group(1) if match.groups() else match.group(0)
            return value or stem
        return stem
    return stem


def check_pattern(pattern: str) -> str | None:
    """Return an error message if the regular expression is not usable."""
    try:
        re.compile(pattern)
    except re.error as exc:
        return str(exc)
    return None
