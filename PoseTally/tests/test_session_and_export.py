import json
import math
import os

import numpy as np
import pytest
from openpyxl import load_workbook

from posetally.demo import make_demo
from posetally.export import describe_rule, export_csv, export_xlsx
from posetally.model import FormulaColumn, Layout, LineShape, Rule
from posetally.session import SESSION_FILE, SESSION_FOLDER, Session


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    folder = tmp_path_factory.mktemp("demo")
    return make_demo(str(folder))


def scores_by_id(table):
    return {(row.video_id, row.animal): row.values for row in table.rows}


def test_demo_maze_gives_the_known_answers(demo):
    session = Session()
    session.open_project(demo["single"])
    assert [item.name for item in session.items] == ["trial_01_groupA", "trial_02_groupA", "trial_03_groupB"]
    assert all(item.tracked for item in session.items)
    assert session.score_all()
    table = session.table()
    assert table.show_animal is False and table.id_header == "Trial"
    got = scores_by_id(table)
    # trial 1 goes left and stays
    assert got[("trial_01", "")]["Left entries"] == 1 and got[("trial_01", "")]["Right entries"] == 0
    assert got[("trial_01", "")]["Chose left"] == 1 and got[("trial_01", "")]["Chose right"] == 0
    # trial 2 goes right first, then left
    assert got[("trial_02", "")]["Left entries"] == 1 and got[("trial_02", "")]["Right entries"] == 1
    assert got[("trial_02", "")]["Chose left"] == 0 and got[("trial_02", "")]["Chose right"] == 1
    # trial 3 hesitates towards the left without crossing, then goes right
    assert got[("trial_03", "")]["Left entries"] == 0 and got[("trial_03", "")]["Chose right"] == 1
    assert got[("trial_02", "")]["Arm entries (total)"] == 2
    # the animal waits 30 frames at 20 frames per second before setting off, plus the time to reach the line
    assert 1.5 < got[("trial_01", "")]["Time to leave start (s)"] < 3.0
    assert got[("trial_03", "")]["Time in start box (s)"] > got[("trial_01", "")]["Time in start box (s)"]
    assert got[("trial_01", "")]["Left goal visits"] == 1 and got[("trial_03", "")]["Left goal visits"] == 0


def test_demo_two_animals_are_tallied_separately(demo):
    session = Session()
    session.open_project(demo["multi"])
    session.score_all()
    table = session.table()
    assert table.show_animal is True
    got = scores_by_id(table)
    assert got[("arena_01", "animal1")]["Midline crossings"] == 4  # two laps
    assert got[("arena_01", "animal2")]["Midline crossings"] == 6  # three laps
    assert got[("arena_01", "animal1")]["Time in centre (s)"] == 0
    assert got[("arena_01", "animal2")]["Centre entries"] == 6


def test_workbook_layout(demo, tmp_path):
    session = Session()
    session.open_project(demo["single"])
    session.score_all()
    out = tmp_path / "scores.xlsx"
    export_xlsx(str(out), session.ordered_results(), session.layout, {"DeepLabCut project": "MazeDemo"})
    wb = load_workbook(out)
    assert wb.sheetnames == ["Scores", "Events", "Quality", "Setup"]
    ws = wb["Scores"]
    header = [c.value for c in ws[1]]
    assert header[0] == "Trial"  # the file label is the left-most column
    assert header[1:] == session.layout.columns()
    rows = {r[0]: dict(zip(header, r)) for r in ws.iter_rows(min_row=2, values_only=True)}
    assert set(rows) == {"trial_01", "trial_02", "trial_03"}
    assert rows["trial_02"]["Right entries"] == 1 and rows["trial_02"]["Arm entries (total)"] == 2
    assert isinstance(rows["trial_02"]["Right entries"], int)
    assert isinstance(rows["trial_02"]["Time in start box (s)"], float)
    events = list(wb["Events"].iter_rows(min_row=2, values_only=True))
    assert any(e[3] == "Right arm" and e[5] == "crossed left to right" and e[8] == 1 for e in events)
    setup_text = [str(c) for row in wb["Setup"].iter_rows(values_only=True) for c in row if c is not None]
    assert "[Left entries] + [Right entries]" in setup_text and "Likelihood cut-off" in setup_text
    quality = list(wb["Quality"].iter_rows(values_only=True))
    assert quality[0][:4] == ("Trial", "Frames", "Frames per second", "Length (s)")
    assert quality[1][2] == 20


def test_workbook_with_animals_untracked_videos_and_empty_cells(demo, tmp_path):
    session = Session()
    session.open_project(demo["multi"])
    # a video that has not been tracked, and a rule whose answer can be empty
    untracked = tmp_path / "not_tracked_yet.mp4"
    untracked.write_bytes(open(session.items[0].video_path, "rb").read())
    session.add_videos([str(untracked)])
    session.layout.lines[0].rules.append(
        Rule(bodypart="head", measure="first_time", direction="top_to_bottom", column="Never happens (s)")
    )
    session.layout.formulas.append(FormulaColumn("Uses empty", "[Never happens (s)] + 1"))
    session.layout.formulas.append(FormulaColumn("Broken", "[No such column] * 2"))
    session.score_all()
    out = tmp_path / "multi.xlsx"
    table = export_xlsx(str(out), session.ordered_results(), session.layout)
    assert "Broken" in table.formula_errors
    ws = load_workbook(out)["Scores"]
    rows = list(ws.iter_rows(values_only=True))
    assert rows[0][:2] == ("File", "Animal")
    by_key = {(r[0], r[1]): dict(zip(rows[0], r)) for r in rows[1:]}
    assert set(by_key) == {("arena_01", "animal1"), ("arena_01", "animal2"), ("not_tracked_yet", None)}
    assert by_key[("arena_01", "animal1")]["Never happens (s)"] is None
    assert by_key[("arena_01", "animal1")]["Uses empty"] is None
    assert by_key[("not_tracked_yet", None)]["Left to right"] is None
    quality = list(load_workbook(out)["Quality"].iter_rows(values_only=True))
    assert any(r[0] == "not_tracked_yet" and "Not tracked yet" in str(r) for r in quality)
    # plain CSV of the scores table
    export_csv(str(tmp_path / "multi.csv"), session.ordered_results(), session.layout)
    text = (tmp_path / "multi.csv").read_text(encoding="utf-8-sig").splitlines()
    assert text[0].startswith("File,Animal,Left to right") and len(text) == 4


def test_label_that_starts_with_equals_stays_text(tmp_path):
    from posetally.batch import VideoResult
    from posetally.scoring import AnimalResult

    layout = Layout()
    layout.lines = [LineShape(name="L", p1=(0, 0), p2=(1, 1), rules=[Rule(bodypart="a", column="Score")])]
    result = VideoResult(key="k", name="=SUM(1,2)", source="/x/=SUM(1,2).mp4", fps=10, n_frames=5)
    result.animals = [AnimalResult(animal="", scores={"Score": 2.0})]
    out = tmp_path / "eq.xlsx"
    export_xlsx(str(out), [result], layout)
    cell = load_workbook(out)["Scores"]["A2"]
    assert cell.value == "=SUM(1,2)" and cell.data_type == "s"


def test_setup_is_saved_with_the_project_and_restored(demo, tmp_path):
    session = Session()
    session.open_project(demo["single"])
    session.layout.lines[0].name = "Renamed line"
    session.layout.settings.margin_px = 7.5
    session.track_options["save_csv"] = True
    saved = session.save()
    assert saved == os.path.join(os.path.dirname(demo["single"]), SESSION_FOLDER, SESSION_FILE)
    stored = json.load(open(saved))
    assert stored["videos"][0] == {"path": "videos/trial_01_groupA.mp4", "tracks_only": False}  # relative: survives moving the folder

    again = Session()
    again.open_project(demo["single"])
    assert again.layout.lines[0].name == "Renamed line" and again.layout.settings.margin_px == 7.5
    assert again.track_options["save_csv"] is True and len(again.items) == 3

    # sharing a setup between projects
    shared = tmp_path / "setup.json"
    session.export_setup(str(shared))
    other = Session()
    other.open_project(demo["multi"])
    other.import_setup(str(shared))
    assert other.layout.lines[0].name == "Renamed line"
    not_a_setup = tmp_path / "something_else.json"
    not_a_setup.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(ValueError):
        other.import_setup(str(not_a_setup))


def test_adding_a_folder_finds_videos_tracking_and_archived_results(demo, tmp_path):
    session = Session()
    session.open_project(demo["multi"])
    session.remove_items([item.key for item in session.items])
    maze_videos = os.path.join(os.path.dirname(demo["single"]), "videos")
    added, orphans = session.add_folder(os.path.dirname(demo["single"]))
    assert (added, orphans) == (3, 0)
    assert session.add_folder(maze_videos) == (0, 0)  # nothing is added twice
    # tracking files whose videos have been archived elsewhere still count
    archive = tmp_path / "archive"
    archive.mkdir()
    for name in os.listdir(maze_videos):
        if name.endswith((".h5", ".pickle")):
            (archive / name).write_bytes(open(os.path.join(maze_videos, name), "rb").read())
    added, orphans = session.add_folder(str(archive))
    assert (added, orphans) == (0, 3)
    only_tracks = [item for item in session.items if item.video_path is None]
    assert sorted(item.name for item in only_tracks) == ["trial_01_groupA", "trial_02_groupA", "trial_03_groupB"]
    session.layout = __import__("posetally.demo", fromlist=["maze_layout"]).maze_layout()
    session.score_all()
    values = [row.values["Right entries"] for row in session.table().rows]
    assert values == [0, 0, 1, 1, 1, 1]  # each trial twice: once from the video folder, once from the archive
    assert session.items_to_track() == []


def test_results_know_when_they_are_out_of_date(demo):
    session = Session()
    session.open_project(demo["single"])
    assert not session.results_are_current()
    session.score_all()
    assert session.results_are_current()
    session.layout.lines[0].rules[0].points = 5
    assert not session.results_are_current()
    session.score_all()
    assert session.table().rows[0].values["Left entries"] == 5


def test_rule_sentences():
    assert describe_rule(Rule(bodypart="snout", direction="left_to_right", points=2, column="A"), "line") == (
        "snout crosses left to right: add 2 to A"
    )
    assert "first choice in group “arm”" in describe_rule(
        Rule(bodypart="snout", column="A", count_mode="first_in_group", group="arm"), "line"
    )
    assert describe_rule(Rule(bodypart="body", measure="time", column="T"), "roi") == "Seconds body spends inside → T"


def test_column_bookkeeping():
    layout = Layout()
    line = LineShape(name="L", p1=(0, 0), p2=(0, 10))
    line.rules = [Rule(bodypart="a", column="One"), Rule(bodypart="a", column="Two"), Rule(bodypart="a", column="One")]
    layout.lines = [line]
    layout.formulas = [FormulaColumn("Sum", "[One] + [Two]")]
    assert layout.columns() == ["One", "Two", "Sum"]
    layout.column_order = ["Sum", "Two"]
    assert layout.columns() == ["Sum", "Two", "One"]
    layout.rename_column("One", "First")
    assert layout.columns() == ["Sum", "Two", "First"] and layout.formulas[0].expression == "[First] + [Two]"
    assert layout.unique_column_name("Two") == "Two 2"
    assert layout.problems(["a"]) == []
    line.rules[1].column = ""
    assert any("no column name" in p for p in layout.problems(["a"]))
    assert any("not tracked" in p for p in layout.problems(["b"]))
    copy = Layout.from_dict(json.loads(json.dumps(layout.to_dict())))
    assert copy.to_dict() == layout.to_dict()
    assert math.isclose(copy.settings.pcutoff, 0.6)
    layout.remove_shape(line.id)
    assert layout.columns() == ["Sum"] and np.isfinite(len(layout.shapes()))
