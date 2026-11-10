"""Drives the real window (without a screen) the way a person would: draws shapes with the
mouse, edits rules in the panel, scores, exports, and runs tracking against the stand-in
for DeepLabCut in tests/fake_dlc."""

import os
import shutil
import time

import pytest
from openpyxl import load_workbook

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtTest, QtWidgets  # noqa: E402

from posetally.demo import make_demo  # noqa: E402
from posetally.session import SESSION_FILE, SESSION_FOLDER  # noqa: E402
from posetally.ui import dialogs  # noqa: E402
from posetally.ui.columnspanel import FormulaDialog, IdColumnDialog  # noqa: E402
from posetally.ui.mainwindow import MainWindow  # noqa: E402
from posetally.ui.trackdialog import TrackDialog  # noqa: E402

Qt = QtCore.Qt
HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "fake_dlc")


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    os.environ["HOME"] = str(home)
    os.environ["USERPROFILE"] = str(home)
    os.environ["XDG_CONFIG_HOME"] = str(home / "config")
    dialogs.QUIET = []
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield application


def pump(ms: int = 60) -> None:
    end = time.time() + ms / 1000.0
    while time.time() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.005)


@pytest.fixture()
def window(app, tmp_path):
    """The window with the maze demo open and no shapes drawn yet."""
    configs = make_demo(str(tmp_path / "demo"))
    os.remove(os.path.join(os.path.dirname(configs["single"]), SESSION_FOLDER, SESSION_FILE))
    dialogs.QUIET.clear()
    w = MainWindow()
    w.resize(1500, 950)
    w.show()
    assert w.open_project(configs["single"])
    w.add_folder_path(os.path.join(os.path.dirname(configs["single"]), "videos"))
    pump(150)
    w.demo = configs
    yield w
    w._save_timer.stop()
    w.close()
    pump(30)


def at(w, x, y) -> QtCore.QPoint:
    """Where a video pixel is on the screen."""
    return w.canvas.mapFromScene(QtCore.QPointF(x, y))


def drag(w, start, end, steps: int = 6) -> None:
    viewport = w.canvas.viewport()
    QtTest.QTest.mousePress(viewport, Qt.LeftButton, Qt.NoModifier, at(w, *start))
    for i in range(1, steps + 1):
        t = i / steps
        QtTest.QTest.mouseMove(viewport, at(w, start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
    QtTest.QTest.mouseRelease(viewport, Qt.LeftButton, Qt.NoModifier, at(w, *end))
    pump(40)


def click(w, x, y) -> None:
    QtTest.QTest.mouseClick(w.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, at(w, x, y))
    pump(30)


def near(a, b, tolerance=2.5) -> bool:
    return abs(a[0] - b[0]) <= tolerance and abs(a[1] - b[1]) <= tolerance


def test_window_opens_a_project_and_lists_videos(window):
    w = window
    assert w.stack.currentWidget() is w.video_pane
    assert w.video_list.tree.topLevelItemCount() == 3
    assert w.current_item().name == "trial_01_groupA"
    assert w.canvas.frame_size() == (640, 480)
    assert w.shape_panel.bodyparts == ["nose", "head", "body", "tailbase"]
    assert "3 tracked" in w.video_list.summary.text()
    assert dialogs.QUIET == []


def test_draw_a_line_and_give_it_rules(window):
    w = window
    w.video_pane.set_tool("line")
    drag(w, (240, 70), (240, 150))
    layout = w.session.layout
    assert len(layout.lines) == 1
    line = layout.lines[0]
    assert near(line.p1, (240, 70)) and near(line.p2, (240, 150))
    assert w.canvas.tool() == "select" and w.canvas.selected_id() == line.id
    assert w.shape_panel.selected_id == line.id
    assert layout.ref_size == (640, 480)
    # it starts with one rule, ready to edit
    (rule,) = line.rules
    assert rule.bodypart == "nose" and rule.measure == "count" and rule.column == "Line 1 crossings"
    (card,) = w.shape_panel._cards
    # a vertical line offers left/right directions only
    offered = [card.direction.itemData(i) for i in range(card.direction.count())]
    assert offered == ["any", "left_to_right", "right_to_left"]
    card.bodypart.setCurrentIndex(card.bodypart.findData("head"))
    card.direction.setCurrentIndex(card.direction.findData("right_to_left"))
    card.column.setCurrentText("Left entries")
    card.column.lineEdit().editingFinished.emit()
    card.points.setValue(2)
    assert (rule.bodypart, rule.direction, rule.column, rule.points) == ("head", "right_to_left", "Left entries", 2.0)
    # "+" adds a second score to the same line
    w.shape_panel.add_button.click()
    pump(30)
    assert len(line.rules) == 2 and len(w.shape_panel._cards) == 2
    second = w.shape_panel._cards[1]
    second.measure.setCurrentIndex(second.measure.findData("first_time"))
    assert line.rules[1].measure == "first_time" and line.rules[1].column == "Line 1 first crossing (s)"
    assert not second.form.isRowVisible(second.points)  # no points for a time measurement
    # renaming the shape renames the columns that still carry the automatic name
    w.shape_panel.name_edit.setText("Left arm")
    w.shape_panel.name_edit.editingFinished.emit()
    assert line.name == "Left arm" and line.rules[1].column == "Left arm first crossing (s)"
    assert layout.columns() == ["Left entries", "Left arm first crossing (s)"]
    # the video being shown is scored straight away
    w._rescore_current()
    assert w._live.animals[0].scores["Left entries"] == 2.0  # one crossing worth 2 points
    assert w.video_pane.timeline.marks()  # with a tick on the timeline
    w.score_all(blocking=True)
    values = {row.video_id: row.values["Left entries"] for row in w.session.table().rows}
    assert values == {"trial_01_groupA": 2.0, "trial_02_groupA": 2.0, "trial_03_groupB": 0.0}


def test_draw_every_kind_of_region(window):
    w = window
    layout = w.session.layout
    w.video_pane.set_tool("rectangle")
    drag(w, (270, 380), (370, 465))
    w.video_pane.set_tool("ellipse")
    drag(w, (50, 60), (150, 160))
    w.video_pane.set_tool("polygon")
    for corner in ((490, 60), (590, 60), (590, 160), (490, 160)):
        click(w, *corner)
    assert w.canvas.is_drawing()
    QtTest.QTest.keyClick(w.canvas, Qt.Key_Return)
    pump(30)
    w.video_pane.set_tool("freehand")
    viewport = w.canvas.viewport()
    loop = [(300, 200), (340, 200), (345, 240), (340, 280), (300, 280), (295, 240)]
    QtTest.QTest.mousePress(viewport, Qt.LeftButton, Qt.NoModifier, at(w, *loop[0]))
    for a, b in zip(loop, loop[1:] + loop[:1]):
        for i in range(1, 9):
            QtTest.QTest.mouseMove(viewport, at(w, a[0] + (b[0] - a[0]) * i / 8, a[1] + (b[1] - a[1]) * i / 8))
    QtTest.QTest.mouseRelease(viewport, Qt.LeftButton, Qt.NoModifier, at(w, *loop[0]))
    pump(40)

    assert [r.form for r in layout.rois] == ["polygon", "ellipse", "polygon", "polygon"]
    assert [r.name for r in layout.rois] == ["Region 1", "Region 2", "Region 3", "Region 4"]
    box, oval, quad, blob = layout.rois
    assert len(box.points) == 4 and near(box.points[0], (270, 380)) and near(box.points[2], (370, 465))
    assert near(oval.points[0], (50, 60)) and near(oval.points[1], (150, 160))
    assert len(quad.points) == 4
    assert 4 <= len(blob.points) <= 20  # a freehand outline is simplified, not kept point by point
    assert all(len(r.rules) == 1 and r.rules[0].measure == "time" for r in layout.rois)
    assert len({r.color for r in layout.rois}) == 4
    # a second score on a region offers the next kind of measure
    w._select_shape(box.id)
    w.shape_panel.add_button.click()
    assert [r.measure for r in box.rules] == ["time", "entries"]
    assert box.rules[1].column == "Region 1 entries"
    w._rescore_current()
    scores = w._live.animals[0].scores
    assert scores["Region 1 time (s)"] > 1.0 and scores["Region 2 time (s)"] > 1.0 and scores["Region 3 time (s)"] == 0.0


def test_select_move_reshape_delete(window):
    w = window
    w.video_pane.set_tool("line")
    drag(w, (400, 70), (400, 150))
    line = w.session.layout.lines[0]
    click(w, 100, 300)  # empty space: nothing selected
    assert w.canvas.selected_id() is None and w.shape_panel.selected_id is None
    click(w, 400, 110)  # on the line
    assert w.canvas.selected_id() == line.id
    drag(w, (400, 110), (430, 110))  # move the whole line
    assert near(line.p1, (430, 70)) and near(line.p2, (430, 150))
    drag(w, (430, 150), (430, 200))  # pull the lower end down
    assert near(line.p1, (430, 70)) and near(line.p2, (430, 200))
    before = line.p1
    QtTest.QTest.keyClick(w.canvas, Qt.Key_Right, Qt.ShiftModifier)
    assert line.p1[0] == pytest.approx(before[0] + 10) and line.p1[1] == pytest.approx(before[1])
    # a tilted line offers all four directions
    drag(w, (440, 200), (520, 150))
    pump(30)
    card = w.shape_panel._cards[0]
    assert card.direction.count() == 5
    QtTest.QTest.keyClick(w.canvas, Qt.Key_Delete)
    pump(30)
    assert w.session.layout.lines == [] and w.session.layout.columns() == []
    assert any(kind == "ask" for kind, _title, _text in dialogs.QUIET)


def test_one_video_can_have_its_own_positions(window):
    w = window
    w.video_pane.set_tool("line")
    drag(w, (240, 70), (240, 150))
    line = w.session.layout.lines[0]
    card = w.shape_panel._cards[0]
    card.bodypart.setCurrentIndex(card.bodypart.findData("head"))
    w.video_pane.positions_mode.setCurrentIndex(1)  # "only this video"
    drag(w, (240, 110), (140, 110))
    layout = w.session.layout
    assert near(line.p1, (240, 70))  # the shared position did not move
    assert layout.has_override("trial_01_groupA", line.id)
    assert w.shape_panel.own_position.isVisibleTo(w.shape_panel)
    w.score_all(blocking=True)
    table = {row.video_id: row for row in w.session.table().rows}
    assert "own shape positions" in table["trial_01_groupA"].notes
    assert table["trial_02_groupA"].notes == ""
    # the next video shows the shared position, and edits go back to "every video"
    w.video_list.select_key(w.session.items[1].key)
    pump(60)
    assert w.video_pane.positions_mode.currentIndex() == 0
    assert near(w.canvas.shape_points(line.id)[0], (240, 70))
    w.video_list.select_key(w.session.items[0].key)
    pump(60)
    assert near(w.canvas.shape_points(line.id)[0], (140, 70))
    w._use_shared_position(line.id)
    assert not layout.overrides and near(w.canvas.shape_points(line.id)[0], (240, 70))


def test_zoom_pan_and_playback(window):
    w = window
    before = w.canvas.view_scale()
    w.canvas.zoom(2.0, QtCore.QPointF(at(w, 320, 240)))
    assert w.canvas.view_scale() == pytest.approx(before * 2.0)
    centre = at(w, 320, 240)
    assert abs(centre.x() - w.canvas.viewport().width() / 2) < 40  # zooming keeps the point under the cursor in place
    w.canvas.fit()
    assert w.canvas.view_scale() == pytest.approx(before, rel=0.02)
    w.video_pane.seek(50)
    assert w.video_pane.current_frame() == 50 and "frame 51 of" in w.video_pane.time_label.text()
    w.video_pane.speed.setCurrentIndex(5)
    w.video_pane.play()
    pump(250)
    w.video_pane.pause()
    assert w.video_pane.current_frame() > 50
    w.video_pane.seek(10_000)
    assert w.video_pane.current_frame() == w.video_pane._n_frames - 1


def test_columns_formulas_and_export(window, tmp_path):
    w = window
    for x, name in ((240, "Left"), (400, "Right")):
        w.video_pane.set_tool("line")
        drag(w, (x, 70), (x, 150))
        card = w.shape_panel._cards[0]
        card.bodypart.setCurrentIndex(card.bodypart.findData("head"))
        card.column.setCurrentText(name)
        card.column.lineEdit().editingFinished.emit()
    layout = w.session.layout
    w._rescore_current()
    dialog = FormulaDialog(layout, None, w.columns_panel.sample_values, w)
    assert not dialog.ok_button.isEnabled()
    dialog.name.setText("Total")
    dialog.expression.setPlainText("[Left] + [Wrong]")
    assert not dialog.ok_button.isEnabled() and "no column called [Wrong]" in dialog.status.text()
    dialog.expression.setPlainText("[Left] + [Right]")
    assert dialog.ok_button.isEnabled() and "gives <b>1</b>" in dialog.status.text()
    dialog._accept()
    assert (dialog.result_name, dialog.result_expression) == ("Total", "[Left] + [Right]")
    from posetally.model import FormulaColumn

    layout.formulas.append(FormulaColumn(dialog.result_name, dialog.result_expression))
    w.columns_panel.refresh()
    assert [w.columns_panel.list.item(i).data(Qt.UserRole) for i in range(3)] == ["Left", "Right", "Total"]
    w.columns_panel.list.setCurrentRow(2)
    w.columns_panel._move(-2 + 1)  # one step up
    assert layout.columns() == ["Left", "Total", "Right"]

    ids = IdColumnDialog(layout.id_column, [i.source for i in w.session.items], w)
    ids.header.setText("Fish")
    ids.mode.setCurrentIndex(ids.mode.findData("pieces"))
    ids.pieces.setText("2")
    assert ids.preview.item(0, 1).text() == "01"
    layout.id_column = ids.result_config

    out = tmp_path / "results.xlsx"
    w.score_all(blocking=True)
    assert w.results_panel.scores.rowCount() == 3
    assert w.export_to(str(out))
    rows = list(load_workbook(out)["Scores"].iter_rows(values_only=True))
    assert rows[0] == ("Fish", "Left", "Total", "Right")
    # the lines count crossings in either direction: going into the right arm and back out is 2
    assert rows[1:] == [("01", 1, 1, 0), ("02", 1, 3, 2), ("03", 0, 2, 2)]
    # the setup is stored with the project and comes back in a new window
    w._save()
    again = MainWindow()
    assert again.open_project(w.demo["single"])
    assert [l.name for l in again.session.layout.lines] == ["Line 1", "Line 2"]
    assert again.session.layout.columns() == ["Left", "Total", "Right"] and len(again.session.items) == 3
    again.close()


def test_nothing_to_score_and_untracked_videos_are_explained(window, tmp_path):
    w = window
    dialogs.QUIET.clear()
    w.score_all(blocking=True)
    assert any("nothing to score" in text for _kind, _title, text in dialogs.QUIET)
    video = tmp_path / "fresh.mp4"
    shutil.copy(w.session.items[0].video_path, video)
    w.session.add_videos([str(video)])
    w._items_changed()
    w.video_list.select_key(str(video))
    pump(60)
    assert "not been tracked yet" in w.shape_panel.summary.text()
    assert "1 to track" in w.video_list.summary.text()


def test_tracking_window_runs_the_network_over_untracked_videos(window, tmp_path, monkeypatch):
    w = window
    project_folder = os.path.dirname(w.demo["single"])
    train = os.path.join(project_folder, "dlc-models-pytorch", "iteration-0", "MazeDemoNov10-trainset95shuffle1", "train")
    os.makedirs(train)
    open(os.path.join(train, "snapshot-200.pt"), "wb").close()
    assert w.open_project(w.demo["single"])  # the trained network is now found
    assert [m.shuffle for m in w.session.project.models] == [1]
    new = tmp_path / "new videos"
    new.mkdir()
    for name in ("fish_a.mp4", "fish_b.mp4", "fish_crash.mp4"):
        shutil.copy(w.session.items[0].video_path, new / name)
    w.add_folder_path(str(new))
    assert len(w.session.items_to_track()) == 3

    monkeypatch.setenv("PYTHONPATH", FAKE)
    dialog = TrackDialog(w.session, w)
    dialog.show()
    deadline = time.time() + 60
    while dialog.check is not None and time.time() < deadline:
        pump(50)
    assert "DeepLabCut 3.0.2-fake" in dialog.environment.text()
    assert "<b>3</b> of 6 videos need tracking" in dialog.summary.text()
    assert dialog.start_button.isEnabled()
    dialog.start()
    while dialog.is_running() and time.time() < deadline:
        pump(50)
    assert not dialog.is_running()
    assert (dialog._done, dialog._failed) == (2, 1)
    assert "2 video(s) tracked" in dialog.stage.text() and "1 could not be tracked" in dialog.stage.text()
    assert "pretend decoding error" in dialog.log.toPlainText()
    # the failures are repeated at the very end of the log, where they are seen without scrolling back
    last_lines = dialog.log.toPlainText().strip().splitlines()[-2:]
    assert last_lines[0] == "Could not be tracked (1):" and "fish_crash" in last_lines[1] and "pretend decoding error" in last_lines[1]
    assert dialog.overall.value() == 3
    dialog.close()
    w._tracking_changed()
    tracked = {item.name: item.tracked for item in w.session.items}
    assert tracked["fish_a"] and tracked["fish_b"] and not tracked["fish_crash"]
    assert "5 tracked" in w.video_list.summary.text()
    # and a freshly tracked video can be scored like any other
    w.video_list.select_key(str(new / "fish_a.mp4"))
    pump(60)
    assert w._tracks is not None and w._tracks.all_bodyparts() == ["nose", "head"]


def test_a_tracking_program_that_dies_is_reported_and_finished_videos_are_kept(window, tmp_path, monkeypatch):
    w = window
    project_folder = os.path.dirname(w.demo["single"])
    train = os.path.join(project_folder, "dlc-models-pytorch", "iteration-0", "MazeDemoNov10-trainset95shuffle1", "train")
    os.makedirs(train)
    open(os.path.join(train, "snapshot-200.pt"), "wb").close()
    assert w.open_project(w.demo["single"])
    new = tmp_path / "new videos"
    new.mkdir()
    for name in ("a_fine.mp4", "b_vanish.mp4", "c_never_reached.mp4"):
        shutil.copy(w.session.items[0].video_path, new / name)
    w.add_folder_path(str(new))

    monkeypatch.setenv("PYTHONPATH", FAKE)
    record = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_DLC_RECORD", str(record))
    dialog = TrackDialog(w.session, w)
    dialog.show()
    deadline = time.time() + 60
    while dialog.check is not None and time.time() < deadline:
        pump(50)
    dialog.batch.setCurrentIndex(dialog.batch.findData(2))  # "Frames at a time": 2
    dialogs.QUIET.clear()
    dialog.start()
    while dialog.is_running() and time.time() < deadline:
        pump(50)
    assert not dialog.is_running()
    assert dialog._fatal and dialog._fatal["kind"] == "crashed"
    assert "Tracking stopped unexpectedly" in dialog.stage.text() and "1 video(s) were tracked before that" in dialog.stage.text()
    assert "stopped unexpectedly (exit code 137)" in dialog.log.toPlainText()
    shown = " ".join(str(entry) for entry in dialogs.QUIET)
    assert "ran out of memory" in shown and "Frames at a time" in shown
    # the chosen number of frames at a time reached DeepLabCut
    import json

    calls = [json.loads(line) for line in record.read_text().splitlines()]
    assert calls and calls[0]["batch_size"] == 2
    dialog.close()
    w._tracking_changed()
    tracked = {item.name: item.tracked for item in w.session.items}
    assert tracked["a_fine"] and not tracked["b_vanish"] and not tracked["c_never_reached"]
    assert w.session.track_options["batch_size"] == 2  # and is remembered for the next run


def test_tracking_window_explains_a_missing_network_and_missing_deeplabcut(window, monkeypatch):
    w = window
    dialog = TrackDialog(w.session, w)
    assert "No trained network was found" in dialog.summary.text()
    assert not dialog.start_button.isEnabled()
    deadline = time.time() + 60
    while dialog.check is not None and time.time() < deadline:
        pump(50)
    try:
        import deeplabcut  # noqa: F401
    except ImportError:
        assert "DeepLabCut is not installed" in dialog.environment.text()
    dialog.close()


def test_export_tracking_as_csv_for_other_tools(window, tmp_path):
    w = window
    out = tmp_path / "csv"
    out.mkdir()
    written, failed = w.export_tracking_csv_to(str(out))
    assert (written, failed) == (3, 0)
    assert sorted(os.listdir(out)) == ["ABOUT THESE FILES.txt", "trial_01_groupA.csv", "trial_02_groupA.csv", "trial_03_groupB.csv"]
    header = (out / "trial_01_groupA.csv").read_text().splitlines()[:3]
    assert header[0].startswith("scorer,") and header[1].startswith("bodyparts,nose,nose,nose,head") and header[2].startswith("coords,x,y,likelihood")


def test_an_unreadable_video_is_explained_and_the_remark_goes_away(window, tmp_path):
    w = window
    more = tmp_path / "more"
    more.mkdir()
    (more / "broken.mp4").write_bytes(b"this is not a video")
    w.add_folder_path(str(more))
    pump(100)
    w.video_list.select_key(str(more / "broken.mp4"))
    pump(100)
    assert "could not be opened. It may be damaged" in w.statusBar().currentMessage()
    good = [item for item in w.session.items if item.tracked][0]
    w.video_list.select_key(good.key)
    pump(100)
    assert "could not be opened" not in w.statusBar().currentMessage()
