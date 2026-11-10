# PoseTally

PoseTally scores animal behaviour from DeepLabCut tracking.

You open a DeepLabCut project, point PoseTally at a batch of videos, and it runs
the project's trained network over them. You then draw lines and regions on the
video and say what each one is worth: "add 1 to *Left arm* each time the head
crosses this line from right to left", "measure the time the body spends in this
region". PoseTally applies that to every video and writes one Excel workbook with
one row per video (and per animal), ready for statistics.

It is not tied to any species, body-part names, maze or computer. Anything
DeepLabCut can track, PoseTally can score.

- Lines that count crossings, with direction (left to right, right to left, top to
  bottom, bottom to top, or either).
- Regions of any shape (corner by corner, rectangle, oval, freehand) that measure
  time inside, entries, exits, or the time until the first entry.
- Any number of scores per line or region (the **+ Add another score** button), each
  with its own body part, points and column.
- "First choice" scoring for mazes: only the arm that was entered first gets the point.
- Single-animal and multi-animal projects. Each animal gets its own tallies.
- Excel export with the file name, or any part of it, in the left-most column.
- Formula columns that combine other columns: totals, percentages, differences, anything arithmetic.
- Playback with the tracked points and every scored event marked, so you can check the scoring by eye.

SimBA is not needed and is not installed. PoseTally reads DeepLabCut's results
directly. If you later want SimBA's behaviour classifiers, see
[Using the tracking in SimBA or other tools](#using-the-tracking-in-simba-or-other-tools).

---

## Contents

1. [Installing](#installing)
2. [Trying it with the demo](#trying-it-with-the-demo)
3. [The five steps](#the-five-steps)
4. [Lines, regions and rules](#lines-regions-and-rules)
5. [The results workbook](#the-results-workbook)
6. [Formula columns](#formula-columns)
7. [How events are decided](#how-events-are-decided)
8. [Settings](#settings)
9. [Where things are saved](#where-things-are-saved)
10. [Using the tracking in SimBA or other tools](#using-the-tracking-in-simba-or-other-tools)
11. [Problems and fixes](#problems-and-fixes)
12. [Networks trained with TensorFlow](#networks-trained-with-tensorflow)
13. [What has been tested](#what-has-been-tested)
14. [For developers](#for-developers)
15. [Citing](#citing)

---

## Installing

You do not need to have Python or DeepLabCut installed. The installer brings its
own private copies and does not touch any Python or DeepLabCut you already have.

**You need**

- A Mac (Apple chip recommended) or a 64-bit Windows 10 or 11 PC. Linux also works.
- An internet connection for the installation.
- About 8 GB of free disk space (about 15 GB on a PC with an NVIDIA graphics card).
- For tracking new videos at a useful speed: an NVIDIA graphics card (Windows, Linux)
  or an Apple chip. Without one, tracking still works but is slow. Scoring videos
  that are already tracked is fast on any computer.

### macOS

1. Unpack the ZIP file (double-click it).
2. In the unpacked folder, double-click **Install-Mac.command**.
3. The first time, macOS may say the file "cannot be opened because it is from an
   unidentified developer". If so, right-click the file, choose **Open**, then
   **Open** again. On recent macOS versions you may instead need to go to
   System Settings → Privacy & Security and press **Open Anyway**.
   (Alternative that always works: open the Terminal app, type `bash ` with a space
   after it, drag **Install-Mac.command** into the Terminal window, press Return.)
4. A Terminal window shows the progress. Leave it until it says
   **PoseTally is installed**. Expect 10 to 30 minutes, mostly downloading.
5. Start PoseTally from the icon on your Desktop, or from the *Applications* folder
   inside your home folder.

### Windows

1. Unpack the ZIP file: right-click it, choose **Extract All…**. Do not run the
   installer from inside the ZIP.
2. In the unpacked folder, double-click **Install-Windows.bat**.
3. If Windows shows "Windows protected your PC", click **More info**, then **Run anyway**.
4. A window shows the progress. Leave it until it says **PoseTally is installed**.
   Expect 10 to 40 minutes. With an NVIDIA graphics card the download is larger (2 to 3 GB extra).
5. Start PoseTally from the shortcut on your Desktop or from the Start menu.

### Linux

Run `bash Install-Linux.sh` in a terminal. It creates the command `posetally` and a menu entry.

### What the installer does

Everything goes into one folder: `.posetally` in your home folder on macOS and Linux,
`%LOCALAPPDATA%\PoseTally` on Windows.

1. Downloads **uv**, a small tool that installs Python packages, from its GitHub releases page.
2. Installs a private copy of **Python 3.12**.
3. Installs **PyTorch** in the build that suits the computer. On Windows and Linux it
   looks for an NVIDIA graphics card, tries the newest PyTorch build the card's driver
   supports, runs a small calculation on the card to confirm it works, and steps down
   to an older build if it does not. With no usable card it installs the processor-only build.
4. Installs **DeepLabCut 3.0.2** and the few extra packages PoseTally needs.
   Package versions are taken as they were on the day PoseTally was tested, so every
   computer gets the same set.
5. Copies the PoseTally program into place, checks that it and DeepLabCut start, and
   creates the icon.

Outside that folder it only creates the icon: `PoseTally.app` in your home *Applications*
folder plus a Desktop link on macOS (and next to the app a second starter, *PoseTally (show
messages)*, for troubleshooting); Desktop and Start-menu shortcuts on Windows.

**To update or repair**, run the installer again. It keeps what is already installed and
working, and replaces the PoseTally program files with the ones next to the installer.

**To remove PoseTally**, double-click **Uninstall-Mac.command** or **Uninstall-Windows.bat**.
Your DeepLabCut projects, videos, results and saved scoring setups are not touched.

### Installer options

Most people never need these. They are given after the installer's name in a
terminal, for example `bash Install-Mac.command --with-tensorflow` or
`Install-Windows.bat --cpu-only`.

| Option | What it does |
| --- | --- |
| `--with-tensorflow` | Also installs TensorFlow, for networks trained with DeepLabCut's older TensorFlow engine. See [below](#networks-trained-with-tensorflow). |
| `--dlc-version 3.0.1` | Installs another DeepLabCut version, for example the one your network was trained with. |
| `--cpu-only` | Ignores an NVIDIA graphics card. |
| `--no-deeplabcut` | Leaves out DeepLabCut and PyTorch (a much smaller install). PoseTally can then score videos that are already tracked but cannot track new ones. |
| `--reinstall` | Installs PyTorch again even if a working one is there, for example after changing the graphics card or its driver. |
| `--no-date-freeze` | Installs the newest package versions instead of the tested set. |
| `--torch-index URL` | Installs PyTorch from a package index you name (advanced). |

---

## Trying it with the demo

On the start screen, press **Try it with the demo** (or Help → Open the demo project).
PoseTally creates a folder called *PoseTally Demo* in your home folder with made-up
videos of a blob in a T-maze, already "tracked", and an example setup: two lines for
the arms with direction and first-choice rules, a line for leaving the start, a
start-box region, a goal region, and a formula column.

Click a line or region to see its rules, press Play, then press **4 Score** and
**5 Export**. A second demo project in the same folder, *Open arena (two animals)*,
shows per-animal tallies; open its `config.yaml` with **1 Open project**.

The demo needs no trained network, so **3 Track** has nothing to do there.

---

## The five steps

The buttons along the top are the workflow.

### 1 Open project

Choose the `config.yaml` at the top of your DeepLabCut project folder. PoseTally reads
the body-part names, whether the project is single- or multi-animal, the likelihood
cut-off, and looks through the project folder for trained networks.

### 2 Add videos

Add a folder (folders inside it are included) or single files. Each video shows as
**Tracked** if DeepLabCut results for it are found next to it, or **Not tracked**.
Results made earlier with DeepLabCut itself are recognised, so nothing is tracked twice.

If only the result files are in a folder and the videos themselves have been archived
elsewhere, they are listed as **Tracking only** and can be scored without the video.

### 3 Track

Runs the project's trained network over every video that is not tracked yet. This is
the slow step: on a graphics card expect very roughly real-time speed or better; on a
processor alone it can take many times the length of the video.

- **Network**: which trained network (DeepLabCut "shuffle") to use, if the project has several.
- **Run on**: leave on *Automatic* unless something goes wrong.
- **Frames at a time**: how many video frames the network handles in one go. Leave it on
  *As in the project* unless tracking stops because the computer ran out of memory; then
  choose a smaller number.
- **Save tracking files**: next to each video (DeepLabCut's usual place), or in one folder.
- **Animals per video** (multi-animal projects only): set it if the videos hold a different
  number of animals than the project was set up for.
- You can **Stop** at any time. Finished videos stay finished, and the next run continues with the rest.
- A video that cannot be read is reported and skipped; the others carry on. The videos
  that could not be tracked are listed again at the end of the progress box.

Tracking runs in a separate program, so the window stays usable, and DeepLabCut's own
messages appear in the progress box.

### 4 Score

Draw your lines and regions and set their rules (next section). The video you are
looking at is scored instantly as you work, so you can see the effect of every
change. **4 Score** applies the setup to every video in the list and fills the
*Scores* table at the bottom.

### 5 Export

Saves the workbook. If the setup changed since the last scoring, the videos are
scored again first, so an export is never out of date.

---

## Lines, regions and rules

### Drawing

Pick a tool above the video.

| Tool | How to draw |
| --- | --- |
| **Line** | Press where it starts, drag, release where it ends. |
| **Region** | Click each corner. Finish with a double-click, Enter, a right-click, or by clicking the first corner. Backspace removes the last corner, Esc cancels. |
| **Rectangle** | Drag from one corner to the opposite corner. |
| **Oval** | Drag out the box the oval should fill. |
| **Freehand** | Hold the button down and draw around the area. It closes when you let go. |

With **Select**: click a shape to edit its rules, drag it to move it, drag a white
square handle to reshape it. Arrow keys nudge the selected shape by one pixel (ten
with Shift). Delete removes it. Scroll to zoom, drag the background to pan, press 0
or **Fit** to see the whole frame.

Draw lines a little longer than the passage they span. A crossing only counts if the
path goes through the drawn line itself; going around its end does not count.

Turn on **Show the path of** a body part to see the whole route the animal took in
this video. It makes placing lines much easier.

### Rules for a line

Click the line. Each rule ("Score 1", "Score 2", …) has:

- **Body part**: the tracked point that has to cross. Any label of the DeepLabCut project.
- **When it crosses**: in either direction, left → right, right → left, top → bottom
  or bottom → top, as seen on the screen. Only the directions that make sense for the
  line are offered: a vertical line has left/right, a horizontal one has top/bottom, a
  diagonal one has all four. Arrows on the line show which way scores.
- **What to score**: add points each time it crosses, or record the time of the first crossing (in seconds).
- **Points to add**: any number. Negative numbers subtract.
- **Into column**: the column of the results this goes into. Pick an existing column
  to add to it, or type a new name. Several rules, on the same shape or on different
  shapes, can feed one column; their contributions are added up.
- **Count**: every time, the first time only, or only if it is the animal's first choice.

**+ Add another score** adds another rule to the same line. There is no limit.

### Rules for a region

- **Body part**: the point that has to be inside.
- **What to score**: time spent inside (seconds), points for each entry, points for
  each exit, or the time of the first entry (seconds).
- The other fields are as for lines.

### First choice (mazes)

To score which arm the animal chose first, give each arm's line a rule with **Count**
set to *Only if it is the animal's first choice* and the same **Choice group** name,
for example `arm`. In each video, and for each animal, only the rule whose event
happened first gets its points. With one point each into the columns *Chose left*
and *Chose right*, exactly one of them is 1.

You can have several groups (for example `arm` and `exit`), and a line can have a
first-choice rule and an every-time rule side by side.

### One video filmed differently

Shapes sit in the same place for every video. If the camera or the maze moved for
some videos, select such a video, set **Moving a shape changes** (under the video)
to *only this video*, and adjust the shapes. That video then has its own positions
(dashed outline, and "own positions" in the video list) while the rules stay shared.
"Use the shared position" in the panel, or the video's right-click menu, undoes it.

If a video has a different frame size from the one the shapes were drawn on (the
same scene saved at another resolution), the shapes are stretched to fit, and the
Quality sheet says so.

### Checking the scoring

- The coloured ticks on the bar under the video are the scored events, in the colour of their shape.
- **[** and **]** (or the double-arrow buttons) jump to the previous or next event.
- The **Events in this video** tab lists them; clicking a row jumps to that moment.
- The tracked points are drawn on the frame. A dotted circle is a point below the likelihood cut-off, which is ignored.
- **This video**, at the bottom of the panel on the right, shows the current video's numbers for every column.

---

## The results workbook

**Scores** is the sheet for your statistics:

| File | Animal | Left entries | Chose left | Time in start box (s) | Arm entries (total) |
| --- | --- | --- | --- | --- | --- |
| trial_01 | | 1 | 1 | 2.3 | 1 |
| trial_02 | | 1 | 0 | 1.8 | 2 |

- The left-most column identifies the video. The **Animal** column only appears when the tracking has several animals.
- Then one column per score and per formula, in the order shown in the **Columns** tab (drag to reorder).
- A video that could not be scored (not tracked, unreadable file) is still listed, with empty cells, so the batch stays complete.
- An empty cell means "no value", for example the time of a first crossing that never happened. It is not zero.

The other sheets document how the numbers were made:

- **Events**: every scored event with its time, frame, shape, body part, column and points. For "time inside" rules it lists each entry and exit.
- **Quality**: for each video and animal, the frame count, frame rate, length, the percentage of frames in which each scoring body part was tracked, how many events happened right after a tracking gap, and any notes.
- **Setup**: the settings, every rule in words, every formula, and the coordinates of every shape. With this sheet a result can be reproduced exactly.

You can also save the Scores table alone as CSV: choose "CSV" as the file type when exporting.

### The first column

In the **Columns** tab, press **Change…** next to "First column".

| Choice | For `fish07_trialB_2024-05-01.mp4` |
| --- | --- |
| File name | `fish07_trialB_2024-05-01` |
| File name with extension | `fish07_trialB_2024-05-01.mp4` |
| Folder and file name | `day3/fish07_trialB_2024-05-01` |
| Part of the file name: split at `_`, keep piece `1` | `fish07` |
| … keep pieces `1-2` | `fish07_trialB` |
| … keep pieces `2,last` | `trialB_2024-05-01` |
| Text between two markers: after `fish`, before `_` | `07` |
| Regular expression `trial([A-Z])` | `B` |

The window shows a preview for your own file names. You can also change the column's heading.

---

## Formula columns

**Columns → + Formula column…** adds a column that is calculated from others. Write
it as you would in a spreadsheet, with column names in square brackets:

```
[Left entries] + [Right entries]
[Correct] / ([Correct] + [Wrong]) * 100
[Time in goal (s)] / 60
IF([Chose left] == 1, [Left first crossing (s)], [Right first crossing (s)])
```

Double-click a column in the list to insert it. The window tells you whether the
formula is valid and what it gives for the video you are looking at.

| You can use | |
| --- | --- |
| Arithmetic | `+  -  *  /  ^` (power), brackets |
| Comparisons | `<  <=  >  >=  ==  !=` give 1 for true and 0 for false |
| Functions | `MIN MAX SUM MEAN COUNT ABS SQRT ROUND(x, digits) FLOOR CEIL LN LOG10 EXP` |
| Logic | `IF(test, value if true, value if false)`, `AND(...)`, `OR(...)`, `NOT(x)` |
| Empty cells | `ISBLANK(x)`, `IFBLANK(x, value)` |

- A formula can use other formula columns.
- **Empty cells**: arithmetic with an empty value gives an empty result. `SUM`, `MIN`, `MAX`, `MEAN` and `COUNT` skip empty values. `IFBLANK([Column], 0)` treats empty as 0.
- Dividing by zero gives an empty cell, not an error.
- Formulas are plain arithmetic. They cannot run code or read files.

Formulas are worked out by PoseTally and written to the workbook as numbers. The
formula text is recorded on the Setup sheet.

---

## How events are decided

This section is exact, so that it can go into a methods section.

**Cleaning the path.** For each body part a rule uses:

1. Frames where DeepLabCut's likelihood is below the cut-off are treated as not tracked.
2. Gaps no longer than the gap limit (default 0.2 s) are bridged by a straight line between the positions on either side. Longer gaps stay empty.
3. Each position is replaced by the median of itself and its neighbours (default window 5 frames), which removes one- or two-frame jumps to a wrong place. Untracked frames stay untracked.

**Crossing a line.** The point's distance from the line is followed frame by frame. A
crossing is registered when the point is more than the *margin* (default 4 pixels) past
the line, having earlier been more than the margin on the other side, and the path
between passed through the drawn segment (not around its end). The event is placed at
the first frame on the new side during that final pass. A point that hovers on the
line, or pokes over by less than the margin and returns, scores nothing.

**Direction** is the direction the point travelled across the line, named as it appears
on screen. With "The animal must face the way it moves" switched on, the crossing also
has to agree with the direction from the *back* body part to the *front* body part at
that frame; a crossing made while backing up is then ignored. If either of those parts
is not tracked at that frame, the crossing is counted.

**Entering and leaving a region.** The same margin applies: the point enters when it
is more than the margin inside the edge, having been more than the margin outside, and
leaves the other way round. For regions narrower than the margin, the inner margin
is reduced automatically so that entering stays possible. An animal that is already
inside at its first tracked frame has not "entered": that is not counted as an entry,
but for "time of the first entry" the time of that first tracked frame (normally zero) is used.

**Time inside** is the number of frames in the inside state divided by the frame rate.

**While not tracked.** By default the last known answer is kept: an animal that
disappears inside a region (under a shelter, say) keeps counting as inside until it
is seen outside. This can be switched to "is not counted". A point that disappears on
one side of a line and reappears beyond the margin on the other side is counted as a
crossing (if the straight line between the two positions passes through the segment)
and is flagged in the Events sheet with the number of untracked frames; this too can
be switched off.

**Time** of an event is its position in the video divided by the frame rate, with the
first frame at time 0. Frame numbers on screen and in the workbook are counted from 1,
so frame 1 is row 0 of DeepLabCut's own table.

**Several animals.** Every animal in the tracking file is scored separately with the
same shapes and rules, and gets its own row. PoseTally relies on DeepLabCut's
identities: if DeepLabCut swaps two animals' identities during a video, their tallies
are swapped from that moment too. Check identity quality in DeepLabCut before
trusting per-animal numbers; the sum over animals is not affected by swaps.

---

## Settings

The **Settings** tab. Hover over any setting for an explanation. They are saved with
the project and written into every workbook.

| Setting | Default | Meaning |
| --- | --- | --- |
| Ignore points with likelihood below | the project's `pcutoff` (usually 0.6) | Below this, a point counts as not tracked. |
| Fill gaps up to | 0.2 s | Longest tracking gap that is bridged. 0 switches bridging off. |
| Smooth over | 5 frames | Median window. 1 switches smoothing off. |
| Distance past a line or edge | 4 px | The margin described above. Raise it if events are counted twice at an edge; lower it for very small animals or low-resolution video. |
| Count crossings made while not tracked | on | See above. |
| Time in a region while not tracked | stays as it was | Or "is not counted". |
| The animal must face the way it moves | off | Needs a front and a back body part. |
| Use DeepLabCut's filtered files when they exist | off | Scores `…_filtered.h5` files (made by DeepLabCut's "filter predictions") instead of the unfiltered ones. |
| Frames per second if a video does not say | 30 | Only for videos without a usable frame rate. |

A reasonable starting margin is about a tenth of the animal's body length in pixels.

---

## Where things are saved

- **Your scoring setup** (shapes, rules, columns, settings, the video list) is saved
  automatically in a folder named `PoseTally` inside the DeepLabCut project folder,
  and restored when you open the project again. Because it lives in the project
  folder, it travels with the project, for example on an external drive between a
  Mac and a Windows PC. Videos inside the project folder are remembered by relative
  path and are found again on the other computer; videos elsewhere need to be added again there.
- **File → Save scoring setup as…** writes the setup to a single file that you can
  load into another project (**Load scoring setup…**) or share with a colleague.
- **Tracking files** are DeepLabCut's own (`<video>DLC_….h5` and companions), stored
  next to each video unless you chose a folder.
- **Messages and errors** are also written to a log file: `~/Library/Logs/PoseTally/posetally.log`
  on macOS, `%LOCALAPPDATA%\PoseTally\logs\posetally.log` on Windows.

DeepLabCut rewrites the `project_path` line of `config.yaml` when a project folder has
moved. That is DeepLabCut's normal behaviour and happens the first time you track on a new computer.

---

## Using the tracking in SimBA or other tools

PoseTally does its own scoring and does not use SimBA. SimBA's region tools count
entries and time in zones but have no line crossings, crossing direction or
first-choice scoring, which is why PoseTally exists.

If you want SimBA's behaviour classifiers on the same videos, **Tools → Export
tracking as CSV files…** writes one CSV per video, named after the video, in
DeepLabCut's own CSV layout. That is the layout SimBA's DeepLabCut import expects.
This export has not been tested against a SimBA installation.

---

## Problems and fixes

**macOS will not open the installer.** Right-click → Open, or use the Terminal method under [Installing](#macos).

**macOS says the installer "could not be executed because you do not have appropriate
access privileges".** The program that unpacked the ZIP dropped the file's permission to
run. Use the Terminal method under [Installing](#macos); it works without that permission.

**The PoseTally icon does nothing, or the window closes at once (macOS).** Open the
*Applications* folder inside your home folder and double-click **PoseTally (show messages)**.
It starts PoseTally from a Terminal window, where any error message stays visible.

**macOS asks to install Rosetta when PoseTally starts.** It should not: on a Mac with an
Apple chip PoseTally runs natively and the installer tells macOS so. If it asks anyway,
you can let it install Rosetta (it does no harm), or start PoseTally with
*PoseTally (show messages)*, which does not ask.

**Windows says "Windows protected your PC".** More info → Run anyway.

**The installer stops with a download error.** Check the internet connection and run
it again; it continues where it stopped. On a network that blocks GitHub, PyPI
(`pypi.org`, `files.pythonhosted.org`) or `download.pytorch.org`, ask IT to allow them.
If the messages mention a *certificate*, the network inspects encrypted traffic (some
companies and universities do). Open a terminal, type `export UV_SYSTEM_CERTS=1` (macOS,
Linux) or `set UV_SYSTEM_CERTS=1` (Windows), and start the installer from that same terminal.

**Windows: "DeepLabCut was installed but does not start", mentioning a DLL.** Install
the Microsoft Visual C++ Redistributable from <https://aka.ms/vs/17/release/vc_redist.x64.exe>
and run the installer again.

**The NVIDIA card is not used.** The installer reports which device tracking will run
on. If it fell back to the processor, update the NVIDIA driver and run the installer
with `--reinstall`.

**"No trained network was found".** The project folder has no finished training
(`dlc-models-pytorch/…/train/snapshot-….pt`). Train the network in DeepLabCut first, or
make sure you opened the `config.yaml` of the right copy of the project.

**Tracking stops with "The tracking program stopped unexpectedly".** The system ended the
program that runs DeepLabCut, nearly always because the computer or its graphics card ran
out of memory. Videos that were finished stay tracked. Close other programs, set **Frames
at a time** to 1 or 2 in the Track window, and press **Start tracking** again; it continues
with the videos that are left. Multi-animal networks need the most memory.

**"DeepLabCut did not find any animal in this video".** The network detected nothing it was
sure about, so there is no tracking for that video. Look at the video in DeepLabCut (for
example with `create_labeled_video`); the network may need more labelled frames from
videos like this one.

**Tracking fails for every video with the same message.** Read the message in the
progress box; it is DeepLabCut's own. Common causes: the network was trained with
TensorFlow ([see below](#networks-trained-with-tensorflow)); the network was trained
with a newer DeepLabCut than the one installed (install that version with
`--dlc-version`); the project drive is read-only.

**A video is skipped because its name ends in `_labeled` or `_full`.** DeepLabCut
reserves those endings for videos it creates itself. Rename the video.

**A video shows as "Tracked" but with the wrong results file.** Right-click it →
*Choose the tracking file by hand…*.

**Events are a frame or two off the picture.** The picture shown for a frame number
relies on the video's index. A few formats (variable frame rate recordings from
phones or screen recorders, damaged files) seek imprecisely. The scores are not
affected, because they come from the tracking table, which was made by reading the
video front to back. Re-encoding the video with a constant frame rate fixes the display.

**Times look wrong.** PoseTally uses the frame rate stored in the video file. Check
the Quality sheet. A variable-frame-rate video has no single correct value; re-encode
it with a constant frame rate before tracking.

**The window opened but something failed silently.** Look at the *Messages* tab at the
bottom and at the log file (see above). On Windows, the file
`%LOCALAPPDATA%\PoseTally\PoseTally (show messages).bat` starts PoseTally with a console that shows errors.

**The menu bar on a Mac says "python" instead of "PoseTally".** Cosmetic: the program
runs inside its private Python.

---

## Networks trained with TensorFlow

DeepLabCut 3 trains with PyTorch. Projects made with DeepLabCut 2.x used TensorFlow,
and their networks are in a folder called `dlc-models` (not `dlc-models-pytorch`).
PoseTally recognises them and says so in the Track window.

You have two options:

1. **Retrain with PyTorch (recommended).** Your labelled frames are reused as they
   are. In DeepLabCut 3, create a new training dataset with the PyTorch engine and
   train it. The new network appears in PoseTally's Network list. This is the only
   way to use an NVIDIA card on Windows, because TensorFlow stopped supporting
   graphics cards on Windows after version 2.10.
2. **Install the TensorFlow add-on.** Run the installer with `--with-tensorflow`.
   DeepLabCut then runs the old network through TensorFlow (on the processor on
   Windows). DeepLabCut plans to drop TensorFlow support in a later version.

The TensorFlow route is written against DeepLabCut's documented interface but has
not been run here; treat it as untested.

---

## What has been tested

Plainly, so you know where to look first if something misbehaves.

**Tested, with automated tests that are included in the `tests` folder**

- The scoring itself: crossings and their direction, margins, going around a line's
  end, regions of every shape, entries, exits, time inside, first-time and first-choice
  rules, points, several rules per column, tracking gaps, low-likelihood points,
  smoothing, body orientation, per-animal tallies, stretched and per-video shape positions.
  The made-up demo videos have known answers and the tests check them.
- Reading DeepLabCut's files: single- and multi-animal `.h5` and `.csv`, cropped
  analyses, the companion `_meta.pickle`, picking the right file among several, and
  reading a project's `config.yaml` and trained-network folders.
- Formula columns and file-label options, including refusing anything that is not arithmetic.
- The Excel workbook, re-opened and checked cell by cell.
- The window itself, driven without a screen: drawing every kind of shape with the
  mouse, moving and reshaping, editing rules in the panel, per-video positions,
  playback, scoring, exporting, saving and restoring the setup.
- The tracking window and its background program, including stopping on a network
  problem, carrying on past a broken video, finishing videos an interrupted run left
  half-done, and reporting a tracking program that the system ended for lack of memory.
- The decisions the installer takes (which PyTorch build to try, stepping down, keeping a
  working one on a second run, the macOS app, the Windows shortcuts), without downloading anything.

**How the DeepLabCut side was checked**

PoseTally was run against a real installation of DeepLabCut 3.0.2 with PyTorch 2.15, on
Linux, on the processor only (the test machine had no graphics card). The projects were
made with DeepLabCut's own functions: new project, training dataset, training.

- PoseTally read each project's `config.yaml` and found the trained networks and the
  snapshot files DeepLabCut's trainer wrote, including its `snapshot-best-…` files.
- The Track window, driven without a screen, tracked videos with the real network: the
  progress display, a broken video file being reported and skipped, and a second run
  finding everything already done. A network without snapshot files was reported with
  DeepLabCut's own message, and tracking stopped cleanly.
- PoseTally found the files DeepLabCut wrote, matched them to the network, read them,
  scored them and exported the workbook. Saving the tracking files into a separate folder
  worked, and DeepLabCut's `.csv` read identically to its `.h5`.
- With a small network trained for this test on the made-up maze videos, the positions
  PoseTally read back were a median of 1.4 pixels from where the animal's body centre was
  drawn (640 × 480 frame; 90% within 3.2 pixels), and matched best with no shift in frame
  number. So coordinates and frame numbers line up, from the video through DeepLabCut to the score.
- A multi-animal project went through detection and DeepLabCut's linking of detections
  into tracks. The small test network found no animals, and PoseTally reported that for
  the video as it should. So **a finished multi-animal tracking table from a real
  DeepLabCut run has not been read**; reading multi-animal tables is tested with files
  written in DeepLabCut's layout.
- On the test machine, which has little memory, that multi-animal run was ended by the
  system until **Frames at a time** was set to 1. That is how the out-of-memory message
  and that option were checked.

The test networks were trained for minutes, not hours, on made-up videos. These runs show
that the pieces connect and line up; they say nothing about how accurate your own network
is, which is DeepLabCut's side.

The other tracking tests use a stand-in for DeepLabCut (`tests/fake_dlc`) with the same
function signature, messages and file names as DeepLabCut 3.0.2, so that they run in
seconds without it.

**Not tested on real hardware**

- The macOS and Windows installers have not been run on a Mac or a Windows PC. They
  are thin wrappers around one shared installer program. On Linux that program was run
  from start to finish without DeepLabCut, and again over an environment that already
  held DeepLabCut and PyTorch, where it kept them, checked that DeepLabCut starts and
  made the launcher. It has not been watched downloading the complete DeepLabCut set in
  one go, because the test machine could not reach PyTorch's download site; that the
  complete set of packages fits together was checked for Linux, Windows and Macs with an
  Apple chip. The Windows scripts were checked for syntax errors. The parts that only
  exist on those systems (the macOS app and its icon, Windows shortcuts, choosing the
  NVIDIA build of PyTorch) are therefore the most likely places for a first-run problem.
- Tracking on a graphics card.
- The TensorFlow add-on and the SimBA hand-off, as noted above.

If the installer fails, the last screenful of its output says which step failed;
that text is what is needed to fix it.

---

## For developers

```
PoseTally/
  Install-Mac.command, Install-Windows.bat, Install-Linux.sh    first-stage installers
  Uninstall-Mac.command, Uninstall-Windows.bat
  installer/
    install_unix.sh, install_windows.ps1    get uv, Python, an empty environment
    posetally_setup.py                      everything else (shared by all systems)
    uninstall_unix.sh
  app/posetally/
    geometry.py      line crossings, point-in-polygon, shape helpers
    model.py         the scoring setup: shapes, rules, settings, columns (plain data, JSON)
    scoring.py       tracking + setup -> events and column values
    formulas.py      safe evaluation of formula columns
    idcolumn.py      building the file label
    tracks.py        finding and reading DeepLabCut result files
    dlcproject.py    reading config.yaml and finding trained networks (does not import DeepLabCut)
    videoio.py       frames and metadata through OpenCV
    batch.py         scoring a list of videos, building the results table
    export.py        the Excel workbook
    session.py       project + videos + setup + results; saving and restoring
    worker.py        separate process that calls deeplabcut.analyze_videos
    demo.py          made-up projects with known answers
    ui/              the PySide6 interface
  tests/             pytest suite; tests/fake_dlc is a stand-in for DeepLabCut
```

- Only `worker.py` imports DeepLabCut, and it runs as its own process
  (`python -m posetally.worker --job job.json`), reporting progress as JSON lines. The
  window never imports DeepLabCut or PyTorch, so it starts quickly and cannot be
  taken down by them.
- Everything except `ui/` is free of Qt and can be scripted:

  ```python
  from posetally.session import Session
  from posetally.export import export_xlsx

  s = Session()
  s.open_project("/path/to/project/config.yaml")   # restores the saved setup
  s.add_folder("/path/to/videos")
  s.score_all()
  export_xlsx("scores.xlsx", s.ordered_results(), s.layout)
  ```

- To run the tests, with the installed environment's Python (path shown for macOS and Linux):

  ```
  ~/.posetally/env/bin/python -m pip install pytest
  ~/.posetally/env/bin/python -m pytest tests
  ```

- To work on the code without reinstalling, run from the source folder:
  `PYTHONPATH=app ~/.posetally/env/bin/python -m posetally`.
- The versions the installer uses are constants at the top of
  `installer/posetally_setup.py` (DeepLabCut version, the date packages are frozen
  to, the list of NVIDIA builds to try) and of the two first-stage scripts (uv and Python versions).

---

## Citing

Tracking is done by DeepLabCut. When you publish, cite it:

- Mathis et al. (2018), *Nature Neuroscience* 21, 1281–1289.
- Nath, Mathis et al. (2019), *Nature Protocols* 14, 2152–2176.
- For multi-animal projects: Lauer et al. (2022), *Nature Methods* 19, 496–504.

and state the DeepLabCut version (shown in the Track window and in the installer's
output) and the PoseTally settings, which are on the Setup sheet of every workbook.

DeepLabCut is licensed under LGPL-3.0; PySide6 (Qt) under LGPL-3.0. PoseTally uses
both as separately installed libraries.
