"""The shared installer program (installer/posetally_setup.py): the decisions it takes,
checked without downloading anything."""

import importlib.util
import os
import plistlib
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


@pytest.fixture()
def setup_module():
    spec = importlib.util.spec_from_file_location("posetally_setup", os.path.join(ROOT, "installer", "posetally_setup.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NVIDIA_SMI = """
Tue Nov 10 10:00:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
"""


def test_reading_the_graphics_driver(setup_module):
    info = setup_module.parse_nvidia(NVIDIA_SMI, "NVIDIA GeForce RTX 3060, 560.94\n")
    assert info == {"cuda": (12, 6), "driver": "560.94", "name": "NVIDIA GeForce RTX 3060"}
    assert setup_module.parse_nvidia("garbage") == {}


def test_which_pytorch_builds_are_tried_for_a_driver(setup_module):
    tags = setup_module.candidate_tags
    # newest build the driver supports first, then older ones, then slightly newer ones of the same generation
    assert tags((12, 6)) == ["cu126", "cu124", "cu121", "cu118", "cu128", "cu129"]
    assert tags((13, 2))[:3] == ["cu132", "cu130", "cu129"] and tags((13, 2))[-1] == "cu134"
    assert tags((11, 8)) == ["cu118"]
    assert tags(None) == setup_module.CUDA_TAGS
    assert setup_module.tag_version("cu118") == (11, 8) and setup_module.tag_version("cu134") == (13, 4)
    assert setup_module.bump_minor("2.15") == "2.16"


def make_setup(module, tmp_path, monkeypatch, argv=(), nvidia=None, torch_states=(), installs_fail=()):
    """A Setup whose commands are recorded instead of run."""
    args = module.parse_args(["--source", ROOT, "--home", str(tmp_path / "home"), "--uv", "/fake/uv", *argv])
    setup = module.Setup(args)
    calls: list[list[str]] = []
    states = list(torch_states)

    def run(command, check=True, capture=False, quiet=False):
        calls.append(list(command))
        failed = any(marker in " ".join(command) for marker in installs_fail)
        return subprocess.CompletedProcess(command, 1 if failed else 0, stdout="")

    monkeypatch.setattr(setup, "run", run)
    monkeypatch.setattr(setup, "find_nvidia", lambda: nvidia)
    monkeypatch.setattr(setup, "python_snippet", lambda code: states.pop(0) if states else {"ok": False})
    monkeypatch.setattr(setup, "installed", lambda: {})
    return setup, calls


def index_of(call):
    return call[call.index("--index-url") + 1] if "--index-url" in call else "PyPI"


def test_nvidia_card_steps_down_until_a_build_really_runs(setup_module, tmp_path, monkeypatch):
    monkeypatch.setattr(setup_module, "IS_MAC", False)
    monkeypatch.setattr(setup_module, "IS_WINDOWS", True)
    card = {"name": "GTX 1080 Ti", "driver": "560.94", "cuda": (12, 6)}
    states = [
        {"ok": False},  # nothing installed yet
        {"ok": True, "cuda": True, "ran_on_gpu": False, "error": "no kernel image is available"},  # cu124: installs, card refuses it
        {"ok": True, "cuda": True, "ran_on_gpu": True, "torch": "2.5.1+cu121", "name": "GTX 1080 Ti"},  # cu121 works
    ]
    setup, calls = make_setup(setup_module, tmp_path, monkeypatch, nvidia=card, torch_states=states, installs_fail=["/whl/cu126"])
    setup.install_torch()
    assert [index_of(c).rsplit("/", 1)[-1] for c in calls] == ["cu126", "cu124", "cu121"]
    assert all("torch<2.16" in c and "torchvision" in c for c in calls)
    assert all("--exclude-newer" not in c for c in calls)  # PyTorch's own index has no dates
    assert setup.warnings == []


def test_no_working_build_falls_back_to_the_processor(setup_module, tmp_path, monkeypatch):
    monkeypatch.setattr(setup_module, "IS_MAC", False)
    monkeypatch.setattr(setup_module, "IS_WINDOWS", True)
    card = {"name": "Old card", "cuda": (11, 8)}
    setup, calls = make_setup(setup_module, tmp_path, monkeypatch, nvidia=card, installs_fail=["/whl/cu118"])
    setup.install_torch()
    assert [index_of(c).rsplit("/", 1)[-1] for c in calls] == ["cu118", "PyPI"]
    assert "--exclude-newer" in calls[-1]
    assert any("processor" in w for w in setup.warnings)


def test_no_card_and_cpu_only(setup_module, tmp_path, monkeypatch):
    monkeypatch.setattr(setup_module, "IS_MAC", False)
    monkeypatch.setattr(setup_module, "IS_WINDOWS", False)  # Linux: prefer the small processor-only build
    setup, calls = make_setup(setup_module, tmp_path, monkeypatch)
    setup.install_torch()
    assert [index_of(c).rsplit("/", 1)[-1] for c in calls] == ["cpu"]
    # if that source cannot be reached, the standard build is used instead
    setup, calls = make_setup(setup_module, tmp_path, monkeypatch, installs_fail=["/whl/cpu"])
    setup.install_torch()
    assert [index_of(c).rsplit("/", 1)[-1] for c in calls] == ["cpu", "PyPI"]
    # a card is ignored when asked
    card = {"name": "RTX", "cuda": (13, 4)}
    setup, calls = make_setup(setup_module, tmp_path, monkeypatch, argv=["--cpu-only"], nvidia=card)
    setup.install_torch()
    assert [index_of(c).rsplit("/", 1)[-1] for c in calls] == ["cpu"]


def test_a_working_pytorch_is_kept_on_a_second_run(setup_module, tmp_path, monkeypatch):
    monkeypatch.setattr(setup_module, "IS_MAC", False)
    monkeypatch.setattr(setup_module, "IS_WINDOWS", True)
    card = {"name": "RTX 3060", "cuda": (12, 6)}
    working = {"ok": True, "cuda": True, "ran_on_gpu": True, "torch": "2.11.0+cu126"}
    setup, calls = make_setup(setup_module, tmp_path, monkeypatch, nvidia=card, torch_states=[working])
    monkeypatch.setattr(setup, "installed", lambda: {"torch": "2.11.0+cu126", "torchvision": "0.26.0+cu126"})
    setup.install_torch()
    assert calls == []
    # but a processor-only PyTorch on a computer that now has a card is replaced
    cpu = {"ok": True, "cuda": False, "torch": "2.15.0"}
    setup, calls = make_setup(setup_module, tmp_path, monkeypatch, nvidia=card, torch_states=[cpu, working])
    monkeypatch.setattr(setup, "installed", lambda: {"torch": "2.15.0", "torchvision": "0.29.2"})
    setup.install_torch()
    assert [index_of(c).rsplit("/", 1)[-1] for c in calls] == ["cu126"]


def test_macs(setup_module, tmp_path, monkeypatch):
    monkeypatch.setattr(setup_module, "IS_MAC", True)
    monkeypatch.setattr(setup_module, "IS_WINDOWS", False)
    monkeypatch.setattr(setup_module.platform, "machine", lambda: "arm64")
    setup, calls = make_setup(setup_module, tmp_path, monkeypatch)
    setup.install_torch()
    assert len(calls) == 1 and index_of(calls[0]) == "PyPI" and "torch>=2" in calls[0]
    monkeypatch.setattr(setup_module.platform, "machine", lambda: "x86_64")
    setup, calls = make_setup(setup_module, tmp_path, monkeypatch)
    setup.install_torch()
    assert "torch>=2,<2.3" in calls[0] and "numpy<2" in calls[0]
    assert any("Intel Mac" in w for w in setup.warnings)


def test_packages_step_pins_pytorch_and_retries_without_the_date(setup_module, tmp_path, monkeypatch):
    monkeypatch.setattr(setup_module, "IS_MAC", False)
    setup, calls = make_setup(setup_module, tmp_path, monkeypatch, argv=["--with-tensorflow"], installs_fail=["--exclude-newer"])
    seen = {}

    def run(command, check=True, capture=False, quiet=False):
        calls.append(list(command))
        if "--constraint" in command:
            seen["constraints"] = open(command[command.index("--constraint") + 1]).read().split()
        return subprocess.CompletedProcess(command, 1 if "--exclude-newer" in command else 0, stdout="")

    monkeypatch.setattr(setup, "run", run)
    monkeypatch.setattr(setup, "installed", lambda: {"torch": "2.11.0+cu126", "torchvision": "0.26.0+cu126", "opencv-python-headless": "4.11.0.86"})
    setup.install_packages()
    assert seen["constraints"] == ["torch==2.11.0+cu126", "torchvision==0.26.0+cu126"]
    assert "deeplabcut[tf]==3.0.2" in calls[0] and "PySide6-Essentials>=6.6" in calls[0]
    assert "--exclude-newer" in calls[0] and "--exclude-newer" not in calls[1]  # second try with newest versions


def test_two_opencv_packages_are_reduced_to_one(setup_module, tmp_path, monkeypatch):
    setup, calls = make_setup(setup_module, tmp_path, monkeypatch)
    monkeypatch.setattr(setup, "installed", lambda: {"opencv-python": "4.11.0.86", "opencv-python-headless": "4.11.0.86"})
    setup.keep_one_opencv()
    assert calls[0][:3] == ["/fake/uv", "pip", "uninstall"] and "opencv-python" in calls[0]
    assert "opencv-python-headless==4.11.0.86" in calls[1]
    setup, calls = make_setup(setup_module, tmp_path, monkeypatch)
    monkeypatch.setattr(setup, "installed", lambda: {"opencv-python-headless": "5.0.0.93"})
    setup.keep_one_opencv()
    assert calls == []


def test_mac_application_bundle(setup_module, tmp_path, monkeypatch):
    home = tmp_path / "user"
    (home / "Desktop").mkdir(parents=True)
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(home) if p == "~" else p)
    setup, _calls = make_setup(setup_module, tmp_path, monkeypatch)
    setup.report["posetally"] = "0.1.0"
    icons = tmp_path / "icons"
    icons.mkdir()
    (icons / "PoseTally.icns").write_bytes(b"icns")
    bundle = home / "Applications" / "PoseTally.app"
    # an Intel Mac: nothing is said about the processor
    monkeypatch.setattr(setup_module.platform, "machine", lambda: "x86_64")
    setup.mac_app(str(icons))
    with open(bundle / "Contents" / "Info.plist", "rb") as handle:
        assert "LSArchitecturePriority" not in plistlib.load(handle)
    # an Apple chip: the app says it is native, so that macOS does not ask for Rosetta
    monkeypatch.setattr(setup_module.platform, "machine", lambda: "arm64")
    setup.mac_app(str(icons))
    with open(bundle / "Contents" / "Info.plist", "rb") as handle:
        plist = plistlib.load(handle)
    assert plist["LSArchitecturePriority"] == ["arm64"]
    assert plist["CFBundleExecutable"] == "PoseTally" and plist["CFBundleIconFile"] == "PoseTally"
    assert plist["CFBundlePackageType"] == "APPL" and plist["CFBundleShortVersionString"] == "0.1.0"
    launcher = bundle / "Contents" / "MacOS" / "PoseTally"
    assert os.access(launcher, os.X_OK)
    text = launcher.read_text()
    assert text.startswith("#!/bin/bash") and f'exec "{sys.executable}" -m posetally "$@"' in text
    assert (bundle / "Contents" / "Resources" / "PoseTally.icns").read_bytes() == b"icns"
    assert os.path.islink(home / "Desktop" / "PoseTally") and os.path.realpath(home / "Desktop" / "PoseTally") == os.path.realpath(bundle)
    plain = home / "Applications" / "PoseTally (show messages).command"
    assert os.access(plain, os.X_OK) and f'"{sys.executable}" -m posetally' in plain.read_text()
    setup.mac_app(str(icons))  # running the installer again replaces it cleanly


def test_windows_shortcuts(setup_module, tmp_path, monkeypatch):
    setup, _calls = make_setup(setup_module, tmp_path, monkeypatch)
    os.makedirs(setup.home, exist_ok=True)
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"], seen["env"] = command, kwargs.get("env", {})
        return subprocess.CompletedProcess(command, 0, stdout=seen.get("answer", "MADE=2\n"), stderr="")

    monkeypatch.setattr(setup_module.subprocess, "run", fake_run)
    setup.windows_shortcuts(str(tmp_path / "icons"))
    script = seen["command"][-1]
    assert seen["command"][:3] == ["powershell", "-NoProfile", "-Command"]
    # PowerShell looks up both folders itself, and gets our paths through the environment
    assert "GetFolderPath('Desktop')" in script and "GetFolderPath('Programs')" in script and "'-m posetally'" in script
    assert seen["env"]["PT_TARGET"].endswith(os.path.basename(sys.executable)) and seen["env"]["PT_ICON"].endswith("PoseTally.ico")
    script.encode("ascii")
    assert setup.report["launcher"] == "PoseTally shortcut" and not setup.warnings
    bat = open(os.path.join(setup.home, "PoseTally (show messages).bat"), "rb").read()
    assert bat.startswith(b"@echo off\r\n") and b"chcp 65001" in bat and b"-m posetally %*\r\n" in bat
    assert b"\n" not in bat.replace(b"\r\n", b"")
    # nothing could be created: the user is told how to start the program anyway
    seen["answer"] = "MADE=0\n"
    setup.windows_shortcuts(str(tmp_path / "icons"))
    assert setup.report["launcher"].endswith("PoseTally (show messages).bat") and "could not be created" in setup.warnings[0]


def test_installing_the_program_files(setup_module, tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    monkeypatch.setattr(setup_module.sysconfig, "get_paths", lambda: {"purelib": str(site)})
    setup, _calls = make_setup(setup_module, tmp_path, monkeypatch)
    setup.install_app()
    target = tmp_path / "home" / "app"
    assert (target / "posetally" / "__main__.py").is_file() and (target / "posetally" / "ui" / "mainwindow.py").is_file()
    assert not list(target.rglob("__pycache__"))
    assert (site / "posetally.pth").read_text().strip() == str(target)
    # a folder that is not the unpacked download is refused with an explanation
    setup.source = str(tmp_path)
    with pytest.raises(setup_module.SetupError, match="Unpack the whole ZIP"):
        setup.install_app()


def test_first_stage_scripts_agree_and_are_well_formed():
    unix = open(os.path.join(ROOT, "installer", "install_unix.sh"), encoding="utf-8").read()
    windows = open(os.path.join(ROOT, "installer", "install_windows.ps1"), "rb").read()
    bat = open(os.path.join(ROOT, "Install-Windows.bat"), "rb").read()
    for raw in (windows, bat, open(os.path.join(ROOT, "Uninstall-Windows.bat"), "rb").read()):
        raw.decode("ascii")  # Windows PowerShell 5 and cmd misread anything else
        assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    import re

    uv_unix = re.search(r'UV_VERSION="([\d.]+)"', unix).group(1)
    uv_win = re.search(rb'\$UvVersion = "([\d.]+)"', windows).group(1).decode()
    py_unix = re.search(r'PYTHON_VERSION="([\d.]+)"', unix).group(1)
    py_win = re.search(rb'\$PythonVersion = "([\d.]+)"', windows).group(1).decode()
    assert uv_unix == uv_win and py_unix == py_win
    for name in ("Install-Mac.command", "Install-Linux.sh", "Uninstall-Mac.command", os.path.join("installer", "install_unix.sh")):
        path = os.path.join(ROOT, name)
        assert os.access(path, os.X_OK), f"{name} must be executable"
        assert subprocess.run(["bash", "-n", path]).returncode == 0
