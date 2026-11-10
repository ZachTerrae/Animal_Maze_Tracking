#!/usr/bin/env python3
"""Second stage of the PoseTally installer.

The small per-system scripts (Install-Mac.command, Install-Windows.bat,
Install-Linux.sh) download the "uv" package tool, install a private copy of
Python and create an empty environment. They then run this file with that
Python. Everything from here on is the same on every system:

1. install PyTorch in the build that suits this computer (NVIDIA card, Apple
   chip, or processor only);
2. install DeepLabCut and the few extra packages PoseTally needs;
3. copy the PoseTally program into place;
4. check that everything starts;
5. create the icon to start it from.

Only the Python standard library is used here.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile

# ----------------------------------------------------------------- what gets installed

DLC_VERSION = "3.0.2"  # the DeepLabCut release PoseTally was tested against
TESTED_ON = "2026-11-10T00:00:00Z"  # other packages are taken as they were on this day
NEWEST_TORCH = "2.15"  # the newest PyTorch that existed on that day
APP_REQUIREMENTS = [
    "PySide6-Essentials>=6.6",
    "openpyxl>=3.1",
    "pyyaml",
    "numpy",
    "pandas>=2.2,<3",
    "tables>3.8",
    "opencv-python-headless>=4.8",
]
TORCH_INDEX = "https://download.pytorch.org/whl/"
# PyTorch builds for NVIDIA cards, newest first. The installer tries the newest one the
# graphics driver supports and steps down until one really runs on the card.
CUDA_TAGS = ["cu134", "cu132", "cu130", "cu129", "cu128", "cu126", "cu124", "cu121", "cu118"]

IS_WINDOWS = os.name == "nt"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

GPU_TEST = r"""
import json, sys
out = {"ok": False}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = bool(torch.cuda.is_available())
    mps = getattr(torch.backends, "mps", None)
    out["mps"] = bool(mps is not None and mps.is_available())
    if out["cuda"]:
        out["name"] = torch.cuda.get_device_name(0)
        layer = torch.nn.Conv2d(3, 8, 3).cuda()
        value = layer(torch.randn(1, 3, 64, 64, device="cuda")).sum().item()
        out["ran_on_gpu"] = value == value
    out["ok"] = True
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
print("RESULT " + json.dumps(out))
"""


class SetupError(Exception):
    """Something the user needs to know about; shown without a traceback."""


def say(text: str = "") -> None:
    print(text, flush=True)


def heading(number: int, total: int, text: str) -> None:
    say()
    say(f"[{number}/{total}] {text}")
    say("-" * (len(text) + 8))


class Setup:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.source = os.path.abspath(args.source)
        self.home = os.path.abspath(args.home)
        self.uv = os.path.abspath(args.uv)
        self.python = sys.executable
        self.env_dir = os.path.dirname(os.path.dirname(self.python))
        self.freeze = not args.no_date_freeze and args.dlc_version == DLC_VERSION
        self.report: dict = {}
        self.warnings: list[str] = []

    # ------------------------------------------------------------------ helpers

    def run(self, command: list[str], check: bool = True, capture: bool = False, quiet: bool = False):
        if not quiet:
            shown = " ".join(os.path.basename(command[0]) if i == 0 else part for i, part in enumerate(command))
            say("  > " + (shown if len(shown) < 300 else shown[:300] + " …"))
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.STDOUT if capture else None,
                text=True,
                encoding="utf-8" if capture else None,
                errors="replace" if capture else None,
            )
        except OSError as exc:  # the program is missing or cannot be started
            result = subprocess.CompletedProcess(command, 127, stdout=f"{exc}\n" if capture else None)
        if check and result.returncode != 0:
            if capture and result.stdout:
                say(result.stdout[-4000:])
            raise SetupError(f"A step failed (exit code {result.returncode}): {os.path.basename(command[0])} {command[1] if len(command) > 1 else ''}")
        return result

    def pip_install(self, packages: list[str], extra: list[str] | None = None, check: bool = True, frozen: bool | None = None):
        command = [self.uv, "pip", "install", "--python", self.python] + list(extra or [])
        if self.freeze if frozen is None else frozen:
            command += ["--exclude-newer", TESTED_ON]
        return self.run(command + packages, check=check)

    def python_snippet(self, code: str) -> dict:
        result = self.run([self.python, "-c", code], check=False, capture=True, quiet=True)
        for line in (result.stdout or "").splitlines():
            if line.startswith("RESULT "):
                try:
                    return json.loads(line[len("RESULT ") :])
                except ValueError:
                    break
        return {"ok": False, "error": (result.stdout or "").strip()[-600:] or "no output"}

    def installed(self) -> dict[str, str]:
        code = (
            "import importlib.metadata as m, json\n"
            "print('RESULT ' + json.dumps({(d.metadata['Name'] or '').lower().replace('_', '-'): d.version for d in m.distributions()}))"
        )
        return {k: v for k, v in self.python_snippet(code).items() if isinstance(v, str)}

    # --------------------------------------------------------------------- torch

    def find_nvidia(self) -> dict | None:
        """Details of the NVIDIA graphics card and its driver, if there is one."""
        tool = shutil.which("nvidia-smi")
        if tool is None and IS_WINDOWS:
            candidate = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
            tool = candidate if os.path.isfile(candidate) else None
        if tool is None:
            return None
        try:
            plain = subprocess.run([tool], capture_output=True, text=True, timeout=30)
            query = subprocess.run(
                [tool, "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if plain.returncode != 0:
            return None
        return parse_nvidia(plain.stdout, query.stdout if query.returncode == 0 else "")

    def install_torch(self) -> None:
        args = self.args
        cap = [] if not self.freeze else [f"torch<{bump_minor(NEWEST_TORCH)}"]
        base = ["torch>=2", "torchvision"] + cap
        machine = platform.machine().lower()

        if not args.reinstall and not args.torch_index:
            have = self.python_snippet(GPU_TEST)
            if have.get("ok") and "torchvision" in self.installed():
                wants_card = not IS_MAC and not args.cpu_only and self.find_nvidia() is not None
                if not wants_card or (have.get("cuda") and have.get("ran_on_gpu")):
                    where = "the NVIDIA graphics card" if have.get("cuda") else ("Apple graphics" if have.get("mps") else "the processor")
                    say(f"PyTorch {have.get('torch')} is already installed and runs on {where}. Keeping it.")
                    return

        if args.torch_index:
            say(f"Using the PyTorch source you asked for: {args.torch_index}")
            self.pip_install(base, ["--index-url", args.torch_index], frozen=False)
            return

        if IS_MAC:
            if machine in ("x86_64", "amd64"):
                say("This is an Intel Mac. PyTorch stopped making new versions for Intel Macs in 2024,")
                say("so an older PyTorch is installed and tracking runs on the processor (slow).")
                self.warnings.append("Intel Mac: tracking runs on the processor only and is slow.")
                self.pip_install(["torch>=2,<2.3", "torchvision", "numpy<2"])
            else:
                say("Apple chip found. Installing PyTorch with Apple graphics (Metal) support.")
                self.pip_install(base)
            return

        nvidia = None if args.cpu_only else self.find_nvidia()
        if nvidia:
            driver_cuda = nvidia.get("cuda")
            say(f"NVIDIA graphics card found: {nvidia.get('name', 'unknown model')} (driver {nvidia.get('driver', '?')}"
                + (f", supports CUDA up to {driver_cuda[0]}.{driver_cuda[1]})" if driver_cuda else ")"))
            say("Installing PyTorch for the graphics card. This is a large download (2 to 3 GB).")
            for tag in candidate_tags(driver_cuda):
                say()
                say(f"Trying the PyTorch build “{tag}” …")
                extra = ["--index-url", TORCH_INDEX + tag, "--reinstall-package", "torch", "--reinstall-package", "torchvision"]
                result = self.pip_install(base, extra, check=False, frozen=False)
                if result.returncode != 0:
                    say(f"  The build “{tag}” is not available for this computer; trying the next one.")
                    continue
                test = self.python_snippet(GPU_TEST)
                if test.get("ok") and test.get("cuda") and test.get("ran_on_gpu"):
                    say(f"  PyTorch {test.get('torch')} runs on the graphics card ({test.get('name')}).")
                    return
                say(f"  That build does not run on this graphics card ({test.get('error') or 'graphics card not usable'}); trying an older one.")
            self.warnings.append(
                "No PyTorch build could use the NVIDIA graphics card, so tracking will run on the processor (slow). "
                "Updating the NVIDIA driver and running the installer again usually fixes this."
            )
            say()
            say("None of the graphics-card builds worked. Installing the processor-only build instead.")
        elif args.cpu_only:
            say("Installing the processor-only PyTorch, as requested.")
        else:
            say("No NVIDIA graphics card was found. Installing the processor-only PyTorch.")
            say("Tracking will work but is slow. Scoring videos that are already tracked is fast either way.")
            self.warnings.append("No NVIDIA graphics card: tracking runs on the processor and is slow.")

        extra = ["--reinstall-package", "torch", "--reinstall-package", "torchvision"]
        if IS_WINDOWS:
            self.pip_install(base, extra)  # the standard Windows build is the processor-only one
        else:
            result = self.pip_install(base, extra + ["--index-url", TORCH_INDEX + "cpu"], check=False, frozen=False)
            if result.returncode != 0:
                self.pip_install(base, extra)

    # ------------------------------------------------------------------ packages

    def install_packages(self) -> None:
        args = self.args
        versions = self.installed()
        constraints = [f"{name}=={versions[name]}" for name in ("torch", "torchvision") if name in versions]
        if IS_MAC and platform.machine().lower() in ("x86_64", "amd64"):
            constraints.append("numpy<2")
        handle = tempfile.NamedTemporaryFile("w", suffix=".txt", prefix="posetally_constraints_", delete=False, encoding="utf-8")
        with handle:
            handle.write("\n".join(constraints) + "\n")
        packages = list(APP_REQUIREMENTS)
        if not args.no_deeplabcut:
            extra_name = "[tf]" if args.with_tensorflow else ""
            packages.insert(0, f"deeplabcut{extra_name}=={args.dlc_version}")
        try:
            result = self.pip_install(packages, ["--constraint", handle.name], check=False)
            if result.returncode != 0 and self.freeze:
                say()
                say("That did not work with the package versions PoseTally was tested with.")
                say("Trying again with the newest versions …")
                self.freeze = False
                result = self.pip_install(packages, ["--constraint", handle.name], check=False)
            if result.returncode != 0:
                raise SetupError(
                    "DeepLabCut and its packages could not be installed. Check the internet connection and the "
                    "messages above, then run the installer again; it continues where it stopped."
                )
        finally:
            try:
                os.remove(handle.name)
            except OSError:
                pass
        self.keep_one_opencv()

    def keep_one_opencv(self) -> None:
        """Two OpenCV packages in one environment break the window toolkit; keep the plain one."""
        versions = self.installed()
        clashing = [name for name in ("opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless") if name in versions]
        if not clashing:
            return
        say()
        say("Tidying up: more than one OpenCV package was installed; keeping the one without its own windows.")
        wanted = versions.get("opencv-python-headless") or versions[clashing[0]]
        self.run([self.uv, "pip", "uninstall", "--python", self.python] + clashing + ["opencv-python-headless"], check=False)
        self.pip_install([f"opencv-python-headless=={wanted}"], ["--reinstall-package", "opencv-python-headless"], frozen=False)

    # ----------------------------------------------------------------------- app

    def install_app(self) -> None:
        package = os.path.join(self.source, "app", "posetally")
        if not os.path.isfile(os.path.join(package, "__init__.py")):
            raise SetupError(
                f"The PoseTally program files were not found in {os.path.join(self.source, 'app')}.\n"
                "Unpack the whole ZIP file first and run the installer from the unpacked folder."
            )
        target_root = os.path.join(self.home, "app")
        target = os.path.join(target_root, "posetally")
        if os.path.abspath(package) != os.path.abspath(target):
            if os.path.isdir(target):
                shutil.rmtree(target)
            shutil.copytree(package, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        site_packages = sysconfig.get_paths()["purelib"]
        os.makedirs(site_packages, exist_ok=True)
        with open(os.path.join(site_packages, "posetally.pth"), "w", encoding="utf-8") as handle:
            handle.write(target_root + "\n")
        say(f"PoseTally program copied to {target}")

    def verify(self) -> None:
        check = self.python_snippet(
            "import json\n"
            "import posetally, yaml, numpy, pandas, tables, openpyxl, cv2\n"
            "from PySide6 import QtCore, QtGui, QtWidgets\n"
            "print('RESULT ' + json.dumps({'ok': True, 'posetally': posetally.__version__, 'qt': QtCore.qVersion(),"
            " 'opencv': cv2.__version__, 'numpy': numpy.__version__, 'pandas': pandas.__version__}))"
        )
        if not check.get("ok"):
            hint = ""
            if IS_LINUX and "lib" in str(check.get("error", "")).lower():
                hint = (
                    "\nOn Linux the window toolkit needs a few system libraries. On Ubuntu or Debian run:\n"
                    "  sudo apt install libegl1 libgl1 libxkbcommon-x11-0 libxcb-cursor0 libfontconfig1 libdbus-1-3\n"
                    "and start the installer again."
                )
            raise SetupError("PoseTally's own packages do not load:\n" + str(check.get("error")) + hint)
        self.report.update({k: v for k, v in check.items() if k != "ok"})
        say(f"PoseTally {check['posetally']} loads (Qt {check['qt']}, OpenCV {check['opencv']}, pandas {check['pandas']}).")

        if self.args.no_deeplabcut:
            self.report["deeplabcut"] = None
            say("DeepLabCut was left out, as requested: PoseTally can score tracked videos but cannot track new ones.")
            return
        result = self.run([self.python, "-m", "posetally.worker", "--check"], check=False, capture=True, quiet=True)
        found = None
        for line in (result.stdout or "").splitlines():
            marker = line.find("@@POSETALLY@@ ")
            if marker >= 0:
                try:
                    found = json.loads(line[marker + len("@@POSETALLY@@ ") :])
                except ValueError:
                    pass
        if not found or not found.get("deeplabcut"):
            detail = (found or {}).get("deeplabcut_error") or (result.stdout or "").strip()[-1500:]
            if IS_WINDOWS and ("dll" in detail.lower() or "winerror 126" in detail.lower()):
                detail += (
                    "\n\nThis usually means the “Microsoft Visual C++ Redistributable” is missing. Install it from\n"
                    "  https://aka.ms/vs/17/release/vc_redist.x64.exe\nand run the installer again."
                )
            raise SetupError("DeepLabCut was installed but does not start:\n" + detail)
        self.report.update({k: found.get(k) for k in ("deeplabcut", "torch", "cuda", "cuda_name", "mps", "tensorflow")})
        device = "processor only"
        if found.get("cuda"):
            device = f"NVIDIA graphics card ({found.get('cuda_name')})"
        elif found.get("mps"):
            device = "Apple graphics (Metal)"
        say(f"DeepLabCut {found['deeplabcut']} starts (PyTorch {found.get('torch')}). Tracking will run on: {device}.")
        if self.args.with_tensorflow:
            if found.get("tensorflow"):
                say(f"TensorFlow {found['tensorflow']} is available for networks trained with the TensorFlow engine.")
            else:
                self.warnings.append("TensorFlow was requested but does not load; TensorFlow-trained networks will not run.")

    # ----------------------------------------------------------------- shortcuts

    def make_icons(self) -> str:
        folder = os.path.join(self.home, "icons")
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        try:
            subprocess.run([self.python, "-m", "posetally.ui.style", folder], env=env, capture_output=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            pass
        return folder

    def make_shortcuts(self) -> None:
        icons = self.make_icons()
        if IS_MAC:
            self.mac_app(icons)
        elif IS_WINDOWS:
            self.windows_shortcuts(icons)
        else:
            self.linux_launcher(icons)

    def mac_app(self, icons: str) -> None:
        apps = os.path.join(os.path.expanduser("~"), "Applications")
        bundle = os.path.join(apps, "PoseTally.app")
        if os.path.isdir(bundle):
            shutil.rmtree(bundle)
        os.makedirs(os.path.join(bundle, "Contents", "MacOS"))
        os.makedirs(os.path.join(bundle, "Contents", "Resources"))
        launcher = os.path.join(bundle, "Contents", "MacOS", "PoseTally")
        with open(launcher, "w", encoding="utf-8") as handle:
            handle.write(
                "#!/bin/bash\n"
                "# Starts PoseTally with its own Python. Made by the PoseTally installer.\n"
                f'exec "{self.python}" -m posetally "$@"\n'
            )
        os.chmod(launcher, 0o755)
        icon_line = ""
        icns = os.path.join(icons, "PoseTally.icns")
        if os.path.isfile(icns):
            shutil.copy(icns, os.path.join(bundle, "Contents", "Resources", "PoseTally.icns"))
            icon_line = "  <key>CFBundleIconFile</key><string>PoseTally</string>\n"
        # The launcher is a script, and macOS takes an app without an Apple-chip program in it
        # for an Intel app: it would ask to install Rosetta. Saying "arm64" here prevents that.
        arch_line = ""
        if platform.machine().lower() in ("arm64", "aarch64"):
            arch_line = "  <key>LSArchitecturePriority</key><array><string>arm64</string></array>\n"
        version = self.report.get("posetally", "0")
        with open(os.path.join(bundle, "Contents", "Info.plist"), "w", encoding="utf-8") as handle:
            handle.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
                '<plist version="1.0">\n<dict>\n'
                "  <key>CFBundleName</key><string>PoseTally</string>\n"
                "  <key>CFBundleDisplayName</key><string>PoseTally</string>\n"
                "  <key>CFBundleIdentifier</key><string>org.posetally.app</string>\n"
                "  <key>CFBundleExecutable</key><string>PoseTally</string>\n"
                "  <key>CFBundlePackageType</key><string>APPL</string>\n"
                f"  <key>CFBundleShortVersionString</key><string>{version}</string>\n"
                f"  <key>CFBundleVersion</key><string>{version}</string>\n"
                + icon_line
                + arch_line
                + "  <key>NSHighResolutionCapable</key><true/>\n"
                "</dict>\n</plist>\n"
            )
        say(f"Created {bundle}")
        # A second way in, which shows error messages: it opens a Terminal window.
        plain = os.path.join(apps, "PoseTally (show messages).command")
        try:
            with open(plain, "w", encoding="utf-8") as handle:
                handle.write(
                    "#!/bin/bash\n"
                    "# Starts PoseTally from a Terminal window, to see error messages.\n"
                    f'"{self.python}" -m posetally "$@"\n'
                )
            os.chmod(plain, 0o755)
        except OSError:
            pass
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        link = os.path.join(desktop, "PoseTally")
        try:
            if os.path.islink(link):
                os.remove(link)
            if os.path.isdir(desktop) and not os.path.exists(link):
                os.symlink(bundle, link)
                say("Added a PoseTally icon to the Desktop")
        except OSError:
            pass
        self.report["launcher"] = bundle

    def windows_shortcuts(self, icons: str) -> None:
        scripts = os.path.dirname(self.python)
        windowed = os.path.join(scripts, "pythonw.exe")
        target = windowed if os.path.isfile(windowed) else self.python
        icon = os.path.join(icons, "PoseTally.ico")
        fallback = os.path.join(self.home, "PoseTally (show messages).bat")
        with open(fallback, "w", encoding="utf-8", newline="") as handle:
            handle.write(
                "@echo off\r\n"
                "rem Starts PoseTally with a console window, to see error messages.\r\n"
                "chcp 65001 >nul\r\n"  # this file is UTF-8: folder names may hold accented letters
                f'"{self.python}" -m posetally %*\r\n'
                "pause\r\n"
            )
        # PowerShell finds the Desktop and Start-menu folders itself (they can be redirected, and
        # their names can hold letters that do not survive being printed and read back). The
        # paths it needs from us travel in environment variables, which are safe for any letter.
        script = (
            "$shell = New-Object -ComObject WScript.Shell; $made = 0; "
            "foreach ($place in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) { "
            "if (-not $place) { continue }; "
            "try { "
            "$s = $shell.CreateShortcut((Join-Path $place 'PoseTally.lnk')); "
            "$s.TargetPath = $env:PT_TARGET; $s.Arguments = '-m posetally'; "
            "$s.WorkingDirectory = $env:USERPROFILE; $s.Description = 'PoseTally'; "
            "if (Test-Path $env:PT_ICON) { $s.IconLocation = $env:PT_ICON }; "
            "$s.Save(); $made = $made + 1 "
            "} catch { } }; "
            "Write-Output ('MADE=' + $made)"
        )
        made = 0
        env = dict(os.environ, PT_TARGET=target, PT_ICON=icon)
        try:
            result = subprocess.run(["powershell", "-NoProfile", "-Command", script], env=env, capture_output=True, text=True, errors="replace")
            found = re.search(r"MADE=(\d+)", result.stdout or "")
            made = int(found.group(1)) if found else 0
        except (OSError, subprocess.SubprocessError):
            made = 0
        if made:
            say("Created the PoseTally shortcut " + ("on the Desktop and in the Start menu." if made > 1 else "(Desktop or Start menu)."))
            self.report["launcher"] = "PoseTally shortcut"
        else:
            self.warnings.append(f"The Desktop shortcut could not be created. Start PoseTally with: {fallback}")
            self.report["launcher"] = fallback

    def linux_launcher(self, icons: str) -> None:
        home = os.path.expanduser("~")
        bin_dir = os.path.join(home, ".local", "bin")
        os.makedirs(bin_dir, exist_ok=True)
        script = os.path.join(bin_dir, "posetally")
        with open(script, "w", encoding="utf-8") as handle:
            handle.write(f'#!/bin/sh\nexec "{self.python}" -m posetally "$@"\n')
        os.chmod(script, 0o755)
        apps = os.path.join(home, ".local", "share", "applications")
        os.makedirs(apps, exist_ok=True)
        entry = os.path.join(apps, "posetally.desktop")
        with open(entry, "w", encoding="utf-8") as handle:
            handle.write(
                "[Desktop Entry]\nType=Application\nName=PoseTally\n"
                "Comment=Score animal tracking from DeepLabCut\n"
                f'Exec="{script}" %f\nIcon={os.path.join(icons, "PoseTally.png")}\nTerminal=false\nCategories=Science;Education;\n'
            )
        say(f"Created the command “posetally” ({script}) and a menu entry.")
        self.report["launcher"] = script

    # ----------------------------------------------------------------------- all

    def write_info(self) -> None:
        info = dict(self.report)
        info["installed"] = datetime.datetime.now().isoformat(timespec="seconds")
        info["python"] = sys.version.split()[0]
        info["home"] = self.home
        info["packages_as_of"] = TESTED_ON if self.freeze else "newest"
        try:
            with open(os.path.join(self.home, "install_info.json"), "w", encoding="utf-8") as handle:
                json.dump(info, handle, indent=1)
        except OSError:
            pass

    def main(self) -> int:
        total = 5
        say("=" * 60)
        say(" PoseTally installer")
        say("=" * 60)
        say(f"Installing into: {self.home}")
        say(f"System: {platform.system()} {platform.release()} ({platform.machine()}), Python {sys.version.split()[0]}")
        if self.args.no_deeplabcut:
            heading(1, total, "PyTorch and DeepLabCut: skipped, as requested")
            heading(2, total, "Installing the packages PoseTally needs")
            self.pip_install(list(APP_REQUIREMENTS))
        else:
            heading(1, total, "Installing PyTorch")
            self.install_torch()
            heading(2, total, f"Installing DeepLabCut {self.args.dlc_version} and the packages PoseTally needs")
            say("Several hundred megabytes are downloaded here; a few minutes on a good connection.")
            self.install_packages()
        heading(3, total, "Installing the PoseTally program")
        self.install_app()
        heading(4, total, "Checking that everything starts")
        self.verify()
        heading(5, total, "Creating the icon to start PoseTally")
        if self.args.no_shortcuts:
            say("Skipped, as requested.")
            self.report["launcher"] = f"{self.python} -m posetally"
        else:
            self.make_shortcuts()
        self.write_info()
        say()
        say("=" * 60)
        say(" PoseTally is installed.")
        say("=" * 60)
        launcher = self.report.get("launcher", "")
        if IS_MAC:
            say("Start it from the PoseTally icon on your Desktop, or from Applications in your home folder.")
        elif IS_WINDOWS:
            say("Start it from the PoseTally shortcut on your Desktop or from the Start menu.")
        else:
            say(f"Start it with:  {launcher}")
        for warning in self.warnings:
            say()
            say("Note: " + warning)
        return 0


# ------------------------------------------------------------------ small helpers


def bump_minor(version: str) -> str:
    """ "2.15" -> "2.16" (used as an upper limit)."""
    major, minor = version.split(".")[:2]
    return f"{major}.{int(minor) + 1}"


def parse_nvidia(plain_output: str, query_output: str = "") -> dict:
    info: dict = {}
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", plain_output)
    if match:
        info["cuda"] = (int(match.group(1)), int(match.group(2)))
    match = re.search(r"Driver Version:\s*([\d.]+)", plain_output)
    if match:
        info["driver"] = match.group(1)
    first = (query_output or "").strip().splitlines()[:1]
    if first and "," in first[0]:
        name, driver = [part.strip() for part in first[0].split(",", 1)]
        info["name"] = name
        info.setdefault("driver", driver)
    return info


def tag_version(tag: str) -> tuple[int, int]:
    digits = tag[2:]
    return (int(digits[:2]), int(digits[2:]))


def candidate_tags(driver_cuda: tuple[int, int] | None) -> list[str]:
    """PyTorch builds to try for a graphics driver, most suitable first."""
    if not driver_cuda:
        return list(CUDA_TAGS)
    fits = [tag for tag in CUDA_TAGS if tag_version(tag) <= tuple(driver_cuda)]
    # a build made for a slightly newer CUDA of the same generation often still runs
    close = [tag for tag in reversed(CUDA_TAGS) if tag_version(tag)[0] == driver_cuda[0] and tag not in fits]
    return fits + close


def parse_args(argv=None) -> argparse.Namespace:
    env = os.environ
    parser = argparse.ArgumentParser(description="PoseTally installer (second stage)")
    parser.add_argument("--source", required=True, help="the unpacked PoseTally folder")
    parser.add_argument("--home", required=True, help="where PoseTally is installed")
    parser.add_argument("--uv", required=True, help="path of the uv program")
    parser.add_argument("--dlc-version", default=env.get("POSETALLY_DLC_VERSION") or DLC_VERSION,
                        help=f"DeepLabCut version to install (default {DLC_VERSION})")
    parser.add_argument("--with-tensorflow", action="store_true", default=bool(env.get("POSETALLY_WITH_TENSORFLOW")),
                        help="also install TensorFlow, for networks trained with DeepLabCut's TensorFlow engine")
    parser.add_argument("--cpu-only", action="store_true", default=bool(env.get("POSETALLY_CPU_ONLY")),
                        help="do not use an NVIDIA graphics card even if there is one")
    parser.add_argument("--torch-index", default=env.get("POSETALLY_TORCH_INDEX") or "",
                        help="install PyTorch from this package index (advanced)")
    parser.add_argument("--no-deeplabcut", action="store_true", default=bool(env.get("POSETALLY_NO_DEEPLABCUT")),
                        help="leave out DeepLabCut and PyTorch: score already-tracked videos only")
    parser.add_argument("--no-date-freeze", action="store_true", default=bool(env.get("POSETALLY_NEWEST_PACKAGES")),
                        help="install the newest package versions instead of the tested ones")
    parser.add_argument("--no-shortcuts", action="store_true", help="do not create Desktop or menu icons")
    parser.add_argument("--reinstall", action="store_true", default=bool(env.get("POSETALLY_REINSTALL")),
                        help="install PyTorch again even if a working one is already there")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    try:
        return Setup(args).main()
    except SetupError as exc:
        say()
        say("!" * 60)
        say("The installation did not finish.")
        say(str(exc))
        say("!" * 60)
        return 1
    except KeyboardInterrupt:
        say("\nStopped. Run the installer again to continue.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
