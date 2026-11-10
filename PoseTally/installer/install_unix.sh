#!/bin/bash
# PoseTally installer for macOS and Linux (first stage).
#
# It downloads the "uv" package tool, installs a private copy of Python and
# creates an empty environment, all inside one folder in your home folder.
# The rest of the installation is done by installer/posetally_setup.py.
#
# Nothing outside that folder is changed, except for the icon that starts
# PoseTally. Running this script again repairs or updates the installation.

set -u

UV_VERSION="0.13.7"
PYTHON_VERSION="3.12"

SRC="$(cd "$(dirname "$0")/.." && pwd)"
APP_HOME="${POSETALLY_HOME:-$HOME/.posetally}"

finish() {
    code="$1"
    echo
    if [ "$code" -ne 0 ]; then
        echo "The installation did not finish."
        echo "Read the messages above. When the problem is fixed, run the installer again:"
        echo "it continues where it stopped."
    fi
    if [ -t 0 ] && [ -z "${POSETALLY_NO_PAUSE:-}" ]; then
        printf "Press Return to close this window. "
        read -r _ || true
    fi
    exit "$code"
}

echo "============================================================"
echo " PoseTally installer"
echo "============================================================"

if [ ! -f "$SRC/app/posetally/__init__.py" ] || [ ! -f "$SRC/installer/posetally_setup.py" ]; then
    echo "The PoseTally program files were not found next to this installer."
    echo "Unpack the whole ZIP file first, then run the installer from the unpacked folder."
    finish 1
fi

case "$(uname -s)" in
    Darwin)
        if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
            UV_ASSET="uv-aarch64-apple-darwin"
        else
            UV_ASSET="uv-x86_64-apple-darwin"
        fi
        ;;
    Linux)
        case "$(uname -m)" in
            x86_64 | amd64) UV_ASSET="uv-x86_64-unknown-linux-gnu" ;;
            aarch64 | arm64) UV_ASSET="uv-aarch64-unknown-linux-gnu" ;;
            *)
                echo "This kind of computer ($(uname -m)) is not supported."
                finish 1
                ;;
        esac
        ;;
    *)
        echo "This installer is for macOS and Linux. On Windows, use Install-Windows.bat."
        finish 1
        ;;
esac

if ! command -v curl >/dev/null 2>&1; then
    echo "The program 'curl' is needed to download files, and it was not found."
    finish 1
fi

mkdir -p "$APP_HOME/bin" || finish 1
export UV_PYTHON_INSTALL_DIR="$APP_HOME/python"
export UV_CACHE_DIR="$APP_HOME/cache"
export UV_PYTHON_PREFERENCE="only-managed"
export UV_NO_CONFIG="1"
UV="$APP_HOME/bin/uv"

if ! "$UV" --version 2>/dev/null | grep -q " $UV_VERSION"; then
    echo
    echo "Downloading the package tool (uv $UV_VERSION) ..."
    TMP="$(mktemp -d)" || finish 1
    URL="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$UV_ASSET.tar.gz"
    if ! curl -fL --retry 3 --progress-bar -o "$TMP/uv.tar.gz" "$URL"; then
        echo "The download failed. Check the internet connection and run the installer again."
        rm -rf "$TMP"
        finish 1
    fi
    if ! tar -xzf "$TMP/uv.tar.gz" -C "$TMP" || ! cp "$TMP/$UV_ASSET/uv" "$UV"; then
        echo "The downloaded file could not be unpacked."
        rm -rf "$TMP"
        finish 1
    fi
    chmod +x "$UV"
    rm -rf "$TMP"
fi

echo
echo "Installing a private copy of Python $PYTHON_VERSION (it does not touch any Python you already have) ..."
"$UV" python install "$PYTHON_VERSION" || finish 1
"$UV" venv --python "$PYTHON_VERSION" --seed --allow-existing "$APP_HOME/env" || finish 1

"$APP_HOME/env/bin/python" "$SRC/installer/posetally_setup.py" --source "$SRC" --home "$APP_HOME" --uv "$UV" "$@"
finish $?
