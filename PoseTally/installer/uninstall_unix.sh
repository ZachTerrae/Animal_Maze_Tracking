#!/bin/bash
# Removes PoseTally from macOS or Linux.
# Your DeepLabCut projects, your videos, your results and the scoring setups saved
# inside your project folders are NOT touched.

set -u
APP_HOME="${POSETALLY_HOME:-$HOME/.posetally}"

echo "This removes PoseTally and its private copy of Python and DeepLabCut:"
echo "  $APP_HOME"
echo "Your projects, videos, results and saved scoring setups are not touched."
echo
if [ -t 0 ] && [ -z "${POSETALLY_NO_PAUSE:-}" ]; then
    printf "Type yes and press Return to remove it: "
    read -r answer || answer=""
    if [ "$answer" != "yes" ]; then
        echo "Nothing was removed."
        exit 0
    fi
fi

case "$APP_HOME" in
    "" | "/" | "$HOME" | "$HOME/")
        echo "Refusing to remove '$APP_HOME'."
        exit 1
        ;;
esac

rm -rf "$APP_HOME"
if [ "$(uname -s)" = "Darwin" ]; then
    rm -rf "$HOME/Applications/PoseTally.app"
    rm -f "$HOME/Applications/PoseTally (show messages).command"
    [ -L "$HOME/Desktop/PoseTally" ] && rm -f "$HOME/Desktop/PoseTally"
    rm -rf "$HOME/Library/Logs/PoseTally"
else
    rm -f "$HOME/.local/bin/posetally" "$HOME/.local/share/applications/posetally.desktop"
fi
echo "PoseTally was removed."
if [ -t 0 ] && [ -z "${POSETALLY_NO_PAUSE:-}" ]; then
    printf "Press Return to close this window. "
    read -r _ || true
fi
