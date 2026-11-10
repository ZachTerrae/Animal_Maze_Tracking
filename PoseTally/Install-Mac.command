#!/bin/bash
# PoseTally installer for macOS. Double-click this file.
# (If macOS refuses to open it: right-click it, choose Open, then Open again.)
exec /bin/bash "$(cd "$(dirname "$0")" && pwd)/installer/install_unix.sh" "$@"
