#!/bin/bash
# PoseTally installer for Linux. Run it in a terminal:  bash Install-Linux.sh
exec /bin/bash "$(cd "$(dirname "$0")" && pwd)/installer/install_unix.sh" "$@"
