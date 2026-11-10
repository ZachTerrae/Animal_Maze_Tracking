#!/bin/bash
# Removes PoseTally from this Mac. Double-click this file.
exec /bin/bash "$(cd "$(dirname "$0")" && pwd)/installer/uninstall_unix.sh" "$@"
