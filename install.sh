#!/bin/sh
set -eu
# Resolve the checkout independently of the caller's current directory.
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
"${PYTHON:-python3}" -m pip install "$script_dir" "$@"
