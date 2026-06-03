#!/usr/bin/env bash

set -euo pipefail

source /opt/spine/setup.sh

echo "which python: $(command -v python)"
python - <<'PY'
import os
import sys

print(f"sys.executable: {sys.executable}")
for key in (
    "ROOTSYS",
    "LARCV_BASEDIR",
    "LARCV_LIBDIR",
    "LARCV_PYTHONDIR",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
):
    print(f"{key}: {os.environ.get(key, '')}")

import ROOT

print(f"ROOT OK: {ROOT.gROOT.GetVersion()}")

import larcv

print(f"larcv OK: {larcv}")

import spine

print(f"spine OK: {spine}")
PY
