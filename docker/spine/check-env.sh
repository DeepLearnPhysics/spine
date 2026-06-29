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
    "LARCV_BUILDDIR",
    "LARCV_LIBDIR",
    "LARCV_PYTHONDIR",
    "LARCV_NUMPY",
    "LARCV_OPENCV",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
):
    print(f"{key}: {os.environ.get(key, '')}")

import ROOT

print(f"ROOT OK: {ROOT.gROOT.GetVersion()}")

import larcv

print(f"larcv OK: {larcv}")

from larcv import larcv as larcv_cpp

if not hasattr(larcv_cpp, "fill_3d_voxels"):
    raise RuntimeError(
        "larcv imported, but larcv.fill_3d_voxels is missing. "
        "This usually means the LArCV PyUtil dictionary was not loaded."
    )

print("larcv PyUtil OK: fill_3d_voxels available")

import spine

print(f"spine OK: {spine}")
PY
