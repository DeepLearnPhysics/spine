#!/usr/bin/env bash

# Canonical runtime environment for SPINE container dependencies.
# This script is primarily a fallback for runtimes that expose the image
# filesystem but fail to apply the container environment automatically.

if [ "${BASH_SOURCE[0]:-}" = "$0" ]; then
    echo "Source this file instead of executing it: source /opt/spine/setup.sh" >&2
    exit 1
fi

_spine_prepend_path() {
    local var_name="$1"
    local entry="$2"
    local current_value

    if [ ! -e "${entry}" ]; then
        return 0
    fi

    current_value="${!var_name:-}"
    case ":${current_value}:" in
        *":${entry}:"*) ;;
        "")
            export "${var_name}=${entry}"
            ;;
        *)
            export "${var_name}=${entry}:${current_value}"
            ;;
    esac
}

export SPINE_RUNTIME_DIR="${SPINE_RUNTIME_DIR:-/opt/spine}"
export ROOTSYS="${ROOTSYS:-/opt/root}"
export LARCV_BASEDIR="${LARCV_BASEDIR:-/app/larcv2}"
export LARCV_LIBDIR="${LARCV_LIBDIR:-${LARCV_BASEDIR}/build/lib}"
export LARCV_PYTHONDIR="${LARCV_PYTHONDIR:-${LARCV_BASEDIR}/python}"
export FMATCH_BASEDIR="${FMATCH_BASEDIR:-/opt/OpT0Finder}"
export FMATCH_BUILDDIR="${FMATCH_BUILDDIR:-${FMATCH_BASEDIR}/build}"
export FMATCH_LIBDIR="${FMATCH_LIBDIR:-${FMATCH_BUILDDIR}/lib}"
export FMATCH_INCDIR="${FMATCH_INCDIR:-${FMATCH_BUILDDIR}/include}"
export FMATCH_BINDIR="${FMATCH_BINDIR:-${FMATCH_BUILDDIR}/bin}"
export FMATCH_DATADIR="${FMATCH_DATADIR:-${FMATCH_BASEDIR}/dat}"

_spine_prepend_path PATH "${ROOTSYS}/bin"
_spine_prepend_path PATH "${FMATCH_BASEDIR}/bin"
_spine_prepend_path PATH "${FMATCH_BINDIR}"

_spine_prepend_path LD_LIBRARY_PATH "${ROOTSYS}/lib"
_spine_prepend_path LD_LIBRARY_PATH "${LARCV_LIBDIR}"
_spine_prepend_path LD_LIBRARY_PATH "${FMATCH_LIBDIR}"

_spine_prepend_path PYTHONPATH "${LARCV_PYTHONDIR}"
_spine_prepend_path PYTHONPATH "${FMATCH_BASEDIR}/python"
_spine_prepend_path PYTHONPATH "${ROOTSYS}/lib"

unset -f _spine_prepend_path
