#!/usr/bin/env bash
# Build the SPINE documentation with the same strictness as CI and RTD.

set -euo pipefail

docs_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$docs_dir"

# Autosummary files are derived output. Removing them prevents deleted or moved
# API members from surviving in a local incremental build.
rm -rf source/api/generated
make clean
make html SPHINXOPTS="-W --keep-going"
make epub SPHINXOPTS="-W --keep-going"

echo "Documentation built at $docs_dir/build/html/index.html and $docs_dir/build/epub/SPINE.epub"
