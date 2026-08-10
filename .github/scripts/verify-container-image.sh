#!/usr/bin/env bash

set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 REPOSITORY REFERENCE [REFERENCE ...]" >&2
  exit 2
fi

repository="${1%/}"
shift

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 2
fi

visited_references=()

verify_reference() {
  local reference="$1"
  local raw
  local digest
  local child
  local visited

  for visited in "${visited_references[@]-}"; do
    if [[ "${visited}" == "${reference}" ]]; then
      return
    fi
  done
  visited_references[${#visited_references[@]}]="${reference}"

  echo "Verifying ${reference}"
  raw="$(docker buildx imagetools inspect "${reference}" --raw)"
  jq -e '.schemaVersion == 2' <<< "${raw}" >/dev/null

  while IFS= read -r digest; do
    [[ -n "${digest}" ]] || continue
    child="${repository}@${digest}"
    verify_reference "${child}"
  done < <(jq -r '.manifests[]?.digest // empty' <<< "${raw}")
}

for reference in "$@"; do
  [[ -n "${reference}" ]] || continue
  if [[ "${reference}" == *"/"* && \
        ( "${reference}" == *":"* || "${reference}" == *"@"* ) ]]; then
    verify_reference "${reference}"
  else
    verify_reference "${repository}:${reference}"
  fi
done

echo "All referenced manifests are available."
