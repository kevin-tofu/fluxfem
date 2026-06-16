#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

if command -v poetry >/dev/null 2>&1 && poetry run python -c 'import sys' >/dev/null 2>&1; then
  runner=(poetry run pytest)
else
  runner=(pytest)
fi

if [ "$#" -eq 0 ]; then
  exec "${runner[@]}" --cov=fluxfem --cov-report=xml
fi

exec "${runner[@]}" "$@"
