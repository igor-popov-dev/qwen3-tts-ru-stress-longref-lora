#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo 'Сначала выполните bash install.sh в каталоге репозитория.' >&2
  exit 1
fi
exec "$ROOT/.venv/bin/python" "$ROOT/synthesize.py" "$@"
