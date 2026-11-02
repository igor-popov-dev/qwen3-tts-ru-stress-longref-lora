#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'Нужен Mac с Apple Silicon (M1 или новее), без Rosetta.' >&2
  exit 1
fi
PYTHON="${PYTHON:-python3.11}"
if ! command -v "$PYTHON" >/dev/null; then
  echo 'Установите Python 3.11: brew install python@3.11' >&2
  exit 1
fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Нужен Python 3.11"'
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
"$PYTHON" -m venv .venv-stress
.venv-stress/bin/python -m pip install -r requirements-stress.txt
.venv/bin/python verify_adapter.py
echo 'Готово. Пример запуска со своим WAV-файлом — в README.md.'
