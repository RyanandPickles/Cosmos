#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required. Install it from https://www.python.org/downloads/macos/"
  read "?Press Return to close..."
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating local Python environment..."
  python3 -m venv .venv
fi

if ! .venv/bin/python -c "import telethon" >/dev/null 2>&1; then
  echo "Installing Telethon..."
  .venv/bin/python -m pip install -r requirements.txt
fi

echo "Starting Telegram message logger..."
echo "Keep this window open. Press Control-C to stop."
set +e
.venv/bin/python bot.py

exit_code=$?
echo "Logger stopped with exit code $exit_code"
read "?Press Return to close..."
exit "$exit_code"
