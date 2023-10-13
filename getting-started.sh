#!/bin/bash

echo "Setting up project ..."

if ! [ -d ".venv" ]; then
  python3 -m venv .venv
  pip install -r requirements.txt
fi

echo "Activating virtual python environment ..."

case "$OSTYPE" in
  darwin*) . .venv/bin/activate ;;
  linux*) . .venv/bin/activate ;;
  msys*) . .venv/Scripts/activate.bat ;;
  *) echo "Cannot activate virtual python environment - unknown os: $OSTYPE" ;;
esac

echo "Done"