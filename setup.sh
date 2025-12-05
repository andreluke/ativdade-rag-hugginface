#!/usr/bin/env bash
set -e

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Ambiente pronto. Ative com: source .venv/bin/activate"